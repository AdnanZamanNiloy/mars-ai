from typing import Any, Dict, List

from app.agents.evidence_utils import (
    dedupe_semantic_facts,
    filter_search_results_by_domain,
    normalize_claim_text,
    source_reliability_score,
)
from app.core.llm import LLMClient, clamp_confidence


SUMMARIZER_SYSTEM_PROMPT = """
You are the Summarizer Agent in a multi-agent research pipeline.
Your job is to extract precise, standalone factual claims from raw web content.
The claims you produce are the only evidence the rest of the pipeline will use.
Accuracy and precision here determine the quality of the entire report.
 
━━━ YOUR RULES ━━━
 
RULE 1 — REWRITE, NEVER COPY
  Do NOT copy sentences verbatim from the source.
  Rewrite every claim in your own words while preserving its exact meaning.
  If you cannot rewrite it without losing precision, quote only the essential phrase.
 
RULE 2 — ONE FACT PER CLAIM
  Each claim must express exactly one idea.
  Split compound sentences into separate claims.
 
  BAD  → "Knowledge is justified true belief, and Gettier challenged this in 1963."
  GOOD → Claim 1: "The justified true belief (JTB) model defines knowledge as belief
          that is true and supported by adequate justification."
         Claim 2: "Edmund Gettier (1963) published counterexamples that showed JTB
          is insufficient as a complete account of knowledge."
 
RULE 3 — ASSIGN CONFIDENCE HONESTLY
  Score each claim 0.0 to 1.0 based on how directly the source supports it:
  0.9–1.0 → explicitly stated with data, definition, or citation in source
  0.6–0.8 → clearly implied or paraphrased from a credible passage
  0.3–0.5 → inferred or extrapolated — flag with low confidence
  0.0–0.2 → speculative, anecdotal, or contradicted elsewhere — omit if possible
 
RULE 4 — DEDUPLICATE WITHIN THIS SOURCE
  If two passages from the same source say the same thing, extract it once.
  Keep the version with higher confidence.
 
RULE 5 — IGNORE IRRELEVANT CONTENT
  Skip: navigation text, cookie notices, ads, author bios, related article lists.
  Skip: claims unrelated to the sub-question being researched.
  Skip: opinions presented without supporting evidence.
 
RULE 6 — MAX 5 CLAIMS PER SOURCE
  Quality over quantity. 3 precise claims beat 10 vague ones.
 
━━━ OUTPUT FORMAT ━━━
 
Return ONLY valid JSON. No markdown fences. No text outside JSON.
 
{
  "sub_question": "<the sub-question this source was searched for>",
  "source_url": "<url>",
  "source_credibility": "<high|medium|low>",
  "claims": [
    {
      "claim": "<rewritten factual claim in your own words>",
      "confidence": <0.0 to 1.0>,
      "direct_quote": "<optional: ≤15-word verbatim fragment if precision requires it>"
    }
  ]
}
""".strip()
 
 
SUMMARIZER_USER_TEMPLATE = """
Sub-question being researched: {sub_question}
Source URL: {source_url}
Source content:
{content}
""".strip()


async def summarizer_agent(
    llm: LLMClient,
    query: str,
    search_results: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    quality_results = filter_search_results_by_domain(search_results)
    if not quality_results:
        return []

    compact_results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("snippet", "")[:500],
            "sub_question": item.get("sub_question", ""),
        }
        for item in quality_results[:16]
    ]

    user_prompt = (
        f"Research query: {query}\n\n"
        f"Search evidence: {compact_results}\n\n"
        "Extract only high-quality claims from reliable sources as JSON in this schema: "
        '{"facts": [{"claim": "...", "source": "https://...", "confidence": 0.0}]}'
        ". Keep claims concise, source-grounded, and normalized. Ignore weak, promotional, or opinion-blog sources."
    )

    try:
        payload = await llm.generate_json(SUMMARIZER_SYSTEM_PROMPT, user_prompt)
        facts = payload.get("facts", []) if isinstance(payload, dict) else []
    except Exception:
        facts = []

    cleaned: List[Dict[str, Any]] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        claim = normalize_claim_text(str(fact.get("claim", "")))
        source = str(fact.get("source", "")).strip()
        source_score = source_reliability_score(source)
        if source_score < 0.55:
            continue

        model_confidence = clamp_confidence(fact.get("confidence", 0.0))
        confidence = clamp_confidence((0.75 * model_confidence) + (0.25 * source_score))
        if claim and source:
            cleaned.append({"claim": claim, "source": source, "confidence": confidence})

    if cleaned:
        return dedupe_semantic_facts(cleaned)

    # Heuristic fallback when the model output is malformed or empty.
    fallback: List[Dict[str, Any]] = []
    for item in quality_results[:6]:
        snippet = item.get("snippet", "").strip()
        if not snippet:
            continue
        fallback.append(
            {
                "claim": normalize_claim_text(snippet[:180]),
                "source": item.get("url", ""),
                "confidence": clamp_confidence((0.30 * 0.45) + (0.70 * source_reliability_score(item.get("url", "")))),
            }
        )
    return dedupe_semantic_facts(fallback)
