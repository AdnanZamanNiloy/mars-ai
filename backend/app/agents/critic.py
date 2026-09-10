from typing import Any, Dict, List

from app.core.logging import get_logger

from app.agents.evidence_utils import dedupe_semantic_facts, extract_domain, filter_facts_by_domain, normalize_claim_text
from app.core.degradation import record_fallback
from app.core.llm import LLMClient, clamp_confidence
from app.core.schemas import CriticVerdictModel

logger = get_logger(__name__)


CRITIC_SYSTEM_PROMPT = """
You are the Critic Agent in a multi-agent research pipeline.
You are the quality gate. Your decision to pass or loop determines
whether the Writer produces a complete, trustworthy report.

You must be demanding but fair:
  - Too strict → unnecessary loops, wasted API calls, slow output
  - Too lenient → shallow reports, missing perspectives, low confidence

━━━ EVALUATION CRITERIA ━━━

1) COMPLETENESS
   Would a knowledgeable reader consider the original query answered?
   Is any obvious major angle of the query entirely absent?

2) SUBSTANTIVE MATERIAL
   Is there enough reliable, non-redundant material to write a clear,
   well-supported answer (e.g. a definition with mechanisms and
   examples, not just fragments)?

3) SOURCE RELIABILITY
   Are the extracted facts backed by credible, authoritative sources?

4) REDUNDANCY / FRAGMENTATION
   Is the evidence mostly duplicated or too fragmented to synthesize?

━━━ RED TEAM QUESTIONS (adversarial) ━━━

Before deciding, attack the evidence yourself:

  RQ1 — What assumption in the current evidence is WEAKEST, i.e. most
        likely to be wrong or unrepresentative?
  RQ2 — What alternative explanation or competing claim would
        INVALIDATE the current conclusion if true?
  RQ3 — What important counter-evidence is conspicuously ABSENT from
        the retrieved material?

If is_sufficient is false, the reason field MUST explicitly name at
least one weak assumption (RQ1), a potentially invalidating alternative
(RQ2), or a missing counter-evidence (RQ3) — not just "insufficient
coverage".

━━━ DECISION RULES ━━━

  - is_sufficient = true  ONLY if the evidence clearly supports a
    complete answer for the original query.
  - If insufficient, improved_queries must contain 1-3 specific,
    search-ready strings targeting the exact gaps — never vague
    suggestions like "search for more information about X".
      BAD  → "Find more about knowledge types"
      GOOD → "metacognitive knowledge definition examples learning research"
  - Be conservative: when in doubt, mark insufficient and say what
    is missing in the reason field.

━━━ OUTPUT FORMAT ━━━

Return ONLY valid JSON. No markdown fences. No text outside JSON.

{
  "is_sufficient": <true|false>,
  "reason": "<one or two sentences: what is complete or what is missing>",
  "improved_queries": ["<targeted search query to fix gap 1>", "..."],
  "confidence": <0.0 to 1.0, your confidence in the current evidence>
}
""".strip()
 


async def critic_agent(
    llm: LLMClient,
    query: str,
    facts: List[Dict[str, Any]],
    iteration: int,
    max_iterations: int,
    contradictions: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    quality_facts = dedupe_semantic_facts(filter_facts_by_domain(facts))
    if not quality_facts:
        return {
            "is_sufficient": False,
            "reason": "No reliable evidence extracted.",
            "improved_queries": [f"Latest evidence for: {query}", f"Key statistics: {query}"],
            "confidence": 0.2,
        }

    contradiction_block = ""
    if contradictions:
        listed = "\n".join(
            f"  - \"{c.get('claim_a', '')[:120]}\" ({c.get('source_a', '')}) vs "
            f"\"{c.get('claim_b', '')[:120]}\" ({c.get('source_b', '')})"
            for c in contradictions[:3]
        )
        contradiction_block = (
            f"\nKnown contradictions between sources (acknowledge these in your "
            f"reason — do NOT silently ignore them):\n{listed}\n"
        )

    user_prompt = (
        f"Main query: {query}\n"
        f"Current iteration: {iteration}/{max_iterations}\n"
        f"Extracted reliable facts: {quality_facts[:10]}\n"
        f"{contradiction_block}\n"
        "Evaluate using these criteria:\n"
        "1) Is the answer complete?\n"
        "2) Is there enough material to write a clear definition?\n"
        "3) Are sources reliable?\n"
        "4) Is information redundant or fragmented?\n\n"
        "Also run the Red Team checks from your instructions: name the weakest\n"
        "assumption, a potentially invalidating alternative, and any missing\n"
        "counter-evidence in your reason when the evidence is insufficient.\n\n"
        "Return JSON: "
        '{"is_sufficient": true/false, "reason": "...", "improved_queries": ["..."], "confidence": 0.0}'
    )

    try:
        payload = await llm.generate_json(
            CRITIC_SYSTEM_PROMPT,
            user_prompt,
            response_model=CriticVerdictModel,
        )
    except Exception as exc:
        logger.warning("[Critic] LLM call failed, treating as insufficient", exc_info=exc)
        record_fallback("critic")
        payload = {}

    is_sufficient = bool(payload.get("is_sufficient", False)) if isinstance(payload, dict) else False
    reason = str(payload.get("reason", "Insufficient assessment.")) if isinstance(payload, dict) else "Insufficient assessment."
    improved_queries = payload.get("improved_queries", []) if isinstance(payload, dict) else []
    confidence = clamp_confidence(payload.get("confidence", 0.4) if isinstance(payload, dict) else 0.4)

    cleaned_queries: List[str] = []
    for item in improved_queries:
        if isinstance(item, str) and item.strip():
            cleaned_queries.append(item.strip())

    reason_lower = reason.lower()
    if any(token in reason_lower for token in ["incomplete", "lacks a clear definition", "insufficient"]):
        is_sufficient = False

    # Deterministic guardrails improve consistency when model judgments are noisy.
    # Synthesis gate: never call evidence sufficient without a verified,
    # multi-source foundation — a stronger model verdict cannot waive this.
    # NOTE: quality_facts above are rebuilt by the prefilters WITHOUT the
    # "verified" key, so verified/source standing is read back off the
    # incoming facts by normalized claim text.
    standing: Dict[str, Dict[str, Any]] = {}
    for f in facts:
        key = normalize_claim_text(str(f.get("claim", "")))
        if key and key not in standing:
            standing[key] = f
    verified_count = sum(
        1 for q in quality_facts
        if standing.get(normalize_claim_text(str(q.get("claim", ""))), {}).get("verified")
    )
    distinct_sources = {
        extract_domain(str(standing.get(normalize_claim_text(str(q.get("claim", ""))), {}).get("source", "")))
        for q in quality_facts
    } - {""}
    has_definition = any(" is " in str(f.get("claim", "")).lower() for f in quality_facts[:5])
    avg_fact_conf = sum(float(f.get("confidence", 0.0) or 0.0) for f in quality_facts) / max(1, len(quality_facts))
    min_facts_required = 4

    if (len(quality_facts) < min_facts_required or not has_definition or avg_fact_conf < 0.74
            or verified_count < 1 or len(distinct_sources) < 2):
        is_sufficient = False
        confidence = min(confidence, 0.58)
        if not cleaned_queries:
            cleaned_queries = [
                f"Authoritative definition of: {query}",
                f"Peer-reviewed or encyclopedia explanation of: {query}",
            ]
        reason = (
            "Evidence is still incomplete for a high-quality synthesis; "
            "the answer lacks enough reliable, non-redundant coverage, a clear definition, "
            "or a verified multi-source foundation "
            f"(facts={len(quality_facts)}, verified={verified_count}, sources={len(distinct_sources)})."
        )
    elif avg_fact_conf >= 0.80 and verified_count >= 1 and len(distinct_sources) >= 2:
        is_sufficient = True
        confidence = max(confidence, 0.78)

    return {
        "is_sufficient": is_sufficient,
        "reason": reason,
        "improved_queries": cleaned_queries[:5],
        "confidence": confidence,
    }
