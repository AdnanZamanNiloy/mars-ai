from typing import Any, Dict, List

from app.core.logging import get_logger

from app.agents.evidence_utils import (
    claim_query_overlap,
    clean_snippet_text,
    dedupe_semantic_facts,
    filter_search_results_by_domain,
    MIN_QUERY_OVERLAP,
    source_reliability_score,
    split_into_sentences,
)
from app.core.llm import LLMClient, clamp_confidence
from app.core.schemas import SummarizerFactsModel
from app.core.cache import cache_key, get_cache
from app.core.degradation import record_fallback

logger = get_logger(__name__)

PROMPT_VERSION = "summarizer-v14"  # BUMP on any claim-shape change (cleaning,
# fields, thresholds): the cache key embeds this, and stale entries would
# otherwise serve pre-fix claims indefinitely (AGENTS.md 4.10).
# v3: full fetched page content feeds the prompt (was snippet-only) and
# facts carry their sub_question for grouped synthesis.
# v4: excerpt-seam hygiene (nav headers, [...]/... splices, link-title
# parens, markdown heading marks) drops page-furniture fragments.
# v6: LaTeX-soup and boilerplate-lead drops (unreadable math markup,
# "Use this page to" instructional leads).
# v7: bare "Title: subtitle" heading claims dropped.
# v8: title rule fixed (trailing period no longer saves a title; title-case
# density separates headings from headed sentences) + question-shaped claims
# ("How does X work?") dropped — questions are not findings.
# v9: article self-reference leads ("This article will explore ...") dropped.
# v10: unbalanced-paren cuts dropped (mid-word tails past the stub check).
# v11: dangling function-word endings ("..., including") dropped + HTML
# entity decoding in both cleaners (&quot seen live).
# v12: leading blockquote markers stripped + full cache break (a stale
# entry from the overlapping-backend incident served pre-rule facts).
# v13: dangling rule covers trailing-period variants ("..., including." —
# normalize strips the period downstream, resurrecting the cut).
# v14: output schema in the system prompt now matches the parser
# ({"facts": [...]}, was a contradictory per-source claims[] schema), the
# prompt footprint shrank (10 sources, shorter excerpts — free-tier TPM/TPD
# budgets were being exhausted by 16-source prompts), and the heuristic
# fallback ranks candidates by query/sub-question overlap (min 0.2) instead
# of taking the first sentences that clear a 0.15 bar.

# Specialist prompt additions (Phase 3.1): routed by the delegation
# contract's domain via AgentContext.specialist_role(). Each specialist
# is the generic summarizer plus a domain evidence-preference overlay —
# same output contract, same fallback path (AGENTS.md 4.7 shape kept).
SPECIALIST_PROMPT_ADDITIONS = {
    "financial": (
        "\n\nSPECIALIST FOCUS — FINANCIAL:\n"
        "Prefer primary financial sources: central-bank releases, IMF/World Bank\n"
        "data, audited filings, regulatory disclosures. Prioritize figures with\n"
        "units, periods, and currency. Treat analyst opinions as low-confidence\n"
        "unless backed by reported numbers. Flag when costs are nominal vs real,\n"
        "and note the fiscal year of any monetary figure."
    ),
    "technical": (
        "\n\nSPECIALIST FOCUS — TECHNICAL:\n"
        "Prefer primary technical sources: papers, specs, benchmarks, official\n"
        "docs, reproducible results. Include methodology details (setup,\n"
        "dataset, version) when extracting performance claims. Treat marketing\n"
        "benchmarks as low-confidence. Preserve version numbers and dates."
    ),
    "market": (
        "\n\nSPECIALIST FOCUS — MARKET:\n"
        "Prefer market-research firms, industry associations, and government\n"
        "statistics over news aggregation. Distinguish market size, share, and\n"
        "growth-rate claims, and keep their scope (region, segment, period)\n"
        "attached to the number. Note when sources disagree on scope."
    ),
    "legal": (
        "\n\nSPECIALIST FOCUS — LEGAL:\n"
        "Prefer primary legal sources: statutes, regulations, case law, official\n"
        "guidance. Quote operative language precisely and always attach the\n"
        "jurisdiction. Distinguish binding authority from commentary, and flag\n"
        "when a rule varies by jurisdiction or is under appeal."
    ),
    "scientific": (
        "\n\nSPECIALIST FOCUS — SCIENTIFIC:\n"
        "Prefer peer-reviewed studies, preprints with methods sections, and\n"
        "official datasets. Capture study design (sample, controls, effect\n"
        "size) with every finding. Treat single-study results as provisional\n"
        "and note replication status when sources discuss it."
    ),
    "policy": (
        "\n\nSPECIALIST FOCUS — POLICY:\n"
        "Prefer government releases, legislative texts, and think-tank analyses\n"
        "with stated methodology. Attach the jurisdiction and status (proposed,\n"
        "enacted, repealed) to every policy claim. Distinguish the policy text\n"
        "itself from analysts' predictions about its effects."
    ),
    "academic": (
        "\n\nSPECIALIST FOCUS — ACADEMIC:\n"
        "Prefer peer-reviewed papers, monographs, and conference proceedings.\n"
        "Attribute schools of thought and name key authors with dates. Keep\n"
        "interpretive claims separate from textual evidence, and flag where\n"
        "scholars disagree rather than smoothing over the debate."
    ),
    "general": "",  # no overlay for unrouted domains
}


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
  "facts": [
    {
      "claim": "<rewritten factual claim in your own words>",
      "source": "<the source URL the claim came from>",
      "confidence": <0.0 to 1.0>
    }
  ]
}

The response is ONE facts list across all provided sources. Every fact's
"source" MUST be one of the URLs given in the input. Optionally attach a
"direct_quote" (≤15-word verbatim fragment) when precision requires it.
 """.strip()


def specialist_system_prompt(role: str = "general") -> str:
    """Generic summarizer prompt + the role's domain overlay (3.1)."""
    overlay = SPECIALIST_PROMPT_ADDITIONS.get(role, "")
    return SUMMARIZER_SYSTEM_PROMPT + overlay


async def summarizer_agent(
    llm: LLMClient,
    query: str,
    search_results: List[Dict[str, str]],
    specialist_role: str = "general",
) -> List[Dict[str, Any]]:
    quality_results = filter_search_results_by_domain(search_results)
    if not quality_results:
        return []

    system_prompt = specialist_system_prompt(specialist_role)

    # Prompt footprint is a real budget line: this agent runs once per
    # sub-question per pass, and 16 sources x (snippet + full content)
    # measured ~6K tokens per call — enough to exhaust free-tier daily
    # token quotas within a handful of runs (observed live on Groq TPD).
    # 10 sources with 800-char excerpts keeps extraction quality while
    # roughly halving tokens per call; ranking already put the best
    # sources first.
    compact_results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("snippet", "")[:400],
            # Full fetched page text (when the fetch loop got it) — the LLM
            # extracts far more claims from pages than from snippets alone,
            # mirroring upstream's summarize-scraped-pages shape.
            "content": item.get("content", "")[:800],
            "sub_question": item.get("sub_question", ""),
        }
        for item in quality_results[:10]
    ]
    sub_question_by_url = {
        str(item.get("url", "")): str(item.get("sub_question", "") or "")
        for item in quality_results
        if item.get("url")
    }

    user_prompt = (
        f"Research query: {query}\n\n"
        f"Search evidence: {compact_results}\n\n"
        "Extract only high-quality claims from reliable sources as JSON in this schema: "
        '{"facts": [{"claim": "...", "source": "https://...", "confidence": 0.0}]}'
        ". Prefer the full page content over the snippet when both are present. "
        "Keep claims concise, source-grounded, and normalized. Ignore weak, promotional, or opinion-blog sources."
    )

    cache = get_cache(llm.settings)
    key = cache_key(
        "summarize_facts",
        PROMPT_VERSION,
        query,
        tuple(sorted(item.get("url", "") for item in compact_results)),
    )
    try:
        cached = cache.get(key)
    except Exception as exc:
        logger.warning("[Summarizer] cache read failed: %s", exc, exc_info=exc)
        cached = None

    if cached is not None:
        logger.info("[Summarizer] cache hit (%s)", PROMPT_VERSION)
        facts = cached
    else:
        logger.info("[Summarizer] cache miss (%s)", PROMPT_VERSION)
        try:
            payload = await llm.generate_json(
                system_prompt,
                user_prompt,
                response_model=SummarizerFactsModel,
            )
            facts = payload.get("facts", []) if isinstance(payload, dict) else []
        except Exception as exc:
            logger.warning("[Summarizer] LLM call failed, using heuristic fallback", exc_info=exc)
            facts = []
        # Cache only successful, non-empty extractions. Caching the empty
        # list on provider failure poisoned the key for a full TTL hour —
        # the summarizer kept "recovering" from cache after the provider
        # was healthy again, silently extending the degradation window.
        if facts:
            try:
                cache.set(key, facts, expire=llm.settings.cache_ttl_sec)
            except Exception as exc:
                logger.warning("[Summarizer] cache write failed, continuing uncached: %s", exc, exc_info=exc)

    cleaned: List[Dict[str, Any]] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        # Model output gets the same fragment hygiene as snippets: echoed
        # truncations ("...from a large dat") must not enter the pool.
        claim = clean_snippet_text(str(fact.get("claim", "")), min_chars=25)
        if not claim:
            continue
        source = str(fact.get("source", "")).strip()
        source_score = source_reliability_score(source)
        if source_score < 0.55:
            continue
        if claim_query_overlap(query, claim) < MIN_QUERY_OVERLAP:
            continue

        model_confidence = clamp_confidence(fact.get("confidence", 0.0))
        confidence = clamp_confidence((0.75 * model_confidence) + (0.25 * source_score))
        if claim and source:
            cleaned.append({"claim": claim, "source": source, "confidence": confidence,
                            "agent": specialist_role,
                            "sub_question": sub_question_by_url.get(source, "")})

    if cleaned:
        return dedupe_semantic_facts(cleaned)

    # Heuristic fallback when the model output is malformed or empty.
    # Reaching here means the model contributed nothing usable. Work
    # sentence-by-sentence from full fetched content first (far richer than
    # snippets) and cover more sources — a thin fallback is what makes
    # degraded answers look empty. Per-sentence gates mean one furniture
    # fragment vetoes its sentence, never the whole page.
    #
    # Relevance: candidates are scored against BOTH the top-level query and
    # the fact's own sub-question (the sub-question is the more specific
    # relevance target), ranked, and only the best sentences per source are
    # kept. Taking the first sentences that clear a bare 0.15 overlap let
    # off-topic passages through (live: a UN transcript's Malawi
    # electrification paragraph surfaced as a top claim on a
    # Bangladesh-nuclear-vs-solar query).
    record_fallback("summarizer")
    MIN_FALLBACK_OVERLAP = 0.2
    fallback: List[Dict[str, Any]] = []
    for item in quality_results[:12]:
        raw = (item.get("content", "") or "")[:1500] or item.get("snippet", "")
        if not raw:
            continue
        sub_q = str(item.get("sub_question", "") or "")
        scored: List[tuple[float, str]] = []
        for sent in split_into_sentences(raw):
            claim = clean_snippet_text(sent)
            if not claim:
                continue
            relevance = max(
                claim_query_overlap(query, claim),
                claim_query_overlap(sub_q, claim) if sub_q else 0.0,
            )
            if relevance < MIN_FALLBACK_OVERLAP:
                continue
            scored.append((relevance, claim))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        for relevance, claim in scored[:3]:
            fallback.append(
                {
                    "claim": claim,
                    "source": item.get("url", ""),
                    "confidence": clamp_confidence((0.30 * 0.45) + (0.70 * source_reliability_score(item.get("url", "")))),
                    "agent": specialist_role,
                    "sub_question": sub_q,
                }
            )
    return dedupe_semantic_facts(fallback)
