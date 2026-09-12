"""Critic Agent — the quality gate that decides whether to loop or write.

What changed and why
--------------------
1. IT NO LONGER RECONSTRUCTS VERIFICATION BY STRING MATCHING. The old code built
   a `standing` dict keyed on normalized claim text, because the prefilters it
   called (`filter_facts_by_domain` -> `dedupe_semantic_facts`) stripped the
   `verified` key off every fact. With that dedup bug fixed in evidence_utils,
   verification standing survives, and the critic reads it directly. The
   workaround was also subtly wrong: two claims normalizing to the same text
   collapsed into whichever came first, so verified counts drifted.

2. THE GATES ARE NO LONGER ARBITRARY CONSTANTS. `avg_confidence < 0.74` failed
   nearly every real pool by a hundredth, which is why runs went to the
   iteration ceiling and every report was stamped "incomplete" regardless of
   quality. Gates now check what actually matters — verified evidence from
   independent domains, coverage of the planned angles, and each contract's own
   `minimum_sources` (a contract field that nothing had ever enforced) — and the
   confidence threshold comes from the mode's target rather than a magic number.

3. IT CONSUMES MEASURED CONFIDENCE INSTEAD OF PRODUCING IT. When a
   ConfidenceReport is supplied the critic reports it rather than substituting
   the model's self-assessment, and it can only lower it.

4. FOLLOW-UP QUERIES ARE DEDUPLICATED AGAINST WHAT WAS ALREADY SEARCHED. The
   critic used to re-emit the same `improved_queries` every pass with no memory,
   so a stubborn gap produced identical searches at full cost forever.

The public contract is unchanged: the returned dict still has `is_sufficient`,
`reason`, `improved_queries` and `confidence`. New keys are additive.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from app.core.degradation import record_fallback
from app.core.llm import LLMClient, clamp_confidence
from app.core.logging import get_logger
from app.core.schemas import CriticVerdictModel

from app.agents.confidence import ConfidenceReport, SUFFICIENCY_THRESHOLD
from app.agents.contradiction import contradiction_followups, summarize_contradictions
from app.agents.evidence_utils import (
    dedupe_semantic_facts,
    evidence_stats,
    extract_domain,
    filter_facts_by_domain,
)
from app.agents.stopping import coverage_gaps

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

3) SOURCE RELIABILITY AND INDEPENDENCE
   Are the facts backed by credible sources, and by more than one
   organisation? Ten claims from one site is one source, not ten.

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
  - Queries listed as ALREADY SEARCHED must not be repeated or
    trivially reworded; a repeat costs a full pass and returns the same
    evidence.
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


DEFINITIONAL_QUERY_RE = re.compile(
    r"^\s*(what\s+is|what\s+are|define|explain)\b", re.IGNORECASE
)

# Minimum independent domains before any pool can be called sufficient. Two is
# the floor, not a target: one publisher's framing cannot be distinguished from
# consensus.
MIN_DISTINCT_DOMAINS = 2
MIN_FACTS_REQUIRED = 4


async def critic_agent(
    llm: LLMClient,
    query: str,
    facts: List[Dict[str, Any]],
    iteration: int,
    max_iterations: int,
    contradictions: List[Dict[str, Any]] | None = None,
    query_type: str = "",
    *,
    confidence_report: Optional[ConfidenceReport] = None,
    plan: Optional[Sequence[Dict[str, Any]]] = None,
    searched_queries: Sequence[str] = (),
    confidence_target: Optional[float] = None,
    redteam_survival: Optional[float] = None,
) -> Dict[str, Any]:
    """Judge the evidence pool. Returns the verdict dict the workflow consumes.

    All new arguments are keyword-only and optional, so the existing call
    signature keeps working unchanged.
    """
    target = float(
        confidence_target if confidence_target is not None else SUFFICIENCY_THRESHOLD
    )

    # Quality prefilters. dedupe now PRESERVES verification metadata, so
    # `quality_facts` is directly inspectable — no claim-text reconstruction.
    quality_facts = dedupe_semantic_facts(filter_facts_by_domain(facts or []))
    if not quality_facts:
        return {
            "is_sufficient": False,
            "reason": "No reliable evidence extracted.",
            "improved_queries": [
                f"Latest evidence for: {query}",
                f"Key statistics: {query}",
            ],
            "confidence": 0.2,
            "gaps": ["no reliable evidence in the pool"],
            "gate_failures": ["facts=0"],
            "stats": evidence_stats([]),
        }

    stats = evidence_stats(quality_facts)
    conflict_summary = summarize_contradictions(contradictions or [])

    contradiction_block = ""
    if contradictions:
        listed = "\n".join(
            f"  - \"{str(c.get('claim_a', ''))[:120]}\" ({c.get('source_a', '')}) vs "
            f"\"{str(c.get('claim_b', ''))[:120]}\" ({c.get('source_b', '')})"
            for c in contradictions[:3]
        )
        contradiction_block = (
            "\nKnown contradictions between sources (acknowledge these in your "
            f"reason — do NOT silently ignore them):\n{listed}\n"
        )

    searched_block = ""
    if searched_queries:
        searched_block = (
            "\nALREADY SEARCHED (do not repeat or trivially reword):\n"
            + "\n".join(f"  - {q}" for q in list(searched_queries)[:12])
            + "\n"
        )

    # The model sees measured facts about the pool, not just the claims. Asking
    # it to judge sufficiency without telling it how many independent domains it
    # is looking at is asking it to guess.
    measurement_block = (
        f"\nMeasured evidence state:\n"
        f"  claims={stats['total']} verified={stats['verified']} "
        f"domains={stats['distinct_domains']} primary_sources={stats['primary_documents']} "
        f"corroborated={stats['corroborated']} angles={stats['axes_covered']}\n"
        f"  cross-source conflicts={conflict_summary['cross_source']} "
        f"(severe={conflict_summary['severe']})\n"
    )

    user_prompt = (
        f"Main query: {query}\n"
        f"Current iteration: {iteration}/{max_iterations}\n"
        f"Extracted reliable facts: {_compact_facts(quality_facts)}\n"
        f"{measurement_block}"
        f"{contradiction_block}"
        f"{searched_block}\n"
        "Evaluate using these criteria:\n"
        "1) Is the answer complete?\n"
        "2) Is there enough material to write a clear, well-supported answer?\n"
        "3) Are sources reliable AND independent of each other?\n"
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

    if not isinstance(payload, dict):
        payload = {}

    is_sufficient = bool(payload.get("is_sufficient", False))
    reason = str(payload.get("reason", "Insufficient assessment.") or "Insufficient assessment.")
    improved_queries = payload.get("improved_queries", []) or []
    model_confidence = clamp_confidence(payload.get("confidence", 0.4))

    cleaned_queries: List[str] = []
    for item in improved_queries:
        if isinstance(item, str) and item.strip():
            cleaned_queries.append(re.sub(r"\s+", " ", item).strip())

    # Contradiction-resolving searches are worth more than generic gap-filling,
    # so they lead the follow-up list.
    for query_text in contradiction_followups(contradictions or [], limit=2):
        if query_text not in cleaned_queries:
            cleaned_queries.insert(0, query_text)

    reason_lower = reason.lower()
    if any(t in reason_lower for t in ("incomplete", "lacks a clear definition", "insufficient")):
        is_sufficient = False

    # ------------------------------------------------------------------
    # Deterministic gates. Model judgement cannot waive these; it can only
    # be overridden BY them. Each gate corresponds to a way a report can be
    # confidently wrong.
    # ------------------------------------------------------------------
    requires_definition = (
        query_type == "factual"
        or (not query_type and bool(DEFINITIONAL_QUERY_RE.match(query or "")))
    )
    has_definition = any(
        " is " in str(f.get("claim", "")).lower() for f in quality_facts[:6]
    )

    gaps = list(coverage_gaps(plan or (), quality_facts))
    gate_failures: List[str] = []

    if stats["total"] < MIN_FACTS_REQUIRED:
        gate_failures.append(f"facts={stats['total']}<{MIN_FACTS_REQUIRED}")
    if requires_definition and not has_definition:
        gate_failures.append("no definitional claim")
    if stats["verified"] < 1:
        gate_failures.append("verified=0")
    if stats["distinct_domains"] < MIN_DISTINCT_DOMAINS:
        gate_failures.append(f"domains={stats['distinct_domains']}<{MIN_DISTINCT_DOMAINS}")
    if conflict_summary["severe"]:
        gate_failures.append(f"severe_conflicts={conflict_summary['severe']}")
    if gaps:
        gate_failures.append(f"uncovered_angles={len(gaps)}")

    # Measured confidence, when available, replaces the model's self-report.
    if confidence_report is not None:
        confidence = confidence_report.overall
        if confidence < target:
            gate_failures.append(
                f"confidence={confidence:.2f}<target={target:.2f}"
            )
    else:
        confidence = model_confidence
        if stats["avg_confidence"] < 0.60:
            gate_failures.append(f"avg_fact_conf={stats['avg_confidence']:.2f}<0.60")

    if redteam_survival is not None and redteam_survival < 0.45:
        gate_failures.append(f"redteam_survival={redteam_survival:.2f}<0.45")

    if gate_failures:
        is_sufficient = False
        confidence = min(confidence, 0.65)
        if not cleaned_queries:
            cleaned_queries = _default_followups(query, gaps, requires_definition)
        reason = _explain_failure(reason, gate_failures, gaps, stats)
    elif confidence >= target:
        # Everything measurable passed and confidence is at target: pass the
        # gate even if the model hedged. A model hedging on evidence that
        # satisfies every objective criterion is what produced needless loops.
        is_sufficient = True

    if not is_sufficient and not cleaned_queries:
        # Model call failed (or hedged) without gates firing: the loop still
        # needs something actionable to search, or the next pass is empty.
        cleaned_queries = _default_followups(query, gaps, requires_definition)

    return {
        "is_sufficient": is_sufficient,
        "reason": reason,
        "improved_queries": cleaned_queries[:5],
        "confidence": clamp_confidence(confidence),
        # --- additive, for the trace, the UI and the stopping controller ---
        "gaps": gaps,
        "gate_failures": gate_failures,
        "stats": stats,
        "model_confidence": model_confidence,
        "conflicts": conflict_summary,
    }


def _compact_facts(facts: Sequence[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    """Trim facts for the prompt.

    The previous version passed `quality_facts[:10]` as full dicts, which after
    the metadata-preservation fix would include verification checks, numbers and
    quotes — several hundred wasted tokens per call for fields the critic does
    not reason over.
    """
    return [
        {
            "claim": str(f.get("claim", ""))[:220],
            "domain": extract_domain(str(f.get("source", ""))),
            "confidence": round(float(f.get("confidence", 0.0) or 0.0), 2),
            "verified": bool(f.get("verified")),
            "primary": bool(f.get("is_primary")),
        }
        for f in list(facts)[:limit]
    ]


def _default_followups(
    query: str, gaps: Sequence[str], requires_definition: bool
) -> List[str]:
    """Concrete follow-ups when the model gave none.

    Derived from the actual gaps where possible. The old fallback emitted two
    definition-hunting queries regardless of what was missing, so a run short on
    statistics searched for definitions it already had.
    """
    out: List[str] = []
    for gap in list(gaps)[:2]:
        angle = gap.split(":", 1)[-1].strip()
        if angle:
            out.append(angle)
    if requires_definition and not out:
        out.append(f"Authoritative definition of: {query}")
    if not out:
        out = [
            f"{query} official statistics data",
            f"{query} criticism limitations counter-evidence",
        ]
    return out


def _explain_failure(
    model_reason: str,
    gate_failures: Sequence[str],
    gaps: Sequence[str],
    stats: Dict[str, Any],
) -> str:
    """A reason a human can act on, not a generic 'still incomplete'."""
    parts: List[str] = []
    if model_reason and "insufficient assessment" not in model_reason.lower():
        parts.append(model_reason.rstrip("."))
    detail: List[str] = []
    if any(f.startswith("domains=") for f in gate_failures):
        detail.append(
            f"evidence comes from only {stats['distinct_domains']} independent domain(s)"
        )
    if "verified=0" in gate_failures:
        detail.append("no claim verified against its cited source")
    if any(f.startswith("severe_conflicts=") for f in gate_failures):
        detail.append("unresolved severe contradiction between sources")
    if gaps:
        detail.append(f"{len(gaps)} planned angle(s) still unsourced: {gaps[0]}")
    if any(f.startswith("confidence=") for f in gate_failures):
        detail.append("measured confidence below the mode's target")
    if detail:
        parts.append("Blocking: " + "; ".join(detail))
    parts.append(f"({', '.join(gate_failures)})")
    return ". ".join(parts) + "."
