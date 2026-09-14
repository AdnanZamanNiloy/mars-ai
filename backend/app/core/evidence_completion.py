"""Evidence-completion targeting — spend the remaining research budget on the
weakest, highest-impact claims instead of on more of the same.

Why this module exists
----------------------
The live deep runs measured corroboration perfectly and then issued follow-up
searches generically: whichever uncorroborated claims grading happened to
return first got a query, regardless of whether the claim was central to the
answer or a peripheral remark. With a bounded expansion budget (a few passes),
that ordering IS the allocation policy, and it was effectively arbitrary.

This module makes the allocation policy explicit and deterministic:

  impact(claim)  = quantitative        (numbers need a second publisher)
                 + used in the summary (executive-summary / key-findings facts)
                 + corroboration need  (single-source definitional claims)
                 + primary-source need (the dimension is thin on primarysources)

`rank_completion_targets` returns the single-source (`needs_corroboration`)
claims ordered by that impact score, highest first, so the expansion routing
can aim its limited budget at the claims that would change the answer.

Pure, LLM-free, and total: an empty/unreadable pool yields [] and never raises.
It does NOT introduce a research loop of its own — callers feed the returned
claims into the existing corroboration/counter-evidence query channel.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

from app.core.logging import get_logger

logger = get_logger(__name__)

# Impact weights. Quantitative claims lead because a wrong/unverified number is
# the highest-consequence failure in a research report; summary use matters
# because those claims are the report's headline; a primary-source need is the
# procurement target the source ledger measures.
IMPACT_QUANTITATIVE = 3
IMPACT_IN_SUMMARY = 3
IMPACT_HIGH_CORROBORATION_NEED = 2
IMPACT_PRIMARY_SOURCE_NEED = 2


def _normalize(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _summary_claims(ctx: Optional[Dict[str, Any]]) -> Set[str]:
    """Normalized claims named in the executive summary / key-findings inputs.

    Accepts either explicit claim strings or fact dicts under the keys the
    workflow already computes (`summary_claims`, `key_findings`,
    `headline_claims`). Missing context is simply an empty set — never a raise.
    """
    if not isinstance(ctx, dict):
        return set()
    out: Set[str] = set()
    for key in ("summary_claims", "key_findings", "headline_claims", "executive_summary_claims"):
        for item in ctx.get(key) or []:
            if isinstance(item, dict):
                text = str(item.get("claim", "") or "")
            else:
                text = str(item or "")
            if text:
                out.add(_normalize(text))
    return out


def claim_impact(
    record: Dict[str, Any],
    *,
    summary_claims: Optional[Set[str]] = None,
    primary_thin_dimensions: Optional[Set[str]] = None,
) -> int:
    """Deterministic impact score for one graded claim record.

    `record` is an EvidenceRecord dict (the `evidence` value from grade_facts).
    Higher = spend the next research query here.
    """
    if not isinstance(record, dict):
        return 0
    score = 0
    if record.get("has_numbers"):
        score += IMPACT_QUANTITATIVE
    claim = _normalize(str(record.get("claim", "") or ""))
    if summary_claims and claim and claim in summary_claims:
        score += IMPACT_IN_SUMMARY
    if record.get("needs_corroboration"):
        score += IMPACT_HIGH_CORROBORATION_NEED
    if primary_thin_dimensions:
        dim = str(record.get("domain", "") or "").strip().lower()
        # The record's `domain` is the publisher domain, not the research
        # dimension; callers pass the thin dimension labels keyed by claim.
        if dim and dim in primary_thin_dimensions:
            score += IMPACT_PRIMARY_SOURCE_NEED
    if int(record.get("contradiction_count", 0) or 0) > 0:
        # A contradicted claim is not a completion target (it needs resolution,
        # handled by the counter-evidence channel); it must not double-count.
        score -= 1
    return score


def rank_completion_targets(
    facts: Sequence[Dict[str, Any]],
    contradictions: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    summary_claims: Optional[Sequence[Any]] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Single-source claims ordered by impact, highest first.

    Returns the graded EvidenceRecord dicts whose `needs_corroboration` flag is
    set, annotated with an `impact` integer (for logging/tests) and sorted by
    (impact desc, claim text) so ordering is stable across runs. Claims already
    independently corroborated (count >= 2) are excluded — they are done.
    Contradicted claims are deprioritized but still returned last, so no gap is
    silently dropped.

    Total and deterministic: a grading failure logs and returns [].
    """
    pool = [f for f in (facts or []) if isinstance(f, dict) and str(f.get("claim", "")).strip()]
    if not pool:
        return []
    try:
        from app.core.evidence_grade import grade_facts

        graded = grade_facts(pool, contradictions=contradictions or [])
    except Exception as exc:  # a grading bug must never break routing
        logger.warning("completion_grading_failed", error=str(exc), exc_info=exc)
        return []

    summaries = {_normalize(s) for s in (summary_claims or []) if str(s or "").strip()}
    targets: List[Dict[str, Any]] = []
    for g in graded:
        if not isinstance(g, dict):
            continue
        ev = g.get("evidence")
        if not isinstance(ev, dict):
            continue
        if not ev.get("needs_corroboration"):
            continue
        record = dict(ev)
        record["impact"] = claim_impact(record, summary_claims=summaries)
        targets.append(record)

    targets.sort(key=lambda r: (-int(r.get("impact", 0)), str(r.get("claim", ""))))
    if limit is not None:
        targets = targets[: max(0, int(limit))]
    return targets


# ---------------------------------------------------------------------------
# Primary-source completion: which research dimensions are thin, and what
# targeted follow-up query closes the gap (reuses the existing query builder).
# ---------------------------------------------------------------------------

# A dimension whose facts are at or below this primary share is "thin" and gets
# a targeted primary follow-up on the next expansion pass. Chosen at 0.34 so a
# dimension with zero, one-of-three or one-of-two primary documents qualifies,
# while a genuinely primary-led dimension (>= half) does not.
PRIMARY_THIN_THRESHOLD = 0.34


def dimension_primary_share(
    facts: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    """Primary-source share per research dimension (keyed by sub_question).

    Uses the fact-level `is_primary` flag stamped by the summarizer from the
    source's tier. Deterministic and total. Dimensions with no facts are absent
    (not reported as 0.0), so callers do not chase a dimension that was never
    researched.
    """
    totals: Dict[str, int] = {}
    primary: Dict[str, int] = {}
    for fact in facts or []:
        if not isinstance(fact, dict):
            continue
        claim = str(fact.get("claim", "") or "").strip()
        source = str(fact.get("source", "") or "").strip()
        if not claim or not source:
            continue
        dim = str(fact.get("sub_question", "") or "").strip() or "__unattributed__"
        totals[dim] = totals.get(dim, 0) + 1
        if bool(fact.get("is_primary", False)):
            primary[dim] = primary.get(dim, 0) + 1
    return {
        dim: round(primary.get(dim, 0) / count, 3)
        for dim, count in totals.items()
        if count
    }


def primary_source_followups(
    facts: Sequence[Dict[str, Any]],
    sub_questions: Sequence[Any],
    *,
    limit: int = 2,
    attempt: int = 0,
) -> List[str]:
    """Targeted primary-source queries for the weakest dimensions.

    For each researched dimension whose primary-source share is at/below
    `PRIMARY_THIN_THRESHOLD`, build a primary-source query (via
    `build_dimension_primary_query`, the same registry the planner uses — no
    parallel system) from that dimension's own question text. Ordered by
    ascending share (weakest first) then by dimension name, so a capped list is
    stable. Returns [] when every dimension is already primary-led or nothing
    was researched — the caller then simply issues no follow-up.
    """
    pool = [f for f in (facts or []) if isinstance(f, dict)]
    if not pool:
        return []
    try:
        from app.agents.sources import build_dimension_primary_query

        shares = dimension_primary_share(pool)
    except Exception as exc:  # never break routing on an import/regression
        logger.warning("primary_followup_failed", error=str(exc), exc_info=exc)
        return []

    by_dim: Dict[str, Dict[str, Any]] = {}
    for item in sub_questions or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("question", "") or "").strip()
        if not text:
            continue
        by_dim.setdefault(text, item)

    thin = [
        (share, dim)
        for dim, share in shares.items()
        if share <= PRIMARY_THIN_THRESHOLD and dim != "__unattributed__"
    ]
    thin.sort(key=lambda pair: (pair[0], pair[1]))

    out: List[str] = []
    for share, dim in thin:
        item = by_dim.get(dim)
        if not isinstance(item, dict):
            continue
        query = build_dimension_primary_query(
            str(item.get("question", "") or ""),
            str(item.get("search_type", "") or ""),
            str(item.get("domain", "") or ""),
            attempt=attempt,
        )
        if query and query not in out:
            out.append(query)
        if len(out) >= max(1, limit):
            break
    return out
