"""Confidence Engine v2 (Phase 2.4 + Feature 10 wiring).

Multi-signal confidence replacing the inline formula in critic_node.
Returns the overall score AND the per-signal breakdown so the frontend can
render the confidence breakdown UI.

v2 wires the three signals the v3 adapter recorded but never fed live:

* contradictions  — a PENALTY, not a weighted term: severe cross-source
                    conflicts subtract from the final score (a weighted
                    average would let strong sources mask direct lies)
* citation_support — the previous iteration's answer-support rate, blended
                    in once a synthesis exists (10% carved from citation
                    coverage — the two measure claim-level vs sentence-level
                    grounding of the same thing)
* axis_coverage   — planned-vs-covered research angles (5% carved from
                    source diversity; an unanswered axis means the number is
                    incomplete no matter how good the covered sources are)

Signals only contribute when their inputs exist, and absent inputs keep
the historical weights exactly (the WEIGHTS_FRESH pattern), so scores stay
comparable across runs and the existing test contracts hold.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List

from app.agents.evidence_utils import extract_domain, source_reliability_score
from app.core.logging import get_logger

logger = get_logger(__name__)

# Corroboration band: similar enough to be about the same thing, below the
# dedupe merge threshold (0.86) so identical claims were already collapsed.
CORROBORATION_SIMILARITY = 0.55

WEIGHTS: Dict[str, float] = {
    "source_quality": 0.20,
    "source_diversity": 0.15,
    "citation_coverage": 0.25,
    "claim_verification_strength": 0.15,
    "cross_source_agreement": 0.15,
    "critic_survival": 0.10,
    "freshness": 0.0,  # not yet measured — no publish dates captured
}

# Freshness takes 0.05 from citation coverage ONLY when publish dates are
# actually present; undated runs keep the exact legacy weights (WEIGHTS),
# so historical scores stay comparable.
WEIGHTS_FRESH: Dict[str, float] = {
    **WEIGHTS,
    "citation_coverage": 0.20,
    "freshness": 0.05,
}

# Contradiction penalty: severe cross-source conflicts subtract from the
# weighted sum (not a weighted term — a weighted average would let strong
# sources mask direct conflicts). Capped so a noisy pool can't zero out a
# otherwise-solid run.
CONTRADICTION_PENALTY_SEVERE = 0.06   # per severe conflict (severity >= 0.60)
CONTRADICTION_PENALTY_MODERATE = 0.03  # per moderate conflict
CONTRADICTION_PENALTY_CAP = 0.18
SEVERE_CONTRADICTION_SEVERITY = 0.60

# Recency curve: fresh under a month, decaying to zero at two years.
FRESHNESS_HALF_LIFE_DAYS = 730


def _freshness(source_dates: List[str] | None) -> tuple[float, bool]:
    """Mean recency over parseable dates. (0.0, False) when none parse —
    unknown stays unmeasured, never faked."""
    from datetime import date

    from app.agents.evidence_utils import parse_published_date

    ages: List[float] = []
    today = date.today()
    for raw in source_dates or []:
        iso = parse_published_date(raw)
        if not iso:
            continue
        try:
            age_days = (today - date.fromisoformat(iso)).days
        except (TypeError, ValueError):
            continue
        if age_days < 0:
            age_days = 0  # future-dated metadata is a provider quirk, not freshness
        ages.append(max(0.0, 1.0 - age_days / FRESHNESS_HALF_LIFE_DAYS))
    if not ages:
        return 0.0, False
    return round(sum(ages) / len(ages), 3), True


def _safe_conf(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _similarity(a: str, b: str) -> float:
    a_norm = " ".join(sorted(a.lower().split()))
    b_norm = " ".join(sorted(b.lower().split()))
    if not a_norm or not b_norm:
        return 0.0
    # Cheap prefilter before the O(n*m) char diff (this is the hot path of
    # _cross_source_agreement's O(n^2) pair loop). Pairs whose word sets
    # barely overlap and share no rare/numeric anchors cannot reach the
    # corroboration band (0.55) on the char ratio: the matched characters
    # are bounded by the shared tokens' length. Short claims skip the gate —
    # their diffs are cheap anyway.
    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())
    if len(a_tokens) >= 4 and len(b_tokens) >= 4:
        inter = a_tokens & b_tokens
        union = a_tokens | b_tokens
        rare_shared = sum(
            1 for t in inter if len(t) >= 7 or any(c.isdigit() for c in t)
        )
        if len(inter) / len(union) < 0.10 and rare_shared < 2:
            return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def _source_quality(facts: List[Dict[str, Any]]) -> float:
    scores = [source_reliability_score(str(f.get("source", ""))) for f in facts if f.get("source")]
    return sum(scores) / len(scores) if scores else 0.0


def _source_diversity(facts: List[Dict[str, Any]]) -> float:
    domains = {extract_domain(str(f.get("source", ""))) for f in facts if f.get("source")}
    domains.discard("")
    if len(facts) < 2 or len(domains) < 2:
        # A single fact or a single domain is by definition not diverse.
        return 0.0
    # Distinct domains relative to fact count, capped at 5 facts.
    return min(1.0, len(domains) / max(2.0, min(len(facts), 5)))


def _citation_coverage(facts: List[Dict[str, Any]]) -> float:
    """Fraction of facts that passed verification (Phase 2.3)."""
    if not facts:
        return 0.0
    if not any("verified" in f for f in facts):
        return 0.0  # verification never ran — don't award coverage credit
    verified = sum(1 for f in facts if f.get("verified"))
    return verified / len(facts)


def _claim_verification_strength(facts: List[Dict[str, Any]]) -> float:
    """Mean lexical-overlap score among facts that were checked (Phase 2.3)."""
    checked = [f for f in facts if isinstance(f.get("verification_score"), (int, float))]
    if not checked:
        return 0.0
    return sum(_safe_conf(f.get("verification_score", 0.0)) for f in checked) / len(checked)


def _cross_source_agreement(facts: List[Dict[str, Any]]) -> float:
    """Fraction of claims corroborated by a similar claim from a DIFFERENT publisher.

    Independence is measured at the registrable-domain level (blog.example.com
    and www.example.com are one publisher), matching the evidence spine — a
    raw host comparison would count a site's own pages as corroboration.
    """
    if len(facts) < 2:
        return 0.0
    from app.core.evidence_grade import registrable_domain

    domains = [
        registrable_domain(str(f.get("source", ""))) or extract_domain(str(f.get("source", "")))
        for f in facts
    ]
    claims = [str(f.get("claim", "")) for f in facts]
    corroborated = 0
    for i, claim in enumerate(claims):
        for j, other in enumerate(claims):
            if i == j or domains[i] == domains[j]:
                continue
            if _similarity(claim, other) >= CORROBORATION_SIMILARITY:
                corroborated += 1
                break
    return corroborated / len(claims)


def _critic_survival(critique: Dict[str, Any], iteration: int, max_iterations: int) -> float:
    """1.0 if the critic passed the run on its own; lower when forced through loops/ceilings."""
    if bool(critique.get("is_sufficient", False)):
        return 1.0
    if iteration >= max_iterations:
        return 0.4  # forced through the ceiling
    return 0.6  # stopped early for another reason (timeout, manual stop, etc.)


# When the summarizer ran on its deterministic fallback, the fact pool is
# extractive rather than model-written: every claim lexically overlaps its
# own source by construction, so source/verification signals read high no
# matter how good the evidence actually is. Capping below the sufficiency
# threshold (0.75) keeps a degraded run from finalizing as "High" confidence
# and keeps the honest signals (degraded list) attached to the breakdown.
DEGRADED_CAP = 0.55
EXTRACTIVE_FALLBACK_AGENTS = frozenset({"summarizer", "synthesizer"})


def _axis_coverage(sub_questions: List[Dict[str, Any]] | None, facts: List[Dict[str, Any]]) -> float:
    """Covered research axes / planned axes.

    Facts attribute to a contract's axis via the fact's own `sub_question`
    (stamped by the summarizer from the source record and preserved by dedup).
    Only verified facts count — unless verification never ran, in which case
    the pool is judged as-is. The previous implementation delegated to the
    depth controller's URL-based attribution against a synthetic state with an
    empty search_results list, so the signal was structurally always 0.0.
    """
    axis_by_question: Dict[str, str] = {}
    for q in sub_questions or []:
        if isinstance(q, dict):
            question = str(q.get("question", "")).strip()
            axis = str(q.get("axis", "")).strip()
            if question and axis:
                axis_by_question[question] = axis
    if not axis_by_question:
        return 0.0

    pool = [f for f in facts or [] if isinstance(f, dict)]
    if any("verified" in f for f in pool):
        pool = [f for f in pool if f.get("verified")]

    covered = {
        axis_by_question[str(f.get("sub_question", "")).strip()]
        for f in pool
        if str(f.get("sub_question", "")).strip() in axis_by_question
    }
    return min(1.0, len(covered) / len(set(axis_by_question.values())))


def _grade_records(facts: List[Dict[str, Any]], contradictions: List[Dict[str, Any]] | None) -> float | None:
    """Evidence-grade quality of the fact pool, or None when ungradeable.

    Grades come from measured per-claim signals (independent corroboration,
    verification, numeric support, contradiction) — never an LLM opinion.
    Returns None (so the confidence weights stay historical) when there is no
    usable pool or grading fails; grading must never break confidence.
    """
    pool = [f for f in (facts or []) if isinstance(f, dict) and str(f.get("claim", "")).strip()]
    if not pool:
        return None
    try:
        from app.core.evidence_grade import grade_facts

        graded = grade_facts(pool, contradictions=contradictions)
        records = [g["evidence"] for g in graded if isinstance(g.get("evidence"), dict)]
        if not records:
            return None
        # grade_facts already emitted serialized records; score them directly
        # (EvidenceRecord reconstruction from partial dicts is fragile).
        weights = {  # same scale as evidence_grade.evidence_quality_score
            "A": 1.0, "B": 0.75, "C": 0.4, "D": 0.1,
        }
        return round(sum(weights.get(str(r.get("grade", "D")), 0.1) for r in records) / len(records), 3)
    except Exception as exc:  # grading must never break confidence
        logger.warning("evidence grading failed, skipping signal: %s", exc)
        return None


def _contradiction_penalty(
    contradictions: List[Dict[str, Any]] | None,
    fact_count: int = 0,
) -> float:
    """Total subtraction for cross-source conflicts (0 .. CONTRADICTION_PENALTY_CAP).

    Pool-size scaled: one severe conflict among 30 facts is noise (or one
    stale page); the same conflict among 3 facts means a third of the
    evidence base disagrees with itself. Small pools pay up to 2x the per-
    conflict penalty, capped as before so a noisy pool still can't zero out
    an otherwise-solid run.
    """
    scale = min(2.0, max(1.0, 6.0 / max(1, fact_count)))
    penalty = 0.0
    for c in contradictions or []:
        if not isinstance(c, dict) or c.get("intra_source"):
            continue
        # Fix C: a contradiction explained by a different period/scope/metric
        # is resolved — recorded for the report, but it must not penalize an
        # otherwise-sound run. Only genuine same-unit/scope/period conflicts
        # subtract.
        if c.get("resolved"):
            continue
        severity = _safe_conf(c.get("severity", 0.5))
        if severity >= SEVERE_CONTRADICTION_SEVERITY:
            penalty += CONTRADICTION_PENALTY_SEVERE * scale
        else:
            penalty += CONTRADICTION_PENALTY_MODERATE * scale
    return min(CONTRADICTION_PENALTY_CAP, penalty)


def compute_confidence(
    facts: List[Dict[str, Any]],
    critique: Dict[str, Any],
    iteration: int,
    max_iterations: int,
    source_dates: List[str] | None = None,
    degraded: List[str] | None = None,
    contradictions: List[Dict[str, Any]] | None = None,
    answer_support: Dict[str, Any] | None = None,
    sub_questions: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Return {"overall": float, "signals": {...}} with per-signal values.

    `degraded` is the per-request fallback list (degradation.take_fallbacks):
    when an evidence-producing stage ran deterministically, the overall score
    is capped — a degraded run must never look more confident than a healthy
    one that was forced through the ceiling.

    `contradictions` (from the live engine), `answer_support` (previous
    iteration's verify_answer_support) and `sub_questions` (the plan) wire
    the three v3 signals; each contributes only when present so historical
    scores stay comparable.
    """
    freshness_value, measured = _freshness(source_dates)
    weights = dict(WEIGHTS_FRESH if measured else WEIGHTS)

    support_rate: float | None = None
    if isinstance(answer_support, dict):
        raw = answer_support.get("rate")
        if raw is not None:
            support_rate = max(0.0, min(1.0, _safe_conf(raw)))

    planned_axes_count = sum(
        1 for q in (sub_questions or []) if isinstance(q, dict) and str(q.get("axis", "")).strip()
    )

    signals: Dict[str, float] = {
        "source_quality": round(_source_quality(facts), 3),
        "source_diversity": round(_source_diversity(facts), 3),
        "citation_coverage": round(_citation_coverage(facts), 3),
        "claim_verification_strength": round(_claim_verification_strength(facts), 3),
        "cross_source_agreement": round(_cross_source_agreement(facts), 3),
        "critic_survival": round(_critic_survival(critique, iteration, max_iterations), 3),
        "freshness": freshness_value,
    }

    # --- v3 signal wiring (only when the inputs exist) --------------------
    if support_rate is not None:
        weights["citation_coverage"] = round(weights.get("citation_coverage", 0.25) - 0.10, 4)
        weights["citation_support"] = 0.10
        signals["citation_support"] = round(support_rate, 3)
    if planned_axes_count > 0:
        weights["source_diversity"] = round(weights.get("source_diversity", 0.15) - 0.05, 4)
        weights["axis_coverage"] = 0.05
        signals["axis_coverage"] = round(_axis_coverage(sub_questions, facts), 3)

    # --- evidence-grade signal (Step 2): how good is the evidence itself? ---
    # Grades are computed from measured, per-claim signals (independence,
    # verification, numeric support, contradiction) — never from an LLM's
    # opinion. Carved 0.05 from source_quality so the historical weights
    # stay intact when grades are absent (older callers/tests).
    grade_records = _grade_records(facts, contradictions)
    if grade_records is not None:
        weights["source_quality"] = round(weights.get("source_quality", 0.20) - 0.05, 4)
        weights["claim_evidence_quality"] = 0.05
        signals["claim_evidence_quality"] = round(grade_records, 3)

    overall = sum(weights.get(name, 0.0) * value for name, value in signals.items())
    overall = round(max(0.0, min(1.0, overall)), 3)

    notes = []
    if not measured:
        notes.append("freshness not yet measured (no publish dates captured) — weighted 0")

    penalty = _contradiction_penalty(contradictions, fact_count=len(facts or []))
    if penalty > 0:
        overall = round(max(0.0, overall - penalty), 3)
        unresolved = sum(
            1 for c in (contradictions or [])
            if isinstance(c, dict) and not c.get("resolved") and not c.get("intra_source")
        )
        notes.append(
            f"confidence penalized {penalty:.2f} for {unresolved} "
            "unresolved cross-source contradiction(s)"
        )

    degraded_agents = {str(a) for a in (degraded or []) if a}
    extractive = sorted(degraded_agents & EXTRACTIVE_FALLBACK_AGENTS)
    if extractive:
        notes.append(
            "confidence capped: " + ", ".join(extractive)
            + " ran on deterministic extraction — claims are unrewritten source text"
        )
        overall = round(min(overall, DEGRADED_CAP), 3)
    return {
        "overall": overall,
        "signals": signals,
        "weights": weights,
        "notes": notes,
    }
