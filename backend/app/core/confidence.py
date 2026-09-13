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
    """Fraction of claims corroborated by a similar claim from a DIFFERENT domain."""
    if len(facts) < 2:
        return 0.0
    domains = [extract_domain(str(f.get("source", ""))) for f in facts]
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
    """Covered research axes / planned axes, attributed via source URL.

    Uses the same URL->axis attribution as the depth controller so the
    stopping rule and the confidence signal never disagree."""
    from app.core.depth_controller import _axes_covered, _planned_axes, _url_to_axis

    state = {"sub_questions": sub_questions or [], "search_results": [], "facts": facts}
    planned = _planned_axes(state)
    if not planned:
        return 0.0
    covered = _axes_covered(state, 1)  # any verified fact counts an axis covered
    return min(1.0, len(covered) / len(planned))


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

    overall = sum(weights.get(name, 0.0) * value for name, value in signals.items())
    overall = round(max(0.0, min(1.0, overall)), 3)

    notes = []
    if not measured:
        notes.append("freshness not yet measured (no publish dates captured) — weighted 0")

    penalty = _contradiction_penalty(contradictions, fact_count=len(facts or []))
    if penalty > 0:
        overall = round(max(0.0, overall - penalty), 3)
        notes.append(
            f"confidence penalized {penalty:.2f} for {len(contradictions or [])} "
            "cross-source contradiction(s)"
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
