"""Confidence Engine (Phase 2.4, Feature 10).

Multi-signal confidence replacing the inline formula in critic_node.
Returns the overall score AND the per-signal breakdown so the frontend can
render the confidence breakdown UI later.

Signal weights sum to 1.0; freshness is explicitly weighted 0 because
publish dates are not captured yet (manual 2.4: don't fake a number).
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
    return 0.6  # stopped early for another reason (budget cutoff etc.)


def compute_confidence(
    facts: List[Dict[str, Any]],
    critique: Dict[str, Any],
    iteration: int,
    max_iterations: int,
    source_dates: List[str] | None = None,
) -> Dict[str, Any]:
    """Return {"overall": float, "signals": {...}} with per-signal values."""
    freshness_value, measured = _freshness(source_dates)
    weights = WEIGHTS_FRESH if measured else WEIGHTS
    signals: Dict[str, float] = {
        "source_quality": round(_source_quality(facts), 3),
        "source_diversity": round(_source_diversity(facts), 3),
        "citation_coverage": round(_citation_coverage(facts), 3),
        "claim_verification_strength": round(_claim_verification_strength(facts), 3),
        "cross_source_agreement": round(_cross_source_agreement(facts), 3),
        "critic_survival": round(_critic_survival(critique, iteration, max_iterations), 3),
        "freshness": freshness_value,
    }
    overall = sum(weights[name] * value for name, value in signals.items())
    notes = []
    if not measured:
        notes.append("freshness not yet measured (no publish dates captured) — weighted 0")
    return {
        "overall": round(max(0.0, min(1.0, overall)), 3),
        "signals": signals,
        "weights": weights,
        "notes": notes,
    }
