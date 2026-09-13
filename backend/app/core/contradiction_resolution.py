"""Contradiction resolution pass (Fix C).

`find_contradictions` surfaces conflicts; it does NOT decide whether the two
claims actually disagree. A live deep run reported 5 contradictions, all of
kind "temporal" — i.e. the same measure for different periods — yet nothing
resolved them, so they kept penalizing confidence and driving expansion even
though the spread was fully explained by the period.

This module is the missing judgment. For each detected pair it asks: do the
two claims MEASURE THE SAME THING? Four axes must agree — unit, scope
(geography/population/segment), period/date and metric definition:

* A different period explains different values  -> reconciled (resolved).
* A different scope explains different values   -> reconciled (resolved).
* A different metric/unit                       -> not comparable (resolved).
* Same unit+scope+period+metric, different value -> GENUINE conflict, the
  only case left `resolved: false` and the only case that keeps penalizing
  confidence and driving expansion.

Pure, deterministic and LLM-free, matching the rest of the evidence spine; a
malformed input is left unresolved (conservative — never silently wave a real
conflict away).
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.agents.evidence_utils import numeric_conflict

# Reuse the contradiction engine's scope vocabulary and year parser so the two
# modules can never drift apart on what "different scope/period" means.
from app.core.contradictions import _scopes_conflict, _scopes_in, _years_in

# Re-exported for callers that only import this module.
__all__ = [
    "resolve_contradiction",
    "resolve_contradictions",
    "unresolved_contradictions",
]


def _unit_of(contradiction: Dict[str, Any]) -> str:
    values = contradiction.get("values")
    if isinstance(values, dict):
        return str(values.get("unit", "") or "").lower()
    return ""


def _values_differ(contradiction: Dict[str, Any]) -> bool:
    """True when the two recorded values are materially different."""
    values = contradiction.get("values")
    if not isinstance(values, dict):
        return True
    try:
        va = float(values.get("value_a"))
        vb = float(values.get("value_b"))
    except (TypeError, ValueError):
        return True
    scale = max(abs(va), abs(vb), 1e-9)
    return abs(va - vb) / scale >= 0.05


def resolve_contradiction(contradiction: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `contradiction` with `resolved`/`resolution` set.

    Total: a non-dict or malformed input is returned with resolved=False (a
    real conflict must never be dismissed by a parsing accident).
    """
    if not isinstance(contradiction, dict):
        return {"resolved": False, "resolution": "unparseable contradiction record"}

    out = dict(contradiction)
    kind = str(contradiction.get("kind", "") or "").lower()
    claim_a = str(contradiction.get("claim_a", "") or "")
    claim_b = str(contradiction.get("claim_b", "") or "")
    unit = _unit_of(contradiction)

    # --- period: different years explain different values ------------------
    years_a, years_b = _years_in(claim_a), _years_in(claim_b)
    if years_a and years_b and set(years_a) != set(years_b):
        out["resolved"] = True
        out["resolution"] = (
            "different periods: the sources report the same measure for "
            f"{sorted(set(years_a))[:3]} vs {sorted(set(years_b))[:3]}; the "
            "values differ because the periods differ, not because the "
            "sources disagree."
        )
        return out
    # A finding the engine already classified as temporal is by definition
    # period-separated even if the year tokens were noisy.
    if kind == "temporal":
        out["resolved"] = True
        out["resolution"] = (
            "different periods: the engine classified this as a period "
            "mismatch, so the value spread is explained by the reporting date."
        )
        return out

    # --- scope: different geography/population explains different values ---
    scopes_a, scopes_b = _scopes_in(claim_a), _scopes_in(claim_b)
    if kind == "scope" or _scopes_conflict(scopes_a, scopes_b):
        out["resolved"] = True
        out["resolution"] = (
            "different scopes: "
            f"{sorted(scopes_a) or ['unspecified']} vs "
            f"{sorted(scopes_b) or ['unspecified']}; the figures describe "
            "different populations, so they are not contradictory."
        )
        return out

    # --- unit/metric: not comparable -------------------------------------
    if kind not in ("numeric", "polarity") and kind != "":
        # Unknown kind (e.g. a future detector) — do not dismiss it.
        out["resolved"] = False
        out["resolution"] = f"unclassified conflict kind {kind!r} left unresolved"
        return out

    # --- genuine conflict: same unit+scope+period+metric, values differ ---
    if kind == "numeric":
        conflict = numeric_conflict(claim_a, claim_b, divergence=0.05)
        if conflict is None:
            out["resolved"] = True
            out["resolution"] = (
                "no comparable like-united quantities remain — the apparent "
                "conflict is a unit or metric mismatch, not a disagreement."
            )
            return out
        unit = str(conflict.get("unit", unit) or unit)
        out["resolved"] = not _values_differ(contradiction)
        if out["resolved"]:
            out["resolution"] = (
                "values agree within tolerance once normalized — not a "
                "material conflict."
            )
        else:
            out["resolution"] = (
                f"unresolved: same {unit or 'dimensionless'} measure, same "
                "scope and period, materially different values — the sources "
                "genuinely disagree and this needs resolution before the "
                "figure can be treated as established."
            )
        return out

    if kind == "polarity":
        out["resolved"] = False
        out["resolution"] = (
            "unresolved: opposite assertions about the same subject, same "
            "scope and period — direct disagreement, not a scope or period "
            "artifact."
        )
        return out

    # No kind at all: conservative default is unresolved.
    out["resolved"] = not _values_differ(contradiction)
    out["resolution"] = (
        "unresolved: values differ without a period, scope or unit "
        "explanation."
        if not out["resolved"]
        else "values agree within tolerance — not a material conflict."
    )
    return out


def resolve_contradictions(
    contradictions: List[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    """Resolve every contradiction, preserving order. Total and non-mutating."""
    return [resolve_contradiction(c) for c in (contradictions or [])]


def unresolved_contradictions(
    contradictions: List[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    """Only the genuinely conflicting (resolved is not True) entries.

    This is what confidence and the stopping policy must count: a period or
    scope difference is recorded for the report but must not penalize a run.
    """
    return [
        c for c in (contradictions or [])
        if isinstance(c, dict) and not c.get("resolved")
    ]
