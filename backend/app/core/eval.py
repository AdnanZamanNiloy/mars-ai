"""Evaluation Lab scoring (4.1): pure functions over finished-run metrics.

A "metric set" is what scripts/run_eval.py collects per query — either from
the live NDJSON stream or from a persisted trace. Scoring never touches
I/O so it stays unit-testable without API keys.
"""

from __future__ import annotations

from typing import Any, Dict, List


def score_query(expectation: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Check one eval query's expectations against its observed metrics.

    Returns per-check booleans plus an overall pass. Missing metrics count
    as failures — an eval that can't observe something shouldn't pass it.
    """
    claims = metrics.get("claims")
    verified = metrics.get("verified")
    decisions = metrics.get("decisions")

    claim_check = (
        isinstance(claims, int)
        and claims >= int(expectation.get("min_claims", 0))
    )
    verified_check = (
        isinstance(verified, int)
        and verified >= int(expectation.get("min_verified", 0))
    )
    if expectation.get("expect_decision_options"):
        decision_check = isinstance(decisions, int) and decisions >= 2
    else:
        decision_check = True
    completed_check = metrics.get("status") == "completed"

    passed = claim_check and verified_check and decision_check and completed_check
    return {
        "claims_ok": claim_check,
        "verified_ok": verified_check,
        "decisions_ok": decision_check,
        "completed_ok": completed_check,
        "passed": passed,
    }


def summarize_batch(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate one eval batch into trend-trackable headline numbers."""
    if not rows:
        return {
            "queries": 0,
            "pass_rate": 0.0,
            "avg_confidence": 0.0,
            "avg_claims": 0.0,
            "avg_verified": 0.0,
            "contradiction_rate": 0.0,
            "avg_cost": 0.0,
        }
    n = len(rows)

    def _mean(key: str) -> float:
        vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "queries": n,
        "pass_rate": sum(1 for r in rows if r.get("passed")) / n,
        "avg_confidence": _mean("confidence"),
        "avg_claims": _mean("claims"),
        "avg_verified": _mean("verified"),
        "contradiction_rate": sum(1 for r in rows if (r.get("contradictions") or 0) > 0) / n,
        "avg_cost": _mean("cost"),
    }


def format_trend(current: Dict[str, Any], previous: Dict[str, Any] | None) -> str:
    """One-line-per-metric comparison of this batch vs the previous one."""
    lines = [
        f"queries: {current['queries']}  pass rate: {current['pass_rate']:.0%}"
        + (f" (was {previous['pass_rate']:.0%})" if previous else " (no prior batch)"),
        _line("avg confidence", current["avg_confidence"], previous["avg_confidence"] if previous else None, ".2f"),
        _line("avg claims", current["avg_claims"], previous["avg_claims"] if previous else None, ".1f"),
        _line("avg verified", current["avg_verified"], previous["avg_verified"] if previous else None, ".1f"),
        _line("contradiction rate", current["contradiction_rate"], previous["contradiction_rate"] if previous else None, "%"),
        _line("avg cost $", current["avg_cost"], previous["avg_cost"] if previous else None, ".4f"),
    ]
    return "\n".join(lines)


def _line(label: str, cur: float, prev: float | None, fmt: str) -> str:
    if fmt == "%":
        cur_s = f"{cur:.0%}"
        prev_s = f"{prev:.0%}" if prev is not None else None
    else:
        cur_s = format(cur, fmt)
        prev_s = format(prev, fmt) if prev is not None else None
    arrow = ""
    if prev is not None and abs(cur - prev) > 1e-9:
        arrow = " ▲" if cur > prev else " ▼"
    return f"{label}: {cur_s}" + (f" (was {prev_s}){arrow}" if prev_s else "") + ""
