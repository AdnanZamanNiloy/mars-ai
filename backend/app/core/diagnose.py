"""Self-Diagnosis aggregates (4.2): pure functions over memory-table rows.

Everything here takes plain row-dicts and returns plain dicts — no I/O,
so the analysis is unit-testable and the fetching script stays thin.
Only real persisted data is analyzed; where the schema cannot attribute
something (e.g. per-specialist claim authorship), the report says so
instead of inventing an attribution.
"""

from __future__ import annotations

from typing import Any, Dict, List


def verification_by_domain(claims: List[Dict[str, Any]], domain_of) -> List[Dict[str, Any]]:
    """Verified-rate and confidence per source domain, busiest first."""
    buckets: Dict[str, Dict[str, Any]] = {}
    for claim in claims:
        domain = domain_of(claim.get("source_url") or "") or "(unsourced)"
        bucket = buckets.setdefault(domain, {"claims": 0, "verified": 0, "confidence_sum": 0.0})
        bucket["claims"] += 1
        if claim.get("verified"):
            bucket["verified"] += 1
        conf = claim.get("confidence")
        if isinstance(conf, (int, float)):
            bucket["confidence_sum"] += float(conf)
    rows = [
        {
            "domain": domain,
            "claims": b["claims"],
            "verified": b["verified"],
            "verified_rate": b["verified"] / b["claims"] if b["claims"] else 0.0,
            "avg_confidence": b["confidence_sum"] / b["claims"] if b["claims"] else 0.0,
        }
        for domain, b in buckets.items()
    ]
    rows.sort(key=lambda r: r["claims"], reverse=True)
    return rows


def critic_efficiency(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    """How hard the critic works: passes per run, expansion rate, gains."""
    by_run: Dict[str, List[Dict[str, Any]]] = {}
    for review in reviews:
        by_run.setdefault(str(review.get("run_id", "")), []).append(review)
    if not by_run:
        return {"runs": 0, "avg_iterations": 0.0, "expansion_rate": 0.0, "avg_confidence_gain": 0.0}

    iterations = []
    expanded = 0
    gains = []
    for run_reviews in by_run.values():
        ordered = sorted(run_reviews, key=lambda r: r.get("iteration") or 0)
        iterations.append(len(ordered))
        if len(ordered) > 1:
            expanded += 1
        confs = [r.get("confidence") for r in ordered if isinstance(r.get("confidence"), (int, float))]
        if len(confs) >= 2:
            gains.append(confs[-1] - confs[0])
    n = len(by_run)
    return {
        "runs": n,
        "avg_iterations": sum(iterations) / n,
        "expansion_rate": expanded / n,
        "avg_confidence_gain": (sum(gains) / len(gains)) if gains else 0.0,
    }


def run_health(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Status mix, confidence and cost of completed runs, failure rate."""
    by_status: Dict[str, int] = {}
    for run in runs:
        status = str(run.get("status", "unknown"))
        by_status[status] = by_status.get(status, 0) + 1
    completed = [r for r in runs if r.get("status") == "completed"]

    def _mean(key: str) -> float | None:
        vals = [r.get(key) for r in completed if isinstance(r.get(key), (int, float))]
        return (sum(vals) / len(vals)) if vals else None

    total = len(runs)
    failed = by_status.get("failed", 0) + by_status.get("timeout", 0)
    return {
        "total": total,
        "by_status": by_status,
        "fail_rate": (failed / total) if total else 0.0,
        "avg_confidence": _mean("confidence"),
        "avg_cost": _mean("estimated_cost"),
    }


def contradiction_watch(report_markdowns: List[str]) -> Dict[str, Any]:
    """Share of final reports that surfaced a Contradictions section."""
    if not report_markdowns:
        return {"reports": 0, "with_contradictions": 0, "rate": 0.0}
    hits = sum(1 for md in report_markdowns if md and "# Contradictions" in md)
    return {"reports": len(report_markdowns), "with_contradictions": hits, "rate": hits / len(report_markdowns)}


def attention_flags(
    health: Dict[str, Any],
    domains: List[Dict[str, Any]],
    critic: Dict[str, Any],
) -> List[str]:
    """Human-readable warnings. Conservative thresholds — flag only
    patterns strong enough to act on, not every below-average number."""
    flags = []
    if health["total"] >= 5 and health["fail_rate"] > 0.2:
        flags.append(
            f"High interruption rate: {health['fail_rate']:.0%} of {health['total']} runs "
            "ended failed/timeout — check timeouts and provider errors before tuning quality."
        )
    for row in domains:
        if row["claims"] >= 10 and row["verified_rate"] < 0.3:
            flags.append(
                f"Low verification yield from {row['domain']}: "
                f"{row['verified_rate']:.0%} verified across {row['claims']} claims — "
                "consider down-weighting or blocklisting this provider."
            )
    if critic["runs"] >= 5 and critic["expansion_rate"] > 0.8 and critic["avg_confidence_gain"] < 0.05:
        flags.append(
            f"Critic expands {critic['expansion_rate']:.0%} of runs but gains only "
            f"{critic['avg_confidence_gain']:+.2f} confidence on average — expansion "
            "may be burning budget for little improvement."
        )
    if health["avg_cost"] is not None and health["avg_cost"] > 0.01:
        flags.append(
            f"Average run cost ${health['avg_cost']:.4f} is above the $0.01 comfort line — "
            "review per-run budget caps."
        )
    return flags
