#!/usr/bin/env python3
"""Deterministic OFFLINE golden evaluator for query routing.

The router decides whether a query needs external evidence (research) or can
be answered from stable general knowledge (direct). The safety-critical
property is that an evidence-requiring query NEVER routes direct — a missed
research run is a quality regression, an ungrounded answer is a correctness
bug.

This evaluator exercises the DETERMINISTIC gate (`deterministic_route`), not
the optional LLM clearance: the gate is always applied and is what the live
router falls back to. Any hard blocker (freshness / quantitative / decision /
contested / query-type / ambiguity) must force research.

No LLM, no network, no clock. Exit code 0 only when every labeled case routes
as expected AND the aggregate routing pass rate clears its threshold.

Usage (from backend/):

    python bench/eval_router.py

Results:
    bench/results/router_eval_v1.json   machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_THRESHOLDS = Path(__file__).resolve().parent / "golden" / "thresholds_router_v1.json"

# The aggregate metric the gate compares against the threshold file.
ROUTER_METRICS = ("router_routing_pass_rate",)


def evaluate_case(case: tuple) -> Dict[str, Any]:
    """Route ONE labeled case through the deterministic gate and score it.

    A case is (query, expected_path, expected_tokens) or
    (query, expected_path, expected_tokens, intent). `expected_path` is
    "research" for evidence-requiring queries, "clear" for stable general
    knowledge that has NO hard blocker (the model may then clear it).
    """
    from app.agents.router import deterministic_route

    query = case[0]
    expected_path = case[1]
    expected_tokens = list(case[2])
    intent = case[3] if len(case) > 3 else None

    decision = deterministic_route(query, intent=intent)
    blockers = list(decision.signals.get("hard_blockers") or [])
    blockers_lower = [b.lower() for b in blockers]

    if expected_path == "research":
        path_ok = decision.path == "research"
        tokens_ok = all(
            token.lower() in blockers_lower for token in expected_tokens
        )
    else:
        # "clear": no hard blocker, so the gate did not force research. The
        # gate still defaults to research pending model clearance — the case
        # asserts only that nothing BLOCKED a direct answer.
        path_ok = not blockers
        tokens_ok = True

    passed = path_ok and tokens_ok
    return {
        "query": query,
        "expected_path": expected_path,
        "actual_path": decision.path,
        "blockers": blockers,
        "expected_tokens": expected_tokens,
        "path_ok": path_ok,
        "tokens_ok": tokens_ok,
        "passed": passed,
    }


def evaluate() -> Dict[str, Any]:
    from bench.datasets import ROUTER_ROUTING_CASES

    rows = [evaluate_case(case) for case in ROUTER_ROUTING_CASES]
    passed = sum(1 for r in rows if r["passed"])
    rate = passed / max(1, len(rows))

    # A missed research route is the dangerous direction: count it explicitly
    # so a regression is legible, not just a lower pass rate.
    missed_research = [
        r["query"] for r in rows
        if r["expected_path"] == "research" and r["actual_path"] != "research"
    ]

    result = {
        "suite": "mars-router-eval",
        "version": "v1",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "cases": rows,
        "metrics": {
            "router_routing_pass_rate": round(rate, 4),
            "missed_research_count": len(missed_research),
        },
        "missed_research": missed_research,
    }
    result["thresholds"] = _load_thresholds()
    result["threshold_failures"] = check_thresholds(result, result["thresholds"])
    result["passed"] = not result["threshold_failures"]
    return result


def _load_thresholds() -> Dict[str, Any]:
    if DEFAULT_THRESHOLDS.exists():
        return json.loads(DEFAULT_THRESHOLDS.read_text())
    # The gate's own contract: every labeled case must route correctly.
    return {"router": {"router_routing_pass_rate": 1.0}}


def check_thresholds(report: Dict[str, Any], thresholds: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compare the router metrics against the threshold file's `router` block.

    A missing metric is a failure, not a pass — an unmeasured router is not a
    healthy one (same rule as bench/eval_offline.check_thresholds).
    """
    failures: List[Dict[str, Any]] = []
    wanted = (thresholds or {}).get("router") or {}
    for metric, floor in wanted.items():
        actual = report.get("metrics", {}).get(metric)
        if actual is None:
            failures.append({"metric": metric, "actual": None, "expected": floor})
        elif float(actual) < float(floor):
            failures.append({"metric": metric, "actual": actual, "expected": floor})
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="results directory (default bench/results)")
    args = parser.parse_args()

    results_dir = Path(args.out) if args.out else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    report = evaluate()
    out_path = results_dir / "router_eval_v1.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))

    m = report["metrics"]
    print("=" * 62)
    print("MARS query-router offline evaluation")
    print("=" * 62)
    print(f"routing pass rate: {m['router_routing_pass_rate']:.4f} "
          f"({sum(1 for c in report['cases'] if c['passed'])}/{len(report['cases'])})")
    if report["missed_research"]:
        print(f"MISSED RESEARCH: {report['missed_research']}")
    for f in report["threshold_failures"]:
        print(f"THRESHOLD FAILURE: {f}")
    print(f"results written: {out_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
