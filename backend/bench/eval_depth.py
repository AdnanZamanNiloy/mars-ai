#!/usr/bin/env python3
"""Deterministic OFFLINE golden evaluator for adaptive-depth routing.

The main golden evaluator (bench/eval_offline.py) runs the production graph
end-to-end, but its offline mocks finalize at the iteration ceiling — so its
regression gate never exercises the adaptive-depth BRANCHES. This evaluator
covers exactly those branches with deterministic ResearchState fixtures
(bench/golden/depth_fixtures_v1.py) routed through the real
`app.core.depth_controller`.

Routing is evaluated at the same seam `workflow.route_after_critic` uses:

  1. `depth_controller.hard_wall_reached(state)` — absolute finish
  2. `workflow._evidence_gaps_remain(state)`   — evidence gate over a model's "enough"
  3. `depth_controller.decide_with_checks(state)` — expand vs finalize + reason

No LLM, no network, no clock. Exit code 0 only when every scenario's routing
decision, reason tokens and limitation recording match expectation, AND the
aggregate `depth_routing_pass_rate` clears its threshold.

Usage (from backend/):

    python bench/eval_depth.py
    python bench/eval_depth.py --thresholds bench/golden/thresholds_depth_v1.json

Results:
    bench/results/depth_eval_v1.json   machine-readable
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
DEFAULT_THRESHOLDS = Path(__file__).resolve().parent / "golden" / "thresholds_depth_v1.json"

# The aggregate metric the gate compares against the threshold file.
DEPTH_METRICS = ("depth_routing_pass_rate",)


def evaluate_scenario(scenario: Dict[str, Any], settings: Any) -> Dict[str, Any]:
    """Route ONE fixture state and score it against its declared expectation.

    Mirrors `route_after_critic`'s ordering so a hard wall can never be
    preempted by an evidence trigger (the exact bug the depth controller's
    ordering guards against)."""
    from app.core import depth_controller
    from app.graph.workflow import _evidence_gaps_remain

    state = scenario["state"]
    expect = scenario["expect"]

    hard_wall = depth_controller.hard_wall_reached(state, settings)
    evidence_gaps = _evidence_gaps_remain(state)
    decision, checks = depth_controller.decide_with_checks(state, settings)
    reason = str(checks.get("decision_reason", "") or "")
    explained = depth_controller.explain(state, settings)

    tokens_present = {
        token: token.lower() in reason.lower()
        for token in (expect.get("reason_tokens") or [])
    }
    limitations: List[str] = []
    if expect.get("expect_limitations"):
        stop = depth_controller.stop_reason(state, settings)
        if stop:
            limitations.append(stop)

    decision_ok = decision == expect["expected_decision"]
    reason_ok = all(tokens_present.values())
    limitations_ok = bool(limitations) if expect.get("expect_limitations") else True
    # Hard wall must win: when the ceiling is reached the decision is finalize
    # even though evidence gaps remain.
    if expect.get("expect_limitations"):
        limitations_ok = limitations_ok and hard_wall and evidence_gaps

    passed = decision_ok and reason_ok and limitations_ok

    return {
        "id": scenario["id"],
        "description": scenario["description"],
        "expected_decision": expect["expected_decision"],
        "actual_decision": decision,
        "decision_ok": decision_ok,
        "reason": reason,
        "expected_reason_tokens": list(expect.get("reason_tokens") or []),
        "reason_tokens_present": tokens_present,
        "reason_ok": reason_ok,
        "expect_limitations": bool(expect.get("expect_limitations")),
        "limitations": limitations,
        "limitations_ok": limitations_ok,
        "hard_wall_reached": hard_wall,
        "evidence_gaps_remain": evidence_gaps,
        "triggers": explained.get("triggers", []),
        "high_impact_uncorroborated": explained.get("high_impact_uncorroborated", 0),
        "severe_contradictions": explained.get("severe_contradictions", 0),
        "thin_dimensions": explained.get("thin_dimensions", []),
        "passed": passed,
    }


def aggregate(per_scenario: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(per_scenario)
    passed = sum(1 for r in per_scenario if r["passed"])
    return {
        "scenarios_total": n,
        "scenarios_passed": passed,
        "scenarios_failed": n - passed,
        "metrics": {
            "depth_routing_pass_rate": round(passed / n, 4) if n else 0.0,
        },
    }


def check_thresholds(aggregate_result: Dict[str, Any],
                     thresholds: Dict[str, Any]) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for name, floor in (thresholds.get("depth") or {}).items():
        actual = aggregate_result["metrics"].get(name)
        if actual is None:
            failures.append({"metric": name, "actual": None, "floor": floor,
                             "reason": "metric not produced"})
            continue
        if float(actual) < float(floor):
            failures.append({"metric": name, "actual": float(actual), "floor": float(floor)})
    return failures


def evaluate(thresholds_path: str | None = None) -> Dict[str, Any]:
    from app.core.config import Settings
    from bench.golden import depth_fixtures_v1

    settings = Settings(groq_api_key="golden-offline", _env_file=None)
    scenarios = depth_fixtures_v1.all_scenarios()
    per_scenario = [evaluate_scenario(s, settings) for s in scenarios]
    aggregate_result = aggregate(per_scenario)

    thresholds = json.loads(
        Path(thresholds_path or DEFAULT_THRESHOLDS).read_text(encoding="utf-8")
    )
    failures = check_thresholds(aggregate_result, thresholds)

    return {
        "suite": "mars-golden-depth-eval",
        "version": thresholds.get("version", "v1"),
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "thresholds_path": str(thresholds_path or DEFAULT_THRESHOLDS),
        "aggregate": aggregate_result,
        "scenarios": per_scenario,
        "thresholds": thresholds,
        "threshold_failures": failures,
        "passed": not failures,
    }


def write_json(report: Dict[str, Any], out_dir: Path = RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "depth_eval_v1.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", default=None, help="depth thresholds JSON path")
    parser.add_argument("--out", default=None, help="results directory")
    args = parser.parse_args()

    try:
        report = evaluate(args.thresholds)
    except Exception as exc:
        print(f"depth evaluation failed to run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else RESULTS_DIR
    json_path = write_json(report, out_dir)

    print("=" * 62)
    print("MARS offline golden adaptive-depth routing evaluation (v1)")
    print("=" * 62)
    for row in report["scenarios"]:
        status = "PASS" if row["passed"] else "FAIL"
        print(f"[{status:>4}] {row['id']:<42} decision={row['actual_decision']:<8} "
              f"reason={row['reason']!r}")
    agg = report["aggregate"]["metrics"]
    for name in DEPTH_METRICS:
        print(f"  {name:<28} {agg.get(name)}")
    print(f"\nthreshold failures: {len(report['threshold_failures'])}")
    for f in report["threshold_failures"]:
        print(f"  - {f['metric']}: {f['actual']} < {f['floor']}")
    print(f"results: {json_path}")
    print(f"RESULT: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
