#!/usr/bin/env python3
"""Deterministic OFFLINE evaluator for contradiction DETECTION + RESOLUTION.

The full-graph golden evaluator (`bench/eval_offline.py`) exercises the
contradiction engine only incidentally; this evaluator scores it directly
against the LABELED fixture set (`bench/golden/contradictions_v1.json`) so a
precision or resolution regression trips the gate.

For each labeled case it runs the production pipeline on the case's claim pool:

    detect   app.core.contradictions.find_contradictions
    resolve  app.core.contradiction_resolution.resolve_contradictions

and maps the outcome to ONE observed label:

    not_contradiction    no contradiction detected for the scored pair
    contradiction        detected, resolved is False (genuine conflict)
    resolved_explained   detected, resolved is True (period/scope/metric
                         explains the values)

Two metrics are reported:

  * detection precision/recall/F1 — positive class is "any contradiction
    detected" (contradiction + resolved_explained). This catches false
    contradictions (unrelated claims sharing numbers) and missed conflicts.
  * resolution precision/recall/F1 — positive class is "resolved is True"
    among DETECTED spreads. This catches false "resolved" (a genuine conflict
    waved away) and false "unresolved" (an explained spread left penalizing).

Only the first two claims of a case form the scored pair (matching the
production engine, which scans every pair); extra claims are context.

No LLM, no network, no clock. Exit 0 only when every case's label matches AND
the aggregate metrics clear their thresholds; the thresholds live in
`bench/golden/thresholds_contradictions_v1.json` and are checked by
`bench/eval_offline.py --with-contradictions` (and by this module's own main()).

Usage (from backend/):

    python bench/eval_contradictions.py
    python bench/eval_contradictions.py --thresholds bench/golden/thresholds_contradictions_v1.json

Results:
    bench/results/contradictions_eval_v1.json   machine-readable
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
DEFAULT_THRESHOLDS = (
    Path(__file__).resolve().parent / "golden" / "thresholds_contradictions_v1.json"
)

# Metrics the gate compares against thresholds.
CONTRADICTION_METRICS = (
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "resolution_precision",
    "resolution_recall",
    "resolution_f1",
    "classification_accuracy",
)

_POSITIVE_DETECTION = {"contradiction", "resolved_explained"}


def classify_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Run detection + resolution on ONE case and derive the observed label."""
    from app.core.contradiction_resolution import resolve_contradictions
    from app.core.contradictions import find_contradictions

    claims = [
        {
            "claim": str(c.get("claim", "")),
            "source": str(c.get("source", "")),
            "verified": bool(c.get("verified", True)),
        }
        for c in case["claims"]
        if isinstance(c, dict)
    ]
    scored_pair = claims[:2]
    detected = find_contradictions(scored_pair)
    resolved = resolve_contradictions(detected)

    if not resolved:
        observed = "not_contradiction"
    elif any(not c.get("resolved") for c in resolved):
        observed = "contradiction"
    else:
        observed = "resolved_explained"

    return {
        "id": case["id"],
        "category": case.get("category", ""),
        "expected_label": case["expected_label"],
        "observed_label": observed,
        "correct": observed == case["expected_label"],
        "detected_count": len(detected),
        "resolved_count": sum(1 for c in resolved if c.get("resolved")),
        "unresolved_count": sum(1 for c in resolved if not c.get("resolved")),
        "kinds": [c.get("kind") for c in resolved],
        "explanation_present": all(
            bool(str(c.get("resolution", "")).strip())
            for c in resolved if c.get("resolved")
        ) if any(c.get("resolved") for c in resolved) else None,
        "claims": [c["claim"] for c in claims],
    }


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def aggregate(per_case: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Precision/recall for detection + resolution, plus label accuracy."""
    # Detection: positive class = any contradiction detected.
    det_tp = det_fp = det_fn = 0
    # Resolution: positive class = resolved is True, only among detected cases.
    res_tp = res_fp = res_fn = 0
    correct = 0

    for row in per_case:
        expected_pos = row["expected_label"] in _POSITIVE_DETECTION
        observed_pos = row["observed_label"] in _POSITIVE_DETECTION
        if expected_pos and observed_pos:
            det_tp += 1
        elif not expected_pos and observed_pos:
            det_fp += 1
        elif expected_pos and not observed_pos:
            det_fn += 1

        if row["correct"]:
            correct += 1

        # Resolution only scored where a spread was DETECTED by the engine (the
        # real decision surface); a not_contradiction case has nothing to
        # resolve, so it is excluded from resolution P/R.
        if row["observed_label"] != "not_contradiction":
            expected_resolved = row["expected_label"] == "resolved_explained"
            observed_resolved = row["observed_label"] == "resolved_explained"
            if expected_resolved and observed_resolved:
                res_tp += 1
            elif not expected_resolved and observed_resolved:
                res_fp += 1
            elif expected_resolved and not observed_resolved:
                res_fn += 1

    detection = _prf(det_tp, det_fp, det_fn)
    resolution = _prf(res_tp, res_fp, res_fn)
    n = len(per_case)
    return {
        "cases_total": n,
        "cases_correct": correct,
        "cases_incorrect": n - correct,
        "detection": {**detection, "tp": det_tp, "fp": det_fp, "fn": det_fn},
        "resolution": {**resolution, "tp": res_tp, "fp": res_fp, "fn": res_fn},
        "metrics": {
            "detection_precision": detection["precision"],
            "detection_recall": detection["recall"],
            "detection_f1": detection["f1"],
            "resolution_precision": resolution["precision"],
            "resolution_recall": resolution["recall"],
            "resolution_f1": resolution["f1"],
            "classification_accuracy": round(correct / n, 4) if n else 0.0,
        },
    }


def check_thresholds(aggregate_result: Dict[str, Any],
                     thresholds: Dict[str, Any]) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for name, floor in (thresholds.get("contradictions") or {}).items():
        actual = aggregate_result["metrics"].get(name)
        if actual is None:
            failures.append({"metric": name, "actual": None, "floor": floor,
                             "reason": "metric not produced"})
            continue
        if float(actual) < float(floor):
            failures.append({"metric": name, "actual": float(actual), "floor": float(floor)})
    return failures


def evaluate(thresholds_path: str | None = None) -> Dict[str, Any]:
    from bench.golden import contradictions_v1

    cases = contradictions_v1.load_cases()
    per_case = [classify_case(c) for c in cases]
    aggregate_result = aggregate(per_case)

    thresholds = json.loads(
        Path(thresholds_path or DEFAULT_THRESHOLDS).read_text(encoding="utf-8")
    )
    failures = check_thresholds(aggregate_result, thresholds)

    return {
        "suite": "mars-golden-contradiction-eval",
        "version": thresholds.get("version", "v1"),
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "thresholds_path": str(thresholds_path or DEFAULT_THRESHOLDS),
        "aggregate": aggregate_result,
        "cases": per_case,
        "thresholds": thresholds,
        "threshold_failures": failures,
        "passed": not failures,
    }


def write_json(report: Dict[str, Any], out_dir: Path = RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "contradictions_eval_v1.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", default=None,
                        help="contradiction thresholds JSON path")
    parser.add_argument("--out", default=None, help="results directory")
    args = parser.parse_args()

    try:
        report = evaluate(args.thresholds)
    except Exception as exc:
        print(f"contradiction evaluation failed to run: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else RESULTS_DIR
    json_path = write_json(report, out_dir)

    print("=" * 62)
    print("MARS offline golden contradiction-resolution evaluation (v1)")
    print("=" * 62)
    for row in report["cases"]:
        status = "PASS" if row["correct"] else "FAIL"
        print(f"[{status:>4}] {row['id']:<44} "
              f"expected={row['expected_label']:<18} observed={row['observed_label']}")
    agg = report["aggregate"]["metrics"]
    for name in CONTRADICTION_METRICS:
        print(f"  {name:<28} {agg.get(name)}")
    print(f"\nthreshold failures: {len(report['threshold_failures'])}")
    for f in report["threshold_failures"]:
        print(f"  - {f['metric']}: {f['actual']} < {f['floor']}")
    print(f"results: {json_path}")
    print(f"RESULT: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
