#!/usr/bin/env python3
"""Deterministic OFFLINE evaluator for provider-failure classification.

Reliability requirement #4: a provider failure must NOT be silently read as
weak evidence, and a transient 429/timeout must not be treated as an outage
that opens the 60s breaker. The main golden evaluator never exercises provider
failure; this evaluator asserts the CLASSIFICATION + REASON MAPPING for the
six contract cases with no network and no real provider:

  1. healthy provider                        -> no degradation
  2. transient 429 then 200                  -> retry succeeds, no degradation
  3. repeated 429 -> fallback provider       -> completes, provider-transient
  4. provider exhaustion                     -> explicit degraded, NOT weak evidence
  5. genuine weak evidence (provider OK)     -> weak-evidence, no provider degradation
  6. provider degradation + weak evidence    -> BOTH represented separately

The provider calls are stubbed at the LLMClient boundary so the evaluator
measures the SAME classification/record code paths the client runs,
deterministically and offline. Exit 0 only when every case matches AND the
aggregate `provider_classification_pass_rate` clears its threshold.

Usage (from backend/):

    python bench/eval_provider.py
    python bench/eval_provider.py --thresholds bench/golden/thresholds_provider_v1.json

Results:
    bench/results/provider_eval_v1.json   machine-readable
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_THRESHOLDS = (
    Path(__file__).resolve().parent / "golden" / "thresholds_provider_v1.json"
)

PROVIDER_METRICS = ("provider_classification_pass_rate",)

# A throwaway DB path keeps the client's provider-store lookup away from the
# developer's real research.db (an active provider there would hijack the
# chain and make this evaluator non-deterministic).
_TMP_DB = str(Path(tempfile.gettempdir()) / "mars-provider-eval.db")


def _settings():
    from app.core.config import Settings

    return Settings(
        groq_api_key="golden-offline",
        huggingface_api_key="golden-hf-offline",
        database_url=_TMP_DB,
        _env_file=None,
    )


def _rate_exc():
    import httpx

    req = httpx.Request("POST", "https://groq.example/v1/chat/completions")
    return httpx.HTTPStatusError("429", request=req, response=httpx.Response(429, json={}))


def _ok_result(name: str, endpoint: str):
    from app.core.llm import CompletionResult

    return CompletionResult(
        text='{"ok": true}', provider=name, model="m",
        endpoint=endpoint, input_tokens=1, output_tokens=1,
    )


def _drive_chain(client, groq_raises=None, hf_raises=None) -> Dict[str, Any]:
    """Run `_generate_with_fallback` with stubbed provider legs and return the
    recorded degradation summary plus what the call raised/returned."""

    async def _groq(s, u):
        if groq_raises is not None:
            raise groq_raises
        return _ok_result("groq", "https://api.groq.com/openai/v1/chat/completions")

    async def _hf(s, u):
        if hf_raises is not None:
            raise hf_raises
        return _ok_result("huggingface", "https://api-inference.huggingface.co/models/m")

    client._call_groq = _groq
    client._call_huggingface = _hf
    return client  # caller runs the chain inside its own event loop


async def _run_chain(client, groq_raises=None, hf_raises=None) -> Dict[str, Any]:
    """Async: drive the stubbed chain and return what happened + the summary."""
    from app.core.degradation import clear_fallbacks, degradation_summary, reset_fallbacks

    _drive_chain(client, groq_raises, hf_raises)
    reset_fallbacks()
    raised = None
    text = None
    try:
        text = await client._generate_with_fallback("sp", "up")
    except BaseException as exc:  # noqa: BLE001 - the case asserts the class
        raised = type(exc).__name__
    summary = degradation_summary()
    clear_fallbacks()
    return {"raised": raised, "text": text, "summary": summary}


async def case_1() -> Dict[str, Any]:
    """Healthy provider: no degradation of any kind."""
    from app.core.llm import LLMClient

    out = await _run_chain(LLMClient(_settings()))
    return {
        "raised": out["raised"],
        "text": out["text"],
        "degraded": out["summary"]["agents"],
        "provider_degraded": out["summary"]["provider_degraded"],
        "provider_kinds": out["summary"]["provider_kinds"],
    }


async def case_2() -> Dict[str, Any]:
    """Transient 429 then 200: the retry recovers IN PLACE.

    Exercises the real tenacity retry + breaker path: two 429s followed by a
    200 inside one call. Nothing must be recorded degraded and the breaker
    must stay closed (a throttle is not an outage)."""
    from app.core.config import Settings
    from app.core.degradation import clear_fallbacks, degradation_summary, reset_fallbacks
    from app.core.llm import LLMClient

    client = LLMClient(Settings(
        custom_llm_api_key="k", custom_llm_base_url="https://llm.example.com/v1",
        custom_llm_model="m", groq_api_key="", huggingface_api_key="",
        database_url=_TMP_DB, _env_file=None,
    ))
    calls = {"n": 0}

    async def _flaky(s, u, c, timeout_base):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _rate_exc()
        return _ok_result("custom", c["endpoint"])

    client._post_custom = _flaky
    reset_fallbacks()
    try:
        text = await client._generate_with_fallback("sp", "up")
        summary = degradation_summary()
        breaker_open = client.custom_breaker.is_open()
    finally:
        clear_fallbacks()
    return {
        "raised": None,
        "text": text,
        "recorded_provider_degradation": summary["provider_degraded"],
        "breaker_open": breaker_open,
        "attempts": calls["n"],
    }


async def case_3() -> Dict[str, Any]:
    """Repeated 429 on Groq: the HF fallback serves the call, run completes,
    and the cause is recorded as provider-transient (not weak evidence)."""
    from app.core.llm import LLMClient

    client = LLMClient(_settings())
    out = await _run_chain(client, groq_raises=_rate_exc())
    return {
        "raised": out["raised"],
        "text": out["text"],
        "provider_degraded": out["summary"]["provider_degraded"],
        "provider_kinds": out["summary"]["provider_kinds"],
        "evidence_agents": out["summary"]["evidence_agents"],
        "breaker_open": client.groq_breaker.is_open(),
    }


async def case_4() -> Dict[str, Any]:
    """Every provider 429s: the chain fails explicitly, classified
    provider-transient — and NOTHING is mislabeled weak evidence."""
    from app.core.config import Settings
    from app.core.llm import LLMClient

    # Only one provider configured: its repeated 429 exhausts the chain, which
    # must surface AllProvidersFailedError (not a bare transport error).
    client = LLMClient(Settings(
        groq_api_key="golden-offline", huggingface_api_key="",
        database_url=_TMP_DB, _env_file=None,
    ))
    out = await _run_chain(client, groq_raises=_rate_exc())
    return {
        "raised": out["raised"],
        "provider_degraded": out["summary"]["provider_degraded"],
        "provider_kinds": out["summary"]["provider_kinds"],
        "agents": out["summary"]["agents"],
        "evidence_agents": out["summary"]["evidence_agents"],
    }


async def case_5() -> Dict[str, Any]:
    """Genuine weak evidence: the provider answered; the stage degraded for a
    non-provider reason. Must be weak-evidence, NOT provider degradation."""
    from app.core.degradation import (
        EVIDENCE_WEAK,
        clear_fallbacks,
        degradation_summary,
        record_fallback,
        reset_fallbacks,
    )

    reset_fallbacks()
    try:
        record_fallback("summarizer", reason=EVIDENCE_WEAK)
        summary = degradation_summary()
    finally:
        clear_fallbacks()
    return {
        "provider_degraded": summary["provider_degraded"],
        "provider_kinds": summary["provider_kinds"],
        "evidence_agents": summary["evidence_agents"],
        "reasons": summary["reasons"],
    }


async def case_6() -> Dict[str, Any]:
    """Provider degradation AND weak evidence in one run: both represented,
    separately attributable."""
    from app.core.degradation import (
        EVIDENCE_WEAK,
        PROVIDER_TRANSIENT,
        clear_fallbacks,
        degradation_summary,
        record_fallback,
        reset_fallbacks,
    )
    from app.core.llm import LLMClient

    # Real transport failure: both providers 429, so the client records a
    # provider-transient failure. `_drive_chain` clears its own context, so we
    # replay the recorded kinds into a fresh context alongside the evidence
    # weakness — mirroring what the pipeline accumulates across stages.
    client = LLMClient(_settings())
    out = await _run_chain(client, groq_raises=_rate_exc(), hf_raises=_rate_exc())
    kinds = out["summary"]["provider_kinds"]
    assert "provider-transient" in kinds, "provider failure must classify transient"

    reset_fallbacks()
    try:
        from app.core.degradation import record_provider_failure

        record_provider_failure(PROVIDER_TRANSIENT, "replayed for case 6")
        record_fallback("summarizer", reason=EVIDENCE_WEAK)
        record_fallback("synthesizer", reason=PROVIDER_TRANSIENT)
        summary = degradation_summary()
    finally:
        clear_fallbacks()
    return {
        "provider_degraded": summary["provider_degraded"],
        "provider_kinds": summary["provider_kinds"],
        "evidence_agents": summary["evidence_agents"],
        "provider_agents": summary["provider_agents"],
        "reasons": summary["reasons"],
        "exhaustion_raised": out["raised"],
    }


CASES: List[Dict[str, Any]] = [
    {
        "id": "case1_healthy_provider",
        "description": "healthy provider: no degradation recorded",
        "run": case_1,
        "check": lambda r: (
            r["raised"] is None
            and r["degraded"] == []
            and r["provider_degraded"] is False
            and r["provider_kinds"] == []
        ),
    },
    {
        "id": "case2_transient_429_recovered",
        "description": "429 then 200: retry succeeds, no degradation, breaker closed",
        "run": case_2,
        "check": lambda r: (
            r["raised"] is None
            and r["recorded_provider_degradation"] is False
            and r["breaker_open"] is False
        ),
    },
    {
        "id": "case3_repeated_429_fallback",
        "description": "repeated 429 -> fallback provider, run completes, provider-transient",
        "run": case_3,
        "check": lambda r: (
            r["raised"] is None
            and r["provider_degraded"] is True
            and "provider-transient" in r["provider_kinds"]
            and r["evidence_agents"] == []
            and r["breaker_open"] is False
        ),
    },
    {
        "id": "case4_provider_exhaustion",
        "description": "provider exhaustion: explicit degraded, never weak evidence",
        "run": case_4,
        "check": lambda r: (
            r["raised"] == "AllProvidersFailedError"
            and r["provider_degraded"] is True
            and "provider-transient" in r["provider_kinds"]
            and r["agents"] == []
            and r["evidence_agents"] == []
        ),
    },
    {
        "id": "case5_weak_evidence",
        "description": "provider OK, evidence thin: weak-evidence, not provider degradation",
        "run": case_5,
        "check": lambda r: (
            r["provider_degraded"] is False
            and r["provider_kinds"] == []
            and r["evidence_agents"] == ["summarizer"]
            and r["reasons"].get("summarizer") == "weak-evidence"
        ),
    },
    {
        "id": "case6_both_classes_separate",
        "description": "provider degradation + weak evidence: both represented separately",
        "run": case_6,
        "check": lambda r: (
            r["provider_degraded"] is True
            and "provider-transient" in r["provider_kinds"]
            and r["evidence_agents"] == ["summarizer"]
            and r["provider_agents"] == ["synthesizer"]
        ),
    },
]


async def evaluate_case(case: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = await case["run"]()
        passed = bool(case["check"](result))
        error = None
    except Exception as exc:  # noqa: BLE001 - one case must not kill the eval
        result = {}
        passed = False
        error = f"{type(exc).__name__}: {exc}"
    return {
        "id": case["id"],
        "description": case["description"],
        "passed": passed,
        "error": error,
        "result": result,
    }


def aggregate(per_case: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(per_case)
    passed = sum(1 for c in per_case if c["passed"])
    return {
        "cases_total": n,
        "cases_passed": passed,
        "cases_failed": n - passed,
        "metrics": {
            "provider_classification_pass_rate": round(passed / n, 4) if n else 0.0,
        },
    }


def check_thresholds(aggregate_result: Dict[str, Any],
                     thresholds: Dict[str, Any]) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for name, floor in (thresholds.get("provider") or {}).items():
        actual = aggregate_result["metrics"].get(name)
        if actual is None:
            failures.append({"metric": name, "actual": None, "floor": floor,
                             "reason": "metric not produced"})
            continue
        if float(actual) < float(floor):
            failures.append({"metric": name, "actual": float(actual), "floor": float(floor)})
    return failures


async def evaluate(thresholds_path: str | None = None) -> Dict[str, Any]:
    per_case = [await evaluate_case(c) for c in CASES]
    aggregate_result = aggregate(per_case)
    thresholds = json.loads(
        Path(thresholds_path or DEFAULT_THRESHOLDS).read_text(encoding="utf-8")
    )
    failures = check_thresholds(aggregate_result, thresholds)
    return {
        "suite": "mars-golden-provider-eval",
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
    path = out_dir / "provider_eval_v1.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", default=None, help="provider thresholds JSON path")
    parser.add_argument("--out", default=None, help="results directory")
    args = parser.parse_args()
    try:
        report = asyncio.run(evaluate(args.thresholds))
    except Exception as exc:  # noqa: BLE001
        print(f"provider evaluation failed to run: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2
    out_dir = Path(args.out) if args.out else RESULTS_DIR
    json_path = write_json(report, out_dir)
    print("=" * 62)
    print("MARS offline golden provider-classification evaluation (v1)")
    print("=" * 62)
    for row in report["cases"]:
        status = "PASS" if row["passed"] else "FAIL"
        print(f"[{status:>4}] {row['id']:<34} {row['description']}")
    agg = report["aggregate"]["metrics"]
    for name in PROVIDER_METRICS:
        print(f"  {name:<36} {agg.get(name)}")
    print(f"\nthreshold failures: {len(report['threshold_failures'])}")
    for f in report["threshold_failures"]:
        print(f"  - {f['metric']}: {f['actual']} < {f['floor']}")
    print(f"results: {json_path}")
    print(f"RESULT: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
