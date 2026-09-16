#!/usr/bin/env python3
"""Deterministic OFFLINE evaluator for retrieval access health.

Reliability requirement: the search layer was the dominant evidence
bottleneck (1,027x 403, 893x 429, 41 fetch timeouts across a 26-query live
run), and provider/host failures must be separable from genuinely weak
evidence. This evaluator drives the REAL `SearchClient` fetch/cooldown/
fallback code paths against scripted mock HTTP outcomes (respx) and reports
retrieval-health metrics with NEW floors:

  * successful_fetch_rate       (succeeded / attempted)
  * forbidden_rate / rate_limited_rate / timeout_rate
  * unique_authoritative_domains_reached
  * primary_source_acquisition_rate
  * cooldown_skip_rate          (skipped because a host was cooling)
  * duplicate_suppression_rate  (known-dead URLs not re-fetched)
  * fallback_acquisition_rate   (substituted authoritative evidence acquired)

No network, no LLM, no real provider. Exit 0 only when every scenario passes
AND the aggregate metrics clear their thresholds. Existing thresholds are
never touched; this file owns its own thresholds JSON.

Usage (from backend/):
    python bench/eval_retrieval.py
    python bench/eval_retrieval.py --thresholds bench/golden/thresholds_retrieval_v1.json

Results:
    bench/results/retrieval_eval_v1.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_THRESHOLDS = (
    Path(__file__).resolve().parent / "golden" / "thresholds_retrieval_v1.json"
)


# ---------------------------------------------------------------------------
# Scenario harness
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    id: str
    description: str
    # routes: list of (url, list_of_responses) for respx
    routes: List[tuple] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)
    fallback_hits: List[str] = field(default_factory=list)
    primary_fallback: bool = False
    settings: Dict[str, Any] = field(default_factory=dict)
    pre_cool: List[str] = field(default_factory=list)
    double_pass: bool = False
    second_pass_urls: List[str] = field(default_factory=list)
    check: Callable[[Dict[str, Any]], bool] = lambda r: True


def _health_metrics(client, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    snap = client.health_snapshot()
    attempts = int(snap.get("fetch_attempts", 0) or 0)
    skips = int(snap.get("skipped_cooldown", 0) or 0)
    dup = int(snap.get("skipped_duplicate_failure", 0) or 0)
    fallback_q = int(snap.get("fallback_queries_issued", 0) or 0)
    fallback_a = int(snap.get("fallback_acquisitions", 0) or 0)

    def _rate(num: float, den: float) -> float:
        return round(num / den, 4) if den else 0.0

    out = {
        **snap,
        "cooldown_skip_rate": _rate(skips, skips + attempts),
        "duplicate_suppression_rate": _rate(dup, dup + attempts),
        "fallback_acquisition_rate": _rate(fallback_a, fallback_q),
        "primary_source_acquisition_rate": _rate(
            int(snap.get("primary_source_acquisitions", 0) or 0),
            max(1, int(snap.get("unique_success_domains", 0) or 0)),
        ),
    }
    if extra:
        out.update(extra)
    return out


async def _run_scenario(scenario: Scenario) -> Dict[str, Any]:
    import respx

    from app.agents import search as search_mod
    from app.agents.search import SearchClient, SearchResult
    from app.core.config import Settings

    settings = Settings(
        groq_api_key="retrieval-offline",
        search_primary_fallback_enabled=scenario.primary_fallback,
        **scenario.settings,
        _env_file=None,
    )
    client = SearchClient(settings)
    for domain in scenario.pre_cool:
        client.domain_registry.record_failure(domain, "forbidden")

    attached_urls: List[str] = []

    def _result(url: str) -> SearchResult:
        return SearchResult(
            title="Result", url=url, snippet="snippet",
            sub_question="offline retrieval probe query",
            search_type="general", provider="ddg_text",
        )

    ranked = [_result(u) for u in scenario.attachments]

    async def _no_sleep(_seconds):
        return None

    original_sleep = search_mod.asyncio.sleep
    original_providers = SearchClient._providers_for
    search_mod.asyncio.sleep = _no_sleep

    async def _fake_providers(self, query, search_type):
        return [_result(u) for u in scenario.fallback_hits]

    if scenario.fallback_hits:
        SearchClient._providers_for = _fake_providers

    try:
        with respx.mock(assert_all_called=False) as mock:
            for url, responses in scenario.routes:
                mock.get(url).mock(side_effect=list(responses))
            await client._attach_content(ranked)
            if scenario.double_pass:
                # Second pass over the same URL: must be suppressed/cooldown-skipped.
                await client._attach_content([_result(u) for u in scenario.attachments])
            if scenario.second_pass_urls:
                # Second pass over DIFFERENT documents on the same (cooled) host.
                await client._attach_content([_result(u) for u in scenario.second_pass_urls])
        for r in ranked:
            if r.content:
                attached_urls.append(r.url)
    finally:
        search_mod.asyncio.sleep = original_sleep
        SearchClient._providers_for = original_providers

    metrics = _health_metrics(client, {
        "attached_content_urls": attached_urls,
        "ranked_urls": [r.url for r in ranked],
    })
    metrics["passed"] = bool(scenario.check(metrics))
    return metrics


# ---------------------------------------------------------------------------
# Scenarios (mirror the deterministic unit tests at the metric level)
# ---------------------------------------------------------------------------

def _resp(status: int, *, body: str = "", ctype: str = "text/html", headers=None):
    import httpx

    h = {"content-type": ctype} if body else {}
    if headers:
        h.update(headers)
    if body:
        return httpx.Response(status, headers=h, text=body)
    return httpx.Response(status, headers=h)


def scenarios() -> List[Scenario]:
    body = "<html><body><p>Authoritative evidence body with measured findings.</p></body></html>"
    return [
        Scenario(
            id="forbidden_host_cools_and_skips",
            description="403 -> no retry, host cooled, second pass skips WITHOUT a request",
            routes=[("https://britannica-like.com/a", [_resp(403)])],
            attachments=["https://britannica-like.com/a"],
            # A DIFFERENT document on the same blocked publisher: the cooldown
            # (not the per-URL failure memory) must be what suppresses it.
            second_pass_urls=["https://britannica-like.com/b"],
            check=lambda m: (
                m["status_forbidden"] == 1
                and m["skipped_cooldown"] >= 1
                and m["cooldown_skip_rate"] > 0
                and m["successful_fetch_rate"] == 0.0
            ),
        ),
        Scenario(
            id="rate_limit_retry_succeeds",
            description="429 then 200 -> bounded retry recovers, host not cooled",
            routes=[("https://crossref-like.org/a",
                     [_resp(429, headers={"Retry-After": "0"}), _resp(200, body=body)])],
            attachments=["https://crossref-like.org/a"],
            settings={"search_fetch_retry_attempts": 2},
            check=lambda m: (
                m["fetch_successes"] == 1
                and m["cooldowns_opened"] == 0
                and m["successful_fetch_rate"] == 1.0
            ),
        ),
        Scenario(
            id="timeout_exhausts_and_cools",
            description="repeated timeouts -> bounded retries, then cooldown",
            routes=[("https://slowsite-like.com/a", [_timeout(), _timeout()])],
            attachments=["https://slowsite-like.com/a"],
            settings={"search_fetch_retry_attempts": 2,
                      "search_domain_failure_threshold": 2},
            check=lambda m: (
                m["fetch_timeouts"] == 1
                and m["retries_attempted"] >= 1
                and m["status_forbidden"] == 0
            ),
        ),
        Scenario(
            id="duplicate_failure_suppressed",
            description="same dead URL is not fetched twice within a run",
            routes=[("https://deadlink-like.com/a", [_resp(404, body="gone")])],
            attachments=["https://deadlink-like.com/a"],
            double_pass=True,
            check=lambda m: (
                m["fetch_attempts"] == 1
                and m["duplicate_suppression_rate"] > 0
            ),
        ),
        Scenario(
            id="primary_fallback_substitutes",
            description="unavailable authoritative host -> alternate authoritative acquisition",
            routes=[("https://blockedpublisher.com/a", [_resp(403)]),
                    ("https://census.gov/data/report", [_resp(200, body=body)])],
            attachments=["https://blockedpublisher.com/a"],
            fallback_hits=["https://census.gov/data/report"],
            primary_fallback=True,
            check=lambda m: (
                m["fallback_queries_issued"] >= 1
                and m["fallback_acquisitions"] >= 1
                and m["fallback_acquisition_rate"] == 1.0
                and m["unique_authoritative_domains"] >= 1
            ),
        ),
        Scenario(
            id="evidence_not_fabricated_on_failure",
            description="failed fetch yields no content and no false success",
            routes=[("https://blockedpublisher.com/b", [_resp(403)])],
            attachments=["https://blockedpublisher.com/b"],
            check=lambda m: (
                m["fetch_successes"] == 0
                and m["attached_content_urls"] == []
                and m["successful_fetch_rate"] == 0.0
            ),
        ),
    ]


def _timeout():
    import httpx

    return httpx.ReadTimeout("timed out")


# ---------------------------------------------------------------------------
# Aggregate / thresholds
# ---------------------------------------------------------------------------

AGGREGATE_KEYS = (
    "successful_fetch_rate",
    "forbidden_rate",
    "rate_limited_rate",
    "timeout_rate",
    "unique_authoritative_domains",
    "primary_source_acquisition_rate",
    "cooldown_skip_rate",
    "duplicate_suppression_rate",
    "fallback_acquisition_rate",
)


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rates = [r for r in rows]
    n = len(rates)
    metrics: Dict[str, Any] = {}
    for key in AGGREGATE_KEYS:
        vals = [float(r.get(key, 0) or 0) for r in rates if r.get(key) is not None]
        metrics[key] = round(sum(vals) / len(vals), 4) if vals else 0.0
    passed = sum(1 for r in rows if r.get("passed"))
    agg = {
        "scenarios_total": n,
        "scenarios_passed": passed,
        "scenarios_failed": n - passed,
        "metrics": metrics,
        "retrieval_health_pass_rate": round(passed / n, 4) if n else 0.0,
    }
    # Mirror the pass rate into `metrics` so the fold-in gate and the markdown
    # writer can print it uniformly with every other aggregate metric.
    metrics["retrieval_health_pass_rate"] = agg["retrieval_health_pass_rate"]
    return agg


def _check_thresholds(agg: Dict[str, Any], thresholds: Dict[str, Any]) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for name, floor in (thresholds.get("retrieval") or {}).items():
        actual = agg["metrics"].get(name)
        if actual is None:
            failures.append({"metric": name, "actual": None, "floor": floor,
                             "reason": "metric not produced"})
            continue
        if float(actual) < float(floor):
            failures.append({"metric": name, "actual": float(actual), "floor": float(floor)})
    return failures


async def evaluate(thresholds_path: str | None = None) -> Dict[str, Any]:
    rows = []
    for scenario in scenarios():
        try:
            metrics = await _run_scenario(scenario)
            metrics["error"] = None
        except Exception as exc:  # noqa: BLE001 - one scenario must not kill the eval
            metrics = {"error": f"{type(exc).__name__}: {exc}", "passed": False}
        metrics["id"] = scenario.id
        metrics["description"] = scenario.description
        rows.append(metrics)
        status = "ok" if metrics.get("passed") else "FAIL"
        print(f"[retr {status:>4}] {scenario.id:<38} {scenario.description}")

    agg = _aggregate(rows)
    thresholds = json.loads(
        Path(thresholds_path or DEFAULT_THRESHOLDS).read_text(encoding="utf-8")
    )
    failures = _check_thresholds(agg, thresholds)
    for failure in failures:
        print(f"  - {failure['metric']}: {failure['actual']} < {failure['floor']}")
    return {
        "suite": "mars-retrieval-health-eval",
        "version": thresholds.get("version", "v1"),
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "thresholds_path": str(thresholds_path or DEFAULT_THRESHOLDS),
        "aggregate": agg,
        "scenarios": rows,
        "thresholds": thresholds,
        "threshold_failures": failures,
        "passed": not failures and agg["scenarios_failed"] == 0,
    }


def write_json(report: Dict[str, Any], out_dir: Path = RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "retrieval_eval_v1.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", default=None, help="retrieval thresholds JSON path")
    parser.add_argument("--out", default=None, help="results directory")
    args = parser.parse_args()
    try:
        report = asyncio.run(evaluate(args.thresholds))
    except Exception as exc:  # noqa: BLE001
        print(f"retrieval evaluation failed to run: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2
    out_dir = Path(args.out) if args.out else RESULTS_DIR
    json_path = write_json(report, out_dir)
    print("=" * 62)
    print("MARS retrieval-health evaluation (v1)")
    print("=" * 62)
    for key in AGGREGATE_KEYS:
        print(f"  {key:<36} {report['aggregate']['metrics'].get(key)}")
    print(f"\nthreshold failures: {len(report['threshold_failures'])}")
    print(f"results: {json_path}")
    print(f"RESULT: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
