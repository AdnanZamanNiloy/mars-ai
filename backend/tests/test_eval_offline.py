"""Deterministic offline golden evaluator (bench/eval_offline.py).

Covers the evaluator's core guarantees without running the full 26-query
suite in every test:
  * the golden query set + thresholds load and validate,
  * a single golden query runs through the offline path with NO network,
  * the produced metric tree has the expected keys,
  * the machine-section guard excludes appended sections from scoring,
  * a degraded metric trips threshold checking (non-zero exit semantics).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from bench.golden.loader import (
    GoldenValidationError,
    load_queries,
    load_thresholds,
    validate_queries,
    validate_thresholds,
)

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "bench" / "golden" / "queries_v1.json"
THRESHOLDS = REPO / "bench" / "golden" / "thresholds_v1.json"


# ---------------------------------------------------------------------------
# Golden set loading / validation
# ---------------------------------------------------------------------------

def test_golden_queries_load_and_validate():
    queries = load_queries(GOLDEN)
    assert 20 <= len(queries) <= 40
    assert len({q["id"] for q in queries}) == len(queries)
    categories = {q["category"] for q in queries}
    assert categories == {
        "factual_explanation", "current_trend", "comparison", "decision_policy",
        "ambiguous_term", "causal", "quantitative",
    }


def test_golden_malformed_query_set_is_rejected():
    data = json.loads(GOLDEN.read_text())
    data["queries"][0]["category"] = "not_a_category"
    errors = validate_queries(data)
    assert any("unknown category" in e for e in errors)


def test_golden_thresholds_load_and_validate():
    queries = load_queries(GOLDEN)
    thresholds = load_thresholds(THRESHOLDS, queries)
    assert thresholds["version"] == "v1"
    assert thresholds["aggregate"]
    assert set(thresholds["per_category"]) == {
        "factual_explanation", "current_trend", "comparison", "decision_policy",
        "ambiguous_term", "causal", "quantitative",
    }


def test_thresholds_cover_only_categories_with_queries():
    queries = load_queries(GOLDEN)
    data = json.loads(THRESHOLDS.read_text())
    data["per_category"]["no_such_category"] = {"answer_quality_mean": 0.0}
    errors = validate_thresholds(data, queries)
    assert any("no_such_category" in e for e in errors)


# ---------------------------------------------------------------------------
# Single-query offline run (no network)
# ---------------------------------------------------------------------------

def _run_one_offline(query_id: str):
    import tempfile

    from app.core.config import Settings
    from bench.eval_offline import run_query
    from bench.golden.fixtures_v1 import expected_by_id

    query = expected_by_id()[query_id]
    tmp = tempfile.mkdtemp(prefix="golden-test-")
    settings = Settings(groq_api_key="test", database_url=f"{tmp}/golden.db",
                        _env_file=None)
    return asyncio.run(run_query(query, settings))


def test_evaluator_runs_offline_with_no_network():
    """A golden query must execute with zero outbound HTTP. respx intercepts
    any httpx request; with no routes registered any call is recorded (and
    404s), so a non-empty call list fails the test."""
    import respx

    with respx.mock(assert_all_called=False) as mock:
        result = _run_one_offline("factual-rag")
        calls = list(mock.calls)
    assert result["error"] is None, result["error"]
    assert calls == [], f"evaluator attempted network calls: {calls}"


def test_evaluator_metric_keys_present():
    result = _run_one_offline("factual-rag")
    assert result["error"] is None
    expected_top = {
        "id", "category", "query", "query_type", "query_type_ok",
        "dimensions_hit", "forbidden_ok", "section_presence_rate",
        "citation", "evidence", "contradictions", "answer_quality",
        "support_rate", "minimums_ok",
    }
    assert expected_top <= set(result)
    assert {"resolution_rate", "no_dangling", "no_unsupported_numbers"} <= set(result["citation"])
    assert {"grade_distribution", "grade_ab_share", "verified_claims",
            "corroborated_claims", "distinct_domains", "primary_share"} <= set(result["evidence"])
    assert {"overall", "passed", "accuracy", "relevance", "answer_relevance"} <= set(result["answer_quality"])
    assert {"total", "resolved", "unresolved"} <= set(result["contradictions"])


# ---------------------------------------------------------------------------
# Machine-section guard
# ---------------------------------------------------------------------------

def test_machine_section_guard_excludes_appended_sections():
    from app.agents.sources import strip_machine_sections
    from bench.eval_offline import _citation_metrics, _unsupported_number_count

    answer = (
        "## Executive Summary\n\n"
        "Renewable capacity additions reached 510 GW in 2023 [1].\n\n"
        "## Key Findings\n\n"
        "- Solar accounted for three-quarters of additions [2].\n\n"
        "## Evidence integrity\n\n"
        "Citation density: 12% of sentences cited.\n\n"
        "## Source ledger\n\n"
        "- Documents read: 2 across 2 independent domain(s)\n\n"
        "## Sources\n\n"
        "[1] iea.org (official, primary) — https://iea.org/a\n\n"
        "[2] irena.org (official, primary) — https://irena.org/b"
    )
    stripped = strip_machine_sections(answer)
    assert "Evidence integrity" not in stripped
    assert "Source ledger" not in stripped
    assert "510 GW" in stripped

    facts = [
        {"claim": "Renewable capacity additions reached 510 GW in 2023",
         "source": "https://iea.org/a", "verified": True},
        {"claim": "Solar accounted for three-quarters of additions",
         "source": "https://irena.org/b", "verified": True},
    ]
    # The machine sections carry no markers and must not be counted as
    # uncited prose or scanned for numbers.
    c = _citation_metrics(answer)
    assert c["markers_total"] == 2
    assert c["no_dangling"] is True
    assert _unsupported_number_count(answer, facts) == 0


def test_machine_section_numbers_do_not_create_false_failures():
    """A machine section containing stray digits (panel counts) must not be
    treated as an unsupported numeric claim in the writer body."""
    from bench.eval_offline import _unsupported_number_count

    answer = (
        "## Executive Summary\n\n"
        "The market grew by 25% in 2024 [1].\n\n"
        "## Evidence & Confidence\n\n"
        "Well-supported: 3 verified facts across 2 sources; grades A=2, B=1.\n\n"
        "## Sources\n\n[1] gov.uk (official, primary) — https://gov.uk/report"
    )
    facts = [{"claim": "The market grew by 25% in 2024",
              "source": "https://gov.uk/report", "verified": True}]
    assert _unsupported_number_count(answer, facts) == 0


# ---------------------------------------------------------------------------
# Threshold failure semantics
# ---------------------------------------------------------------------------

def test_thresholds_fail_when_a_metric_is_degraded():
    from bench.eval_offline import check_thresholds

    thresholds = {"aggregate": {"answer_quality_mean": 62.0}, "per_category": {}}
    good = {"metrics": {"answer_quality_mean": 69.0}}
    bad = {"metrics": {"answer_quality_mean": 40.0}}
    assert check_thresholds(good, {}, thresholds) == []
    failures = check_thresholds(bad, {}, thresholds)
    assert len(failures) == 1
    assert failures[0]["metric"] == "answer_quality_mean"


def test_per_category_threshold_failure_is_scoped():
    from bench.eval_offline import check_thresholds

    thresholds = {"aggregate": {}, "per_category": {"causal": {"citation_resolution_rate": 1.0}}}
    category_result = {"causal": {"metrics": {"citation_resolution_rate": 0.5}}}
    failures = check_thresholds({"metrics": {}}, category_result, thresholds)
    assert failures and failures[0]["scope"] == "causal"


def test_missing_metric_is_a_failure_not_a_pass():
    """A metric the evaluator failed to produce must fail the gate, never
    silently pass — a missing measurement is not evidence of health."""
    from bench.eval_offline import check_thresholds

    failures = check_thresholds({"metrics": {}}, {},
                                {"aggregate": {"verified_claims_mean": 1.0},
                                 "per_category": {}})
    assert failures and failures[0]["actual"] is None


# ---------------------------------------------------------------------------
# Full aggregate is deterministic
# ---------------------------------------------------------------------------

def test_aggregate_is_deterministic_across_identical_runs():
    from bench.eval_offline import aggregate

    rows = [{
        "error": None, "query_type_ok": True, "dimensions_hit": 2,
        "dimensions_required": ["a", "b"], "forbidden_ok": True,
        "section_presence_rate": 1.0, "citation": {"resolution_rate": 1.0},
        "evidence": {"verified_claims": 2, "corroborated_claims": 1,
                     "distinct_domains": 2, "primary_share": 0.5,
                     "grade_ab_share": 0.9},
        "answer_quality": {"overall": 70, "answer_relevance": 0.5},
        "support_rate": 0.5, "minimums_ok": True,
    }]
    a = aggregate(rows)
    b = aggregate(rows)
    assert a == b
    assert a["metrics"]["query_type_accuracy"] == 1.0
    assert a["metrics"]["answer_quality_mean"] == 70.0


def test_golden_validation_error_is_raised_on_missing_file(tmp_path):
    with pytest.raises(GoldenValidationError):
        load_queries(tmp_path / "does_not_exist.json")


# ---------------------------------------------------------------------------
# Adaptive-depth routing golden coverage (bench/eval_depth.py)
# ---------------------------------------------------------------------------

def _depth_settings():
    from app.core.config import Settings

    return Settings(groq_api_key="test", _env_file=None)


def test_depth_scenarios_all_pass_and_cover_every_branch():
    """Every declared adaptive-depth scenario routes as expected. This is the
    regression contract for continue/stop/hard-wall."""
    from bench import eval_depth
    from bench.golden import depth_fixtures_v1

    settings = _depth_settings()
    rows = [eval_depth.evaluate_scenario(s, settings)
            for s in depth_fixtures_v1.all_scenarios()]
    failed = [r for r in rows if not r["passed"]]
    assert not failed, failed
    agg = eval_depth.aggregate(rows)
    assert agg["metrics"]["depth_routing_pass_rate"] == 1.0


def test_depth_expand_scenarios_continue():
    from bench import eval_depth
    from bench.golden import depth_fixtures_v1

    settings = _depth_settings()
    for builder in (
        depth_fixtures_v1.scenario_high_impact_uncorroborated,
        depth_fixtures_v1.scenario_thin_dimension,
        depth_fixtures_v1.scenario_unresolved_severe_contradiction,
    ):
        scenario = builder()
        row = eval_depth.evaluate_scenario(scenario, settings)
        assert row["actual_decision"] == "expand", scenario["id"]
        assert row["reason_ok"], (scenario["id"], row["reason"])


def test_depth_stop_scenarios_finalize():
    from bench import eval_depth
    from bench.golden import depth_fixtures_v1

    settings = _depth_settings()
    for builder in (
        depth_fixtures_v1.scenario_sufficient,
        depth_fixtures_v1.scenario_hard_wall,
    ):
        scenario = builder()
        row = eval_depth.evaluate_scenario(scenario, settings)
        assert row["actual_decision"] == "finalize", scenario["id"]


def test_depth_hard_wall_records_limitations():
    from bench import eval_depth
    from bench.golden import depth_fixtures_v1

    scenario = depth_fixtures_v1.scenario_hard_wall()
    row = eval_depth.evaluate_scenario(scenario, _depth_settings())
    assert row["hard_wall_reached"] is True
    assert row["evidence_gaps_remain"] is True
    assert row["limitations"], "hard wall must record the outstanding gaps"
    assert "gap" in row["limitations"][0].lower()


def test_depth_routing_gate_bites_on_a_broken_fixture():
    """Prove the routing gate actually fails when a scenario's decision
    regresses: flip one fixture's expected decision and assert the aggregate
    drops below the 1.0 threshold, producing a violation."""
    from bench import eval_depth
    from bench.golden import depth_fixtures_v1

    scenarios = depth_fixtures_v1.all_scenarios()
    scenarios[0]["expect"]["expected_decision"] = "finalize"  # deliberately wrong

    settings = _depth_settings()
    rows = [eval_depth.evaluate_scenario(s, settings) for s in scenarios]
    agg = eval_depth.aggregate(rows)
    assert agg["metrics"]["depth_routing_pass_rate"] == 0.8
    failures = eval_depth.check_thresholds(agg, {"depth": {"depth_routing_pass_rate": 1.0}})
    assert failures and failures[0]["metric"] == "depth_routing_pass_rate"


def test_depth_reason_token_mismatch_fails():
    """A scenario whose expected reason token is absent must fail — a routing
    decision with the wrong WHY is still a regression."""
    from bench import eval_depth
    from bench.golden import depth_fixtures_v1

    scenario = depth_fixtures_v1.scenario_sufficient()
    scenario["expect"]["reason_tokens"] = ["a token that never appears"]
    row = eval_depth.evaluate_scenario(scenario, _depth_settings())
    assert row["actual_decision"] == "finalize"
    assert row["reason_ok"] is False
    assert row["passed"] is False


def test_depth_thresholds_file_loads_and_gates_at_one():
    from bench import eval_depth

    report = eval_depth.evaluate()
    assert report["passed"] is True
    assert report["threshold_failures"] == []
    assert report["thresholds"]["depth"]["depth_routing_pass_rate"] == 1.0
