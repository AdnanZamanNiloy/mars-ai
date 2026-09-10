"""Evaluation Lab (4.1): scoring, persistence, and query-set validity.

Live pipeline runs are exercised by scripts/run_eval.py on demand — never
from pytest (no API keys, no network, no wall-clock cost here).
"""

import asyncio
import json
from pathlib import Path

import pytest

from app.core.eval import format_trend, score_query, summarize_batch
from app.db.sqlite import (
    eval_batch_rows,
    init_db,
    latest_eval_batches,
    save_evaluation_run,
)


def _expectation(**over):
    base = {"min_claims": 5, "min_verified": 2, "expect_decision_options": True}
    base.update(over)
    return base


def _metrics(**over):
    base = {
        "status": "completed", "confidence": 0.7, "claims": 6, "verified": 4,
        "sources": 8, "decisions": 3, "contradictions": 1, "cost": 0.001,
    }
    base.update(over)
    return base


def test_score_query_passes_on_healthy_run():
    checks = score_query(_expectation(), _metrics())
    assert checks["passed"] is True
    assert all([checks["claims_ok"], checks["verified_ok"], checks["decisions_ok"], checks["completed_ok"]])


def test_score_query_fails_failed_run():
    checks = score_query(_expectation(), _metrics(status="failed"))
    assert checks["passed"] is False
    assert checks["completed_ok"] is False


def test_score_query_fails_thin_evidence():
    checks = score_query(_expectation(), _metrics(claims=2, verified=0))
    assert checks == {
        "claims_ok": False, "verified_ok": False, "decisions_ok": True,
        "completed_ok": True, "passed": False,
    }


def test_score_query_skips_decision_check_for_factual():
    checks = score_query(_expectation(expect_decision_options=False), _metrics(decisions=1))
    assert checks["decisions_ok"] is True and checks["passed"] is True


def test_score_query_missing_metrics_fail_closed():
    checks = score_query(_expectation(), {"status": "completed"})
    assert checks["passed"] is False


def test_summarize_batch_headlines():
    rows = [
        {**_metrics(confidence=0.8, claims=6, verified=4, contradictions=0, cost=0.001), "passed": True},
        {**_metrics(confidence=0.4, claims=2, verified=0, contradictions=2, cost=0.002), "passed": False},
    ]
    summary = summarize_batch(rows)
    assert summary["queries"] == 2
    assert summary["pass_rate"] == 0.5
    assert summary["avg_confidence"] == pytest.approx(0.6)
    assert summary["avg_claims"] == 4.0
    assert summary["avg_verified"] == 2.0
    assert summary["contradiction_rate"] == 0.5
    assert summary["avg_cost"] == pytest.approx(0.0015)


def test_summarize_batch_empty():
    assert summarize_batch([])["queries"] == 0


def test_format_trend_with_and_without_previous():
    cur = summarize_batch([{**_metrics(), "passed": True}])
    first = format_trend(cur, None)
    assert "no prior batch" in first
    prev = summarize_batch([{**_metrics(confidence=0.5), "passed": False}])
    second = format_trend(cur, prev)
    assert "was" in second


def test_evaluation_rows_round_trip(tmp_path):
    db_path = str(tmp_path / "eval.db")
    asyncio.run(init_db(db_path))
    asyncio.run(save_evaluation_run(db_path, {
        "eval_batch": "b1", "query_id": "eval-01", "query": "q?", "mode": "quick",
        "run_id": "r1", "status": "completed", "confidence": 0.7, "claims": 6,
        "verified": 4, "sources": 8, "contradictions": 1,
        "recommended_option": "A", "cost": 0.001, "passed": True,
    }))
    rows = asyncio.run(eval_batch_rows(db_path, "b1"))
    assert len(rows) == 1
    assert rows[0]["passed"] is True
    assert rows[0]["contradictions"] == 1
    assert asyncio.run(latest_eval_batches(db_path)) == ["b1"]


def test_eval_queries_file_valid():
    path = Path(__file__).parent / "eval_queries.json"
    queries = json.loads(path.read_text(encoding="utf-8"))
    assert 15 <= len(queries) <= 20, f"expected 15-20 queries, got {len(queries)}"
    ids = [q["id"] for q in queries]
    assert len(set(ids)) == len(ids), "duplicate query ids"
    for q in queries:
        assert len(q["query"]) >= 5, q["id"]
        assert q["mode"] in ("quick", "standard", "deep"), q["id"]
        assert isinstance(q["min_claims"], int) and isinstance(q["min_verified"], int)
        assert isinstance(q["expect_decision_options"], bool)


def test_run_one_reads_quality_metrics_from_trace():
    """Stream findings lack verified flags (length-based emission), so the
    harness must score persisted trace truth — verified here end to end
    with a mocked transport, no network."""
    import httpx
    from scripts.run_eval import run_one

    stream_body = (
        '{"type": "progress", "request_id": "run-1", "message": "Query received"}\n'
        '{"type": "findings", "items": [{"claim": "c1", "source": "s", "confidence": 0.5}]}\n'
        '{"type": "budget", "estimated_cost": 0.0012, "limit": 0.5, "llm_calls": 3, "over_budget": false}\n'
        '{"type": "final_report", "report": "# Final Answer\\nok", "confidence": 0.7}\n'
    )
    trace_body = {
        "status": "completed",
        "claims": [
            {"claim": "c1", "verified": 1},
            {"claim": "c2", "verified": 1},
            {"claim": "c3", "verified": 0},
        ],
        "sources": [{}, {}, {}, {}],
        "decisions": [
            {"option_label": "A", "is_recommended": False},
            {"option_label": "B", "is_recommended": True},
        ],
        "final_report": {
            "report_markdown": "# Final Answer\nok\n\n# Contradictions\n- a vs b\n- c vs d\n",
            "confidence": 0.7,
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/trace"):
            return httpx.Response(200, json=trace_body)
        return httpx.Response(200, content=stream_body.encode())

    async def _run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await run_one(client, "http://test", {"query": "q?", "mode": "quick"}, 30.0)

    metrics = asyncio.run(_run())
    assert metrics["run_id"] == "run-1"
    assert metrics["status"] == "completed"
    assert metrics["claims"] == 3
    assert metrics["verified"] == 2  # from trace, NOT the unverified stream item
    assert metrics["sources"] == 4
    assert metrics["decisions"] == 2
    assert metrics["recommended_option"] == "B"
    assert metrics["contradictions"] == 2
    assert metrics["confidence"] == 0.7
    assert metrics["cost"] == 0.0012


def test_run_eval_rejects_malformed_queries_file(tmp_path):
    """CLI validation fires before any network/DB touch — cheap subprocess."""
    import subprocess
    import sys

    bad = tmp_path / "bad.json"
    bad.write_text('[{"id": "x", "query": "too short to matter"}]', encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "scripts/run_eval.py", "--queries", str(bad)],
        capture_output=True, text=True, cwd=Path(__file__).parent.parent,
    )
    assert proc.returncode == 2
    assert "must be a list" in proc.stderr


def test_migration_adds_degraded_column_to_old_table(tmp_path):
    """Pre-degraded evaluation_runs tables gain the column in place."""
    import aiosqlite

    db_path = str(tmp_path / "old_eval.db")

    async def _setup():
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "CREATE TABLE evaluation_runs (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " eval_batch TEXT NOT NULL, query_id TEXT NOT NULL, query TEXT NOT NULL,"
                " mode TEXT NOT NULL, run_id TEXT, status TEXT NOT NULL, confidence REAL,"
                " claims INTEGER, verified INTEGER, sources INTEGER, contradictions INTEGER,"
                " recommended_option TEXT, cost REAL, checks_passed INTEGER NOT NULL,"
                " created_at TEXT NOT NULL)"
            )
            await db.execute(
                "INSERT INTO evaluation_runs (eval_batch, query_id, query, mode, status,"
                " checks_passed, created_at) VALUES ('b0','q','q?','quick','completed',1,'now')"
            )
            await db.commit()

    asyncio.run(_setup())
    asyncio.run(init_db(db_path))
    rows = asyncio.run(eval_batch_rows(db_path, "b0"))
    assert len(rows) == 1
    assert rows[0]["degraded"] == []


def test_degraded_round_trip(tmp_path):
    db_path = str(tmp_path / "deg.db")
    asyncio.run(init_db(db_path))
    asyncio.run(save_evaluation_run(db_path, {
        "eval_batch": "b1", "query_id": "q1", "query": "q?", "mode": "quick",
        "run_id": "r1", "status": "completed", "confidence": 0.6, "claims": 5,
        "verified": 2, "sources": 6, "contradictions": 0,
        "recommended_option": None, "cost": 0.001, "passed": True,
        "degraded": ["synthesizer", "planner", "planner"],
    }))
    rows = asyncio.run(eval_batch_rows(db_path, "b1"))
    assert rows[0]["degraded"] == ["planner", "synthesizer"]


def test_summarize_counts_degraded():
    rows = [
        {**_metrics(), "passed": True, "degraded": []},
        {**_metrics(), "passed": True, "degraded": ["planner"]},
    ]
    summary = summarize_batch(rows)
    assert summary["degraded"] == 1
    assert summary["degraded_rate"] == 0.5
    assert "degraded rows" in format_trend(summary, None)


def test_run_one_reads_degraded_from_trace_fallbacks():
    """Degraded agents come from persisted fallback events, not the stream."""
    import httpx
    from scripts.run_eval import run_one

    stream_body = (
        '{"type": "progress", "request_id": "run-9", "message": "ok"}\n'
        '{"type": "final_report", "report": "# Final Answer\\nok", "confidence": 0.6}\n'
    )
    trace_body = {
        "status": "completed",
        "claims": [{"claim": "c1", "verified": 1}],
        "sources": [{}],
        "decisions": [],
        "events": [
            {"node": "synthesizer", "event_type": "fallback"},
            {"node": "planner", "event_type": "fallback"},
            {"node": "search", "event_type": "end"},
        ],
        "final_report": {"report_markdown": "# Final Answer\nok", "confidence": 0.6},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/trace"):
            return httpx.Response(200, json=trace_body)
        return httpx.Response(200, content=stream_body.encode())

    async def _run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await run_one(client, "http://test", {"query": "q?", "mode": "quick"}, 30.0)

    metrics = asyncio.run(_run())
    assert metrics["degraded"] == ["planner", "synthesizer"]
