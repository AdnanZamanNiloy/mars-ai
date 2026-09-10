"""Verification-domain memory: contradictions, verification_results, citations."""

import asyncio

import aiosqlite

from app.db.sqlite import (
    get_run_trace,
    init_db,
    save_citations,
    save_contradictions,
    save_verification_results,
    start_research_run,
)


def _seed(db_path):
    async def _run():
        await init_db(db_path)
        await start_research_run(db_path, "run-v", "what is RAG", complexity="low", agent_count=2)

    asyncio.run(_run())


async def _count(db_path, table):
    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id = 'run-v'")
        return (await cur.fetchone())[0]


def test_contradictions_round_trip(tmp_path):
    db_path = str(tmp_path / "v.db")
    _seed(db_path)
    n = asyncio.run(save_contradictions(db_path, "run-v", [
        {"claim_a": "Costs fall", "source_a": "https://a.com",
         "claim_b": "Costs plateau", "source_b": "https://b.com"},
        {"claim_a": "", "claim_b": "Incomplete pair"},
    ]))
    assert n == 1
    assert asyncio.run(_count(db_path, "contradictions")) == 1


def test_verification_results_round_trip(tmp_path):
    db_path = str(tmp_path / "v.db")
    _seed(db_path)
    n = asyncio.run(save_verification_results(db_path, "run-v", [
        {"claim": "RAG combines search", "verified": True,
         "verification_score": 0.9, "verification_reason": "overlap 0.9"},
        {"claim": "", "verified": False},
    ]))
    assert n == 1
    assert asyncio.run(_count(db_path, "verification_results")) == 1


def test_citations_parse_legend(tmp_path):
    db_path = str(tmp_path / "v.db")
    _seed(db_path)
    report = "# Final Answer\nok [1].\n\nSources:\n[1] en.wikipedia.org — https://en.wikipedia.org/wiki/RAG\n[2] no-url-here"
    n = asyncio.run(save_citations(db_path, "run-v", report))
    assert n == 1
    assert asyncio.run(_count(db_path, "citations")) == 1
    assert asyncio.run(save_citations(db_path, "run-v", "no legend here")) == 0


def test_trace_includes_verification_tables(tmp_path):
    db_path = str(tmp_path / "v.db")
    _seed(db_path)
    asyncio.run(save_contradictions(db_path, "run-v", [
        {"claim_a": "A", "source_a": "https://a.com", "claim_b": "B", "source_b": "https://b.com"}]))
    trace = asyncio.run(get_run_trace(db_path, "run-v"))
    assert len(trace["contradictions"]) == 1
    assert trace["contradictions"][0]["claim_a"] == "A"
    assert trace["verification_results"] == []
    assert trace["citations"] == []
