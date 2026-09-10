"""Research Memory persistence tests (Phase 2.7 / 2.10).

Runs the real workflow with stubbed agents + a real temp SQLite DB, then
asserts every memory table has rows joinable by run_id.
"""
import json

import aiosqlite
import pytest

import app.graph.workflow as wf
from app.core.budget import BudgetTracker
from app.core.config import Settings
from app.core.llm import LLMClient

RUN_ID = "run-test-1234"


async def test_memory_tables_populated_and_joinable(tmp_path, monkeypatch):
    db_path = str(tmp_path / "memory_test.db")
    settings = Settings(groq_api_key="k", _env_file=None)
    from app.db.sqlite import init_db
    await init_db(db_path)

    tracker = BudgetTracker(settings)
    state = wf.build_initial_state("what is RAG", 3)
    state["budget_tracker"] = tracker

    sub_questions = [{
        "id": 1, "question": "what is RAG definition", "axis": "definition",
        "search_type": "encyclopedia", "priority": 1, "depends_on": [],
        "coverage_goal": "", "domain": "general", "minimum_sources": 2,
        "stop_condition": "sufficient evidence for this axis",
    }]
    search_results = [{
        "url": "https://arxiv.org/abs/2005.11401", "title": "RAG",
        "snippet": "retrieval augmented generation", "reliability_score": 0.95,
        "content": "retrieval augmented generation grounds answers",
    }]
    facts = [{
        "claim": "RAG retrieves documents before generation",
        "source": "https://arxiv.org/abs/2005.11401", "confidence": 0.9,
        "verified": True, "verification_score": 0.8, "verification_reason": "ok",
    }]

    async def fake_planner(llm, query, critique_feedback="", today=""):
        return sub_questions

    async def fake_summarizer(llm, query, search_results=None, specialist_role="general"):
        return facts

    async def fake_critic(llm, query, facts=None, iteration=1, max_iterations=3, contradictions=None):
        return {"is_sufficient": True, "reason": "complete", "improved_queries": [], "confidence": 0.9}

    async def fake_synthesizer(llm, query, facts=None):
        return "synthesized answer"

    monkeypatch.setattr(wf, "planner_agent", fake_planner)
    monkeypatch.setattr(wf, "summarizer_agent", fake_summarizer)
    monkeypatch.setattr(wf, "critic_agent", fake_critic)
    monkeypatch.setattr(wf, "synthesizer_agent", fake_synthesizer)

    class NoSearch:
        async def run_search(self, questions):
            return search_results

    workflow = wf.create_workflow(LLMClient(settings), NoSearch())
    final = None
    async for snap in workflow.astream(state, stream_mode="values"):
        final = snap
    assert final and final.get("final_report")

    # Now emulate the routes.py persistence calls (the route owns run_id and
    # incremental writes; workflow state carries the data).
    from app.db.sqlite import (
        complete_research_run, record_event, save_agent_tasks, save_claims,
        save_sources, start_research_run,
    )
    await start_research_run(db_path, RUN_ID, "what is RAG", complexity="low", agent_count=1)
    await save_agent_tasks(db_path, RUN_ID, sub_questions)
    await save_sources(db_path, RUN_ID, search_results)
    await save_claims(db_path, RUN_ID, facts)
    await record_event(db_path, RUN_ID, "planner", "end", payload=json.dumps({"sub_questions": 1}))
    await record_event(db_path, RUN_ID, "finalize", "end", payload="")
    await complete_research_run(db_path, RUN_ID, "completed", confidence=0.8, estimated_cost=0.01)

    async with aiosqlite.connect(db_path) as db:
        counts = {}
        for table in ("research_runs", "agent_tasks", "sources", "claims", "agent_events", "research_reports"):
            cur = await db.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = (await cur.fetchone())[0]
        joined = await db.execute(
            "SELECT r.id, t.question, s.url, c.claim, e.node "
            "FROM research_runs r "
            "JOIN agent_tasks t ON t.run_id = r.id "
            "JOIN sources s ON s.run_id = r.id "
            "JOIN claims c ON c.run_id = r.id "
            "JOIN agent_events e ON e.run_id = r.id "
            "WHERE r.id = ?",
            (RUN_ID,),
        )
        joined_rows = await joined.fetchall()

    assert counts["research_runs"] == 1
    assert counts["agent_tasks"] == 1
    assert counts["sources"] == 1
    assert counts["claims"] == 1
    assert counts["agent_events"] == 2
    assert joined_rows, "all five tables must join on run_id"

    run_row = await _fetchone(db_path, "SELECT status, confidence FROM research_runs WHERE id = ?", (RUN_ID,))
    assert run_row["status"] == "completed"
    assert run_row["confidence"] == 0.8


async def _fetchone(db_path, query, params):
    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    try:
        cur = await db.execute(query, params)
        return dict(await cur.fetchone())
    finally:
        await db.close()
