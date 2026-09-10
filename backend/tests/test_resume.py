"""Durable checkpointing + resume tests (Phase 3.3)."""
import asyncio

import httpx
import pytest
from fastapi import FastAPI

from app.core.config import Settings
from app.db.sqlite import (
    complete_research_run,
    init_db,
    load_state_for_resume,
    save_agent_tasks,
    save_critic_review,
    save_sources,
    save_claims,
    start_research_run,
)


def _seed(db_path):
    async def _run():
        await init_db(db_path)
        await start_research_run(db_path, "run-fail-1", "what is RAG", complexity="low", agent_count=1)
        await save_agent_tasks(db_path, "run-fail-1", [
            {"question": "what is RAG definition", "axis": "definition", "search_type": "encyclopedia", "priority": 1},
        ])
        await save_sources(db_path, "run-fail-1", [{"url": "https://arxiv.org/a", "reliability_score": 0.95}])
        await save_claims(db_path, "run-fail-1", [{
            "claim": "RAG retrieves documents before generation",
            "source_url": "https://arxiv.org/a", "confidence": 0.9, "verified": 1,
        }])
        await save_critic_review(db_path, "run-fail-1", 2, {
            "is_sufficient": False, "reason": "incomplete", "confidence": 0.5,
        })
        await complete_research_run(db_path, "run-fail-1", "failed", confidence=0.5, estimated_cost=0.01)

    asyncio.run(_run())


def test_load_state_rebuilds_from_persisted_rows(tmp_path):
    db_path = str(tmp_path / "resume.db")
    _seed(db_path)
    state = asyncio.run(load_state_for_resume(db_path, "run-fail-1"))
    assert state is not None
    assert state["query"] == "what is RAG"
    assert state["sub_questions"][0]["question"] == "what is RAG definition"
    assert state["search_results"][0]["url"] == "https://arxiv.org/a"
    assert state["facts"][0]["verified"] is True
    assert state["iteration"] == 2
    assert state["critique"]["is_sufficient"] is False


def test_completed_run_is_not_resumable(tmp_path):
    db_path = str(tmp_path / "resume2.db")

    async def _run():
        await init_db(db_path)
        await start_research_run(db_path, "run-ok", "q", complexity="low", agent_count=1)
        await complete_research_run(db_path, "run-ok", "completed", confidence=0.9, estimated_cost=0.0)

    asyncio.run(_run())
    state = asyncio.run(load_state_for_resume(db_path, "run-ok"))
    assert state is None, "completed runs must not be resumable"


def test_unknown_run_is_not_resumable(tmp_path):
    db_path = str(tmp_path / "resume3.db")
    asyncio.run(init_db(db_path))
    assert asyncio.run(load_state_for_resume(db_path, "ghost")) is None


def test_resume_endpoint_rejects_unknown_run(tmp_path):
    db_path = str(tmp_path / "resume4.db")
    asyncio.run(init_db(db_path))

    from app.api.routes import router as api_router

    app = FastAPI()
    app.state.settings = Settings(groq_api_key="k", database_url=db_path, _env_file=None)
    app.state.workflow = object()  # must not be reached for unknown runs
    app.include_router(api_router, prefix="/api")

    async def _call():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/api/research/ghost/resume")

    response = asyncio.run(_call())
    assert response.status_code == 409


def test_resume_reruns_critic_not_planner_or_search(tmp_path, monkeypatch):
    """3.3 DoD: resume continues from persisted state instead of re-running
    the whole pipeline — planner/search must never be invoked."""
    db_path = str(tmp_path / "resume5.db")
    _seed(db_path)

    import app.graph.workflow as wf
    from app.core.llm import LLMClient

    visited = []
    seen_tracker = {}

    async def fake_critic(llm, query, facts=None, iteration=1, max_iterations=3, contradictions=None):
        visited.append("critic")
        # Not sufficient → forces route_after_critic through depth_controller.
        return {"is_sufficient": False, "reason": "need more", "improved_queries": [], "confidence": 0.4}

    def spy_decide(state):
        # Depth controller + report builder read the tracker from state, not
        # the ContextVar — resume must wire it in or budget checks are lost.
        seen_tracker["present"] = state.get("budget_tracker") is not None
        return "synthesizer"

    async def fake_synthesizer(llm, query, facts=None):
        visited.append("synthesizer")
        return "resumed answer"

    monkeypatch.setattr(wf, "critic_agent", fake_critic)
    monkeypatch.setattr(wf.depth_controller, "decide", spy_decide)
    monkeypatch.setattr(wf, "synthesizer_agent", fake_synthesizer)

    from app.api.routes import router as api_router

    app = FastAPI()
    app.state.settings = Settings(groq_api_key="k", database_url=db_path, _env_file=None)
    app.include_router(api_router, prefix="/api")

    async def _call():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/api/research/run-fail-1/resume")

    response = asyncio.run(_call())
    assert response.status_code == 200
    assert visited == ["critic", "synthesizer"], visited
    assert seen_tracker.get("present") is True, "budget_tracker missing from resume state"

    # Run marked completed, report persisted.
    import aiosqlite

    async def _check():
        async with aiosqlite.connect(db_path) as db:
            cur = await db.execute("SELECT status FROM research_runs WHERE id = 'run-fail-1'")
            status = (await cur.fetchone())[0]
            cur = await db.execute("SELECT report_markdown FROM final_reports WHERE run_id = 'run-fail-1'")
            report = (await cur.fetchone())[0]
        return status, report

    status, report = asyncio.run(_check())
    assert status == "completed"
    assert "# Final Answer" in report


def test_incremental_sources_saved_without_duplicates(tmp_path):
    """Expansion passes persist only unseen source URLs (no re-inserts)."""
    from app.api.routes import limiter
    from app.api.routes import router as api_router
    from app.db.sqlite import init_db

    db_path = str(tmp_path / "incsrc.db")
    asyncio.run(init_db(db_path))

    class StubSourceWorkflow:
        async def astream(self, state, stream_mode=None):
            yield {"search_results": [
                {"url": "https://a.com/1", "snippet": "s1"},
                {"url": "https://b.com/2", "snippet": "s2"},
            ]}
            yield {"search_results": [
                {"url": "https://a.com/1", "snippet": "s1"},
                {"url": "https://b.com/2", "snippet": "s2"},
                {"url": "https://c.com/3", "snippet": "s3"},
            ]}
            yield {"final_report": "# Final Answer\nok", "confidence": 0.6}

    settings = Settings(groq_api_key="k", database_url=db_path, _env_file=None)
    app = FastAPI()
    app.state.workflow = StubSourceWorkflow()
    app.state.settings = settings
    app.state.limiter = limiter
    app.include_router(api_router, prefix="/api")
    limiter.reset()

    async def _call():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/api/research/stream", json={"query": "valid research query here"})

    response = asyncio.run(_call())
    limiter.reset()
    assert response.status_code == 200

    import aiosqlite

    async def _check():
        async with aiosqlite.connect(db_path) as db:
            cur = await db.execute("SELECT url FROM sources ORDER BY id")
            urls = [r[0] for r in await cur.fetchall()]
            cur = await db.execute("SELECT COUNT(*) FROM evidence")
            ev = (await cur.fetchone())[0]
        return urls, ev

    urls, ev = asyncio.run(_check())
    assert urls == ["https://a.com/1", "https://b.com/2", "https://c.com/3"], urls
    assert ev == 3, ev


def test_resume_restores_run_max_iterations(tmp_path):
    """load_state_for_resume returns the run's own ceiling, not hardcoded 3."""
    from app.db.sqlite import init_db, load_state_for_resume

    db_path = str(tmp_path / "maxiter.db")
    asyncio.run(init_db(db_path))

    async def _seed():
        await start_research_run(db_path, "run-deep", "q", complexity="high",
                                 agent_count=5, max_iterations=5)
        await complete_research_run(db_path, "run-deep", "timeout", confidence=0.0,
                                    estimated_cost=0.0)

    asyncio.run(_seed())
    state = asyncio.run(load_state_for_resume(db_path, "run-deep"))
    assert state is not None
    assert state["max_iterations"] == 5
