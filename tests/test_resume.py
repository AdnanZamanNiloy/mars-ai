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
