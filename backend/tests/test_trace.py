"""Research Replay trace endpoint tests (Phase 3.4)."""
import json

import httpx
import pytest

from app.api.routes import router as api_router
from app.core.config import Settings
from app.db.sqlite import (
    complete_research_run,
    init_db,
    record_event,
    save_agent_tasks,
    save_claims,
    save_sources,
    start_research_run,
)

RUN_ID = "trace-test-run"


@pytest.fixture
def app_with_db(tmp_path):
    db_path = str(tmp_path / "trace.db")

    async def _setup():
        await init_db(db_path)
        await start_research_run(db_path, RUN_ID, "what is RAG", complexity="low", agent_count=1)
        await save_agent_tasks(db_path, RUN_ID, [{
            "id": 1, "question": "what is RAG", "axis": "definition",
            "search_type": "encyclopedia", "priority": 1,
        }])
        await save_sources(db_path, RUN_ID, [{"url": "https://arxiv.org/a", "reliability_score": 0.95}])
        await save_claims(db_path, RUN_ID, [{
            "claim": "RAG retrieves documents", "source": "https://arxiv.org/a",
            "confidence": 0.9, "verified": True,
        }])
        await record_event(db_path, RUN_ID, "planner", "end", payload=json.dumps({"sub_questions": 1}))
        await record_event(db_path, RUN_ID, "budget", "budget_check", payload=json.dumps({"estimated_cost": 0.01}))
        await complete_research_run(db_path, RUN_ID, "completed", confidence=0.8, estimated_cost=0.01)

    import asyncio
    asyncio.run(_setup())

    from fastapi import FastAPI

    app = FastAPI()
    app.state.settings = Settings(groq_api_key="k", database_url=db_path, _env_file=None)
    app.include_router(api_router, prefix="/api")
    return app


async def test_trace_returns_complete_ordered_reconstruction(app_with_db):
    transport = httpx.ASGITransport(app=app_with_db)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/api/research/{RUN_ID}/trace")
    assert response.status_code == 200
    trace = response.json()

    assert trace["run_id"] == RUN_ID
    assert trace["status"] == "completed"
    assert trace["confidence"] == 0.8
    assert len(trace["plan"]) == 1
    assert trace["plan"][0]["axis"] == "definition"
    assert len(trace["sources"]) == 1
    assert len(trace["claims"]) == 1
    assert trace["claims"][0]["verified"] == 1
    # Events include the budget check, not just node transitions.
    event_types = [(e["node"], e["event_type"]) for e in trace["events"]]
    assert ("planner", "end") in event_types
    assert ("budget", "budget_check") in event_types
    # Chronological order preserved.
    ids = [e["id"] for e in trace["events"]]
    assert ids == sorted(ids)


async def test_trace_unknown_run_404(app_with_db):
    transport = httpx.ASGITransport(app=app_with_db)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/research/nope/trace")
    assert response.status_code == 404
