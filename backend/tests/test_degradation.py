"""Degradation flag: fallback paths record themselves; the stream reports them."""

import asyncio
import json

import httpx
from fastapi import FastAPI

from app.api.routes import limiter, router as api_router
from app.core.config import Settings
from app.core.degradation import (
    clear_fallbacks,
    record_fallback,
    reset_fallbacks,
    take_fallbacks,
)


def test_recorder_scoped_and_deduped():
    record_fallback("planner")  # outside a tracked request: no-op
    assert take_fallbacks() == []
    reset_fallbacks()
    record_fallback("planner")
    record_fallback("planner")
    record_fallback("synthesizer")
    assert take_fallbacks() == ["planner", "synthesizer"]
    clear_fallbacks()
    record_fallback("planner")
    assert take_fallbacks() == []


def test_planner_fallback_records():
    from app.agents.planner import planner_agent

    class ExplodingLLM:
        async def generate_json(self, *a, **k):
            raise RuntimeError("down")

    reset_fallbacks()
    try:
        result = asyncio.run(planner_agent(ExplodingLLM(), "Some deep query here?"))
        assert len(result) == 4  # fallback plan still returned
        assert take_fallbacks() == ["planner"]
    finally:
        clear_fallbacks()


def test_synthesizer_fallback_records():
    from app.agents.synthesizer import synthesizer_agent

    class ExplodingLLM:
        async def generate_json(self, *a, **k):
            raise RuntimeError("down")

    facts = [{"claim": "Distinct transformer study finding", "source": "https://arxiv.org/x",
              "confidence": 0.9}]
    reset_fallbacks()
    try:
        asyncio.run(synthesizer_agent(ExplodingLLM(), "What is RAG?", facts))
        assert take_fallbacks() == ["synthesizer"]
    finally:
        clear_fallbacks()


def test_final_report_event_carries_degraded_list(tmp_path):
    """Route drains the per-request recorder into the final_report event and
    persists one fallback row per agent in agent_events."""
    from app.db.sqlite import init_db

    db_path = str(tmp_path / "deg.db")
    asyncio.run(init_db(db_path))

    class StubDegradedWorkflow:
        async def astream(self, state, stream_mode=None):
            record_fallback("planner")
            yield {"final_report": "# Final Answer\nok", "confidence": 0.5}

    settings = Settings(groq_api_key="k", database_url=db_path, _env_file=None)
    app = FastAPI()
    app.state.workflow = StubDegradedWorkflow()
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
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    finals = [e for e in events if e.get("type") == "final_report"]
    assert len(finals) == 1
    assert finals[0]["degraded"] == ["planner"]

    import aiosqlite

    async def _check():
        async with aiosqlite.connect(db_path) as db:
            cur = await db.execute(
                "SELECT node, event_type FROM agent_events WHERE event_type = 'fallback'")
            return await cur.fetchall()

    rows = asyncio.run(_check())
    assert ("planner", "fallback") in [(r[0], r[1]) for r in rows]
