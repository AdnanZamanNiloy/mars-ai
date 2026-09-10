"""Route-level behavior: per-request timeout (1.3) and rate limiting (1.7)."""
import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI

from app.api.routes import limiter, router as api_router
from app.core.config import Settings


class StubSlowWorkflow:
    """A workflow that hangs longer than the request timeout."""

    def __init__(self, delay: float):
        self.delay = delay

    async def astream(self, state, stream_mode=None):
        await asyncio.sleep(self.delay)
        yield {}


class StubFastWorkflow:
    async def astream(self, state, stream_mode=None):
        yield {"final_report": "# Final Answer\nok", "confidence": 0.5}


def _build_app(workflow, settings: Settings) -> FastAPI:
    app = FastAPI()
    app.state.workflow = workflow
    app.state.settings = settings
    app.state.limiter = limiter
    app.include_router(api_router, prefix="/api")
    return app


@pytest.fixture(autouse=True)
def _reset_limiter():
    """The route decorator captures the module limiter's closure, so its
    in-memory storage is shared across tests — reset it before each test."""
    limiter.reset()
    yield
    limiter.reset()


async def _post_stream(client, query):
    return await client.post("/api/research/stream", json={"query": query})


async def test_timeout_emits_clean_error_event():
    """1.3 DoD: a run exceeding RESEARCH_TIMEOUT_SEC yields an error NDJSON event, not a hang."""
    settings = Settings(groq_api_key="test-key", research_timeout_sec=0.2, _env_file=None)
    app = _build_app(StubSlowWorkflow(delay=5.0), settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await _post_stream(client, "valid research query here")
    lines = [line for line in response.text.splitlines() if line.strip()]
    error_events = [line for line in lines if '"error"' in line]
    assert error_events, f"expected an error event, got: {lines}"
    assert "timed out" in error_events[0]


async def test_rate_limit_returns_429_after_fifth_request():
    """1.7 DoD: more than RATE_LIMIT requests in a minute get a 429 on the expensive route."""
    settings = Settings(groq_api_key="test-key", rate_limit="5/minute", _env_file=None)
    app = _build_app(StubFastWorkflow(), settings)
    transport = httpx.ASGITransport(app=app)
    statuses = []
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for i in range(6):
            response = await _post_stream(client, f"rate limit probe {i}")
            statuses.append(response.status_code)
    assert statuses[:5] == [200] * 5, statuses
    assert statuses[5] == 429, statuses


async def test_normal_run_completes_within_timeout():
    settings = Settings(groq_api_key="test-key", research_timeout_sec=5.0, _env_file=None)
    app = _build_app(StubFastWorkflow(), settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await _post_stream(client, "valid research query here")
    lines = [line for line in response.text.splitlines() if line.strip()]
    assert any('"final_report"' in line for line in lines), lines


class StubCriticWorkflow:
    """Yields plan → critic (with breakdown) → final_report like the real route."""

    async def astream(self, state, stream_mode=None):
        yield {"sub_questions": [{"question": "what is RAG today?"}]}
        yield {"sub_questions": [{"question": "what is RAG today?"}],
               "iteration": 1,
               "critique": {"is_sufficient": True, "reason": "ok"},
               "confidence_breakdown": {"overall": 0.8, "signals": {"source_quality": 0.9}}}
        yield {"final_report": "# Final Answer\nok", "confidence": 0.8}


async def test_critic_event_carries_confidence_breakdown(tmp_path):
    from app.db.sqlite import init_db

    db_path = str(tmp_path / "bd.db")
    await init_db(db_path)
    settings = Settings(groq_api_key="test-key", database_url=db_path, _env_file=None)
    app = _build_app(StubCriticWorkflow(), settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/research/stream", json={"query": "valid research query here"})
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    critics = [e for e in events if e.get("type") == "critic"]
    assert len(critics) == 1
    assert critics[0]["breakdown"] == {"overall": 0.8, "signals": {"source_quality": 0.9}}

    import aiosqlite

    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT breakdown FROM critic_reviews")
        rows = await cur.fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0][0]) == {"overall": 0.8, "signals": {"source_quality": 0.9}}


class StubVerifyWorkflow:
    """Pre-verifier snapshot (unverified facts) then post-verifier snapshot
    (same length, annotated in place) — the re-emission case."""

    async def astream(self, state, stream_mode=None):
        yield {"facts": [
            {"claim": "RAG combines search with generation", "source": "https://a.com", "confidence": 0.8},
            {"claim": "Dense indexes serve the retriever", "source": "https://b.com", "confidence": 0.7},
        ], "iteration": 0}
        yield {"facts": [
            {"claim": "RAG combines search with generation", "source": "https://a.com", "confidence": 0.8,
             "verified": True, "verification_score": 0.9},
            {"claim": "Dense indexes serve the retriever", "source": "https://b.com", "confidence": 0.7,
             "verified": False, "verification_score": 0.2},
        ], "iteration": 0}
        yield {"final_report": "# Final Answer\nok", "confidence": 0.7}


async def test_verified_facts_reemit_when_flags_change(tmp_path):
    from app.db.sqlite import init_db

    db_path = str(tmp_path / "reemit.db")
    await init_db(db_path)
    settings = Settings(groq_api_key="test-key", database_url=db_path, _env_file=None)
    app = _build_app(StubVerifyWorkflow(), settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/research/stream", json={"query": "valid research query here"})
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    findings = [e for e in events if e.get("type") == "findings"]
    assert len(findings) == 2, findings
    assert findings[1].get("verified_update") is True
    assert findings[1]["items"][0]["verified"] is True
    assert findings[1]["items"][1]["verified"] is False


async def test_eval_batches_endpoint_returns_summaries(tmp_path):
    from app.db.sqlite import init_db, save_evaluation_run

    db_path = str(tmp_path / "evalapi.db")
    await init_db(db_path)
    await save_evaluation_run(db_path, {
        "eval_batch": "b1", "query_id": "q1", "query": "What is RAG?", "mode": "quick",
        "run_id": "r1", "status": "completed", "confidence": 0.8, "claims": 6,
        "verified": 4, "sources": 8, "contradictions": 0,
        "recommended_option": None, "cost": 0.001, "passed": True, "degraded": [],
    })
    settings = Settings(groq_api_key="test-key", database_url=db_path, _env_file=None)
    app = _build_app(StubFastWorkflow(), settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/eval/batches?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert len(data["batches"]) == 1
    batch = data["batches"][0]
    assert batch["batch"] == "b1"
    assert batch["summary"]["queries"] == 1
    assert batch["summary"]["pass_rate"] == 1.0
    assert batch["rows"][0]["query_id"] == "q1"
    assert batch["rows"][0]["mode"] == "quick"


async def test_eval_batches_rejects_bad_limit(tmp_path):
    from app.db.sqlite import init_db

    db_path = str(tmp_path / "evalapi2.db")
    await init_db(db_path)
    settings = Settings(groq_api_key="test-key", database_url=db_path, _env_file=None)
    app = _build_app(StubFastWorkflow(), settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/eval/batches?limit=abc")
    assert response.status_code == 422
