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

    async def astream(self, state, stream_mode=None, config=None):
        await asyncio.sleep(self.delay)
        yield {}


class StubFastWorkflow:
    async def astream(self, state, stream_mode=None, config=None):
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

    async def astream(self, state, stream_mode=None, config=None):
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

    async def astream(self, state, stream_mode=None, config=None):
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


class StubManyFactsWorkflow:
    """One snapshot carrying 6 facts — the old emitter sliced only 3 per
    snapshot while marking the whole batch consumed, so facts 4..N never
    streamed to the UI."""

    async def astream(self, state, stream_mode=None, config=None):
        yield {"facts": [
            {"claim": f"Claim number {i}", "source": f"https://s{i}.com", "confidence": 0.8}
            for i in range(6)
        ], "iteration": 1, "critique": {"is_sufficient": True, "reason": "ok"}}
        yield {"final_report": "# Final Answer\nok", "confidence": 0.8}


async def test_all_new_findings_emitted_not_just_three(tmp_path):
    from app.db.sqlite import init_db

    db_path = str(tmp_path / "findings.db")
    await init_db(db_path)
    settings = Settings(groq_api_key="test-key", database_url=db_path, _env_file=None)
    app = _build_app(StubManyFactsWorkflow(), settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/research/stream", json={"query": "valid research query here"})
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    regular = [e for e in events if e.get("type") == "findings" and not e.get("verified_update")]
    assert regular, events
    streamed = [item["claim"] for e in regular for item in e["items"]]
    assert sorted(streamed) == [f"Claim number {i}" for i in range(6)], streamed


async def test_preflight_fails_fast_when_no_provider_reachable(tmp_path, monkeypatch):
    """A run with zero reachable LLM providers would only degrade to
    extraction after minutes. The pre-flight probe must end it in seconds
    with a clear error event — and never reach the workflow."""
    import respx
    import httpx as _httpx

    from app.core.llm import LLMClient

    settings = Settings(groq_api_key="test-key", database_url=str(tmp_path / "pf.db"), _env_file=None)
    app = _build_app(StubFastWorkflow(), settings)
    app.state.llm = LLMClient(settings)

    async def _boom(*a, **k):
        raise AssertionError("workflow must not run when the pre-flight probe fails")

    app.state.workflow = type("NoRun", (), {"astream": staticmethod(_boom)})()

    with respx.mock(assert_all_called=False) as mock:
        mock.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=_httpx.Response(429, json={"error": "TPD exhausted"}))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await _post_stream(client, "valid research query here")
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    types = [e["type"] for e in events]
    assert "error" in types, events
    assert "final_report" not in types, events
    error_event = next(e for e in events if e["type"] == "error")
    assert "No LLM provider is reachable" in error_event["message"]
    assert "groq" in error_event["message"]


class StubRouteWorkflow:
    """Yields an intent then a route decision — the R2 surface."""

    async def astream(self, state, stream_mode=None, config=None):
        yield {"intent": {"query_type": "factual", "domain": "software",
                          "ambiguity": False, "origin": "llm"}}
        yield {"intent": {"query_type": "factual", "domain": "software",
                          "ambiguity": False, "origin": "llm"},
               "route": {"path": "research", "reason": "needs evidence",
                         "confidence": 0.9, "origin": "llm",
                         "signals": {"hard_blockers": ["freshness"]}}}
        yield {"final_report": "# Final Answer\nok", "confidence": 0.7}


async def test_route_event_emitted_once(tmp_path):
    """R2: the router decision is surfaced on the wire exactly once, with
    the path/reason/signals the UI needs."""
    from app.db.sqlite import init_db

    db_path = str(tmp_path / "route.db")
    await init_db(db_path)
    settings = Settings(groq_api_key="test-key", database_url=db_path, _env_file=None)
    app = _build_app(StubRouteWorkflow(), settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/research/stream", json={"query": "valid research query here"})
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    routes = [e for e in events if e.get("type") == "route"]
    assert len(routes) == 1, events
    assert routes[0]["path"] == "research"
    assert routes[0]["reason"] == "needs evidence"
    assert routes[0]["confidence"] == 0.9
    assert routes[0]["signals"] == {"hard_blockers": ["freshness"]}


async def test_preflight_passes_when_a_provider_responds(tmp_path):
    """One reachable provider is enough: the run proceeds normally."""
    import respx
    import httpx as _httpx

    from app.core.llm import LLMClient
    from app.db.sqlite import init_db

    db_path = str(tmp_path / "pf2.db")
    await init_db(db_path)
    settings = Settings(groq_api_key="test-key", database_url=db_path, _env_file=None)
    app = _build_app(StubFastWorkflow(), settings)
    app.state.llm = LLMClient(settings)

    with respx.mock(assert_all_called=False) as mock:
        mock.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=_httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await _post_stream(client, "valid research query here")
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert any(e["type"] == "final_report" for e in events), events


class StubExpansionWorkflow:
    """Pass 1 verifies two facts; pass 2 adds a third, annotated only at its
    own verifier snapshot. Exercises both expansion bugs: the new claim must
    stream a verified_update containing ONLY the new item (the old code
    re-emitted once per run, leaving pass-2 claims stuck at null), and must
    be persisted with its real verified flag (the old code saved it at the
    summarizer snapshot with verified=0 and never re-saved — the length
    never changes at the verifier snapshot)."""

    async def astream(self, state, stream_mode=None, config=None):
        f1 = {"claim": "Claim one", "source": "https://a.com", "confidence": 0.8}
        f2 = {"claim": "Claim two", "source": "https://b.com", "confidence": 0.8}
        f1v = {**f1, "verified": True, "verification_score": 0.9}
        f2v = {**f2, "verified": False, "verification_score": 0.2}
        f3 = {"claim": "Claim three", "source": "https://c.com", "confidence": 0.8}
        f3v = {**f3, "verified": True, "verification_score": 0.85}
        yield {"facts": [f1, f2], "iteration": 0}
        yield {"facts": [f1v, f2v], "iteration": 0}
        yield {"facts": [f1v, f2v], "iteration": 1,
               "critique": {"is_sufficient": False, "reason": "gap", "improved_queries": ["more"]}}
        yield {"facts": [f1v, f2v, f3], "iteration": 1,
               "critique": {"is_sufficient": False, "reason": "gap", "improved_queries": ["more"]}}
        yield {"facts": [f1v, f2v, f3v], "iteration": 1,
               "critique": {"is_sufficient": False, "reason": "gap", "improved_queries": ["more"]}}
        yield {"facts": [f1v, f2v, f3v], "iteration": 2,
               "critique": {"is_sufficient": True, "reason": "ok"}}
        yield {"final_report": "# Final Answer\nok", "confidence": 0.8}


async def test_expansion_pass_findings_stream_and_persist_with_flags(tmp_path):
    import aiosqlite

    from app.db.sqlite import init_db

    db_path = str(tmp_path / "expansion.db")
    await init_db(db_path)
    settings = Settings(groq_api_key="test-key", database_url=db_path, _env_file=None)
    app = _build_app(StubExpansionWorkflow(), settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await _post_stream(client, "valid research query here")
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]

    updates = [e for e in events if e.get("type") == "findings" and e.get("verified_update")]
    assert len(updates) == 2, events
    assert len(updates[0]["items"]) == 2
    assert len(updates[1]["items"]) == 1, updates
    assert updates[1]["items"][0]["claim"] == "Claim three"
    assert updates[1]["items"][0]["verified"] is True

    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute("SELECT claim, verified FROM claims ORDER BY id")
        rows = await cur.fetchall()
    assert len(rows) == 3, rows
    by_claim = {claim: verified for claim, verified in rows}
    assert by_claim["Claim one"] == 1
    assert by_claim["Claim two"] == 0
    assert by_claim["Claim three"] == 1, "pass-2 claim must persist with its real verified flag"


async def test_client_disconnect_marks_run_cancelled(tmp_path):
    """When the stream is cancelled (client disconnect), the run must be
    marked 'timeout' via a detached task instead of sitting in 'running'
    forever — abandoned runs previously made the trace lie."""
    import asyncio

    import aiosqlite
    import httpx

    from app.db.sqlite import init_db

    class StubCancelledWorkflow:
        """Yields one snapshot, then the stream gets cancelled — the same
        shape a real client disconnect produces inside the generator."""

        async def astream(self, state, stream_mode=None, config=None):
            yield {"sub_questions": [{"question": "q1"}]}
            raise asyncio.CancelledError()

    db_path = str(tmp_path / "disconnect.db")
    await init_db(db_path)
    settings = Settings(groq_api_key="test-key", database_url=db_path, _env_file=None)
    app = _build_app(StubCancelledWorkflow(), settings)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        try:
            await client.post("/api/research/stream", json={"query": "valid research query here"})
        except Exception:
            pass  # the re-raised cancellation may abort the response transport

    # the detached completion task runs after the generator unwinds
    row = None
    for _ in range(20):
        await asyncio.sleep(0.05)
        async with aiosqlite.connect(db_path) as db:
            cur = await db.execute("SELECT status FROM research_runs")
            row = await cur.fetchone()
        if row and row[0] == "cancelled":
            break
    assert row and row[0] == "cancelled", f"run stuck in status {row}"
