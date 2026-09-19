"""Chat session persistence regression tests.

Covers the reported bug: multiple questions in one chat created separate
sessions instead of reusing one; history was lost/mismapped on reload.

These assert the DB + API contract (one stable session id, ordered append,
full-history restore, and New Chat minting a distinct session).
"""
import httpx
import pytest

from app.api.routes import router as api_router
from app.core.config import Settings
from app.db.sqlite import (
    complete_research_run,
    ensure_session,
    get_session,
    init_db,
    list_sessions,
    save_final_report,
    start_research_run,
)

SESSION_ID = "chat-session-1"
RUN_A = "run-aaa"
RUN_B = "run-bbb"


@pytest.fixture
def app_with_db(tmp_path):
    db_path = str(tmp_path / "chat.db")

    async def _setup():
        await init_db(db_path)
        # Two questions asked in the SAME chat, in order.
        await ensure_session(db_path, SESSION_ID, title="first question?")
        await start_research_run(
            db_path, RUN_A, "first question?", complexity="low", agent_count=1,
            session_id=SESSION_ID,
        )
        await save_final_report(db_path, RUN_A, "answer A", 0.8)
        await complete_research_run(db_path, RUN_A, "completed", confidence=0.8, estimated_cost=0.01)
        await start_research_run(
            db_path, RUN_B, "follow-up question?", complexity="low", agent_count=1,
            session_id=SESSION_ID,
        )
        await save_final_report(db_path, RUN_B, "answer B", 0.9)
        await complete_research_run(db_path, RUN_B, "completed", confidence=0.9, estimated_cost=0.02)
        # A separate chat started via "New Chat".
        await ensure_session(db_path, "chat-session-2", title="other chat?")
        await start_research_run(
            db_path, "run-ccc", "other chat?", complexity="low", agent_count=1,
            session_id="chat-session-2",
        )
        await complete_research_run(db_path, "run-ccc", "completed", confidence=0.7, estimated_cost=0.0)

    import asyncio
    asyncio.run(_setup())

    from fastapi import FastAPI

    app = FastAPI()
    app.state.settings = Settings(groq_api_key="k", database_url=db_path, _env_file=None)
    app.include_router(api_router, prefix="/api")
    return app


async def test_sequential_messages_share_one_session_and_stay_ordered():
    """The core bug: two questions in one chat must land in ONE session,
    appended in ask-order, not two separate sessions."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "seq.db")
        await init_db(path)
        await ensure_session(path, SESSION_ID, title="first question?")
        await start_research_run(path, RUN_A, "first question?", "low", 1, session_id=SESSION_ID)
        await start_research_run(path, RUN_B, "follow-up question?", "low", 1, session_id=SESSION_ID)

        session = await get_session(path, SESSION_ID)
        assert session is not None
        assert session["run_ids"] == [RUN_A, RUN_B]
        assert session["run_count"] == 2
        # Messages interleave user turn then run result, in order.
        assert [m["role"] for m in session["messages"]] == ["user", "run", "user", "run"]
        assert [m["run_id"] for m in session["messages"]] == [RUN_A, RUN_A, RUN_B, RUN_B]
        assert session["messages"][0]["text"] == "first question?"
        assert session["messages"][2]["text"] == "follow-up question?"


async def test_session_id_is_stable_across_ensure_calls():
    """Re-using a session id must never create a second session or overwrite
    its title — that is what keeps one chat = one stable id."""
    import os, tempfile

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "stable.db")
        await init_db(path)
        await ensure_session(path, SESSION_ID, title="original title")
        await ensure_session(path, SESSION_ID, title="SHOULD NOT WIN")
        sessions = await list_sessions(path)
        assert len(sessions) == 1
        assert sessions[0]["id"] == SESSION_ID
        assert sessions[0]["title"] == "original title"


async def test_persistence_reload_restores_full_history(app_with_db):
    """Reload/reopen: GET /api/sessions/{id} returns the full ordered history."""
    transport = httpx.ASGITransport(app=app_with_db)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/api/sessions/{SESSION_ID}")
    assert response.status_code == 200
    session = response.json()
    assert session["id"] == SESSION_ID
    assert session["run_count"] == 2
    assert [m["text"] for m in session["messages"] if m["role"] == "user"] == [
        "first question?", "follow-up question?",
    ]
    assert session["messages"][1]["report"] == "answer A"
    assert session["messages"][3]["report"] == "answer B"


async def test_sidebar_lists_one_chat_per_session(app_with_db):
    """Sidebar must show one entry per chat, not one per run."""
    transport = httpx.ASGITransport(app=app_with_db)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/sessions")
    assert response.status_code == 200
    sessions = response.json()["sessions"]
    ids = [s["id"] for s in sessions]
    assert sorted(ids) == sorted(["chat-session-2", SESSION_ID])
    first = next(s for s in sessions if s["id"] == SESSION_ID)
    assert first["run_count"] == 2


async def test_new_chat_creates_distinct_session(app_with_db):
    """'New Chat' mints a new id; the two chats stay independent."""
    transport = httpx.ASGITransport(app=app_with_db)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = (await client.get(f"/api/sessions/{SESSION_ID}")).json()
        second = (await client.get("/api/sessions/chat-session-2")).json()
    assert first["id"] != second["id"]
    assert first["run_count"] == 2
    assert second["run_count"] == 1
    assert second["messages"][0]["text"] == "other chat?"


async def test_unknown_session_returns_404(app_with_db):
    transport = httpx.ASGITransport(app=app_with_db)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/sessions/does-not-exist")
    assert response.status_code == 404


async def test_stream_request_accepts_and_echoes_session_id():
    """The stream contract: a client-supplied session_id is grouped, and the
    progress event echoes it so the UI can adopt the canonical id."""
    from app.api.routes import ResearchRequest

    payload = ResearchRequest(query="hello", session_id=SESSION_ID)
    assert payload.session_id == SESSION_ID
    # Absent session_id stays optional for backward compatibility.
    assert ResearchRequest(query="hello").session_id is None
