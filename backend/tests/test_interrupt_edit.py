"""Interrupt-and-edit regression tests.

The user must be able to stop generation, edit the previous message and
resend, without partial answers being persisted or the chat history breaking.

Backend contract under test:
- a user-cancelled run is recorded as 'cancelled' (not 'timeout'/'completed')
- no final report row exists for a cancelled run
- the session still lists the chat with the cancelled run counted
- a cancelled run is NOT offered for resume (the user chose to stop)
"""
import asyncio
import os
import tempfile

import httpx
import pytest

from app.api.routes import router as api_router
from app.core.config import Settings
from app.db.sqlite import (
    complete_research_run,
    ensure_session,
    get_run_trace,
    get_session,
    init_db,
    load_state_for_resume,
    save_final_report,
    start_research_run,
)

SESSION_ID = "edit-session-1"


async def _seed_cancelled(db_path: str) -> None:
    await init_db(db_path)
    await ensure_session(db_path, SESSION_ID, title="first question?")
    # A completed earlier turn that survived the interrupt.
    await start_research_run(
        db_path, "run-done", "first question?", "low", 1, session_id=SESSION_ID,
    )
    await save_final_report(db_path, "run-done", "answer one", 0.8)
    await complete_research_run(db_path, "run-done", "completed", 0.8, 0.01)
    # The interrupted turn: status cancelled, and crucially NO final report.
    await start_research_run(
        db_path, "run-stopped", "second question?", "low", 1, session_id=SESSION_ID,
    )
    await complete_research_run(db_path, "run-stopped", "cancelled", 0.0, 0.0)


@pytest.fixture
def app_with_db(tmp_path):
    db_path = str(tmp_path / "interrupt.db")
    asyncio.run(_seed_cancelled(db_path))

    from fastapi import FastAPI

    app = FastAPI()
    app.state.settings = Settings(groq_api_key="k", database_url=db_path, _env_file=None)
    app.include_router(api_router, prefix="/api")
    return app


async def test_cancelled_run_has_no_partial_report():
    """The core guarantee: stopping mid-run must not persist a partial answer."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cancel.db")
        await init_db(path)
        await ensure_session(path, SESSION_ID, title="q")
        await start_research_run(path, "run-stopped", "q", "low", 1, session_id=SESSION_ID)
        await complete_research_run(path, "run-stopped", "cancelled", 0.0, 0.0)

        trace = await get_run_trace(path, "run-stopped")
        assert trace is not None
        assert trace["status"] == "cancelled"
        assert trace["final_report"] is None
        assert trace["claims"] == []


async def test_cancelled_run_is_not_resumable():
    """A user stop is a decision, not a failure to recover from."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "notresume.db")
        await init_db(path)
        await ensure_session(path, SESSION_ID, title="q")
        await start_research_run(path, "run-stopped", "q", "low", 1, session_id=SESSION_ID)
        await complete_research_run(path, "run-stopped", "cancelled", 0.0, 0.0)

        assert await load_state_for_resume(path, "run-stopped") is None


async def test_session_history_survives_interrupt_and_edit():
    """After a stop, the completed earlier turn stays and the cancelled run is
    counted but has no report — so editing + resending can append cleanly."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "history.db")
        await _seed_cancelled(path)

        session = await get_session(path, SESSION_ID)
        assert session["run_ids"] == ["run-done", "run-stopped"]
        assert session["run_count"] == 2
        reports = {m["run_id"]: m.get("report") for m in session["messages"] if m["role"] == "run"}
        assert reports["run-done"] == "answer one"
        # The stopped turn has no report to render as a partial answer.
        assert reports["run-stopped"] in ("", None)


async def test_edit_then_resend_appends_to_same_session():
    """Simulate edit-and-resend: truncate from the interrupted turn, then ask
    a new question under the SAME session id. History must stay one chat."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "resend.db")
        await _seed_cancelled(path)

        # The edited resend: a fresh run in the same session.
        await start_research_run(
            path, "run-resend", "second question edited?", "low", 1, session_id=SESSION_ID,
        )
        await save_final_report(path, "run-resend", "answer two", 0.9)
        await complete_research_run(path, "run-resend", "completed", 0.9, 0.02)

        session = await get_session(path, SESSION_ID)
        assert session["run_count"] == 3
        assert session["run_ids"][-1] == "run-resend"
        # Still exactly one chat.
        from app.db.sqlite import list_sessions
        sessions = await list_sessions(path)
        assert [s for s in sessions if s["id"] == SESSION_ID]
        assert len([s for s in sessions if s["id"] == SESSION_ID]) == 1


async def test_cancelled_status_surfaces_in_session_api(app_with_db):
    transport = httpx.ASGITransport(app=app_with_db)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session = (await client.get(f"/api/sessions/{SESSION_ID}")).json()
    statuses = [r["status"] for r in session["runs"]]
    assert "cancelled" in statuses
    assert "completed" in statuses
