"""Claim attribution: agent tags persist, contradictions flag challenged."""

import asyncio

from app.db.sqlite import (
    init_db,
    mark_challenged_claims,
    save_claims,
)


def _db_rows(db_path, table="claims"):
    import aiosqlite

    async def _q():
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(f"SELECT * FROM {table} ORDER BY id")
            return [dict(r) for r in await cur.fetchall()]

    return asyncio.run(_q())


def test_agent_tag_round_trip(tmp_path):
    db_path = str(tmp_path / "attr.db")
    asyncio.run(init_db(db_path))
    asyncio.run(save_claims(db_path, "r1", [
        {"claim": "Markets rise on data", "source": "https://x.com/1",
         "confidence": 0.8, "verified": True, "agent": "financial"},
        {"claim": "No agent here", "source": "https://x.com/2", "confidence": 0.5},
    ]))
    rows = _db_rows(db_path)
    assert rows[0]["agent"] == "financial"
    assert rows[1]["agent"] == ""


def test_mark_challenged_flags_contradiction_sides(tmp_path):
    db_path = str(tmp_path / "chal.db")
    asyncio.run(init_db(db_path))
    asyncio.run(save_claims(db_path, "r1", [
        {"claim": "Costs will keep falling fast", "source": "https://x.com/1", "confidence": 0.8},
        {"claim": "Unrelated stable fact here", "source": "https://y.com/2", "confidence": 0.7},
    ]))
    contradictions = [{"claim_a": "Costs will keep falling fast",
                       "claim_b": "Costs have plateaued since last year"}]
    flagged = asyncio.run(mark_challenged_claims(db_path, "r1", contradictions))
    assert flagged == 1
    rows = _db_rows(db_path)
    assert rows[0]["challenged"] == 1
    assert rows[1]["challenged"] == 0
    assert asyncio.run(mark_challenged_claims(db_path, "r1", [])) == 0


def test_migration_adds_claim_columns_to_old_table(tmp_path):
    import aiosqlite

    db_path = str(tmp_path / "old.db")

    async def _setup():
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "CREATE TABLE claims (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,"
                " claim TEXT NOT NULL, source_url TEXT NOT NULL, confidence REAL, verified INTEGER,"
                " created_at TEXT NOT NULL)"
            )
            await db.execute(
                "INSERT INTO claims (run_id, claim, source_url, created_at)"
                " VALUES ('r1', 'Old claim here', 'https://x.com', 'now')"
            )
            await db.commit()

    asyncio.run(_setup())
    asyncio.run(init_db(db_path))
    rows = _db_rows(db_path)
    assert rows[0]["agent"] == ""
    assert rows[0]["challenged"] == 0


def test_summarizer_attaches_specialist_role(tmp_path):
    from app.agents.summarizer import summarizer_agent
    from app.core.config import Settings

    class FakeLLM:
        settings = Settings(groq_api_key="k", database_url=str(tmp_path / "t.db"), _env_file=None)

        async def generate_json(self, *a, **k):
            return {"facts": [
                {"claim": "Capital expenditure shapes nuclear project economics",
                 "source": "https://imf.org/report", "confidence": 0.9},
            ]}

    facts = asyncio.run(summarizer_agent(
        FakeLLM(), "Nuclear economics?", [{"url": "https://imf.org/report", "snippet": "s", "content": "c"}],
        specialist_role="financial"))
    assert facts and facts[0]["agent"] == "financial"
