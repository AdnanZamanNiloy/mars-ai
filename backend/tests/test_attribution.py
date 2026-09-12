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


def test_summarizer_quote_grounding_end_to_end(tmp_path):
    """direct_quote must survive generate_json validation (FactModel field)
    and verify against the source text: located quotes boost, invented
    quotes penalize."""
    import asyncio

    from app.agents.summarizer import summarizer_agent
    from app.core.config import Settings
    from app.core.degradation import clear_fallbacks, reset_fallbacks
    from app.core.schemas import SummarizerFactsModel

    content = (
        "Solar capacity in Bangladesh doubled in 2025 as imports surged. "
        "Grid upgrades lagged behind the new installations."
    )

    class QuoteLLM:
        def __init__(self):
            self.settings = Settings(
                groq_api_key="k", database_url=str(tmp_path / "q.db"), _env_file=None
            )

        async def generate_json(self, system_prompt, user_prompt, response_model=None):
            payload = {"facts": [
                {"claim": "Solar capacity in Bangladesh doubled in 2025",
                 "source": "https://example.com/solar",
                 "confidence": 0.8,
                 "direct_quote": "Solar capacity in Bangladesh doubled in 2025"},
                {"claim": "Grid upgrades lagged behind the new installations",
                 "source": "https://example.com/solar",
                 "confidence": 0.8,
                 "direct_quote": "Ministers promised a tenfold budget increase soon"},
            ]}
            if response_model is not None:
                return response_model.model_validate(payload).model_dump()
            return payload

    results = [{"url": "https://example.com/solar", "snippet": "", "content": content,
                "sub_question": "How fast did solar grow in Bangladesh?"}]
    reset_fallbacks()
    try:
        facts = asyncio.run(summarizer_agent(
            QuoteLLM(), "Solar growth and grid upgrades in Bangladesh", results))
    finally:
        clear_fallbacks()

    by_claim = {f["claim"]: f for f in facts}
    located = by_claim["Solar capacity in Bangladesh doubled in 2025"]
    assert located["direct_quote"] != ""  # survived validation
    assert located["quote_verified"] is True
    invented = by_claim["Grid upgrades lagged behind the new installations"]
    assert invented["quote_verified"] is False
    assert invented["confidence"] < located["confidence"]
