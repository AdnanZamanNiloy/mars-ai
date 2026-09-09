from datetime import datetime, timezone

import aiosqlite


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS research_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    report TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research_runs (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    complexity TEXT,
    agent_count INTEGER,
    status TEXT NOT NULL,
    estimated_cost REAL,
    confidence REAL,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS agent_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES research_runs(id),
    question TEXT NOT NULL,
    axis TEXT,
    search_type TEXT,
    priority INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES research_runs(id),
    url TEXT NOT NULL,
    reliability_score REAL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES research_runs(id),
    claim TEXT NOT NULL,
    source_url TEXT NOT NULL,
    confidence REAL,
    verified INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES research_runs(id),
    node TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT,
    started_at TEXT,
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES research_runs(id),
    source_id INTEGER REFERENCES sources(id),
    raw_snippet TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS critic_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES research_runs(id),
    iteration INTEGER NOT NULL,
    is_sufficient INTEGER,
    reason TEXT,
    confidence REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES research_runs(id),
    option_label TEXT NOT NULL,
    description TEXT NOT NULL,
    is_recommended INTEGER,
    rationale TEXT,
    risk_note TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS final_reports (
    run_id TEXT PRIMARY KEY REFERENCES research_runs(id),
    report_markdown TEXT NOT NULL,
    confidence REAL,
    generated_at TEXT NOT NULL
);
"""

SCHEMA_VERSION = 3


async def init_db(database_path: str) -> None:
    async with aiosqlite.connect(database_path) as db:
        # WAL allows concurrent readers alongside a writer; busy_timeout
        # stops spurious "database is locked" errors under contention.
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA busy_timeout=5000;")
        # executescript handles the multi-statement CREATE TABLE block.
        await db.executescript(CREATE_TABLE_SQL)
        await db.execute(
            "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);"
        )
        cursor = await db.execute("SELECT MAX(version) FROM schema_version")
        row = await cursor.fetchone()
        current = row[0] if row and row[0] is not None else 0
        if current < SCHEMA_VERSION:
            await db.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        await db.commit()


async def save_report(database_path: str, query: str, report: str, confidence: float) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(database_path) as db:
        await db.execute(
            "INSERT INTO research_reports (query, report, confidence, created_at) VALUES (?, ?, ?, ?)",
            (query, report, confidence, created_at),
        )
        await db.commit()


# ======================================================================
# Research Memory (Phase 2.7 / 2.10) — incremental run persistence
# ======================================================================

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def start_research_run(database_path: str, run_id: str, query: str, complexity: str, agent_count: int) -> None:
    async with aiosqlite.connect(database_path) as db:
        await db.execute(
            "INSERT OR IGNORE INTO research_runs (id, query, complexity, agent_count, status, created_at) "
            "VALUES (?, ?, ?, ?, 'running', ?)",
            (run_id, query, complexity, agent_count, _now()),
        )
        await db.commit()


async def complete_research_run(
    database_path: str,
    run_id: str,
    status: str,
    confidence: float,
    estimated_cost: float,
) -> None:
    async with aiosqlite.connect(database_path) as db:
        await db.execute(
            "UPDATE research_runs SET status = ?, confidence = ?, estimated_cost = ?, completed_at = ? WHERE id = ?",
            (status, confidence, estimated_cost, _now(), run_id),
        )
        await db.commit()


async def save_agent_tasks(database_path: str, run_id: str, sub_questions: list) -> None:
    rows = [
        (
            run_id,
            str(q.get("question", "")).strip(),
            str(q.get("axis", "")),
            str(q.get("search_type", "")),
            int(q.get("priority", 2) or 2),
            _now(),
        )
        for q in sub_questions
        if isinstance(q, dict) and str(q.get("question", "")).strip()
    ]
    if not rows:
        return
    async with aiosqlite.connect(database_path) as db:
        await db.executemany(
            "INSERT INTO agent_tasks (run_id, question, axis, search_type, priority, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        await db.commit()


async def save_sources(database_path: str, run_id: str, search_results: list) -> None:
    seen = set()
    rows = []
    for item in search_results or []:
        url = str(item.get("url", "")).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        rows.append((run_id, url, float(item.get("reliability_score", 0.0) or 0.0), _now()))
    if not rows:
        return
    async with aiosqlite.connect(database_path) as db:
        await db.executemany(
            "INSERT INTO sources (run_id, url, reliability_score, fetched_at) VALUES (?, ?, ?, ?)",
            rows,
        )
        await db.commit()


async def save_claims(database_path: str, run_id: str, facts: list) -> None:
    rows = [
        (
            run_id,
            str(f.get("claim", "")).strip(),
            str(f.get("source", "")).strip(),
            float(f.get("confidence", 0.0) or 0.0),
            1 if f.get("verified") else 0,
            _now(),
        )
        for f in facts
        if str(f.get("claim", "")).strip() and str(f.get("source", "")).strip()
    ]
    if not rows:
        return
    async with aiosqlite.connect(database_path) as db:
        await db.executemany(
            "INSERT INTO claims (run_id, claim, source_url, confidence, verified, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        await db.commit()


async def record_event(
    database_path: str,
    run_id: str,
    node: str,
    event_type: str,
    payload: str = "",
    started_at: str = "",
    ended_at: str = "",
) -> None:
    async with aiosqlite.connect(database_path) as db:
        await db.execute(
            "INSERT INTO agent_events (run_id, node, event_type, payload, started_at, ended_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, node, event_type, payload, started_at or _now(), ended_at or _now()),
        )
        await db.commit()


async def save_evidence(database_path: str, run_id: str, search_results: list) -> None:
    """Evidence = the raw material claims were derived from (distinct from
    the rewritten claims): one row per source with its raw snippet."""
    if not search_results:
        return
    async with aiosqlite.connect(database_path) as db:
        for item in search_results:
            url = str(item.get("url", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            if not url or not snippet:
                continue
            cur = await db.execute(
                "SELECT id FROM sources WHERE run_id = ? AND url = ? LIMIT 1",
                (run_id, url),
            )
            row = await cur.fetchone()
            source_id = row[0] if row else None
            await db.execute(
                "INSERT INTO evidence (run_id, source_id, raw_snippet, created_at) "
                "VALUES (?, ?, ?, ?)",
                (run_id, source_id, snippet, _now()),
            )
        await db.commit()


async def save_critic_review(database_path: str, run_id: str, iteration: int, critique: dict) -> None:
    """Persist EVERY critic iteration (not just the final verdict) — Replay
    needs the actual back-and-forth, not only the outcome."""
    if not isinstance(critique, dict):
        return
    async with aiosqlite.connect(database_path) as db:
        await db.execute(
            "INSERT INTO critic_reviews (run_id, iteration, is_sufficient, reason, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                int(iteration),
                1 if critique.get("is_sufficient") else 0,
                str(critique.get("reason", "")),
                float(critique.get("confidence", 0.0) or 0.0),
                _now(),
            ),
        )
        await db.commit()


async def save_decisions(database_path: str, run_id: str, options: list) -> None:
    """One row per strategic option from the Decision Layer (3.5)."""
    rows = [
        (
            run_id,
            str(o.get("option_label", "")).strip(),
            str(o.get("description", "")).strip(),
            1 if o.get("is_recommended") else 0,
            str(o.get("rationale", "")),
            str(o.get("risk_note", "")),
            _now(),
        )
        for o in options
        if isinstance(o, dict) and str(o.get("option_label", "")).strip() and str(o.get("description", "")).strip()
    ]
    if not rows:
        return
    async with aiosqlite.connect(database_path) as db:
        await db.executemany(
            "INSERT INTO decisions (run_id, option_label, description, is_recommended, rationale, risk_note, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        await db.commit()


async def save_final_report(database_path: str, run_id: str, report_markdown: str, confidence: float) -> None:
    """Canonical report row keyed by run_id (research_reports stays for
    backward compatibility)."""
    async with aiosqlite.connect(database_path) as db:
        await db.execute(
            "INSERT OR REPLACE INTO final_reports (run_id, report_markdown, confidence, generated_at) "
            "VALUES (?, ?, ?, ?)",
            (run_id, report_markdown, float(confidence), _now()),
        )
        await db.commit()


async def get_run_trace(database_path: str, run_id: str) -> dict | None:
    """Research Replay (Phase 3.4): read-only reconstruction of a run.

    Joins agent_events (node timing, retries, budget checks, failures) with
    agent_tasks / sources / claims so the trace answers "why did this report
    reach this confidence", not just "what was the answer".
    """
    async with aiosqlite.connect(database_path) as db:
        db.row_factory = aiosqlite.Row

        cur = await db.execute("SELECT * FROM research_runs WHERE id = ?", (run_id,))
        run_row = await cur.fetchone()
        if run_row is None:
            return None

        async def _all(query: str, params: tuple = ()) -> list:
            c = await db.execute(query, params)
            return [dict(r) for r in await c.fetchall()]

        tasks = await _all(
            "SELECT id, question, axis, search_type, priority, created_at "
            "FROM agent_tasks WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        sources = await _all(
            "SELECT id, url, reliability_score, fetched_at "
            "FROM sources WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        claims = await _all(
            "SELECT id, claim, source_url, confidence, verified, created_at "
            "FROM claims WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        events = await _all(
            "SELECT id, node, event_type, payload, started_at, ended_at "
            "FROM agent_events WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        critic_reviews = await _all(
            "SELECT id, iteration, is_sufficient, reason, confidence, created_at "
            "FROM critic_reviews WHERE run_id = ? ORDER BY iteration, id",
            (run_id,),
        )
        evidence = await _all(
            "SELECT id, source_id, raw_snippet, created_at "
            "FROM evidence WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        decisions = await _all(
            "SELECT id, option_label, description, is_recommended, rationale, risk_note, created_at "
            "FROM decisions WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        cur = await db.execute(
            "SELECT report_markdown, confidence, generated_at FROM final_reports WHERE run_id = ?",
            (run_id,),
        )
        final_report_row = await cur.fetchone()

        return {
            "run_id": run_id,
            "query": run_row["query"],
            "status": run_row["status"],
            "complexity": run_row["complexity"],
            "agent_count": run_row["agent_count"],
            "confidence": run_row["confidence"],
            "estimated_cost": run_row["estimated_cost"],
            "created_at": run_row["created_at"],
            "completed_at": run_row["completed_at"],
            "plan": tasks,
            "sources": sources,
            "claims": claims,
            "events": events,
            "critic_reviews": critic_reviews,
            "evidence": evidence,
            "decisions": decisions,
            "final_report": dict(final_report_row) if final_report_row else None,
        }
