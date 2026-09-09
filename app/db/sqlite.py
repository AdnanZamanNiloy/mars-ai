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
"""

SCHEMA_VERSION = 2


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
