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
"""

SCHEMA_VERSION = 1


async def init_db(database_path: str) -> None:
    async with aiosqlite.connect(database_path) as db:
        # WAL allows concurrent readers alongside a writer; busy_timeout
        # stops spurious "database is locked" errors under contention.
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA busy_timeout=5000;")
        await db.execute(CREATE_TABLE_SQL)
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
