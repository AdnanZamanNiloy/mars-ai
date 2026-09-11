from datetime import datetime, timezone
import json
import re

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
    max_iterations INTEGER NOT NULL DEFAULT 3,
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
    agent TEXT NOT NULL DEFAULT '',
    challenged INTEGER NOT NULL DEFAULT 0,
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
    breakdown TEXT NOT NULL DEFAULT '{}',
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

CREATE TABLE IF NOT EXISTS contradictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES research_runs(id),
    claim_a TEXT NOT NULL,
    source_a TEXT NOT NULL,
    claim_b TEXT NOT NULL,
    source_b TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verification_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES research_runs(id),
    claim TEXT NOT NULL,
    verified INTEGER NOT NULL,
    score REAL,
    reason TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES research_runs(id),
    marker INTEGER NOT NULL,
    domain TEXT NOT NULL,
    url TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_providers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    base_url TEXT NOT NULL,
    api_key_enc TEXT NOT NULL,
    key_hint TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
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
        review_cols = await db.execute("PRAGMA table_info(critic_reviews)")
        if "breakdown" not in {r[1] for r in await review_cols.fetchall()}:
            await db.execute("ALTER TABLE critic_reviews ADD COLUMN breakdown TEXT NOT NULL DEFAULT '{}'")
        run_cols = await db.execute("PRAGMA table_info(research_runs)")
        if "max_iterations" not in {r[1] for r in await run_cols.fetchall()}:
            await db.execute("ALTER TABLE research_runs ADD COLUMN max_iterations INTEGER NOT NULL DEFAULT 3")
        claim_cols = await db.execute("PRAGMA table_info(claims)")
        claim_names = {r[1] for r in await claim_cols.fetchall()}
        if "agent" not in claim_names:
            await db.execute("ALTER TABLE claims ADD COLUMN agent TEXT NOT NULL DEFAULT ''")
        if "challenged" not in claim_names:
            await db.execute("ALTER TABLE claims ADD COLUMN challenged INTEGER NOT NULL DEFAULT 0")
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


async def start_research_run(database_path: str, run_id: str, query: str, complexity: str, agent_count: int,
                           max_iterations: int = 3) -> None:
    async with aiosqlite.connect(database_path) as db:
        await db.execute(
            "INSERT OR IGNORE INTO research_runs (id, query, complexity, agent_count, status, max_iterations, created_at) "
            "VALUES (?, ?, ?, ?, 'running', ?, ?)",
            (run_id, query, complexity, agent_count, int(max_iterations), _now()),
        )
        await db.commit()


async def complete_research_run(
    database_path: str,
    run_id: str,
    status: str,
    confidence: float,
    estimated_cost: float | None,
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
            str(f.get("source") or f.get("source_url") or "").strip(),
            float(f.get("confidence", 0.0) or 0.0),
            1 if f.get("verified") else 0,
            str(f.get("agent", "") or ""),
            _now(),
        )
        for f in facts
        if str(f.get("claim", "")).strip() and str(f.get("source") or f.get("source_url") or "").strip()
    ]
    if not rows:
        return
    async with aiosqlite.connect(database_path) as db:
        await db.executemany(
            "INSERT INTO claims (run_id, claim, source_url, confidence, verified, agent, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        await db.commit()


async def mark_challenged_claims(database_path: str, run_id: str, contradictions: list) -> int:
    """Flag claims that appear on either side of a detected contradiction.

    Called once per run at finalization, when contradictions are known —
    claims are saved incrementally, contradictions only at the end.
    Returns the number of rows flagged. Matching is normalized-text
    equality against both contradiction sides.
    """
    sides = set()
    for c in contradictions or []:
        if not isinstance(c, dict):
            continue
        for key in ("claim_a", "claim_b"):
            text = re.sub(r"\s+", " ", str(c.get(key, "") or "").strip().lower())
            if text:
                sides.add(text)
    if not sides:
        return 0
    flagged = 0
    async with aiosqlite.connect(database_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT id, claim FROM claims WHERE run_id = ?", (run_id,))
        ids = [
            r["id"] for r in await cur.fetchall()
            if re.sub(r"\s+", " ", str(r["claim"] or "").strip().lower()) in sides
        ]
        for claim_id in ids:
            await db.execute("UPDATE claims SET challenged = 1 WHERE id = ?", (claim_id,))
            flagged += 1
        await db.commit()
    return flagged


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


async def save_critic_review(database_path: str, run_id: str, iteration: int, critique: dict,
                           breakdown: dict | None = None) -> None:
    """Persist EVERY critic iteration (not just the final verdict) — Replay
    needs the actual back-and-forth, not only the outcome."""
    if not isinstance(critique, dict):
        return
    try:
        breakdown_json = json.dumps(breakdown or {})
    except (TypeError, ValueError):
        breakdown_json = "{}"
    async with aiosqlite.connect(database_path) as db:
        await db.execute(
            "INSERT INTO critic_reviews (run_id, iteration, is_sufficient, reason, confidence, breakdown, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                int(iteration),
                1 if critique.get("is_sufficient") else 0,
                str(critique.get("reason", "")),
                float(critique.get("confidence", 0.0) or 0.0),
                breakdown_json,
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


async def save_contradictions(database_path: str, run_id: str, contradictions: list) -> int:
    """Persist detected contradiction pairs for replay and diagnosis."""
    rows = [
        (
            run_id,
            str(c.get("claim_a", "")).strip(),
            str(c.get("source_a", "") or "").strip(),
            str(c.get("claim_b", "")).strip(),
            str(c.get("source_b", "") or "").strip(),
            _now(),
        )
        for c in contradictions or []
        if isinstance(c, dict) and str(c.get("claim_a", "")).strip() and str(c.get("claim_b", "")).strip()
    ]
    if not rows:
        return 0
    async with aiosqlite.connect(database_path) as db:
        await db.executemany(
            "INSERT INTO contradictions (run_id, claim_a, source_a, claim_b, source_b, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        await db.commit()
    return len(rows)


async def save_verification_results(database_path: str, run_id: str, facts: list) -> int:
    """Persist per-fact verification detail (the claims table keeps only
    the verified flag; scores and reasons live here)."""
    rows = [
        (
            run_id,
            str(f.get("claim", "")).strip(),
            1 if f.get("verified") else 0,
            float(f.get("verification_score", 0.0) or 0.0)
            if isinstance(f.get("verification_score"), (int, float)) else None,
            str(f.get("verification_reason", "") or ""),
            _now(),
        )
        for f in facts or []
        if isinstance(f, dict) and str(f.get("claim", "")).strip()
    ]
    if not rows:
        return 0
    async with aiosqlite.connect(database_path) as db:
        await db.executemany(
            "INSERT INTO verification_results (run_id, claim, verified, score, reason, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        await db.commit()
    return len(rows)


async def save_citations(database_path: str, run_id: str, report_markdown: str) -> int:
    """Parse the emitted Sources legend back into citation rows, so the
    [n] markers in the report resolve to URLs without reparsing markdown.
    Handles both the current `## Sources` heading and the legacy bare
    `Sources:` line."""
    import re as _re

    count = 0
    text = report_markdown or ""
    match = _re.search(r"\n+#{0,6}\s*Sources:?\s*\n", text)
    if match:
        legend = text[match.end():]
    else:
        _, _, legend = text.partition("\nSources:")
    async with aiosqlite.connect(database_path) as db:
        for line in legend.splitlines():
            match = _re.match(r"^\[(\d+)\]\s+(\S+)\s+—\s*(\S+)\s*$", line.strip())
            if not match:
                continue
            await db.execute(
                "INSERT INTO citations (run_id, marker, domain, url, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, int(match.group(1)), match.group(2), match.group(3) or "", _now()),
            )
            count += 1
        await db.commit()
    return count


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


async def load_state_for_resume(database_path: str, run_id: str) -> dict | None:
    """Durable checkpointing (Phase 3.3): rebuild a ResearchState from the
    persisted rows of a failed/timeout run so resume can skip planner/search.

    Returns None if the run doesn't exist or is not resumable.
    """
    async with aiosqlite.connect(database_path) as db:
        db.row_factory = aiosqlite.Row

        cur = await db.execute(
            "SELECT * FROM research_runs WHERE id = ? AND status IN ('failed', 'timeout')",
            (run_id,),
        )
        run_row = await cur.fetchone()
        if run_row is None:
            return None

        async def _all(query: str, params: tuple = ()) -> list:
            c = await db.execute(query, params)
            return [dict(r) for r in await c.fetchall()]

        tasks = await _all(
            "SELECT question, axis, search_type, priority FROM agent_tasks WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        sources = await _all(
            "SELECT url, reliability_score FROM sources WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        claims = await _all(
            "SELECT claim, source_url, confidence, verified, agent FROM claims WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        review_rows = await _all(
            "SELECT iteration, is_sufficient, reason, confidence FROM critic_reviews "
            "WHERE run_id = ? ORDER BY iteration DESC LIMIT 1",
            (run_id,),
        )
        report_rows = await _all(
            "SELECT report_markdown, confidence FROM final_reports WHERE run_id = ? LIMIT 1",
            (run_id,),
        )

    last_review = review_rows[0] if review_rows else None
    iteration = int(last_review["iteration"]) if last_review else 0

    sub_questions = [
        {
            "id": i + 1,
            "question": t["question"],
            "axis": t["axis"] or "general",
            "search_type": t["search_type"] or "encyclopedia",
            "priority": int(t["priority"] or 2),
            "depends_on": [],
            "coverage_goal": "",
            "domain": "general",
            "minimum_sources": 2,
            "stop_condition": "sufficient evidence for this axis",
        }
        for i, t in enumerate(tasks)
    ]

    search_results = [
        {"url": s["url"], "reliability_score": s["reliability_score"] or 0.0, "snippet": ""}
        for s in sources
    ]
    facts = [
        {
            "claim": c["claim"],
            "source": c["source_url"],
            "confidence": c["confidence"] or 0.0,
            "verified": bool(c["verified"]),
            "agent": c.get("agent", "") or "",
        }
        for c in claims
    ]

    state: dict = {
        "query": run_row["query"],
        "sub_questions": sub_questions,
        "search_results": search_results,
        "facts": facts,
        "critique": {},
        "critique_feedback": "",
        "iteration": iteration,
        "max_iterations": int(run_row["max_iterations"] or 3)
        if "max_iterations" in run_row.keys() else 3,
        "final_report": "",
        "synthesized_answer": "",
        "confidence": run_row["confidence"] or 0.0,
        "confidence_history": [run_row["confidence"] or 0.0] if run_row["confidence"] else [],
        "orchestration": {
            "complexity_level": run_row["complexity"] or "unknown",
            "target_agents": run_row["agent_count"] or 3,
            "max_parallel_agents": 3,
            "clamped": False,
            "deep_research": False,
            "notes": [],
        },
        "resumed_from": run_id,
    }
    if last_review:
        state["critique"] = {
            "is_sufficient": bool(last_review["is_sufficient"]),
            "reason": last_review["reason"] or "",
            "improved_queries": [],
            "confidence": last_review["confidence"] or 0.0,
        }
        state["critique_feedback"] = last_review["reason"] or ""
    if report_rows:
        state["final_report"] = report_rows[0]["report_markdown"]

    return state


async def mark_run_resumable_reset(database_path: str, run_id: str) -> None:
    """Flip a failed/timeout run back to 'running' when a resume starts."""
    async with aiosqlite.connect(database_path) as db:
        await db.execute(
            "UPDATE research_runs SET status = 'running', completed_at = NULL WHERE id = ?",
            (run_id,),
        )
        await db.commit()


async def get_run_trace(database_path: str, run_id: str) -> dict | None:
    """Research Replay (Phase 3.4): read-only reconstruction of a run.

    Joins agent_events (node timing, retries, fallbacks, failures) with
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
            "SELECT id, claim, source_url, confidence, verified, agent, challenged, created_at "
            "FROM claims WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        events = await _all(
            "SELECT id, node, event_type, payload, started_at, ended_at "
            "FROM agent_events WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        critic_reviews = await _all(
            "SELECT id, iteration, is_sufficient, reason, confidence, breakdown, created_at "
            "FROM critic_reviews WHERE run_id = ? ORDER BY iteration, id",
            (run_id,),
        )
        for review in critic_reviews:
            try:
                review["breakdown"] = json.loads(review.get("breakdown") or "{}")
            except (TypeError, ValueError):
                review["breakdown"] = {}
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
        contradictions = await _all(
            "SELECT id, claim_a, source_a, claim_b, source_b, created_at "
            "FROM contradictions WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        verification_results = await _all(
            "SELECT id, claim, verified, score, reason, created_at "
            "FROM verification_results WHERE run_id = ? ORDER BY id",
            (run_id,),
        )
        citations = await _all(
            "SELECT id, marker, domain, url, created_at "
            "FROM citations WHERE run_id = ? ORDER BY marker, id",
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
            "contradictions": contradictions,
            "verification_results": verification_results,
            "citations": citations,
            "final_report": dict(final_report_row) if final_report_row else None,
        }
