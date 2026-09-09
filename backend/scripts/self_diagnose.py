#!/usr/bin/env python3
"""Self-Diagnosis (4.2): read-only health report over the research memory.

Inspects research_runs / claims / critic_reviews / final_reports /
evaluation_runs for patterns: failing providers, verification yield,
critic efficiency, cost drift. Prints a human-readable report (or JSON).

The database is opened in SQLite read-only mode — this script cannot
write, no matter the bug. It is a manual report, not an autonomous loop.

Usage:
    .venv/bin/python scripts/self_diagnose.py [--db research.db] [--days 30] [--json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import aiosqlite

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.evidence_utils import extract_domain  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.diagnose import (  # noqa: E402
    attention_flags,
    contradiction_watch,
    critic_efficiency,
    run_health,
    verification_by_domain,
)
from app.core.eval import summarize_batch  # noqa: E402


async def _all(db: aiosqlite.Connection, query: str, params: tuple = ()) -> list:
    cur = await db.execute(query, params)
    return [dict(r) for r in await cur.fetchall()]


async def fetch(db_path: str, days: int) -> dict:
    # mode=ro: the OS + SQLite both refuse writes through this handle.
    uri = f"file:{db_path}?mode=ro"
    async with aiosqlite.connect(uri, uri=True) as db:
        db.row_factory = aiosqlite.Row
        since = f"-{int(days)} days" if days > 0 else "-100 years"
        runs = await _all(
            db,
            "SELECT id, query, status, confidence, estimated_cost, created_at "
            "FROM research_runs WHERE created_at >= datetime('now', ?) ORDER BY id",
            (since,),
        )
        run_ids = [r["id"] for r in runs]
        claims, reviews, reports = [], [], []
        if run_ids:
            placeholders = ",".join("?" for _ in run_ids)
            claims = await _all(
                db,
                f"SELECT run_id, claim, source_url, confidence, verified FROM claims "
                f"WHERE run_id IN ({placeholders})",
                tuple(run_ids),
            )
            reviews = await _all(
                db,
                f"SELECT run_id, iteration, is_sufficient, confidence FROM critic_reviews "
                f"WHERE run_id IN ({placeholders})",
                tuple(run_ids),
            )
            report_rows = await _all(
                db,
                f"SELECT report_markdown FROM final_reports WHERE run_id IN ({placeholders})",
                tuple(run_ids),
            )
            reports = [r["report_markdown"] for r in report_rows]
        eval_rows = await _all(
            db,
            "SELECT eval_batch, confidence, claims, verified, contradictions, cost, checks_passed "
            "FROM evaluation_runs ORDER BY id DESC LIMIT 200",
        )
    return {"runs": runs, "claims": claims, "reviews": reviews, "reports": reports, "eval_rows": eval_rows}


def render_text(data: dict, days: int) -> str:
    health = run_health(data["runs"])
    domains = verification_by_domain(data["claims"], extract_domain)
    critic = critic_efficiency(data["reviews"])
    contra = contradiction_watch(data["reports"])
    flags = attention_flags(health, domains, critic)

    lines = [f"MARS self-diagnosis (last {days} days, {health['total']} runs)", ""]
    lines.append("Run health:")
    for status, count in sorted(health["by_status"].items()):
        lines.append(f"  {status}: {count}")
    if health["avg_confidence"] is not None:
        lines.append(f"  avg confidence (completed): {health['avg_confidence']:.2f}")
    if health["avg_cost"] is not None:
        lines.append(f"  avg cost (completed): ${health['avg_cost']:.6f}")
    lines.append("")

    lines.append("Verification yield by source domain (top 8):")
    if domains:
        for row in domains[:8]:
            lines.append(
                f"  {row['domain']}: {row['verified']}/{row['claims']} verified "
                f"({row['verified_rate']:.0%}), avg confidence {row['avg_confidence']:.2f}"
            )
    else:
        lines.append("  no claims in window")
    lines.append("  note: claims carry no specialist tag, so per-specialist yield")
    lines.append("  cannot be measured from the current schema — domain yield is the proxy.")
    lines.append("")

    lines.append("Critic efficiency:")
    lines.append(
        f"  {critic['runs']} reviewed runs, {critic['avg_iterations']:.1f} avg passes, "
        f"{critic['expansion_rate']:.0%} expanded, "
        f"{critic['avg_confidence_gain']:+.2f} avg confidence gain"
    )
    lines.append("")
    lines.append(
        f"Contradictions surfaced in {contra['with_contradictions']}/{contra['reports']} "
        f"reports ({contra['rate']:.0%})"
    )
    lines.append("")

    batches: dict = {}
    for row in data["eval_rows"]:
        batches.setdefault(row["eval_batch"], []).append({
            "confidence": row["confidence"], "claims": row["claims"],
            "verified": row["verified"], "contradictions": row["contradictions"],
            "cost": row["cost"], "passed": bool(row["checks_passed"]),
        })
    if batches:
        newest = sorted(batches)[-1]
        summary = summarize_batch(batches[newest])
        lines.append(
            f"Latest eval batch {newest}: {summary['queries']} queries, "
            f"{summary['pass_rate']:.0%} pass, {summary['avg_confidence']:.2f} avg confidence"
        )
        lines.append("")

    lines.append("Attention:")
    if flags:
        for flag in flags:
            lines.append(f"  ! {flag}")
    else:
        lines.append("  nothing above action thresholds — system looks healthy.")
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description="MARS Self-Diagnosis (read-only)")
    parser.add_argument("--db", default=None)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    db_path = args.db or get_settings().database_url
    try:
        data = await fetch(db_path, args.days)
    except Exception as exc:
        print(f"self-diagnosis failed to read {db_path}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        health = run_health(data["runs"])
        print(json.dumps({
            "days": args.days,
            "health": health,
            "domains": verification_by_domain(data["claims"], extract_domain)[:20],
            "critic": critic_efficiency(data["reviews"]),
            "contradictions": contradiction_watch(data["reports"]),
            "flags": attention_flags(
                health,
                verification_by_domain(data["claims"], extract_domain),
                critic_efficiency(data["reviews"]),
            ),
        }, indent=2))
    else:
        print(render_text(data, args.days))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
