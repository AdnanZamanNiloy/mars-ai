#!/usr/bin/env python3
"""Evaluation Lab (4.1): run the fixed query set against the real pipeline.

On demand only — there is no scheduler on this host. Each query streams a
live research run from the serving backend, scores it against its
expectations from tests/eval_queries.json, and persists one row per query
to the evaluation_runs table for trend comparison across batches.

Usage:
    .venv/bin/python scripts/run_eval.py [--limit 2] [--query-id eval-01]
    .venv/bin/python scripts/run_eval.py --server http://127.0.0.1:8000 --db research.db

Exit code is 0 when every evaluated query passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.eval import (  # noqa: E402
    JUDGE_MODEL_DEFAULT,
    _extract_json_object,
    format_trend,
    judge_prompt,
    parse_judge_scores,
    score_query,
    summarize_batch,
)
from app.db.sqlite import (  # noqa: E402
    eval_batch_rows,
    init_db,
    latest_eval_batches,
    save_evaluation_run,
)

DEFAULT_QUERIES = Path(__file__).resolve().parent.parent / "tests" / "eval_queries.json"


def count_section_lines(markdown: str, heading: str) -> int:
    """Non-empty lines under one markdown heading (up to the next heading)."""
    if not markdown or heading not in markdown:
        return 0
    body = markdown.split(heading, 1)[1]
    lines = []
    for line in body.splitlines():
        if line.startswith("#"):
            break
        if line.strip():
            lines.append(line)
    return len(lines)


async def fetch_judge_score(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    query: str,
    report_markdown: str,
) -> float | None:
    """One rater call to Groq. Returns the overall 1-5 score or None —
    judging must never fail an eval batch."""
    if not api_key or not report_markdown.strip():
        return None
    try:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": judge_prompt(query, report_markdown)}],
                "temperature": 0,
                "max_tokens": 200,
            },
            timeout=60.0,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        scores = parse_judge_scores(_extract_json_object(text))
        return scores["overall"] if scores else None
    except Exception:
        return None


async def run_one(client: httpx.AsyncClient, server: str, query: dict, timeout: float) -> dict:
    """Run one live evaluation; return observed metrics (never raises).

    The stream is used for run identity/completion only. All quality
    metrics come from the persisted trace afterwards: findings events are
    length-based, so facts annotated in place by the verifier (same list
    length) never re-emit — stream-observed `verified` would always read 0
    on single-pass runs. The trace is the persisted truth.
    """
    metrics: dict = {
        "status": "failed",
        "run_id": None,
        "confidence": None,
        "claims": 0,
        "verified": 0,
        "sources": 0,
        "decisions": 0,
        "contradictions": 0,
        "recommended_option": None,
        "cost": None,
        "error": None,
        "degraded": [],
        "report": "",
        "judge_score": None,
    }
    run_id: str | None = None
    try:
        async with client.stream(
            "POST",
            f"{server}/api/research/stream",
            json={"query": query["query"], "mode": query.get("mode", "standard")},
            timeout=timeout,
        ) as response:
            if response.status_code != 200:
                metrics["error"] = f"HTTP {response.status_code}"
                return metrics
            async for raw in response.aiter_lines():
                line = raw.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = evt.get("type")
                if t == "progress" and evt.get("request_id"):
                    run_id = evt["request_id"]
                    metrics["run_id"] = run_id
                elif t == "budget" and isinstance(evt.get("estimated_cost"), (int, float)):
                    metrics["cost"] = float(evt["estimated_cost"])
                elif t == "error":
                    metrics["status"] = "failed"
                    metrics["error"] = str(evt.get("message", "stream error"))[:200]
    except Exception as exc:  # transport/timeout — record, don't crash the batch
        metrics["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return metrics

    if run_id is None:
        metrics["error"] = metrics["error"] or "no run_id streamed"
        return metrics

    # Persisted truth for everything quality-related.
    try:
        resp = await client.get(f"{server}/api/research/{run_id}/trace", timeout=30.0)
        if resp.status_code != 200:
            metrics["error"] = (metrics["error"] or "") + f" trace HTTP {resp.status_code}"
            return metrics
        trace = resp.json()
        claims = trace.get("claims") or []
        decisions = trace.get("decisions") or []
        report_md = (trace.get("final_report") or {}).get("report_markdown") or ""
        metrics["report"] = report_md
        metrics["status"] = trace.get("status") or metrics["status"]
        metrics["claims"] = len(claims)
        metrics["verified"] = sum(1 for c in claims if c.get("verified"))
        metrics["sources"] = len(trace.get("sources") or [])
        metrics["decisions"] = len(decisions)
        metrics["contradictions"] = count_section_lines(report_md, "# Contradictions")
        # Persisted fallback events — stream-observed degraded would miss
        # fallbacks from retries the stream never surfaced cleanly.
        metrics["degraded"] = sorted({
            str(e.get("node", "")) for e in (trace.get("events") or [])
            if e.get("event_type") == "fallback" and e.get("node")
        })
        rec = next((o for o in decisions if o.get("is_recommended")), None)
        metrics["recommended_option"] = rec.get("option_label") if rec else None
        conf = (trace.get("final_report") or {}).get("confidence")
        if isinstance(conf, (int, float)):
            metrics["confidence"] = float(conf)
    except Exception as exc:
        metrics["error"] = (metrics["error"] or "") + f" trace failed: {type(exc).__name__}"[:120]
    return metrics


async def main() -> int:
    parser = argparse.ArgumentParser(description="MARS Evaluation Lab")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--db", default=None, help="SQLite path (default: configured database_url)")
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N queries")
    parser.add_argument("--query-id", action="append", default=[], help="evaluate only these ids (repeatable)")
    parser.add_argument("--batch", default=None, help="batch label (default: UTC timestamp)")
    parser.add_argument("--judge", action="store_true",
                        help="rate each report with an LLM judge (extra model calls)")
    parser.add_argument("--judge-model", default=JUDGE_MODEL_DEFAULT)
    args = parser.parse_args()

    with open(args.queries, encoding="utf-8") as fh:
        queries = json.load(fh)
    if not isinstance(queries, list) or not all(
        isinstance(q, dict) and isinstance(q.get("id"), str) and isinstance(q.get("query"), str)
        and q.get("mode") in ("quick", "standard", "deep")
        for q in queries
    ):
        print("queries file must be a list of {id, query, mode} objects", file=sys.stderr)
        return 2
    if args.query_id:
        wanted = set(args.query_id)
        queries = [q for q in queries if q.get("id") in wanted]
        missing = wanted - {q.get("id") for q in queries}
        if missing:
            print(f"unknown query ids: {sorted(missing)}", file=sys.stderr)
            return 2
    if args.limit > 0:
        queries = queries[: args.limit]
    if not queries:
        print("no queries selected", file=sys.stderr)
        return 2

    try:
        db_path = args.db or get_settings().database_url
    except Exception as exc:
        print(f"cannot load settings (missing API key in .env?): {exc}", file=sys.stderr)
        return 2
    await init_db(db_path)
    batch = args.batch or datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"eval batch {batch}: {len(queries)} queries vs {args.server}")

    settings = get_settings()
    stream_timeout = float(settings.research_timeout_sec) + 60.0
    failures = 0
    async with httpx.AsyncClient() as client:
        for i, query in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] {query['id']}: {query['query'][:60]}…", flush=True)
            metrics = await run_one(client, args.server.rstrip("/"), query, stream_timeout)
            if args.judge and metrics["status"] == "completed":
                metrics["judge_score"] = await fetch_judge_score(
                    client, settings.groq_api_key, args.judge_model,
                    query["query"], metrics["report"],
                )
            checks = score_query(query, metrics)
            status = "PASS" if checks["passed"] else "FAIL"
            if not checks["passed"]:
                failures += 1
            print(
                f"  -> {status} status={metrics['status']} claims={metrics['claims']} "
                f"verified={metrics['verified']} sources={metrics['sources']} "
                f"decisions={metrics['decisions']} contradictions={metrics['contradictions']} "
                f"conf={metrics['confidence']} cost={metrics['cost']}"
                + (f" error={metrics['error']}" if metrics.get("error") else "")
                + (f" DEGRADED={','.join(metrics['degraded'])}" if metrics.get("degraded") else "")
                + (f" judge={metrics['judge_score']}" if metrics.get("judge_score") is not None else "")
            )
            await save_evaluation_run(db_path, {
                "eval_batch": batch,
                "query_id": query.get("id", ""),
                "query": query.get("query", ""),
                "mode": query.get("mode", "standard"),
                "run_id": metrics["run_id"],
                "status": metrics["status"],
                "confidence": metrics["confidence"],
                "claims": metrics["claims"],
                "verified": metrics["verified"],
                "sources": metrics["sources"],
                "contradictions": metrics["contradictions"],
                "recommended_option": metrics["recommended_option"],
                "cost": metrics["cost"],
                "passed": checks["passed"],
                "degraded": metrics.get("degraded") or [],
                "judge_score": metrics.get("judge_score"),
            })

    rows = await eval_batch_rows(db_path, batch)
    clean = [r for r in rows if not r.get("degraded")]
    skipped = len(rows) - len(clean)
    if not clean:
        print(f"\nbatch {batch}: all {len(rows)} rows degraded — no clean trend available.")
        if skipped:
            print(f"degraded agents seen: {sorted({a for r in rows for a in r.get('degraded', [])})}")
        return 1 if failures else 0
    current = summarize_batch(clean)
    batches = await latest_eval_batches(db_path, limit=2)
    previous = None
    if len(batches) == 2:
        prev_rows = await eval_batch_rows(db_path, batches[1])
        previous = summarize_batch([r for r in prev_rows if not r.get("degraded")])
    print(f"\nbatch {batch} summary (trend over {len(clean)} clean rows"
          + (f", {skipped} degraded excluded" if skipped else "") + "):"
          f"\n{format_trend(current, previous)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
