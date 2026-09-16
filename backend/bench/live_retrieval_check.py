#!/usr/bin/env python3
"""Bounded LIVE retrieval-only check (NOT the full benchmark).

Runs the real `SearchClient` on a handful of queries with NO LLM calls and
prints retrieval-health metrics: successful fetch rate, 403/429/timeout counts,
unique authoritative domains, primary-source acquisitions. Small, bounded and
safe to run before/after a retrieval change to compare.

Usage (from backend/):
    python bench/live_retrieval_check.py            # default 3 queries
    python bench/live_retrieval_check.py --queries 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_QUERIES = [
    "what is a transformer?",
    "state of nuclear energy in 2026",
    "compare solar vs nuclear for grid reliability",
    "retrieval augmented generation benchmarks",
    "global renewable capacity statistics",
]


async def _run(queries: int) -> dict:
    from dotenv import load_dotenv

    backend = Path(__file__).resolve().parents[1]
    load_dotenv(backend / ".env")

    from app.agents import search as search_mod
    from app.agents.search import SearchClient
    from app.core.config import Settings

    # A warm search cache would return results with zero fetches and zero
    # health counters, so disable it for this live check.
    def _no_cache(_settings):
        raise RuntimeError("live check: search cache disabled")

    search_mod.get_cache = _no_cache

    settings = Settings(_env_file=str(backend / ".env"))
    client = SearchClient(settings)

    t0 = time.perf_counter()
    results = await client.run_search([q for q in DEFAULT_QUERIES[:queries]])
    elapsed = time.perf_counter() - t0

    # Unique authoritative domains reached among the retrieved URLs.
    from app.agents.sources import classify_source

    urls = [str(r.get("url", "")) for r in results if isinstance(r, dict)]
    authoritative = {
        classify_source(u).domain
        for u in urls
        if classify_source(u).authority >= 0.75
    }

    snapshot = client.health_snapshot()
    snapshot["queries"] = queries
    snapshot["results_returned"] = len(results)
    snapshot["elapsed_sec"] = round(elapsed, 2)
    snapshot["unique_authoritative_domains_reached"] = len(authoritative)
    snapshot["results_with_content"] = sum(
        1 for r in results if isinstance(r, dict) and r.get("content")
    )
    # `run_search` returns dicts; health counters are populated by `_search`.
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--out", default=None, help="write the snapshot JSON here")
    args = parser.parse_args()
    snapshot = asyncio.run(_run(max(1, args.queries)))
    if args.out:
        Path(args.out).write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    print(json.dumps(snapshot, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
