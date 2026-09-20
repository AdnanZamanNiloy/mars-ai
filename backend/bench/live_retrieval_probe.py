#!/usr/bin/env python3
"""Bounded LIVE retrieval probe, VERSION-AGNOSTIC (works on baseline + after).

Runs the real SearchClient over a few queries and counts, at the httpx layer,
how many page fetches succeeded / 403'd / 429'd / timed out, plus how many
unique domains and primary-source URLs the run acquired. Uses a monkeypatch on
httpx.AsyncClient.get so it does not depend on any new telemetry API — the same
script runs identically on commit 5242617 (before) and on the hardened build
(after), which is what makes the before/after numbers comparable.

    python bench/live_retrieval_probe.py --out /tmp/before.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

QUERIES = [
    "what is a transformer?",
    "state of nuclear energy in 2026",
    "compare solar vs nuclear for grid reliability",
]


async def _run(n: int) -> dict:
    import httpx
    from dotenv import load_dotenv

    backend = Path(__file__).resolve().parents[1]
    load_dotenv(backend / ".env")

    from app.agents import search as search_mod
    from app.agents.search import SearchClient
    from app.agents.sources import classify_source
    from app.core.config import Settings

    # A warm search cache would serve results without a single fetch and make
    # the before/after comparison meaningless. Force every run to fetch.
    def _no_cache(_settings):
        raise RuntimeError("probe: search cache disabled")

    search_mod.get_cache = _no_cache

    stats = {"attempts": 0, "success": 0, "403": 0, "429": 0, "timeout": 0,
             "5xx": 0, "4xx_other": 0, "connection": 0}

    original_get = httpx.AsyncClient.get

    async def _counting_get(self, url, **kwargs):
        stats["attempts"] += 1
        try:
            response = await original_get(self, url, **kwargs)
        except httpx.TimeoutException:
            stats["timeout"] += 1
            raise
        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError):
            stats["connection"] += 1
            raise
        status = int(getattr(response, "status_code", 0) or 0)
        if 200 <= status < 300:
            stats["success"] += 1
        elif status == 403:
            stats["403"] += 1
        elif status == 429:
            stats["429"] += 1
        elif 500 <= status < 600:
            stats["5xx"] += 1
        else:
            stats["4xx_other"] += 1
        return response

    httpx.AsyncClient.get = _counting_get
    try:
        settings = Settings(_env_file=str(backend / ".env"))
        client = SearchClient(settings)
        t0 = time.perf_counter()
        results = await client.run_search(list(QUERIES[:n]))
        elapsed = time.perf_counter() - t0
    finally:
        httpx.AsyncClient.get = original_get

    urls = [str(r.get("url", "")) for r in results if isinstance(r, dict)]
    domains = {classify_source(u).domain for u in urls if classify_source(u).domain}
    authoritative = {
        classify_source(u).domain for u in urls if classify_source(u).authority >= 0.75
    }
    primary = [u for u in urls if classify_source(u).is_primary]
    snap = {
        **stats,
        "fetch_success_rate": round(stats["success"] / stats["attempts"], 4)
        if stats["attempts"] else 0.0,
        "results_returned": len(results),
        "results_with_content": sum(
            1 for r in results if isinstance(r, dict) and r.get("content")
        ),
        "unique_domains": len(domains),
        "unique_authoritative_domains": len(authoritative),
        "primary_source_urls": len(primary),
        "elapsed_sec": round(elapsed, 2),
    }
    return snap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    snapshot = asyncio.run(_run(max(1, args.queries)))
    if args.out:
        Path(args.out).write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(json.dumps(snapshot, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
