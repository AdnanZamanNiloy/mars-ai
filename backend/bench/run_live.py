#!/usr/bin/env python3
"""MARS live benchmark: real providers, real searches, real latency.

The offline suite (run_offline.py) measures the deterministic intelligence
components. This script measures what only a live run can: planner quality,
retrieval reality, synthesizer wording, end-to-end latency with real
providers, and the accuracy/citation metrics on fresh web data.

Cost profile: one research run per query (free-tier providers by default).
Typical spend with Groq free tier: 20-60k tokens per query. Point
MARS_LIVE_QUERIES at a smaller list to spend less.

Usage (from backend/):

    # Uses .env providers; writes bench/results/live_results.json
    python bench/run_live.py

    # Custom queries
    MARS_LIVE_QUERIES="q1|q2|q3" python bench/run_live.py

Requires: at least one configured LLM provider (GROQ_API_KEY or the
CUSTOM_LLM_* trio in backend/.env).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_QUERIES = [
    "What were the largest renewable energy capacity additions by country in the most recent year?",
    "What is the current evidence on retrieval augmented generation reducing hallucination rates?",
    "How do leading small language models compare on CPU inference benchmarks?",
]


def _load_queries() -> List[str]:
    raw = os.environ.get("MARS_LIVE_QUERIES", "")
    if raw.strip():
        return [q.strip() for q in raw.split("|") if q.strip()]
    return DEFAULT_QUERIES


async def run_one(query: str, settings: Any) -> Dict[str, Any]:
    """One full live research run through the production graph."""
    from app.agents.search import SearchClient
    from app.core.llm import LLMClient
    from app.core.usage import clear_run_usage, start_run_usage
    from app.graph.workflow import build_initial_state, create_workflow

    llm = LLMClient(settings)
    search_client = SearchClient(settings)
    workflow = create_workflow(llm, search_client)
    state = build_initial_state(query, settings.max_iterations, mode="standard")

    usage = start_run_usage(f"live-{int(time.time())}", settings, mode="standard")
    t0 = time.perf_counter()
    final: Dict[str, Any] = dict(state)
    error = None
    try:
        async for snapshot in workflow.astream(state, stream_mode="values"):
            final = {**final, **{k: v for k, v in snapshot.items() if v}}
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        clear_run_usage()
    elapsed = time.perf_counter() - t0

    facts = final.get("facts") or []
    results = final.get("search_results") or []
    domains = {str(r.get("url", "")).split("/")[2] for r in results if str(r.get("url", "")).startswith("http")}
    support = final.get("answer_support") or {}
    budget = usage.snapshot()
    return {
        "query": query,
        "error": error,
        "wall_sec": round(elapsed, 2),
        "iterations": int(final.get("iteration", 0)),
        "sub_questions": len(final.get("sub_questions") or []),
        "sources": len(results),
        "distinct_domains": len(domains),
        "facts": len(facts),
        "verified_facts": sum(1 for f in facts if f.get("verified")),
        "verification_rate": round(
            sum(1 for f in facts if f.get("verified")) / max(1, len(facts)), 4
        ),
        "contradictions": len(final.get("contradictions") or []),
        "confidence": final.get("confidence"),
        "answer_support_rate": support.get("rate"),
        "answer_support_sentences": support.get("cited"),
        "citation_health": (final.get("citation_health") or {}).get("summary", {}),
        "report_chars": len(str(final.get("final_report", ""))),
        "waves": len(final.get("wave_report") or []),
        "budget": {
            "llm_calls": budget.get("llm_calls"),
            "tokens": budget.get("spent_tokens"),
            "usd": budget.get("spent_usd"),
            "search_calls": budget.get("search_calls"),
            "cache_hit_rate": budget.get("cache_hit_rate"),
        },
    }


async def _main_async() -> int:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    from app.core.config import Settings

    try:
        settings = Settings(_env_file=str(Path(__file__).resolve().parents[1] / ".env"))
    except Exception as exc:
        print(f"No LLM provider configured ({exc}). Set GROQ_API_KEY or CUSTOM_LLM_* in backend/.env.")
        return 2

    queries = _load_queries()
    print("=" * 62)
    print(f"MARS live benchmark — {len(queries)} query(ies), real providers")
    print("=" * 62)

    runs: List[Dict[str, Any]] = []
    for query in queries:
        print(f"\n>> {query[:80]}")
        result = await run_one(query, settings)
        runs.append(result)
        if result["error"]:
            print(f"   ERROR: {result['error']}")
        else:
            print(
                f"   {result['wall_sec']}s | iters={result['iterations']} | "
                f"sources={result['sources']} ({result['distinct_domains']} domains) | "
                f"facts={result['facts']} verified={result['verified_facts']} | "
                f"support={result['answer_support_rate']} | conf={result['confidence']}"
            )

    ok_runs = [r for r in runs if not r["error"]]
    report: Dict[str, Any] = {
        "suite": "mars-live-benchmarks",
        "version": "2.0",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "summary": {
            "completed": len(ok_runs),
            "failed": len(runs) - len(ok_runs),
            "mean_wall_sec": round(
                sum(r["wall_sec"] for r in ok_runs) / max(1, len(ok_runs)), 2
            ) if ok_runs else None,
            "mean_verification_rate": round(
                sum(r["verification_rate"] for r in ok_runs) / max(1, len(ok_runs)), 4
            ) if ok_runs else None,
            "mean_support_rate": round(
                sum(r["answer_support_rate"] or 0 for r in ok_runs) / max(1, len(ok_runs)), 4
            ) if ok_runs else None,
            "mean_confidence": round(
                sum(r["confidence"] or 0 for r in ok_runs) / max(1, len(ok_runs)), 4
            ) if ok_runs else None,
            "total_tokens": sum((r["budget"] or {}).get("tokens") or 0 for r in ok_runs),
            "total_usd": round(sum((r["budget"] or {}).get("usd") or 0 for r in ok_runs), 6),
        },
    }

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "live_results.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nlive results written: {out_path}")
    print(f"summary: {json.dumps(report['summary'])}")
    return 0


def main() -> int:
    return asyncio.run(_main_async())


if __name__ == "__main__":
    raise SystemExit(main())
