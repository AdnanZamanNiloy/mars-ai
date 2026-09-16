#!/usr/bin/env python3
"""Deterministic performance/cost benchmark for the improvement task.

Runs the FULL production graph with the scripted mock LLM + mock search (zero
network, zero nondeterminism) and reports the cost metrics the optimization
targets:

* total LLM calls (and per-stage breakdown)
* retry calls (prompt-identical re-asks within a stage)
* estimated tokens (the run ledger's own accounting)
* search queries issued, unique queries, duplicate queries
* wall-clock (offline, deterministic-ish)

Run from backend/:

    python bench/bench_perf.py

The numbers are an OFFLINE proxy: the mock LLM never re-tries a provider the
way a live 429 would, so "retry calls" here counts prompt-identical calls to
the same stage (the redundant re-ask class), not HTTP retries. Live latency is
measured separately by `bench/run_live.py`.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings  # noqa: E402


def _stage(system_prompt: str, user_prompt: str) -> str:
    """Stable stage label from the SYSTEM prompt (mirrors the mock contract)."""
    s = (system_prompt or "").lower()
    if "final synthesis agent" in s:
        return "synthesizer"
    if "extract high-quality claims" in s:
        return "summarizer"
    if "critic" in s or "sufficien" in s:
        return "critic"
    if "delegation contract" in s or "planner" in s or "sub-question" in s:
        return "planner"
    if "intent" in s:
        return "intent"
    return "unknown"


async def _run(settings: Settings) -> dict:
    from app.core import llm_cache as _lc
    from app.core.usage import clear_run_usage, start_run_usage
    from app.graph.workflow import build_initial_state, create_workflow
    from bench.mock_pipeline import FakeLLM, FakeSearch

    _lc._force_disabled = True  # measure the pipeline, not a warm response cache

    llm = FakeLLM(settings, critic_pass_on_iteration=2)
    search = FakeSearch(settings)
    seen_prompts: Counter = Counter()
    retry_calls = 0

    orig = llm.generate_json

    async def patched(system_prompt, user_prompt, retries=3, response_model=None):
        nonlocal retry_calls
        key = (system_prompt, user_prompt)
        if seen_prompts[key]:
            retry_calls += 1
        seen_prompts[key] += 1
        return await orig(system_prompt, user_prompt, retries, response_model)

    llm.generate_json = patched
    workflow = create_workflow(llm, search_client=search)
    state = build_initial_state(
        "What is retrieval augmented generation, how widely is it adopted, "
        "what is the evidence for its effectiveness, and what are the criticisms?",
        max_iterations=3,
        mode="standard",
    )
    usage = start_run_usage("bench-perf", settings, mode="standard")
    t0 = time.perf_counter()
    final = dict(state)
    try:
        async for snapshot in workflow.astream(state, stream_mode="values"):
            final = {**final, **{k: v for k, v in snapshot.items() if v}}
    finally:
        budget = usage.snapshot()
        clear_run_usage()
    wall = time.perf_counter() - t0

    stages = Counter(c["stage"] for c in llm.calls)
    queries = list(search.calls)
    return {
        "wall_sec": round(wall, 3),
        "llm_calls": len(llm.calls),
        "llm_calls_by_stage": dict(stages),
        "retry_calls": retry_calls,
        "tokens_estimated": budget.get("spent_tokens"),
        "spent_usd": budget.get("spent_usd"),
        "search_queries_issued": len(queries),
        "search_queries_unique": len(set(queries)),
        "search_queries_duplicate": len(queries) - len(set(queries)),
        "facts_extracted": len(final.get("facts") or []),
        "facts_verified": sum(1 for f in (final.get("facts") or []) if f.get("verified")),
        "quality": (final.get("quality") or {}).get("overall"),
        "confidence": final.get("confidence"),
        "iterations": int(final.get("iteration", 0)),
    }


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="mars-perf-")
    settings = Settings(
        groq_api_key="bench-key", database_url=f"{tmp}/perf.db", _env_file=None
    )
    report = {"suite": "mars-perf-benchmark", "version": "1.0", **_run_sync(settings)}
    print(json.dumps(report, indent=2, default=str))
    return 0


def _run_sync(settings: Settings) -> dict:
    return asyncio.run(_run(settings))


if __name__ == "__main__":
    raise SystemExit(main())
