#!/usr/bin/env python3
"""Scenario Engine (4.3): compare named alternative assumptions side by side.

Given a completed base run (or a fresh base query) plus a scenario file of
one-sentence assumptions, this script:
  1. derives assumption-scoped sub-questions from the BASE plan,
  2. runs each scenario through the existing research pipeline as a
     subroutine (assumption embedded in the query — no pipeline changes),
  3. reuses the base run's verified claims as shared context and isolates
     each scenario's NEW claims,
  4. compares confidence / findings / recommended option across scenarios
     and states explicitly whether the recommendation is robust or flips.

Usage:
    .venv/bin/python scripts/run_scenarios.py --run-id <completed-run> --scenarios scripts/scenarios.example.json
    .venv/bin/python scripts/run_scenarios.py --query "Should we ..." --scenarios my.json --mode quick --out result.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.scenarios import (  # noqa: E402
    build_scenario_query,
    compare_scenarios,
    scenario_new_claims,
    scope_questions,
)


async def stream_run_id(client: httpx.AsyncClient, server: str, query: str, mode: str, timeout: float) -> str | None:
    """Run one pipeline query to completion; return its run_id.

    The stream MUST be drained fully — returning early disconnects the
    client, which cancels the server-side run before it completes.
    """
    run_id: str | None = None
    try:
        async with client.stream(
            "POST", f"{server}/api/research/stream",
            json={"query": query, "mode": mode}, timeout=timeout,
        ) as response:
            if response.status_code != 200:
                print(f"  stream HTTP {response.status_code}", flush=True)
                return None
            async for raw in response.aiter_lines():
                line = raw.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if evt.get("type") == "progress" and evt.get("request_id"):
                    run_id = evt["request_id"]
                if evt.get("type") == "error":
                    print(f"  stream error: {evt.get('message', '')[:160]}", flush=True)
    except Exception as exc:
        print(f"  transport failure: {type(exc).__name__}: {exc}"[:200], flush=True)
        return None
    return run_id


async def wait_trace(client: httpx.AsyncClient, server: str, run_id: str, tries: int = 12) -> dict | None:
    """Poll the trace until the run leaves running state (stream may end first)."""
    for _ in range(tries):
        try:
            resp = await client.get(f"{server}/api/research/{run_id}/trace", timeout=30.0)
            if resp.status_code == 200:
                trace = resp.json()
                if trace.get("status") in ("completed", "failed", "timeout"):
                    return trace
        except Exception as exc:
            print(f"  trace poll failed: {type(exc).__name__}", flush=True)
        await asyncio.sleep(10)
    return None


def trace_summary(trace: dict) -> dict:
    decisions = trace.get("decisions") or []
    rec = next((o for o in decisions if o.get("is_recommended")), None)
    conf = (trace.get("final_report") or {}).get("confidence")
    return {
        "status": trace.get("status"),
        "confidence": float(conf) if isinstance(conf, (int, float)) else None,
        "claims": trace.get("claims") or [],
        "recommended_option": rec.get("option_label") if rec else None,
    }


def render(comparison: Dict, base_query: str) -> str:
    lines = [f"Scenario comparison for: {base_query}", ""]
    lines.append(f"Base recommendation: Option {comparison['base_option']}" if comparison["base_option"] else "Base run: no recommendation")
    lines.append("")
    header = f"{'scenario':<14}{'confidence':<12}{'claims':<8}{'new':<6}recommendation"
    lines.append(header)
    lines.append("-" * len(header))
    for row in comparison["rows"]:
        conf = f"{row['confidence']:.2f}" if isinstance(row["confidence"], (int, float)) else "—"
        rec = f"Option {row['recommended_option']}" if row["recommended_option"] else "—"
        lines.append(f"{row['id']:<14}{conf:<12}{row['total_claims']:<8}{row['new_claims']:<6}{rec}")
        lines.append(f"  assumption: {row['assumption']}")
    lines.append("")
    lines.append(f"Verdict: {comparison['verdict']}")
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description="MARS Scenario Engine")
    parser.add_argument("--run-id", default=None, help="completed base run to branch from")
    parser.add_argument("--query", default=None, help="base query (runs a fresh base run first)")
    parser.add_argument("--scenarios", required=True, help="JSON file: list of {id, assumption}")
    parser.add_argument("--mode", default="quick", choices=["quick", "standard", "deep"])
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--out", default=None, help="write full comparison JSON here")
    args = parser.parse_args()

    if not args.run_id and not args.query:
        print("need --run-id or --query", file=sys.stderr)
        return 2
    with open(args.scenarios, encoding="utf-8") as fh:
        scenarios = json.load(fh)
    if not isinstance(scenarios, list) or not scenarios:
        print("scenario file must be a non-empty JSON list", file=sys.stderr)
        return 2
    for s in scenarios:
        if not s.get("id") or not s.get("assumption"):
            print("each scenario needs id + one-sentence assumption", file=sys.stderr)
            return 2

    server = args.server.rstrip("/")
    stream_timeout = float(get_settings().research_timeout_sec) + 60.0
    async with httpx.AsyncClient() as client:
        if args.run_id:
            trace = await wait_trace(client, server, args.run_id, tries=1)
            if trace is None:
                print(f"cannot load base run {args.run_id}", file=sys.stderr)
                return 1
            base_query = trace.get("query", "")
        else:
            print(f"base run: {args.query[:70]}…", flush=True)
            base_id = await stream_run_id(client, server, args.query, args.mode, stream_timeout)
            if not base_id:
                return 1
            trace = await wait_trace(client, server, base_id)
            if trace is None:
                return 1
            base_query = args.query
        base = trace_summary(trace)
        if base["status"] != "completed":
            print(f"base run status is {base['status']} — need a completed run", file=sys.stderr)
            return 1

        base_questions = [t.get("question", "") for t in trace.get("plan") or [] if t.get("question")]
        base_claims = base["claims"]
        print(f"base: {len(base_questions)} planned questions, {len(base_claims)} claims, "
              f"confidence {base['confidence']}, recommendation {base['recommended_option']}")

        results = []
        for s in scenarios:
            scoped = scope_questions(s["assumption"], base_questions)
            scenario_query = build_scenario_query(base_query, s["assumption"], scoped)
            print(f"\n[{s['id']}] assuming: {s['assumption']}", flush=True)
            for q in scoped:
                print(f"  scoped: {q[:100]}")
            scen_id = await stream_run_id(client, server, scenario_query, args.mode, stream_timeout)
            if not scen_id:
                results.append({"id": s["id"], "assumption": s["assumption"],
                                "confidence": None, "claims": [], "new_claims": [],
                                "recommended_option": None})
                continue
            scen_trace = await wait_trace(client, server, scen_id)
            if scen_trace is None or scen_trace.get("status") != "completed":
                print(f"  scenario run did not complete", flush=True)
                results.append({"id": s["id"], "assumption": s["assumption"],
                                "confidence": None, "claims": [], "new_claims": [],
                                "recommended_option": None})
                continue
            summary = trace_summary(scen_trace)
            new_claims = scenario_new_claims(summary["claims"], base_claims)
            print(f"  -> confidence {summary['confidence']}, {len(summary['claims'])} claims "
                  f"({len(new_claims)} new), recommendation {summary['recommended_option']}", flush=True)
            results.append({"id": s["id"], "assumption": s["assumption"],
                            "confidence": summary["confidence"], "claims": summary["claims"],
                            "new_claims": new_claims,
                            "recommended_option": summary["recommended_option"]})

    comparison = compare_scenarios(
        {"recommended_option": base["recommended_option"]},
        [{k: r[k] for k in ("id", "assumption", "confidence", "claims", "new_claims", "recommended_option")} for r in results],
    )
    output = {"base_query": base_query, "base": {
        "confidence": base["confidence"], "claims": len(base_claims),
        "recommended_option": base["recommended_option"]}, "comparison": comparison}
    print("\n" + render(comparison, base_query))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
