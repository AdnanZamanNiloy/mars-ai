"""Diverse-query bench: capture MARS's plan axes + full answer + metrics.

Generalizes bench_two.py to N queries and additionally records the planner's
axis/dimension set and coverage note so plan diversity across query TYPES can
be measured (the dynamic-planning task's Phase 1/2 evidence).

Usage:
    python -m scripts.bench_multi [--out DIR] [--queries "q1" "q2" ...]
Writes one <slug>.md per query plus results.jsonl, and prints RESULT_JSON.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from pathlib import Path

from app.core.config import get_settings
from app.core.llm import LLMClient
from app.agents.search import SearchClient
from app.graph.workflow import create_workflow, build_initial_state, graph_recursion_limit
from app.core.evidence_grade import grade_facts
from app.core.degradation import reset_fallbacks, take_fallbacks

DEFAULT_QUERIES = [
    "What is the current trend of AI?",
    "What is a transformer?",
    "Should Bangladesh increase nuclear energy investment over 20 years?",
    "How does mRNA vaccine technology work and what are its limitations?",
    "Compare solar vs nuclear for grid baseload power",
    "What caused the 2023 regional banking crisis?",
    "What share of global electricity will renewables supply by 2030?",
    "Is social media harmful to teenagers?",
]


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s[:60] or "query").strip("-")


def _plan_axes(plan) -> list[str]:
    axes: list[str] = []
    for item in plan or []:
        if isinstance(item, dict):
            axis = str(item.get("axis", "") or "").strip()
        else:
            axis = ""
        if axis and axis not in axes:
            axes.append(axis)
    return axes


def _plan_questions(plan) -> list[str]:
    out: list[str] = []
    for item in plan or []:
        if isinstance(item, dict):
            q = str(item.get("question", "") or "").strip()
            if q:
                out.append(q)
    return out


async def run(query: str) -> dict:
    s = get_settings()
    reset_fallbacks()
    wf = create_workflow(LLMClient(s), SearchClient(s))
    state = build_initial_state(query, max_iterations=3, mode="deep")
    snaps, err = [], None
    try:
        async for snap in wf.astream(
            state,
            stream_mode="values",
            config={"recursion_limit": graph_recursion_limit(state)},
        ):
            snaps.append(snap)
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:200]}"
    last = snaps[-1] if snaps else {}
    facts = [f for f in last.get("facts", []) if isinstance(f, dict)]
    ver = [f for f in facts if f.get("verified")]
    graded = grade_facts(ver, contradictions=last.get("contradictions") or [])
    corr = [(g.get("evidence") or {}).get("corroboration_count", 1) for g in graded]
    dist = Counter((g.get("evidence") or {}).get("grade") for g in graded)
    total_graded = sum(dist.values()) or 1
    cons = last.get("contradictions", []) or []
    q = last.get("quality") or {}
    answer = str(last.get("synthesized_answer") or last.get("final_report") or "")
    plan = last.get("sub_questions", []) or []
    orch = last.get("orchestration", {}) or {}
    degraded = take_fallbacks()
    return {
        "query": query,
        "loop_error": err,
        "terminated": err is None,
        "iterations": last.get("iteration"),
        "max_iterations": last.get("max_iterations"),
        "plan_contracts": len(plan),
        "plan_axes": _plan_axes(plan),
        "plan_questions": _plan_questions(plan),
        "plan_injected_axes": orch.get("required_axes", []),
        "query_type": orch.get("query_type"),
        "complexity_level": orch.get("complexity_level"),
        "facts_total": len(facts),
        "verified": len(ver),
        "grade_dist": dict(dist),
        "primary_share": (
            sum(dist.get(g, 0) for g in ("A", "B")) / total_graded
        ),
        "corroborated_ge2": sum(1 for c in corr if c >= 2),
        "needs_corroboration": sum(
            1 for g in graded if (g.get("evidence") or {}).get("needs_corroboration")
        ),
        "contradictions_resolved": sum(1 for c in cons if c.get("resolved")),
        "contradictions_unresolved": sum(1 for c in cons if not c.get("resolved")),
        "confidence": last.get("confidence"),
        "quality": q,
        "degraded": degraded,
        "synthesis_llm_written": "synthesizer" not in degraded,
        "answer_chars": len(answer),
        "answer": answer,
    }


def _write_md(out_dir: Path, o: dict) -> None:
    path = out_dir / f"{_slug(o['query'])}.md"
    lines = [
        f"# {o['query']}",
        "",
        f"- query_type: {o.get('query_type')} | complexity: {o.get('complexity_level')}",
        f"- plan_axes: {', '.join(o.get('plan_axes') or [])}",
        f"- required_axes (orchestration): {', '.join(o.get('plan_injected_axes') or [])}",
        f"- contracts: {o.get('plan_contracts')} | iterations: {o.get('iterations')}/{o.get('max_iterations')}",
        f"- confidence: {o.get('confidence')} | primary_share: {round(o.get('primary_share') or 0, 3)}",
        f"- grade_dist: {o.get('grade_dist')} | degraded: {o.get('degraded')}",
        "",
        "## Plan questions",
        "",
    ]
    for i, qq in enumerate(o.get("plan_questions") or [], 1):
        lines.append(f"{i}. {qq}")
    lines += ["", "## Answer", "", o.get("answer", ""), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/mars_multi")
    ap.add_argument("--queries", nargs="*", default=None)
    args = ap.parse_args()
    queries = args.queries or DEFAULT_QUERIES
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    outs = []
    for qq in queries:
        try:
            o = await run(qq)
        except Exception as e:  # never lose the whole bench to one query
            o = {"query": qq, "loop_error": f"{type(e).__name__}: {e}", "answer": ""}
        outs.append(o)
        _write_md(out_dir, o)
        print("RESULT_JSON", json.dumps(o))
        print(f"[bench_multi] {qq!r}: axes={(o.get('plan_axes') or [])} "
              f"conf={o.get('confidence')} err={o.get('loop_error')}")
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for o in outs:
            fh.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
