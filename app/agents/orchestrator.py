"""Adaptive Orchestrator (Phase 2.1, Feature 01).

Scores query complexity with cheap heuristics — deliberately NOT another
LLM call — then maps the score to a target sub-question count, clamped by
MAX_PARALLEL_AGENTS. The hardware cap (8GB host) wins over what the raw
score suggests; very_high complexity needs an explicit deep_research flag
on the request to exceed the default cap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

COMPARATIVE_MARKERS = (
    " vs ", " versus ", "compare", "compared", "comparison",
    "better", "worse", "difference between", "which is better",
)
ANALYTICAL_MARKERS = (
    "should", "why ", "how do", "how does", "impact", "effect of",
    "consequence", "implication", "trade-off", "tradeoff", "evaluate",
    "assess", "risk", "strategy", "policy", "influence",
)
EXPLORATORY_MARKERS = (
    "overview", "landscape", "state of the art", "survey", "future of",
    "trends", "evolution", "history of",
)

# Distinct-domain hints: multiple entities joined by and/or/comma lists.
_ENTITY_SPLIT_RE = re.compile(r",|\band\b|\bor\b|/")

QUERY_TYPES = ("factual", "comparative", "analytical", "exploratory")


@dataclass
class ComplexityScore:
    score: int
    level: str                # low | medium | high | very_high
    query_type: str
    markers_found: List[str] = field(default_factory=list)


def classify_query_type(query: str) -> str:
    q = query.lower()
    if any(m in q for m in COMPARATIVE_MARKERS):
        return "comparative"
    if any(m in q for m in ANALYTICAL_MARKERS):
        return "analytical"
    if any(m in q for m in EXPLORATORY_MARKERS):
        return "exploratory"
    return "factual"


def score_complexity(query: str) -> ComplexityScore:
    q = query.lower().strip()
    markers: List[str] = []
    score = 0

    words = len(q.split())
    if words > 25:
        score += 3
        markers.append("very_long_query")
    elif words > 12:
        score += 2
        markers.append("long_query")
    elif words > 6:
        score += 1
        markers.append("medium_query")

    if any(m in q for m in COMPARATIVE_MARKERS):
        score += 3
        markers.append("comparative")
    if any(m in q for m in ANALYTICAL_MARKERS):
        score += 2
        markers.append("analytical")
    if any(m in q for m in EXPLORATORY_MARKERS):
        score += 2
        markers.append("exploratory")

    entities = [e for e in _ENTITY_SPLIT_RE.split(q) if len(e.split()) >= 1 and e.strip()]
    if len(entities) >= 3:
        score += 1
        markers.append("multi_entity")

    if re.search(r"\b(20\d{2}|over|last|past|since|within|decade|year)\b", q):
        score += 1
        markers.append("time_bound")

    if score >= 7:
        level = "very_high"
    elif score >= 5:
        level = "high"
    elif score >= 3:
        level = "medium"
    else:
        level = "low"

    return ComplexityScore(score=score, level=level, query_type=classify_query_type(query), markers_found=markers)


# Level → target sub-question count before clamping.
LEVEL_TARGETS: Dict[str, int] = {
    "low": 3,
    "medium": 4,
    "high": 5,
    "very_high": 6,
}


@dataclass
class OrchestrationPlan:
    complexity: ComplexityScore
    target_agents: int
    clamped: bool
    deep_research: bool
    max_parallel_agents: int
    notes: List[str] = field(default_factory=list)


def orchestrate(
    query: str,
    max_parallel_agents: int,
    deep_research: bool = False,
) -> OrchestrationPlan:
    """Map a query to a target agent count under the hardware cap."""
    complexity = score_complexity(query)
    raw_target = LEVEL_TARGETS[complexity.level]
    notes: List[str] = []

    effective_cap = max_parallel_agents
    if complexity.level == "very_high" and not deep_research:
        notes.append(
            "very_high complexity clamped to MAX_PARALLEL_AGENTS; "
            "pass deep_research=true to lift the cap"
        )

    target_agents = min(raw_target, effective_cap)
    clamped = target_agents < raw_target

    return OrchestrationPlan(
        complexity=complexity,
        target_agents=target_agents,
        clamped=clamped,
        deep_research=deep_research,
        max_parallel_agents=effective_cap,
        notes=notes,
    )


def plan_metadata_for_event(plan: OrchestrationPlan) -> Dict[str, Any]:
    """Compact dict for embedding in the plan NDJSON event metadata."""
    return {
        "complexity_score": plan.complexity.score,
        "complexity_level": plan.complexity.level,
        "query_type": plan.complexity.query_type,
        "target_agents": plan.target_agents,
        "max_parallel_agents": plan.max_parallel_agents,
        "clamped": plan.clamped,
        "deep_research": plan.deep_research,
        "notes": plan.notes,
    }
