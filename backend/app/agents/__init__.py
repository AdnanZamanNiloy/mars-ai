"""MARS upgraded agents package.

Deliberately lazy: importing `app.agents` costs nothing, and each name pulls
in only the module that defines it. That matters in the host application,
where `app.agents.search` drags in httpx and provider SDKs that a worker
running only synthesis should not pay for.

Public surface (stable names, additive-only evolution):

    run_mission, MissionResult, BudgetedLLM      — the composition layer
    orchestrate, OrchestrationPlan, ...          — adaptive orchestration
    planner_agent, execution_waves, ...          — planning contracts
    SearchClient, contract_queries, ...          — multi-provider search
    summarizer_agent, verifier helpers           — extraction + verification
    critic_agent, redteam_agent                  — judgement + adversarial
    detect_contradictions, compute_confidence    — conflicts + scoring
    DepthController, StopDecision               — dynamic depth
    ResearchBudget, retry/breaker helpers        — cost + reliability
    ResearchTrace, merge_traces                  — traceability
    decision_briefing, DecisionBrief             — executive decision layer
    evaluate_mission, run_benchmark              — evaluation lab
"""
from __future__ import annotations

from typing import Any

_LAZY: dict[str, str] = {
    # mission layer
    "run_mission": "app.agents.mission",
    "MissionResult": "app.agents.mission",
    "BudgetedLLM": "app.agents.mission",
    # orchestration
    "orchestrate": "app.agents.orchestrator",
    "OrchestrationPlan": "app.agents.orchestrator",
    "ComplexityScore": "app.agents.orchestrator",
    "PlanTargets": "app.agents.orchestrator",
    "score_complexity": "app.agents.orchestrator",
    "recommend_mode": "app.agents.orchestrator",
    "MODE_PRESETS": "app.agents.orchestrator",
    "VALID_MODES": "app.agents.orchestrator",
    # planning
    "planner_agent": "app.agents.planner",
    "execution_waves": "app.agents.planner",
    "fallback_plan": "app.agents.planner",
    "enforce_axis_coverage": "app.agents.planner",
    "sanitize_dependencies": "app.agents.planner",
    # search
    "SearchClient": "app.agents.search",
    "contract_queries": "app.agents.search",
    "SearchResult": "app.agents.search",
    # extraction / verification
    "summarizer_agent": "app.agents.summarizer",
    "specialist_system_prompt": "app.agents.summarizer",
    "verify_facts": "app.agents.verifier",
    "verification_summary": "app.agents.verifier",
    # judgement / adversarial
    "critic_agent": "app.agents.critic",
    "redteam_agent": "app.agents.redteam",
    "RedTeamReport": "app.agents.redteam",
    # conflicts / scoring
    "detect_contradictions": "app.agents.contradiction",
    "summarize_contradictions": "app.agents.contradiction",
    "contradiction_followups": "app.agents.contradiction",
    "compute_confidence": "app.agents.confidence",
    "ConfidenceReport": "app.agents.confidence",
    # depth control
    "DepthController": "app.agents.stopping",
    "StopDecision": "app.agents.stopping",
    "coverage_gaps": "app.agents.stopping",
    # budget / reliability
    "ResearchBudget": "app.agents.budget",
    "BudgetExceeded": "app.agents.budget",
    "estimate_tokens": "app.agents.budget",
    "retry_async": "app.agents.reliability",
    "CircuitBreaker": "app.agents.reliability",
    "gather_bounded": "app.agents.reliability",
    "ok_results": "app.agents.reliability",
    # traceability
    "ResearchTrace": "app.agents.trace",
    "merge_traces": "app.agents.trace",
    # sources / evidence
    "classify_source": "app.agents.sources",
    "authority_score": "app.agents.sources",
    "is_primary_source": "app.agents.sources",
    "build_primary_source_query": "app.agents.sources",
    "dedupe_semantic_facts": "app.agents.evidence_utils",
    "evidence_stats": "app.agents.evidence_utils",
    "verify_answer_support": "app.agents.evidence_utils",
    "numbers_grounded": "app.agents.evidence_utils",
    # decision layer
    "decision_briefing": "app.agents.decision",
    "DecisionBrief": "app.agents.decision",
    "DecisionOption": "app.agents.decision",
    # evaluation lab
    "evaluate_mission": "app.agents.evaluation",
    "run_benchmark": "app.agents.evaluation",
    "EvaluationReport": "app.agents.evaluation",
    "BenchmarkReport": "app.agents.evaluation",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    try:
        value = getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - drift guard
        raise AttributeError(
            f"{module_path} no longer exports {name!r}; update agents/__init__.py"
        ) from exc
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY.keys()))
