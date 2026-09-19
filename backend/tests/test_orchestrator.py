"""Adaptive Orchestrator tests (Phase 2.1)."""
from app.agents.orchestrator import orchestrate, score_complexity, classify_query_type

VERY_HIGH = (
    "Should Bangladesh invest in nuclear vs solar energy over the next 20 years, "
    "considering financing, grid impact, and political trade-offs between "
    "regional power strategies?"
)


def test_very_high_clamped_without_flag():
    plan = orchestrate(VERY_HIGH, max_parallel_agents=3, deep_research=False)
    assert plan.complexity.level == "very_high"
    assert plan.target_agents == 3
    assert plan.clamped
    assert any("deep_research" in n for n in plan.notes)


def test_very_high_deep_research_lifts_cap():
    """2.1 DoD: deep_research=true is what allows exceeding the default cap."""
    plan = orchestrate(VERY_HIGH, max_parallel_agents=3, deep_research=True)
    assert plan.target_agents > 3
    assert not plan.clamped
    assert plan.notes == []


def test_low_and_medium_never_lift_cap():
    easy = orchestrate("what is RAG", 3, deep_research=True)
    assert easy.target_agents == 3 and not easy.clamped

    medium = orchestrate("compare X and Y costs in the Bangladesh energy market today", 3, deep_research=True)
    assert medium.target_agents == 3 and medium.clamped
    assert any("clamped" in n for n in medium.notes)


def test_clamp_note_present_whenever_clamped():
    plan = orchestrate("compare X and Y costs in the Bangladesh energy market today", 3, deep_research=False)
    assert plan.clamped and plan.notes, "clamped plans must carry a metadata note"


def test_query_type_classification():
    assert classify_query_type("compare X and Y") == "comparative"
    assert classify_query_type("what is RAG") == "factual"
    assert classify_query_type("should we do X") == "analytical"
    assert classify_query_type("overview of the field") == "exploratory"


def test_score_complexity_levels():
    assert score_complexity("what is RAG").level == "low"
    assert score_complexity(VERY_HIGH).level == "very_high"


def test_plan_targets_require_evidence_and_criticism():
    """Non-quick plans must hunt evidence and counter-arguments, not just
    background: required_axes reach the planner via build_initial_state."""
    from app.graph.workflow import build_initial_state

    plan = orchestrate(VERY_HIGH, max_parallel_agents=3, deep_research=False)
    assert {"evidence", "criticism"} <= set(plan.targets.required_axes)
    assert plan.targets.sub_questions == plan.target_agents
    assert plan.targets.min_sources_per_axis >= 2

    state = build_initial_state(VERY_HIGH, max_iterations=3, max_parallel_agents=3)
    orch = state["orchestration"]
    assert {"evidence", "criticism"} <= set(orch["required_axes"])
    assert orch["target_sub_questions"] == orch["target_agents"]


def test_explicit_quick_mode_stays_quick():
    plan = orchestrate(VERY_HIGH, max_parallel_agents=3, mode="quick")
    assert plan.mode == "quick"
    assert plan.target_agents <= 2


def test_redteam_heuristics_run_llm_free():
    """Red-team review must work with no model client: heuristics always run."""
    import asyncio

    from app.agents.redteam import redteam_agent

    facts = [
        {"claim": "Solar capacity doubled in 2025", "source": "https://a.org/x",
         "confidence": 0.8, "verified": True},
        {"claim": "Solar capacity doubled in 2025", "source": "https://b.org/y",
         "confidence": 0.75, "verified": True},
    ]
    report = asyncio.run(redteam_agent(
        None, "How fast did solar grow?", facts, use_llm=False))
    d = report.to_dict()
    assert 0.0 <= d["survival_score"] <= 1.0
    assert isinstance(d["findings"], list) and isinstance(d["targeted_queries"], list)
    assert all(set(f) >= {"kind", "statement", "severity"} for f in d["findings"])


def test_heuristic_survival_does_not_collapse_with_more_findings():
    """Regression: the old multiplicative score fell to the floor as the
    red team did its job (the prompt asks for several 0.8+ attacks), so every
    run reported ~0% survival in the UI. The score must be driven by the
    strongest attack, remain absolute, and never collapse to 0 from a normal
    number of findings."""
    from app.agents.redteam import (
        KIND_INVALIDATING,
        RedTeamFinding,
        heuristic_survival,
    )

    def findings(n: int, severity: float = 1.0):
        return [RedTeamFinding(KIND_INVALIDATING, "attack", severity) for _ in range(n)]

    assert heuristic_survival([]) == 1.0
    assert heuristic_survival(findings(1)) > 0.0
    # More severe findings lower survival, but never to zero.
    scores = [heuristic_survival(findings(n)) for n in range(1, 7)]
    assert all(s > 0.0 for s in scores), scores
    assert scores == sorted(scores, reverse=True), "must decrease monotonically"
    # A lone severity-1.0 attack is unresolved (below the 0.6 survives line)
    # but not a total loss.
    assert 0.3 <= heuristic_survival(findings(1)) < 0.6
    # A moderate attack barely moves the score.
    assert heuristic_survival(findings(1, severity=0.3)) > 0.8
