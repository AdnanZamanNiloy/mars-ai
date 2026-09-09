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
