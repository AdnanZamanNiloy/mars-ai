"""Research Modes tests (Phase 3.7)."""
from app.graph.workflow import build_initial_state
from app.agents.orchestrator import MODE_PRESETS, VALID_MODES


def test_presets_shape():
    assert set(MODE_PRESETS) == {"quick", "standard", "deep", "executive", "audit", "redteam"}
    assert MODE_PRESETS["quick"]["max_iterations"] < MODE_PRESETS["standard"]["max_iterations"]
    assert MODE_PRESETS["standard"]["max_iterations"] < MODE_PRESETS["deep"]["max_iterations"]
    assert MODE_PRESETS["quick"]["max_agents"] < MODE_PRESETS["deep"]["max_agents"]
    assert set(VALID_MODES) == {"quick", "standard", "deep", "executive", "audit", "redteam"}
    # Distinct resource profiles: no two modes share a tuple.
    assert len({(p["max_agents"], p["max_iterations"]) for p in MODE_PRESETS.values()}) == 6


def test_quick_mode_respects_one_iteration():
    """GAP-8 regression: the max(3,...) iteration floor must not defeat quick mode."""
    state = build_initial_state("what is RAG", 3, mode="quick")
    assert state["max_iterations"] == 1
    assert state["mode"] == "quick"
    assert state["orchestration"]["target_agents"] == 2


def test_only_deep_and_executive_exceed_default_cap():
    hard = (
        "Should Bangladesh invest in nuclear vs solar energy over the next 20 years, "
        "considering financing, grid impact, and political trade-offs between "
        "regional power strategies?"
    )
    standard = build_initial_state(hard, 3, mode="standard")
    deep = build_initial_state(hard, 3, mode="deep")
    executive = build_initial_state(hard, 3, mode="executive")
    audit = build_initial_state(hard, 3, mode="audit")
    redteam = build_initial_state(hard, 3, mode="redteam")
    assert standard["orchestration"]["target_agents"] == 4  # standard plans 4 angles
    assert audit["orchestration"]["target_agents"] <= 3
    assert redteam["orchestration"]["target_agents"] <= 3
    assert deep["orchestration"]["target_agents"] > 3  # explicit opt-in modes exceed it
    assert executive["orchestration"]["target_agents"] > 3


def test_unknown_mode_falls_back_to_standard():
    state = build_initial_state("what is RAG", 3, mode="bogus")
    assert state["mode"] == "standard"
    assert state["max_iterations"] == 3
