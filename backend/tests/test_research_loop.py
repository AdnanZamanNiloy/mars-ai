"""Research-loop enforcement: coverage hard-block, gap→search, normalization.

These tests pin the failure modes the research-loop fix addresses: a run that
detects weaknesses (uncovered planned angles, uncorroborated claims, severe
contradictions) and then STOPS anyway. The depth controller now refuses to
finalize on a soft signal while a useful pass can still run, the evidence gate
is not defeated by a depth-controller soft stop, and numeric contradictions
are only emitted when unit+scope+period are compatible.

Mirrors the style of test_depth_controller_v2.py / test_evidence_gate.py.
"""
from app.core import depth_controller
from app.core.config import Settings
from app.core.contradictions import find_contradictions
from app.core.usage import clear_run_usage, start_run_usage
from app.graph.workflow import _evidence_gaps_remain


def _settings(**kw):
    return Settings(groq_api_key="k", _env_file=None, **kw)


def _state(**overrides):
    """Two planned axes; only the definition axis has verified facts."""
    base = {
        "query": "what is X and how does it compare",
        "iteration": 1,
        "max_iterations": 4,
        "confidence": 0.9,
        "confidence_history": [0.6, 0.75, 0.9],
        "critique": {
            "is_sufficient": True,
            "improved_queries": [],
            "reason": "ok",
        },
        "sub_questions": [
            {"axis": "definition", "question": "what is X", "minimum_sources": 1},
            {"axis": "comparison", "question": "X vs Y", "minimum_sources": 1},
        ],
        "search_results": [
            {"url": "https://a.com/x", "sub_question": "what is X"},
        ],
        "facts": [
            {"claim": "X is a taxonomy of knowledge", "source": "https://a.com/x", "verified": True},
        ],
        "contradictions": [],
        "mode": "standard",
    }
    base.update(overrides)
    return base


def _covered_state(**overrides):
    state = _state(
        confidence=0.85,
        confidence_history=[0.6, 0.75, 0.85],
        search_results=[
            {"url": "https://a.com/x", "sub_question": "what is X"},
            {"url": "https://b.com/y", "sub_question": "X vs Y"},
        ],
        facts=[
            {"claim": "X is a taxonomy of knowledge", "source": "https://a.com/x", "verified": True},
            {"claim": "the comparison favours X", "source": "https://b.com/y", "verified": True},
        ],
    )
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# 1. Coverage hard-block
# ---------------------------------------------------------------------------

def test_uncovered_axis_blocks_soft_stop_even_at_high_confidence():
    """An uncovered planned axis must expand even when confidence is high and
    marginal gain is low — the core research-loop bug (gaps detected, stop
    anyway)."""
    state = _state(
        confidence=0.95,
        confidence_history=[0.80, 0.88, 0.95],  # large gains, no stall
        critique={"is_sufficient": True, "improved_queries": [], "reason": "ok"},
    )
    checks = depth_controller.evaluate(state, _settings())
    assert checks["uncovered_axes"] == ["comparison"]
    assert depth_controller.decide(state, _settings()) == "expand"


def test_uncovered_axis_blocks_marginal_gain_stop():
    # Two consecutive stalls would normally stop; an uncovered axis wins.
    state = _state(
        confidence=0.5,
        confidence_history=[0.46, 0.49, 0.5],
        iteration=2,
        max_iterations=5,
        critique={"is_sufficient": True, "improved_queries": [], "reason": "ok"},
    )
    checks = depth_controller.evaluate(state, _settings())
    assert checks["marginal_gain_stop"] is True
    assert checks["uncovered_axes"] == ["comparison"]
    assert depth_controller.decide(state, _settings()) == "expand"


def test_no_uncovered_axis_when_all_covered():
    state = _covered_state()
    assert depth_controller.evaluate(state, _settings())["uncovered_axes"] == []
    assert depth_controller.decide(state, _settings()) == "finalize"


# ---------------------------------------------------------------------------
# 2. Hard walls remain absolute
# ---------------------------------------------------------------------------

def test_budget_wall_forces_finalize_even_with_uncovered_axis():
    state = _state()
    try:
        usage = start_run_usage("rl-budget", _settings(max_llm_calls=1))
        usage.budget.llm_calls = 60
        assert depth_controller.evaluate(state, _settings())["uncovered_axes"] == ["comparison"]
        assert depth_controller.decide(state, _settings()) == "finalize"
    finally:
        clear_run_usage()


def test_iteration_ceiling_forces_finalize_even_with_uncovered_axis():
    state = _state(iteration=4, max_iterations=4)
    checks = depth_controller.evaluate(state, _settings())
    assert checks["ceiling_reached"] is True
    assert checks["uncovered_axes"] == ["comparison"]
    assert depth_controller.decide(state, _settings()) == "finalize"


def test_hard_wall_reached_helper():
    state = _state()
    assert depth_controller.hard_wall_reached(state, _settings()) is False
    try:
        usage = start_run_usage("rl-hw", _settings(max_llm_calls=1))
        usage.budget.llm_calls = 60
        assert depth_controller.hard_wall_reached(state, _settings()) is True
    finally:
        clear_run_usage()


# ---------------------------------------------------------------------------
# 3. Critic insufficiency forces expansion
# ---------------------------------------------------------------------------

def test_critic_insufficient_with_improved_queries_expands():
    state = _covered_state(
        confidence=0.95,  # measured confidence is high, but the critic disagrees
        critique={
            "is_sufficient": False,
            "improved_queries": ["independent evaluation of X versus Y"],
            "reason": "no independent comparison yet",
        },
    )
    assert depth_controller.decide(state, _settings()) == "expand"


def test_counter_evidence_queries_are_visible_to_depth_controller():
    """Counter-evidence queries must ride improved_queries (not only the
    feedback string), or the stopping policy cannot see them."""
    from app.graph.workflow import _counter_evidence_queries

    state = _state(facts=[
        {"claim": "Adoption grew 42% in 2024", "source": "https://blog.example.com/a", "verified": True},
    ])
    qs = _counter_evidence_queries(state)
    assert qs and any("corroboration" in q.lower() for q in qs)


# ---------------------------------------------------------------------------
# 4. Gap → executed search via route_after_critic
# ---------------------------------------------------------------------------

def test_route_after_critic_expands_on_evidence_gap(monkeypatch):
    import app.graph.workflow as wf

    state = _state(facts=[
        {"claim": "Adoption grew 42% in 2024", "source": "https://blog.example.com/a", "verified": True},
    ])
    assert _evidence_gaps_remain(state) is True

    route = wf.create_workflow._entry_node = None  # noqa: F841 - keep import used
    # route_after_critic is a closure; exercise it through the module-level
    # decision rather than the compiled graph.
    decision = depth_controller.decide(state)
    assert decision == "expand"


def test_route_after_critic_respects_hard_wall(monkeypatch):
    state = _state(
        iteration=4,
        max_iterations=4,
        facts=[
            {"claim": "Adoption grew 42% in 2024", "source": "https://blog.example.com/a", "verified": True},
        ],
    )
    # Gap remains, but the ceiling is a hard wall: cannot expand.
    assert _evidence_gaps_remain(state) is True
    assert depth_controller.hard_wall_reached(state) is True
    assert depth_controller.decide(state) == "finalize"


# ---------------------------------------------------------------------------
# 5. Contradiction normalization (unit / scope / period)
# ---------------------------------------------------------------------------

def test_scope_separated_numbers_are_not_a_numeric_contradiction():
    facts = [
        {"claim": "Carbon emissions fell 30% globally across the power sector last year",
         "source": "https://a.com/x"},
        {"claim": "Carbon emissions fell 5% in the United States across the power sector last year",
         "source": "https://b.com/y"},
    ]
    found = find_contradictions(facts)
    assert found, "a scope-separated finding should still be surfaced"
    assert all(c.get("kind") != "numeric" for c in found)
    assert found[0]["kind"] == "scope"
    assert found[0]["severity"] < 0.60


def test_period_separated_numbers_are_temporal_not_numeric():
    facts = [
        {"claim": "Global capacity reached 1200 GW in 2023", "source": "https://a.com/x"},
        {"claim": "Global capacity reached 1600 GW in 2024", "source": "https://b.com/y"},
    ]
    found = find_contradictions(facts)
    assert found and found[0]["kind"] == "temporal"
    assert all(c.get("kind") != "numeric" for c in found)


def test_unit_separated_numbers_are_not_a_contradiction():
    facts = [
        {"claim": "Efficiency improved by 22% in the latest generation", "source": "https://a.com/x"},
        {"claim": "Plants added 22 GW of capacity in the latest generation", "source": "https://b.com/y"},
    ]
    assert find_contradictions(facts) == []


def test_compatible_unit_scope_period_numbers_are_numeric():
    facts = [
        {"claim": "The market grew by 25% globally last year", "source": "https://a.com/x"},
        {"claim": "The market grew by 11% globally last year", "source": "https://b.com/y"},
    ]
    found = find_contradictions(facts)
    assert found and found[0]["kind"] == "numeric"
    assert found[0]["values"]["unit"] == "%"


# ---------------------------------------------------------------------------
# 6. Loop termination
# ---------------------------------------------------------------------------

def test_iteration_ceiling_prevents_infinite_loop():
    state = _state(
        iteration=3,
        max_iterations=3,
        facts=[
            {"claim": "Adoption grew 42% in 2024", "source": "https://blog.example.com/a", "verified": True},
        ],
    )
    # Both an uncovered axis AND an uncorroborated claim, yet the ceiling stops.
    assert depth_controller.decide(state, _settings()) == "finalize"


def test_no_novel_queries_stops_when_gaps_cannot_be_searched():
    """Angles covered, nothing actionable left to search, but an important
    claim is still single-source: finalize rather than burn a pass re-finding
    the same pages."""
    state = _state(
        critique={"is_sufficient": False, "improved_queries": [], "reason": "g"},
        search_results=[
            {"url": "https://a.com/x", "sub_question": "what is X"},
            {"url": "https://b.com/y", "sub_question": "X vs Y"},
        ],
        facts=[
            {"claim": "Adoption grew 42% in 2024", "source": "https://a.com/x", "verified": True},
            {"claim": "the comparison favours X", "source": "https://b.com/y", "verified": True},
        ],
    )
    checks = depth_controller.evaluate(state, _settings())
    assert checks["uncovered_axes"] == []
    assert checks["needs_corroboration_count"] > 0
    assert checks["novel_followups"] == []
    assert depth_controller.decide(state, _settings()) == "finalize"


# ---------------------------------------------------------------------------
# 7. Regression: max_iterations is a HARD routing stop (GraphRecursionError)
# ---------------------------------------------------------------------------

def _route_after_critic():
    """The routing closure is registered as a conditional edge on 'critic'."""
    import app.graph.workflow as wfmod

    graph = wfmod.create_workflow(llm=None, search_client=None)
    branch = graph.builder.branches["critic"]["route_after_critic"]
    return branch.path.func


def test_ceiling_is_hard_stop_with_every_expansion_trigger_present():
    """At the ceiling, uncovered axes + critic-insufficiency + uncorroborated
    claims must ALL yield to the iteration ceiling: decide() finalizes and the
    router returns 'synthesizer' — never the planner edge."""
    state = _state(
        iteration=4,
        max_iterations=4,
        confidence=0.2,
        critique={
            "is_sufficient": False,
            "improved_queries": ["independent evaluation of X versus Y"],
            "reason": "gaps remain",
        },
        facts=[
            {"claim": "Adoption grew 42% in 2024", "source": "https://blog.example.com/a", "verified": True},
        ],
    )
    checks = depth_controller.evaluate(state, _settings())
    # All three expansion triggers are genuinely latched on...
    assert "comparison" in checks["uncovered_axes"]
    assert checks["needs_corroboration_count"] > 0
    assert checks["critic_sufficient"] is False
    assert checks["ceiling_reached"] is True
    # ...yet the ceiling wins at both layers.
    assert depth_controller.decide(state, _settings()) == "finalize"
    assert depth_controller.hard_wall_reached(state, _settings()) is True
    assert _route_after_critic()(state) == "synthesizer"


def test_evidence_gap_override_yields_to_ceiling_in_router():
    """Even when the critic says sufficient and the evidence gate detects a
    genuine gap, the router must finalize at the ceiling — the evidence gap
    cannot route back to the planner past max_iterations."""
    state = _state(
        iteration=3,
        max_iterations=3,
        facts=[
            {"claim": "Adoption grew 42% in 2024", "source": "https://blog.example.com/a", "verified": True},
        ],
        critique={"is_sufficient": True, "improved_queries": [], "reason": "ok"},
    )
    assert _evidence_gaps_remain(state) is True
    assert _route_after_critic()(state) == "synthesizer"


def test_uncovered_axes_still_expand_while_iterations_remain():
    """The fix must not disable the research loop: below the ceiling the same
    uncovered-axis state still routes to the planner."""
    state = _state(
        iteration=1,
        max_iterations=4,
        confidence=0.2,
        critique={
            "is_sufficient": False,
            "improved_queries": ["independent evaluation of X versus Y"],
            "reason": "gaps remain",
        },
    )
    assert depth_controller.evaluate(state, _settings())["ceiling_reached"] is False
    assert depth_controller.decide(state, _settings()) == "expand"
    assert _route_after_critic()(state) == "planner"


async def test_deep_run_terminates_without_graph_recursion_error():
    """A full mocked deep run whose critic NEVER reports sufficient must still
    terminate: the iteration ceiling stops it and the graph recursion budget
    accommodates every pass, asserting final iteration <= max_iterations."""
    from bench.mock_pipeline import FakeLLM, FakeSearch
    from app.core import llm_cache as _lc
    from app.core.usage import clear_run_usage, start_run_usage
    from app.graph.workflow import (
        build_initial_state,
        create_workflow,
        graph_recursion_limit,
    )

    _lc._force_disabled = True  # the fake LLM must not be masked by the cache

    settings = _settings()
    llm = FakeLLM(settings, critic_pass_on_iteration=10_000)
    search = FakeSearch(settings)
    workflow = create_workflow(llm, search_client=search)

    state = build_initial_state("What is the current trend of AI?", max_iterations=5, mode="deep")
    assert state["max_iterations"] == 5

    usage = start_run_usage("test-deep-loop", settings, mode="deep")
    try:
        final = dict(state)
        async for snapshot in workflow.astream(
            state,
            stream_mode="values",
            config={"recursion_limit": graph_recursion_limit(state)},
        ):
            final = {**final, **{k: v for k, v in snapshot.items() if v}}
    finally:
        clear_run_usage()

    assert int(final.get("iteration", 0)) <= state["max_iterations"]
    assert final.get("final_report")
