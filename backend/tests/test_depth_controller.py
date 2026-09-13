"""Dynamic Research Depth controller tests (Phase 2.8)."""
from app.core import depth_controller as dc


def _state(**overrides):
    base = {
        "query": "compare X and Y",
        "iteration": 1,
        "max_iterations": 3,
        "confidence": 0.5,
        "confidence_history": [0.2, 0.35, 0.5],
        "critique": {"is_sufficient": False, "improved_queries": ["more on Y costs"]},
        "sub_questions": [
            {"axis": "definition", "question": "what is X", "minimum_sources": 2},
            {"axis": "comparison", "question": "X vs Y costs", "minimum_sources": 2},
        ],
        "search_results": [
            {"url": "https://arxiv.org/a", "sub_question": "what is X"},
            {"url": "https://arxiv.org/b", "sub_question": "what is X"},
            {"url": "https://arxiv.org/c", "sub_question": "what is X"},
            {"url": "https://arxiv.org/d", "sub_question": "what is X"},
        ],
        "facts": [
            {"claim": "the first fact about X", "source": "https://arxiv.org/a", "verified": True},
            {"claim": "the second fact about X", "source": "https://arxiv.org/b", "verified": True},
            {"claim": "fact 3 about X", "source": "https://arxiv.org/c", "verified": True},
            {"claim": "fact 4 about X", "source": "https://arxiv.org/d", "verified": True},
        ],
    }
    base.update(overrides)
    return base


def test_high_confidence_with_all_axes_covered_finalizes():
    # A genuine critic pass (is_sufficient=True) plus full axis coverage and
    # confidence at target finalizes. (An *insufficient* critic now forces a
    # pass even here — see test_critic_insufficiency_forces_expansion — so the
    # critique is set to the passing verdict this test is about.)
    state = _state(
        confidence=0.85,
        confidence_history=[0.6, 0.75, 0.85],
        critique={"is_sufficient": True, "improved_queries": [], "reason": "ok"},
        search_results=[
            {"url": "https://arxiv.org/a", "sub_question": "what is X"},
            {"url": "https://arxiv.org/b", "sub_question": "what is X"},
            {"url": "https://arxiv.org/c", "sub_question": "X vs Y costs"},
            {"url": "https://arxiv.org/d", "sub_question": "X vs Y costs"},
        ],
        facts=[
            {"claim": "the first fact about X", "source": "https://arxiv.org/a", "verified": True},
            {"claim": "the second fact about X", "source": "https://arxiv.org/b", "verified": True},
            {"claim": "the third fact about Y", "source": "https://arxiv.org/c", "verified": True},
            {"claim": "the fourth fact about Y", "source": "https://arxiv.org/d", "verified": True},
        ],
    )
    assert dc.evaluate(state)["sufficiency_met"] is True
    assert dc.decide(state) == "finalize"


def test_coverage_gap_and_low_confidence_expands():
    # Only the "definition" axis has verified facts; comparison axis is empty.
    state = _state(confidence=0.5, confidence_history=[0.3, 0.4, 0.5])
    assert dc.evaluate(state)["coverage_gap"] is True
    assert dc.evaluate(state)["axes_below_threshold"] == ["comparison"]
    assert dc.decide(state) == "expand"


def test_marginal_gain_stall_stops_when_angles_covered():
    """DoD: diminishing returns finalize before MAX_ITERATIONS.

    Both planned angles must be *covered* for the stall to stop the run — an
    uncovered angle now hard-blocks the soft stop (test_uncovered_axis_blocks_-
    marginal_gain_stop), which is the research-loop fix.
    """
    state = _state(
        confidence=0.5,
        confidence_history=[0.46, 0.49, 0.5],  # deltas 0.03, 0.01 → stalled
        max_iterations=5,
        iteration=2,
        # Insufficient critic with NO actionable follow-up: the mass of the
        # pool is what stops the run on diminishing returns, not the verdict.
        critique={"is_sufficient": False, "improved_queries": [], "reason": "g"},
        search_results=[
            {"url": "https://arxiv.org/a", "sub_question": "what is X"},
            {"url": "https://arxiv.org/b", "sub_question": "what is X"},
            {"url": "https://arxiv.org/c", "sub_question": "X vs Y costs"},
            {"url": "https://arxiv.org/d", "sub_question": "X vs Y costs"},
        ],
        facts=[
            {"claim": "the first fact about X", "source": "https://arxiv.org/a", "verified": True},
            {"claim": "the second fact about X", "source": "https://arxiv.org/b", "verified": True},
            {"claim": "the third fact about Y", "source": "https://arxiv.org/c", "verified": True},
            {"claim": "the fourth fact about Y", "source": "https://arxiv.org/d", "verified": True},
        ],
    )
    checks = dc.evaluate(state)
    assert checks["marginal_gain_stop"] is True
    assert checks["ceiling_reached"] is False
    assert checks["uncovered_axes"] == []
    assert dc.decide(state) == "finalize"
    reason = dc.stop_reason(state)
    assert reason and "marginal" in reason


def test_ceiling_reached_finalizes():
    state = _state(iteration=3, max_iterations=3)
    assert dc.evaluate(state)["ceiling_reached"] is True
    assert dc.decide(state) == "finalize"


def test_critic_pass_alone_finalizes():
    # Critic pass finalizes only when every planned angle is covered.
    state = _state(
        search_results=[
            {"url": "https://arxiv.org/a", "sub_question": "what is X"},
            {"url": "https://arxiv.org/b", "sub_question": "what is X"},
            {"url": "https://arxiv.org/c", "sub_question": "X vs Y costs"},
            {"url": "https://arxiv.org/d", "sub_question": "X vs Y costs"},
        ],
        facts=[
            {"claim": "the first fact about X", "source": "https://arxiv.org/a", "verified": True},
            {"claim": "the second fact about X", "source": "https://arxiv.org/b", "verified": True},
            {"claim": "the third fact about Y", "source": "https://arxiv.org/c", "verified": True},
            {"claim": "the fourth fact about Y", "source": "https://arxiv.org/d", "verified": True},
        ],
    )
    state["critique"] = {"is_sufficient": True, "improved_queries": []}
    assert dc.decide(state) == "finalize"


def test_early_stop_note_in_report(monkeypatch):
    """The report limitations must state an early stop on marginal gain."""
    import app.graph.workflow as wf

    state = _state(
        confidence=0.5,
        confidence_history=[0.46, 0.49, 0.5],
        max_iterations=5,
        iteration=2,
    )
    state["critique"] = {"is_sufficient": False, "improved_queries": ["q"]}
    state["synthesized_answer"] = "answer"
    report = wf.build_markdown_report(state)
    assert "marginal" in report.lower()
