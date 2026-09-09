"""Dynamic Research Depth controller tests (Phase 2.8)."""
import pytest

from app.core import depth_controller as dc
from app.core.config import Settings


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
            {"claim": "fact 1 about X", "source": "https://arxiv.org/a", "verified": True},
            {"claim": "fact 2 about X", "source": "https://arxiv.org/b", "verified": True},
            {"claim": "fact 3 about X", "source": "https://arxiv.org/c", "verified": True},
            {"claim": "fact 4 about X", "source": "https://arxiv.org/d", "verified": True},
        ],
    }
    base.update(overrides)
    return base


def test_high_confidence_with_all_axes_covered_finalizes():
    state = _state(
        confidence=0.85,
        confidence_history=[0.6, 0.75, 0.85],
        search_results=[
            {"url": "https://arxiv.org/a", "sub_question": "what is X"},
            {"url": "https://arxiv.org/b", "sub_question": "what is X"},
            {"url": "https://arxiv.org/c", "sub_question": "X vs Y costs"},
            {"url": "https://arxiv.org/d", "sub_question": "X vs Y costs"},
        ],
        facts=[
            {"claim": "fact 1 about X", "source": "https://arxiv.org/a", "verified": True},
            {"claim": "fact 2 about X", "source": "https://arxiv.org/b", "verified": True},
            {"claim": "fact 3 about X", "source": "https://arxiv.org/c", "verified": True},
            {"claim": "fact 4 about X", "source": "https://arxiv.org/d", "verified": True},
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


def test_marginal_gain_stall_stops_even_with_gaps():
    """DoD: diminishing returns finalize before MAX_ITERATIONS."""
    state = _state(
        confidence=0.5,
        confidence_history=[0.46, 0.49, 0.5],  # deltas 0.03, 0.01 → stalled
        max_iterations=5,
        iteration=2,
    )
    checks = dc.evaluate(state)
    assert checks["marginal_gain_stop"] is True
    assert checks["ceiling_reached"] is False
    assert dc.decide(state) == "finalize"
    reason = dc.stop_reason(state)
    assert reason and "marginal" in reason


def test_budget_cutoff_blocks_expansion():
    class FakeTracker:
        over_budget = True
        limit_usd = 0.5
        estimated_cost_usd = 0.6

    state = _state(confidence=0.5, confidence_history=[0.3, 0.4, 0.5])
    state["budget_tracker"] = FakeTracker()
    assert dc.decide(state) == "finalize"


def test_low_remaining_budget_disables_expansion():
    class LowTracker:
        over_budget = False
        limit_usd = 1.0
        estimated_cost_usd = 0.95  # 5% remaining < safety margin

    state = _state(confidence=0.5, confidence_history=[0.3, 0.4, 0.5])
    state["budget_tracker"] = LowTracker()
    assert dc.evaluate(state)["budget_low"] is True
    assert dc.decide(state) == "finalize"


def test_ceiling_reached_finalizes():
    state = _state(iteration=3, max_iterations=3)
    assert dc.evaluate(state)["ceiling_reached"] is True
    assert dc.decide(state) == "finalize"


def test_critic_pass_alone_finalizes():
    state = _state()
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
