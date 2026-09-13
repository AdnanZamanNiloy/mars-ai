"""Depth controller v2: budget stops, novel-query memory, mode targets."""

from app.core import depth_controller
from app.core.config import Settings
from app.core.usage import clear_run_usage, start_run_usage


def _state(**kw):
    base = {
        "query": "q",
        "sub_questions": [
            {"question": "what exactly defines transformer architecture", "axis": "definition", "minimum_sources": 1},
            {"question": "how strong is the benchmark evidence for transformers", "axis": "evidence", "minimum_sources": 1},
        ],
        "search_results": [
            {"url": "https://a.com/x", "sub_question": "what exactly defines transformer architecture"},
            {"url": "https://b.com/y", "sub_question": "how strong is the benchmark evidence for transformers"},
        ],
        "facts": [
            {"claim": "Transformers are attention-based architectures", "source": "https://a.com/x", "verified": True, "sub_question": "what exactly defines transformer architecture"},
            {"claim": "Two benchmark studies support the claims", "source": "https://b.com/y", "verified": True, "sub_question": "how strong is the benchmark evidence for transformers"},
        ],
        "critique": {"is_sufficient": False, "improved_queries": ["compare transformer efficiency against recurrent models"], "reason": "gaps"},
        "iteration": 1,
        "max_iterations": 4,
        "confidence": 0.5,
        "confidence_history": [0.5],
        "mode": "standard",
    }
    base.update(kw)
    return base


def _settings(**kw):
    return Settings(groq_api_key="k", _env_file=None, **kw)


def test_budget_exhaustion_forces_finalize():
    state = _state()
    try:
        usage = start_run_usage("t-1", _settings(max_llm_calls=1))
        usage.budget.llm_calls = 60  # simulate spend past ceiling
        assert depth_controller.decide(state, _settings()) == "finalize"
        checks = depth_controller.last_decision(state, _settings())
        assert checks["budget_stop"] is True
    finally:
        clear_run_usage()


def test_unaffordable_pass_forces_finalize():
    state = _state()
    try:
        usage = start_run_usage("t-2", _settings(max_llm_calls=1))
        usage.budget.llm_calls = 1  # ceiling exactly reached: next pass unaffordable
        assert depth_controller.decide(state, _settings()) == "finalize"
    finally:
        clear_run_usage()


def test_no_budget_outside_run():
    # No ledger active: budget checks inert, decision unchanged.
    state = _state()
    assert depth_controller.decide(state, _settings()) in ("expand", "finalize")
    checks = depth_controller.last_decision(state, _settings())
    assert checks["budget"]["active"] is False
    assert checks["budget_stop"] is False


def test_no_novel_queries_stops():
    # Critic proposes follow-ups that duplicate already-searched questions.
    state = _state(search_results=[
        {"url": "https://a.com/x", "sub_question": "what exactly defines transformer architecture"},
        {"url": "https://b.com/y", "sub_question": "how strong is the benchmark evidence for transformers"},
        {"url": "https://c.com/z", "sub_question": "compare transformer efficiency against recurrent models"},
    ])
    assert depth_controller.decide(state, _settings()) == "finalize"
    checks = depth_controller.last_decision(state, _settings())
    assert checks["no_novel_queries"] is True


def test_novel_followups_allow_expansion():
    state = _state()  # improved query not yet searched
    assert depth_controller.decide(state, _settings()) == "expand"


def test_mode_confidence_target_respected():
    # Confidence 0.62: quick target (0.60) met -> finalize; audit (0.85) not.
    state = _state(confidence=0.62, confidence_history=[0.62],
                   critique={"is_sufficient": False, "improved_queries": ["compare transformer efficiency against recurrent models"], "reason": "g"})
    quick = depth_controller.evaluate(dict(state, mode="quick"), _settings())
    audit = depth_controller.evaluate(dict(state, mode="audit"), _settings())
    assert quick["confidence_target"] == 0.60
    assert audit["confidence_target"] == 0.85
    assert quick["sufficiency_met"] is True
    assert audit["sufficiency_met"] is False


def test_min_iterations_blocks_premature_stop():
    # Audit demands >= 2 passes: a sufficient-but-shallow pass-1 must not stop.
    state = _state(mode="audit", confidence=0.9, confidence_history=[0.9],
                   critique={"is_sufficient": True, "improved_queries": [], "reason": "ok"})
    assert depth_controller.decide(state, _settings()) == "expand"
    state["iteration"] = 2
    assert depth_controller.decide(state, _settings()) == "finalize"


def test_stop_reason_mentions_budget():
    state = _state()
    try:
        usage = start_run_usage("t-3", _settings())
        usage.budget.llm_calls = 60
        reason = depth_controller.stop_reason(state, _settings())
        assert reason and "budget" in reason.lower()
    finally:
        clear_run_usage()


def test_stop_reason_mentions_no_novel_queries():
    state = _state(search_results=[
        {"url": "https://a.com/x", "sub_question": "what exactly defines transformer architecture"},
        {"url": "https://b.com/y", "sub_question": "how strong is the benchmark evidence for transformers"},
        {"url": "https://c.com/z", "sub_question": "compare transformer efficiency against recurrent models"},
    ])
    reason = depth_controller.stop_reason(state, _settings())
    assert reason and "duplicated" in reason.lower()


def test_checks_are_per_call_not_global():
    """Two different states must yield their own checks, in any order.

    Regression for the removed module-level `_last_decision` cache: reading
    state A after evaluating state B used to return B's checks, so a
    concurrent run could see another run's decision.
    """
    state_a = _state(search_results=[
        {"url": "https://a.com/x", "sub_question": "what exactly defines transformer architecture"},
        {"url": "https://b.com/y", "sub_question": "how strong is the benchmark evidence for transformers"},
        {"url": "https://c.com/z", "sub_question": "compare transformer efficiency against recurrent models"},
    ])  # no novel queries
    state_b = _state()  # novel query pending
    checks_a = depth_controller.last_decision(state_a, _settings())
    checks_b = depth_controller.last_decision(state_b, _settings())
    assert checks_a["no_novel_queries"] is True
    assert checks_b["no_novel_queries"] is False
    # Re-reading A after B still returns A's result.
    assert depth_controller.last_decision(state_a, _settings())["no_novel_queries"] is True
