"""Step 3: evidence-sufficiency driven adaptive research depth.

The depth controller must decide expand/stop from MEASURED evidence
sufficiency — high-impact uncorroborated claims, unresolved high-impact
contradictions, thinly-evidenced planned dimensions — not from iteration count,
plan size or marginal gain alone. Every hard wall (budget/time/ceiling/
expansion-cap) still wins, and the decision must be explainable.

Mirrors the fixtures in test_depth_controller_v2.py / test_research_loop.py.
"""
from app.core import depth_controller
from app.core.config import Settings
from app.core.investigation_state import (
    STATUS_ATTEMPTED,
    STATUS_EXHAUSTED,
    STATUS_OPEN,
    investigation_key,
)
from app.core.usage import clear_run_usage, start_run_usage


def _settings(**kw):
    return Settings(groq_api_key="k", _env_file=None, **kw)


def _fact(claim, source, dim, **kw):
    base = {
        "claim": claim,
        "source": source,
        "verified": True,
        "sub_question": dim,
        "is_primary": True,
    }
    base.update(kw)
    return base


def _state(**overrides):
    """Two planned dimensions, both covered by primary corroborated facts."""
    base = {
        "query": "cost and mechanism of X",
        "iteration": 1,
        "max_iterations": 4,
        "confidence": 0.9,
        "confidence_history": [0.8, 0.9],
        "mode": "standard",
        "critique": {"is_sufficient": False, "improved_queries": [], "reason": "ok"},
        "contradictions": [],
        "sub_questions": [
            {"axis": "evidence", "question": "what is the cost data", "minimum_sources": 1},
            {"axis": "mechanism", "question": "how does the mechanism work", "minimum_sources": 1},
        ],
        "search_results": [
            {"url": "https://gov.uk/x", "sub_question": "what is the cost data"},
            {"url": "https://who.int/y", "sub_question": "how does the mechanism work"},
        ],
        "facts": [
            _fact(
                "Global spending reached 200 billion dollars in 2025",
                "https://gov.uk/x",
                "what is the cost data",
                corroborating_sources=["https://gov.uk/x", "https://who.int/data"],
                corroboration_count=2,
            ),
            _fact(
                "The mechanism works via a multi-stage pipeline process",
                "https://who.int/y",
                "how does the mechanism work",
            ),
        ],
    }
    base.update(overrides)
    return base


def _state_with_uncorroborated_high_impact(**overrides):
    """One dimension, whose only claim is a single-publisher quantitative one."""
    state = _state(
        sub_questions=[
            {"axis": "evidence", "question": "what is the cost data", "minimum_sources": 1},
        ],
        search_results=[
            {"url": "https://blog.example.com/a", "sub_question": "what is the cost data"},
        ],
        facts=[
            _fact(
                "Global spending reached 200 billion dollars in 2025",
                "https://blog.example.com/a",
                "what is the cost data",
                is_primary=False,
            ),
        ],
        critique={
            "is_sufficient": False,
            "improved_queries": ["independent corroboration for the spending figure"],
            "reason": "single-source figure",
        },
    )
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# 1. High-impact uncorroborated claims
# ---------------------------------------------------------------------------

def test_high_impact_uncorroborated_expands_while_budget_remains():
    state = _state_with_uncorroborated_high_impact()
    checks = depth_controller.evaluate(state, _settings())
    assert checks["high_impact_uncorroborated_count"] == 1
    assert checks["high_impact_uncorroborated"][0]["impact"] > 0
    assert checks["ceiling_reached"] is False
    assert depth_controller.decide(state, _settings()) == "expand"


def test_high_impact_uncorroborated_finalizes_at_hard_wall():
    state = _state_with_uncorroborated_high_impact(iteration=4, max_iterations=4)
    checks = depth_controller.evaluate(state, _settings())
    assert checks["high_impact_uncorroborated_count"] == 1
    assert checks["ceiling_reached"] is True
    assert depth_controller.decide(state, _settings()) == "finalize"


def test_corroborated_high_impact_claim_does_not_force_expand():
    # The same quantitative claim, now with a second publisher: no gap.
    state = _state()
    checks = depth_controller.evaluate(state, _settings())
    assert checks["high_impact_uncorroborated_count"] == 0
    assert checks["needs_corroboration_count"] == 0
    assert checks["thin_dimensions"] == []
    assert checks["evidence_sufficient"] is True
    assert depth_controller.decide(state, _settings()) == "finalize"


# ---------------------------------------------------------------------------
# 2. Unresolved high-impact contradictions
# ---------------------------------------------------------------------------

SEVERE_UNRESOLVED = [
    {"claim_a": "a", "claim_b": "b", "kind": "numeric", "severity": 0.8, "resolved": False},
]
SEVERE_RESOLVED = [
    {
        "claim_a": "a",
        "claim_b": "b",
        "kind": "numeric",
        "severity": 0.8,
        "resolved": True,
        "resolution": "different period",
    },
]


def test_unresolved_severe_contradiction_forces_expand():
    state = _state(
        contradictions=SEVERE_UNRESOLVED,
        critique={
            "is_sufficient": False,
            "improved_queries": ["resolve the contested figure independently"],
            "reason": "conflict",
        },
    )
    checks = depth_controller.evaluate(state, _settings())
    assert checks["severe_contradictions"] == 1
    assert depth_controller.decide(state, _settings()) == "expand"
    assert "severe contradiction" in checks["decision_reason"]


def test_resolved_contradiction_does_not_force_expand():
    # A resolved contradiction is an explained spread, not a disagreement: with
    # a sufficient critic and no other gaps the run finalizes.
    state = _state(
        contradictions=SEVERE_RESOLVED,
        critique={"is_sufficient": True, "improved_queries": [], "reason": "ok"},
    )
    checks = depth_controller.evaluate(state, _settings())
    assert checks["severe_contradictions"] == 0
    assert checks["evidence_sufficient"] is True
    assert depth_controller.decide(state, _settings()) == "finalize"


def test_unresolved_contradiction_finalizes_when_nothing_novel_left():
    # Contradiction present but no novel query to run: stop, record as
    # limitation — never burn a pass re-finding the same pages.
    state = _state(
        contradictions=SEVERE_UNRESOLVED,
        critique={"is_sufficient": False, "improved_queries": [], "reason": "conflict"},
    )
    assert depth_controller.decide(state, _settings()) == "finalize"
    reason = depth_controller.explain(state, _settings())["reason"]
    assert "nothing novel" in reason.lower()


# ---------------------------------------------------------------------------
# 3. Thin dimensions
# ---------------------------------------------------------------------------

def test_thin_dimension_triggers_targeted_expand():
    # The mechanism dimension has no fact attributed at all -> thin.
    state = _state(
        facts=[
            _fact(
                "Global spending reached 200 billion dollars in 2025",
                "https://gov.uk/x",
                "what is the cost data",
                corroborating_sources=["https://gov.uk/x", "https://who.int/data"],
                corroboration_count=2,
            ),
        ],
        critique={
            "is_sufficient": False,
            "improved_queries": ["evidence on the mechanism"],
            "reason": "mechanism thin",
        },
    )
    checks = depth_controller.evaluate(state, _settings())
    assert "how does the mechanism work" in checks["thin_dimensions"]
    assert depth_controller.decide(state, _settings()) == "expand"
    assert "thin dimension" in checks["decision_reason"]


def test_primary_thin_dimension_triggers_expand():
    # Dimension has facts, but none primary -> primary share 0.0 <= threshold.
    state = _state(
        facts=[
            _fact(
                "Global spending reached 200 billion dollars in 2025",
                "https://gov.uk/x",
                "what is the cost data",
                corroborating_sources=["https://gov.uk/x", "https://who.int/data"],
                corroboration_count=2,
            ),
            _fact(
                "A secondary comment on how the mechanism works",
                "https://blog.example.com/m",
                "how does the mechanism work",
                is_primary=False,
            ),
        ],
        critique={
            "is_sufficient": False,
            "improved_queries": ["primary source on the mechanism"],
            "reason": "mechanism not primary",
        },
    )
    checks = depth_controller.evaluate(state, _settings())
    assert "how does the mechanism work" in checks["thin_dimensions"]
    assert depth_controller.decide(state, _settings()) == "expand"


def test_all_dimensions_covered_finalizes_even_with_iterations_remaining():
    state = _state(iteration=1, max_iterations=9)
    checks = depth_controller.evaluate(state, _settings())
    assert checks["thin_dimensions"] == []
    assert checks["ceiling_reached"] is False
    assert depth_controller.decide(state, _settings()) == "finalize"


# ---------------------------------------------------------------------------
# 4. No unnecessary expansion when sufficient
# ---------------------------------------------------------------------------

def test_sufficient_evidence_does_not_expand_from_bare_iterations():
    # Evidence is complete and the critic agrees there is nothing more to do:
    # remaining iterations alone must not trigger another pass.
    state = _state(
        iteration=2,
        max_iterations=9,
        critique={"is_sufficient": True, "improved_queries": [], "reason": "ok"},
    )
    checks = depth_controller.evaluate(state, _settings())
    assert checks["evidence_sufficient"] is True
    assert checks["ceiling_reached"] is False
    assert depth_controller.decide(state, _settings()) == "finalize"


def test_critic_novel_followups_still_expand_when_evidence_complete():
    # A model verdict IS actionable when it proposes genuinely new queries —
    # that is not "unnecessary" research, so the critic branch still expands.
    state = _state(
        iteration=2,
        max_iterations=9,
        critique={
            "is_sufficient": False,
            "improved_queries": ["independent evaluation of X versus Y"],
            "reason": "curiosity",
        },
    )
    assert depth_controller.evaluate(state, _settings())["evidence_sufficient"] is True
    assert depth_controller.decide(state, _settings()) == "expand"


# ---------------------------------------------------------------------------
# 5. Hard walls always win
# ---------------------------------------------------------------------------

def test_iteration_ceiling_beats_every_evidence_trigger():
    state = _state_with_uncorroborated_high_impact(
        iteration=4,
        max_iterations=4,
        contradictions=SEVERE_UNRESOLVED,
    )
    checks = depth_controller.evaluate(state, _settings())
    assert checks["high_impact_uncorroborated_count"] == 1
    assert checks["severe_contradictions"] == 1
    assert checks["ceiling_reached"] is True
    assert depth_controller.decide(state, _settings()) == "finalize"
    assert depth_controller.hard_wall_reached(state, _settings()) is True


def test_budget_wall_beats_every_evidence_trigger():
    state = _state_with_uncorroborated_high_impact()
    try:
        usage = start_run_usage("ad-budget", _settings(max_llm_calls=1))
        usage.budget.llm_calls = 60
        assert depth_controller.decide(state, _settings()) == "finalize"
        assert depth_controller.hard_wall_reached(state, _settings()) is True
    finally:
        clear_run_usage()


def test_hard_wall_does_not_hide_outstanding_gaps():
    # The report limitations must still name the gaps that were left behind.
    state = _state_with_uncorroborated_high_impact(iteration=4, max_iterations=4)
    reason = depth_controller.stop_reason(state, _settings())
    assert reason and "gap" in reason.lower()
    assert "uncorroborated" in reason.lower()


# ---------------------------------------------------------------------------
# 6. Explainability
# ---------------------------------------------------------------------------

def test_explain_reports_concrete_triggers():
    state = _state_with_uncorroborated_high_impact(
        contradictions=SEVERE_UNRESOLVED,
    )
    explained = depth_controller.explain(state, _settings())
    assert explained["decision"] == "expand"
    assert explained["high_impact_uncorroborated"] == 1
    assert explained["severe_contradictions"] == 1
    joined = "; ".join(explained["triggers"])
    assert "high-impact claim" in joined
    assert "severe contradiction" in joined


def test_decision_reason_is_stable_and_empty_when_sufficient():
    state = _state()
    checks = depth_controller.evaluate(state, _settings())
    assert checks["decision_reasons"] == []
    assert checks["decision_reason"] == "evidence sufficient"
    assert depth_controller.explain(state, _settings())["decision"] == "finalize"


def test_reasons_are_ordered_and_singular_plural_correct():
    state = _state_with_uncorroborated_high_impact(
        facts=[
            _fact(
                "Global spending reached 200 billion dollars in 2025",
                "https://blog.example.com/a",
                "what is the cost data",
                is_primary=False,
            ),
        ],
        contradictions=SEVERE_UNRESOLVED,
    )
    checks = depth_controller.evaluate(state, _settings())
    reasons = checks["decision_reasons"]
    # contradictions first, then high-impact claims.
    assert reasons[0].startswith("1 unresolved severe contradiction")
    assert "1 high-impact claim uncorroborated" in reasons


# ---------------------------------------------------------------------------
# 7. Backwards compatibility / robustness
# ---------------------------------------------------------------------------

def test_absent_inputs_are_neutral():
    # Legacy state without sub_question stamps or corroboration metrics must
    # not be flagged thin, and grading failure must not change routing.
    state = _state(
        sub_questions=[],
        facts=[
            {"claim": "the first fact about X", "source": "https://a.com/x", "verified": True},
        ],
    )
    checks = depth_controller.evaluate(state, _settings())
    assert checks["thin_dimensions"] == []
    assert "decision_reason" in checks


def test_thin_dimension_grading_failure_is_neutral(monkeypatch):
    import app.core.evidence_completion as ec

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(ec, "dimension_primary_share", _boom)
    state = _state()
    # Primary-share failure only removes the ADDITIONAL signal; the run still
    # evaluates and decides without raising.
    checks = depth_controller.evaluate(state, _settings())
    assert isinstance(checks["thin_dimensions"], list)


def test_checks_are_per_call_not_global():
    gap_state = _state_with_uncorroborated_high_impact()
    clean_state = _state()
    gap_checks = depth_controller.explain(gap_state, _settings())
    clean_checks = depth_controller.explain(clean_state, _settings())
    assert gap_checks["high_impact_uncorroborated"] == 1
    assert clean_checks["high_impact_uncorroborated"] == 0
    # Re-reading after the other call still returns the first state's answer.
    assert depth_controller.explain(gap_state, _settings())["high_impact_uncorroborated"] == 1


# ---------------------------------------------------------------------------
# 8. Investigation state wiring: exhausted gaps are limitations, open gaps
#    still drive expansion (app/core/investigation_state.py)
# ---------------------------------------------------------------------------

_HIGH_IMPACT_CLAIM = "Global spending reached 200 billion dollars in 2025"


def _inv_entry(claim, status, attempts=0, max_attempts=2):
    key = investigation_key(claim)
    return {
        key: {
            "claim": claim,
            "attempts": attempts,
            "queries": [f"q{i}" for i in range(attempts)],
            "status": status,
            "last_outcome": "still_single_source" if status != STATUS_OPEN else "",
            "max_attempts": max_attempts,
        }
    }


def _state_primary_high_impact(**overrides):
    """A single high-impact claim from a primary publisher (still single-source).

    Primary avoids the separate `thin_dimensions` signal so these cases isolate
    the investigation-state corroboration wiring under test.
    """
    state = _state_with_uncorroborated_high_impact(
        facts=[
            _fact(
                _HIGH_IMPACT_CLAIM,
                "https://blog.example.com/a",
                "what is the cost data",
                is_primary=True,
            ),
        ],
    )
    state.update(overrides)
    return state


def test_open_high_impact_gap_still_expands():
    # Un-attempted (open) high-impact gap: a real, actionable gap -> expand.
    state = _state_with_uncorroborated_high_impact(
        investigation_state=_inv_entry(_HIGH_IMPACT_CLAIM, STATUS_OPEN)
    )
    checks = depth_controller.evaluate(state, _settings())
    assert checks["active_high_impact_uncorroborated_count"] == 1
    assert checks["needs_corroboration_count"] >= 1
    assert depth_controller.decide(state, _settings()) == "expand"
    assert any("uncorroborated" in r for r in checks["decision_reasons"])


def test_attempted_with_budget_remaining_still_expands():
    # Attempted once, budget (2) not yet spent -> still actionable -> expand.
    state = _state_with_uncorroborated_high_impact(
        investigation_state=_inv_entry(_HIGH_IMPACT_CLAIM, STATUS_ATTEMPTED, attempts=1)
    )
    checks = depth_controller.evaluate(state, _settings())
    assert checks["active_high_impact_uncorroborated_count"] == 1
    assert depth_controller.decide(state, _settings()) == "expand"


def test_exhausted_high_impact_gap_does_not_force_expand():
    # Exhausted gap is an acknowledged limitation, NOT a continue-reason.
    state = _state_primary_high_impact(
        investigation_state=_inv_entry(_HIGH_IMPACT_CLAIM, STATUS_EXHAUSTED, attempts=2),
        critique={"is_sufficient": True, "improved_queries": [], "reason": "ok"},
    )
    checks = depth_controller.evaluate(state, _settings())
    # Raw count still reports the single-source claim (backwards compat).
    assert checks["needs_corroboration_count"] >= 1
    assert checks["high_impact_uncorroborated_count"] == 1
    # ... but the ACTIVE count excludes it.
    assert checks["exhausted_gap_count"] >= 1
    assert checks["active_high_impact_uncorroborated_count"] == 0
    assert all("uncorroborated" not in r for r in checks["decision_reasons"])
    assert depth_controller.decide(state, _settings()) == "finalize"


def test_exhausted_gap_with_sufficient_evidence_finalizes():
    # Only remaining gap is exhausted and the evidence is otherwise sufficient:
    # a genuine sufficiency STOP, naming the exhausted gap as a limitation.
    state = _state_primary_high_impact(
        investigation_state=_inv_entry(_HIGH_IMPACT_CLAIM, STATUS_EXHAUSTED, attempts=2),
        confidence=0.9,
        confidence_history=[0.8, 0.9],
        critique={"is_sufficient": True, "improved_queries": [], "reason": "ok"},
    )
    checks = depth_controller.evaluate(state, _settings())
    assert checks["evidence_sufficient"] is True
    assert checks["active_high_impact_uncorroborated_count"] == 0
    decision, checks2 = depth_controller.decide_with_checks(state, _settings())
    assert decision == "finalize"
    assert "sufficient" in checks2["decision_reason"].lower()
    assert "acknowledged limitation" in checks2["decision_reason"].lower()


def test_hard_wall_wins_even_with_open_gaps():
    # An OPEN high-impact gap cannot defeat the iteration ceiling.
    state = _state_with_uncorroborated_high_impact(
        investigation_state=_inv_entry(_HIGH_IMPACT_CLAIM, STATUS_OPEN),
        iteration=4,
        max_iterations=4,
    )
    checks = depth_controller.evaluate(state, _settings())
    assert checks["active_high_impact_uncorroborated_count"] == 1
    assert checks["ceiling_reached"] is True
    assert depth_controller.decide(state, _settings()) == "finalize"


def test_exhausted_gaps_are_not_continue_reasons_but_open_are():
    exhausted = _state_with_uncorroborated_high_impact(
        investigation_state=_inv_entry(_HIGH_IMPACT_CLAIM, STATUS_EXHAUSTED, attempts=2)
    )
    open_state = _state_with_uncorroborated_high_impact(
        investigation_state=_inv_entry(_HIGH_IMPACT_CLAIM, STATUS_OPEN)
    )
    exhausted_reasons = depth_controller.evaluate(exhausted, _settings())["decision_reasons"]
    open_reasons = depth_controller.evaluate(open_state, _settings())["decision_reasons"]
    assert not any("uncorroborated" in r for r in exhausted_reasons)
    assert any("uncorroborated" in r for r in open_reasons)


def test_malformed_investigation_state_is_neutral():
    # A garbage investigation state must never raise or change routing.
    state = _state_with_uncorroborated_high_impact(investigation_state="not-a-dict")
    checks = depth_controller.evaluate(state, _settings())
    assert checks["exhausted_gap_count"] == 0
    assert checks["active_high_impact_uncorroborated_count"] == 1
    assert depth_controller.decide(state, _settings()) == "expand"
