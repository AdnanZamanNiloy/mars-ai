"""Per-claim investigation state — closing the adaptive-investigation loop.

The live baseline: `needs_corroboration` stuck at 20-50 claims per query and
every run hitting the iteration ceiling still "not sufficient". MARS detected
and targeted gaps but never tracked whether a targeted attempt WORKED, so
budget could be re-spent on the same dead-end gap and an exhausted gap could
not be told apart from an un-attempted one.

These tests pin the outcome memory:
  * a needs_corroboration claim is recorded as open;
  * a targeted query for it increments attempts;
  * once a new publisher corroborates it, it leaves the open set;
  * after N failed attempts it is exhausted and NOT re-queried;
  * exhausted claims surface as acknowledged limitations;
  * un-attempted gaps are prioritized over already-attempted ones;
  * state is run-scoped (a fresh dict per run) and total on garbage input.

All deterministic and LLM-free.
"""
from app.core.investigation_state import (
    STATUS_ATTEMPTED,
    STATUS_CORROBORATED,
    STATUS_EXHAUSTED,
    STATUS_OPEN,
    DEFAULT_MAX_ATTEMPTS,
    exhausted_limitations,
    investigation_key,
    investigation_summary,
    open_targets,
    reconcile_outcomes,
    record_attempts,
    sanitize_investigation_state,
)

_CLAIM_QUANT = "Enterprise adoption of retrieval augmented generation reached 62% of companies in 2024"
_CLAIM_DEF = "Retrieval augmented generation is a technique that grounds language model outputs"
_UNRELATED = "Solar capacity in Bangladesh grew 40% last year"


def _fact(claim, source="https://blog.example.com/report", **kw):
    base = {"claim": claim, "source": source, "verified": True}
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# 1. a needs_corroboration claim is recorded as open
# ---------------------------------------------------------------------------

def test_needs_corroboration_claim_recorded_as_open():
    targets = [{"claim": _CLAIM_QUANT, "impact": 8}]
    state = record_attempts(None, targets, queries_by_claim={})
    key = investigation_key(_CLAIM_QUANT)
    assert key in state
    assert state[key]["status"] == STATUS_OPEN
    assert state[key]["attempts"] == 0
    assert investigation_summary(state)[STATUS_OPEN] == 1


def test_targeted_query_increments_attempts():
    key = investigation_key(_CLAIM_QUANT)
    targets = [{"claim": _CLAIM_QUANT}]
    state = record_attempts(
        None, targets, queries_by_claim={key: ["query a"]}, max_attempts=2
    )
    assert state[key]["attempts"] == 1
    assert state[key]["status"] == STATUS_ATTEMPTED


def test_duplicate_query_text_does_not_double_count():
    key = investigation_key(_CLAIM_QUANT)
    state = record_attempts(
        None, [{"claim": _CLAIM_QUANT}], queries_by_claim={key: ["q1"]}, max_attempts=3
    )
    state = record_attempts(
        state, [{"claim": _CLAIM_QUANT}], queries_by_claim={key: ["q1"]}, max_attempts=3
    )
    assert state[key]["attempts"] == 1
    assert state[key]["queries"] == ["q1"]


# ---------------------------------------------------------------------------
# 2. success removes the claim from the open set
# ---------------------------------------------------------------------------

def test_corroborated_claim_leaves_open_set():
    key = investigation_key(_CLAIM_QUANT)
    state = record_attempts(
        None, [{"claim": _CLAIM_QUANT}], queries_by_claim={key: ["q1"]}, max_attempts=3
    )
    # A second publisher landed: the fact no longer needs corroboration.
    state = reconcile_outcomes(state, [_fact(_CLAIM_QUANT, corroborating_sources=["https://oecd.org/x"], corroboration_count=2)], max_attempts=3)
    assert state[key]["status"] == STATUS_CORROBORATED
    summary = investigation_summary(state)
    assert summary[STATUS_CORROBORATED] == 1
    # And it is excluded from what a new pass may fund.
    actionable, excluded = open_targets(state, [{"claim": _CLAIM_QUANT}])
    assert actionable == []
    assert excluded == [{"claim": _CLAIM_QUANT}]


# ---------------------------------------------------------------------------
# 3. N failed attempts -> exhausted, not re-queried, surfaced as limitation
# ---------------------------------------------------------------------------

def test_claim_exhausts_after_n_failed_attempts():
    key = investigation_key(_CLAIM_QUANT)
    fact = _fact(_CLAIM_QUANT)
    max_attempts = 2
    state = record_attempts(None, [{"claim": _CLAIM_QUANT}], queries_by_claim={key: ["q1"]}, max_attempts=max_attempts)
    state = reconcile_outcomes(state, [fact], max_attempts=max_attempts)
    assert state[key]["status"] == STATUS_ATTEMPTED
    state = record_attempts(state, [{"claim": _CLAIM_QUANT}], queries_by_claim={key: ["q2"]}, max_attempts=max_attempts)
    state = reconcile_outcomes(state, [fact], max_attempts=max_attempts)
    assert state[key]["status"] == STATUS_EXHAUSTED
    assert state[key]["attempts"] == 2


def test_exhausted_claim_not_re_queried_by_open_targets():
    key = investigation_key(_CLAIM_QUANT)
    state = {
        key: {"claim": _CLAIM_QUANT, "attempts": 2, "queries": ["q1", "q2"],
              "status": STATUS_EXHAUSTED, "last_outcome": "still_single_source",
              "max_attempts": 2}
    }
    actionable, excluded = open_targets(state, [{"claim": _CLAIM_QUANT}])
    assert actionable == [], "an exhausted claim must not be re-queried"
    assert excluded == [{"claim": _CLAIM_QUANT}]


def test_exhausted_claims_surface_as_limitations():
    key = investigation_key(_CLAIM_QUANT)
    state = {
        key: {"claim": _CLAIM_QUANT, "attempts": 2, "queries": ["q1", "q2"],
              "status": STATUS_EXHAUSTED, "last_outcome": "still_single_source",
              "max_attempts": 2}
    }
    lines = exhausted_limitations(state)
    assert len(lines) == 1
    assert "62%" in lines[0]
    assert "acknowledged limitation" in lines[0]
    # A non-exhausted claim is never reported as a limitation.
    open_state = record_attempts(None, [{"claim": _CLAIM_DEF}], max_attempts=2)
    assert exhausted_limitations(open_state) == []


# ---------------------------------------------------------------------------
# 4. un-attempted gaps are prioritized over already-attempted ones
# ---------------------------------------------------------------------------

def test_unattempted_gaps_prioritized_over_attempted_ones():
    attempted_key = investigation_key(_CLAIM_DEF)
    state = record_attempts(
        None,
        [{"claim": _CLAIM_DEF}],
        queries_by_claim={attempted_key: ["q1"]},
        max_attempts=3,
    )
    # Impact order puts the already-attempted claim first; open_targets must
    # still fund the un-attempted one first.
    targets = [{"claim": _CLAIM_DEF}, {"claim": _CLAIM_QUANT}]
    actionable, _ = open_targets(state, targets)
    assert actionable[0]["claim"] == _CLAIM_QUANT
    assert actionable[1]["claim"] == _CLAIM_DEF


# ---------------------------------------------------------------------------
# 5. state is run-scoped + total on garbage input
# ---------------------------------------------------------------------------

def test_state_is_run_scoped_new_dict_per_attempt_record():
    base = record_attempts(None, [{"claim": _CLAIM_QUANT}], queries_by_claim={}, max_attempts=2)
    mutated = record_attempts(
        base, [{"claim": _CLAIM_QUANT}],
        queries_by_claim={investigation_key(_CLAIM_QUANT): ["q1"]},
        max_attempts=2,
    )
    # The input state passed to record_attempts is never mutated in place.
    assert base[investigation_key(_CLAIM_QUANT)]["attempts"] == 0
    assert mutated[investigation_key(_CLAIM_QUANT)]["attempts"] == 1


def test_empty_and_garbage_inputs_are_total():
    assert sanitize_investigation_state(None) == {}
    assert sanitize_investigation_state("garbage") == {}
    assert sanitize_investigation_state([1, 2]) == {}
    assert sanitize_investigation_state({"k": "not-a-dict"})["k"]["status"] == STATUS_OPEN
    assert investigation_summary(None) == {
        STATUS_OPEN: 0, STATUS_ATTEMPTED: 0, STATUS_EXHAUSTED: 0, STATUS_CORROBORATED: 0,
    }
    assert open_targets(None, []) == ([], [])
    assert open_targets(None, [{"no_claim": 1}]) == ([], [])
    assert record_attempts(None, [None, 3, {"claim": ""}], queries_by_claim=None) == {}
    assert reconcile_outcomes(None, None) == {}
    assert exhausted_limitations(None) == []
    assert investigation_key("") == ""
    # Grading is never asked to raise on an ungradeable pool.
    assert reconcile_outcomes(
        {investigation_key(_CLAIM_QUANT): {"claim": _CLAIM_QUANT, "attempts": 1}},
        [{"no_claim": True}],
    ) is not None


def test_default_max_attempts_matches_settings_default():
    from app.core.config import Settings

    settings = Settings(groq_api_key="k", _env_file=None)
    assert DEFAULT_MAX_ATTEMPTS == settings.max_corroboration_attempts


# ---------------------------------------------------------------------------
# 6. end-to-end: workflow surfaces the investigation state + limitations
# ---------------------------------------------------------------------------

def test_workflow_surfaces_investigation_limitations():
    from app.graph.workflow import _measured_coverage_gaps

    key = investigation_key(_CLAIM_QUANT)
    state = {
        "query": _CLAIM_QUANT,
        "facts": [_fact(_CLAIM_QUANT)],
        "contradictions": [],
        "investigation_state": {
            key: {"claim": _CLAIM_QUANT, "attempts": 2, "queries": ["q1", "q2"],
                  "status": STATUS_EXHAUSTED, "last_outcome": "still_single_source",
                  "max_attempts": 2}
        },
    }
    gaps = _measured_coverage_gaps(state)
    joined = " ".join(gaps).lower()
    assert "acknowledged limitation" in joined
