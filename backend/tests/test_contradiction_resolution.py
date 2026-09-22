"""Fix C — contradiction resolution pass.

A live deep run reported 5 contradictions, all kind="temporal", and nothing
resolved them: they kept penalizing confidence and driving expansion even
though the spread was fully explained by the period. These tests pin the rule:

* same unit+scope+period+metric but different values -> unresolved, still
  penalizes and still blocks;
* differing only by period/scope/metric -> resolved=true with an explanation
  and NOT counted as an unresolved conflict.
"""
from app.core.confidence import compute_confidence
from app.core.contradiction_resolution import (
    resolve_contradiction,
    resolve_contradictions,
    unresolved_contradictions,
)
from app.core.contradictions import find_contradictions
from app.core.depth_controller import _severe_contradictions
from app.core.evidence_grade import grade_claim


def _facts(*claims_and_sources):
    return [
        {"claim": claim, "source": source, "verified": True}
        for claim, source in claims_and_sources
    ]


# ---------------------------------------------------------------------------
# 1. genuine conflict stays unresolved
# ---------------------------------------------------------------------------

def test_same_unit_scope_period_different_values_unresolved():
    facts = _facts(
        ("The market grew by 25% globally last year", "https://a.com/x"),
        ("The market grew by 11% globally last year", "https://b.com/y"),
    )
    found = find_contradictions(facts)
    assert found and found[0]["kind"] == "numeric"
    resolved = resolve_contradictions(found)
    assert resolved[0]["resolved"] is False
    assert resolved[0]["resolution"]
    # It remains a counted unresolved conflict.
    assert unresolved_contradictions(resolved) == [resolved[0]]
    assert _severe_contradictions({"contradictions": resolved}) >= 1


def test_resolved_numeric_conflict_does_not_penalize_confidence():
    facts = _facts(
        ("The market grew by 25% globally last year", "https://a.com/x"),
        ("The market grew by 11% globally last year", "https://b.com/y"),
    )
    found = resolve_contradictions(find_contradictions(facts))
    critique = {"is_sufficient": True, "reason": "ok"}
    with_conflict = compute_confidence(
        facts=facts, critique=critique, iteration=2, max_iterations=4,
        contradictions=found,
    )
    # unresolved -> penalty applies, score no greater than the no-conflict run
    baseline = compute_confidence(
        facts=facts, critique=critique, iteration=2, max_iterations=4,
        contradictions=[],
    )
    assert with_conflict["overall"] <= baseline["overall"]


# ---------------------------------------------------------------------------
# 2. period / scope / metric differences resolve
# ---------------------------------------------------------------------------

def test_different_period_resolves_with_explanation():
    facts = _facts(
        ("Global capacity reached 1200 GW in 2023", "https://a.com/x"),
        ("Global capacity reached 1600 GW in 2024", "https://b.com/y"),
    )
    found = find_contradictions(facts)
    assert found and found[0]["kind"] == "temporal"
    resolved = resolve_contradictions(found)
    assert resolved[0]["resolved"] is True
    assert "period" in resolved[0]["resolution"].lower()
    assert unresolved_contradictions(resolved) == []
    assert _severe_contradictions({"contradictions": resolved}) == 0


def test_temporal_resolved_does_not_penalize_confidence():
    facts = _facts(
        ("Global capacity reached 1200 GW in 2023", "https://a.com/x"),
        ("Global capacity reached 1600 GW in 2024", "https://b.com/y"),
    )
    resolved = resolve_contradictions(find_contradictions(facts))
    critique = {"is_sufficient": True, "reason": "ok"}
    with_conflict = compute_confidence(
        facts=facts, critique=critique, iteration=2, max_iterations=4,
        contradictions=resolved,
    )
    baseline = compute_confidence(
        facts=facts, critique=critique, iteration=2, max_iterations=4,
        contradictions=[],
    )
    assert with_conflict["overall"] == baseline["overall"]


def test_different_scope_resolves():
    facts = _facts(
        ("Carbon emissions fell 30% globally across the power sector", "https://a.com/x"),
        ("Carbon emissions fell 5% in the United States across the power sector", "https://b.com/y"),
    )
    found = resolve_contradictions(find_contradictions(facts))
    assert found and found[0]["resolved"] is True
    assert "scope" in found[0]["resolution"].lower()
    assert unresolved_contradictions(found) == []


def test_grade_ignores_resolved_contradiction():
    claim = "Global capacity reached 1200 GW in 2023"
    fact = {"claim": claim, "source": "https://a.com/x", "verified": True}
    unresolved = [{
        "claim_a": claim, "claim_b": "Global capacity reached 1600 GW in 2024",
        "source_a": "https://a.com/x", "source_b": "https://b.com/y",
        "kind": "numeric", "severity": 0.8, "resolved": False,
    }]
    resolved = [{
        "claim_a": claim, "claim_b": "Global capacity reached 1600 GW in 2024",
        "source_a": "https://a.com/x", "source_b": "https://b.com/y",
        "kind": "temporal", "severity": 0.4, "resolved": True,
    }]
    assert grade_claim(fact, contradictions=unresolved).contradiction_count == 1
    assert grade_claim(fact, contradictions=resolved).contradiction_count == 0


def test_resolve_contradiction_is_total_on_malformed_input():
    assert resolve_contradiction(None)["resolved"] is False  # type: ignore[arg-type]
    assert resolve_contradiction({"kind": "mystery"})["resolved"] is False


# ---------------------------------------------------------------------------
# 2b. evidence-semantics guards (labeled-fixture regressions)
# ---------------------------------------------------------------------------

def test_period_does_not_explain_opposite_polarity():
    """A differing year cannot reconcile "fell" with "rose". The period branch
    must not auto-resolve a polarity conflict — that was a false 'resolved'."""
    facts = _facts(
        ("Coal capacity fell 12% in 2020 across the region", "https://a.example/x"),
        ("Coal capacity rose 12% in 2023 across the region", "https://b.example/y"),
    )
    found = find_contradictions(facts)
    assert found and found[0]["kind"] == "polarity"
    resolved = resolve_contradictions(found)
    assert resolved[0]["resolved"] is False
    assert unresolved_contradictions(resolved) == [resolved[0]]


def test_incidental_context_year_is_not_a_temporal_conflict():
    """Two claims that agree on the measure and only differ by an extra
    historical year are not a period conflict. `numeric_conflict` keeps the
    last value per unit, so it sees 5 GW vs 26.5 GW while missing the agreeing
    26.5 GW — the shared agreeing quantity is the guard."""
    facts = _facts(
        ("Bangladesh capacity reached 26.5 GW in 2024, up from 5 GW in 2009",
         "https://a.example/x"),
        ("Bangladesh capacity was 26.5 GW in 2024", "https://b.example/y"),
    )
    assert find_contradictions(facts) == []


def test_different_population_segments_are_a_scope_note_not_a_conflict():
    """Urban vs rural adults are different populations: the engine records a
    scope spread (resolved=true) and never counts it as an unresolved
    contradiction."""
    facts = _facts(
        ("The literacy rate among urban adults was 82% in 2020", "https://a.example/x"),
        ("The literacy rate among rural adults was 61% in 2020", "https://b.example/y"),
    )
    found = resolve_contradictions(find_contradictions(facts))
    assert found and found[0]["kind"] == "scope"
    assert found[0]["resolved"] is True
    assert "scope" in found[0]["resolution"].lower()
    assert unresolved_contradictions(found) == []


def test_primary_vs_secondary_conflict_stays_unresolved():
    """A primary journal and a secondary blog disagreeing on the same measure,
    scope and period is a genuine conflict — the primary/secondary weighting
    must not silently mark it resolved."""
    facts = _facts(
        ("The trial reported a mortality reduction of 15% versus placebo",
         "https://nejm.org/a"),
        ("The trial reported a mortality reduction of 40% versus placebo",
         "https://someblog.example/b"),
    )
    resolved = resolve_contradictions(find_contradictions(facts))
    assert resolved and resolved[0]["resolved"] is False
    assert "unresolved" in resolved[0]["resolution"].lower()


def test_unresolved_genuine_conflict_still_penalizes_confidence():
    """The whole point of keeping a genuine conflict unresolved: it must keep
    penalizing confidence (never be waved away by the guard changes)."""
    facts = _facts(
        ("The trial reported a mortality reduction of 15% versus placebo",
         "https://nejm.org/a"),
        ("The trial reported a mortality reduction of 40% versus placebo",
         "https://someblog.example/b"),
    )
    resolved = resolve_contradictions(find_contradictions(facts))
    critique = {"is_sufficient": True, "reason": "ok"}
    with_conflict = compute_confidence(
        facts=facts, critique=critique, iteration=2, max_iterations=4,
        contradictions=resolved,
    )
    baseline = compute_confidence(
        facts=facts, critique=critique, iteration=2, max_iterations=4,
        contradictions=[],
    )
    assert with_conflict["overall"] < baseline["overall"]
    assert _severe_contradictions({"contradictions": resolved}) >= 1


def test_unresolved_conflict_reaches_synthesis_counterarguments():
    """An unresolved conflict must appear in the report's Counterarguments
    section (not silently dropped)."""
    from app.agents.synthesizer import _add_required_sections, select_profile

    facts = [
        {"claim": "The trial reported a mortality reduction of 15% versus placebo",
         "source": "https://nejm.org/a", "verified": True},
        {"claim": "The trial reported a mortality reduction of 40% versus placebo",
         "source": "https://someblog.example/b", "verified": True},
    ]
    resolved = resolve_contradictions(find_contradictions(facts))
    assert unresolved_contradictions(resolved)
    ctx = {"contradictions": resolved}
    report, _ = _add_required_sections(
        "## Executive Summary\n\nA short draft.",
        ctx=ctx,
        usable_facts=facts,
        contradictions=resolved,
        profile=select_profile(ctx, fact_count=len(facts)),
    )
    counter = report.lower()
    assert "counterarguments" in counter
    assert "mortality reduction" in counter



# ---------------------------------------------------------------------------
# 3. deep-mode depth scaling (Fix B.1)
# ---------------------------------------------------------------------------

def test_deep_mode_iterations_scale_with_map_and_stay_bounded():
    from app.agents.orchestrator import (
        MODE_PRESETS,
        DEEP_MODE_ITERATION_FLOOR,
        scaled_max_iterations,
    )

    # A 12-contract map needs at least 6 passes (one per 2 contracts).
    assert scaled_max_iterations("deep", 12) == 6
    assert scaled_max_iterations("deep", 20) == 10
    # Bounded: never below the preset floor.
    assert scaled_max_iterations("deep", 1) == DEEP_MODE_ITERATION_FLOOR
    # Unknown mode falls back to the preset ceiling, no scaling.
    assert scaled_max_iterations("unknown", 12) == MODE_PRESETS["standard"]["max_iterations"]


def test_quick_and_standard_iterations_unchanged():
    from app.agents.orchestrator import MODE_PRESETS, scaled_max_iterations

    assert scaled_max_iterations("quick", 12) == MODE_PRESETS["quick"]["max_iterations"]
    assert scaled_max_iterations("standard", 12) == MODE_PRESETS["standard"]["max_iterations"]


def test_build_initial_state_scales_deep_and_scales_graph_limit():
    from app.graph.workflow import build_initial_state, graph_recursion_limit

    state = build_initial_state("What is the current trend of AI?", max_iterations=5, mode="deep")
    target_agents = int(state["orchestration"]["target_agents"])
    import math

    expected = max(5, math.ceil(target_agents / 2))
    assert state["max_iterations"] == expected
    # The recursion budget scales with the raised ceiling, so a deep run still
    # terminates via the routing ceiling rather than GraphRecursionError.
    assert graph_recursion_limit(state) >= 8 + 5 * expected
