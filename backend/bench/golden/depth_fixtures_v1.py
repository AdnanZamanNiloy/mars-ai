"""Golden adaptive-depth routing fixtures (v1).

The main golden set (queries_v1.json) drives the FULL production graph with
scripted offline mocks, but those mocks are arranged so the run finalizes at
the iteration ceiling — so none of the adaptive-depth BRANCHES in
`app.core.depth_controller` are regression-covered by the golden gate.

This module supplies deterministic ResearchState fixtures that exercise each
adaptive-depth branch directly, and declares the EXPECTED routing outcome for
each. The states are built with the same shape the depth controller reads in
production (verified facts with `sub_question` stamps, `is_primary`,
`corroborating_sources`, planned `sub_questions` with `minimum_sources`,
critique follow-ups, contradictions). Nothing here fakes a metric or inflates
a score: the states are minimal, honest inputs that isolate the routing input.

Why fixture states (approach b) rather than driving the real graph (approach a):
the offline graph mocks return evidence for every planned contract, so a
naturally-produced "thin dimension" or "single-publisher high-impact claim"
state cannot be constructed without weakening the mocks (which would fake the
pipeline, not test it). The routing decision is therefore exercised at the
exact seam `workflow.route_after_critic` calls — `depth_controller.decide_with_checks`
(after its `hard_wall_reached` / evidence-gap pre-checks) — which is
end-to-end for the ROUTING unit while staying deterministic and honest.

Each scenario dict:
    id            stable scenario id
    description   human-readable trigger
    state         the ResearchState fixture
    expect        expected_decision ("expand" | "finalize"),
                  reason_tokens (substrings that must appear in the reason),
                  expect_limitations (bool; hard-wall limitations recorded)
"""

from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Fact / state builders (mirror tests/test_adaptive_depth.py shapes)
# ---------------------------------------------------------------------------


def _fact(claim: str, source: str, dim: str, **kw: Any) -> Dict[str, Any]:
    base = {
        "claim": claim,
        "source": source,
        "verified": True,
        "sub_question": dim,
        "is_primary": True,
    }
    base.update(kw)
    return base


def _base_state(**overrides: Any) -> Dict[str, Any]:
    """Two planned dimensions, both covered by primary corroborated facts.

    This is the KNOWN-SUFFICIENT reference state: no thin dimension, no
    uncorroborated high-impact claim, no severe contradiction, confidence at
    target — the STOP side of the routing decision.
    """
    base: Dict[str, Any] = {
        "query": "cost and mechanism of X",
        "iteration": 1,
        "max_iterations": 4,
        "confidence": 0.9,
        "confidence_history": [0.8, 0.9],
        "mode": "standard",
        "critique": {"is_sufficient": True, "improved_queries": [], "reason": "ok"},
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


# ---------------------------------------------------------------------------
# Scenario builders — one per adaptive-depth branch
# ---------------------------------------------------------------------------


def scenario_high_impact_uncorroborated() -> Dict[str, Any]:
    """BRANCH 1 (expand): a high-impact quantitative claim resting on one
    publisher, with a novel follow-up available to corroborate it."""
    state = _base_state(
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
    return {
        "id": "depth-high-impact-uncorroborated",
        "description": "high-impact quantitative claim on a single publisher -> expand",
        "state": state,
        "expect": {
            "expected_decision": "expand",
            "reason_tokens": ["high-impact claim", "uncorroborated"],
            "expect_limitations": False,
        },
    }


def scenario_thin_dimension() -> Dict[str, Any]:
    """BRANCH 2 (expand): a planned dimension that HAS a fact but is
    primary-source thin (no primary publisher), with a novel follow-up that
    targets it.

    Deliberately not the "zero facts" case: an axis with zero verified facts
    is caught earlier by the `uncovered_axes` branch (a distinct, simpler
    gate). This fixture isolates the primary-thin signal that only the
    `thin_dimensions` branch detects."""
    state = _base_state(
        search_results=[
            {"url": "https://gov.uk/x", "sub_question": "what is the cost data"},
            {"url": "https://blog.example.com/m", "sub_question": "how does the mechanism work"},
        ],
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
    return {
        "id": "depth-thin-dimension",
        "description": "planned dimension with no attributed fact -> expand",
        "state": state,
        "expect": {
            "expected_decision": "expand",
            "reason_tokens": ["thin dimension"],
            "expect_limitations": False,
        },
    }


SEVERE_UNRESOLVED = [
    {"claim_a": "a", "claim_b": "b", "kind": "numeric", "severity": 0.8, "resolved": False},
]


def scenario_unresolved_severe_contradiction() -> Dict[str, Any]:
    """BRANCH 3 (expand): an unresolved severe numeric contradiction with a
    novel follow-up available to resolve it."""
    state = _base_state(
        contradictions=list(SEVERE_UNRESOLVED),
        critique={
            "is_sufficient": False,
            "improved_queries": ["resolve the contested figure independently"],
            "reason": "conflict",
        },
    )
    return {
        "id": "depth-unresolved-severe-contradiction",
        "description": "unresolved severe numeric contradiction -> expand",
        "state": state,
        "expect": {
            "expected_decision": "expand",
            "reason_tokens": ["severe contradiction"],
            "expect_limitations": False,
        },
    }


def scenario_sufficient() -> Dict[str, Any]:
    """BRANCH 4 (finalize): evidence is complete and the critic agrees — no
    important gap remains and iterations are available, so STOP."""
    state = _base_state(
        iteration=2,
        max_iterations=9,
        critique={"is_sufficient": True, "improved_queries": [], "reason": "ok"},
    )
    return {
        "id": "depth-sufficient",
        "description": "sufficient evidence, no important gaps -> finalize",
        "state": state,
        "expect": {
            "expected_decision": "finalize",
            "reason_tokens": [],  # sufficient path: "evidence sufficient"
            "expect_limitations": False,
        },
    }


def scenario_hard_wall() -> Dict[str, Any]:
    """BRANCH 5 (finalize + limitations): the iteration ceiling is reached
    while a high-impact claim is still uncorroborated. The hard wall must win,
    and `stop_reason` must name the outstanding gap so it is recorded as a
    limitation rather than silently dropped."""
    state = _base_state(
        iteration=4,
        max_iterations=4,
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
    return {
        "id": "depth-hard-wall",
        "description": "iteration ceiling reached with gaps remaining -> finalize + limitations",
        "state": state,
        "expect": {
            "expected_decision": "finalize",
            "reason_tokens": ["hard wall", "ceiling"],
            "expect_limitations": True,
        },
    }


def all_scenarios() -> List[Dict[str, Any]]:
    return [
        scenario_high_impact_uncorroborated(),
        scenario_thin_dimension(),
        scenario_unresolved_severe_contradiction(),
        scenario_sufficient(),
        scenario_hard_wall(),
    ]
