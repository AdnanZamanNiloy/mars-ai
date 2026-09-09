"""Scenario Engine (4.3): scoping, novelty, and verdict logic.

The scripts/run_scenarios.py driver is I/O over live APIs and is verified
live; pytest covers the pure comparison core here only.
"""

from app.core.scenarios import (
    build_scenario_query,
    compare_scenarios,
    scenario_new_claims,
    scope_questions,
)


def test_scope_questions_names_assumption():
    scoped = scope_questions("prices fall 50%", ["What does X cost?", "How fast is Y?"])
    assert len(scoped) == 2
    assert all("prices fall 50%" in q for q in scoped)
    assert "What does X cost?" in scoped[0]


def test_scope_questions_skips_blanks_and_caps():
    scoped = scope_questions("a", ["", "  ", "Q1", "Q2", "Q3", "Q4", "Q5"], limit=3)
    assert scoped == ["Given that a, Q1", "Given that a, Q2", "Given that a, Q3"]


def test_build_scenario_query_embeds_all_parts():
    q = build_scenario_query("Should we buy?", "prices fall", ["Given that prices fall, what?"])
    assert "prices fall" in q and "Should we buy?" in q and "(1)" in q


def test_scenario_new_claims_separates_shared_from_novel():
    base = [
        {"claim": "Solar panels convert sunlight into electricity efficiently"},
        {"claim": "Battery storage costs declined over the last decade"},
    ]
    scen = [
        {"claim": "Solar panels convert sunlight to electricity"},  # restatement
        {"claim": "Under 50% cheaper GPUs, cloud training dominates budgets"},  # novel
        {"claim": "ok"},  # too short to judge — dropped
    ]
    new = scenario_new_claims(scen, base)
    assert len(new) == 1
    assert "cloud training" in new[0]["claim"]


def test_compare_robust_verdict():
    comparison = compare_scenarios(
        {"recommended_option": "A"},
        [
            {"id": "s1", "assumption": "a1", "confidence": 0.7,
             "claims": [{}, {}], "new_claims": [{}], "recommended_option": "A"},
            {"id": "s2", "assumption": "a2", "confidence": 0.5,
             "claims": [{}], "new_claims": [{}], "recommended_option": "A"},
        ],
    )
    assert comparison["rows"][0]["total_claims"] == 2
    assert comparison["rows"][0]["new_claims"] == 1
    assert "robust across scenarios" in comparison["verdict"]
    assert "Option A" in comparison["verdict"]


def test_compare_flipped_verdict():
    comparison = compare_scenarios(
        {"recommended_option": "A"},
        [
            {"id": "s1", "assumption": "a1", "confidence": 0.7,
             "claims": [], "new_claims": [], "recommended_option": "A"},
            {"id": "s2", "assumption": "a2", "confidence": 0.6,
             "claims": [], "new_claims": [], "recommended_option": "B"},
        ],
    )
    assert "flips" in comparison["verdict"]
    assert "s1" in comparison["verdict"] and "s2" in comparison["verdict"]


def test_compare_no_recommendations():
    comparison = compare_scenarios(
        {"recommended_option": None},
        [{"id": "s1", "assumption": "a1", "confidence": 0.7,
          "claims": [], "new_claims": [], "recommended_option": None}],
    )
    assert "nothing to compare" in comparison["verdict"]


def test_build_scenario_query_fits_api_limit():
    long_qs = [f"Given that X, {'word ' * 40}{i}" for i in range(4)]
    q = build_scenario_query("Should we buy?", "prices fall", long_qs, max_len=500)
    assert len(q) <= 500
    assert "prices fall" in q and "Should we buy?" in q
    assert "(1)" in q  # at least one scoped question survives


def test_build_scenario_query_head_survives_overflow():
    q = build_scenario_query("B?" * 400, "a", ["Q1"], max_len=500)
    assert len(q) <= 500
    assert "Under the assumption that a" in q


def test_compare_robust_notes_base_agreement_and_divergence():
    scen = lambda opt: {"id": "s", "assumption": "a", "confidence": 0.7,
                        "claims": [], "new_claims": [], "recommended_option": opt}
    agree = compare_scenarios({"recommended_option": "A"}, [scen("A"), scen("A")])
    assert "base run agreed" in agree["verdict"]
    split = compare_scenarios({"recommended_option": "A"}, [scen("B"), scen("B")])
    assert "robust across scenarios" in split["verdict"]
    assert "shift the conclusion" in split["verdict"]
