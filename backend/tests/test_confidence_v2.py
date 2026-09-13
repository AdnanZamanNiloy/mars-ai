"""Confidence Engine v2: contradiction penalty, citation support, axis coverage."""

from app.core.confidence import compute_confidence


GOOD_FACTS = [
    {"claim": "Solar capacity grew 40% in 2024", "source": "https://iea.org/x",
     "verified": True, "verification_score": 0.8, "sub_question": "how much did solar grow"},
    {"claim": "Solar capacity expanded by 40 percent during 2024", "source": "https://irena.org/y",
     "verified": True, "verification_score": 0.75, "sub_question": "how much did solar grow"},
    {"claim": "Wind additions slowed last year", "source": "https://ember-energy.org/z",
     "verified": True, "verification_score": 0.7, "sub_question": "how much did wind grow"},
]
CRITIC_PASS = {"is_sufficient": True, "reason": "ok"}
SUB_QUESTIONS = [
    {"question": "how much did solar grow", "axis": "evidence"},
    {"question": "how much did wind grow", "axis": "evidence"},
]


def _base(**kw):
    args = dict(
        facts=GOOD_FACTS,
        critique=CRITIC_PASS,
        iteration=1,
        max_iterations=3,
    )
    args.update(kw)
    return compute_confidence(**args)


def test_baseline_unchanged_without_new_inputs():
    result = _base()
    assert "citation_support" not in result["signals"]
    assert "axis_coverage" not in result["signals"]
    assert result["overall"] > 0.5


def test_contradictions_reduce_confidence():
    severe = [{"kind": "numeric", "severity": 0.91, "intra_source": False}]
    base = _base()["overall"]
    penalized = _base(contradictions=severe)["overall"]
    # 3-fact pool: scale = 6/3 = 2 -> penalty 0.06 * 2 = 0.12.
    assert penalized == round(max(0.0, base - 0.12), 3)


def test_large_pool_reduces_penalty_scale():
    severe = [{"kind": "numeric", "severity": 0.91, "intra_source": False}]
    big_pool = GOOD_FACTS * 4  # 12 facts -> scale = 1.0
    base = _base(facts=big_pool)["overall"]
    penalized = _base(facts=big_pool, contradictions=severe)["overall"]
    assert penalized == round(max(0.0, base - 0.06), 3)


def test_multiple_contradictions_penalty_capped():
    many = [{"kind": "numeric", "severity": 0.9, "intra_source": False} for _ in range(5)]
    big_pool = GOOD_FACTS * 4  # scale 1.0: 2 severe -> 0.12; 5 -> capped 0.18
    p_two = _base(facts=big_pool, contradictions=many[:2])["overall"]
    p_five = _base(facts=big_pool, contradictions=many)["overall"]
    base = _base(facts=big_pool)["overall"]
    assert abs((base - p_two) - 0.12) < 0.005
    assert abs((base - p_five) - 0.18) < 0.005


def test_intra_source_conflicts_do_not_penalize():
    intra = [{"kind": "numeric", "severity": 0.9, "intra_source": True}]
    assert _base(contradictions=intra)["overall"] == _base()["overall"]


def test_citation_support_blends_when_present():
    result = _base(answer_support={"rate": 0.9, "cited": 10, "supported": 9})
    assert result["signals"]["citation_support"] == 0.9
    assert result["weights"]["citation_support"] == 0.10
    assert result["weights"]["citation_coverage"] < 0.25


def test_citation_support_ignored_when_rate_missing():
    result = _base(answer_support={"rate": None, "cited": 0, "supported": 0})
    assert "citation_support" not in result["signals"]


def test_axis_coverage_signal_when_plan_present():
    result = _base(sub_questions=SUB_QUESTIONS)
    assert "axis_coverage" in result["signals"]
    assert 0.0 <= result["signals"]["axis_coverage"] <= 1.0
    assert result["weights"]["axis_coverage"] == 0.05


def test_axis_coverage_penalizes_uncovered_plan():
    # Plan asks for two axes; facts cover only solar (both sources).
    covered = _base(sub_questions=SUB_QUESTIONS)
    partial = _base(sub_questions=SUB_QUESTIONS[:1] + [
        {"question": "regulatory outlook", "axis": "policy"},
    ])
    assert partial["signals"]["axis_coverage"] <= covered["signals"]["axis_coverage"]


def test_axis_coverage_full_coverage_scores_one():
    # Regression: the signal was structurally 0.0 (URL attribution against an
    # empty search_results list) even when verified facts covered every axis.
    result = _base(sub_questions=SUB_QUESTIONS)
    assert result["signals"]["axis_coverage"] == 1.0


def test_axis_coverage_partial_coverage_scores_fraction():
    result = _base(sub_questions=SUB_QUESTIONS[:1] + [
        {"question": "regulatory outlook", "axis": "policy"},
    ])
    assert result["signals"]["axis_coverage"] == 0.5


def test_axis_coverage_unverified_facts_do_not_count():
    unverified = [
        {**f, "verified": False} for f in GOOD_FACTS
    ]
    result = _base(facts=unverified, sub_questions=SUB_QUESTIONS)
    assert result["signals"]["axis_coverage"] == 0.0


def test_axis_coverage_counts_all_facts_when_verification_never_ran():
    no_flags = [
        {k: v for k, v in f.items() if k != "verified"} for f in GOOD_FACTS
    ]
    result = _base(facts=no_flags, sub_questions=SUB_QUESTIONS)
    assert result["signals"]["axis_coverage"] == 1.0


def test_penalty_note_recorded():
    result = _base(contradictions=[{"kind": "numeric", "severity": 0.7}])
    assert any("penalized" in n for n in result["notes"])


def test_historical_weights_untouched_by_absent_signals():
    result = _base()
    assert result["weights"]["citation_coverage"] == 0.25
    assert result["weights"]["source_diversity"] == 0.15
