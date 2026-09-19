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


def test_cross_source_agreement_uses_measured_corroboration():
    """Fix: CSA must read the measured corroboration the pipeline records, not
    re-derive it by pairwise claim similarity. A single claim held by 3
    publishers is full independent corroboration."""
    facts = [{
        "claim": "The Eiffel Tower is 330 metres tall",
        "source": "https://en.wikipedia.org/wiki/Eiffel_Tower",
        "corroboration_count": 3,
        "corroborating_sources": [
            "https://en.wikipedia.org/wiki/Eiffel_Tower",
            "https://www.britannica.com/topic/Eiffel-Tower",
            "https://structurae.net/en/structures/eiffel-tower",
        ],
    }]
    result = _base(facts=facts, critique={"is_sufficient": False})
    assert result["signals"]["cross_source_agreement"] == 1.0


def test_cross_source_agreement_single_publisher_is_zero():
    facts = [{
        "claim": "A lone claim from one publisher",
        "source": "https://randomblog.com/a",
        "corroboration_count": 1,
    }]
    result = _base(facts=facts, critique={"is_sufficient": False})
    assert result["signals"]["cross_source_agreement"] == 0.0


def test_cross_source_agreement_two_publishers_is_full():
    facts = [{
        "claim": "A claim two independent publishers assert",
        "source": "https://a.example/news",
        "corroboration_count": 2,
    }]
    result = _base(facts=facts, critique={"is_sufficient": False})
    assert result["signals"]["cross_source_agreement"] == 1.0


def test_cross_source_agreement_falls_back_without_measured_count():
    """Legacy facts with no corroboration_count still use the pairwise path."""
    disjoint = [
        {"claim": "The Transformer is a neural network architecture from 2017",
         "source": "https://arxiv.org/abs/1"},
        {"claim": "A transformer changes AC voltage by electromagnetic induction",
         "source": "https://www.britannica.com/topic/transformer"},
    ]
    result = _base(facts=disjoint, critique={"is_sufficient": False})
    # Genuinely disjoint claims are not corroboration -> fallback scores 0.
    assert result["signals"]["cross_source_agreement"] == 0.0
    same = [
        {"claim": "Solar capacity grew 40% in 2024", "source": "https://iea.org/x"},
        {"claim": "Solar capacity expanded by 40 percent during 2024", "source": "https://irena.org/y"},
    ]
    result2 = _base(facts=same, critique={"is_sufficient": False})
    assert result2["signals"]["cross_source_agreement"] == 1.0


def test_critic_survival_is_evidence_grounded_not_constant():
    """A fail with many named gaps survives less than a bare hedge with none.
    The old constant made every fail identical at the ceiling."""
    hedge = _base(critique={"is_sufficient": False})  # no gaps named
    criticized = _base(critique={
        "is_sufficient": False,
        "gaps": ["uncovered angle: cost", "uncovered angle: regulation",
                 "uncovered angle: critics"],
        "gate_failures": ["domains=1<2", "verified=0"],
    })
    assert criticized["signals"]["critic_survival"] < hedge["signals"]["critic_survival"]


def test_critic_survival_pass_is_one_and_fail_capped():
    assert _base(critique={"is_sufficient": True})["signals"]["critic_survival"] == 1.0
    forced = _base(critique={
        "is_sufficient": False,
        "gaps": ["g1", "g2", "g3"],
        "gate_failures": ["g4", "g5"],
    }, iteration=3, max_iterations=3)
    assert forced["signals"]["critic_survival"] <= 0.6
    # A FAIL can never outrank a PASS.
    assert forced["signals"]["critic_survival"] < 1.0


def test_critic_survival_does_not_collapse_to_zero():
    """Regression: a fixed denominator over an unbounded count drove survival
    to a flat 0.0 for any critic naming enough items (~4-7 on a normal run),
    which the UI rendered as "0% critic survival" next to an overall of ~0.8.
    Gate failures must weigh more than advisory gaps, the score must fall
    monotonically, and it must never reach 0."""
    from app.core.confidence import _critic_survival

    def survival(gates: int, gaps: int = 0) -> float:
        return _critic_survival(
            {
                "is_sufficient": False,
                "gaps": [f"gap {i}" for i in range(gaps)],
                "gate_failures": [f"gate {i}" for i in range(gates)],
            },
            1,
            3,
        )

    # A realistic critic (2 gates + a few advisory gaps) keeps meaningful
    # survival instead of collapsing to zero.
    assert survival(2, 5) > 0.0
    assert survival(2, 5) >= 0.15, "the floor keeps a working critic off zero"
    # Monotonic in both dimensions, and never below the floor.
    for g in range(0, 15):
        assert survival(g + 1, g) <= survival(g, g)
        assert survival(g, g) >= 0.15
    # Objective gate failures cost more than advisory gaps.
    assert survival(3, 0) < survival(0, 3)
    # A FAIL never outranks the pass/hedge values.
    assert survival(0, 0) == 0.6
    assert _critic_survival({"is_sufficient": True}, 1, 3) == 1.0


def test_freshness_uses_fact_dates_and_shared_curve():
    """Freshness is measured over facts actually in the pool, using the same
    exponential curve as the verifier (not a separate linear one)."""
    from app.agents.sources import freshness_score

    facts = [
        {"claim": "c1", "source": "https://a.com/1", "verified": True,
         "verification_score": 0.8, "published_at": "2026-09-01", "search_type": "default"},
        {"claim": "c2", "source": "https://b.com/2", "verified": True,
         "verification_score": 0.8, "published_at": "2026-09-01", "search_type": "default"},
    ]
    result = _base(facts=facts)
    # Both are rounded to 3 dp by their respective producers; compare on that
    # scale so a 4th-decimal difference is not treated as a regression.
    assert result["signals"]["freshness"] == round(freshness_score("2026-09-01", "default"), 3)


def test_freshness_ignores_undated_search_results_when_facts_present():
    facts = [
        {"claim": "c1", "source": "https://a.com/1", "verified": True,
         "verification_score": 0.8, "published_at": "2026-09-01", "search_type": "default"},
    ]
    with_dates = _base(facts=facts, source_dates=["2001-01-01", "1999-01-01"])
    # Fact-level dates win over irrelevant stale search-result dates.
    assert with_dates["signals"]["freshness"] > 0.9


def test_freshness_unmeasured_stays_zero_and_weightless():
    result = _base(facts=[
        {"claim": "c1", "source": "https://a.com/1", "verified": True, "verification_score": 0.8},
    ])
    assert result["signals"]["freshness"] == 0.0
    assert result["weights"]["freshness"] == 0.0
    assert any("freshness" in n for n in result["notes"])


def test_axis_coverage_requires_min_two_facts_per_axis():
    """One verified fact per angle is not "covered": the axis needs
    AXIS_MIN_FACTS before it counts."""
    from app.core.confidence import AXIS_MIN_FACTS
    assert AXIS_MIN_FACTS == 2

    # Two DISTINCT axes (definition, evidence), one fact each.
    plan = [
        {"question": "what is solar", "axis": "definition"},
        {"question": "how much did solar grow", "axis": "evidence"},
    ]
    one_per_axis = [
        {"claim": "solar is a renewable energy source", "source": "https://iea.org/x",
         "verified": True, "verification_score": 0.8, "sub_question": "what is solar"},
        {"claim": "solar grew", "source": "https://irena.org/y", "verified": True,
         "verification_score": 0.8, "sub_question": "how much did solar grow"},
    ]
    result = _base(facts=one_per_axis, sub_questions=plan)
    assert result["signals"]["axis_coverage"] == 0.0

    two_per_axis = [
        {**one_per_axis[0]},
        {"claim": "solar is a photovoltaic energy source", "source": "https://energy.gov/a",
         "verified": True, "verification_score": 0.8, "sub_question": "what is solar"},
        {**one_per_axis[1]},
        {"claim": "solar capacity grew last year", "source": "https://ember.org/z",
         "verified": True, "verification_score": 0.8, "sub_question": "how much did solar grow"},
    ]
    result2 = _base(facts=two_per_axis, sub_questions=plan)
    assert result2["signals"]["axis_coverage"] == 1.0


def test_source_diversity_counts_registrable_domains_not_subdomains():
    """arxiv.org + ar5iv.labs.arxiv.org are ONE publisher; subdomains must not
    inflate diversity."""
    facts = [
        {"claim": "a", "source": "https://arxiv.org/abs/1", "verified": True, "verification_score": 0.8},
        {"claim": "b", "source": "https://ar5iv.labs.arxiv.org/html/1", "verified": True, "verification_score": 0.8},
        {"claim": "c", "source": "https://www.britannica.com/x", "verified": True, "verification_score": 0.8},
        {"claim": "d", "source": "https://kids.britannica.com/y", "verified": True, "verification_score": 0.8},
    ]
    result = _base(facts=facts)
    # Two registrable publishers across four facts -> 2 / min(4,5) = 0.5,
    # not the old raw-host 4/4 = 1.0.
    assert result["signals"]["source_diversity"] == 0.5


def test_quote_matching_uses_corroboration_excerpt(monkeypatch):
    """A quote present only in the retained excerpt (raw content blanked) must
    verify, instead of failing as "direct quote not found"."""
    from app.agents import verifier

    quote = "solar capacity grew forty percent during the year"
    excerpt = f"Intro text. {quote}. More text."
    fact = {
        "claim": "Solar capacity grew by forty percent during the year.",
        "source": "https://example.gov/report",
        "direct_quote": quote,
        "corroboration_excerpt": excerpt,
    }
    results = [{
        "url": "https://example.gov/report",
        "content": "",
        "snippet": "Solar capacity grew.",
        "title": "Solar report",
    }]
    out = verifier.verify_facts([fact], results)
    assert out[0]["verification_checks"]["quote_verified"] is True


def test_cross_source_agreement_prefilter_preserves_corroboration():
    """The cheap prefilter in _similarity must be score-preserving: pairs
    inside the corroboration band keep their exact SequenceMatcher score,
    unrelated pairs are skipped (they can never reach the 0.55 band)."""
    from difflib import SequenceMatcher

    from app.core.confidence import _similarity

    def old_similarity(a, b):
        a_norm = " ".join(sorted(a.lower().split()))
        b_norm = " ".join(sorted(b.lower().split()))
        if not a_norm or not b_norm:
            return 0.0
        return SequenceMatcher(None, a_norm, b_norm).ratio()

    corroborating = [
        ("Solar capacity grew 40% in 2024", "Solar capacity expanded by 40 percent during 2024"),
        ("RAG retrieves documents before generation", "RAG retrieves external documents first"),
    ]
    for a, b in corroborating:
        assert _similarity(a, b) == old_similarity(a, b)

    unrelated = ("Solar capacity grew 40% in 2024",
                 "The chef prepared pasta with tomato sauce and basil tonight")
    assert _similarity(*unrelated) == 0.0
    assert old_similarity(*unrelated) < 0.55, "prefilter assumes this pair is out of band"


def test_cross_source_agreement_scored_per_angle_not_per_claim():
    """A specific claim restated nowhere should not sink an angle that IS
    independently corroborated. Angle-level scoring measures research coverage,
    and any one corroborated claim establishes the angle is corroborated."""
    facts = [
        # Angle A: three claims, one of which is independently corroborated.
        {"claim": "Solar capacity grew 40%", "source": "https://iea.org/a",
         "corroboration_count": 2, "sub_question": "how much did solar grow"},
        {"claim": "Solar added 300 GW", "source": "https://iea.org/b",
         "corroboration_count": 1, "sub_question": "how much did solar grow"},
        {"claim": "Solar costs fell 20%", "source": "https://iea.org/c",
         "corroboration_count": 1, "sub_question": "how much did solar grow"},
        # Angle B: fully corroborated.
        {"claim": "Wind capacity rose", "source": "https://irena.org/a",
         "corroboration_count": 2, "sub_question": "how much did wind grow"},
    ]
    result = _base(facts=facts, critique={"is_sufficient": False})
    # Both angles have at least one corroborated claim -> 1.0, not per-claim 1/4.
    assert result["signals"]["cross_source_agreement"] == 1.0


def test_cross_source_agreement_angle_without_any_corroboration_is_zero():
    """An angle whose every claim is single-publisher is not corroborated."""
    facts = [
        {"claim": "Solar capacity grew 40%", "source": "https://iea.org/a",
         "corroboration_count": 2, "sub_question": "how much did solar grow"},
        {"claim": "Wind added 50 GW", "source": "https://irena.org/b",
         "corroboration_count": 1, "sub_question": "how much did wind grow"},
        {"claim": "Wind costs fell 10%", "source": "https://irena.org/c",
         "corroboration_count": 1, "sub_question": "how much did wind grow"},
    ]
    result = _base(facts=facts, critique={"is_sufficient": False})
    # Solar angle corroborated, wind angle not -> 0.5.
    assert result["signals"]["cross_source_agreement"] == 0.5


def test_cross_source_agreement_uncorroborated_angles_stay_low():
    facts = [
        {"claim": "a", "source": "https://x.com/1", "corroboration_count": 1,
         "sub_question": "angle one"},
        {"claim": "b", "source": "https://x.com/2", "corroboration_count": 1,
         "sub_question": "angle two"},
    ]
    result = _base(facts=facts, critique={"is_sufficient": False})
    assert result["signals"]["cross_source_agreement"] == 0.0


def test_breakdown_carries_engine_version_for_replay():
    """Every breakdown must identify the engine that produced it. Persisted
    reviews are replayed verbatim by the UI, so a versionless row (pre-fix)
    cannot be told apart from a current one and froze critic_survival at a
    stale 0% indefinitely. The stamp is the source-of-truth discriminator."""
    from app.core.confidence import ENGINE_VERSION

    result = compute_confidence(list(GOOD_FACTS), CRITIC_PASS, 1, 3)
    assert result["engine_version"] == ENGINE_VERSION


def test_epistemics_adjust_the_live_confidence_path():
    """The live workflow calls app.core.confidence.compute_confidence directly.
    Epistemic adjustments must therefore be applied INSIDE it — not only in the
    unused app.agents.confidence wrapper. A genuine unresolved conflict must
    lower the score when epistemics are supplied, and the breakdown must still
    carry engine_version."""
    from app.agents.epistemics import assess_epistemics
    from app.core.confidence import ENGINE_VERSION

    facts = [
        {"claim": "Revenue was 2 billion dollars", "source": "https://a.com/x",
         "confidence": 0.9, "verified": True, "corroboration_count": 2},
        {"claim": "Revenue was 3 billion dollars", "source": "https://b.com/y",
         "confidence": 0.9, "verified": True, "corroboration_count": 2},
    ]
    contradictions = [{
        "claim_a": "Revenue was 2 billion dollars",
        "claim_b": "Revenue was 3 billion dollars",
        "source_a": "https://a.com/x", "source_b": "https://b.com/y",
    }]
    query = "What is current revenue?"
    epistemics = assess_epistemics(query, facts, contradictions)

    without = compute_confidence(list(facts), CRITIC_PASS, 1, 3)
    with_epi = compute_confidence(
        list(facts), CRITIC_PASS, 1, 3,
        contradictions=contradictions, epistemics=epistemics, query=query,
    )

    assert with_epi["overall"] < without["overall"], (
        "the live path must apply epistemic adjustments (the caps were dead "
        "before the wiring: a conflict cleared the same bar as a clean pool)"
    )
    assert with_epi["engine_version"] == ENGINE_VERSION
