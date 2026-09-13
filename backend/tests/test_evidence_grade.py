"""Evidence grading (Step 1): claim-level evidence quality, independent
corroboration, numeric-support flagging, contradiction demotion, and the
pool-quality score. Pure and deterministic — no LLM, no network."""
from app.core.evidence_grade import (
    GRADE_A, GRADE_C, GRADE_D,
    coverage_gaps_from_records,
    evidence_quality_score,
    grade_claim,
    grade_distribution,
    grade_facts,
    independent_corroboration,
    registrable_domain,
)


# --- independence must be measured by publisher, not URL count ---------------

def test_registrable_domain_collapses_subdomains():
    assert registrable_domain("https://news.bbc.co.uk/x") == "bbc.co.uk"
    assert registrable_domain("https://www.gov.uk/report") == "gov.uk"
    assert registrable_domain("https://x.com/a") == "x.com"
    assert registrable_domain("") == ""


def test_same_publisher_is_not_independent_corroboration():
    """Two URLs on one registrable domain are ONE source — the exact
    false-independence the brief forbids."""
    domains, count = independent_corroboration(
        ["https://news.example.com/a", "https://blog.example.com/b"],
        primary_source="https://example.com/root",
    )
    assert count == 1
    assert domains == ["example.com"]


def test_distinct_publishers_are_independent():
    domains, count = independent_corroboration(
        ["https://who.int/a", "https://worldbank.org/b"],
        primary_source="https://gov.uk/report",
    )
    assert count == 3
    assert set(domains) == {"who.int", "worldbank.org", "gov.uk"}


# --- grading -----------------------------------------------------------------

def test_single_source_quantitative_needs_corroboration():
    r = grade_claim({
        "claim": "Adoption grew 42% in 2024",
        "source": "https://blog.example.com/a",
        "verified": True,
    })
    assert r.has_numbers is True
    assert r.corroboration_count == 1
    assert r.needs_corroboration is True
    assert r.grade == GRADE_C


def test_primary_and_independently_corroborated_grades_a():
    r = grade_claim({
        "claim": "Adoption grew 42% in 2024",
        "source": "https://www.gov.uk/report",
        "verified": True,
        "corroborating_sources": ["https://www.gov.uk/report", "https://who.int/data"],
    })
    assert r.grade == GRADE_A
    assert r.needs_corroboration is False
    assert r.corroboration_count == 2


def test_unverified_weak_source_grades_d():
    r = grade_claim({"claim": "Some claim", "source": "https://randomblog.example/a"})
    assert r.grade == GRADE_D


def test_unverified_primary_is_only_c():
    """A primary source we could not verify against is not worth A/B."""
    r = grade_claim({"claim": "Some claim", "source": "https://www.gov.uk/report"})
    assert r.grade == GRADE_C


def test_contradiction_demotes_even_a_well_sourced_claim():
    r = grade_claim(
        {
            "claim": "Rate fell to 3%",
            "source": "https://who.int/x",
            "verified": True,
            "corroborating_sources": ["https://who.int/x", "https://worldbank.org/y"],
        },
        contradictions=[{"claim_a": "Rate fell to 3%", "claim_b": "Rate rose", "kind": "numeric"}],
    )
    assert r.contradiction_count == 1
    assert r.numeric_conflict is True
    assert r.grade == GRADE_C  # never A/B while conflicted


def test_quantitative_claim_with_explicit_number_detected():
    r = grade_claim({"claim": "Revenue reached $1.2B", "source": "https://gov.uk/x", "verified": True})
    assert r.has_numbers is True


# --- pool quality: few strong beats many weak --------------------------------

def test_quality_score_favours_strong_pool_over_weak():
    strong = grade_claim({
        "claim": "Adoption grew 42% in 2024",
        "source": "https://www.gov.uk/report",
        "verified": True,
        "corroborating_sources": ["https://www.gov.uk/report", "https://who.int/data"],
    })
    weak = grade_claim({"claim": "unverified thing", "source": ""})
    assert evidence_quality_score([strong, strong]) > evidence_quality_score([weak] * 10)


def test_empty_pool_scores_zero():
    assert evidence_quality_score([]) == 0.0


# --- annotations are additive and non-mutating -------------------------------

def test_grade_facts_adds_fields_without_mutating_original():
    original = {"claim": "Adoption grew 42%", "source": "https://gov.uk/a", "verified": True}
    out = grade_facts([original])
    assert out[0]["evidence_grade"] == GRADE_A or out[0]["evidence_grade"] in (GRADE_A, "B", "C")
    assert "evidence" in out[0]
    assert "evidence_grade" not in original  # original untouched


def test_grade_distribution_counts_every_grade():
    r_a = grade_claim({
        "claim": "grew 42%",
        "source": "https://gov.uk/a",
        "verified": True,
        "corroborating_sources": ["https://gov.uk/a", "https://who.int/b"],
    })
    r_d = grade_claim({"claim": "x", "source": ""})
    dist = grade_distribution([r_a, r_d])
    assert dist[GRADE_A] == 1
    assert dist[GRADE_D] == 1


def test_coverage_gaps_names_missing_corroboration():
    r = grade_claim({
        "claim": "Adoption grew 42% in 2024",
        "source": "https://blog.example.com/a",
        "verified": True,
    })
    gaps = coverage_gaps_from_records([r])
    assert gaps and any("independent corroboration" in g.lower() for g in gaps)


def test_coverage_gaps_flags_contradiction():
    r = grade_claim(
        {"claim": "Rate fell to 3%", "source": "https://who.int/x", "verified": True},
        contradictions=[{"claim_a": "Rate fell to 3%", "claim_b": "Rate rose", "kind": "numeric"}],
    )
    gaps = coverage_gaps_from_records([r])
    assert gaps and any("contradicted" in g.lower() for g in gaps)
