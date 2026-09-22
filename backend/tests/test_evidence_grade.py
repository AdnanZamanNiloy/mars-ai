"""Evidence grading (Step 1): claim-level evidence quality, independent
corroboration, numeric-support flagging and contradiction demotion. Pure
and deterministic — no LLM, no network."""
from app.core.evidence_grade import (
    GRADE_A, GRADE_C, GRADE_D,
    coverage_gaps_from_records,
    grade_claim,
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

# --- annotations are additive and non-mutating -------------------------------

def test_grade_facts_adds_fields_without_mutating_original():
    original = {"claim": "Adoption grew 42%", "source": "https://gov.uk/a", "verified": True}
    out = grade_facts([original])
    assert out[0]["evidence_grade"] == GRADE_A or out[0]["evidence_grade"] in (GRADE_A, "B", "C")
    assert "evidence" in out[0]
    assert "evidence_grade" not in original  # original untouched

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


def test_document_fingerprint_links_arxiv_and_doi_mirrors():
    from app.core.evidence_grade import document_fingerprint

    native = document_fingerprint("https://arxiv.org/abs/1706.03762")
    pdf = document_fingerprint("https://arxiv.org/pdf/1706.03762v5")
    doi = document_fingerprint("https://doi.org/10.48550/arXiv.1706.03762")
    assert native == "arxiv:1706.03762"
    assert pdf == native
    assert doi == native


def test_distinct_publisher_count_does_not_count_mirrors_of_one_work():
    from app.core.evidence_grade import distinct_publisher_count

    mirrors = [
        "https://arxiv.org/abs/1706.03762",
        "https://doi.org/10.48550/arXiv.1706.03762",
    ]
    assert distinct_publisher_count(mirrors) == 1


def test_distinct_publisher_count_counts_distinct_works():
    from app.core.evidence_grade import distinct_publisher_count

    different = [
        "https://arxiv.org/abs/1706.03762",
        "https://arxiv.org/abs/2401.12345",
    ]
    assert distinct_publisher_count(different) == 2


def test_is_new_publisher_rejects_mirror_across_hosts():
    from app.core.evidence_grade import is_new_publisher

    existing = ["https://arxiv.org/abs/1706.03762"]
    assert is_new_publisher("https://doi.org/10.48550/arXiv.1706.03762", existing) is False


def test_is_new_publisher_still_rejects_same_domain():
    from app.core.evidence_grade import is_new_publisher

    assert is_new_publisher(
        "https://blog.example.com/deep", ["https://www.example.com/"]
    ) is False


def test_independent_corroboration_merges_mirrors():
    _, count = independent_corroboration(
        ["https://doi.org/10.48550/arXiv.1706.03762"],
        primary_source="https://arxiv.org/abs/1706.03762",
    )
    assert count == 1


def test_apply_corroboration_rejects_mirror_of_same_work():
    from app.core.evidence_grade import apply_corroboration

    fact = {
        "claim": "The Transformer was introduced in 2017.",
        "source": "https://arxiv.org/abs/1706.03762",
        "corroboration_count": 1,
    }
    assert apply_corroboration(fact, "https://doi.org/10.48550/arXiv.1706.03762") is False
    assert fact["corroboration_count"] == 1
