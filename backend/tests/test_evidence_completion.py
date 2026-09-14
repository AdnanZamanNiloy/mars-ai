"""Evidence-completion targeting (workstream B).

Before synthesis, the remaining research budget must be aimed at the
highest-impact claims that are still single-source — quantitative claims, the
claims an executive summary / key findings uses, and high-corroboration-need
claims — rather than at whichever uncorroborated claim grading happened to
return first.

All deterministic and LLM-free.
"""
from app.core.evidence_completion import (
    PRIMARY_THIN_THRESHOLD,
    claim_impact,
    dimension_primary_share,
    primary_source_followups,
    rank_completion_targets,
)


def _fact(claim, source, **kw):
    base = {"claim": claim, "source": source, "verified": True}
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# impact ordering
# ---------------------------------------------------------------------------

def test_quantitative_claim_outranks_definitional_one():
    facts = [
        _fact("Retrieval augmented generation is a grounding technique", "https://a.com/x"),
        _fact("Global spending reached 200 billion dollars in 2025", "https://b.com/y"),
    ]
    ranked = rank_completion_targets(facts)
    assert len(ranked) == 2, "both single-source important claims need corroboration"
    assert "200 billion" in ranked[0]["claim"]
    assert ranked[0]["impact"] > ranked[-1]["impact"]


def test_summary_claim_outranks_unused_claim():
    facts = [
        _fact("A peripheral detail about the topic appears here", "https://a.com/x"),
        _fact("The headline finding is that adoption doubled in 2024", "https://b.com/y"),
    ]
    summary = ["The headline finding is that adoption doubled in 2024"]
    ranked = rank_completion_targets(facts, summary_claims=summary)
    assert "headline finding" in ranked[0]["claim"]


def test_claim_impact_scores_quantitative_and_summary():
    record = {
        "claim": "Output reached 12 gigawatts in 2025",
        "has_numbers": True,
        "needs_corroboration": True,
        "contradiction_count": 0,
    }
    assert claim_impact(record, summary_claims={"output reached 12 gigawatts in 2025"}) >= 8


def test_corroborated_claims_are_not_targets():
    facts = [_fact("Adoption grew 42% in 2024", "https://a.com/x")]
    facts[0]["corroborating_sources"] = ["https://b.org/y"]
    facts[0]["corroboration_count"] = 2
    assert rank_completion_targets(facts) == []


def test_ranking_is_total_on_empty_and_bad_input():
    assert rank_completion_targets([]) == []
    assert rank_completion_targets([{"no_claim": 1}]) == []
    assert claim_impact({}) == 0
    # Grading failure must never raise.
    assert rank_completion_targets([{"claim": "", "source": "x"}]) == []


# ---------------------------------------------------------------------------
# primary-source completion
# ---------------------------------------------------------------------------

def test_dimension_primary_share_reports_per_dimension():
    facts = [
        _fact("A finding about nuclear cost", "https://gov.uk/a", is_primary=True,
              sub_question="nuclear cost"),
        _fact("Another finding about nuclear cost", "https://blog.com/b", is_primary=False,
              sub_question="nuclear cost"),
        _fact("A finding about solar", "https://solar.gov/c", is_primary=True,
              sub_question="solar"),
    ]
    shares = dimension_primary_share(facts)
    assert shares["nuclear cost"] == 0.5
    assert shares["solar"] == 1.0


def test_primary_source_followups_targets_thin_dimensions_only():
    facts = [
        _fact("A finding about nuclear economics", "https://blog.com/b", is_primary=False,
              sub_question="nuclear economics"),
        _fact("A finding about solar economics", "https://solar.blog/c", is_primary=False,
              sub_question="solar economics"),
        _fact("A primary finding about banking", "https://fed.gov/d", is_primary=True,
              sub_question="banking"),
    ]
    sub_questions = [
        {"question": "nuclear economics", "search_type": "statistical", "domain": "economics"},
        {"question": "solar economics", "search_type": "statistical", "domain": "economics"},
        {"question": "banking", "search_type": "statistical", "domain": "economics"},
    ]
    followups = primary_source_followups(facts, sub_questions, limit=5)
    joined = " ".join(followups).lower()
    assert "nuclear economics" in joined
    assert "solar economics" in joined
    # A primary-led dimension gets no follow-up query.
    assert "banking" not in joined
    assert PRIMARY_THIN_THRESHOLD >= 0.0


def test_primary_source_followups_empty_when_nothing_researched():
    assert primary_source_followups([], [{"question": "x", "search_type": "news"}]) == []
