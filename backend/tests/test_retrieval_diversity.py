"""Retrieval-diversity regression: the primary-source query must actually run.

The live deep run's source ledger reported only ~27-48% primary sources and
leaned on vendor blogs (sureprompts, statworx, turingpost). The planner already
builds a `primary_source_query` (site:-scoped at the publisher that owns the
fact) and the search layer already knows how to run it — but it was appended
LAST in `contract_queries` and truncated away on any contract with two
variants, which is every high-value contract. These tests pin the slot
reservation and the authoritative-domain targeting.
"""
from app.agents.planner import _contract
from app.agents.search import contract_queries
from app.agents.sources import (
    authoritative_site_terms,
    build_corroboration_query,
    build_primary_source_query,
    primary_source_hints,
)


def _evidence_contract():
    return _contract(
        index=1,
        question="global AI capital expenditure 2025",
        axis="evidence",
        search_type="statistical",
        priority=1,
        domain="economics",
        variants=["AI capex statistics 2025", "AI infrastructure spending figures 2025"],
    )


def test_primary_source_query_survives_the_variant_budget():
    """With two variants and a budget of 3, the site:-scoped primary query
    must still be present — it is the query that reaches the publisher."""
    contract = _evidence_contract()
    assert contract["primary_source_query"], "planner did not build a primary query"
    queries = contract_queries(contract, max_queries=3)
    assert contract["primary_source_query"] in queries
    assert len(queries) <= 3


def test_contract_queries_dedupes_and_caps():
    contract = _evidence_contract()
    queries = contract_queries(contract, max_queries=2)
    assert len(queries) <= 2
    # Base question first — the contract's own phrasing leads.
    assert queries[0] == contract["question"]


def test_primary_source_query_targets_registered_publishers():
    """A statistical/economics question must be aimed at the agencies that
    publish the numbers, not left to a general web search."""
    query = build_primary_source_query(
        "global AI capital expenditure 2025", "statistical", "economics"
    )
    assert "site:" in query
    assert any(host in query for host in primary_source_hints("statistical", "economics"))


def test_no_hint_means_no_primary_query():
    """An unregistered (type, domain) pair must not fabricate a site: query —
    an over-constrained query returns nothing, which is worse than a broad one."""
    assert build_primary_source_query("what is a quark", "comparison", "general") == ""


def test_authoritative_rotation_reaches_different_publishers():
    """Successive corroboration attempts for one claim must target different
    authoritative hosts, so the second pass is not a repeat of the first."""
    first = authoritative_site_terms(max_sites=3, offset=0)
    second = authoritative_site_terms(max_sites=3, offset=3)
    assert first and second
    assert set(first) != set(second)


def test_corroboration_query_excludes_the_known_publisher():
    query = build_corroboration_query(
        "global AI capital expenditure reached 200 billion",
        exclude_domain="example.com",
        quantitative=True,
    )
    assert "-site:example.com" in query
    assert "site:" in query
