"""v3 Set A: source-intelligence registry + evidence-primitive coverage.

Offline and deterministic. Locks in the behaviors the rest of the v3 port
depends on: canonical URLs, tier/primacy classification, legacy score
preservation where claimed, numeric grounding, polarity, corroborating
dedup, and aggregate evidence stats.
"""
from app.agents import sources
from app.agents.evidence_utils import (
    claim_polarity,
    dedupe_semantic_facts,
    evidence_stats,
    extract_numbers,
    numbers_grounded,
    verify_answer_support,
)


def test_canonical_url_strips_tracking_and_normalizes():
    assert sources.canonical_url("http://www.Example.com/Path/?utm_source=x&b=2#frag") == \
        "https://example.com/Path?b=2"
    assert sources.canonical_url("https://example.com/a//b/") == "https://example.com/a/b"
    assert sources.canonical_url("https://example.com/news/amp") == "https://example.com/news"
    assert sources.same_document("https://example.com/x?utm_campaign=a&fbclid=z", "http://www.example.com/x")


def test_classify_tiers_and_primacy():
    assert sources.classify_source("https://www.who.int/data").tier == sources.TIER_OFFICIAL
    assert sources.classify_source("https://www.nature.com/articles/x").tier == sources.TIER_PEER_REVIEWED
    assert sources.classify_source("https://arxiv.org/abs/1234").tier == sources.TIER_PREPRINT
    assert sources.classify_source("https://en.wikipedia.org/wiki/X").tier == sources.TIER_REFERENCE
    assert sources.classify_source("https://www.reuters.com/world/").tier == sources.TIER_MEDIA
    assert sources.classify_source("https://example.com/report").tier == sources.TIER_SECONDARY
    assert sources.classify_source("https://www.reddit.com/r/x").tier == sources.TIER_LOW
    assert sources.is_primary_source("https://census.gov/data") is True
    assert sources.is_primary_source("https://www.reuters.com/world/") is False


def test_legacy_scores_preserved_where_claimed():
    # Unregistered-domain TLD ladder is unchanged: thresholds downstream
    # (0.55 keep / 0.62 facts) keep their meaning.
    assert sources.authority_score("https://example.com/x") == 0.58
    assert sources.authority_score("https://example.org/x") == 0.75
    assert sources.authority_score("https://example.co/x") == 0.62
    assert sources.authority_score("https://example.io/x") == 0.55
    assert sources.authority_score("https://www.reddit.com/r/x") == 0.0
    assert sources.authority_score("not a url") == 0.0


def test_freshness_decay_ordering():
    fresh = sources.freshness_score("2026-09-01", "news")
    old = sources.freshness_score("2026-01-01", "news")
    assert 0.0 < old < fresh <= 1.0
    assert sources.freshness_score("", "news") == 0.45  # unknown, not stale
    assert sources.freshness_score("2026-01-01", "encyclopedia") > old  # slower decay


def test_primary_source_share_counts_distinct_documents():
    urls = [
        "https://census.gov/a?utm_source=x",
        "http://www.census.gov/a",
        "https://example.com/b",
    ]
    assert sources.primary_source_share(urls) == 0.5


def test_extract_numbers_folds_scale_and_flags_years():
    # Scale words fold into the value and consume the unit slot by design,
    # so "2.4bn" and "2,400,000,000" compare equal downstream.
    (q,) = extract_numbers("capacity reached 2.4 billion")
    assert q.value == 2.4e9 and q.unit == "" and not q.is_year
    (u,) = extract_numbers("capacity reached 2400 mwh")
    assert u.value == 2400.0 and u.unit == "mwh" and not u.is_year
    (y,) = extract_numbers("in 2019 the policy changed")
    assert y.is_year is True


def test_numbers_grounded_tolerance():
    assert numbers_grounded("costs fell 40%", "costs fell 40 percent last year") is True
    assert numbers_grounded("costs fell 40%", "costs fell 4% last year") is False
    assert numbers_grounded("1,200,000,000 people", "population of 1.2 billion") is True
    assert numbers_grounded("no figures here", "nothing numeric") is True  # vacuous


def test_claim_polarity_catches_inversion():
    assert claim_polarity("costs increased sharply") == 1
    assert claim_polarity("costs decreased sharply") == -1
    assert claim_polarity("costs did not increase") == -1


def test_dedupe_preserves_metadata_and_records_corroboration():
    facts = [
        {"claim": "Solar capacity doubled in 2025", "source": "https://a.com/x",
         "confidence": 0.7, "verified": True, "verification_score": 0.9,
         "verification_reason": "overlap", "published_at": "2025-01-01"},
        {"claim": "Solar capacity doubled in 2025!", "source": "https://b.org/y",
         "confidence": 0.6, "verified": False},
    ]
    (merged,) = dedupe_semantic_facts(facts)
    assert merged["verified"] is True  # verified copy never displaced
    assert merged["verification_score"] == 0.9
    assert merged["published_at"] == "2025-01-01"
    assert merged["corroboration_count"] == 2
    assert set(merged["corroborating_sources"]) == {"https://a.com/x", "https://b.org/y"}


def test_evidence_stats_single_pass_counts():
    facts = [
        {"claim": "A", "source": "https://a.com/1", "confidence": 0.8,
         "verified": True, "sub_question": "q1", "corroboration_count": 2},
        {"claim": "B", "source": "https://b.org/2", "confidence": 0.6,
         "verified": False, "sub_question": "q1"},
    ]
    stats = evidence_stats(facts)
    assert stats["total"] == 2
    assert stats["verified"] == 1
    assert stats["distinct_domains"] == 2
    assert stats["axes_covered"] == 1
    assert stats["corroborated"] == 1


def test_verify_answer_support_numeric_gate():
    answer = (
        "Costs fell 40% last year [1].\n\n"
        "## Sources\n[1] Costs fell 4% last year — https://a.com/x\n"
    )
    facts = [{"claim": "Costs fell 4% last year", "source": "https://a.com/x",
              "confidence": 0.8, "verified": True}]
    report = verify_answer_support(answer, facts)
    assert report["cited"] == 1
    assert report["supported"] == 0  # lexical hit, numeric miss
    assert len(report["numeric_failures"]) == 1
    assert report["numeric_rate"] == 0.0
