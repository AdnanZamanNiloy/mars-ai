"""Pure-function tests for evidence_utils (Phase 1.8). No mocking needed."""
from app.agents.evidence_utils import (
    dedupe_semantic_facts,
    extract_domain,
    filter_facts_by_domain,
    normalize_claim_text,
    source_reliability_score,
)


def test_extract_domain_strips_www():
    assert extract_domain("https://www.nature.com/article") == "nature.com"
    assert extract_domain("http://arxiv.org/abs/2005.11401") == "arxiv.org"
    assert extract_domain("not a url") == ""


def test_source_reliability_score_ordering():
    high_authority = source_reliability_score("https://nature.com/paper")
    gov = source_reliability_score("https://who.int/report")
    org = source_reliability_score("https://example.org/x")
    com = source_reliability_score("https://randomsite.com/x")
    blocked = source_reliability_score("https://reddit.com/r/ml")
    empty = source_reliability_score("")

    assert high_authority >= gov > org > com >= 0.55
    assert blocked == 0.0
    assert empty == 0.0


def test_normalize_claim_text_truncates_and_capitalizes():
    long = "word " * 100
    normalized = normalize_claim_text(long)
    assert normalized.endswith("...")
    assert len(normalized) <= 260
    assert normalize_claim_text("the claim.") == "The claim"
    assert normalize_claim_text("   ") == ""


def test_dedupe_semantic_facts_merges_near_duplicates_keeps_best():
    facts = [
        {"claim": "RAG retrieves documents before generating answers", "source": "https://arxiv.org/a", "confidence": 0.7},
        {"claim": "RAG retrieves documents before generating answers.", "source": "https://arxiv.org/b", "confidence": 0.9},
        {"claim": "Completely different topic about solar panels", "source": "https://arxiv.org/c", "confidence": 0.8},
    ]
    deduped = dedupe_semantic_facts(facts)
    assert len(deduped) == 2
    # The higher-confidence duplicate must win.
    rag = next(f for f in deduped if "RAG" in f["claim"])
    assert rag["confidence"] == 0.9


def test_dedupe_semantic_facts_drops_empty_claims():
    facts = [
        {"claim": "", "source": "https://arxiv.org/a", "confidence": 0.9},
        {"claim": "Valid claim", "source": "", "confidence": 0.9},
        {"claim": "Valid claim two", "source": "https://arxiv.org/c", "confidence": 0.8},
    ]
    assert len(dedupe_semantic_facts(facts)) == 1


def test_filter_facts_by_domain_blocks_low_quality_sources():
    facts = [
        {"claim": "Reddit says X", "source": "https://reddit.com/r/x", "confidence": 0.9},
        {"claim": "Nature says Y", "source": "https://nature.com/y", "confidence": 0.8},
    ]
    kept = filter_facts_by_domain(facts)
    assert all("reddit" not in str(f["source"]) for f in kept)
    assert any("nature" in str(f["source"]) for f in kept)


def test_filter_facts_by_domain_uses_fallback_when_no_strong_sources():
    facts = [
        {"claim": "Blog claim A", "source": "https://someone.medium.com/a", "confidence": 0.4},
        {"claim": "Org claim B", "source": "https://example.org/b", "confidence": 0.7},
    ]
    kept = filter_facts_by_domain(facts)
    assert any("example.org" in str(f["source"]) for f in kept)
    assert all("medium.com" not in str(f["source"]) for f in kept)
