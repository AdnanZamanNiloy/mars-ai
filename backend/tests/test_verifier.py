"""Verification Agent tests (Phase 2.3)."""
from app.agents.verifier import verify_facts

RESULTS = [
    {
        "url": "https://arxiv.org/abs/2005.11401",
        "content": (
            "Retrieval augmented generation retrieves external documents relevant to the "
            "query and conditions generation on the retrieved evidence, reducing hallucination."
        ),
        "snippet": "",
    },
    {
        "url": "https://en.wikipedia.org/wiki/RAG",
        "snippet": "RAG grounds model outputs in sources.",
    },
]

FACTS = [
    {"claim": "RAG retrieves external documents before generating answers", "source": "https://arxiv.org/abs/2005.11401", "confidence": 0.9},
    {"claim": "Photosynthesis converts sunlight into chemical energy inside chloroplasts", "source": "https://arxiv.org/abs/2005.11401", "confidence": 0.9},
    {"claim": "RAG grounds outputs in cited sources", "source": "https://en.wikipedia.org/wiki/RAG", "confidence": 0.8},
    {"claim": "Some claim from an unfetched source", "source": "https://www.iea.org/reports/x", "confidence": 0.8},
]


def test_supported_fact_passes_verification():
    out = verify_facts(FACTS, RESULTS)
    assert out[0]["verified"] is True
    assert out[0]["verification_score"] >= 0.35


def test_zero_overlap_fact_is_flagged_unverified():
    out = verify_facts(FACTS, RESULTS)
    assert out[1]["verified"] is False, "near-zero lexical overlap must fail"
    assert "overlap" in out[1]["verification_reason"]


def test_snippet_only_source_still_verifiable():
    out = verify_facts(FACTS, RESULTS)
    # Wikipedia snippet: "RAG grounds model outputs in sources."
    assert out[2]["verification_score"] > 0.0


def test_unavailable_source_content_is_unverified():
    out = verify_facts(FACTS, RESULTS)
    assert out[3]["verified"] is False
    assert "unavailable" in out[3]["verification_reason"]


def test_facts_kept_in_state_even_when_unverified():
    """Transparency rule: verifier must not drop facts, only flag them."""
    out = verify_facts(FACTS, RESULTS)
    assert len(out) == len(FACTS)
    assert all("verified" in f for f in out)


def test_verifier_drops_truncated_fragments():
    from app.agents.verifier import verify_facts

    facts = [
        {"claim": "Transfer learning reuses models learned from a large dat",
         "source": "https://en.wikipedia.org/wiki/X", "confidence": 0.9},
        {"claim": "Transfer learning reuses models trained before on related tasks.",
         "source": "https://en.wikipedia.org/wiki/X", "confidence": 0.9},
    ]
    results = [{"url": "https://en.wikipedia.org/wiki/X",
                "content": "Transfer learning reuses models trained before on related tasks and data"}]
    out = verify_facts(facts, results)
    assert len(out) == 1
    assert "large dat" not in out[0]["claim"]


def test_numeric_mismatch_fails_despite_high_overlap():
    """'fell 40%' verified against a 'fell 4%' source scored ~0.95 on pure
    overlap before: the numeric hard-fail must catch it."""
    facts = [
        {"claim": "Solar costs fell 40 percent in 2024", "source": "https://example.com/solar",
         "confidence": 0.9},
    ]
    results = [
        {"url": "https://example.com/solar",
         "content": "Solar costs fell 4 percent in 2024 according to the annual review."},
    ]
    (out,) = verify_facts(facts, results)
    assert out["verified"] is False
    assert "figure in the claim does not appear" in out["verification_reason"]
    assert out["verification_checks"]["numbers_grounded"] is False


def test_polarity_inversion_fails_despite_shared_vocabulary():
    facts = [
        {"claim": "Solar costs increased sharply in 2024", "source": "https://example.com/solar",
         "confidence": 0.9},
    ]
    results = [
        {"url": "https://example.com/solar",
         "content": "Solar costs fell sharply in 2024 as demand dropped."},
    ]
    (out,) = verify_facts(facts, results)
    assert out["verified"] is False
    assert "contradicts the source" in out["verification_reason"]


def test_stale_evidence_flagged_not_dropped():
    facts = [
        {"claim": "Solar costs decreased in recent years", "source": "https://example.com/solar",
         "confidence": 0.8, "published_at": "2020-01-01", "search_type": "news"},
    ]
    results = [
        {"url": "https://example.com/solar",
         "content": "Solar costs decreased in recent years across major markets."},
    ]
    (out,) = verify_facts(facts, results)
    assert out["verified"] is True
    assert out["is_stale"] is True  # discounted downstream, never dropped
