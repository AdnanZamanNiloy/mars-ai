"""Contradiction Engine tests (Phase 3.2)."""
from app.core.contradictions import extract_numbers, find_contradictions


def test_extract_numbers_scales():
    assert extract_numbers("growth of 25% in 2024") == [25.0, 2024.0]
    assert extract_numbers("1.5 million users") == [1_500_000.0]
    assert extract_numbers("3,000 MW capacity") == [3000.0]
    assert extract_numbers("cost fell to $30 per MWh") == [30.0]
    assert extract_numbers("no numbers here") == []


def test_detects_numeric_conflict_between_similar_claims():
    facts = [
        {"claim": "Market grew at 25% annually", "source": "https://arxiv.org/a", "confidence": 0.9},
        {"claim": "Market grew at 40% annually", "source": "https://nature.com/b", "confidence": 0.8},
    ]
    out = find_contradictions(facts)
    assert len(out) == 1
    assert out[0]["value_a"] == [25.0]
    assert out[0]["value_b"] == [40.0]


def test_no_conflict_when_numbers_agree():
    facts = [
        {"claim": "Market grew at 25% annually", "source": "https://arxiv.org/a", "confidence": 0.9},
        {"claim": "Market grew at 25% yearly", "source": "https://nature.com/b", "confidence": 0.8},
    ]
    assert find_contradictions(facts) == []


def test_no_conflict_between_dissimilar_claims():
    facts = [
        {"claim": "Solar capacity reached 3000 MW", "source": "https://arxiv.org/a", "confidence": 0.9},
        {"claim": "Population of Dhaka exceeded 20 million", "source": "https://nature.com/b", "confidence": 0.8},
    ]
    assert find_contradictions(facts) == []


def test_same_source_never_contradicts_itself():
    facts = [
        {"claim": "Market grew at 25% annually", "source": "https://arxiv.org/a", "confidence": 0.9},
        {"claim": "Market grew at 40% annually", "source": "https://arxiv.org/a", "confidence": 0.8},
    ]
    assert find_contradictions(facts) == []


def test_near_duplicate_restatement_not_double_counted():
    """A third near-copy of claim A (slightly reworded) must not create a
    second conflict entry for the same underlying disagreement."""
    facts = [
        {"claim": "Market grew at 25% annually", "source": "https://arxiv.org/a", "confidence": 0.9},
        {"claim": "Market grew at 40% annually", "source": "https://nature.com/b", "confidence": 0.8},
        {"claim": "Market grew at 25% annually according to analysts", "source": "https://arxiv.org/c", "confidence": 0.7},
    ]
    out = find_contradictions(facts)
    assert len(out) == 1, out


def test_cap_on_contradictions():
    facts = [
        {"claim": f"Sector growth was {10 + i}% this year", "source": f"https://arxiv.org/{i}", "confidence": 0.9}
        for i in range(8)
    ]
    assert len(find_contradictions(facts)) <= 5
