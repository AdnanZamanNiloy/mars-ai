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


def test_unrelated_year_claims_are_not_a_temporal_conflict():
    """Regression: the temporal branch had no topical floor, so ANY two claims
    carrying different 4-digit years and a shared-unit number were reported as
    a period conflict. A live deep run paired a Rooppur cost claim with "38
    countries endorsed the tripling declaration" and "100 reactors in the US"
    purely because both contained a year — 5 bogus contradictions rendered into
    the report. Real temporal pairs (same measure, different years) score 0.51+;
    these unrelated pairs score 0.02-0.25 and must be ignored."""
    facts = [
        {"claim": "The Rooppur Nuclear Power Plant in Bangladesh has an estimated "
                  "construction cost of 12.65 billion US dollars, with cumulative "
                  "expenditure reaching 73,746.06 crore Bangladeshi taka as of June 2024",
         "source": "https://wikipedia.org/x", "confidence": 0.9},
        {"claim": "The Declaration to Triple Nuclear Energy by 2050 has been endorsed "
                  "by 38 countries aiming to at least triple global capacity.",
         "source": "https://world-nuclear.org/y", "confidence": 0.8},
        {"claim": "As of 2005 there were more than 100 operating nuclear reactors in "
                  "the United States producing about one fifth of the nations electricity.",
         "source": "https://britannica.com/z", "confidence": 0.8},
    ]
    assert find_contradictions(facts) == []


def test_genuine_temporal_conflict_still_detected():
    """The topical floor must not suppress a real period-mismatch conflict:
    the same measure (installed capacity in GW) reported for different years
    with materially different values."""
    facts = [
        {"claim": "Bangladesh's installed power generation capacity was 5 GW in 2009.",
         "source": "https://a.example/x", "confidence": 0.9},
        {"claim": "Bangladesh's installed power generation capacity reached 26.5 GW by 2024.",
         "source": "https://b.example/y", "confidence": 0.8},
    ]
    out = find_contradictions(facts)
    assert len(out) == 1, out
    assert out[0]["kind"] == "temporal"


def test_indian_numbering_currency_scale_parsed():
    """Indian-numbering scale words must fold in, so a crore amount keeps its
    true magnitude instead of parsing as a small scale-free count (part of why
    unrelated figures could be "compared")."""
    from app.agents.evidence_utils import extract_numbers
    nums = extract_numbers("73,746.06 crore Bangladeshi taka")
    assert nums and nums[0].value == 73746.06 * 1e7

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
