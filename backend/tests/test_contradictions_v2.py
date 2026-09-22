"""Contradiction Engine v2: numeric, polarity and temporal detection."""

from app.core.contradictions import find_contradictions


def test_polarity_contradiction_detected():
    facts = [
        {"claim": "The new model outperforms the baseline on all benchmarks",
         "source": "https://a.com/x", "confidence": 0.9},
        {"claim": "The new model does not outperform the baseline on all benchmarks",
         "source": "https://b.com/y", "confidence": 0.9},
    ]
    found = find_contradictions(facts)
    assert len(found) == 1
    assert found[0]["kind"] == "polarity"
    assert found[0]["severity"] >= 0.5
    assert "opposite" in found[0]["note"]


def test_temporal_contradiction_detected():
    facts = [
        {"claim": "Global installed solar capacity reached 1200 GW in 2023",
         "source": "https://a.com/x", "confidence": 0.9},
        {"claim": "Global installed solar capacity reached 1600 GW in 2024",
         "source": "https://b.com/y", "confidence": 0.9},
    ]
    found = find_contradictions(facts)
    assert len(found) == 1
    assert found[0]["kind"] == "temporal"
    assert found[0]["values"]["years_a"] == [2023]
    assert found[0]["values"]["years_b"] == [2024]
    assert "period" in found[0]["note"]


def test_numeric_conflict_uses_units():
    # A % vs a GW is NOT a conflict — the unit-aware check skips it.
    facts = [
        {"claim": "Module efficiency improved by 22% in the latest generation",
         "source": "https://a.com/x", "confidence": 0.9},
        {"claim": "New plants added 22 GW of capacity in the latest generation",
         "source": "https://b.com/y", "confidence": 0.9},
    ]
    assert find_contradictions(facts) == []


def test_same_metric_different_period_without_years_is_numeric():
    facts = [
        {"claim": "The market grew by 25% last year", "source": "https://a.com/x"},
        {"claim": "The market grew by 11% last year", "source": "https://b.com/y"},
    ]
    found = find_contradictions(facts)
    assert found and found[0]["kind"] == "numeric"
    assert found[0]["values"]["unit"] == "%"
    assert found[0]["severity"] > 0.5


def test_agreeing_claims_not_flagged():
    facts = [
        {"claim": "The company employs 40000 people across 12 countries",
         "source": "https://a.com/x"},
        {"claim": "The company employs 40000 people across 12 countries",
         "source": "https://b.com/y"},
    ]
    # Near-identical: either deduped upstream or similarity above the band.
    assert find_contradictions(facts) == []


def test_severity_orders_numeric_above_temporal():
    numeric = find_contradictions([
        {"claim": "The market grew by 25% last year", "source": "https://a.com/x"},
        {"claim": "The market grew by 5% last year", "source": "https://b.com/y"},
    ])
    temporal = find_contradictions([
        {"claim": "Capacity reached 1200 GW in 2023", "source": "https://a.com/x"},
        {"claim": "Capacity reached 1600 GW in 2024", "source": "https://b.com/y"},
    ])
    assert numeric[0]["severity"] > temporal[0]["severity"]


def test_backward_compatible_keys_present():
    found = find_contradictions([
        {"claim": "The market grew by 25% last year", "source": "https://a.com/x"},
        {"claim": "The market grew by 5% last year", "source": "https://b.com/y"},
    ])
    c = found[0]
    for key in ("topic_similarity", "claim_a", "source_a", "value_a",
                "claim_b", "source_b", "value_b", "note"):
        assert key in c, f"missing legacy key {key}"
