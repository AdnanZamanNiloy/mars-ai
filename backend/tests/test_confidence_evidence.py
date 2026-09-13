"""Step 2: the evidence-grade confidence signal.

The signal must (a) be derived from measured per-claim evidence quality,
(b) appear only when the pool is gradeable, and (c) never break confidence
when grading fails.
"""
from app.core.confidence import compute_confidence, _grade_records


STRONG = [{
    "claim": "Adoption grew 42% in 2024",
    "source": "https://www.gov.uk/report",
    "verified": True,
    "corroborating_sources": ["https://www.gov.uk/report", "https://who.int/data"],
}]
WEAK = [{"claim": "thing happened", "source": "https://blog.example/a", "verified": False}]


def test_grade_records_scales_with_evidence_quality():
    assert _grade_records(STRONG, []) == 1.0
    assert _grade_records(WEAK, []) == 0.1


def test_grade_records_none_when_pool_ungradeable():
    assert _grade_records([], []) is None
    assert _grade_records([{"no_claim": "x"}], []) is None


def test_confidence_includes_evidence_signal_for_gradeable_pool():
    result = compute_confidence(
        facts=STRONG, critique={"is_sufficient": True}, iteration=1, max_iterations=3,
    )
    assert "claim_evidence_quality" in result["signals"]
    assert result["signals"]["claim_evidence_quality"] == 1.0
    # Weights stay a valid distribution after the 0.05 carve.
    assert abs(sum(result["weights"].values()) - 1.0) < 1e-6


def test_confidence_omits_evidence_signal_for_empty_pool():
    result = compute_confidence(
        facts=[], critique={"is_sufficient": True}, iteration=1, max_iterations=3,
    )
    assert "claim_evidence_quality" not in result["signals"]


def test_strong_pool_scores_higher_than_weak_pool():
    strong = compute_confidence(
        facts=STRONG, critique={"is_sufficient": True}, iteration=1, max_iterations=3,
    )
    weak = compute_confidence(
        facts=WEAK, critique={"is_sufficient": True}, iteration=1, max_iterations=3,
    )
    assert strong["overall"] > weak["overall"]


def test_grading_failure_never_breaks_confidence(monkeypatch):
    import app.core.evidence_grade as eg
    monkeypatch.setattr(eg, "grade_facts", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    # Helper swallows the failure and returns None.
    assert _grade_records(STRONG, []) is None
    result = compute_confidence(
        facts=STRONG, critique={"is_sufficient": True}, iteration=1, max_iterations=3,
    )
    assert isinstance(result["overall"], float)
    assert "claim_evidence_quality" not in result["signals"]
