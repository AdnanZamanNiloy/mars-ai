"""Step 3: evidence-first sufficiency gate + counter-evidence queries.

A critic's "sufficient" verdict must not finalize a run whose evidence base
still has uncorroborated important claims or unresolved contradictions. The
gate only ever ADDS research; when the pool is clean/empty/ungradeable the
critic wins exactly as before.
"""
from app.graph.workflow import _evidence_gaps_remain, _counter_evidence_queries


def _state(facts, contradictions=None):
    return {
        "facts": facts,
        "contradictions": contradictions or [],
        "sub_questions": [{"question": "q", "axis": "evidence"}],
        "iteration": 1,
        "max_iterations": 3,
        "critique": {"is_sufficient": True, "improved_queries": [], "reason": "ok"},
        "search_results": [],
        "confidence": 0.8,
        "confidence_history": [0.8],
        "mode": "standard",
    }


UNCORROBORATED_QUANT = [{
    "claim": "Adoption grew 42% in 2024",
    "source": "https://blog.example.com/a",
    "verified": True,
}]

CORROBORATED = [{
    "claim": "Adoption grew 42% in 2024",
    "source": "https://www.gov.uk/report",
    "verified": True,
    "corroborating_sources": ["https://www.gov.uk/report", "https://who.int/data"],
}]


def test_gap_detected_for_single_source_quantitative_claim():
    assert _evidence_gaps_remain(_state(UNCORROBORATED_QUANT)) is True


def test_no_gap_for_corroborated_primary_claim():
    assert _evidence_gaps_remain(_state(CORROBORATED)) is False


def test_no_gap_for_empty_pool():
    assert _evidence_gaps_remain(_state([])) is False


def test_gap_detected_for_contradiction():
    state = _state(
        CORROBORATED,
        contradictions=[{"claim_a": CORROBORATED[0]["claim"], "claim_b": "other", "kind": "numeric"}],
    )
    assert _evidence_gaps_remain(state) is True


def test_grading_failure_is_neutral(monkeypatch):
    import app.core.evidence_grade as eg
    monkeypatch.setattr(eg, "grade_facts", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert _evidence_gaps_remain(_state(UNCORROBORATED_QUANT)) is False


def test_counter_evidence_query_for_uncorroborated_claim():
    qs = _counter_evidence_queries(_state(UNCORROBORATED_QUANT))
    assert qs and any("corroboration" in q.lower() or "verification" in q.lower() for q in qs)


def test_counter_evidence_query_for_contradiction():
    state = _state(
        CORROBORATED,
        contradictions=[{"claim_a": CORROBORATED[0]["claim"], "claim_b": "other", "kind": "numeric"}],
    )
    qs = _counter_evidence_queries(state)
    assert qs and any("conflict" in q.lower() or "disagree" in q.lower() for q in qs)


def test_counter_evidence_empty_for_clean_pool():
    assert _counter_evidence_queries(_state(CORROBORATED)) == []
