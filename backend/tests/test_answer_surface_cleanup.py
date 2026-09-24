"""Phase 11.1 — final answer-surface cleanup regression tests.

Three fixes are pinned here:

  1. the internal Decision layer is hidden from the normal answer surface
     (no `decisions` NDJSON event is emitted, and the answer card no longer
     renders a decisions block);
  2. uncertainty / audit language is compressed so the answer leads with the
     best-supported conclusion (`_reduce_redundant_audit_language`);
  3. the deterministic/degraded fallback produces a direct synthesized answer
     rather than extraction/audit meta-language.

Deterministic and LLM-free.
"""
from __future__ import annotations

from app.agents.synthesizer import (
    _deterministic_report,
    _reduce_redundant_audit_language,
)
from app.core.decision import build_decision_layer


# --- 1. decision layer hidden from the answer surface ------------------------

def test_decisions_ndjson_event_is_not_emitted():
    """The route must persist decision options for the audit/trace but NOT send
    a user-facing `decisions` event (the internal Option A/B/C/D machinery)."""
    import app.api.routes as routes
    import inspect

    src = inspect.getsource(routes)
    assert 'event_line("decisions"' not in src, (
        "the internal decision layer must not be emitted on the user-facing stream"
    )
    # Persistence/audit for decisions must remain intact.
    assert "save_decisions" in src


def test_decision_options_are_not_rendered_into_the_primary_answer():
    """build_decision_layer output belongs to the audit layer only."""
    import app.graph.workflow as wf

    state = {
        "query": "Should we invest in nuclear over solar?",
        "sub_questions": [
            {"question": "nuclear cost", "axis": "cost"},
            {"question": "solar cost", "axis": "comparison"},
        ],
        "facts": [
            {"claim": "Nuclear capex is high per MW in 2025.",
             "source": "https://iea.org/a", "verified": True, "axis": "cost"},
            {"claim": "Solar capex fell in 2025.",
             "source": "https://irena.org/a", "verified": True, "axis": "comparison"},
        ],
        "synthesized_answer": "Nuclear and solar both remain viable; solar is cheaper now [1][2].",
        "intent": {"query_type": "comparative"},
    }
    options = build_decision_layer(state)
    report = wf.build_markdown_report(state, decision_options=options)
    # The primary answer is exactly the synthesized prose — no option rows.
    assert report.strip() == state["synthesized_answer"].strip()
    assert "Option A" not in report
    assert "RECOMMENDED" not in report


# --- 2. uncertainty / audit compression --------------------------------------

def test_redundant_uncertainty_is_compressed_to_one_statement():
    text = (
        "AI adoption is rising quickly [1].\n\n"
        "The evidence does not establish a reliable 2027 ranking. "
        "What the evidence does not contain is a labor forecast. "
        "This cannot be determined from the collected sources. "
        "It is unclear which roles will dominate."
    )
    out = _reduce_redundant_audit_language(text)
    # The direct answer survives untouched.
    assert "AI adoption is rising quickly [1]." in out
    # Only one uncited limitation sentence remains.
    lowered = out.lower()
    assert lowered.count("does not") <= 1
    assert "cannot be determined" not in lowered
    assert "it is unclear" not in lowered


def test_cited_limitation_is_always_preserved():
    text = (
        "The direct answer is X [1].\n\n"
        "The evidence does not establish a 2027 ranking [1]. "
        "The evidence does not establish a 2028 ranking [2]."
    )
    out = _reduce_redundant_audit_language(text)
    # Cited qualifications are substantive: both are kept (grounding intact).
    assert out.count("[1]") >= 1
    assert "[2]" in out


def test_compression_leaves_clean_text_unchanged():
    text = "AI adoption is rising [1]. Funding also rose [2]."
    assert _reduce_redundant_audit_language(text) == text


# --- 3. degraded fallback produces a direct synthesized answer ---------------

def _facts():
    return [
        {"claim": "AI adoption reached 78 percent in 2025.",
         "source": "https://mckinsey.com/a", "verified": True, "confidence": 0.9,
         "sub_question": "adoption"},
        {"claim": "AI funding hit 100 billion dollars in 2025.",
         "source": "https://cbinsights.com/a", "verified": True, "confidence": 0.85,
         "sub_question": "funding"},
        {"claim": "Model training costs fell 90 percent over 18 months.",
         "source": "https://a16z.com/b", "verified": True, "confidence": 0.8,
         "sub_question": "costs"},
    ]


def test_degraded_fallback_leads_with_a_direct_answer_not_meta_language():
    facts = _facts()
    result = _deterministic_report(
        "What is the current trend of AI?", facts, facts,
        {"verified_count": 3}, ["adoption", "funding", "costs"],
    )
    answer = result.answer
    assert result.used_fallback is True
    # No self-referential "the summary below reflects..." extraction language.
    assert "summary below reflects" not in answer.lower()
    # It leads with the best-supported finding (the answer), not process notes.
    first_line = answer.strip().splitlines()[0]
    assert "adoption reached 78 percent" in first_line.lower()
    # Process/audit accounting is NOT in the answer body.
    assert "deterministic fallback" not in answer.lower()
    assert "evidence & confidence" not in answer.lower()


def test_degraded_fallback_has_no_repeated_audit_language():
    facts = _facts()
    result = _deterministic_report(
        "What is the current trend of AI?", facts, facts,
        {"verified_count": 3}, ["adoption", "funding", "costs"],
    )
    # The compressed body does not stack limitation phrasing.
    assert result.answer.lower().count("does not establish") <= 1


def test_degraded_fallback_with_no_evidence_is_honest_and_plain():
    result = _deterministic_report(
        "What is the current trend of AI?", [], [],
        {"verified_count": 0}, [],
    )
    assert "No reliable evidence was retrieved" in result.answer
    assert "summary below reflects" not in result.answer.lower()
