"""Decision Intelligence Layer tests (Phase 3.5).

DoD: a comparative query produces 2+ genuinely different options, exactly
one recommended, rationale referencing verified claims, risk per option;
a factual query produces NO options — no Decision Layer section at all.
"""
from app.core.decision import build_decision_layer


def _comparative_state(**overrides):
    state = {
        "query": "Should Bangladesh invest in nuclear vs solar energy?",
        "orchestration": {"query_type": "comparative"},
        "sub_questions": [
            {"axis": "cost", "question": "costs"},
            {"axis": "financing", "question": "financing"},
            {"axis": "policy", "question": "policy"},
        ],
        "facts": [
            {"claim": "Nuclear LCOE is higher than solar LCOE in cost terms", "source": "https://iea.org/a", "verified": True, "confidence": 0.9},
            {"claim": "Cost estimates for nuclear vary across studies", "source": "https://arxiv.org/b", "verified": True, "confidence": 0.8},
            {"claim": "Financing terms drive nuclear project viability for financing", "source": "https://imf.org/c", "verified": True, "confidence": 0.85},
            {"claim": "Policy support exists for solar", "source": "https://worldbank.org/d", "verified": False, "confidence": 0.5},
        ],
        "contradictions": [
            {"claim_a": "25% growth", "source_a": "a", "claim_b": "40% growth", "source_b": "b"},
        ],
    }
    state.update(overrides)
    return state


def test_comparative_query_yields_multiple_distinct_options():
    options = build_decision_layer(_comparative_state())
    assert len(options) >= 2
    descriptions = {o["description"] for o in options}
    assert len(descriptions) == len(options), "options must be genuinely distinct"
    labels = [o["option_label"] for o in options]
    assert labels == ["A", "B", "C"][: len(options)]


def test_exactly_one_recommended_and_it_has_most_support():
    options = build_decision_layer(_comparative_state())
    recommended = [o for o in options if o["is_recommended"]]
    assert len(recommended) == 1
    # Cost axis has 2 verified supporting claims — the most of any axis.
    assert recommended[0]["description"].startswith("Frame the decision primarily around the 'cost'")


def test_rationale_references_verified_claims():
    options = build_decision_layer(_comparative_state())
    for o in options:
        assert "verified claim" in o["rationale"]


def test_every_option_has_risk_note():
    options = build_decision_layer(_comparative_state())
    assert all(o.get("risk_note") for o in options)
    # Contradictions are reflected in risk notes.
    assert any("contradiction" in o["risk_note"] for o in options)


def test_factual_query_produces_no_options():
    """Factual/definitional queries emit no decision options: the old
    'no material decision' placeholder read as manufactured filler in the
    UI and the report."""
    state = {
        "query": "what is RAG",
        "orchestration": {"query_type": "factual"},
        "sub_questions": [{"axis": "definition", "question": "q"}],
        "facts": [],
        "contradictions": [],
    }
    assert build_decision_layer(state) == []


def test_comparative_with_single_axis_produces_no_options():
    state = _comparative_state(sub_questions=[{"axis": "cost", "question": "costs"}])
    assert build_decision_layer(state) == []


def test_factual_report_has_no_decision_layer_in_answer_or_audit():
    import app.graph.workflow as wf

    state = {
        "query": "what is RAG",
        "orchestration": {"query_type": "factual"},
        "sub_questions": [{"axis": "definition", "question": "q"}],
        "facts": [],
        "contradictions": [],
        "synthesized_answer": "answer",
        "critique": {"is_sufficient": True},
    }
    report = wf.build_markdown_report(state)
    # The primary answer is the synthesizer's text, verbatim — no injected
    # report skeleton, and so no decision layer.
    assert report == "answer"
    audit = wf.build_answer_audit(state)
    assert "Decision layer" not in audit


def test_decision_layer_lives_in_audit_not_in_the_answer():
    """Evidence/judgment separation is now structural: the answer carries the
    synthesized prose only, and the strategic options are rendered in the
    separate audit document (never in the answer body)."""
    import app.graph.workflow as wf

    state = _comparative_state()
    state["synthesized_answer"] = "answer"
    state["critique"] = {"is_sufficient": True}
    report = wf.build_markdown_report(state)
    audit = wf.build_answer_audit(state)
    # The answer is untouched: no Decision Layer, no injected headings.
    assert "# Decision Layer" not in report
    assert report == "answer"
    # The audit carries the options and their recommendation.
    assert "Decision layer" in audit
    assert "(RECOMMENDED)" in audit
