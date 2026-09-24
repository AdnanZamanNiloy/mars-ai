"""Phase 12 — answer conformance regression tests.

`answer_conformance.check_answer_conformance` is a deterministic,
LLM-free check that the FINISHED prose does the work the question's shape
demands, instead of merely citing evidence. Five contracts are pinned here:

  1. SHAPE CONFORMANCE — a "why" explains a mechanism; a comparison compares
     criterion by criterion and gives a verdict; a forecast marks projections
     and anchors an as-of; a decision gives a conditioned recommendation; a
     how-to gives ordered steps.
  2. INFERENCE CALIBRATION — a cross-source conclusion the analyst declared is
     not promoted to a measured fact in the prose.
  3. CONTRADICTION SYNTHESIS — a named disagreement is explained, not just
     announced.
  4. UNCERTAINTY PROPORTIONALITY — limitation language does not dominate.
  5. DEPTH FIT — a deep-research answer reasons beyond reporting.

The check is observational: failures are actionable sentences fed to the
SAME single revision pass (no new loop, no LLM call). Grounding is untouched.
"""
from __future__ import annotations

from app.agents.answer_conformance import (
    ConformanceReport,
    check_answer_conformance,
)
from app.agents.analyst import AnalyticalBrief


# --- shape conformance -------------------------------------------------------

def test_causal_question_without_mechanism_fails():
    answer = (
        "AI adoption rose to 78 percent [1]. Funding reached 100 billion [2]. "
        "Model costs fell 90 percent [3]."
    )
    report = check_answer_conformance(
        answer, "Why did AI adoption grow?", {"query_type": "causal"}
    )
    assert report.shape_conformant is False
    assert any("mechanism" in f.lower() for f in report.failures)


def test_causal_question_with_mechanism_passes():
    answer = (
        "AI adoption grew because model costs fell 90 percent [1], which made "
        "deployment affordable for smaller firms [2]. This drove the funding "
        "increase [3]."
    )
    report = check_answer_conformance(
        answer, "Why did AI adoption grow?", {"query_type": "causal"}
    )
    assert report.shape_conformant is True


def test_comparison_without_verdict_fails():
    answer = (
        "Solar costs fell in 2025 [1]. Nuclear capex remains high [2]. "
        "Nuclear provides firm capacity [3]."
    )
    report = check_answer_conformance(
        answer, "Solar vs nuclear", {"query_type": "comparison"}
    )
    assert report.shape_conformant is False
    assert any("comparison" in f.lower() for f in report.failures)


def test_comparison_with_criterion_and_verdict_passes():
    answer = (
        "Solar is cheaper than nuclear on levelised cost [1], whereas nuclear "
        "provides firm capacity that solar lacks [2]. For most new builds the "
        "better option is solar, though the verdict depends on the need for "
        "baseload power [3]."
    )
    report = check_answer_conformance(
        answer, "Solar vs nuclear", {"query_type": "comparison"}
    )
    assert report.shape_conformant is True


def test_forecast_without_projection_signal_fails():
    answer = (
        "The labour market shifted toward AI roles [1]. Demand is concentrated "
        "in engineering [2]."
    )
    report = check_answer_conformance(
        answer, "The most demanding jobs in 2027", {"query_type": "forecast"}
    )
    assert report.shape_conformant is False
    assert any("projection" in f.lower() or "forward" in f.lower()
               for f in report.failures)


def test_forecast_with_projection_and_asof_passes():
    answer = (
        "As of 2025 the market measured 40 billion [1]. It is projected to reach "
        "90 billion by 2027, assuming current adoption rates hold [2]."
    )
    report = check_answer_conformance(
        answer, "The most demanding jobs in 2027", {"query_type": "forecast"}
    )
    assert report.shape_conformant is True


def test_decision_without_conditioned_recommendation_fails():
    answer = (
        "Nuclear has high capex [1]. Solar has low capex [2]. Both have trade-offs.\n\n"
        "- Nuclear: firm power [3]\n- Solar: cheap [4]"
    )
    report = check_answer_conformance(
        answer, "Should we invest in nuclear over solar?",
        {"query_type": "decision"},
    )
    assert report.shape_conformant is False
    assert any("recommendation" in f.lower() for f in report.failures)


def test_decision_with_conditioned_recommendation_passes():
    answer = (
        "For a grid that needs firm baseload, nuclear is the better choice [1]. "
        "If you value low upfront cost over dispatchability, then solar is "
        "preferable [2]. The recommendation would change if storage costs fall "
        "below the nuclear benchmark [3]."
    )
    report = check_answer_conformance(
        answer, "Should we invest in nuclear over solar?",
        {"query_type": "decision"},
    )
    assert report.shape_conformant is True


def test_howto_without_ordered_steps_fails():
    answer = (
        "First you configure the client [1] and then you connect it. The setup "
        "involves several parameters [2]."
    )
    report = check_answer_conformance(
        answer, "How do I set up the client?", {"query_type": "howto"}
    )
    assert report.shape_conformant is False
    assert any("step" in f.lower() for f in report.failures)


def test_howto_with_numbered_steps_passes():
    answer = (
        "1. Install the client [1].\n"
        "2. Configure the API key [2].\n"
        "3. Run the connection check [3]."
    )
    report = check_answer_conformance(
        answer, "How do I set up the client?", {"query_type": "howto"}
    )
    assert report.shape_conformant is True


def test_unrecognised_shape_is_never_failed_for_structure():
    """Phase 12 forbids a rigid template: an unclassified question must not be
    failed for lacking a heading list."""
    report = check_answer_conformance(
        "A short answer with one cited fact [1].", "Tell me about X", {}
    )
    assert report.shape_conformant is True


# --- inference calibration ---------------------------------------------------

def test_unhedged_cross_source_conclusion_is_flagged():
    """The analyst declared an inference; the answer dropped the hedge and
    stated it as measured fact."""
    brief = AnalyticalBrief(
        cross_source_conclusions=[
            "the evidence suggests the two markets are converging on a shared "
            "regulatory standard",
        ],
        origin="llm",
    )
    answer = (
        "The two markets converge on a shared regulatory standard [1][2]. "
        "Adoption rose [3]. Costs fell [4]."
    )
    report = check_answer_conformance(
        answer, "How do the markets compare?", {"query_type": "comparison"},
        brief=brief,
    )
    assert report.inference_calibrated is False
    assert any("inference" in f.lower() for f in report.failures)


def test_hedged_cross_source_conclusion_is_calibrated():
    brief = AnalyticalBrief(
        cross_source_conclusions=[
            "the evidence suggests the two markets are converging on a shared "
            "regulatory standard",
        ],
        origin="llm",
    )
    answer = (
        "Taken together, the evidence suggests the two markets converge on a "
        "shared regulatory standard [1][2]. Whereas each differs in scope [3], "
        "the direction points the same way [4]."
    )
    report = check_answer_conformance(
        answer, "How do the markets compare?", {"query_type": "comparison"},
        brief=brief,
    )
    assert report.inference_calibrated is True


def test_plain_corroborated_conclusion_is_not_treated_as_inference():
    """A conclusion the brief states plainly is established content, not an
    inference — the answer may state it directly without a hedge."""
    brief = AnalyticalBrief(
        cross_source_conclusions=[
            "the two markets share a common regulatory standard",
        ],
        origin="llm",
    )
    answer = (
        "The two markets share a common regulatory standard [1][2]. Adoption "
        "rose [3]. Costs fell [4]."
    )
    report = check_answer_conformance(
        answer, "How do the markets compare?", {"query_type": "comparison"},
        brief=brief,
    )
    assert report.inference_calibrated is True


def test_missing_brief_never_penalises_inference_calibration():
    report = check_answer_conformance(
        "The evidence suggests a convergence [1].", "Compare X and Y", {}
    )
    assert report.inference_calibrated is True


# --- contradiction synthesis -------------------------------------------------

def test_named_but_unexplained_conflict_is_flagged():
    contradictions = [{
        "claim_a": "Solar LCOE is $30/MWh", "source_a": "a",
        "claim_b": "Solar LCOE is $50/MWh", "source_b": "b",
    }]
    answer = (
        "Sources disagree on the solar cost [1][2]. Adoption nevertheless rose [3]."
    )
    report = check_answer_conformance(
        answer, "What is the solar cost?", {}, contradictions=contradictions
    )
    assert report.contradiction_synthesised is False
    assert any("contradiction" in f.lower() for f in report.failures)


def test_explained_conflict_passes():
    contradictions = [{
        "claim_a": "Solar LCOE is $30/MWh", "source_a": "a",
        "claim_b": "Solar LCOE is $50/MWh", "source_b": "b",
    }]
    answer = (
        "Sources disagree on the solar cost [1][2]; the gap likely reflects "
        "different timeframes, since one figure is unsubsidised and the other "
        "includes storage [3]."
    )
    report = check_answer_conformance(
        answer, "What is the solar cost?", {}, contradictions=contradictions
    )
    assert report.contradiction_synthesised is True


def test_no_conflict_needs_no_synthesis():
    report = check_answer_conformance(
        "Costs fell [1]. Adoption rose [2].", "What is the trend?", {}
    )
    assert report.contradiction_synthesised is True


# --- uncertainty proportionality ---------------------------------------------

def test_stacked_limitations_are_flagged():
    answer = (
        "The answer is a rising trend [1]. "
        "The evidence does not establish a ranking. "
        "The data is insufficient to compare regions. "
        "It remains uncertain which factor dominates. "
        "We do not know the 2027 outcome. "
        "No evidence sets a precise figure. "
        "Adoption still increased overall [2]."
    )
    report = check_answer_conformance(answer, "What is the trend?", {})
    assert report.uncertainty_proportionate is False
    assert report.limitation_ratio > 0.30
    assert any("proportionality" in f.lower() for f in report.failures)


def test_one_limitation_is_proportionate():
    answer = (
        "Adoption rose 20 percent [1]. Funding doubled [2]. Costs fell [3]. "
        "The evidence does not establish a 2027 ranking [4]."
    )
    report = check_answer_conformance(answer, "What is the trend?", {})
    assert report.uncertainty_proportionate is True


# --- depth fit ---------------------------------------------------------------

def test_shallow_deep_answer_is_flagged():
    answer = (
        "Adoption rose 20 percent [1]. Funding doubled [2]. Costs fell [3]. "
        "Headcount grew [4]."
    )
    report = check_answer_conformance(
        answer, "Analyse the state of the field", {}, mode="deep"
    )
    assert report.depth_fit is False
    assert any("deep" in f.lower() for f in report.failures)


def test_reasoned_deep_answer_passes():
    answer = (
        "Adoption rose 20 percent because costs fell [1], whereas funding grew "
        "more slowly [2]. However, the trade-off is that scaling raises "
        "infrastructure demand [3]. Taken together, this suggests growth is "
        "real but supply-constrained [4]."
    )
    report = check_answer_conformance(
        answer, "Analyse the state of the field", {}, mode="deep"
    )
    assert report.depth_fit is True


def test_simple_question_is_not_penalised_for_brevity():
    report = check_answer_conformance(
        "The capital of France is Paris [1].", "What is the capital of France?",
        {"query_type": "definition"}, mode="quick",
    )
    assert report.depth_fit is True


# --- totality / safety -------------------------------------------------------

def test_empty_answer_is_neutral():
    report = check_answer_conformance("", "Any question", {})
    assert isinstance(report, ConformanceReport)
    assert report.failures == []
    assert report.score == 1.0


def test_report_serializes():
    report = check_answer_conformance(
        "A statement [1].", "Why X?", {"query_type": "causal"}
    )
    data = report.to_dict()
    assert data["query_type"] == "causal"
    assert "score" in data and "failures" in data


def test_score_is_fraction_of_checks_passed():
    # A clean comparison question with no brief/conflicts/mode pressure passes
    # all five checks and scores 1.0.
    answer = (
        "Solar is cheaper than nuclear [1], whereas nuclear is firmer [2]. For "
        "most builds solar is the better option [3]."
    )
    report = check_answer_conformance(answer, "Solar vs nuclear",
                                       {"query_type": "comparison"})
    assert report.score == 1.0


# --- workflow integration: conformance reaches the audit layer ---------------

import app.graph.workflow as wf
from app.core.config import Settings
from app.core.llm import LLMClient


async def _run_stub_workflow(monkeypatch, query, answer, intent_query_type,
                             mode="standard"):
    """Run the production graph with stub agents; return final state + the list
    of synthesizer prompts (so a test can inspect revision feedback)."""
    settings = Settings(groq_api_key="k", _env_file=None)
    prompts = []

    async def fake_planner(llm, query, critique_feedback="", today="", **kwargs):
        return [{"id": 1, "question": f"what is {query}", "axis": "definition",
                 "search_type": "encyclopedia", "priority": 1, "depends_on": [],
                 "coverage_goal": "", "domain": "general", "minimum_sources": 1,
                 "stop_condition": "enough", "variants": []}]

    async def fake_summarizer(llm, query, search_results=None,
                              specialist_role="general"):
        return [{"claim": "Solar is cheaper than nuclear in 2025.",
                 "source": "https://a.com/x", "verified": True,
                 "sub_question": "definition"}]

    async def fake_critic(llm, query, facts=None, iteration=1,
                          max_iterations=3, contradictions=None, **kwargs):
        return {"is_sufficient": True, "reason": "ok", "improved_queries": [],
                "confidence": 0.9}

    async def fake_synth(llm, query, facts, context=None):
        prompts.append(str((context or {}).get("quality_feedback") or []))
        return answer

    class StubSearch:
        async def run_search(self, questions):
            first = questions[0]
            text = first[0] if isinstance(first, (tuple, list)) else first
            return [{"url": "https://a.com/x", "sub_question": text,
                     "snippet": "snip", "content": "content"}]

    monkeypatch.setattr(wf, "planner_agent", fake_planner)
    monkeypatch.setattr(wf, "summarizer_agent", fake_summarizer)
    monkeypatch.setattr(wf, "critic_agent", fake_critic)
    monkeypatch.setattr(wf, "synthesizer_agent", fake_synth)
    monkeypatch.setattr(
        wf, "verify_facts",
        lambda facts, search_results: [{**f, "verified": True} for f in facts],
    )

    state = wf.build_initial_state(query, 3, mode=mode)
    state["intent"] = {"query_type": intent_query_type}
    workflow = wf.create_workflow(LLMClient(settings), StubSearch())
    final = {}
    async for snap in workflow.astream(state, stream_mode="values"):
        final = snap
    return final, prompts


async def test_conformance_is_carried_into_final_state_and_audit(monkeypatch):
    """The conformance report reaches state and the audit document, while the
    primary answer stays exactly the synthesizer's prose."""
    answer = (
        "Solar is cheaper than nuclear [1]. Nuclear is firmer [2].\n\n"
        "Sources:\n[1] a.com — https://a.com/x\n[2] a.com — https://a.com/x"
    )
    final, prompts = await _run_stub_workflow(
        monkeypatch, "What is the trend?", answer, "comparison", mode="quick"
    )
    conf = final.get("answer_conformance")
    assert isinstance(conf, dict) and conf.get("query_type")
    # The answer is unchanged: conformance is observational, never rewrites.
    assert final["final_report"] == final["synthesized_answer"]
    audit = final.get("final_audit", "")
    assert "Answer conformance" in audit


async def test_conformance_failure_is_actionable_and_does_not_crash_the_run(
    monkeypatch,
):
    """A non-conformant comparison answer must still ship (fail-safe) and its
    run must not raise; the failure is available for the revision pass."""
    answer = (
        "Solar is cheaper than nuclear [1]. Nuclear is firmer [2].\n\n"
        "Sources:\n[1] a.com — https://a.com/x\n[2] a.com — https://a.com/x"
    )
    final, prompts = await _run_stub_workflow(
        monkeypatch, "Compare solar and nuclear", answer, "comparison",
        mode="standard",
    )
    conf = final.get("answer_conformance") or {}
    # The comparison has no criterion/verdict, so shape conformance is missed.
    assert conf.get("shape_conformant") is False
    assert final["final_report"].strip()
    # The failure is actionable: it rides the SAME single revision pass (the
    # second synthesizer call carries it in quality_feedback). No new loop.
    assert len(prompts) == 2
    assert "Question fit" in prompts[1]
