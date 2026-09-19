"""Direct Answer Agent + R3 graph routing: answer without research, safely.

The properties that matter:
  * a confident, non-refusing model answer is delivered with a CAPPED
    confidence (never mistakable for a researched answer);
  * a refusal, low confidence, empty answer or provider failure falls
    through to the research pipeline;
  * the graph actually branches, and the route event/state agree.
"""

from app.agents.direct_answer import (
    MIN_ANSWER_CONFIDENCE,
    DirectAnswer,
    direct_answer_agent,
)


class FakeLLM:
    def __init__(self, payload=None, error=None, settings=None):
        self.payload = payload
        self.error = error
        self.settings = settings
        self.calls = []

    async def generate_json(self, system_prompt, user_prompt, retries=3, response_model=None):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if self.error is not None:
            raise self.error
        if response_model is not None and isinstance(self.payload, dict):
            return response_model.model_validate(self.payload).model_dump()
        return self.payload


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

async def test_confident_answer_delivered():
    llm = FakeLLM({
        "answer": "A list comprehension builds a list in one expression.",
        "needs_research": False,
        "confidence": 0.95,
        "reason": "Stable language feature.",
    })
    result = await direct_answer_agent(llm, "What is a Python list comprehension?")
    assert result.needs_research is False
    assert result.usable is True
    assert result.answer.startswith("A list comprehension")


async def test_refusal_honored():
    llm = FakeLLM({
        "answer": "",
        "needs_research": True,
        "confidence": 0.9,
        "reason": "Depends on current data.",
    })
    result = await direct_answer_agent(llm, "What is the price of gold today?")
    assert result.needs_research is True
    assert result.usable is False


async def test_needs_research_true_beats_a_present_answer():
    """Even with a non-empty answer, an explicit refusal is honoured."""
    llm = FakeLLM({
        "answer": "Probably around $2000.",
        "needs_research": True,
        "confidence": 0.99,
        "reason": "Cannot verify a live price.",
    })
    result = await direct_answer_agent(llm, "What is the price of gold today?")
    assert result.needs_research is True
    assert result.usable is False


async def test_low_confidence_answer_not_usable():
    llm = FakeLLM({
        "answer": "Something plausible.",
        "needs_research": False,
        "confidence": MIN_ANSWER_CONFIDENCE - 0.1,
        "reason": "Not sure.",
    })
    result = await direct_answer_agent(llm, "What is a for-loop?")
    assert result.usable is False


async def test_llm_failure_refuses_to_research():
    llm = FakeLLM(error=RuntimeError("provider down"))
    result = await direct_answer_agent(llm, "What is a for-loop?")
    assert result.needs_research is True
    assert result.origin == "fallback"


async def test_empty_payload_refuses_to_research():
    llm = FakeLLM(payload=None)
    result = await direct_answer_agent(llm, "What is a for-loop?")
    assert result.needs_research is True
    assert result.origin == "fallback"


def test_usable_requires_answer_and_confidence_and_no_refusal():
    assert DirectAnswer("", False, 0.9, "").usable is False
    assert DirectAnswer("x", True, 0.9, "").usable is False
    assert DirectAnswer("x", False, 0.1, "").usable is False
    assert DirectAnswer("x", False, 0.9, "").usable is True


# ---------------------------------------------------------------------------
# Graph routing (R3)
# ---------------------------------------------------------------------------

def _direct_state(query="What is a for-loop?"):
    from app.graph.workflow import build_initial_state

    return build_initial_state(query, 3, mode="quick")


async def _run_graph(route_path, answer_payload, monkeypatch, query="What is a for-loop?"):
    """Run the real graph with intent/planner/etc. stubbed, forcing a route
    path, and return (final_state, captured)."""
    import app.graph.workflow as wf
    from app.core.config import Settings
    from app.core.llm import LLMClient

    settings = Settings(groq_api_key="k", _env_file=None)
    llm = LLMClient(settings)
    captured = {"planner_called": False, "report_built": False}

    async def fake_classify(llm_arg, q, context_snippets=None):
        from app.agents.intent import heuristic_intent

        return heuristic_intent(q)

    async def fake_route(llm_arg, q, intent=None, complexity=None):
        from app.agents.router import RouteDecision

        return RouteDecision(
            path=route_path, reason="forced", confidence=0.9,
            signals={}, origin="llm",
        )

    async def fake_direct(llm_arg, q, intent=None):
        from app.agents.direct_answer import DirectAnswer as DA

        if answer_payload is None:
            return DA("", True, 0.0, "refused", origin="llm")
        return DA(answer_payload, False, 0.95, "confident", origin="llm")

    async def fake_planner(**kwargs):
        captured["planner_called"] = True
        return []

    async def fake_summarizer(*a, **k):
        return []

    async def fake_critic(**kwargs):
        return {"is_sufficient": True, "reason": "ok", "improved_queries": [], "confidence": 0.8}

    async def fake_synth(llm=None, query=None, facts=None, context=None):
        return "## Executive Summary\n\nResearched answer."

    class _FakeSearch:
        def __init__(self):
            self.settings = settings

        async def run_search(self, queries):
            return []

    monkeypatch.setattr(wf, "classify_intent", fake_classify)
    monkeypatch.setattr(wf, "route_query", fake_route)
    monkeypatch.setattr(wf, "direct_answer_agent", fake_direct)
    monkeypatch.setattr(wf, "planner_agent", fake_planner)
    monkeypatch.setattr(wf, "summarizer_agent", fake_summarizer)
    monkeypatch.setattr(wf, "critic_agent", fake_critic)
    monkeypatch.setattr(wf, "synthesizer_agent", fake_synth)

    graph = wf.create_workflow(llm, _FakeSearch())
    state = _direct_state(query)
    final = None
    async for snap in graph.astream(state, stream_mode="values"):
        final = snap
    return final, captured


async def test_direct_route_delivers_ungrounded_answer_with_capped_confidence(monkeypatch):
    final, captured = await _run_graph(
        "direct", "A for-loop iterates over a sequence.", monkeypatch
    )
    assert final["direct_answer"].startswith("A for-loop")
    # Research stages never ran.
    assert captured["planner_called"] is False
    assert final.get("sub_questions") in ([], None)
    # Confidence capped (self 0.95 -> cap 0.55) and below the sufficiency bar.
    assert final["confidence"] <= 0.55
    # The report is exactly the answer text: no research scaffolding, no
    # duplicated provenance sections.
    report = final["final_report"]
    assert report.strip() == "A for-loop iterates over a sequence."
    assert "# Supporting Evidence" not in report
    assert "# Limitations" not in report


async def test_direct_refusal_falls_through_to_research(monkeypatch):
    """The escape hatch: a direct answer that refuses must research, not
    ship an empty report."""
    final, captured = await _run_graph("direct", None, monkeypatch)
    assert captured["planner_called"] is True
    assert final.get("direct_answer", "") == ""
    # Research report shape, not the direct one.
    assert "# Final Answer" in final["final_report"]


async def test_research_route_never_calls_direct_answer(monkeypatch):
    final, captured = await _run_graph(
        "research", "unused", monkeypatch, query="What is the latest AI news?"
    )
    assert captured["planner_called"] is True
    assert final.get("direct_answer", "") == ""
