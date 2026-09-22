"""Query Router Agent: direct-answer vs. research decision (R1).

R1 is unwired — these tests pin the decision logic, not the graph wiring.
The two properties that matter most:
  * hard signals (freshness/quantitative/decision/contested/type/ambiguity)
    always force research, even against a model that says "direct";
  * the deterministic fallback NEVER grants a direct answer, so a router
    failure degrades to research, never to an ungrounded answer.
"""

from app.agents.router import (
    DEFAULT_MIN_DIRECT_CONFIDENCE,
    DIRECT,
    RESEARCH,
    RouteDecision,
    deterministic_route,
    route_query,
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
        return self.payload


# ---------------------------------------------------------------------------
# Deterministic gate
# ---------------------------------------------------------------------------

def test_freshness_forces_research():
    d = deterministic_route("What is the latest price of gold?")
    assert d.path == RESEARCH
    assert "freshness" in d.signals["hard_blockers"]


def test_bare_year_forces_research():
    d = deterministic_route("How many electric vehicles were sold in 2025?")
    assert d.path == RESEARCH
    assert "freshness" in d.signals["hard_blockers"]


def test_quantitative_forces_research():
    d = deterministic_route("What is the market size of the battery industry?")
    assert d.path == RESEARCH
    assert "quantitative" in d.signals["hard_blockers"]


def test_decision_framing_forces_research():
    d = deterministic_route("Should we invest in nuclear energy for our grid?")
    assert d.path == RESEARCH
    assert "decision" in d.signals["hard_blockers"]


def test_contested_topic_forces_research():
    d = deterministic_route("Is this supplement safe to take daily?")
    assert d.path == RESEARCH
    assert "contested" in d.signals["hard_blockers"]


def test_comparative_query_forces_research():
    d = deterministic_route("Solar vs nuclear energy for Bangladesh")
    assert d.path == RESEARCH
    assert "query_type" in d.signals["hard_blockers"]


def test_ambiguity_forces_research():
    intent = {"ambiguity": True, "query_type": "factual"}
    d = deterministic_route("What is transformer?", intent=intent)
    assert d.path == RESEARCH
    assert "ambiguity" in d.signals["hard_blockers"]


def test_deterministic_never_grants_direct():
    """The fallback can block, never grant: an unremarkable factual query
    still routes to research until the model explicitly clears it."""
    d = deterministic_route("What is a for-loop?")
    assert d.path == RESEARCH
    assert d.origin == "heuristic"


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

async def test_model_clearance_grants_direct():
    llm = FakeLLM({
        "path": "direct",
        "confidence": 0.9,
        "reason": "Stable programming concept.",
        "needs_research": False,
        "answer_sketch": "A for-loop iterates over a sequence.",
    })
    d = await route_query(llm, "What is a for-loop?")
    assert d.path == DIRECT
    assert d.origin == "llm"
    assert d.answer_sketch.startswith("A for-loop")


async def test_model_direct_vetoed_by_freshness_gate():
    """A model that wrongly says 'direct' on a freshness query is overridden
    by the deterministic gate — and no LLM call is even made."""
    llm = FakeLLM({
        "path": "direct",
        "confidence": 0.99,
        "reason": "I know this.",
        "needs_research": False,
        "answer_sketch": "Gold is about $2000.",
    })
    d = await route_query(llm, "What is the latest price of gold?")
    assert d.path == RESEARCH
    assert llm.calls == [], "hard-blocked queries skip the LLM call entirely"


async def test_low_confidence_stays_research():
    llm = FakeLLM({
        "path": "direct",
        "confidence": DEFAULT_MIN_DIRECT_CONFIDENCE - 0.1,
        "reason": "Probably know it.",
        "needs_research": False,
        "answer_sketch": "Answer.",
    })
    d = await route_query(llm, "What is a for-loop?")
    assert d.path == RESEARCH


async def test_explicit_needs_research_stays_research():
    llm = FakeLLM({
        "path": "direct",
        "confidence": 0.99,
        "reason": "Uncertain.",
        "needs_research": True,
        "answer_sketch": "",
    })
    d = await route_query(llm, "What is a for-loop?")
    assert d.path == RESEARCH


async def test_llm_failure_falls_back_to_research():
    llm = FakeLLM(error=RuntimeError("provider down"))
    d = await route_query(llm, "What is a for-loop?")
    assert d.path == RESEARCH
    assert d.origin == "heuristic"


async def test_empty_payload_falls_back_to_research():
    llm = FakeLLM(payload={})
    d = await route_query(llm, "What is a for-loop?")
    assert d.path == RESEARCH
    assert d.origin == "heuristic"


async def test_router_disabled_uses_deterministic():
    from app.core.config import Settings

    settings = Settings(router_enabled=False, groq_api_key="k", _env_file=None)
    llm = FakeLLM(payload={"path": "direct", "confidence": 0.99,
                           "reason": "", "needs_research": False}, settings=settings)
    d = await route_query(llm, "What is a for-loop?")
    assert d.path == RESEARCH
    assert llm.calls == []


async def test_min_direct_confidence_setting_honored():
    from app.core.config import Settings

    settings = Settings(router_min_direct_confidence=0.95, groq_api_key="k", _env_file=None)
    llm = FakeLLM(payload={"path": "direct", "confidence": 0.9,
                           "reason": "", "needs_research": False}, settings=settings)
    d = await route_query(llm, "What is a for-loop?")
    assert d.path == RESEARCH


def test_to_dict_shape():
    d = RouteDecision(path=DIRECT, reason="r", confidence=0.9)
    assert set(d.to_dict()) == {
        "path", "reason", "confidence", "signals", "origin", "answer_sketch"
    }


# ---------------------------------------------------------------------------
# R5: conversation path — greetings/meta are not research questions
# ---------------------------------------------------------------------------

from app.agents.router import CONVERSATION, conversation_kind  # noqa: E402


def test_greetings_route_to_conversation():
    for q in ("hey", "hi", "hello", "hello there", "hi again", "good morning", "yo"):
        d = deterministic_route(q)
        assert d.path == CONVERSATION, q
        assert d.signals.get("conversation_kind") == "greeting"
        assert d.answer_sketch, q


def test_thanks_meta_farewell_route_to_conversation():
    assert deterministic_route("thanks!").signals["conversation_kind"] == "thanks"
    assert deterministic_route("thank you so much").signals["conversation_kind"] == "thanks"
    assert deterministic_route("who are you?").signals["conversation_kind"] == "meta"
    assert deterministic_route("what can you do?").signals["conversation_kind"] == "meta"
    assert deterministic_route("bye").signals["conversation_kind"] == "farewell"


def test_real_questions_are_not_conversation():
    """A question that merely CONTAINS a greeting word must not be classified
    as conversation — whole-utterance matching only."""
    for q in ("What is a histogram?",
              "How does the internet handle high traffic?",
              "What is the history of hello in programming?"):
        assert conversation_kind(q) == "", q


def test_conversation_kind_is_bounded_and_total():
    assert conversation_kind("") == ""
    assert conversation_kind("   ") == ""
    assert conversation_kind("hi " * 100) == ""  # over the length cap


async def test_route_query_handles_conversation_without_llm_call():
    llm = FakeLLM(payload={"path": "direct", "confidence": 0.99,
                           "reason": "", "needs_research": False})
    d = await route_query(llm, "hey")
    assert d.path == CONVERSATION
    assert llm.calls == [], "conversation must never spend an LLM routing call"


async def test_graph_routes_greeting_to_conversation_not_planner(monkeypatch):
    """R5 end-to-end: a greeting must finalize through the conversation node
    without ever invoking the planner, search, or an LLM research call."""
    import app.graph.workflow as wf
    from app.core.config import Settings
    from app.core.llm import LLMClient

    settings = Settings(groq_api_key="k", _env_file=None)
    llm = LLMClient(settings)
    captured = {"planner_called": False}

    from app.agents.intent import heuristic_intent

    async def fake_classify(llm_arg, q, context_snippets=None):
        return heuristic_intent(q)

    async def fake_planner(**kwargs):
        captured["planner_called"] = True
        return []

    class _FakeSearch:
        def __init__(self):
            self.settings = settings

        async def run_search(self, queries):
            raise AssertionError("greeting must not search")

    monkeypatch.setattr(wf, "classify_intent", fake_classify)
    monkeypatch.setattr(wf, "planner_agent", fake_planner)

    graph = wf.create_workflow(llm, _FakeSearch())
    state = wf.build_initial_state("hey", 3, mode="quick")
    final = None
    async for snap in graph.astream(state, stream_mode="values"):
        final = snap

    assert captured["planner_called"] is False
    assert final["route"]["path"] == "conversation"
    assert final["direct_answer"], "the greeting reply must be delivered"
    assert "research assistant" in final["final_report"].lower()
