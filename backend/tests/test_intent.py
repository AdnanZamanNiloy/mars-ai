"""Intent Classification Agent: ambiguity resolution before research."""

import pytest

from app.agents.intent import (
    IntentReport,
    classify_intent,
    heuristic_intent,
)


class FakeLLM:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    async def generate_json(self, system_prompt, user_prompt, retries=3, response_model=None):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if self.error is not None:
            raise self.error
        if response_model is not None:
            return response_model.model_validate(self.payload).model_dump()
        return self.payload


def test_fallback_flags_transformer_ambiguity():
    report = heuristic_intent("What is transformer?")
    assert report.ambiguity is True
    assert len(report.senses) == 2
    assert report.senses[0].domain == "machine_learning"
    assert report.senses[1].domain == "engineering"
    # 0.75 vs 0.20 gap > 0.30 -> research the dominant sense only.
    assert report.recommended_action == "research_dominant"
    assert report.research_senses == [report.senses[0]]


def test_fallback_unambiguous_query_passes_through():
    report = heuristic_intent("What is retrieval augmented generation?")
    assert report.ambiguity is False
    assert report.senses == []
    assert report.query_type == "factual"
    assert report.recommended_action == "research_dominant"


def test_fallback_levels():
    assert heuristic_intent("explain transformer simply").explanation_level == "basic"
    assert heuristic_intent("transformer architecture internals").explanation_level == "expert"
    assert heuristic_intent("What is transformer?").explanation_level == "practical"


def test_fallback_classifies_question_type():
    assert heuristic_intent("nuclear vs solar energy in Bangladesh").query_type == "comparative"
    assert heuristic_intent("should we invest in nuclear energy").query_type == "analytical"


async def test_llm_intent_used_when_valid():
    llm = FakeLLM({
        "query_type": "factual",
        "domain": "electrical_engineering",
        "explanation_level": "practical",
        "ambiguity": True,
        "senses": [
            {"label": "Transformer neural network architecture", "domain": "machine_learning",
             "probability": 0.6, "note": "AI sense"},
            {"label": "Electrical transformer", "domain": "electrical_engineering",
             "probability": 0.35, "note": "device sense"},
        ],
        "reasoning": "likely AI, plausibly electrical",
    })
    report = await classify_intent(llm, "What is transformer?")
    assert report.origin == "llm"
    assert report.ambiguity is True
    # probabilities ranked most-likely first
    assert report.senses[0].probability >= report.senses[1].probability
    # non-planner domains aliased into the planner vocabulary
    assert report.domain == "engineering"
    assert report.senses[1].domain == "engineering"
    # gap 0.25 < 0.30 -> both senses researched
    assert report.recommended_action == "research_both"
    assert len(report.research_senses) == 2


async def test_llm_hedge_cannot_hide_a_real_second_sense():
    """The deterministic ambiguity floor: a model flagging ambiguity=false with
    a 0.4 second sense must not silently drop the alternative reading."""
    llm = FakeLLM({
        "query_type": "factual",
        "domain": "machine_learning",
        "explanation_level": "practical",
        "ambiguity": False,
        "senses": [
            {"label": "AI architecture", "domain": "machine_learning", "probability": 0.55},
            {"label": "Electrical device", "domain": "engineering", "probability": 0.40},
        ],
        "reasoning": "",
    })
    report = await classify_intent(llm, "What is transformer?")
    assert report.ambiguity is True


async def test_llm_failure_falls_back_deterministically():
    llm = FakeLLM(error=RuntimeError("provider down"))
    report = await classify_intent(llm, "What is transformer?")
    assert report.origin == "heuristic"
    assert report.ambiguity is True
    assert report.senses[0].domain == "machine_learning"


async def test_llm_invalid_payload_falls_back():
    llm = FakeLLM(payload={"senses": "not-a-list", "query_type": "nonsense"})
    report = await classify_intent(llm, "What is python?")
    assert report.origin == "heuristic"


async def test_context_snippets_reach_the_prompt():
    llm = FakeLLM(payload={
        "query_type": "factual", "domain": "machine_learning",
        "explanation_level": "practical", "ambiguity": False,
        "senses": [], "reasoning": "ok",
    })
    await classify_intent(llm, "What is transformer?",
                          context_snippets=["Attention is all you need — ML architecture"])
    assert "Attention is all you need" in llm.calls[0]["user_prompt"]


def test_research_both_when_genuinely_split():
    report = IntentReport(
        query="what is jaguar", query_type="factual", domain="general",
        explanation_level="practical", ambiguity=True,
        senses=[
            __import__("app.agents.intent", fromlist=["SenseCandidate"]).SenseCandidate(
                "Jaguar car", "engineering", 0.45),
            __import__("app.agents.intent", fromlist=["SenseCandidate"]).SenseCandidate(
                "Jaguar animal", "science", 0.40),
        ],
    )
    assert report.recommended_action == "research_both"
    assert len(report.research_senses) == 2


def test_to_dict_round_trip_shape():
    report = heuristic_intent("What is transformer?")
    d = report.to_dict()
    assert set(d) == {"query", "query_type", "domain", "explanation_level", "ambiguity",
                      "senses", "recommended_action", "reasoning", "origin"}
    assert d["senses"][0]["probability"] == 0.75


def test_planner_assigns_senses_and_fallback_uses_them():
    """The intent -> planner handoff: contracts get sense tags, domains upgrade
    from general, and the deterministic fallback plans per sense."""
    from app.agents.planner import _assign_intent_senses, fallback_plan

    intent = heuristic_intent("What is transformer?").to_dict()

    plan = _assign_intent_senses([
        {"question": "transformer architecture attention mechanism", "domain": "general",
         "specialist": "general", "sense": ""},
        {"question": "transformer statistics official data", "domain": "general",
         "specialist": "general", "sense": "Electrical transformer (AC voltage device)"},
    ], intent)
    # dominant sense only -> every contract targets it
    assert all(c["sense"] == "Transformer neural network architecture" for c in plan)
    # general-domain contracts upgrade to the sense's domain + specialist
    assert plan[0]["domain"] == "machine_learning"
    assert plan[0]["specialist"] == "technical"

    fb = fallback_plan("What is transformer?", target_count=3,
                       intent=heuristic_intent("What is transformer?").to_dict())
    assert fb, "fallback plan must still build"
    senses = {c.get("sense") for c in fb}
    assert senses == {"Transformer neural network architecture"}
    assert all("neural network architecture" in c["question"] for c in fb)

    # research_both: fallback interleaves both senses
    both = dict(intent)
    both["recommended_action"] = "research_both"
    fb2 = fallback_plan("What is transformer?", target_count=4, intent=both)
    used = {c.get("sense") for c in fb2}
    assert used == {"Transformer neural network architecture",
                    "Electrical transformer (AC voltage device)"}


def test_unambiguous_intent_leaves_plan_untagged():
    from app.agents.planner import _assign_intent_senses

    intent = heuristic_intent("What is retrieval augmented generation?").to_dict()
    plan = _assign_intent_senses([
        {"question": "RAG definition", "domain": "machine_learning", "specialist": "technical", "sense": ""},
    ], intent)
    assert plan[0]["sense"] == ""


class _FakeSearchClient:
    settings = None

    async def run_search(self, queries):
        return [{
            "title": "Attention Is All You Need",
            "snippet": "The transformer is a neural network architecture for sequence transduction.",
            "url": "https://arxiv.org/abs/1706.03762",
        }]


async def test_graph_runs_intent_before_planner(monkeypatch):
    """Full-graph wiring: intent resolves first, its context search is shared,
    and the planner receives the intent report."""
    import app.graph.workflow as wf
    from app.core.config import Settings
    from app.core.llm import LLMClient

    _FakeSearchClient.settings = Settings(groq_api_key="k", _env_file=None)
    llm = LLMClient(_FakeSearchClient.settings)

    captured = {}

    async def fake_classify(llm_arg, query, context_snippets=None):
        captured["context"] = list(context_snippets or [])
        return heuristic_intent(query)

    async def fake_planner(**kwargs):
        captured["intent"] = kwargs.get("intent")
        return [{
            "id": 1, "question": "transformer neural network architecture definition",
            "axis": "definition", "search_type": "encyclopedia", "priority": 1,
            "depends_on": [], "domain": "machine_learning", "minimum_sources": 2,
            "coverage_goal": "", "stop_condition": "", "variants": [], "agent": "",
            "tools": ["web_search"], "scope": [], "output_format": "structured_findings",
            "specialist": "technical", "preferred_domains": [], "primary_source_query": "",
            "wave": 0, "sense": "",
        }]

    async def fake_summarizer(llm_arg, query, search_results, specialist_role="general",
                              prior_findings=None, sense=""):
        captured["sense"] = sense
        return []

    async def fake_critic(**kwargs):
        return {"is_sufficient": False, "reason": "thin", "improved_queries": [], "confidence": 0.4}

    async def fake_synthesizer(llm=None, query=None, facts=None, context=None):
        return "## Executive Summary\n\nAnswer."

    monkeypatch.setattr(wf, "classify_intent", fake_classify)
    monkeypatch.setattr(wf, "planner_agent", fake_planner)
    monkeypatch.setattr(wf, "summarizer_agent", fake_summarizer)
    monkeypatch.setattr(wf, "critic_agent", fake_critic)
    monkeypatch.setattr(wf, "synthesizer_agent", fake_synthesizer)

    graph = wf.create_workflow(llm, _FakeSearchClient())
    state = wf.build_initial_state("What is transformer?", 3, mode="quick")
    final = None
    async for snap in graph.astream(state, stream_mode="values"):
        final = snap

    assert captured["context"], "grounding search must feed the intent classifier"
    assert captured["intent"]["ambiguity"] is True
    assert captured["intent"]["domain"] == "machine_learning"
    # intent lands in state for the synthesizer
    assert final["intent"]["ambiguity"] is True
