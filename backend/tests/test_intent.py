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
