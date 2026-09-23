"""Answer support: cited sentences checked against cited evidence."""

import app.graph.workflow as wf
from app.agents.evidence_utils import verify_answer_support
from app.core.config import Settings
from app.core.llm import LLMClient

ANSWER = (
    "Retrieval augmented generation combines search with language models [1]. "
    "It was invented on Mars by pigeons [2]. "
    "This sentence is short."
    "\n\nSources:\n[1] en.wikipedia.org — https://en.wikipedia.org/wiki/RAG\n"
    "[2] example.com — https://example.com/mars-pigeons"
)

FACTS = [
    {"claim": "Retrieval augmented generation is a technique combining search with language models",
     "source": "https://en.wikipedia.org/wiki/RAG", "confidence": 0.9, "verified": True},
    {"claim": "Pigeons are birds commonly found in cities",
     "source": "https://example.com/mars-pigeons", "confidence": 0.8, "verified": True},
]


def test_supported_and_unsupported_sentences():
    result = verify_answer_support(ANSWER, FACTS)
    assert result["cited"] == 2
    assert result["supported"] == 1
    assert result["uncited"] == 0  # short sentence ignored
    assert result["rate"] == 0.5
    assert len(result["unsupported"]) == 1
    assert "pigeons" in result["unsupported"][0]


def test_unverified_facts_do_not_support():
    facts = [{**FACTS[0], "verified": False}, FACTS[1]]
    result = verify_answer_support(ANSWER, facts)
    assert result["supported"] == 0
    assert result["rate"] == 0.0


def test_no_legend_means_nothing_resolves():
    result = verify_answer_support("Some claim here [1].", FACTS)
    assert result["cited"] == 1 and result["supported"] == 0


def test_no_citations_rate_none():
    result = verify_answer_support("In my view this topic deserves much wider attention overall.", FACTS)
    assert result["cited"] == 0 and result["rate"] is None


def test_unknown_number_is_unsupported():
    answer = "Wild claim [99].\n\nSources:\n[1] en.wikipedia.org — https://en.wikipedia.org/wiki/RAG"
    result = verify_answer_support(answer, FACTS)
    assert result["cited"] == 1 and result["supported"] == 0


async def test_synthesizer_node_records_support(monkeypatch):
    """Node computes support from the emitted answer and full facts."""
    settings = Settings(groq_api_key="k", _env_file=None)

    async def fake_planner(llm, query, critique_feedback="", today="", **kwargs):
        return [{"id": 1, "question": "What is RAG today?", "axis": "definition",
                 "search_type": "encyclopedia", "priority": 1, "depends_on": [],
                 "coverage_goal": "", "domain": "general", "minimum_sources": 1,
                 "stop_condition": "enough", "variants": []}]

    async def fake_summarizer(llm, query, search_results=None, specialist_role="general"):
        return FACTS

    async def fake_critic(llm, query, facts=None, iteration=1, max_iterations=3, contradictions=None, **kwargs):
        return {"is_sufficient": True, "reason": "ok", "improved_queries": [], "confidence": 0.9}

    async def fake_synthesizer(llm, query, facts, context=None):
        return ANSWER

    monkeypatch.setattr(wf, "planner_agent", fake_planner)
    monkeypatch.setattr(wf, "summarizer_agent", fake_summarizer)
    monkeypatch.setattr(wf, "critic_agent", fake_critic)
    monkeypatch.setattr(wf, "synthesizer_agent", fake_synthesizer)
    monkeypatch.setattr(wf, "verify_facts",
                        lambda facts, search_results: [{**f, "verified": True} for f in facts])

    class StubSearch:
        async def run_search(self, questions):
            first = questions[0]
            text = first[0] if isinstance(first, (tuple, list)) else first
            return [{"url": "https://en.wikipedia.org/wiki/RAG", "sub_question": text,
                     "snippet": "snip", "content": "content"}]

    state = wf.build_initial_state("What is RAG?", 3)
    workflow = wf.create_workflow(LLMClient(settings), StubSearch())

    final = None
    async for snap in workflow.astream(state, stream_mode="values"):
        final = snap

    support = final.get("answer_support", {})
    assert support["cited"] == 2
    assert support["supported"] == 1
    assert support["rate"] == 0.5
    # The primary answer is the synthesizer's prose, not a report skeleton.
    assert final["final_report"].strip()
    assert final["final_report"] == final["synthesized_answer"]


# --- cross-source synthesis is attributed, not unsupported -------------------

_SYNTH_FACTS = [
    {"claim": "AI adoption is rising across enterprise workflows.",
     "source": "https://a.com/x", "verified": True},
    {"claim": "Inference costs have fallen sharply.",
     "source": "https://b.com/y", "verified": True},
]


def test_multi_source_synthesis_is_attributed_not_unsupported():
    answer = (
        "Taken together, these trends indicate AI is shifting toward operational "
        "deployment [1][2].\n\n"
        "Sources:\n[1] a.com — https://a.com/x\n[2] b.com — https://b.com/y"
    )
    support = verify_answer_support(answer, _SYNTH_FACTS)
    detail = support["sentence_details"][0]
    assert detail["status"] == "synthesis"
    assert detail["multi_source"] is True
    # A synthesis sentence is never counted as supported fact, and never as
    # unsupported contamination.
    assert support["synthesis"] == 1
    assert support["supported"] == 0
    assert support["unsupported"] == []


def test_fabricated_number_in_synthesis_still_fails():
    answer = (
        "Taken together, the market grew 9000 percent in 2026 [1][2].\n\n"
        "Sources:\n[1] a.com — https://a.com/x\n[2] b.com — https://b.com/y"
    )
    support = verify_answer_support(answer, _SYNTH_FACTS)
    assert support["sentence_details"][0]["status"] == "numeric_failure"
    assert support["synthesis"] == 0


def test_single_source_sentence_still_requires_support_threshold():
    # One cited source, no lexical overlap -> unsupported, never synthesis.
    answer = (
        "Quantum tunnelling drives macroeconomic inflation cycles [1].\n\n"
        "Sources:\n[1] a.com — https://a.com/x"
    )
    support = verify_answer_support(answer, _SYNTH_FACTS)
    assert support["sentence_details"][0]["status"] == "unsupported"
    assert support["sentence_details"][0]["multi_source"] is False
