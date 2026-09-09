"""Agent Context Isolation tests (Phase 2.9).

DoD: a specialist/worker function's call signature only receives its own
AgentContext — never the full ResearchState or another sub-question's
results; raw content never persists past the summarization call that used it.
"""
import asyncio

import app.graph.workflow as wf
from app.core.isolation import AgentContext, build_contexts
from app.core.config import Settings
from app.core.llm import LLMClient


def _make_state():
    return {
        "query": "compare X and Y",
        "sub_questions": [
            {"id": 1, "question": "what is X", "axis": "definition", "search_type": "encyclopedia",
             "priority": 1, "depends_on": [], "coverage_goal": "", "domain": "general",
             "minimum_sources": 2, "stop_condition": "enough"},
            {"id": 2, "question": "X vs Y costs", "axis": "comparison", "search_type": "statistical",
             "priority": 1, "depends_on": [], "coverage_goal": "", "domain": "economics",
             "minimum_sources": 2, "stop_condition": "enough"},
        ],
        "search_results": [
            {"url": "https://arxiv.org/a", "sub_question": "what is X", "content": "RAW CONTENT OF X SOURCE", "snippet": "x snippet"},
            {"url": "https://arxiv.org/b", "sub_question": "what is X", "content": "MORE X RAW", "snippet": "x snippet 2"},
            {"url": "https://arxiv.org/c", "sub_question": "X vs Y costs", "content": "RAW COSTS CONTENT", "snippet": "costs snippet"},
        ],
        "facts": [],
        "critique": {},
        "iteration": 0,
        "max_iterations": 3,
        "confidence_history": [],
    }


def test_contexts_are_scoped_to_own_results():
    state = _make_state()
    contexts = build_contexts(state["sub_questions"], state["search_results"])
    assert len(contexts) == 2

    definition_ctx = next(c for c in contexts if c.axis() == "definition")
    comparison_ctx = next(c for c in contexts if c.axis() == "comparison")

    assert {r["url"] for r in definition_ctx.own_results} == {"https://arxiv.org/a", "https://arxiv.org/b"}
    assert {r["url"] for r in comparison_ctx.own_results} == {"https://arxiv.org/c"}
    assert all("RAW COSTS" not in str(r.get("content", "")) for r in definition_ctx.own_results)
    assert all("RAW CONTENT OF X" not in str(r.get("content", "")) for r in comparison_ctx.own_results)


async def test_worker_receives_context_not_state():
    """The summarizer worker's call signature gets AgentContext data only."""
    state = _make_state()
    contexts = build_contexts(state["sub_questions"], state["search_results"])
    received = []

    async def worker(ctx: AgentContext):
        received.append(ctx)
        return [{"claim": "c", "source": "https://arxiv.org/a", "confidence": 0.9}]

    await asyncio.gather(*(worker(ctx) for ctx in contexts))
    assert len(received) == 2
    for ctx in received:
        assert isinstance(ctx, AgentContext)
        assert not hasattr(ctx, "get"), "AgentContext must not be a dict/state-like object"
        assert ctx.question() in {"what is X", "X vs Y costs"}


async def test_raw_content_dropped_after_summarization(monkeypatch):
    """DoD: raw `content` fields never reach shared state past summarization."""
    captured_results = []

    async def fake_summarizer(llm, query, search_results=None):
        captured_results.extend(search_results or [])
        return [{"claim": "fact", "source": search_results[0]["url"], "confidence": 0.9}] if search_results else []

    async def fake_critic(llm, query, facts=None, iteration=1, max_iterations=3):
        return {"is_sufficient": True, "reason": "ok", "improved_queries": [], "confidence": 0.9}

    async def fake_planner(llm, query, critique_feedback=""):
        return _make_state()["sub_questions"]

    async def fake_synthesizer(llm, query, facts=None):
        return "answer"

    monkeypatch.setattr(wf, "summarizer_agent", fake_summarizer)
    monkeypatch.setattr(wf, "critic_agent", fake_critic)
    monkeypatch.setattr(wf, "planner_agent", fake_planner)
    monkeypatch.setattr(wf, "synthesizer_agent", fake_synthesizer)

    settings = Settings(groq_api_key="k", _env_file=None)

    class NoSearch:
        async def run_search(self, questions):
            return _make_state()["search_results"]

    workflow = wf.create_workflow(LLMClient(settings), NoSearch())

    state = _make_state()
    final = None
    async for snap in workflow.astream(state, stream_mode="values"):
        final = snap

    assert final is not None
    # After the summarizer node, no raw content survives in state.
    contents = [r.get("content", "") for r in final.get("search_results", [])]
    assert all(not c for c in contents), f"raw content leaked into shared state: {contents}"
    # Snippets remain for transparency/verification.
    assert any(r.get("snippet") for r in final.get("search_results", []))
