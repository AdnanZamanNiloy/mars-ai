"""Per-axis expansion: plans append gap questions, search skips answered ones."""

import app.graph.workflow as wf
from app.core.budget import BudgetTracker, current_budget
from app.core.config import Settings
from app.core.llm import LLMClient
from app.graph.workflow import _merge_questions, _unanswered_questions


def _q(i: int, text: str) -> dict:
    return {"id": i, "question": text, "axis": "general", "search_type": "encyclopedia",
            "priority": 2, "depends_on": [], "coverage_goal": "", "domain": "general",
            "minimum_sources": 2, "stop_condition": "enough"}


def test_merge_questions_appends_only_new_with_continuing_ids():
    existing = [_q(1, "What is RAG?"), _q(2, "How does retrieval work?")]
    new = [_q(1, "What is RAG?"), _q(2, "What are RAG benchmarks 2026?")]
    merged = _merge_questions(existing, new)
    assert [q["question"] for q in merged] == [
        "What is RAG?", "How does retrieval work?", "What are RAG benchmarks 2026?",
    ]
    assert [q["id"] for q in merged] == [1, 2, 3]


def test_merge_questions_empty_existing():
    merged = _merge_questions([], [_q(1, "Fresh question here?")])
    assert len(merged) == 1 and merged[0]["id"] == 1


def test_unanswered_questions_skips_searched():
    subs = [_q(1, "What is RAG?"), _q(2, "How does retrieval work?")]
    results = [{"url": "https://a.com", "sub_question": "What is RAG?"}]
    assert _unanswered_questions(subs, results) == ["How does retrieval work?"]
    assert _unanswered_questions(subs, []) == ["What is RAG?", "How does retrieval work?"]
    assert _unanswered_questions(subs, None) == ["What is RAG?", "How does retrieval work?"]


async def test_expansion_searches_only_new_questions(monkeypatch):
    """Full-graph: pass 2 searches just the gap question, results accumulate."""
    settings = Settings(groq_api_key="k", _env_file=None)
    tracker = BudgetTracker(settings)
    token = current_budget.set(tracker)

    plans = [
        [_q(1, "What is RAG today?"), _q(2, "How does dense retrieval work now?")],
        [_q(1, "What are RAG benchmarks this year?")],
    ]
    search_inputs = []

    async def fake_planner(llm, query, critique_feedback="", today=""):
        return plans.pop(0)

    async def fake_summarizer(llm, query, search_results=None, specialist_role="general"):
        return [{"claim": f"Fact about {r.get('sub_question', '')[:20]} is established",
                 "source": r.get("url", ""), "confidence": 0.9}
                for r in (search_results or [])]

    calls = {"n": 0}

    async def fake_critic(llm, query, facts=None, iteration=1, max_iterations=3, contradictions=None):
        calls["n"] += 1
        if calls["n"] >= 2:
            return {"is_sufficient": True, "reason": "ok", "improved_queries": [], "confidence": 0.9}
        return {"is_sufficient": False, "reason": "thin", "improved_queries": ["RAG benchmarks 2026"],
                "confidence": 0.4}

    async def fake_synthesizer(llm, query, facts):
        return "synthesized answer"

    monkeypatch.setattr(wf, "planner_agent", fake_planner)
    monkeypatch.setattr(wf, "summarizer_agent", fake_summarizer)
    monkeypatch.setattr(wf, "critic_agent", fake_critic)
    monkeypatch.setattr(wf, "synthesizer_agent", fake_synthesizer)
    monkeypatch.setattr(wf, "verify_facts", lambda facts, search_results: facts)

    class StubSearch:
        async def run_search(self, questions):
            search_inputs.append(list(questions))
            return [{"url": f"https://test.com/{q[:10]}", "sub_question": q,
                     "snippet": "snip", "content": "content here"} for q in questions]

    state = wf.build_initial_state("What is RAG?", 3)
    state["budget_tracker"] = tracker
    workflow = wf.create_workflow(LLMClient(settings), StubSearch())

    final = None
    async for snap in workflow.astream(state, stream_mode="values"):
        final = snap
    current_budget.reset(token)

    assert calls["n"] == 2
    assert search_inputs[0] == ["What is RAG today?", "How does dense retrieval work now?"]
    assert search_inputs[1] == ["What are RAG benchmarks this year?"], search_inputs
    assert len(final["sub_questions"]) == 3
    assert len(final["search_results"]) == 3
    assert "# Final Answer" in final["final_report"]


async def test_variant_queries_searched_and_attributed(monkeypatch):
    """Variants fan out in search and their results reach the parent
    contract's summarizer context (no orphans)."""
    settings = Settings(groq_api_key="k", _env_file=None)
    tracker = BudgetTracker(settings)
    token = current_budget.set(tracker)
    search_inputs = []

    async def fake_planner(llm, query, critique_feedback="", today=""):
        return [{**_q(1, "What is RAG today?"),
                 "variants": ["RAG definition overview 2026"]}]

    async def fake_summarizer(llm, query, search_results=None, specialist_role="general"):
        return [{"claim": f"Finding about {r.get('sub_question', '')[:30]} is documented here",
                 "source": r.get("url", ""), "confidence": 0.9}
                for r in (search_results or [])]

    async def fake_critic(llm, query, facts=None, iteration=1, max_iterations=3, contradictions=None):
        return {"is_sufficient": True, "reason": "ok", "improved_queries": [], "confidence": 0.9}

    async def fake_synthesizer(llm, query, facts):
        return "synthesized answer"

    monkeypatch.setattr(wf, "planner_agent", fake_planner)
    monkeypatch.setattr(wf, "summarizer_agent", fake_summarizer)
    monkeypatch.setattr(wf, "critic_agent", fake_critic)
    monkeypatch.setattr(wf, "synthesizer_agent", fake_synthesizer)
    monkeypatch.setattr(wf, "verify_facts", lambda facts, search_results: facts)

    class StubSearch:
        async def run_search(self, questions):
            search_inputs.append(list(questions))
            return [{"url": f"https://test.com/{i}", "sub_question": q,
                     "snippet": "snip", "content": "content here"}
                    for i, q in enumerate(questions)]

    state = wf.build_initial_state("What is RAG?", 3)
    state["budget_tracker"] = tracker
    workflow = wf.create_workflow(LLMClient(settings), StubSearch())

    final = None
    async for snap in workflow.astream(state, stream_mode="values"):
        final = snap
    current_budget.reset(token)

    assert search_inputs[0] == ["What is RAG today?", "RAG definition overview 2026"], search_inputs
    assert len(final["facts"]) == 2, final["facts"]
    assert "# Final Answer" in final["final_report"]
