"""Cost Governor tests (Phase 2.2)."""
import app.graph.workflow as wf
from app.core.budget import BudgetTracker, current_budget
from app.core.config import Settings
from app.core.llm import LLMClient


def _tracker(**overrides) -> BudgetTracker:
    settings = Settings(groq_api_key="k", **overrides)
    return BudgetTracker(settings)


def test_real_tokens_preferred_over_estimate():
    t = _tracker()
    t.record_llm_call("groq:llama-3.1-8b-instant", input_tokens=1000, output_tokens=500)
    assert t.input_tokens == 1000
    assert t.output_tokens == 500
    # $0.05/1M in, $0.08/1M out → (1000*0.00005 + 500*0.00008)/1000
    assert abs(t.estimated_cost_usd - (1000 * 0.00005 + 500 * 0.00008) / 1000) < 1e-9


def test_estimate_fallback_when_no_usage_field():
    t = _tracker()
    t.record_llm_call("groq:x", request_chars=400, response_chars=800)
    assert t.input_tokens == 100
    assert t.output_tokens == 200


def test_over_budget_flag_and_note():
    t = _tracker(research_max_cost_usd=0.01)
    assert not t.over_budget
    t.record_llm_call("groq:x", input_tokens=100_000, output_tokens=100_000)
    assert t.over_budget
    note = t.limitation_note()
    assert "budget" in note.lower()
    assert t.snapshot()["over_budget"] is True


def test_hf_pricing_used_for_non_groq_models():
    t = _tracker()
    t.record_llm_call("hf:Qwen/Qwen2.5-7B-Instruct", input_tokens=10_000, output_tokens=0)
    assert abs(t.estimated_cost_usd - (10_000 * 0.0002 / 1000)) < 1e-9


async def test_budget_cutoff_stops_loop_and_notes_limitation(monkeypatch):
    """2.2 DoD: a tiny budget stops the pipeline early, still producing a report."""
    settings = Settings(groq_api_key="k", research_max_cost_usd=0.001, _env_file=None)
    tracker = BudgetTracker(settings)
    token = current_budget.set(tracker)

    state = wf.build_initial_state("what is RAG", 5)
    state["budget_tracker"] = tracker

    big_usage = dict(input_tokens=50_000, output_tokens=50_000)

    async def fake_planner(llm, query, critique_feedback="", today=""):
        return [{"id": 1, "question": "q one", "axis": "definition", "search_type": "encyclopedia",
                 "priority": 1, "depends_on": [], "coverage_goal": "", "domain": "general",
                 "minimum_sources": 2, "stop_condition": "enough"}]

    async def fake_summarizer(llm, query, search_results=None, specialist_role="general"):
        # One summarizer call costs ~$0.0065 — 6x the $0.001 budget.
        tracker.record_llm_call("groq:llama-3.1-8b-instant", **big_usage)
        return [{"claim": "RAG is retrieval augmented generation is here", "source": "https://arxiv.org/a", "confidence": 0.9}]

    critic_calls = {"n": 0}

    async def fake_critic(llm, query, facts=None, iteration=1, max_iterations=3, contradictions=None):
        critic_calls["n"] += 1
        return {"is_sufficient": False, "reason": "not done", "improved_queries": ["more"], "confidence": 0.4}

    async def fake_synthesizer(llm, query, facts):
        return "synthesized answer"

    monkeypatch.setattr(wf, "planner_agent", fake_planner)
    monkeypatch.setattr(wf, "summarizer_agent", fake_summarizer)
    monkeypatch.setattr(wf, "critic_agent", fake_critic)
    monkeypatch.setattr(wf, "synthesizer_agent", fake_synthesizer)

    class StubSearch:
        SEARCH_RESULTS = [
            {"url": "https://arxiv.org/a", "sub_question": "q one", "content": "content for the sub-question", "snippet": "snip"},
        ]

        async def run_search(self, questions):
            return self.SEARCH_RESULTS

    llm = LLMClient(settings)
    workflow = wf.create_workflow(llm, StubSearch())

    final = None
    async for snap in workflow.astream(state, stream_mode="values"):
        final = snap

    current_budget.reset(token)
    assert tracker.over_budget
    # Budget cutoff must route to synthesizer before the iteration ceiling.
    assert critic_calls["n"] == 1, critic_calls
    assert "# Final Answer" in final["final_report"]
    assert "budget" in final["final_report"].lower()
