"""Wave-based parallel execution (Feature 03): dependency-ordered summarization
with prerequisite context passing."""

import asyncio

import app.graph.workflow as wf
from app.core.config import Settings


class _FakeSearch:
    def __init__(self, results):
        self._results = results
        self.settings = Settings(groq_api_key="k", _env_file=None)

    async def run_search(self, queries):
        out = []
        for q in queries or []:
            text = q[0] if isinstance(q, tuple) else q
            for r in self._results:
                out.append({**r, "sub_question": text})
        return out


async def test_wave_execution_passes_prerequisites(monkeypatch):
    """Wave-1 contracts must receive wave-0's findings as grounding."""
    import app.graph.workflow as wfmod

    calls = []

    async def fake_summarizer(llm, query, search_results, specialist_role="general", prior_findings=None):
        calls.append({
            "n_results": len(search_results),
            "prior": list(prior_findings) if prior_findings else [],
        })
        return [{
            "claim": f"fact from {len(calls)}",
            "source": search_results[0].get("url", "https://x.com/a"),
            "confidence": 0.8,
        }]

    monkeypatch.setattr(wfmod, "summarizer_agent", fake_summarizer)
    monkeypatch.setattr(wfmod, "dedupe_semantic_facts", lambda facts: facts)

    sub_questions = [
        {"question": "what is X", "axis": "definition", "wave": 0},
        {"question": "what is Y", "axis": "definition", "wave": 0},
        {"question": "compare X and Y", "axis": "comparison", "wave": 1, "depends_on": [1, 2]},
    ]
    search_results = [
        {"url": "https://x.com/a", "sub_question": "what is X", "snippet": "X stuff", "content": ""},
        {"url": "https://y.com/b", "sub_question": "what is Y", "snippet": "Y stuff", "content": ""},
        {"url": "https://xy.com/c", "sub_question": "compare X and Y", "snippet": "comparison", "content": ""},
    ]

    graph = wf.create_workflow(llm=None, search_client=None, entry_node=None)
    # Invoke the summarizer node directly via the compiled graph's node map.
    state = {
        "query": "compare X and Y",
        "sub_questions": sub_questions,
        "search_results": search_results,
        "facts": [],
    }
    node = graph.nodes["summarizer"]
    update = await node.ainvoke(state)
    facts = update["facts"]
    wave_report = update["wave_report"]

    # Two waves ran: wave 0 (2 contracts, no prerequisites) then wave 1 (1, with).
    assert [w["wave"] for w in wave_report] == [0, 1]
    assert wave_report[0]["contracts"] == 2
    assert wave_report[0]["with_prerequisites"] is False
    assert wave_report[1]["contracts"] == 1
    assert wave_report[1]["with_prerequisites"] is True

    # Wave-0 calls saw no prior findings; the wave-1 call saw 2.
    assert calls[0]["prior"] == []
    assert calls[1]["prior"] == []
    assert len(calls[2]["prior"]) == 2

    # All three contracts produced facts.
    assert len(facts) == 3


async def test_flat_plan_single_wave(monkeypatch):
    """No wave metadata (or all wave 0) → one wave, no prerequisite passing."""
    import app.graph.workflow as wfmod

    calls = []

    async def fake_summarizer(llm, query, search_results, specialist_role="general", prior_findings=None):
        calls.append({"prior": prior_findings})
        return [{"claim": "c", "source": "https://x.com/a", "confidence": 0.5}]

    monkeypatch.setattr(wfmod, "summarizer_agent", fake_summarizer)
    monkeypatch.setattr(wfmod, "dedupe_semantic_facts", lambda facts: facts)

    state = {
        "query": "q",
        "sub_questions": [
            {"question": "a", "axis": "definition"},
            {"question": "b", "axis": "evidence"},
        ],
        "search_results": [
            {"url": "https://x.com/a", "sub_question": "a", "snippet": "s", "content": ""},
            {"url": "https://x.com/b", "sub_question": "b", "snippet": "s", "content": ""},
        ],
        "facts": [],
    }
    graph = wf.create_workflow(llm=None, search_client=None, entry_node=None)
    node = graph.nodes["summarizer"]
    update = await node.ainvoke(state)
    assert len(update["wave_report"]) == 1
    assert update["wave_report"][0]["contracts"] == 2
    assert all(c["prior"] is None for c in calls)


def test_planner_update_exposes_wave_shape():
    """PlannerUpdate carries execution_waves so the UI can render the plan's
    dependency structure."""
    assert "execution_waves" in wf.PlannerUpdate.__annotations__
    assert "wave_report" in wf.SummarizerUpdate.__annotations__
    assert "execution_waves" in wf.ResearchState.__annotations__
