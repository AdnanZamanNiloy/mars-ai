"""Latency remediation: independent I/O must actually run concurrently.

Evidence (baseline run 0bfd5f9e, 236s total): the search node alone took
111s — with Tavily's circuit open every contract query paid a serial
DDG text→news rescue, and primary-fallback substitution queries ran one
awaited round at a time. Section-wise synthesis wrote each section with a
serial awaited LLM call, and the router ran AFTER the intent gather instead
of overlapping the grounding search.

Each test pins concurrency with an in-flight peak counter or completion
ordering — never bare wall-clock thresholds (those flake under load).
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime

import httpx
import pytest
from fastapi import FastAPI

from app.agents.intent import heuristic_intent
from app.agents.outline import build_outline
from app.agents.search import SearchClient, SearchResult
from app.agents.synthesizer import synthesize
from app.api.routes import limiter, router as api_router
from app.core.config import Settings


def _settings(**over) -> Settings:
    base = {"groq_api_key": "k", "_env_file": None}
    base.update(over)
    return Settings(**base)


def _result(url: str, *, sub_question: str = "what is a transformer",
            search_type: str = "general") -> SearchResult:
    return SearchResult(
        title="Result", url=url, snippet="snippet text",
        sub_question=sub_question, search_type=search_type, provider="ddg_text",
    )


def _sub_questions():
    return [
        {"question": "AI definition", "axis": "definition"},
        {"question": "AI market data", "axis": "evidence"},
        {"question": "AI risks", "axis": "criticism"},
        {"question": "AI outlook", "axis": "outlook"},
    ]


def _facts():
    return [
        {"claim": "Artificial intelligence simulates human intelligence in machines.",
         "axis": "definition", "source": "https://a.example/x", "confidence": 0.8,
         "verified": True, "sub_question": "AI definition"},
        {"claim": "Global AI spending reached 200 billion dollars in 2025.",
         "axis": "evidence", "source": "https://b.example/y", "confidence": 0.7,
         "verified": True, "sub_question": "AI market data"},
        {"claim": "AI systems can encode societal bias at scale.",
         "axis": "criticism", "source": "https://c.example/z", "confidence": 0.6,
         "verified": True, "sub_question": "AI risks"},
        {"claim": "Adoption is expected to keep rising through 2026.",
         "axis": "outlook", "source": "https://d.example/w", "confidence": 0.5,
         "verified": True, "sub_question": "AI outlook"},
    ]


# ---------------------------------------------------------------------------
# Section-wise synthesis: writes overlap (peak >= 2), order preserved
# ---------------------------------------------------------------------------

class _PeakSectionLLM:
    def __init__(self):
        self.in_flight = 0
        self.peak = 0
        self.calls = 0

    async def generate_json(self, system_prompt, user_prompt, **kwargs):
        self.calls += 1
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.05)
            return {"answer": "Section body with a cited claim [1]."}
        finally:
            self.in_flight -= 1


def test_section_wise_writes_run_concurrently():
    """Sections are independent I/O and must be gathered, not awaited one
    by one: with a serial loop the peak in-flight count stays at 1."""
    outline = build_outline("What is the current trend of AI?", _facts(), _sub_questions())
    llm = _PeakSectionLLM()
    result = asyncio.run(
        synthesize(llm, "What is the current trend of AI?", _facts(),
                   {"intent": {}, "sub_questions": _sub_questions()},
                   outline=outline, section_wise=True, compress_context=False)
    )
    assert result is not None
    assert llm.calls >= 3  # exec summary + at least two section writes
    assert llm.peak >= 2, "section writes did not overlap"
    # Submission order = outline order regardless of completion order.
    for section in outline.sections:
        assert f"## {section.title}" in result.answer


# ---------------------------------------------------------------------------
# Tavily -> DDG rescue: text and news overlap
# ---------------------------------------------------------------------------

async def test_ddg_rescue_runs_text_and_news_concurrently():
    client = SearchClient(_settings())
    state = {"in_flight": 0, "peak": 0}

    async def _slow_text(query):
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        try:
            await asyncio.sleep(0.05)
            return [_result("https://text.example/a")]
        finally:
            state["in_flight"] -= 1

    async def _slow_news(query):
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        try:
            await asyncio.sleep(0.05)
            return [_result("https://news.example/b", search_type="news")]
        finally:
            state["in_flight"] -= 1

    client._ddg_text = _slow_text
    client._ddg_news = _slow_news
    results = await client._ddg_rescue("some query")
    assert state["peak"] >= 2, "ddg text and news ran serially"
    # Concatenation order stays text-then-news (stable ranking input).
    assert [r.url for r in results] == ["https://text.example/a", "https://news.example/b"]


# ---------------------------------------------------------------------------
# Primary fallback: substitution queries issue concurrently
# ---------------------------------------------------------------------------

async def test_primary_fallback_queries_run_concurrently(monkeypatch):
    settings = _settings(
        search_primary_fallback_enabled=True,
        search_primary_fallback_max=2,
        search_fetch_top_n=4,
    )
    client = SearchClient(settings)
    state = {"in_flight": 0, "peak": 0}

    async def slow_providers(self, query, search_type):
        assert "site:" in query
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        try:
            await asyncio.sleep(0.05)
            return []
        finally:
            state["in_flight"] -= 1

    monkeypatch.setattr(SearchClient, "_providers_for", slow_providers)
    ranked = [
        _result("https://blocked-a.com/x"),
        _result("https://blocked-b.com/y"),
    ]
    await client._primary_fallback(ranked, ["blocked-a.com", "blocked-b.com"])
    assert state["peak"] >= 2, "primary-fallback queries ran serially"
    assert client.health.snapshot()["fallback_queries_issued"] == 2


# ---------------------------------------------------------------------------
# Intent node: router overlaps the grounding search
# ---------------------------------------------------------------------------

async def test_route_overlaps_grounding_search(monkeypatch):
    """route_query must finish BEFORE the grounding search does — the old
    shape awaited it after the search/classify gather, adding one serial
    LLM round-trip to every run. Completion ordering proves the overlap."""
    import app.graph.workflow as wf
    from app.core.llm import LLMClient

    stamps: dict = {}
    settings = Settings(groq_api_key="k", _env_file=None)
    llm = LLMClient(settings)

    class _SlowSearch:
        async def run_search(self, sub_questions):
            await asyncio.sleep(0.35)
            if "search" not in stamps:
                stamps["search"] = time.monotonic()
            return []

    # Class bodies resolve via LOAD_NAME (global/builtins only) — attach
    # the settings after the block instead of `settings = settings` inside.
    _SlowSearch.settings = settings

    async def fake_classify(llm_arg, query, context_snippets=None):
        await asyncio.sleep(0.10)
        stamps["classify"] = time.monotonic()
        return heuristic_intent(query)

    class _RouteStub:
        def to_dict(self):
            return {"path": "research", "reason": "test", "confidence": 0.9,
                    "origin": "llm", "signals": {}}

    async def fake_route(llm_arg, query, intent=None):
        await asyncio.sleep(0.10)
        stamps["route"] = time.monotonic()
        return _RouteStub()

    async def fake_planner(**kwargs):
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
        return []

    async def fake_critic(**kwargs):
        return {"is_sufficient": True, "reason": "enough", "improved_queries": [],
                "confidence": 0.8}

    async def fake_synthesizer(llm=None, query=None, facts=None, context=None):
        return "## Executive Summary\n\nAnswer."

    monkeypatch.setattr(wf, "classify_intent", fake_classify)
    monkeypatch.setattr(wf, "route_query", fake_route)
    monkeypatch.setattr(wf, "planner_agent", fake_planner)
    monkeypatch.setattr(wf, "summarizer_agent", fake_summarizer)
    monkeypatch.setattr(wf, "critic_agent", fake_critic)
    monkeypatch.setattr(wf, "synthesizer_agent", fake_synthesizer)

    graph = wf.create_workflow(llm, _SlowSearch())
    state = wf.build_initial_state("What is transformer?", 3, mode="quick")
    async for _snap in graph.astream(state, stream_mode="values"):
        pass

    assert {"classify", "route", "search"} <= set(stamps)
    # classify+route together (0.20s) finish before the search (0.35s):
    # overlap holds even with generous scheduler jitter. The old serial
    # shape produced route >= search, failing this assertion.
    assert stamps["route"] < stamps["search"], (
        "route_query did not overlap the grounding search"
    )


# ---------------------------------------------------------------------------
# Trace instrumentation: started_at is a real superstep boundary
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_limiter():
    limiter.reset()
    yield
    limiter.reset()


class StubTimedWorkflow:
    """Initial snapshot + one snapshot per superstep, like LangGraph
    stream_mode="values", with gaps so boundary timestamps are distinct."""

    async def astream(self, state, stream_mode=None, config=None):
        yield {}
        await asyncio.sleep(0.02)
        yield {"sub_questions": [{"question": "what is RAG?"}]}
        await asyncio.sleep(0.02)
        yield {"search_results": [{"url": "https://a.com", "title": "t", "snippet": "s"}]}
        await asyncio.sleep(0.02)
        yield {"facts": [{"claim": "c", "source": "https://a.com", "confidence": 0.9}],
               "iteration": 0}
        await asyncio.sleep(0.02)
        yield {"final_report": "# ok", "confidence": 0.5}


async def test_agent_events_carry_real_started_at(tmp_path):
    from app.db.sqlite import init_db
    import aiosqlite

    db_path = str(tmp_path / "timing.db")
    await init_db(db_path)
    settings = Settings(groq_api_key="test-key", database_url=db_path, _env_file=None)
    app = FastAPI()
    app.state.workflow = StubTimedWorkflow()
    app.state.settings = settings
    app.state.limiter = limiter
    app.include_router(api_router, prefix="/api")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/research/stream", json={"query": "valid research query here"}
        )
    assert response.status_code == 200

    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "SELECT node, started_at, ended_at FROM agent_events "
            "WHERE node IN ('planner', 'search', 'summarizer') ORDER BY id"
        )
        rows = await cur.fetchall()
    by_node = {node: (started, ended) for node, started, ended in rows}
    assert {"planner", "search", "summarizer"} <= set(by_node)
    for node, (started, ended) in by_node.items():
        assert started and ended, f"{node} missing timestamps"
        # Previously started_at was stubbed equal to ended_at; the boundary
        # must make intra-run stage duration measurable.
        assert datetime.fromisoformat(started) < datetime.fromisoformat(ended), (
            f"{node} started_at still equals ended_at (unmeasurable duration)"
        )
