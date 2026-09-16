"""Performance/cost optimizations (perf task): deterministic regression tests.

The measured defect these lock down: `search_node` executed an IDENTICAL search
query on consecutive expansion passes. The depth controller already built an
executed-query memory (`coverage_searched` -> `_searched_queries`) but nothing
ever WROTE it, so primary-source / corroboration follow-ups that regenerate the
same text every pass were re-issued at full retrieval cost for zero new
evidence. `search_node` now enforces exact normalized-text dedup and records
what actually ran.

Invariants these tests protect:
* an identical query (including within one batch) is issued at most once;
* a DIFFERENT query is never dropped — no research path or source is lost;
* the executed-query memory is recorded on state AND visible to the depth
  controller's existing `_searched_queries` reader;
* the full mocked pipeline keeps the same verified-fact footprint.
"""
from app.core.config import Settings
from app.core.depth_controller import _searched_queries


def _settings(**kw):
    return Settings(groq_api_key="k", database_url="/tmp/mars-perf.db", _env_file=None, **kw)


class _CapturingSearch:
    def __init__(self):
        self.calls = []

    async def run_search(self, sub_questions):
        for q in sub_questions or []:
            text = q[0] if isinstance(q, tuple) else str(q)
            self.calls.append(text)
        return [
            {
                "url": f"https://newpublisher{i}.org/{i}",
                "sub_question": (q[0] if isinstance(q, tuple) else str(q)),
                "content": "",
                "snippet": "",
            }
            for i, q in enumerate(sub_questions or [])
        ]


def _search_node(settings):
    import app.graph.workflow as wfmod

    search = _CapturingSearch()
    graph = wfmod.create_workflow(llm=None, search_client=search)
    return graph.nodes["search"], search


# ---------------------------------------------------------------------------
# 1. identical query is never re-issued
# ---------------------------------------------------------------------------

async def test_identical_corroboration_query_not_reissued():
    """A query already executed this run must not run again."""
    settings = _settings()
    node, search = _search_node(settings)
    prior = "what evidence supports rag reducing hallucinations site:arxiv.org OR site:doi.org"
    state = {
        "query": "broad query",
        "iteration": 1,
        "sub_questions": [],
        "search_results": [{"url": "https://seed.com/a", "sub_question": "seed"}],
        "corroboration_queries": [prior],
        # The query already ran on an earlier pass — recorded in the memory.
        "executed_queries": [prior],
        "coverage_searched": [prior],
        "expansion_passes": 0,
    }
    update = await node.ainvoke(state)
    assert search.calls == [], "an identical query must never be re-issued"
    assert update["search_results"]  # evidence still flows to the summarizer


async def test_identical_query_in_same_batch_is_issued_once():
    """Two sources in one pass naming the same query text -> one execution."""
    settings = _settings()
    node, search = _search_node(settings)
    dup = "what is retrieval augmented generation site:britannica.com"
    state = {
        "query": "broad query",
        "iteration": 1,
        "sub_questions": [
            {"question": dup, "search_type": "encyclopedia"},
        ],
        "search_results": [],
        "corroboration_queries": [dup],
        "expansion_passes": 0,
    }
    update = await node.ainvoke(state)
    assert search.calls.count(dup) <= 1, "within-batch duplicates must collapse"
    assert update["expansion_passes"] == 1


async def test_distinct_queries_are_never_dropped():
    """The safety invariant: only exact repeats are removed, never new work."""
    settings = _settings()
    node, search = _search_node(settings)
    first = "what is retrieval augmented generation site:britannica.com"
    second = "what evidence supports rag reducing hallucinations site:arxiv.org"
    state = {
        "query": "broad query",
        "iteration": 1,
        "sub_questions": [{"question": second, "search_type": "academic"}],
        "search_results": [{"url": "https://seed.com/a", "sub_question": "seed"}],
        "corroboration_queries": [first],
        "executed_queries": [first],  # only `first` already ran
        "coverage_searched": [first],
        "expansion_passes": 0,
    }
    update = await node.ainvoke(state)
    assert second in search.calls, "a novel query must still be executed"
    assert first not in search.calls, "the already-run query must be skipped"
    assert second in update["executed_queries"]


async def test_case_and_whitespace_only_repeats_collapse():
    """Normalization: case/whitespace variants are the same query."""
    settings = _settings()
    node, search = _search_node(settings)
    state = {
        "query": "broad query",
        "iteration": 1,
        "sub_questions": [],
        "search_results": [{"url": "https://seed.com/a", "sub_question": "seed"}],
        "corroboration_queries": ["RAG   Adoption   STATISTICS"],
        "executed_queries": ["rag adoption statistics"],
        "expansion_passes": 0,
    }
    update = await node.ainvoke(state)
    assert search.calls == []


# ---------------------------------------------------------------------------
# 2. the executed-query memory is recorded and consumable downstream
# ---------------------------------------------------------------------------

async def test_executed_queries_recorded_and_visible_to_depth_controller():
    settings = _settings()
    node, search = _search_node(settings)
    q = "novel primary source query -site:example.com"
    state = {
        "query": "broad query",
        "iteration": 1,
        "sub_questions": [],
        "search_results": [{"url": "https://seed.com/a", "sub_question": "seed"}],
        "corroboration_queries": [q],
        "expansion_passes": 0,
    }
    update = await node.ainvoke(state)
    assert q in update["executed_queries"]
    assert q in update["coverage_searched"]
    # The depth controller's existing reader must now see it (it was read-only
    # dead code before this change).
    merged_state = {**state, **update}
    assert _searched_queries(merged_state), "executed queries feed the stopping memory"


# ---------------------------------------------------------------------------
# 3. quality-preserving invariant on the mocked pipeline
# ---------------------------------------------------------------------------

async def test_mock_pipeline_dedup_preserves_fact_footprint():
    """Dropping only exact repeats must not change extracted/verified facts."""
    import tempfile

    from app.core import llm_cache as _lc
    from app.core.usage import start_run_usage, clear_run_usage
    from app.graph.workflow import build_initial_state, create_workflow
    from bench.mock_pipeline import FakeLLM, FakeSearch

    tmp = tempfile.mkdtemp()
    settings = Settings(
        groq_api_key="bench-key", database_url=f"{tmp}/bench.db", _env_file=None
    )
    _lc._force_disabled = True
    llm = FakeLLM(settings, critic_pass_on_iteration=2)
    search = FakeSearch(settings)
    workflow = create_workflow(llm, search_client=search)
    state = build_initial_state(
        "What is retrieval augmented generation, how widely is it adopted, "
        "what is the evidence for its effectiveness, and what are the criticisms?",
        max_iterations=3,
        mode="standard",
    )
    usage = start_run_usage("perf-test", settings, mode="standard")
    final = dict(state)
    try:
        async for snapshot in workflow.astream(state, stream_mode="values"):
            final = {**final, **{k: v for k, v in snapshot.items() if v}}
    finally:
        clear_run_usage()

    calls = [c for c in search.calls]
    assert len(calls) == len(set(calls)), "no duplicate search query may be issued"
    facts = final.get("facts") or []
    assert len(facts) == 4, "verified fact footprint must be preserved"
    assert sum(1 for f in facts if f.get("verified")) == 4
    assert final.get("executed_queries"), "executed-query memory must persist"

