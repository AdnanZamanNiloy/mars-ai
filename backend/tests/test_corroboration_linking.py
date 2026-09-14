"""Deterministic multi-publisher corroboration linking (root-cause regression).

The primitives (`find_corroborating_sources` / `is_new_publisher` /
`grade_claim`) worked; the bug was that `search_node` merged freshly-returned
expansion results into state but never matched them back to the pending
needs_corroboration facts, so `corroborating_sources` never grew past one
publisher and live deep runs ended with `corroborated_ge2 = 0`. These tests
invoke the compiled graph's `search` node directly (the same version-robust
pattern as `test_wave_execution.py`) and lock the linking step in place.
"""
import app.graph.workflow as wf
from app.core.config import Settings
from app.core.evidence_grade import grade_claim

_CLAIM = "Global AI capital expenditure reached $200 billion in 2025"
_REUTERS = "https://reuters.com/tech/ai-capex"


def _settings(**kw):
    return Settings(groq_api_key="k", _env_file=None, **kw)


class _FakeSearch:
    """SearchClient-compatible stub returning a scripted result list."""

    def __init__(self, settings, results):
        self.settings = settings
        self._results = results
        self.calls = []

    async def run_search(self, queries):
        self.calls.append(list(queries))
        return list(self._results)


def _pending_state():
    """A deep-run state mid-expansion with one uncorroborated Reuters claim
    and its claim-specific corroboration query."""
    state = wf.build_initial_state("What is the current trend of AI?", 3, mode="deep")
    state.update({
        "facts": [{"claim": _CLAIM, "source": _REUTERS, "verified": True}],
        "corroboration_queries": [
            f"{_CLAIM} independent corroboration official data -site:reuters.com"
        ],
        "iteration": 1,  # expansion pass: corroboration queries are issued
        "sub_questions": [],
    })
    return state


async def _run_search(settings, results, state):
    graph = wf.create_workflow(
        llm=None, search_client=_FakeSearch(settings, results)
    )
    return await graph.nodes["search"].ainvoke(state)


async def test_search_node_links_new_publisher_to_pending_claim():
    """A pending needs_corroboration claim gains a second independent domain."""
    settings = _settings()
    oecd = "https://www.oecd.org/digital/ai-investment"
    results = [{
        "url": oecd,
        "snippet": f"{_CLAIM} according to OECD data.",
        "content": f"{_CLAIM} according to OECD data.",
        "sub_question": "",
    }]
    out = await _run_search(settings, results, _pending_state())

    facts = out.get("facts")
    assert facts, "linking must return the annotated facts"
    linked = [f for f in facts if _CLAIM in str(f.get("claim", ""))]
    assert linked, "pending claim must be preserved (never dropped)"
    assert linked[0].get("corroboration_count", 1) == 2
    srcs = linked[0].get("corroborating_sources") or []
    assert oecd in srcs, f"new publisher must be attached, got {srcs}"

    ev = grade_claim(linked[0])
    assert ev.corroboration_count >= 2
    assert ev.needs_corroboration is False


async def test_search_node_links_non_verbatim_new_publisher():
    """Regression: live expansion hits paraphrase the claim and scored ~0.30,
    so snippet-only matching never linked them. Title+snippet+excerpt with the
    numeric anchor path must link this realistic OECD page."""
    settings = _settings()
    oecd = "https://www.oecd.org/ai/investment-outlook"
    results = [{
        "url": oecd,
        "title": "OECD AI investment outlook",
        "snippet": (
            "Spending on AI infrastructure hit about $200 billion last year, "
            "per OECD figures."
        ),
        "content": "",
        "sub_question": "",
    }]
    out = await _run_search(settings, results, _pending_state())

    facts = out.get("facts")
    assert facts, "linking must return the annotated facts"
    linked = [f for f in facts if _CLAIM in str(f.get("claim", ""))]
    assert linked[0].get("corroboration_count", 1) == 2
    assert oecd in (linked[0].get("corroborating_sources") or [])

    ev = grade_claim(linked[0])
    assert ev.corroboration_count >= 2
    assert ev.needs_corroboration is False


async def test_search_node_same_publisher_does_not_raise_corroboration():
    """Another reuters.com URL is the SAME publisher: count stays 1."""
    settings = _settings()
    results = [{
        "url": "https://www.reuters.com/markets/ai-spending-2025",
        "snippet": f"{_CLAIM} according to Reuters.",
        "content": f"{_CLAIM} according to Reuters.",
        "sub_question": "",
    }]
    out = await _run_search(settings, results, _pending_state())

    # No new publisher -> no fact changed -> the update omits `facts`
    # entirely (shape preserved) and the run is not polluted with duplicates.
    assert "facts" not in out, "same-publisher hit must not annotate facts"

    # The claim itself is still uncorroborated.
    ev = grade_claim({"claim": _CLAIM, "source": _REUTERS, "verified": True})
    assert ev.corroboration_count == 1
    assert ev.needs_corroboration is True


async def test_mocked_pipeline_reaches_corroborated_ge2_without_recursion(monkeypatch):
    """End-to-end mocked graph: a corroboration-query result reaches the
    linkage, the final state carries a corroborated fact, and the run
    terminates without GraphRecursionError."""
    from app.graph.workflow import build_initial_state, create_workflow

    settings = _settings()
    claim = _CLAIM
    reuters = _REUTERS
    oecd = "https://www.oecd.org/digital/ai-investment"
    search_calls = []

    class FakeLLM:
        def __init__(self):
            self.settings = settings

        async def generate_json(self, *a, **k):
            return {}  # intent/stages handled by monkeypatched agents

    class StubSearch:
        def __init__(self):
            self.settings = settings

        async def run_search(self, queries):
            search_calls.append(list(queries))
            out = []
            for q in queries or []:
                text = q[0] if isinstance(q, (tuple, list)) else str(q)
                low = text.lower()
                # The corroboration procurement query excludes the current
                # publisher (reuters) and targets authoritative agencies.
                if "-site:reuters.com" in low or "oecd" in low:
                    out.append({
                        "url": oecd,
                        "snippet": f"{claim} according to OECD data.",
                        "content": f"{claim} according to OECD data.",
                        "sub_question": text,
                    })
                else:
                    out.append({
                        "url": reuters,
                        "snippet": f"{claim}.",
                        "content": f"{claim}.",
                        "sub_question": text,
                    })
            return out

    async def fake_planner(llm, query, critique_feedback="", today="", **kwargs):
        return [{
            "id": 1, "question": "What is the current trend of AI?",
            "axis": "evidence", "search_type": "news", "priority": 1,
            "depends_on": [], "coverage_goal": "", "domain": "general",
            "minimum_sources": 1, "stop_condition": "enough",
        }]

    async def fake_summarizer(llm, query, search_results=None,
                              specialist_role="general", prior_findings=None):
        # Always sourced to Reuters: the ONLY path to a second publisher is
        # search_node linking the OECD corroboration result to this claim.
        return [{"claim": claim, "source": reuters, "confidence": 0.9}]

    async def fake_critic(llm, query, facts=None, iteration=1,
                          max_iterations=3, contradictions=None, **kwargs):
        return {"is_sufficient": True, "reason": "ok",
                "improved_queries": [], "confidence": 0.9}

    async def fake_synthesizer(llm, query, facts, context=None):
        return "synthesized answer"

    monkeypatch.setattr(wf, "planner_agent", fake_planner)
    monkeypatch.setattr(wf, "summarizer_agent", fake_summarizer)
    monkeypatch.setattr(wf, "critic_agent", fake_critic)
    monkeypatch.setattr(wf, "synthesizer_agent", fake_synthesizer)
    monkeypatch.setattr(wf, "verify_facts", lambda facts, search_results: facts)

    state = build_initial_state("What is the current trend of AI?", 3, mode="standard")
    workflow = create_workflow(llm=FakeLLM(), search_client=StubSearch())

    final = None
    async for snapshot in workflow.astream(state, stream_mode="values"):
        final = {**(final or {}), **snapshot}

    assert final is not None
    facts = final.get("facts") or []
    corroborated = [
        f for f in facts
        if grade_claim(f).corroboration_count >= 2
        and not grade_claim(f).needs_corroboration
    ]
    assert corroborated, (
        "a corroboration-query result must lift a pending claim to two "
        f"independent domains; facts={facts}"
    )
