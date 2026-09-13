"""Fix A — independent corroboration PROCUREMENT.

The live deep run measured corroboration perfectly and then never went
looking for it: 0 claims reached two independent registrable domains while
121 needed corroboration. These tests pin the procurement path:

* a needs_corroboration claim yields a query that targets a DIFFERENT
  publisher (contains a `-site:` exclusion or an explicit new-publisher ask);
* `independent_corroboration` can never count the same registrable domain
  twice, however many URLs that domain contributes;
* `is_new_publisher` agrees with that independence rule;
* corroboration queries are actually EXECUTED by search_node, not merely
  stored on the critique.
"""
from app.core.evidence_grade import (
    distinct_publisher_count,
    independent_corroboration,
    is_new_publisher,
    registrable_domain,
)
from app.core.config import Settings
from app.graph.workflow import _corroboration_queries


def _settings(**kw):
    return Settings(groq_api_key="k", _env_file=None, **kw)


def _state(**overrides):
    base = {
        "query": "current trend of AI",
        "facts": [],
        "contradictions": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. domain independence is a hard invariant
# ---------------------------------------------------------------------------

def test_independent_corroboration_never_counts_same_domain_twice():
    domains, count = independent_corroboration(
        [
            "https://blog.example.com/deep/article",
            "https://www.example.com/",
            "https://example.com/other",
        ],
        primary_source="https://example.com/primary",
    )
    assert domains == ["example.com"]
    assert count == 1


def test_independent_corroboration_counts_distinct_publishers():
    domains, count = independent_corroboration(
        ["https://b.org/x", "https://c.net/y"],
        primary_source="https://a.com/p",
    )
    assert count == 3
    assert registrable_domain("https://a.com/p") in domains
    assert registrable_domain("https://b.org/x") in domains


def test_is_new_publisher_behaves():
    existing = ["https://blog.example.com/a", "https://other.org/b"]
    assert is_new_publisher("https://www.example.com/deep", existing) is False
    assert is_new_publisher("https://third.net/c", existing) is True
    # Unparseable URLs are never a new publisher.
    assert is_new_publisher("", existing) is False


def test_distinct_publisher_count_is_per_domain():
    assert distinct_publisher_count(
        ["https://a.com/1", "https://www.a.com/2", "https://b.org/3"]
    ) == 2


def test_dedupe_corroboration_count_is_per_publisher():
    """A second page on the same domain must not raise corroboration_count."""
    from app.agents.evidence_utils import dedupe_semantic_facts

    claim = "Retrieval augmented generation reduces hallucination rates by 30%"
    facts = [
        {"claim": claim, "source": "https://example.com/a", "confidence": 0.8},
        {"claim": claim, "source": "https://www.example.com/b", "confidence": 0.7},
    ]
    merged = dedupe_semantic_facts(facts)
    assert len(merged) == 1
    assert merged[0]["corroboration_count"] == 1


# ---------------------------------------------------------------------------
# 2. corroboration query generation targets a different publisher
# ---------------------------------------------------------------------------

def test_needs_corroboration_claim_generates_different_domain_query():
    state = _state(facts=[
        {
            "claim": "Enterprise adoption reached 62% of companies in 2024",
            "source": "https://blog.example.com/report",
            "verified": True,
        },
    ])
    queries = _corroboration_queries(state)
    assert queries, "a single-publisher quantitative claim must procure corroboration"
    joined = " ".join(queries).lower()
    assert "-site:example.com" in joined
    # Quantitative claims lean on primary/official vocabulary.
    assert any(term in joined for term in ("official report", "government data", "dataset", "peer-reviewed"))


def test_definitional_uncorroborated_claim_targets_new_publisher():
    state = _state(facts=[
        {
            "claim": "Retrieval augmented generation is a technique that grounds outputs",
            "source": "https://secondary.com/what-is-rag",
            "verified": True,
        },
    ])
    queries = _corroboration_queries(state)
    assert queries
    joined = " ".join(queries).lower()
    assert "-site:secondary.com" in joined
    assert "independent source" in joined or "independent publisher" in joined


def test_corroborated_claim_yields_no_procurement_query():
    state = _state(facts=[
        {
            "claim": "Enterprise adoption reached 62% of companies in 2024",
            "source": "https://a.com/report",
            "verified": True,
            "corroborating_sources": ["https://b.org/x", "https://c.net/y"],
        },
    ])
    assert _corroboration_queries(state) == []


# ---------------------------------------------------------------------------
# 3. procurement queries are actually EXECUTED by search_node
# ---------------------------------------------------------------------------

class _CapturingSearch:
    def __init__(self, settings):
        self.settings = settings
        self.calls = []

    async def run_search(self, sub_questions):
        for q in sub_questions or []:
            text = q[0] if isinstance(q, tuple) else str(q)
            self.calls.append(text)
        return [
            {
                "url": f"https://newpublisher{ i }.org/{ i }",
                "sub_question": (q[0] if isinstance(q, tuple) else str(q)),
                "content": "",
                "snippet": "",
            }
            for i, q in enumerate(sub_questions or [])
        ]


async def test_search_node_executes_corroboration_queries(monkeypatch):
    import app.graph.workflow as wfmod

    settings = _settings()
    search = _CapturingSearch(settings)
    graph = wfmod.create_workflow(llm=None, search_client=search)
    node = graph.nodes["search"]
    state = {
        "query": "current trend of AI",
        "iteration": 1,  # expansion pass
        "sub_questions": [],
        "search_results": [{"url": "https://seed.com/a", "sub_question": "seed"}],
        "corroboration_queries": [
            "enterprise adoption official report dataset -site:example.com",
        ],
        "expansion_passes": 0,
    }
    update = await node.ainvoke(state)
    assert any("-site:example.com" in c for c in search.calls)
    assert update.get("counter_evidence_attempted") is True
    assert update.get("expansion_passes") == 1


async def test_search_node_stops_expansion_at_hard_wall():
    """Fix B.3: a per-run cap on expansion passes bounds the loop."""
    import app.graph.workflow as wfmod

    settings = _settings(max_expansion_passes=2)
    search = _CapturingSearch(settings)
    graph = wfmod.create_workflow(llm=None, search_client=search)
    node = graph.nodes["search"]
    state = {
        "query": "broad query",
        "iteration": 5,
        "sub_questions": [],
        "search_results": [{"url": "https://seed.com/a", "sub_question": "seed"}],
        "corroboration_queries": ["another query -site:example.com"],
        "expansion_passes": 3,  # already at the wall
    }
    update = await node.ainvoke(state)
    assert search.calls == [], "the expansion wall must stop new searches"
    assert update["expansion_passes"] == 3
