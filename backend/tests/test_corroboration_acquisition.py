"""Corroboration ACQUISITION (not just measurement).

The confirmed live failure: 143 facts / 15 contracts, corroborated_ge2 = 0 —
not one claim reached two independent registrable domains, while 112 needed
corroboration. Measurement (`independent_corroboration`, `is_new_publisher`)
was already correct; the gap was that corroboration searches ran and their
results were never matched back to the claims that needed a second publisher.

These tests pin the acquisition path:

* a needs_corroboration claim produces a query that TARGETS authoritative
  publishers (site:gov/edu/int/official/dataset) AND excludes its domain;
* the same claim's corroboration query is not re-issued beyond a bounded limit;
* a search result from a NEW registrable domain whose text supports the claim
  raises corroboration to 2 (needs_corroboration clears); a SAME-domain result
  does not;
* an exhausted claim stops being re-queried (bounded);
* a full mocked deep run terminates and leaves the tree state consistent.
"""
from app.agents.sources import (
    authoritative_site_terms,
    build_corroboration_query,
)
from app.core.config import Settings
from app.core.evidence_grade import (
    apply_corroboration,
    find_corroborating_sources,
    grade_claim,
    registrable_domain,
)
from app.graph.workflow import (
    _acquire_corroboration,
    _corroboration_queries,
    build_initial_state,
    create_workflow,
)


def _settings(**kw):
    return Settings(groq_api_key="k", _env_file=None, **kw)


def _state(**overrides):
    base = {"query": "current trend of AI", "facts": [], "contradictions": []}
    base.update(overrides)
    return base


QUANT_CLAIM = (
    "Enterprise adoption of retrieval augmented generation reached 62% of "
    "companies in 2024"
)
QUANT_SUPPORT = (
    "Enterprise adoption of retrieval augmented generation reached 62% of "
    "surveyed companies in 2024 according to official statistics."
)
DEF_CLAIM = (
    "Retrieval augmented generation is a technique that grounds language model "
    "outputs in retrieved evidence"
)
DEF_SUPPORT = (
    "Retrieval augmented generation is a technique that grounds language model "
    "outputs in retrieved evidence rather than relying on parametric memory."
)
UNRELATED = "Solar capacity in Bangladesh grew 40% last year according to the grid operator."


# ---------------------------------------------------------------------------
# 1. queries target authoritative publishers and exclude the current domain
# ---------------------------------------------------------------------------

def test_authoritative_registry_reuses_tiering_sources():
    terms = authoritative_site_terms(max_sites=40)
    assert terms, "the authoritative registry must not be empty"
    # Reuses the sources.py registries rather than a parallel invented list.
    assert any(t in {"worldbank.org", "who.int", "oecd.org", "nist.gov", "ipcc.ch"} for t in terms)
    assert any(t == "gov" or t.endswith(".gov") for t in terms)
    assert any(t == "edu" or t.endswith(".edu") or t in {"arxiv.org", "nature.com"} for t in terms)
    # Priority order is stable; rotation changes the starting host.
    assert authoritative_site_terms(max_sites=4)[0] == "worldbank.org"
    assert authoritative_site_terms(max_sites=4, offset=1)[0] != "worldbank.org"


def test_corroboration_query_targets_authoritative_and_excludes_domain():
    query = build_corroboration_query("adoption of retrieval augmented generation", "example.com")
    lowered = query.lower()
    assert "site:" in lowered
    assert any(term in lowered for term in ("gov", "edu", "int"))
    assert "-site:example.com" in lowered
    assert query == query.strip()


def test_needs_corroboration_claim_produces_authoritative_targeting_query():
    state = _state(facts=[
        {"claim": QUANT_CLAIM, "source": "https://blog.example.com/report", "verified": True},
    ])
    queries, registry = _corroboration_queries(state)
    assert queries, "a single-publisher quantitative claim must procure corroboration"
    joined = " ".join(queries).lower()
    assert "-site:example.com" in joined
    assert "site:" in joined
    assert any(term in joined for term in ("official report", "government data", "dataset", "peer-reviewed"))
    # Per-claim attempt state is threaded (keyed by normalized claim).
    assert registry
    entry = next(iter(registry.values()))
    assert entry["attempts"] == 1
    assert "example.com" in entry["domains_queried"]


# ---------------------------------------------------------------------------
# 2. bounded identical re-issue
# ---------------------------------------------------------------------------

def test_same_claim_query_not_reissued_beyond_bounded_limit():
    settings = _settings(max_corroboration_attempts=2)
    facts = [
        {"claim": QUANT_CLAIM, "source": "https://blog.example.com/report", "verified": True},
    ]
    state = _state(facts=facts)
    registry = {}
    seen_queries = []
    for _ in range(5):
        state["corroboration_registry"] = registry
        queries, registry = _corroboration_queries(state, settings=settings)
        seen_queries.extend(queries)
    # Two attempt budgets, then the claim is exhausted; identical query text is
    # never re-issued (attempt 2 rotates to different authoritative hosts).
    assert len(seen_queries) == 2
    assert len(set(seen_queries)) == 2
    entry = next(iter(registry.values()))
    assert entry["attempts"] == 2


def test_exhausted_claim_stops_being_queried():
    settings = _settings(max_corroboration_attempts=1)
    from app.agents.planner import normalize_text

    key = normalize_text(QUANT_CLAIM)
    state = _state(
        facts=[{"claim": QUANT_CLAIM, "source": "https://blog.example.com/report", "verified": True}],
        corroboration_registry={
            key: {"claim": QUANT_CLAIM, "domains_queried": ["example.com"],
                  "query_keys": [], "attempts": 1}
        },
    )
    queries, _ = _corroboration_queries(state, settings=settings)
    assert queries == [], "an exhausted claim must not be re-queried"


# ---------------------------------------------------------------------------
# 3. matching: new domain supports the claim -> count rises to 2
# ---------------------------------------------------------------------------

def test_new_domain_supporting_text_raises_corroboration_to_two():
    fact = {"claim": QUANT_CLAIM, "source": "https://blog.example.com/report", "verified": True}
    assert grade_claim(fact).corroboration_count == 1
    assert grade_claim(fact).needs_corroboration is True

    results = [
        {"url": "https://www.oecd.org/data/report", "snippet": QUANT_SUPPORT, "content": ""},
    ]
    matches = find_corroborating_sources(
        QUANT_CLAIM, results, [fact["source"]]
    )
    assert matches == ["https://www.oecd.org/data/report"]
    assert apply_corroboration(fact, matches[0]) is True

    record = grade_claim(fact)
    assert record.corroboration_count == 2
    assert record.needs_corroboration is False


def test_same_domain_result_does_not_raise_corroboration():
    fact = {"claim": QUANT_CLAIM, "source": "https://blog.example.com/report", "verified": True}
    results = [
        # Same registrable domain as the primary source.
        {"url": "https://news.example.com/other", "snippet": QUANT_SUPPORT, "content": ""},
    ]
    assert find_corroborating_sources(QUANT_CLAIM, results, [fact["source"]]) == []
    assert apply_corroboration(fact, "https://news.example.com/other") is False
    assert grade_claim(fact).corroboration_count == 1
    assert grade_claim(fact).needs_corroboration is True


def test_unrelated_new_domain_text_is_not_corroboration():
    fact = {"claim": QUANT_CLAIM, "source": "https://blog.example.com/report", "verified": True}
    results = [{"url": "https://random.org/page", "snippet": UNRELATED, "content": ""}]
    assert find_corroborating_sources(QUANT_CLAIM, results, [fact["source"]]) == []
    assert grade_claim(fact).corroboration_count == 1


def test_acquire_corroboration_attaches_new_domain_only():
    facts = [{"claim": QUANT_CLAIM, "source": "https://blog.example.com/report", "verified": True}]
    results = [
        {"url": "https://news.example.com/dup", "snippet": QUANT_SUPPORT, "content": ""},
        {"url": "https://www.worldbank.org/data", "snippet": QUANT_SUPPORT, "content": ""},
    ]
    out = _acquire_corroboration(facts, results)
    assert out[0]["corroboration_count"] == 2
    domains = {registrable_domain(u) for u in out[0]["corroborating_sources"]}
    assert domains == {"example.com", "worldbank.org"}


def test_definitional_claim_corroborated_by_new_domain():
    fact = {"claim": DEF_CLAIM, "source": "https://secondary.com/what-is-rag", "verified": True}
    results = [{"url": "https://arxiv.org/abs/2005.11401", "snippet": DEF_SUPPORT, "content": ""}]
    out = _acquire_corroboration([fact], results)
    assert out[0]["corroboration_count"] == 2
    assert grade_claim(out[0]).needs_corroboration is False


# ---------------------------------------------------------------------------
# 4. full mocked deep run terminates and stays consistent
# ---------------------------------------------------------------------------

class _FakeSearch:
    """Returns a NEW-domain authoritative hit for corroboration queries and
    distinct-source hits otherwise, like the real providers should."""

    def __init__(self, settings):
        self.settings = settings
        self.calls = []

    async def run_search(self, sub_questions):
        out = []
        for q in sub_questions or []:
            text = q[0] if isinstance(q, tuple) else str(q)
            self.calls.append(text)
            lowered = text.lower()
            if "-site:" in lowered or "site:" in lowered:
                url = "https://www.worldbank.org/corroboration"
                snippet = QUANT_SUPPORT
            else:
                slug = "".join(ch if ch.isalnum() else "-" for ch in lowered)[:30].strip("-") or "q"
                url = f"https://source-{slug}.org/page"
                snippet = (f"{text} is discussed in this source with supporting detail "
                           "and grounding context for the claim.")
            out.append({
                "url": url,
                "title": "source",
                "snippet": snippet,
                "content": "",
                "sub_question": text,
                "published_at": "2024-06-15",
            })
        return out


class _FakeLLM:
    def __init__(self, settings):
        self.settings = settings
        self.calls = []

    async def generate_json(self, system_prompt, user_prompt, retries=3, response_model=None):
        blob = f"{system_prompt}\n{user_prompt}".lower()
        self.calls.append(blob[:80])
        if "extract high-quality claims" in blob:
            return {"facts": []}
        if "sufficient" in blob or "critic" in blob:
            return {"is_sufficient": False, "reason": "gaps remain",
                    "improved_queries": ["follow up"]}
        if "final answer" in blob or "synthesize" in blob:
            return {"answer": "A concise grounded answer [1]."}
        return {"sub_questions": [{"question": "what is retrieval augmented generation",
                                   "axis": "definition", "search_type": "general",
                                   "wave": 0, "priority": 1, "minimum_sources": 1}]}


async def test_full_mocked_deep_run_terminates_and_consistent():
    settings = _settings(max_iterations=3, max_expansion_passes=3)
    llm = _FakeLLM(settings)
    search = _FakeSearch(settings)
    workflow = create_workflow(llm, search_client=search)
    state = build_initial_state("what is retrieval augmented generation", max_iterations=3,
                                mode="standard")
    final = dict(state)
    async for snapshot in workflow.astream(state, stream_mode="values"):
        final = {**final, **{k: v for k, v in snapshot.items() if v}}

    assert int(final.get("iteration", 0)) <= int(final.get("max_iterations", 3))
    assert final.get("final_report")
    # Registry (if any) is a plain dict keyed by claim text — no module-level
    # state leaked, and every entry is bounded.
    registry = final.get("corroboration_registry") or {}
    assert isinstance(registry, dict)
    for entry in registry.values():
        assert entry["attempts"] <= settings.max_corroboration_attempts
    # Tree state shape is consistent.
    assert isinstance(final.get("facts"), list)
    assert isinstance(final.get("search_results"), list)
