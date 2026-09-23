"""Central investigation-intelligence allocator (app/core/investigation_planner).

The allocator ranks candidate investigations across the EXISTING evidence
channels (corroboration, counter-evidence, primary-source, contradiction
resolution) by a single deterministic expected-value score and returns the
top-K within the per-pass budget. These tests lock in the scoring policy, the
deterministic tie-breaks, the novelty rule (exhausted/corroborated = zero gain)
and the workflow wiring.

All deterministic and LLM-free.
"""
import app.graph.workflow as wf
from app.core.config import Settings
from app.core.investigation_planner import (
    KIND_CONTRADICTION_RESOLUTION,
    KIND_CORROBORATION,
    KIND_COUNTER_EVIDENCE,
    KIND_DIMENSION_COVERAGE,
    KIND_PRIMARY_SOURCE,
    explain,
    select_investigations,
)

_AUTHORITATIVE_CLAIM = "Global AI capital expenditure reached $200 billion in 2025"
_PERIPHERAL_CLAIM = "RAG is a grounding technique"
_REUTERS = "https://reuters.com/tech/ai-capex"


def _fact(claim, source, **kw):
    base = {"claim": claim, "source": source, "verified": True}
    base.update(kw)
    return base


def _contradiction(claim_a=_AUTHORITATIVE_CLAIM, severity=0.9):
    return {
        "claim_a": claim_a,
        "claim_b": "A different source reports 300 billion instead",
        "kind": "numeric",
        "severity": severity,
        "resolved": False,
    }


def _base_state(**kw):
    state = {
        "facts": [
            _fact(_AUTHORITATIVE_CLAIM, _REUTERS),
            _fact(_PERIPHERAL_CLAIM, "https://blog.example.com/rag"),
        ],
        "contradictions": [],
        "sub_questions": [],
        "investigation_state": {},
    }
    state.update(kw)
    return state


# ---------------------------------------------------------------------------
# candidate generation from existing channels
# ---------------------------------------------------------------------------

def test_candidates_cover_existing_channels():
    state = _base_state(contradictions=[_contradiction()])
    result = select_investigations(state, 10)
    kinds = {c["kind"] for c in result["ranked"]}
    assert KIND_CORROBORATION in kinds
    assert KIND_COUNTER_EVIDENCE in kinds
    assert KIND_CONTRADICTION_RESOLUTION in kinds
    # Only the declared channel kinds ever appear (no new detector).
    assert kinds <= {
        KIND_CORROBORATION,
        KIND_COUNTER_EVIDENCE,
        KIND_CONTRADICTION_RESOLUTION,
        KIND_PRIMARY_SOURCE,
    }


def test_primary_candidates_only_for_thin_researched_dimensions():
    facts = [
        _fact("Output is described in secondary commentary", "https://blog.example.com/a",
              sub_question="What is the growth?"),
        _fact("Output is characterised differently here", "https://blog.example.com/b",
              sub_question="What is the growth?"),
    ]
    state = {
        "facts": facts,
        "contradictions": [],
        "sub_questions": [
            {"question": "What is the growth?", "search_type": "general", "domain": "general"},
        ],
    }
    result = select_investigations(state, 5)
    primary = [c for c in result["ranked"] if c["kind"] == KIND_PRIMARY_SOURCE]
    assert primary, "a thin researched dimension must produce a primary-source candidate"
    assert primary[0]["query"]


def test_no_candidate_for_dimension_with_zero_facts():
    """A planned dimension nobody researched is NOT detected here — that is the
    depth controller's uncovered-axis signal, not this channel's."""
    state = {
        "facts": [_fact("Output grew to 12 gigawatts", "https://blog.example.com/a",
                        sub_question="Some other dimension")],
        "contradictions": [],
        "sub_questions": [{"question": "Totally unresearched angle", "search_type": "general"}],
    }
    result = select_investigations(state, 5)
    assert not [c for c in result["ranked"] if c["kind"] == KIND_PRIMARY_SOURCE]


# ---------------------------------------------------------------------------
# determinism + explainability
# ---------------------------------------------------------------------------

def test_ranking_is_deterministic_and_explainable():
    state = _base_state(contradictions=[_contradiction()])
    a = select_investigations(state, 10)
    b = select_investigations(state, 10)
    assert [c["query"] for c in a["ranked"]] == [c["query"] for c in b["ranked"]]
    assert [c["score"] for c in a["ranked"]] == [c["score"] for c in b["ranked"]]
    for c in a["selected"]:
        assert c["reason"]
        assert "score" in c and "target" in c
    assert explain(a["selected"])[0]["reason"]


def test_scores_are_sorted_descending():
    state = _base_state(contradictions=[_contradiction()])
    ranked = select_investigations(state, 10)["ranked"]
    scores = [c["score"] for c in ranked]
    assert scores == sorted(scores, reverse=True)


def test_tie_break_is_stable():
    """Two corroboration candidates with identical scores order by target/query."""
    facts = [
        _fact("Alpha metric is defined as a measure", "https://b.example.com/x"),
        _fact("Beta metric is defined as a measure", "https://a.example.com/y"),
    ]
    state = {"facts": facts, "contradictions": [], "sub_questions": []}
    ranked = select_investigations(state, 10)["ranked"]
    targets = [c["target"] for c in ranked]
    assert targets == sorted(targets, key=lambda t: " ".join(t.lower().split()))
    # Re-running keeps the same order.
    assert [c["target"] for c in select_investigations(state, 10)["ranked"]] == targets


# ---------------------------------------------------------------------------
# novelty: exhausted / corroborated / attempted
# ---------------------------------------------------------------------------

def test_exhausted_and_corroborated_targets_have_zero_gain_and_are_not_selected():
    state = _base_state(
        investigation_state={
            "rag is a grounding technique": {
                "claim": _PERIPHERAL_CLAIM,
                "attempts": 2,
                "status": "exhausted",
                "max_attempts": 2,
            },
        }
    )
    result = select_investigations(state, 10)
    assert not [c for c in result["selected"] if c["target"] == _PERIPHERAL_CLAIM]
    exhausted = [c for c in result["ranked"] if c["target"] == _PERIPHERAL_CLAIM]
    assert exhausted and exhausted[0]["expected_gain"] == 0.0


def test_attempted_target_is_deprioritized_below_open_but_selectable():
    facts = [
        _fact("Alpha metric is defined as a measure", "https://b.example.com/x"),
        _fact("Beta metric is defined as a measure", "https://a.example.com/y"),
    ]
    open_state = {"facts": facts, "contradictions": [], "sub_questions": []}
    # Same pool, but the Beta claim has been attempted once (still fundable).
    attempted_state = {
        **open_state,
        "investigation_state": {
            "beta metric is defined as a measure": {
                "claim": "Beta metric is defined as a measure",
                "attempts": 1,
                "status": "attempted",
                "max_attempts": 2,
            },
        },
    }
    result = select_investigations(attempted_state, 10)
    by_target = {c["target"]: c for c in result["ranked"]}
    open_c = by_target["Alpha metric is defined as a measure"]
    attempted_c = by_target["Beta metric is defined as a measure"]
    assert attempted_c["expected_gain"] < open_c["expected_gain"]
    assert attempted_c["score"] < open_c["score"]
    # Still selectable with enough budget.
    assert attempted_c in result["selected"]


# ---------------------------------------------------------------------------
# budget selection + dedup
# ---------------------------------------------------------------------------

def test_top_k_never_exceeds_cap():
    state = _base_state(contradictions=[_contradiction()])
    for cap in (1, 2, 3):
        result = select_investigations(state, cap)
        assert len(result["selected"]) <= cap
        assert result["summary"]["budget_cap"] == cap


def test_budget_cap_equal_scores_deterministic_subset():
    state = _base_state(contradictions=[_contradiction()])
    first = select_investigations(state, 2)["selected"]
    second = select_investigations(state, 2)["selected"]
    assert [c["query"] for c in first] == [c["query"] for c in second]
    assert len(first) <= 2


def test_already_executed_queries_are_excluded():
    state = _base_state(contradictions=[_contradiction()])
    baseline = select_investigations(state, 10)
    top_query = baseline["selected"][0]["query"]
    executed_state = {**_base_state(contradictions=[_contradiction()]),
                      "executed_queries": [top_query]}
    rerun = select_investigations(executed_state, 10)
    assert top_query not in [c["query"] for c in rerun["selected"]]
    assert rerun["summary"]["excluded_executed"] >= 1


def test_selected_queries_are_deduped():
    state = _base_state(contradictions=[_contradiction()])
    queries = [c["query"] for c in select_investigations(state, 10)["selected"]]
    normalized = [" ".join(q.lower().split()) for q in queries]
    assert len(normalized) == len(set(normalized))


# ---------------------------------------------------------------------------
# priority: contradicted high-impact vs peripheral uncorroborated
# ---------------------------------------------------------------------------

def test_contradicted_high_impact_outranks_peripheral_uncorroborated():
    state = _base_state(contradictions=[_contradiction()])
    ranked = select_investigations(state, 10)["ranked"]
    contradicted_scores = [
        c["score"] for c in ranked if c["target"] == _AUTHORITATIVE_CLAIM
    ]
    peripheral_scores = [c["score"] for c in ranked if c["target"] == _PERIPHERAL_CLAIM]
    assert contradicted_scores and peripheral_scores
    assert max(peripheral_scores) < max(contradicted_scores)
    # The very first selected investigation targets the contradicted claim.
    assert select_investigations(state, 1)["selected"][0]["target"] == _AUTHORITATIVE_CLAIM


# ---------------------------------------------------------------------------
# fail-safe / total
# ---------------------------------------------------------------------------

def test_empty_and_garbage_state_never_raise():
    assert select_investigations({}, 3)["selected"] == []
    assert select_investigations(None, 3)["selected"] == []
    assert select_investigations({"facts": "garbage"}, 3)["selected"] == []
    assert select_investigations(
        {"facts": [1, 2, None], "contradictions": "nope", "investigation_state": 5}, 3
    )["selected"] == []
    assert select_investigations({}, 0)["selected"] == []


# ---------------------------------------------------------------------------
# workflow integration
# ---------------------------------------------------------------------------

def _settings(**kw):
    return Settings(groq_api_key="k", _env_file=None, **kw)


class _FakeSearch:
    def __init__(self, settings):
        self.settings = settings
        self.calls = []

    async def run_search(self, queries):
        self.calls.append(list(queries))
        return [
            {
                "url": "https://www.oecd.org/ai/investment-outlook",
                "title": "OECD AI investment outlook",
                "snippet": f"{_AUTHORITATIVE_CLAIM} per OECD figures.",
                "content": "",
                "sub_question": "",
            }
        ]


async def _run_search(state, settings):
    fake = _FakeSearch(settings)
    graph = wf.create_workflow(llm=None, search_client=fake)
    out = await graph.nodes["search"].ainvoke(state)
    return out, fake.calls


def _expansion_state(**kw):
    state = wf.build_initial_state("What is the current trend of AI?", 3, mode="deep")
    state.update(
        {
            "facts": [_fact(_AUTHORITATIVE_CLAIM, _REUTERS)],
            "contradictions": [],
            "corroboration_queries": [],
            "iteration": 1,
            "sub_questions": [],
            "investigation_state": {},
        }
    )
    state.update(kw)
    return state


async def test_search_node_issues_allocator_selected_queries():
    settings = _settings()
    out, calls = await _run_search(_expansion_state(), settings)
    assert calls, "search must still run a pass"
    issued = [q[0] if isinstance(q, (tuple, list)) else q for q in calls[0]]
    # The uncorroborated high-impact claim's allocation query is issued: it
    # targets the claim's terms and excludes the current publisher.
    assert any(
        "capital expenditure" in q and "-site:reuters.com" in q for q in issued
    ), issued


async def test_search_node_falls_back_to_channel_queries_when_allocator_empty():
    """No facts -> the allocator yields nothing -> the historical
    corroboration_queries channel is executed unchanged (deterministic fallback)."""
    settings = _settings()
    fallback_query = "legacy channel fallback query about RAG"
    state = _expansion_state(
        facts=[],
        contradictions=[],
        corroboration_queries=[fallback_query],
    )
    out, calls = await _run_search(state, settings)
    issued = [
        q[0] if isinstance(q, (tuple, list)) else q
        for call in calls
        for q in call
    ]
    assert fallback_query in issued, (calls, out)


async def test_search_node_preserves_executed_query_memory():
    settings = _settings()
    state = _expansion_state()
    out, calls = await _run_search(state, settings)
    issued = [q[0] if isinstance(q, (tuple, list)) else q for q in calls[0]]
    executed = set(out.get("executed_queries") or [])
    assert all(" ".join(q.lower().split()) in executed for q in issued)


# --- synthesis-plan-driven dimension-coverage channel ------------------------

def test_under_researched_dimension_gets_a_targeted_query():
    """A central-but-thin dimension produces a dimension_coverage candidate
    whose query is the dimension's own question text."""
    state = {
        "query": "What is the current state of the AI market?",
        "facts": [
            _fact(
                "AI market size reached 200 billion dollars in 2025",
                "https://reuters.com/ai-market",
                has_numbers=True,
                sub_question="market size",
                axis="evidence",
            ),
        ],
        "sub_questions": [
            {"question": "market size", "axis": "evidence"},
            {"question": "what is the adoption mechanism", "axis": "mechanism"},
        ],
        "contradictions": [],
        "intent": {"query_type": "analytical"},
    }
    result = select_investigations(state, budget_cap=8)
    coverage = [c for c in result["ranked"] if c["kind"] == KIND_DIMENSION_COVERAGE]
    assert coverage, "a central thin dimension must produce a coverage candidate"
    assert any(c["query"] == "market size" for c in coverage), coverage


def test_no_dimension_coverage_when_every_dimension_is_well_covered():
    """Two corroborated findings per dimension -> nothing is under-researched."""
    state = {
        "query": "What is the AI market?",
        "facts": [
            _fact(
                "AI market size reached 200 billion dollars",
                "https://reuters.com/a", has_numbers=True,
                corroborating_sources=["https://ft.com/a"],
                sub_question="market size", axis="evidence",
            ),
            _fact(
                "AI market grew 20 percent year over year",
                "https://who.int/a", has_numbers=True,
                corroborating_sources=["https://oecd.org/a"],
                sub_question="market size", axis="evidence",
            ),
        ],
        "sub_questions": [{"question": "market size", "axis": "evidence"}],
        "contradictions": [],
        "intent": {"query_type": "factual"},
    }
    result = select_investigations(state, budget_cap=8)
    assert not any(c["kind"] == KIND_DIMENSION_COVERAGE for c in result["ranked"])


def test_dimension_coverage_channel_is_total_on_garbage():
    for state in ({}, {"facts": None}, {"facts": ["garbage"], "sub_questions": None}):
        result = select_investigations(state, budget_cap=4)
        assert isinstance(result.get("selected"), list)


def test_uncovered_required_dimension_gets_targeted_followup():
    """A dimension the question REQUIRES but retrieval never covered must
    produce a targeted dimension_coverage candidate — the coverage-aware
    follow-up that fixes broad questions being researched too narrowly."""
    state = {
        "query": "What is the current trend of AI?",
        "facts": [
            _fact(
                "AI adoption reached 78 percent in 2025",
                "https://mckinsey.com/a", has_numbers=True,
                corroborating_sources=["https://gartner.com/a"],
                sub_question="adoption rates", axis="adoption rates",
            ),
        ],
        # The plan required regulation coverage; retrieval returned none.
        "sub_questions": [
            {"question": "adoption rates", "axis": "adoption rates"},
            {"question": "what regulation applies to AI", "axis": "regulation"},
        ],
        "contradictions": [],
        "intent": {"query_type": "analytical"},
    }
    result = select_investigations(state, budget_cap=8)
    coverage = [c for c in result["ranked"] if c["kind"] == KIND_DIMENSION_COVERAGE]
    queries = " ".join(c["query"].lower() for c in coverage)
    assert "regulation" in queries, coverage


def test_uncovered_required_dimension_outranks_thin_dimension():
    """A fully missing required dimension is a bigger gap than a thin-but-
    present one, so it ranks higher within the coverage channel."""
    state = {
        "query": "What is the current trend of AI?",
        "facts": [
            _fact(
                "AI adoption reached 78 percent in 2025",
                "https://mckinsey.com/a", has_numbers=True,
                sub_question="adoption rates", axis="adoption rates",
            ),
        ],
        "sub_questions": [
            {"question": "adoption rates", "axis": "adoption rates"},
            {"question": "what regulation applies to AI", "axis": "regulation"},
        ],
        "contradictions": [],
        "intent": {"query_type": "analytical"},
    }
    result = select_investigations(state, budget_cap=8)
    coverage = [c for c in result["ranked"] if c["kind"] == KIND_DIMENSION_COVERAGE]
    assert coverage
    # The uncovered "regulation" dimension leads the coverage candidates.
    assert "regulation" in coverage[0]["query"].lower(), coverage


def test_adequately_covered_required_dimension_triggers_no_followup():
    """Every required dimension has corroborated evidence -> no coverage
    follow-up is issued (no unnecessary search)."""
    state = {
        "query": "What is the current trend of AI?",
        "facts": [
            _fact(
                "AI adoption reached 78 percent in 2025",
                "https://mckinsey.com/a", has_numbers=True,
                corroborating_sources=["https://gartner.com/a"],
                sub_question="adoption rates", axis="adoption rates",
            ),
            _fact(
                "AI adoption grew 20 percent year over year",
                "https://pwc.com/a", has_numbers=True,
                corroborating_sources=["https://deloitte.com/a"],
                sub_question="adoption rates", axis="adoption rates",
            ),
            _fact(
                "AI regulation expanded across 40 jurisdictions in 2025",
                "https://oecd.org/a", has_numbers=True,
                corroborating_sources=["https://europa.eu/a"],
                sub_question="regulation", axis="regulation",
            ),
            _fact(
                "AI regulation added 12 new compliance rules in 2025",
                "https://nist.gov/a", has_numbers=True,
                corroborating_sources=["https://iso.org/a"],
                sub_question="regulation", axis="regulation",
            ),
        ],
        "sub_questions": [
            {"question": "adoption rates", "axis": "adoption rates"},
            {"question": "regulation", "axis": "regulation"},
        ],
        "contradictions": [],
        "intent": {"query_type": "analytical"},
    }
    result = select_investigations(state, budget_cap=8)
    coverage = [
        c for c in result["ranked"]
        if c["kind"] == KIND_DIMENSION_COVERAGE
        and any(w in c["query"].lower() for w in ("adoption", "regulation"))
    ]
    assert not coverage, coverage


