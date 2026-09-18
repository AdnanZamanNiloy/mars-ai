"""Planner regression test (Phase 1.8, mandatory per manual).

BUG-1 history: PLANNER_SYSTEM_PROMPT was referenced but never defined, so
every LLM planning call raised NameError and silently fell back to a fixed
4-question template. This test must never silently pass again: if the planner
falls back when it shouldn't, these tests fail.
"""
import app.agents.planner as planner_mod
from app.agents.planner import fallback_plan, planner_agent

LLM_PLAN = {
    "query_type": "comparative",
    "query_scope": "broad",
    "dominant_domain": "economics",
    "sub_questions": [
        {
            "id": 1,
            "question": "Levelized cost per MWh of nuclear vs solar in Bangladesh 2024",
            "axis": "comparison",
            "search_type": "statistical",
            "priority": 1,
            "depends_on": [],
            "coverage_goal": "cost comparison",
            "domain": "economics",
        },
        {
            "id": 2,
            "question": "Financing structure of the Rooppur nuclear plant in Bangladesh",
            "axis": "mechanism",
            "search_type": "news",
            "priority": 2,
            "depends_on": [],
            "coverage_goal": "financing details",
            "domain": "economics",
        },
    ],
    "coverage_note": "needs cost, financing and grid feasibility angles",
}


class FakeLLM:
    """Records whether the planner passed a defined system prompt."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def generate_json(self, system_prompt, user_prompt, retries=3, response_model=None):
        self.calls.append(system_prompt)
        if response_model is not None:
            return response_model.model_validate(self.payload).model_dump()
        return self.payload


QUERY = "Compare the economics of nuclear vs solar energy in Bangladesh"


async def test_planner_uses_llm_for_non_trivial_query():
    llm = FakeLLM(LLM_PLAN)
    result = await planner_agent(llm, QUERY)
    fallback = fallback_plan(QUERY)
    assert result != fallback, "Planner silently fell back to the fixed template (BUG-1 regression)"
    # The model's own contract survives verbatim (mandatory frontier tracks are
    # appended after it, so compare the first contract only).
    assert result[0] == {
            "id": 1,
            "question": "Levelized cost per MWh of nuclear vs solar in Bangladesh 2024",
            "axis": "comparison",
            "search_type": "statistical",
            "priority": 1,
            "depends_on": [],
            "coverage_goal": "cost comparison",
            "domain": "economics",
            "minimum_sources": 2,
            "stop_condition": "sufficient evidence for this axis",
            "variants": [],
            "agent": "financial_researcher",
            "tools": ["web_search"],
            "scope": [],
            "output_format": "structured_findings",
            # v3 contract enrichment: specialist overlay + primary-source
            # steering from the source registry.
            "specialist": "financial",
            "preferred_domains": ["worldbank.org", "imf.org", "oecd.org"],
            "primary_source_query": (
                "Levelized cost per MWh of nuclear vs solar in Bangladesh 2024 "
                "site:worldbank.org OR site:imf.org"
            ),
            "wave": 0,
            "sense": "",
        }
    assert result[1] == {
        "id": 2,
        "question": "Financing structure of the Rooppur nuclear plant in Bangladesh",
        "axis": "mechanism",
        "search_type": "news",
        "priority": 2,
        "depends_on": [],
        "coverage_goal": "financing details",
        "domain": "economics",
        "minimum_sources": 2,
        "stop_condition": "sufficient evidence for this axis",
        "variants": [],
        "agent": "financial_researcher",
        "tools": ["web_search"],
        "scope": [],
        "output_format": "structured_findings",
        "specialist": "financial",
        "preferred_domains": ["reuters.com", "apnews.com", "ft.com"],
        "primary_source_query": (
            "Financing structure of the Rooppur nuclear plant in Bangladesh "
            "site:reuters.com OR site:apnews.com"
        ),
        "wave": 0,
        "sense": "",
    }
    assert llm.calls, "planner never called the LLM"
    assert all("PLANNER_SYSTEM_PROMPT" not in c for c in llm.calls)


async def test_planner_llm_failure_returns_fallback():
    class ExplodingLLM:
        async def generate_json(self, *args, **kwargs):
            raise RuntimeError("LLM down")

    result = await planner_agent(ExplodingLLM(), QUERY)
    # Failure fallback is sized to the planner's default target (5), not the
    # bare fallback_plan() default (4): a degraded run is a smaller research
    # plan, not a different kind of plan.
    assert result == fallback_plan(QUERY, 5)


def test_planner_system_prompt_is_defined():
    # Direct regression guard for BUG-1: the constant must exist at import time.
    assert hasattr(planner_mod, "PLANNER_SYSTEM_PROMPT")
    assert "sub_questions" in planner_mod.PLANNER_SYSTEM_PROMPT


def test_fallback_plan_structure():
    plan = fallback_plan("what is retrieval augmented generation")
    assert len(plan) == 4
    # Axis-complete fallback: evidence + criticism guaranteed (the two angles
    # that separate research from recall), not definition variants.
    assert {q["axis"] for q in plan} == {"definition", "evidence", "criticism", "mechanism"}


async def test_planner_uses_llm_for_what_is_query():
    """No fast-path bypass: definitional queries get methodology-planned
    questions, not the fixed template (the cost governor caps spend)."""
    llm = FakeLLM(LLM_PLAN)
    result = await planner_agent(llm, "What is retrieval augmented generation?")
    assert result != fallback_plan("What is retrieval augmented generation?")
    assert llm.calls, "planner bypassed the LLM on a definitional query"
    # The model's two contracts survive; mandatory frontier tracks are appended.
    assert len(result) >= 2
    assert result[0]["question"] == LLM_PLAN["sub_questions"][0]["question"]


def test_planner_prompt_has_methodology_teeth():
    prompt = planner_mod.PLANNER_SYSTEM_PROMPT
    for marker in ("statistical", "criticism", "coverage_note", "CONCRETE"):
        assert marker in prompt, f"methodology marker missing: {marker}"
    assert "PLANNER_SYSTEM_PROMPT" not in prompt


def _llm_plan_with_variants():
    plan = dict(LLM_PLAN)
    plan["sub_questions"] = [
        {**LLM_PLAN["sub_questions"][0],
         "variants": ["Bangladesh nuclear solar LCOE comparison 2024",
                      "Levelized cost per MWh of nuclear vs solar in Bangladesh 2024"]},
        {**LLM_PLAN["sub_questions"][1], "variants": ["x"]},
    ]
    return plan


async def test_planner_keeps_valid_variants_drops_dupes_and_junk():
    llm = FakeLLM(_llm_plan_with_variants())
    result = await planner_agent(llm, QUERY)
    assert result[0]["variants"] == ["Bangladesh nuclear solar LCOE comparison 2024"]
    assert result[1]["variants"] == []


async def test_planner_passes_today_into_prompt():
    captured = {}

    class SpyLLM(FakeLLM):
        async def generate_json(self, system_prompt, user_prompt, retries=3, response_model=None):
            captured["user"] = user_prompt
            return await super().generate_json(system_prompt, user_prompt, retries, response_model)

    await planner_agent(SpyLLM(LLM_PLAN), QUERY, today="2026-09-10")
    assert "2026-09-10" in captured["user"]


def test_planner_prompt_has_temporal_and_variant_markers():
    prompt = planner_mod.PLANNER_SYSTEM_PROMPT
    assert "current year" in prompt
    assert "variants" in prompt
    assert "VARIANTS" in prompt


def test_fallback_plan_has_empty_variants():
    for item in fallback_plan("What is RAG?"):
        assert item["variants"] == []


async def test_planner_receives_web_context_snippets():
    """Search-informed planning: the planner prompt must carry the raw
    query's real search snippets so the plan targets what the web actually
    contains (gpt-researcher parity) — never plan blind."""
    captured = {}

    class SpyLLM(FakeLLM):
        async def generate_json(self, system_prompt, user_prompt, retries=3, response_model=None):
            captured["user"] = user_prompt
            return await super().generate_json(system_prompt, user_prompt, retries, response_model)

    await planner_agent(
        SpyLLM(LLM_PLAN), QUERY,
        context_snippets=[
            "Rooppur nuclear power plant: 2,400 MW VVER-1200 reactors, Rosatom turnkey contract",
            "Solar LCOE in Bangladesh fell below grid parity in 2024 per IRENA",
        ],
    )
    assert "Web context" in captured["user"]
    assert "Rooppur nuclear power plant" in captured["user"]
    assert "ground" in captured["user"], "context block must tell the planner to ground, not answer"


async def test_planner_omits_empty_context_block():
    captured = {}

    class SpyLLM(FakeLLM):
        async def generate_json(self, system_prompt, user_prompt, retries=3, response_model=None):
            captured["user"] = user_prompt
            return await super().generate_json(system_prompt, user_prompt, retries, response_model)

    await planner_agent(SpyLLM(LLM_PLAN), QUERY, context_snippets=["", "  "])
    assert "Web context" not in captured["user"]


def _six_question_plan():
    axes = ["definition", "mechanism", "application", "evidence", "criticism", "outlook"]
    plan = dict(LLM_PLAN)
    plan["sub_questions"] = [
        {**LLM_PLAN["sub_questions"][0], "id": i + 1,
         "question": f"Distinct research angle number {i + 1} on Bangladesh energy policy options",
         "axis": axis}
        for i, axis in enumerate(axes)
    ]
    return plan


async def test_planner_honours_target_count():
    """The orchestrator's budget caps the MODEL's plan. The mandatory frontier
    tracks are then appended on top (they are never subject to the target)."""
    from app.agents.planner import COUNTER_EVIDENCE_AXIS, FRONTIER_AXES

    llm = FakeLLM(_six_question_plan())
    result = await planner_agent(llm, QUERY, target_count=3)
    model_contracts = [
        c for c in result
        if str(c.get("axis", "")) not in set(FRONTIER_AXES) | {COUNTER_EVIDENCE_AXIS}
    ]
    assert len(model_contracts) <= 3
    # Every mandatory track is present regardless of the target.
    axes = {str(c.get("axis", "")) for c in result}
    assert set(FRONTIER_AXES) <= axes
    assert COUNTER_EVIDENCE_AXIS in axes


async def test_planner_enforces_required_axes():
    """A plan missing the required evidence/criticism angles gets them
    injected deterministically instead of shipping background-only."""
    llm = FakeLLM(LLM_PLAN)  # comparison + mechanism only
    result = await planner_agent(llm, QUERY, required_axes=("evidence", "criticism"))
    axes = {q["axis"] for q in result}
    assert {"evidence", "criticism"} <= axes


def test_execution_waves_orders_dependencies():
    from app.agents.planner import execution_waves, sanitize_dependencies

    plan = sanitize_dependencies([
        {"id": 1, "question": "a", "depends_on": []},
        {"id": 2, "question": "b", "depends_on": [1]},
        {"id": 3, "question": "c", "depends_on": [1]},
    ])
    waves = execution_waves(plan)
    assert [q["id"] for q in waves[0]] == [1]
    assert sorted(q["id"] for q in waves[1]) == [2, 3]


def test_sanitize_dependencies_breaks_cycles_to_roots():
    from app.agents.planner import execution_waves, sanitize_dependencies

    plan = sanitize_dependencies([
        {"id": 1, "question": "a", "depends_on": [2]},
        {"id": 2, "question": "b", "depends_on": [1]},
    ])
    # Cycle broken into an order: node 1 is the re-normalized root (wave 0),
    # node 2 follows (wave 1) instead of both stranding in a later wave.
    assert plan[0]["depends_on"] == [] and plan[0]["wave"] == 0
    waves = execution_waves(plan)
    assert [q["id"] for q in waves[0]] == [1]
    assert [q["id"] for q in waves[1]] == [2]


def test_axis_enforcement_injects_mechanism_why_angle():
    """The live deep run produced plans whose axes were always
    [definition, evidence, criticism] — no WHY/mechanism angle — so the report
    recited WHAT without explaining WHY. Mechanism injection closes it."""
    from app.agents.planner import _contract, enforce_axis_coverage

    plan = [_contract(index=1, question="ai definition", axis="definition",
                      search_type="encyclopedia", priority=1, domain="general")]
    plan, injected = enforce_axis_coverage(
        plan, "What is the current trend of AI?",
        ["evidence", "criticism", "mechanism"], today="2026-09-14",
    )
    assert "mechanism" in injected
    mech = [p for p in plan if p["axis"] == "mechanism"]
    assert mech, [p["axis"] for p in plan]
    # The injected angle must read as a search-ready causal query, not a label.
    assert "why" in mech[0]["question"].lower() or "cause" in mech[0]["question"].lower()


def test_plan_has_no_overlapping_questions_after_dedupe():
    """Two questions differing only in phrasing must not both survive — a
    duplicate angle burns a whole search agent for the same retrieval."""
    from app.agents.planner import deduplicate_semantic

    plan = deduplicate_semantic([
        {"question": "what is the current trend of AI in business",
         "axis": "definition"},
        {"question": "current AI business trend what is it",
         "axis": "definition"},
        {"question": "global AI capital expenditure 2025 official statistics",
         "axis": "evidence"},
    ])
    assert len(plan) == 2
    axes = {p["axis"] for p in plan}
    assert axes == {"definition", "evidence"}


def test_planner_schema_accepts_surplus_variants_without_failing():
    """Regression: 3+ variants must not fail the whole plan payload.

    SubQuestionModel capped variants at 2, but models routinely emit 3-4 — one
    surplus variant raised a pydantic ValidationError that discarded the ENTIRE
    plan and silently degraded to the deterministic template (observed live as
    the same generic axes every pass). The parser already slices to 2, so the
    schema must accept surplus and let the parser trim."""
    from app.core.schemas import PlannerOutputModel

    model = PlannerOutputModel.model_validate({
        "sub_questions": [{
            "id": 1,
            "question": "global AI capex 2025",
            "variants": ["a", "b", "c", "d"],
            "scope": ["x", "y", "z", "p", "q", "r"],
        }],
    })
    assert len(model.sub_questions[0].variants) == 4


def test_planner_trims_surplus_variants_to_two():
    """The prompt promises 1-2 variants; trimming happens in the parser."""
    from app.agents.planner import planner_agent
    import asyncio

    payload = {
        "query_type": "analytical",
        "query_scope": "broad",
        "dominant_domain": "general",
        "sub_questions": [{
            "id": 1,
            "question": "why is enterprise AI adoption accelerating 2026",
            "axis": "mechanism",
            "search_type": "academic",
            "priority": 1,
            "variants": ["v one", "v two", "v three", "v four"],
        }],
        "coverage_note": "",
    }
    result = asyncio.run(planner_agent(FakeLLM(payload), "why is AI accelerating?"))
    assert result
    assert len(result[0]["variants"]) <= 2


def test_mechanism_required_for_trend_query():
    """A recency/trend query must require a WHY angle even when the lexical
    classifier types it 'factual' (the live 'current trend of AI' case)."""
    from app.agents.orchestrator import plan_targets, score_complexity

    targets = plan_targets(score_complexity("What is the current trend of AI?"), "deep", 5)
    assert "mechanism" in targets.required_axes


def test_required_axes_survive_truncation():
    """Required axes are a contract, not a preference: with a target smaller
    than the required-axis set, every required axis must still be selected.

    Regression: `select_plan` broke its required-axis loop at target_count, so
    on the live deep run (target=3, axes=[definition, evidence, criticism,
    mechanism, outlook]) the injected mechanism and outlook contracts were
    chosen and immediately dropped — the plan collapsed to the same three axes
    on every pass, and the WHY angle never reached the report."""
    from app.agents.planner import _contract, select_plan

    axes = ["definition", "evidence", "criticism", "mechanism", "outlook"]
    plan = [
        _contract(index=i + 1, question=f"q{i}", axis=a,
                  search_type="encyclopedia", priority=1, domain="general")
        for i, a in enumerate(axes)
    ]
    out = select_plan(plan, target_count=3, required_axes=axes)
    assert set(axes) <= {p["axis"] for p in out}


def test_optional_axes_still_respect_the_budget():
    """Raising required axes above the budget must not let optional angles in."""
    from app.agents.planner import _contract, select_plan

    plan = [
        _contract(index=1, question="a", axis="evidence",
                  search_type="statistical", priority=1, domain="general"),
        _contract(index=2, question="b", axis="outlook",
                  search_type="news", priority=2, domain="general"),
        _contract(index=3, question="c", axis="history",
                  search_type="encyclopedia", priority=3, domain="general"),
    ]
    out = select_plan(plan, target_count=2, required_axes=["evidence"])
    assert [p["axis"] for p in out] == ["evidence", "outlook"]
