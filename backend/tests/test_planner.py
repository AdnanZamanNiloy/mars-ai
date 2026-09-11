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
    assert result == [
        {
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
            "agent": "",
            "tools": ["web_search"],
            "scope": [],
            "output_format": "structured_findings",
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
            "minimum_sources": 2,
            "stop_condition": "sufficient evidence for this axis",
            "variants": [],
            "agent": "",
            "tools": ["web_search"],
            "scope": [],
            "output_format": "structured_findings",
        },
    ]
    assert llm.calls, "planner never called the LLM"
    assert all("PLANNER_SYSTEM_PROMPT" not in c for c in llm.calls)


async def test_planner_llm_failure_returns_fallback():
    class ExplodingLLM:
        async def generate_json(self, *args, **kwargs):
            raise RuntimeError("LLM down")

    result = await planner_agent(ExplodingLLM(), QUERY)
    assert result == fallback_plan(QUERY)


def test_planner_system_prompt_is_defined():
    # Direct regression guard for BUG-1: the constant must exist at import time.
    assert hasattr(planner_mod, "PLANNER_SYSTEM_PROMPT")
    assert "sub_questions" in planner_mod.PLANNER_SYSTEM_PROMPT


def test_fallback_plan_structure():
    plan = fallback_plan("what is retrieval augmented generation")
    assert len(plan) == 4
    assert {q["axis"] for q in plan} == {"definition", "mechanism", "application", "criticism"}


async def test_planner_uses_llm_for_what_is_query():
    """No fast-path bypass: definitional queries get methodology-planned
    questions, not the fixed template (the cost governor caps spend)."""
    llm = FakeLLM(LLM_PLAN)
    result = await planner_agent(llm, "What is retrieval augmented generation?")
    assert result != fallback_plan("What is retrieval augmented generation?")
    assert llm.calls, "planner bypassed the LLM on a definitional query"
    assert len(result) == 2


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
