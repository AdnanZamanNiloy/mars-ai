"""Dynamic planning (Phase 2): query-type-specific dimensions.

The defect these tests lock down: the old pipeline injected a fixed axis
template (mechanism/outlook/risk/cost/history/regulation) whenever the plan
model omitted a required axis, so different query TYPES converged on the same
axes. Dynamic planning asks the directive stage which dimensions THIS query
needs, and the plan must cover whatever it returns.
"""
import asyncio

import app.agents.planner as planner_mod
from app.agents.planner import (
    _heuristic_dimensions,
    dimension_to_axis,
    enforce_axis_coverage,
    fallback_plan,
    plan_dimensions,
    planner_agent,
    select_plan,
    _contract,
)
from app.core.schemas import PlanningDirectiveModel


class DirectiveLLM:
    """Returns a scripted dimension directive, then a scripted plan."""

    def __init__(self, dimensions, plan_payload=None):
        self.dimensions = dimensions
        self.plan_payload = plan_payload or {
            "query_type": "analytical",
            "query_scope": "broad",
            "dominant_domain": "general",
            "sub_questions": [{
                "id": 1,
                "question": "A specific search-ready sub-question about the topic",
                "axis": "general",
                "search_type": "academic",
                "priority": 1,
                "depends_on": [],
            }],
            "coverage_note": "",
        }
        self.system_prompts = []
        self.user_prompts = []

    async def generate_json(self, system_prompt, user_prompt, retries=3, response_model=None):
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        if response_model is PlanningDirectiveModel:
            payload = {
                "query_type": "analytical",
                "dominant_domain": "general",
                "reasoning": "test",
                "dimensions": list(self.dimensions),
                "must_cover": list(self.dimensions[:2]),
                "coverage_note": "test",
            }
            return response_model.model_validate(payload).model_dump()
        if response_model is not None:
            return response_model.model_validate(self.plan_payload).model_dump()
        return self.plan_payload


def test_directive_prompt_and_schema_are_in_sync():
    """AGENTS.md 4.2: the prompt's declared schema, the pydantic model and the
    parser must agree on the keys."""
    prompt = planner_mod.PLANNING_DIRECTIVE_PROMPT
    for key in ("dimensions", "must_cover", "query_type", "dominant_domain",
                "reasoning", "coverage_note"):
        assert f'"{key}"' in prompt, f"directive prompt missing key {key}"
    for field in ("dimensions", "must_cover", "query_type", "dominant_domain"):
        assert field in PlanningDirectiveModel.model_fields


def test_heuristic_dimensions_differ_by_query_type():
    """Deterministic fallback must also be query-type-aware, not one template."""
    decision = _heuristic_dimensions("Should we invest in nuclear power?")
    mechanism = _heuristic_dimensions("How does mRNA vaccine technology work?")
    causal = _heuristic_dimensions("What caused the 2023 banking crisis?")
    comparative = _heuristic_dimensions("Compare solar vs nuclear for baseload")

    assert any("cost" in d or "financ" in d for d in decision)
    assert any("risk" in d or "feasib" in d for d in decision)
    assert any("mechanism" in d for d in mechanism)
    assert any("cause" in d or "causal" in d for d in causal)
    assert any("comparison" in d or "head-to-head" in d for d in comparative)

    assert set(decision) != set(mechanism) != set(causal)
    # The old failure mode: identical axis sets across unrelated query types.
    assert decision != comparative != causal


def test_llm_dimensions_flow_into_required_axes_and_contracts():
    """The directive's dimensions become the plan's hard requirements: a plan
    that omits one gets a contract for it."""
    llm = DirectiveLLM(["cost and financing", "implementation feasibility"])
    result = asyncio.run(planner_agent(llm, "Should Bangladesh build nuclear?"))
    axes = {q["axis"] for q in result}
    # Aliases map to canonical axes: cost and financing -> cost, feasibility
    # -> its own slug.
    assert "cost" in axes
    assert len(result) >= 2


def test_different_query_types_produce_different_dimension_sets():
    """The core diversity invariant."""
    outputs = {}
    for query, dims in {
        "decision": ["cost and financing", "risk and feasibility"],
        "mechanism": ["mechanism of action", "delivery system"],
        "causal": ["causal mechanism", "alternative explanations"],
    }.items():
        llm = DirectiveLLM(dims)
        plan = asyncio.run(planner_agent(llm, f"query for {query}"))
        outputs[query] = frozenset(
            dimension_to_axis(q["axis"]) for q in plan
        )
    assert outputs["decision"] != outputs["mechanism"]
    assert outputs["mechanism"] != outputs["causal"]
    assert outputs["decision"] != outputs["causal"]


def test_dimension_directive_fallback_fires_when_llm_fails():
    class ExplodingLLM:
        async def generate_json(self, *a, **k):
            raise RuntimeError("directive LLM down")

    dims, must, meta = asyncio.run(
        plan_dimensions(ExplodingLLM(), "Should we invest in nuclear?")
    )
    assert meta["planned_by"] == "heuristic"
    assert dims, "fallback must still yield dimensions"
    assert must, "fallback must yield must-cover dimensions"


def test_directive_empty_dimensions_falls_back():
    class EmptyLLM:
        async def generate_json(self, system_prompt, user_prompt, retries=3, response_model=None):
            if response_model is PlanningDirectiveModel:
                return response_model.model_validate(
                    {"dimensions": [], "must_cover": []}
                ).model_dump()
            return {"sub_questions": []}

    dims, must, meta = asyncio.run(plan_dimensions(EmptyLLM(), "what is a transformer?"))
    assert meta["planned_by"] == "heuristic"
    assert dims


def test_enforce_axis_coverage_synthesizes_dynamic_dimension():
    """A required dynamic dimension with no plan contract is synthesized as a
    search-ready query carrying that dimension, not a generic template."""
    plan, injected = enforce_axis_coverage(
        [_contract(index=1, question="q1", axis="definition",
                   search_type="encyclopedia", priority=1, domain="general")],
        "Should Bangladesh build nuclear?",
        ["cost and financing"],
        today="2026-09-14",
    )
    assert "cost" in injected
    cost = [p for p in plan if p["axis"] == "cost"]
    assert cost
    # Canonical dimensions use their canonical retrieval phrasing...
    assert "cost" in cost[0]["question"].lower()

    # ...while a genuinely query-specific dimension is synthesized from its own
    # label + the query concept (no stock template).
    plan2, injected2 = enforce_axis_coverage(
        [_contract(index=1, question="q1", axis="definition",
                   search_type="encyclopedia", priority=1, domain="general")],
        "What caused the 2023 banking crisis?",
        ["contagion dynamics"],
        today="2026-09-14",
    )
    assert "contagion_dynamics" in injected2
    synth = [p for p in plan2 if p["axis"] == "contagion_dynamics"]
    assert synth
    assert "contagion dynamics" in synth[0]["question"].lower()


def test_select_plan_protects_dynamic_dimensions():
    """Required dimensions survive truncation even below the budget."""
    plan = [
        _contract(index=1, question="a", axis="definition",
                  search_type="encyclopedia", priority=1, domain="general"),
        _contract(index=2, question="b", axis="cost",
                  search_type="statistical", priority=1, domain="general"),
        _contract(index=3, question="c", axis="risk",
                  search_type="academic", priority=1, domain="general"),
    ]
    out = select_plan(plan, target_count=1, required_axes=["cost and financing", "risk"])
    assert {"cost", "risk"} <= {p["axis"] for p in out}


def test_planner_does_not_replan_dimensions_on_expansion():
    """An expansion pass (critique_feedback present) must not re-run the
    directive and undo per-axis expansion."""
    llm = DirectiveLLM(["some dynamic dimension"])
    asyncio.run(planner_agent(
        llm, "What is the current trend of AI?",
        critique_feedback="gap: add a cost angle",
        required_axes=["definition", "evidence"],
    ))
    # The directive prompt must not have been sent on the expansion pass.
    assert all("Planning Directive" not in p for p in llm.system_prompts)


def test_expansion_passes_existing_axes_not_static_orchestration_axes(monkeypatch):
    """Regression: the workflow reseeded required_axes from the orchestration's
    static axis list on every expansion pass, re-injecting the generic
    definition/evidence/criticism/mechanism contracts alongside the dynamic
    dimensions the first pass derived. Expansion must pass the axes already
    under research (the existing contracts' axes), not the static list."""
    import app.graph.workflow as wf
    from app.core.config import Settings
    from app.core.llm import LLMClient

    captured = {}

    async def fake_planner(llm, query, critique_feedback="", **kwargs):
        captured.setdefault("required_axes_calls", []).append(
            list(kwargs.get("required_axes") or [])
        )
        return [{
            "id": 1, "question": "What is RAG?",
            "axis": "retrieval_architecture", "search_type": "academic",
            "priority": 1, "depends_on": [],
        }]

    async def fake_summarizer(llm, query, search_results=None, specialist_role="general"):
        return []

    async def fake_critic(llm, query, facts=None, iteration=1, max_iterations=3, **kwargs):
        # Force exactly one expansion pass.
        if iteration == 1:
            return {"is_sufficient": False, "reason": "thin",
                    "improved_queries": ["more"], "confidence": 0.4}
        return {"is_sufficient": True, "reason": "ok",
                "improved_queries": [], "confidence": 0.9}

    monkeypatch.setattr(wf, "planner_agent", fake_planner)
    monkeypatch.setattr(wf, "summarizer_agent", fake_summarizer)
    monkeypatch.setattr(wf, "critic_agent", fake_critic)
    monkeypatch.setattr(wf, "verify_facts", lambda facts, search_results: facts)

    class StubSearch:
        async def run_search(self, questions):
            return [{"url": f"https://test.com/{i}", "sub_question":
                     q[0] if isinstance(q, (tuple, list)) else q,
                     "snippet": "s", "content": "c"} for i, q in enumerate(questions)]

    settings = Settings(groq_api_key="k", _env_file=None)
    state = wf.build_initial_state("What is RAG?", 3)
    workflow = wf.create_workflow(LLMClient(settings), StubSearch())
    import asyncio as _asyncio

    async def _drive():
        async for _ in workflow.astream(state, stream_mode="values"):
            pass

    _asyncio.run(_drive())

    calls = captured.get("required_axes_calls") or []
    assert len(calls) >= 2, calls
    # Expansion (call 2) must carry the dynamic axis under research, not the
    # orchestration's static definition/evidence/criticism list.
    assert "retrieval_architecture" in calls[1]
    assert "evidence" not in calls[1]


def test_fallback_plan_still_axis_complete():
    """Legacy contract preserved: fallback still yields the canonical safety
    net dimensions (AGENTS.md 4.7 / 3.7)."""
    plan = fallback_plan("what is retrieval augmented generation")
    assert {q["axis"] for q in plan} == {
        "definition", "evidence", "criticism", "mechanism",
    }


def test_dimension_to_axis_maps_and_preserves():
    assert dimension_to_axis("head-to-head comparison") == "comparison"
    assert dimension_to_axis("counter-evidence and criticism") == "criticism"
    assert dimension_to_axis("cost and financing") == "cost"
    assert dimension_to_axis("self-attention mechanism") == "mechanism"
    # A genuinely query-specific dimension keeps its own slug, not "general".
    assert dimension_to_axis("proximate failure mechanics") == "proximate_failure_mechanics"
