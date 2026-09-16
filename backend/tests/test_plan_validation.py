"""Deterministic plan validator (required-angle enforcement).

The directive stage's prompt REQUESTS a quantitative angle for quantitative
queries (R3) and decision/comparison/counter-evidence angles for their query
shapes (R4), but nothing verified compliance — the same class of bug as the
prompt/parser drift in AGENTS.md 4.2. `validate_plan_dimensions` enforces those
requirements deterministically (no new LLM calls): missing angles are APPENDED
to the model's plan and added to `must_cover`; an already-valid plan is
returned unchanged and never reordered.
"""
import asyncio

from app.agents.planner import (
    dimension_to_axis,
    plan_dimensions,
    validate_plan_dimensions,
)
from app.core.schemas import PlanningDirectiveModel


def _dimensions_for(query, dims, must=None):
    out_dims, out_cover, meta = validate_plan_dimensions(query, dims, must or [])
    return out_dims, out_cover, meta


def test_quantitative_query_without_quantitative_dimension_gets_one():
    dims, cover, meta = _dimensions_for(
        "How much did solar electricity cost per MWh in 2024?",
        ["definition", "mechanism of action"],
    )
    assert "quantitative evidence" in dims
    assert "quantitative evidence" in cover
    assert "quantitative evidence" in meta["added"]
    # The model's plan is preserved and not reordered.
    assert dims[:2] == ["definition", "mechanism of action"]


def test_decision_query_gets_tradeoff_dimension():
    dims, cover, meta = _dimensions_for(
        "Should we invest in nuclear power for Bangladesh?",
        ["definition", "mechanism of action"],
    )
    assert "decision angle" in dims
    assert "decision angle" in cover


def test_contested_topic_gets_counter_evidence_dimension():
    dims, _, meta = _dimensions_for(
        "Is the claim that mRNA vaccines are harmful controversial?",
        ["definition", "mechanism of action"],
    )
    assert "contested angle" in dims


def test_comparative_query_gets_comparison_dimension():
    dims, cover, meta = _dimensions_for(
        "Compare nuclear vs solar for baseload power",
        ["definition", "mechanism of action"],
    )
    assert "comparison angle" in dims
    assert "comparison angle" in cover


def test_already_valid_plan_is_returned_unchanged():
    original = ["quantitative evidence", "decision tradeoffs and risks", "cost per unit"]
    dims, cover, meta = _dimensions_for(
        "Should we invest in solar given levelized cost per MWh?",
        list(original),
        ["quantitative evidence"],
    )
    assert dims == original
    assert cover == ["quantitative evidence"]
    assert meta["added"] == []


def test_exact_duplicates_are_deduplicated_without_reordering():
    dims, cover, meta = _dimensions_for(
        "what is retrieval augmented generation?",
        ["Definition", "definition", "evidence and data", "Evidence and Data"],
    )
    assert dims == ["Definition", "evidence and data"]
    assert meta["duplicate_dimensions"] is True


def test_required_angles_are_validated_against_axis_not_just_tokens():
    """A contract already serving the required canonical axis counts as
    coverage even when its label lacks a marker (no redundant injection)."""
    dims, _, meta = _dimensions_for(
        "Compare nuclear vs solar for baseload power",
        ["head-to-head comparison"],
    )
    comparison = [d for d in dims if dimension_to_axis(d) == "comparison"]
    assert len(comparison) == 1
    assert meta["added"] == []


def test_token_only_axis_label_is_still_covered_by_vocabulary():
    """The second half of coverage: a 'cost per unit' label is quantitative by
    vocabulary even though its canonical axis is `cost`, not `evidence`."""
    dims, _, meta = _dimensions_for(
        "How much does utility-scale solar cost per MW?",
        ["cost per unit"],
    )
    assert "quantitative" not in " ".join(meta["added"])
    assert meta["added"] == []


def test_must_cover_entries_all_map_to_dimensions():
    dims, cover, _ = _dimensions_for(
        "How much does utility-scale solar cost per MW?",
        ["definition"],
        ["definition", "a dimension that does not exist"],
    )
    dim_keys = {d.lower() for d in dims}
    assert all(c.lower() in dim_keys for c in cover)
    assert "a dimension that does not exist" not in cover


def test_empty_input_does_not_raise_and_returns_unchanged():
    dims, cover, meta = _dimensions_for("Should we invest?", [])
    assert dims == [] and cover == []
    assert meta["applied"] is False


def test_classifier_failure_skips_validation(monkeypatch):
    import app.agents.orchestrator as orch

    def _boom(query):
        raise RuntimeError("classifier down")

    monkeypatch.setattr(orch, "score_complexity", _boom)
    dims, cover, meta = validate_plan_dimensions(
        "How much does solar cost?", ["definition"], ["definition"]
    )
    assert meta["applied"] is False
    assert dims == ["definition"]


class _DirectiveLLM:
    """Returns a scripted directive, then a scripted plan payload."""

    def __init__(self, directive_payload, plan_payload=None):
        self.directive_payload = directive_payload
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

    async def generate_json(self, system_prompt, user_prompt, retries=3, response_model=None):
        if response_model is PlanningDirectiveModel:
            return response_model.model_validate(self.directive_payload).model_dump()
        if response_model is not None:
            return response_model.model_validate(self.plan_payload).model_dump()
        return self.plan_payload


def test_heuristic_return_path_is_also_validated():
    """When the directive yields no dimensions, the heuristic list is validated
    too — a decision query must still carry the required decision angle even
    though the coarse heuristic list has none ('prioritize' is a decision
    marker for the orchestrator, not in the heuristic's own short list)."""
    llm = _DirectiveLLM(
        {"dimensions": [], "must_cover": [], "query_type": "analytical"},
    )
    dims, cover, meta = asyncio.run(
        plan_dimensions(llm, "Prioritize the options for grid storage rollouts")
    )
    assert meta["planned_by"] == "heuristic"
    assert "decision angle" in dims
    assert "decision angle" in cover
    assert "decision angle" in meta["plan_validation"]["added"]


def test_llm_return_path_is_validated_and_meta_records_additions():
    llm = _DirectiveLLM(
        {
            "query_type": "analytical",
            "dominant_domain": "general",
            "reasoning": "test",
            "dimensions": ["definition", "mechanism of action"],
            "must_cover": ["definition"],
            "coverage_note": "test",
        },
    )
    dims, cover, meta = asyncio.run(
        plan_dimensions(llm, "How much does utility-scale solar cost per MW?")
    )
    assert meta["planned_by"] == "llm"
    assert "quantitative evidence" in dims
    assert "quantitative evidence" in cover
    assert "quantitative evidence" in meta["plan_validation"]["added"]
