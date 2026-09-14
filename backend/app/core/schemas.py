"""Pydantic response models mirroring each agent's expected LLM output schema.

Phase 1.2: passed as `response_model` to LLMClient.generate_json so malformed
LLM output is rejected and retried instead of silently accepted.

Schema rule (AGENTS.md 4.2): every field here must match the keys the
corresponding agent's system prompt documents AND the keys its parsing
code reads. Keep the three in sync.
"""
from typing import Any, List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class PlanningDirectiveModel(BaseModel):
    """LLM-authored research plan: the dimensions THIS query needs.

    Phase 2 (dynamic planning) replaces the generic axis-template injection
    (mechanism/outlook/risk/cost/history/regulation strings added whenever the
    model omitted one) with the model's own query-specific dimensions. The
    schema mirrors the PLANNING_DIRECTIVE_PROMPT output block and the parser in
    planner_agent — all three must stay in sync (AGENTS.md 4.2).
    """
    query_type: Literal["factual", "comparative", "analytical", "exploratory"] = "factual"
    dominant_domain: str = "general"
    reasoning: str = ""
    # Free-form, query-specific dimension labels (2-4 words). These become the
    # contract axes; the prompt tells the model to name what the query needs,
    # not to pick from a fixed enum.
    dimensions: List[str] = Field(default_factory=list, max_length=8)
    # The one dimension that, if missing, would leave the question unanswered.
    must_cover: List[str] = Field(default_factory=list, max_length=4)
    coverage_note: str = ""
    # Optional retrieval vocabulary the model proposes for the primary sources.
    preferred_search_types: List[str] = Field(default_factory=list, max_length=8)


class SubQuestionModel(BaseModel):
    id: int = 1
    question: str = Field(min_length=1)
    axis: str = "general"
    search_type: str = "encyclopedia"
    priority: int = 2
    depends_on: List[int] = Field(default_factory=list)
    coverage_goal: str = ""
    domain: str = "general"
    minimum_sources: int = Field(default=2, ge=1)
    stop_condition: str = "sufficient evidence for this axis"
    # The planner prompt asks for 1-2 variants, but models routinely emit 3-4.
    # max_length=2 made that a HARD validation error on the WHOLE plan payload:
    # one extra variant discarded the entire plan and silently degraded to the
    # deterministic template (observed live as the same 3 generic axes every
    # pass). planner._clean_str_list / the `variants[:2]` slice already resolve
    # the surplus, so the schema accepts up to 8 and lets the parser trim —
    # validation should reject malformed output, not well-formed-but-extra.
    variants: List[str] = Field(default_factory=list, max_length=8)
    agent: str = ""
    tools: List[str] = Field(default_factory=lambda: ["web_search"])
    scope: List[str] = Field(default_factory=list, max_length=8)
    output_format: str = "structured_findings"
    # Intent sense label this sub-question researches ("" when unambiguous).
    sense: str = ""


class PlannerOutputModel(BaseModel):
    query_type: Literal["factual", "comparative", "analytical", "exploratory"] = "factual"
    query_scope: Literal["narrow", "broad"] = "narrow"
    dominant_domain: str = "general"
    sub_questions: List[SubQuestionModel] = Field(min_length=1)
    coverage_note: str = ""


class FactModel(BaseModel):
    claim: str = Field(min_length=1)
    source: str = ""
    confidence: float = 0.0
    # Optional verbatim fragment from the cited source grounding this claim.
    # Retained (not validated) so downstream quote-verification can use it;
    # "" means "no quote supplied", never "quote checked and missing".
    direct_quote: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0


class SummarizerFactsModel(BaseModel):
    # An EMPTY facts list is a valid model outcome (the sources genuinely had
    # nothing extractable for this contract). It must NOT be a validation
    # error: min_length=1 turned "found nothing" into a transient-looking
    # failure, retried 3x, then cascaded into the heuristic fallback and a
    # false "providers unavailable / degraded" report on every run. Malformed
    # payloads (no list at all) still fail via _coerce_fact_lists below.
    facts: List[FactModel] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_fact_lists(cls, data: Any) -> Any:
        """Accept the shapes models actually return: a bare list, or the
        list under `claims`/`results`/`items` instead of `facts` (observed
        live: fast models answer in a near-miss schema). Rejecting a good
        fact list over its key name wastes calls and quota, then cascades
        into rate limits — normalize instead. Genuinely malformed payloads
        (no list anywhere) still fail validation and retry as before."""
        if isinstance(data, list):
            return {"facts": data}
        if isinstance(data, dict):
            for key in ("facts", "claims", "results", "items"):
                value = data.get(key)
                if isinstance(value, list):
                    return {"facts": value}
            # A dict with NO recognized list key is genuinely malformed
            # (e.g. {"nope": []}) — fail validation rather than silently
            # treating it as "no facts found".
            return {"facts": None}
        return data


class CriticVerdictModel(BaseModel):
    is_sufficient: bool
    reason: str = ""
    improved_queries: List[str] = Field(default_factory=list)
    confidence: float = 0.0

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0


class SynthesizerAnswerModel(BaseModel):
    answer: str = Field(min_length=1)


class IntentSenseModel(BaseModel):
    """One candidate meaning of an ambiguous query term."""
    label: str = Field(min_length=1)
    domain: str = "general"
    probability: float = 0.5
    note: str = ""

    @field_validator("probability", mode="before")
    @classmethod
    def _clamp_probability(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5


class IntentOutputModel(BaseModel):
    """Intent Classification Agent output — the schema INTENT_SYSTEM_PROMPT
    documents and the parser reads (AGENTS.md 4.2)."""
    query_type: Literal["factual", "comparative", "analytical", "exploratory"] = "factual"
    domain: str = "general"
    explanation_level: Literal["basic", "practical", "expert"] = "practical"
    ambiguity: bool = False
    senses: List[IntentSenseModel] = Field(default_factory=list)
    reasoning: str = ""
