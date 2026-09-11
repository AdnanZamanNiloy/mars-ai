"""Pydantic response models mirroring each agent's expected LLM output schema.

Phase 1.2: passed as `response_model` to LLMClient.generate_json so malformed
LLM output is rejected and retried instead of silently accepted.

Schema rule (AGENTS.md 4.2): every field here must match the keys the
corresponding agent's system prompt documents AND the keys its parsing
code reads. Keep the three in sync.
"""
from typing import Any, List, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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
    variants: List[str] = Field(default_factory=list, max_length=2)
    agent: str = ""
    tools: List[str] = Field(default_factory=lambda: ["web_search"])
    scope: List[str] = Field(default_factory=list, max_length=5)
    output_format: str = "structured_findings"


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

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0


class SummarizerFactsModel(BaseModel):
    facts: List[FactModel] = Field(min_length=1)

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
