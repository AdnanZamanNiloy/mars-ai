"""Pydantic response models mirroring each agent's expected LLM output schema.

Phase 1.2: passed as `response_model` to LLMClient.generate_json so malformed
LLM output is rejected and retried instead of silently accepted.

Schema rule (AGENTS.md 4.2): every field here must match the keys the
corresponding agent's system prompt documents AND the keys its parsing
code reads. Keep the three in sync.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


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
