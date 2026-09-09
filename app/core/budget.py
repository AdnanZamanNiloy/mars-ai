"""Cost / Token Governor (Phase 2.2).

Tracks approximate token usage and dollar cost per research run, stopping
the critic loop early when the run exceeds RESEARCH_MAX_COST_USD.

Token accounting prefers the provider's real `usage` block (Groq supplies
prompt_tokens/completion_tokens) and falls back to a chars/4 estimate.
Pricing comes from Settings — real provider rates, never hardcoded here.
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Dict, Optional

from app.core.config import Settings

logger = logging.getLogger(__name__)

# Per-run budget lives in a ContextVar so concurrent requests each track
# their own usage without global mutable state (AGENTS.md 4.3).
current_budget: ContextVar[Optional["BudgetTracker"]] = ContextVar("current_budget", default=None)

CHARS_PER_TOKEN_ESTIMATE = 4


def get_current_budget() -> Optional["BudgetTracker"]:
    return current_budget.get()


class BudgetTracker:
    """Accumulates per-run token usage and estimated cost."""

    def __init__(self, settings: Settings, limit_usd: Optional[float] = None):
        self.settings = settings
        self.limit_usd = float(limit_usd if limit_usd is not None else settings.research_max_cost_usd)
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.estimated_cost_usd: float = 0.0
        self.llm_calls: int = 0
        self.over_budget: bool = False

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_llm_call(
        self,
        model: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        request_chars: int = 0,
        response_chars: int = 0,
    ) -> None:
        """Record one LLM call. Real token counts win over estimates."""
        if input_tokens is None:
            input_tokens = request_chars // CHARS_PER_TOKEN_ESTIMATE
        if output_tokens is None:
            output_tokens = response_chars // CHARS_PER_TOKEN_ESTIMATE

        self.input_tokens += max(0, int(input_tokens))
        self.output_tokens += max(0, int(output_tokens))
        self.llm_calls += 1

        cost = self._cost_for(model, int(input_tokens), int(output_tokens))
        self.estimated_cost_usd += cost

        if not self.over_budget and self.estimated_cost_usd >= self.limit_usd:
            self.over_budget = True
            logger.warning(
                "[Budget] cost $%.4f exceeded limit $%.4f after %d LLM calls — pipeline will stop expanding",
                self.estimated_cost_usd,
                self.limit_usd,
                self.llm_calls,
            )

    def _cost_for(self, model: str, input_tokens: int, output_tokens: int) -> float:
        if model and "groq" in model.lower():
            in_rate = self.settings.groq_cost_per_1k_input_tokens
            out_rate = self.settings.groq_cost_per_1k_output_tokens
        else:
            in_rate = self.settings.hf_cost_per_1k_tokens
            out_rate = in_rate
        return (input_tokens * in_rate + output_tokens * out_rate) / 1000.0

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def snapshot(self) -> Dict[str, Any]:
        return {
            "estimated_cost": round(self.estimated_cost_usd, 4),
            "limit": self.limit_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "llm_calls": self.llm_calls,
            "over_budget": self.over_budget,
        }

    def limitation_note(self) -> str:
        return (
            f"Research stopped expanding early: estimated LLM cost "
            f"${self.estimated_cost_usd:.4f} reached the ${self.limit_usd:.2f} budget "
            f"after {self.llm_calls} model calls."
        )
