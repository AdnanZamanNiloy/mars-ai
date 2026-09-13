"""Per-run usage ledger: the wire connecting LLM calls to the budget governor.

The LLMClient is app-scoped (one instance per process), but spend is per-run.
Threading a budget object through every agent signature would touch a dozen
call sites and the LangGraph state schema; instead the ledger rides a
ContextVar — the same pattern degradation flags already use — so the LLM
client, the search layer, the depth controller and the API routes can all
read the same per-run accounting with zero signature changes.

Starlette serves each request inside its own task context, so concurrent
research runs never see each other's ledgers.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Any, Dict, Optional

from app.agents.budget import ResearchBudget
from app.core.logging import get_logger

logger = get_logger(__name__)


class RunUsage:
    """One research run's spend: budget + cache counters + timing."""

    __slots__ = (
        "request_id", "budget", "mode", "started_at",
        "cache_hits", "cache_misses", "search_calls",
        "stage_hint",
    )

    def __init__(self, request_id: str, budget: ResearchBudget, mode: str = "standard"):
        self.request_id = request_id
        self.budget = budget
        self.mode = mode
        self.started_at = time.time()
        self.cache_hits = 0
        self.cache_misses = 0
        self.search_calls = 0
        self.stage_hint = ""

    # -- recording (called from llm.py / search.py) ------------------------

    def record_llm(
        self,
        stage: str,
        prompt: str = "",
        completion: str = "",
        *,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        model: Optional[str] = None,
        cached: bool = False,
    ) -> None:
        """Record one LLM call. Cached hits record tokens (they still count
        against free-tier daily quotas conceptually) but at zero cost —
        diskcache serves them without a provider round-trip. The budget does
        the zero-cost bookkeeping itself, so the run total is correct even if
        another record lands between this call and the next."""
        self.budget.record_llm(
            stage,
            prompt=prompt,
            completion=completion,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            cached=cached,
        )
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def record_search(self, provider: str, stage: str = "search", count: int = 1) -> None:
        self.search_calls += max(1, count)
        self.budget.record_search(provider, stage=stage, count=count)

    # -- decisions ----------------------------------------------------------

    @property
    def exhausted(self) -> bool:
        return self.budget.exhausted

    def can_afford_pass(self, sub_questions: int, avg_prompt_chars: int = 6000) -> bool:
        return self.budget.afford_pass(sub_questions, avg_prompt_chars)

    # -- reporting ----------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        snap = self.budget.snapshot()
        snap.update({
            "request_id": self.request_id,
            "mode": self.mode,
            "wall_sec": round(time.time() - self.started_at, 2),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                round(self.cache_hits / (self.cache_hits + self.cache_misses), 4)
                if (self.cache_hits + self.cache_misses) else 0.0
            ),
            "search_calls": self.search_calls,
        })
        return snap


_RUN_USAGE: ContextVar[Optional[RunUsage]] = ContextVar("mars_run_usage", default=None)


def start_run_usage(request_id: str, settings: Any, mode: str = "standard") -> RunUsage:
    """Install a fresh ledger for this run (call at request start)."""
    budget = ResearchBudget.from_settings(settings, mode=mode)
    usage = RunUsage(request_id=request_id, budget=budget, mode=mode)
    _RUN_USAGE.set(usage)
    return usage


def get_run_usage() -> Optional[RunUsage]:
    """The current run's ledger, or None outside a research request."""
    return _RUN_USAGE.get()


def clear_run_usage() -> None:
    _RUN_USAGE.set(None)


def run_seconds_remaining() -> float:
    """Wall-clock seconds left in the current research run's budget.

    Ladders consult this before spending a second provider attempt: a
    budget-aware retry is worth it, a retry that will be killed by the run
    timeout anyway is not. Outside a run (tests, scripts) the answer is
    'plenty' — the caller's own timeout still protects it.
    """
    usage = _RUN_USAGE.get()
    if usage is None:
        return 10_000.0
    return usage.budget.remaining_seconds


def set_stage_hint(stage: str) -> None:
    """Best-effort stage label so ledger entries attribute correctly
    (planner/summarizer/critic/synthesizer) even though the LLM client's
    public API doesn't carry a stage parameter."""
    usage = _RUN_USAGE.get()
    if usage is not None:
        usage.stage_hint = str(stage or "")
