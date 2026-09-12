"""Cost & token governor (vision Feature 12).

The pipeline's cost is dominated by LLM calls whose count is a *product*, not
a sum: sub_questions x passes x agents. Raising the plan from 3 to 5
sub-questions and the cap from 3 to 5 passes multiplies spend by ~2.8x, and
nothing in the previous code could observe that, let alone stop it. This module
makes every spend decision explicit and refusable.

Design constraints that shaped it:

* Prices change and vary per provider, so the table is a default, overridable
  from settings (`llm_price_per_1k_input` / `..._output` or `model_prices`).
* An estimate before a call is worth more than an exact number after it: the
  governor answers "can I afford another pass?" *before* spending, using a
  reserve/commit protocol.
* Free tiers are rate-limited, not priced. The governor therefore tracks
  tokens and calls independently of dollars, so a $0 budget still bounds work.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.core.logging import get_logger

logger = get_logger(__name__)

# USD per 1K tokens (input, output). Rough public list prices; the point is the
# ratio and order of magnitude, not four-decimal accuracy.
DEFAULT_MODEL_PRICES: Dict[str, Tuple[float, float]] = {
    "llama-3.1-8b-instant": (0.00005, 0.00008),
    "llama-3.3-70b-versatile": (0.00059, 0.00079),
    "llama-3.1-70b-versatile": (0.00059, 0.00079),
    "mixtral-8x7b-32768": (0.00024, 0.00024),
    "gemma2-9b-it": (0.0002, 0.0002),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
    "claude-3-5-haiku": (0.0008, 0.004),
    "claude-3-5-sonnet": (0.003, 0.015),
    "default": (0.0004, 0.0008),
}

# Non-LLM unit costs. Tavily's free tier is credit-metered rather than billed,
# but a credit is a scarce resource and belongs in the same budget.
SEARCH_UNIT_COST: Dict[str, float] = {
    "tavily": 0.008,
    "ddg": 0.0,
    "wikipedia": 0.0,
    "arxiv": 0.0,
    "crossref": 0.0,
    "fetch": 0.0,
}

# Average characters per token for English prose. Used only for estimates;
# actual usage is recorded from provider responses when available.
CHARS_PER_TOKEN = 3.9


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return int(math.ceil(len(text) / CHARS_PER_TOKEN))


class BudgetExceeded(RuntimeError):
    """Raised only when a caller opts into strict enforcement."""


@dataclass
class SpendRecord:
    stage: str
    kind: str            # "llm" | "search" | "fetch"
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "kind": self.kind,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "at": self.at,
        }


@dataclass
class ResearchBudget:
    """Per-mission budget across dollars, tokens, calls and wall-clock.

    Four independent ceilings because they bind in different situations: paid
    APIs bind on dollars, free tiers on tokens/day and calls/minute, and an
    interactive UI binds on latency. A run stops at whichever hits first.
    """

    max_usd: float = 0.50
    max_tokens: int = 400_000
    max_llm_calls: int = 60
    max_seconds: float = 300.0
    model: str = "default"
    prices: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_MODEL_PRICES))
    strict: bool = False

    spent_usd: float = field(default=0.0, init=False)
    spent_tokens: int = field(default=0, init=False)
    llm_calls: int = field(default=0, init=False)
    search_calls: int = field(default=0, init=False)
    reserved_usd: float = field(default=0.0, init=False)
    started_at: float = field(default_factory=time.monotonic, init=False)
    records: List[SpendRecord] = field(default_factory=list, init=False)
    refusals: List[str] = field(default_factory=list, init=False)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_settings(cls, settings: Any, mode: str = "standard") -> "ResearchBudget":
        """Build from app settings, with per-mode multipliers.

        Every field is read with getattr defaults, so this works against a
        Settings object that has never heard of budgets.
        """
        multiplier = MODE_BUDGET_MULTIPLIER.get(str(mode or "standard").lower(), 1.0)
        prices = dict(DEFAULT_MODEL_PRICES)
        custom = getattr(settings, "model_prices", None)
        if isinstance(custom, dict):
            for key, value in custom.items():
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    prices[str(key)] = (float(value[0]), float(value[1]))
        model = str(
            getattr(settings, "llm_model", None)
            or getattr(settings, "model", None)
            or "default"
        )
        return cls(
            max_usd=float(getattr(settings, "max_budget_usd", 0.50) or 0.50) * multiplier,
            max_tokens=int(float(getattr(settings, "max_budget_tokens", 400_000) or 400_000) * multiplier),
            max_llm_calls=int(float(getattr(settings, "max_llm_calls", 60) or 60) * multiplier),
            max_seconds=float(getattr(settings, "max_research_seconds", 300.0) or 300.0) * multiplier,
            model=model,
            prices=prices,
            strict=bool(getattr(settings, "strict_budget", False)),
        )

    # -- pricing -----------------------------------------------------------

    def price_for(self, model: Optional[str] = None) -> Tuple[float, float]:
        name = str(model or self.model or "default")
        if name in self.prices:
            return self.prices[name]
        for key, value in self.prices.items():
            if key != "default" and key in name:
                return value
        return self.prices.get("default", DEFAULT_MODEL_PRICES["default"])

    def cost_of(self, input_tokens: int, output_tokens: int, model: Optional[str] = None) -> float:
        pin, pout = self.price_for(model)
        return (max(0, input_tokens) / 1000.0) * pin + (max(0, output_tokens) / 1000.0) * pout

    # -- state -------------------------------------------------------------

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.max_usd - self.spent_usd - self.reserved_usd)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.spent_tokens)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed)

    @property
    def exhausted(self) -> bool:
        return (
            self.remaining_usd <= 0.0
            or self.remaining_tokens <= 0
            or self.llm_calls >= self.max_llm_calls
            or self.remaining_seconds <= 0.0
        )

    def utilization(self) -> float:
        """Worst-case fraction consumed across all four ceilings (0-1)."""
        ratios = [
            self.spent_usd / self.max_usd if self.max_usd > 0 else 0.0,
            self.spent_tokens / self.max_tokens if self.max_tokens > 0 else 0.0,
            self.llm_calls / self.max_llm_calls if self.max_llm_calls > 0 else 0.0,
            self.elapsed / self.max_seconds if self.max_seconds > 0 else 0.0,
        ]
        return round(min(1.0, max(ratios)), 4)

    # -- decisions ---------------------------------------------------------

    def can_afford(
        self,
        *,
        stage: str = "llm",
        est_input_tokens: int = 0,
        est_output_tokens: int = 0,
        calls: int = 1,
        model: Optional[str] = None,
    ) -> bool:
        """Would this operation fit? Never raises — callers decide what to do."""
        if self.remaining_seconds <= 0.0:
            self._refuse(stage, "time budget exhausted")
            return False
        if self.llm_calls + calls > self.max_llm_calls:
            self._refuse(stage, f"call ceiling {self.max_llm_calls} reached")
            return False
        if self.spent_tokens + est_input_tokens + est_output_tokens > self.max_tokens:
            self._refuse(stage, f"token ceiling {self.max_tokens} would be exceeded")
            return False
        cost = self.cost_of(est_input_tokens, est_output_tokens, model)
        if cost > self.remaining_usd:
            self._refuse(stage, f"cost ${cost:.4f} exceeds remaining ${self.remaining_usd:.4f}")
            return False
        return True

    def afford_pass(self, sub_questions: int, avg_prompt_chars: int = 6000) -> bool:
        """Can a whole additional research pass be paid for?

        This is the decision that actually protects the budget: one refused
        pass saves N summarizer calls plus a critic call, where the previous
        code could only refuse (or not) one call at a time and always
        discovered the problem after spending.
        """
        est_in = estimate_tokens("x" * avg_prompt_chars) * max(1, sub_questions)
        est_out = int(est_in * 0.25)
        return self.can_afford(
            stage="pass",
            est_input_tokens=est_in,
            est_output_tokens=est_out,
            calls=max(1, sub_questions) + 1,
        )

    def _refuse(self, stage: str, reason: str) -> None:
        message = f"{stage}: {reason}"
        if message not in self.refusals:
            self.refusals.append(message)
        logger.info("[budget] refused %s", message)

    # -- accounting --------------------------------------------------------

    def reserve(self, usd: float) -> None:
        self.reserved_usd += max(0.0, usd)

    def release(self, usd: float) -> None:
        self.reserved_usd = max(0.0, self.reserved_usd - max(0.0, usd))

    def record_llm(
        self,
        stage: str,
        prompt: str = "",
        completion: str = "",
        *,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> SpendRecord:
        """Record one LLM call. Prefers provider-reported token counts and
        falls back to a character estimate."""
        tin = int(input_tokens) if input_tokens is not None else estimate_tokens(prompt)
        tout = int(output_tokens) if output_tokens is not None else estimate_tokens(completion)
        cost = self.cost_of(tin, tout, model)
        record = SpendRecord(
            stage=stage, kind="llm", model=str(model or self.model),
            input_tokens=tin, output_tokens=tout, cost_usd=cost,
        )
        self.records.append(record)
        self.spent_usd += cost
        self.spent_tokens += tin + tout
        self.llm_calls += 1
        if self.strict and self.spent_usd > self.max_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.4f} exceeds budget ${self.max_usd:.4f}"
            )
        return record

    def record_search(self, provider: str, stage: str = "search", count: int = 1) -> SpendRecord:
        unit = SEARCH_UNIT_COST.get(str(provider).lower(), 0.0)
        cost = unit * max(1, count)
        record = SpendRecord(
            stage=stage, kind="search", model=str(provider),
            input_tokens=0, output_tokens=0, cost_usd=cost,
        )
        self.records.append(record)
        self.spent_usd += cost
        self.search_calls += max(1, count)
        return record

    # -- reporting ---------------------------------------------------------

    def by_stage(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for record in self.records:
            bucket = out.setdefault(
                record.stage, {"calls": 0, "tokens": 0, "cost_usd": 0.0}
            )
            bucket["calls"] += 1
            bucket["tokens"] += record.input_tokens + record.output_tokens
            bucket["cost_usd"] = round(bucket["cost_usd"] + record.cost_usd, 6)
        return out

    def snapshot(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "spent_usd": round(self.spent_usd, 6),
            "max_usd": round(self.max_usd, 6),
            "remaining_usd": round(self.remaining_usd, 6),
            "spent_tokens": self.spent_tokens,
            "max_tokens": self.max_tokens,
            "llm_calls": self.llm_calls,
            "max_llm_calls": self.max_llm_calls,
            "search_calls": self.search_calls,
            "elapsed_sec": round(self.elapsed, 2),
            "max_seconds": self.max_seconds,
            "utilization": self.utilization(),
            "exhausted": self.exhausted,
            "by_stage": self.by_stage(),
            "refusals": list(self.refusals),
        }


# Deep modes are allowed to cost more — that depth is the product — but the
# multiplier is explicit and bounded rather than emergent from a loop count.
MODE_BUDGET_MULTIPLIER: Dict[str, float] = {
    "quick": 0.35,
    "standard": 1.0,
    "audit": 1.2,
    "redteam": 1.2,
    "executive": 2.0,
    "deep": 2.5,
}
