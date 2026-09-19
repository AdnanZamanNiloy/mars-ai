"""Query Router Agent — answer directly or go research (R1).

Why this module exists
----------------------
Every query in MARS currently enters the full research pipeline
(intent → plan → search → summarize → verify → critic → synthesize), even
when the question is a stable, well-known fact the model can answer from
general knowledge ("What is a for-loop?", "What is the capital of France?").
Spending retrieval, multiple LLM calls and minutes of latency on those is
pure waste.

The router is the pre-research decision the vision's Research Modes table
already promises ("Quick → Direct answer"): decide whether this question
needs EXTERNAL EVIDENCE at all, before any research is shaped.

Decision policy
---------------
`needs_external` is true when ANY of these holds:

  * freshness      — the answer changes over time (latest / current / 2026 /
                     a recency marker); model recall has a training cutoff.
  * quantitative   — numbers, prices, statistics, forecasts, market data.
  * decision       — "should we / recommend / invest" framing.
  * contested      — debate / risk / controversy vocabulary.
  * query_type     — comparative / analytical / exploratory need evidence.
  * ambiguity      — an ambiguous query must not be answered from one reading;
                     it needs the disambiguation machinery (or at least the
                     full report).
  * low confidence — the model itself signals it is unsure (LLM path only).

When none hold, the LLM is asked a single strict question: "can you answer
this confidently from general knowledge?" A DirectDecision is returned. Any
model answer that sets `needs_research=true`, or that reports low
self-confidence, routes to research.

Safety properties
-----------------
* Fail-safe direction is ALWAYS "research". A router LLM failure, invalid
  payload or disabled LLM returns the deterministic decision, and the
  deterministic decision never invents a "direct" verdict without the
  model's own clearance.
* R1 is intentionally UNWIRED: nothing calls this yet. The graph, the API
  and the frontend are untouched until R2/R3. This lets the decision logic
  be evaluated (bench + tests) before it can affect a live answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.agents.orchestrator import ComplexityScore, score_complexity
from app.core.degradation import record_fallback
from app.core.llm import LLMClient
from app.core.logging import get_logger

logger = get_logger(__name__)

DIRECT = "direct"
RESEARCH = "research"

# Self-confidence at/above which a model's own clearance is trusted for a
# direct answer. Below it, research — the model must be sure, not merely
# willing. Overridable via Settings.router_min_direct_confidence.
DEFAULT_MIN_DIRECT_CONFIDENCE = 0.75

# Query types that always need evidence. Only a plain factual question is
# even a candidate for a direct answer.
_RESEARCH_QUERY_TYPES = ("comparative", "analytical", "exploratory")

# A bare date mention (2024, 2026, ...) is a freshness signal even when the
# orchestrator's marker list does not carry that exact year.
_DATE_RE = re.compile(r"\b20[2-9]\d\b")

# Questions about the world as it currently is demand live retrieval. The
# orchestrator's RECENCY_MARKERS cover most of these; the extra terms below
# are the ones that appeared in the failures this feature exists to prevent.
_FRESHNESS_RE = re.compile(
    r"\b(latest|current|currently|recent|recently|today|now|newest|"
    r"up[\s-]?to[\s-]?date|breaking|live|ongoing|this (year|month|week)|"
    r"as of|right now|these days)\b",
    re.IGNORECASE,
)


ROUTER_SYSTEM_PROMPT = """
You are the Query Router in a multi-agent research system. Your ONLY job is
to decide whether a question requires searching the web for fresh,
verifiable evidence, or whether it can be answered correctly from your own
general knowledge.

━━━ ROUTE TO RESEARCH ("research") WHEN ANY OF THESE IS TRUE ━━━

1. FRESHNESS — the answer changes over time or depends on the present:
   "latest", "current", "today", "this year", "as of 2026", live prices,
   election results, recent events, ongoing situations. Your knowledge has
   a training cutoff; you cannot know today's facts.

2. NUMBERS / VERIFICATION — the user asks for statistics, prices, market
   data, forecasts, dates of recent events, or any figure that must be
   sourced to be trusted.

3. DECISIONS — the user asks what to do ("should we", "is it worth it",
   "recommend", "which is better for us"). Decisions require evidence,
   not just recall.

4. CONTESTED / HIGH-STAKES — health, safety, legal, financial, political
   or otherwise debated topics. These require corroboration and source
   attribution.

5. AMBIGUITY — the question could mean several different things and you
   are not certain which is intended.

6. YOUR OWN UNCERTAINTY — if you are not genuinely confident you can give
   a correct, complete answer from memory, route to research.

━━━ ROUTE DIRECT ("direct") ONLY WHEN ALL OF THESE ARE TRUE ━━━

- The answer is STABLE general knowledge that does not change over time
  (definitions, concepts, established facts, how something works).
- You can answer it correctly and completely from general knowledge.
- There is no legal, medical, financial or safety risk in answering
  without sources.
- The question is not ambiguous.

Examples that are DIRECT: "What is a Python list comprehension?",
"What is the capital of France?", "Explain how TCP handles packet loss."
Examples that are RESEARCH: "What is the latest price of gold?",
"How many electric vehicles were sold in 2025?", "Should we invest in
nuclear energy for our grid?", "Is this supplement safe?"

━━━ OUTPUT FORMAT ━━━

Return ONLY valid JSON. No markdown fences. No text outside JSON.

{
  "path": "<direct|research>",
  "confidence": <0.0-1.0, how sure you are this routing is correct>,
  "reason": "<one sentence naming the deciding factor>",
  "needs_research": <true|false>,
  "answer_sketch": "<only when path=direct: a concise, correct answer; otherwise empty>"
}
""".strip()


@dataclass
class RouteDecision:
    """The router's verdict: which pipeline path this query takes."""

    path: str                              # direct | research
    reason: str
    confidence: float
    signals: Dict[str, Any] = field(default_factory=dict)
    origin: str = "heuristic"              # llm | heuristic
    answer_sketch: str = ""

    @property
    def is_direct(self) -> bool:
        return self.path == DIRECT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "signals": dict(self.signals),
            "origin": self.origin,
            "answer_sketch": self.answer_sketch,
        }


def _collect_signals(
    query: str,
    complexity: ComplexityScore,
    intent: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Deterministic evidence about whether the query needs external data.

    Pure and total: no LLM, no I/O, no exceptions. `intent` is the
    IntentReport.to_dict() shape (or None when intent has not run).
    """
    q = str(query or "")
    lowered = q.lower()
    intent = intent or {}
    query_type = str(
        complexity.query_type or intent.get("query_type", "") or "factual"
    )
    ambiguity = bool(intent.get("ambiguity", False))
    freshness = bool(
        complexity.needs_recency
        or _FRESHNESS_RE.search(lowered)
        or _DATE_RE.search(lowered)
    )
    quantitative = bool(complexity.needs_quantitative)
    decision = bool(complexity.is_decision)
    contested = bool(complexity.is_contested)
    research_type = query_type in _RESEARCH_QUERY_TYPES
    hard_blockers: List[str] = []
    if freshness:
        hard_blockers.append("freshness")
    if quantitative:
        hard_blockers.append("quantitative")
    if decision:
        hard_blockers.append("decision")
    if contested:
        hard_blockers.append("contested")
    if research_type:
        hard_blockers.append("query_type")
    if ambiguity:
        hard_blockers.append("ambiguity")
    return {
        "query_type": query_type,
        "ambiguity": ambiguity,
        "freshness": freshness,
        "quantitative": quantitative,
        "decision": decision,
        "contested": contested,
        "research_query_type": research_type,
        "hard_blockers": hard_blockers,
    }


def deterministic_route(
    query: str,
    complexity: Optional[ComplexityScore] = None,
    intent: Optional[Dict[str, Any]] = None,
) -> RouteDecision:
    """The no-LLM decision, used as the fallback and as the hard gate.

    Any hard blocker routes to research. Otherwise the route is research too
    — WITHOUT the model's explicit clearance a direct answer is never
    assumed (fail-safe direction). This is why `deterministic_route` alone
    cannot produce `direct`: it can only block a direct answer, never grant
    one. `route_query` grants `direct` only when the LLM path clears it.
    """
    complexity = complexity if complexity is not None else score_complexity(query)
    signals = _collect_signals(query, complexity, intent)
    blockers = signals["hard_blockers"]
    if blockers:
        return RouteDecision(
            path=RESEARCH,
            reason=f"Query requires external evidence ({', '.join(blockers)}).",
            confidence=0.9,
            signals=signals,
            origin="heuristic",
        )
    return RouteDecision(
        path=RESEARCH,
        reason=(
            "No hard evidence requirement detected, but a direct answer "
            "requires explicit model clearance — defaulting to research."
        ),
        confidence=0.5,
        signals=signals,
        origin="heuristic",
    )


def _coerce_confidence(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _finalize(
    query: str,
    payload: Dict[str, Any],
    base: RouteDecision,
    confidence: float,
    min_direct_confidence: float = DEFAULT_MIN_DIRECT_CONFIDENCE,
) -> RouteDecision:
    """Validate the LLM's verdict against the deterministic hard gate.

    Even when the model says `direct`, any hard blocker from the
    deterministic signals forces research. The model may not override
    freshness/verification requirements; it may only clear an otherwise
    unremarkable question — and only when it is confident and explicitly
    says no research is needed.
    """
    signals = base.signals
    blockers = signals.get("hard_blockers") or []
    model_path = str(payload.get("path", "") or "").strip().lower()
    needs_research = bool(payload.get("needs_research", False))
    reason = str(payload.get("reason", "") or "").strip()

    if blockers:
        return RouteDecision(
            path=RESEARCH,
            reason=(
                reason
                or f"Externally-grounded answer required ({', '.join(blockers)})."
            ),
            confidence=max(confidence, 0.8),
            signals=signals,
            origin="llm",
        )

    if model_path != DIRECT or needs_research or confidence < min_direct_confidence:
        return RouteDecision(
            path=RESEARCH,
            reason=reason or "Model did not clear a direct answer.",
            confidence=confidence,
            signals=signals,
            origin="llm",
        )

    return RouteDecision(
        path=DIRECT,
        reason=reason or "Stable general knowledge; no external evidence required.",
        confidence=confidence,
        signals=signals,
        origin="llm",
        answer_sketch=str(payload.get("answer_sketch", "") or "").strip(),
    )


async def route_query(
    llm: LLMClient,
    query: str,
    complexity: Optional[ComplexityScore] = None,
    intent: Optional[Dict[str, Any]] = None,
) -> RouteDecision:
    """Decide whether `query` takes the direct or research path.

    Standard agent shape (AGENTS.md 4.7): the deterministic decision is
    computed first and is ALWAYS the fallback. The LLM may only refine it —
    and only in the direction of a direct answer, only when it is confident
    and explicitly declares no research is needed. Any failure yields the
    deterministic (research) decision.
    """
    complexity = complexity if complexity is not None else score_complexity(query)
    base = deterministic_route(query, complexity=complexity, intent=intent)

    # Optional external LLM clearance — import is module-level and settings
    # are read defensively so tests with a duck-typed LLM keep working.
    router_enabled = True
    min_direct_confidence = DEFAULT_MIN_DIRECT_CONFIDENCE
    settings = getattr(llm, "settings", None)
    if settings is not None:
        router_enabled = bool(getattr(settings, "router_enabled", True))
        try:
            min_direct_confidence = float(
                getattr(
                    settings,
                    "router_min_direct_confidence",
                    DEFAULT_MIN_DIRECT_CONFIDENCE,
                )
            )
        except (TypeError, ValueError):
            min_direct_confidence = DEFAULT_MIN_DIRECT_CONFIDENCE

    # A query with a hard blocker is research regardless of the model; there
    # is no reason to spend a call asking. This is both correct and cheap.
    if not router_enabled or base.signals.get("hard_blockers"):
        logger.info(
            "[Router] deterministic path=%s blockers=%s",
            base.path, base.signals.get("hard_blockers"),
        )
        return base

    user_prompt = (
        f"Question: {query}\n\n"
        "Decide whether this question requires web research "
        "(freshness, numbers needing sources, decisions, high-stakes or "
        "ambiguous topics, or your own uncertainty) or can be answered "
        "directly from stable general knowledge. Return JSON only."
    )
    try:
        payload = await llm.generate_json(ROUTER_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:
        logger.warning("[Router] LLM call failed, using deterministic fallback", exc_info=exc)
        record_fallback("router")
        return base

    if not isinstance(payload, dict) or not payload:
        logger.warning("[Router] empty/invalid LLM output, using fallback")
        record_fallback("router")
        return base

    confidence = _coerce_confidence(payload.get("confidence"), default=0.0)
    decision = _finalize(query, payload, base, confidence, min_direct_confidence)
    logger.info(
        "[Router] path=%s origin=%s confidence=%.2f reason=%s",
        decision.path, decision.origin, decision.confidence, decision.reason,
    )
    return decision
