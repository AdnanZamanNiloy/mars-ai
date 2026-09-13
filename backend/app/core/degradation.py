"""Degradation tracking: which agents fell back to deterministic defaults.

The pipeline's core promise is "never crash" — but a silently degraded run
looks exactly like a mediocre one. Agents call record_fallback() on every
deterministic-fallback path so the stream can report WHICH parts of an
answer came from fallbacks instead of the model.

Scoped per request via ContextVar: reset at stream start and drained at
final_report. Outside a tracked request, record_fallback() is a no-op — unit
tests and scripts that never reset simply record nothing.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Dict, List, Optional

_fallbacks: ContextVar[Optional[List[str]]] = ContextVar("degraded_agents", default=None)
# Why each agent fell back (agent -> short reason code). Bounded by agent
# count and set once per agent; reset/cleared alongside _fallbacks.
_fallback_reasons: ContextVar[Optional[Dict[str, str]]] = ContextVar(
    "degraded_reasons", default=None
)


def reset_fallbacks() -> None:
    _fallbacks.set([])
    _fallback_reasons.set({})


def clear_fallbacks() -> None:
    _fallbacks.set(None)
    _fallback_reasons.set(None)


def record_fallback(agent: str, reason: str = "") -> None:
    current = _fallbacks.get()
    if current is None:
        return
    if agent not in current:
        current.append(agent)
    reasons = _fallback_reasons.get()
    if reasons is not None and reason and agent not in reasons:
        reasons[agent] = str(reason)


def take_fallbacks() -> List[str]:
    return list(_fallbacks.get() or [])


def fallback_reasons() -> Dict[str, str]:
    """Per-agent reason codes for why each fallback fired (may be empty for
    older call sites that pass no reason)."""
    return dict(_fallback_reasons.get() or {})
