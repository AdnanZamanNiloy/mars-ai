"""Degradation tracking: which agents fell back to deterministic defaults.

The pipeline's core promise is "never crash" — but a silently degraded run
looks exactly like a mediocre one. Agents call record_fallback() on every
deterministic-fallback path so the stream can report WHICH parts of an
answer came from fallbacks instead of the model.

Scoped per request via ContextVar (same pattern as the budget tracker):
routes reset at stream start and drain at final_report. Outside a tracked
request, record_fallback() is a no-op — unit tests and scripts that never
reset simply record nothing.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import List, Optional

_fallbacks: ContextVar[Optional[List[str]]] = ContextVar("degraded_agents", default=None)


def reset_fallbacks() -> None:
    _fallbacks.set([])


def clear_fallbacks() -> None:
    _fallbacks.set(None)


def record_fallback(agent: str) -> None:
    current = _fallbacks.get()
    if current is None:
        return
    if agent not in current:
        current.append(agent)


def take_fallbacks() -> List[str]:
    return list(_fallbacks.get() or [])
