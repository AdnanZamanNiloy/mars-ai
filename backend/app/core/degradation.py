"""Degradation tracking: which agents fell back to deterministic defaults.

The pipeline's core promise is "never crash" — but a silently degraded run
looks exactly like a mediocre one. Agents call record_fallback() on every
deterministic-fallback path so the stream can report WHICH parts of an
answer came from fallbacks instead of the model.

Scoped per request via ContextVar: reset at stream start and drained at
final_report. Outside a tracked request, record_fallback() is a no-op — unit
tests and scripts that never reset simply record nothing.

Two DIFFERENT failure classes are tracked separately and must never be
conflated (reliability requirement #4):

  * evidence weakness   — the provider answered, but the fact pool is thin
                          (few facts, no verification, uncorroborated).
                          Recorded per-agent via record_fallback(reason=...).
  * provider degradation — the provider failed (transient or hard), so the
                          run fell back to deterministic extraction for
                          transport reasons, not evidence reasons. Recorded
                          via record_provider_failure(); a transient provider
                          failure must NOT be reported as weak evidence.

Reason codes for both live here so the stream/report can state WHY a run is
degraded, not merely that it is.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Dict, List, Optional

# Provider-failure classifications (classification is done in app.core.llm so
# the HTTP-status knowledge lives in one place).
PROVIDER_TRANSIENT = "provider-transient"
PROVIDER_HARD = "provider-hard"
EVIDENCE_WEAK = "weak-evidence"

_fallbacks: ContextVar[Optional[List[str]]] = ContextVar("degraded_agents", default=None)
# Why each agent fell back (agent -> short reason code). Bounded by agent
# count and set once per agent; reset/cleared alongside _fallbacks.
_fallback_reasons: ContextVar[Optional[Dict[str, str]]] = ContextVar(
    "degraded_reasons", default=None
)
# Provider-level failures observed during the request (bounded list of
# {kind, detail}). Distinct from per-agent fallback reasons: a provider
# failure can degrade several stages and must be reported as a transport
# cause, never as thin evidence.
_provider_failures: ContextVar[Optional[List[Dict[str, str]]]] = ContextVar(
    "provider_failures", default=None
)


def reset_fallbacks() -> None:
    _fallbacks.set([])
    _fallback_reasons.set({})
    _provider_failures.set([])


def clear_fallbacks() -> None:
    _fallbacks.set(None)
    _fallback_reasons.set(None)
    _provider_failures.set(None)


def record_fallback(agent: str, reason: str = "") -> None:
    current = _fallbacks.get()
    if current is None:
        return
    if agent not in current:
        current.append(agent)
    reasons = _fallback_reasons.get()
    if reasons is not None and reason and agent not in reasons:
        reasons[agent] = str(reason)


def record_provider_failure(kind: str, detail: str = "") -> None:
    """Record that an LLM provider failed for TRANSPORT reasons.

    `kind` is one of PROVIDER_TRANSIENT / PROVIDER_HARD. Bounded: identical
    (kind, detail) pairs collapse, and the list caps at 20 entries so a burst
    of failures cannot grow state without limit (AGENTS.md §4.3)."""
    failures = _provider_failures.get()
    if failures is None:
        return
    entry = {"kind": str(kind or ""), "detail": str(detail or "")[:200]}
    if entry in failures:
        return
    if len(failures) >= 20:
        return
    failures.append(entry)


def take_fallbacks() -> List[str]:
    return list(_fallbacks.get() or [])


def fallback_reasons() -> Dict[str, str]:
    """Per-agent reason codes for why each fallback fired (may be empty for
    older call sites that pass no reason)."""
    return dict(_fallback_reasons.get() or {})


def provider_failures() -> List[Dict[str, str]]:
    """Provider-level failures observed this request ([] when none)."""
    return [dict(f) for f in (_provider_failures.get() or [])]


def provider_failure_kinds() -> List[str]:
    """Distinct provider-failure classifications, order-preserved."""
    kinds: List[str] = []
    for f in _provider_failures.get() or []:
        kind = str(f.get("kind", "") or "")
        if kind and kind not in kinds:
            kinds.append(kind)
    return kinds


def has_provider_degradation() -> bool:
    """True when any provider failed for transport reasons this request."""
    return bool(_provider_failures.get())


def degradation_summary() -> Dict[str, Any]:
    """Structured, additive view of WHY a run is degraded.

    `reasons` maps each degraded agent to its short reason code. `provider`
    names the distinct provider-failure classes so a transient outage is
    never silently read as weak evidence. All keys always present."""
    agents = take_fallbacks()
    reasons = fallback_reasons()
    provider_kinds = provider_failure_kinds()
    provider_agents = sorted(
        a for a, r in reasons.items()
        if r in (PROVIDER_TRANSIENT, PROVIDER_HARD) or str(r).startswith("provider_")
    )
    provider_reason_agents = {
        a for a, r in reasons.items()
        if r in (PROVIDER_TRANSIENT, PROVIDER_HARD) or str(r).startswith("provider_")
    }
    return {
        "agents": agents,
        "reasons": reasons,
        "provider_failures": provider_failures(),
        "provider_degraded": bool(provider_kinds),
        "provider_kinds": provider_kinds,
        "provider_agents": provider_agents,
        # Degraded agents whose reason is NOT provider-transport: evidence
        # weakness, parse failure, etc. A genuine evidence weakness must stay
        # distinguishable from a provider outage.
        "evidence_agents": sorted(
            a for a, r in reasons.items() if r and a not in provider_reason_agents
        ),
    }
