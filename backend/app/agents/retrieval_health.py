"""Run-scoped retrieval health: domain cooldowns, failed-fetch memory, telemetry.

Why this exists
---------------
A 26-query live run measured 1,027 HTTP 403s, 893 HTTP 429s and 41 fetch
timeouts against the same handful of hosts (britannica.com 509, ssrn 104,
oecd.org 92, sciencedirect 71, crossref 29, wikipedia 19). Every one of those
was PAID FOR: each 403 was a full request that returned nothing, and because
nothing remembered the failure, the next query variant, the next expansion
pass and the next corroboration attempt re-hit the exact same blocked host and
re-paid for the same empty answer. The research budget went into re-learning
that a host was down instead of acquiring evidence.

Three primitives, all run-scoped and bounded:

  DomainCooldown   after repeated 403/429/timeout on a registrable domain,
                   that host is skipped for a cooldown window. Later passes
                   stop spending on a known-blocked publisher.
  FailedFetchLog   canonical URLs (and domains) that already failed this run
                   are never fetched twice. A 404'd link is not re-fetched by
                   the next pass's identically-ranked result.
  RetrievalHealth  counted outcomes (attempts, successes, 403/429/timeouts,
                   unique authoritative domains, primary acquisitions,
                   skipped-due-to-cooldown) so provider/host failures are
                   distinguishable from genuinely weak evidence.

This module is PURE and OFFLINE: no network, no LLM, no asyncio. The search
client owns the calls and the counters; everything here is a bounded registry.

Bounded by construction (AGENTS.md 4.3 / Section 5): the domain map has a hard
size cap with LRU eviction, the failed-URL set is capped the same way, and a
run that never resets cannot grow either. `reset()` is the explicit cleanup
path — `SearchClient` calls it at the start of every run.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from app.core.logging import get_logger

logger = get_logger(__name__)

# Default cooldown/window knobs. Overridden per-run from Settings; these exist
# so the dataclass is safe to construct in a unit test with no config.
DEFAULT_COOLDOWN_SEC = 90.0
DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_MAX_DOMAINS = 512
DEFAULT_MAX_FAILED_URLS = 2048


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------

# 403 is a hard wall: the host has decided this client may not read the page.
# Retrying it cannot succeed without changing identity (which this project
# will not do — no UA spoofing to defeat access control), so it is NOT
# transient. 429/timeout/5xx/connection errors can recover and are retried.
_HARD_BLOCK_STATUS = frozenset({401, 403, 451})
_TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 522, 524})


def classify_fetch_failure(status: Optional[int], exc: Optional[BaseException] = None) -> str:
    """Bucket a fetch outcome into the reason vocabulary the counters use.

    Returns one of: "ok", "forbidden", "rate_limited", "timeout",
    "server_error", "client_error", "connection", "other".
    """
    if status is None:
        if exc is None:
            return "ok"
        name = type(exc).__name__.lower()
        text = f"{name}: {exc}".lower()
        if "timeout" in name or "timed out" in text:
            return "timeout"
        if "connect" in name or "connection" in text or "disconnect" in text:
            return "connection"
        return "other"
    if 200 <= status < 300:
        return "ok"
    if status in _HARD_BLOCK_STATUS:
        return "forbidden"
    if status == 429:
        return "rate_limited"
    if status in _TRANSIENT_STATUS:
        return "server_error"
    if 400 <= status < 500:
        return "client_error"
    return "other"


def failure_is_transient(reason: str) -> bool:
    """Transient failures are worth a bounded retry; forbidden is not."""
    return reason in ("rate_limited", "timeout", "server_error", "connection")


def failure_cools_host(reason: str) -> bool:
    """Which failure kinds mark the HOST unavailable for a window.

    A 403 (forbidden) and repeated 429/timeout mean the host is not usable
    right now; cooling it down is what stops later passes re-paying for the
    same wall. A lone 404 (client_error) is URL-specific and must NOT cool the
    whole domain — the rest of the site may be perfectly readable.
    """
    return reason in ("forbidden", "rate_limited", "timeout", "connection", "server_error")


# ---------------------------------------------------------------------------
# Domain registry: cooldown + per-domain failure streak
# ---------------------------------------------------------------------------

@dataclass
class DomainState:
    failures: int = 0
    cooled_until: float = 0.0
    last_reason: str = ""

    def is_cooling(self, now: float) -> bool:
        return self.cooled_until > now


class DomainRegistry:
    """Bounded registrable-domain cooldown registry.

    A domain enters cooldown once it accumulates `failure_threshold` failures
    (or immediately on a hard 403). While cooling, `is_cooling()` is True and
    the search client skips it — so a blocked host costs one probe per window
    instead of one request per ranked result per pass.

    LRU-bounded: at most `max_domains` entries survive; the oldest-touched is
    evicted when the cap is crossed, so a long multi-run process cannot leak.
    """

    def __init__(
        self,
        *,
        cooldown_sec: float = DEFAULT_COOLDOWN_SEC,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        max_domains: int = DEFAULT_MAX_DOMAINS,
        clock=time.monotonic,
    ) -> None:
        self.cooldown_sec = max(0.0, float(cooldown_sec))
        self.failure_threshold = max(1, int(failure_threshold))
        self.max_domains = max(1, int(max_domains))
        self._clock = clock
        self._states: "OrderedDict[str, DomainState]" = OrderedDict()

    def _touch(self, domain: str) -> DomainState:
        state = self._states.get(domain)
        if state is None:
            state = DomainState()
            self._states[domain] = state
        else:
            self._states.move_to_end(domain)
        while len(self._states) > self.max_domains:
            self._states.popitem(last=False)
        return state

    def record_success(self, domain: str) -> None:
        if not domain:
            return
        state = self._states.get(domain)
        if state is not None and state.failures:
            state.failures = 0
            state.cooled_until = 0.0
            state.last_reason = ""

    def record_failure(self, domain: str, reason: str) -> None:
        """Count a host failure and cool the domain when warranted.

        A hard block (403/451) cools immediately: there is no streak to wait
        for, the host has already answered. Transient failures accumulate to
        the threshold, because one timeout is not evidence the host is down.
        """
        if not domain:
            return
        state = self._touch(domain)
        state.failures += 1
        state.last_reason = reason
        hard = reason == "forbidden"
        if hard or state.failures >= self.failure_threshold:
            state.cooled_until = self._clock() + self.cooldown_sec
            logger.info(
                "[retrieval] domain cooled down: %s (%s, failures=%d, %.0fs)",
                domain, reason, state.failures, self.cooldown_sec,
            )

    def is_cooling(self, domain: str) -> bool:
        if not domain:
            return False
        state = self._states.get(domain)
        return bool(state and state.is_cooling(self._clock()))

    def remaining(self, domain: str) -> float:
        state = self._states.get(domain)
        if state is None:
            return 0.0
        return max(0.0, state.cooled_until - self._clock())

    def cooling_domains(self) -> List[str]:
        now = self._clock()
        return [d for d, s in self._states.items() if s.is_cooling(now)]

    def reset(self) -> None:
        self._states.clear()


# ---------------------------------------------------------------------------
# Failed-fetch memory (canonical URLs + domains that already failed)
# ---------------------------------------------------------------------------

class FailedFetchLog:
    """Run-scoped memory of URLs/domains that already failed this run.

    Two different questions, two different answers:
      * "have we already tried this exact document?" -> URL-level memory,
        so a dead link is not re-fetched by the next pass.
      * "is this host unusable right now?" -> domain cooldown (above).

    Kept separate because a 404 on one URL must not blacklist the host, and a
    host that is merely cooling is not the same as a URL that is gone.

    Bounded with LRU eviction (insertion-order OrderedDict).
    """

    def __init__(self, max_urls: int = DEFAULT_MAX_FAILED_URLS) -> None:
        self.max_urls = max(1, int(max_urls))
        self._urls: "OrderedDict[str, str]" = OrderedDict()

    def mark(self, canonical: str, reason: str) -> None:
        key = (canonical or "").strip()
        if not key or key in self._urls:
            return
        self._urls[key] = reason
        while len(self._urls) > self.max_urls:
            self._urls.popitem(last=False)

    def seen(self, canonical: str) -> bool:
        return (canonical or "").strip() in self._urls

    def reason(self, canonical: str) -> str:
        return self._urls.get((canonical or "").strip(), "")

    def __len__(self) -> int:
        return len(self._urls)

    def reset(self) -> None:
        self._urls.clear()


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

@dataclass
class RetrievalHealth:
    """Counters that separate retrieval-access failure from weak evidence.

    A run that reports "3 verified claims" because every publisher returned
    403 is a RETRIEVAL failure, not a research result — and the two must never
    look the same in the artifact. Every field below is additive telemetry
    only: nothing here changes ranking, grading, corroboration or any gate.
    """

    fetch_attempts: int = 0
    fetch_successes: int = 0
    status_forbidden: int = 0
    status_rate_limited: int = 0
    fetch_timeouts: int = 0
    status_server_error: int = 0
    connection_errors: int = 0
    other_failures: int = 0
    retries_attempted: int = 0
    retries_succeeded: int = 0
    skipped_cooldown: int = 0
    skipped_duplicate_failure: int = 0
    cooldowns_opened: int = 0
    fallback_queries_issued: int = 0
    fallback_acquisitions: int = 0
    provider_attempts: int = 0
    provider_successes: int = 0
    _success_domains: Set[str] = field(default_factory=set)
    _primary_urls: Set[str] = field(default_factory=set)
    _authoritative_domains: Set[str] = field(default_factory=set)

    # -- recording ---------------------------------------------------------

    def record_attempt(self) -> None:
        self.fetch_attempts += 1

    def record_success(self, domain: str = "", primary: bool = False,
                       url: str = "", authoritative: bool = False) -> None:
        self.fetch_successes += 1
        if domain:
            self._success_domains.add(domain)
            if authoritative:
                self._authoritative_domains.add(domain)
        if primary and url:
            self._primary_urls.add(url)

    def record_failure(self, reason: str) -> None:
        if reason == "forbidden":
            self.status_forbidden += 1
        elif reason == "rate_limited":
            self.status_rate_limited += 1
        elif reason == "timeout":
            self.fetch_timeouts += 1
        elif reason == "server_error":
            self.status_server_error += 1
        elif reason == "connection":
            self.connection_errors += 1
        elif reason != "ok":
            self.other_failures += 1

    def record_retry(self, succeeded: bool) -> None:
        self.retries_attempted += 1
        if succeeded:
            self.retries_succeeded += 1

    def record_skip(self, reason: str) -> None:
        if reason == "cooldown":
            self.skipped_cooldown += 1
        elif reason == "duplicate_failure":
            self.skipped_duplicate_failure += 1

    def record_cooldown_opened(self) -> None:
        self.cooldowns_opened += 1

    def record_provider_call(self, succeeded: bool) -> None:
        self.provider_attempts += 1
        if succeeded:
            self.provider_successes += 1

    def record_fallback_query(self, acquisitions: int) -> None:
        self.fallback_queries_issued += 1
        if acquisitions > 0:
            self.fallback_acquisitions += 1

    # -- derived -----------------------------------------------------------

    def snapshot(self) -> Dict[str, object]:
        attempts = self.fetch_attempts
        return {
            "fetch_attempts": attempts,
            "fetch_successes": self.fetch_successes,
            "successful_fetch_rate": round(self.fetch_successes / attempts, 4) if attempts else 0.0,
            "status_forbidden": self.status_forbidden,
            "status_rate_limited": self.status_rate_limited,
            "fetch_timeouts": self.fetch_timeouts,
            "status_server_error": self.status_server_error,
            "connection_errors": self.connection_errors,
            "other_failures": self.other_failures,
            "forbidden_rate": round(self.status_forbidden / attempts, 4) if attempts else 0.0,
            "rate_limited_rate": round(self.status_rate_limited / attempts, 4) if attempts else 0.0,
            "timeout_rate": round(self.fetch_timeouts / attempts, 4) if attempts else 0.0,
            "retries_attempted": self.retries_attempted,
            "retries_succeeded": self.retries_succeeded,
            "skipped_cooldown": self.skipped_cooldown,
            "skipped_duplicate_failure": self.skipped_duplicate_failure,
            "cooldowns_opened": self.cooldowns_opened,
            "fallback_queries_issued": self.fallback_queries_issued,
            "fallback_acquisitions": self.fallback_acquisitions,
            "unique_success_domains": len(self._success_domains),
            "unique_authoritative_domains": len(self._authoritative_domains),
            "primary_source_acquisitions": len(self._primary_urls),
            "provider_attempts": self.provider_attempts,
            "provider_successes": self.provider_successes,
        }

    def reset(self) -> None:
        self.fetch_attempts = 0
        self.fetch_successes = 0
        self.status_forbidden = 0
        self.status_rate_limited = 0
        self.fetch_timeouts = 0
        self.status_server_error = 0
        self.connection_errors = 0
        self.other_failures = 0
        self.retries_attempted = 0
        self.retries_succeeded = 0
        self.skipped_cooldown = 0
        self.skipped_duplicate_failure = 0
        self.cooldowns_opened = 0
        self.fallback_queries_issued = 0
        self.fallback_acquisitions = 0
        self.provider_attempts = 0
        self.provider_successes = 0
        self._success_domains.clear()
        self._primary_urls.clear()
        self._authoritative_domains.clear()


__all__ = [
    "DEFAULT_COOLDOWN_SEC",
    "DEFAULT_FAILURE_THRESHOLD",
    "DEFAULT_MAX_DOMAINS",
    "DEFAULT_MAX_FAILED_URLS",
    "DomainState",
    "DomainRegistry",
    "FailedFetchLog",
    "RetrievalHealth",
    "classify_fetch_failure",
    "failure_cools_host",
    "failure_is_transient",
]
