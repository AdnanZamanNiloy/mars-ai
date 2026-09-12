"""Fault tolerance primitives: retries, circuit breakers, bulkheads, timeouts.

The pipeline's failure modes are all remote-call failure modes: a provider
rate-limits, a page hangs, a JSON body comes back truncated. Before this
module, each call site handled that with a bare `try/except -> return []`,
which turns a 200ms transient 429 into permanently missing evidence and hides
the outage from every downstream quality gate.

Three things live here, in dependency order:

  retry_async     bounded retries with exponential backoff + full jitter,
                  restricted to errors that can actually succeed on retry
  CircuitBreaker  stops calling a provider that is definitively down, so a
                  dead provider costs one timeout per cooldown instead of
                  one timeout per sub-question per pass
  Bulkhead        named concurrency ceilings, so one slow stage cannot consume
                  every slot on an 8GB host

No third-party dependencies, no global state that survives a process, and
every knob has a default that is safe on a laptop.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional, Sequence, Tuple, TypeVar

from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Retry classification
# ---------------------------------------------------------------------------

# Retrying a 400 ("your query is malformed") burns budget and latency for a
# guaranteed second failure. Only transient shapes are retried; everything
# else fails fast to the caller's fallback.
_TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 522, 524})

_TRANSIENT_MESSAGE_MARKERS: Tuple[str, ...] = (
    "timeout", "timed out", "temporarily unavailable", "rate limit",
    "too many requests", "connection reset", "connection aborted",
    "connection error", "server disconnected", "remote protocol error",
    "read error", "eof occurred", "overloaded", "capacity", "try again",
    "service unavailable", "bad gateway", "gateway timeout", "503", "429",
)


def _status_code(exc: BaseException) -> Optional[int]:
    """Pull an HTTP status off an exception without importing httpx."""
    response = getattr(exc, "response", None)
    for holder in (response, exc):
        code = getattr(holder, "status_code", None)
        if isinstance(code, int):
            return code
        code = getattr(holder, "status", None)
        if isinstance(code, int):
            return code
    return None


def is_transient(exc: BaseException) -> bool:
    """True when retrying `exc` has a realistic chance of succeeding."""
    if isinstance(exc, asyncio.CancelledError):
        return False
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return True
    code = _status_code(exc)
    if code is not None:
        return code in _TRANSIENT_STATUS
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MESSAGE_MARKERS)


def retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Honour a provider's own Retry-After when it sends one. Guessing a
    backoff while the provider is telling you the answer is how a 429 storm
    turns into a ban."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:  # pragma: no cover - exotic header containers
        return None
    if raw is None:
        return None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return max(0.0, min(60.0, value))


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------

@dataclass
class RetryPolicy:
    attempts: int = 3
    base_delay: float = 0.4
    max_delay: float = 8.0
    timeout: Optional[float] = None   # per-attempt, not total
    jitter: bool = True

    def delay_for(self, attempt: int) -> float:
        """Full-jitter exponential backoff (attempt is 1-based).

        Full jitter rather than fixed backoff: N sub-questions failing on the
        same provider at the same moment would otherwise retry in lockstep and
        re-trigger the same rate limit.
        """
        raw = min(self.max_delay, self.base_delay * (2 ** max(0, attempt - 1)))
        return random.uniform(0.0, raw) if self.jitter else raw


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    policy: Optional[RetryPolicy] = None,
    label: str = "call",
    retry_on: Callable[[BaseException], bool] = is_transient,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> T:
    """Await `fn()` with bounded retries. Re-raises the last error on failure.

    `fn` is a zero-arg factory, not a coroutine object, because a coroutine
    can only be awaited once — passing one in makes retrying impossible, a
    mistake that is easy to make and silently degrades to no retries at all.
    """
    policy = policy or RetryPolicy()
    attempts = max(1, int(policy.attempts))
    last: BaseException = RuntimeError(f"{label}: no attempt was made")

    for attempt in range(1, attempts + 1):
        try:
            if policy.timeout:
                return await asyncio.wait_for(fn(), timeout=policy.timeout)
            return await fn()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            last = exc
            if attempt >= attempts or not retry_on(exc):
                break
            delay = retry_after_seconds(exc)
            if delay is None:
                delay = policy.delay_for(attempt)
            logger.warning(
                "[reliability] %s attempt %d/%d failed (%s); retrying in %.2fs",
                label, attempt, attempts, type(exc).__name__, delay,
            )
            if on_retry:
                try:
                    on_retry(attempt, exc, delay)
                except Exception:  # pragma: no cover - observer must not break flow
                    pass
            await asyncio.sleep(delay)

    raise last


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class CircuitOpen(RuntimeError):
    """Raised instead of calling a provider that is known to be down."""

    def __init__(self, name: str, retry_in: float) -> None:
        super().__init__(f"circuit '{name}' is open; retry in {retry_in:.1f}s")
        self.name = name
        self.retry_in = retry_in


@dataclass
class CircuitBreaker:
    """Per-provider breaker with closed -> open -> half-open states.

    Why a research pipeline needs one: a sub-question fan-out calls the same
    provider once per contract per pass. When that provider is hard-down, the
    old code paid the full timeout on every one of those calls, every pass —
    tens of wasted seconds per run and a completely empty evidence pool. The
    breaker converts that into one timeout, then instant failures that fall
    straight through to the alternate provider.
    """

    name: str
    failure_threshold: int = 4
    cooldown: float = 30.0
    half_open_successes: int = 1

    _failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _state: str = field(default="closed", init=False)
    _probe_successes: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @property
    def state(self) -> str:
        if self._state == "open" and (time.monotonic() - self._opened_at) >= self.cooldown:
            return "half_open"
        return self._state

    def retry_in(self) -> float:
        if self._state != "open":
            return 0.0
        return max(0.0, self.cooldown - (time.monotonic() - self._opened_at))

    async def guard(self) -> None:
        async with self._lock:
            if self._state == "open":
                if (time.monotonic() - self._opened_at) < self.cooldown:
                    raise CircuitOpen(self.name, self.retry_in())
                self._state = "half_open"
                self._probe_successes = 0
                logger.info("[reliability] circuit '%s' half-open, probing", self.name)

    async def record_success(self) -> None:
        async with self._lock:
            if self._state == "half_open":
                self._probe_successes += 1
                if self._probe_successes >= max(1, self.half_open_successes):
                    self._state = "closed"
                    self._failures = 0
                    logger.info("[reliability] circuit '%s' closed", self.name)
                return
            self._failures = 0

    async def record_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            if self._state == "half_open" or self._failures >= max(1, self.failure_threshold):
                self._state = "open"
                self._opened_at = time.monotonic()
                logger.warning(
                    "[reliability] circuit '%s' OPEN after %d failures; cooling down %.0fs",
                    self.name, self._failures, self.cooldown,
                )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "failures": self._failures,
            "retry_in": round(self.retry_in(), 2),
        }


_BREAKERS: Dict[str, CircuitBreaker] = {}


def get_breaker(
    name: str, failure_threshold: int = 4, cooldown: float = 30.0
) -> CircuitBreaker:
    breaker = _BREAKERS.get(name)
    if breaker is None:
        breaker = CircuitBreaker(
            name=name, failure_threshold=failure_threshold, cooldown=cooldown
        )
        _BREAKERS[name] = breaker
    return breaker


def breaker_snapshot() -> Dict[str, Dict[str, Any]]:
    return {name: b.snapshot() for name, b in _BREAKERS.items()}


def reset_breakers() -> None:
    """Test hook: breakers are process-global by design (a provider is down
    for the whole process, not per request)."""
    _BREAKERS.clear()


async def call_protected(
    fn: Callable[[], Awaitable[T]],
    *,
    name: str,
    policy: Optional[RetryPolicy] = None,
    breaker: Optional[CircuitBreaker] = None,
    fallback: Optional[Callable[[], Awaitable[T]]] = None,
) -> T:
    """Retry + breaker + optional fallback around one remote dependency.

    The composition order matters: the breaker is checked once per logical
    call (not per retry), retries happen inside it, and only an exhausted
    retry budget counts as a breaker failure. Checking the breaker per retry
    would open it on a single slow-but-recovering provider.
    """
    breaker = breaker or get_breaker(name)
    try:
        await breaker.guard()
    except CircuitOpen as exc:
        logger.warning("[reliability] skipping '%s': %s", name, exc)
        if fallback is not None:
            return await fallback()
        raise

    try:
        result = await retry_async(fn, policy=policy, label=name)
    except asyncio.CancelledError:
        raise
    except BaseException as exc:  # noqa: BLE001
        await breaker.record_failure()
        if fallback is not None:
            logger.warning(
                "[reliability] '%s' failed (%s); using fallback", name, type(exc).__name__
            )
            return await fallback()
        raise
    await breaker.record_success()
    return result


# ---------------------------------------------------------------------------
# Bulkheads
# ---------------------------------------------------------------------------

class Bulkhead:
    """Named concurrency ceiling.

    Separate pools per resource kind so a burst of page fetches cannot starve
    LLM calls (or vice versa) on a memory-constrained host. `asyncio.Semaphore`
    is created lazily so a Bulkhead can be constructed outside a running loop.
    """

    def __init__(self, name: str, limit: int) -> None:
        self.name = name
        self.limit = max(1, int(limit))
        self._sem: Optional[asyncio.Semaphore] = None
        self._in_flight = 0
        self._peak = 0

    def _semaphore(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.limit)
        return self._sem

    async def __aenter__(self) -> "Bulkhead":
        await self._semaphore().acquire()
        self._in_flight += 1
        self._peak = max(self._peak, self._in_flight)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._in_flight -= 1
        self._semaphore().release()

    async def run(self, fn: Callable[[], Awaitable[T]]) -> T:
        async with self:
            return await fn()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "limit": self.limit,
            "in_flight": self._in_flight,
            "peak": self._peak,
        }


async def gather_bounded(
    factories: Sequence[Callable[[], Awaitable[T]]],
    limit: int,
    *,
    return_exceptions: bool = True,
) -> list:
    """Run coroutine factories with at most `limit` in flight.

    `asyncio.gather` over N coroutines starts all N immediately; the existing
    code relied on an inner semaphore to hold them back, which still allocated
    every coroutine frame and every provider client up front. Factories keep
    unstarted work unallocated — the point of a concurrency cap on 8GB.
    """
    sem = asyncio.Semaphore(max(1, int(limit)))
    results: list = [None] * len(factories)

    async def _run(index: int, factory: Callable[[], Awaitable[T]]) -> None:
        async with sem:
            try:
                results[index] = await factory()
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001
                if not return_exceptions:
                    raise
                results[index] = exc

    await asyncio.gather(*(_run(i, f) for i, f in enumerate(factories)))
    return results


def ok_results(results: Iterable[Any]) -> list:
    """Drop exception placeholders from a `gather_bounded` result list."""
    return [r for r in results if not isinstance(r, BaseException)]
