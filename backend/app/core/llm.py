import asyncio
import json
from app.core.logging import get_logger
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Tuple, Type

import httpx
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.config import Settings
from app.core import llm_cache
from app.core.usage import get_run_usage

logger = get_logger(__name__)


@dataclass
class CompletionResult:
    """One provider completion with the accounting the budget governor needs.

    `input_tokens`/`output_tokens` come from the provider's `usage` field
    when present; HF's text-only responses fall back to char estimates.
    """

    text: str
    provider: str
    model: str
    endpoint: str
    input_tokens: int = 0
    output_tokens: int = 0


def _parse_usage(data: Any) -> Tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from an OpenAI-compatible
    response; (0, 0) when the provider omits usage (callers estimate)."""
    try:
        usage = data.get("usage") or {}
        return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    except (AttributeError, TypeError, ValueError):
        return 0, 0

HF_FALLBACK_MODELS: List[str] = [
    "Qwen/Qwen2.5-7B-Instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]


class AllProvidersFailedError(RuntimeError):
    """Every configured provider refused the call within this attempt.

    Distinct from "no keys configured": keys exist but each provider failed
    (rate limit, quota, auth, outage). Deterministic within a request —
    retrying the same chain immediately only burns the research budget —
    so generate_json treats it as fail-fast and the caller's deterministic
    fallback engages.
    """


class PromptTooLargeError(RuntimeError):
    """The provider rejected the request size (HTTP 413).

    Observed live: Groq 413s between ~21KB and ~41KB request payloads —
    exactly where the summarizer's 12-source excerpt prompt lands. Retry-
    ing the identical payload can only fail identically, so this is fail-
    fast at every level; the CALLER (summarizer/synthesizer) retries with
    a smaller prompt via its own ladder instead of degrading to extraction.
    """


# HTTP 400 bodies that mark a PERMANENT condition. Retrying them burns the
# research budget on a guaranteed failure (live case: a provider whose
# credits ran out 400s every call — the old code retried it 4x per LLM call).
_PERMANENT_400_SIGNATURES = (
    "insufficient balance", "insufficient credit", "credit insufficient",
    "no credits", "quota exceeded", "exceeded your quota",
    "invalid api key", "invalid_api_key", "model_not_found",
    "model not found", "does not exist", "deprecated",
)


def _permanent_client_error(exc: BaseException) -> bool:
    if not isinstance(exc, httpx.HTTPStatusError) or exc.response is None:
        return False
    if exc.response.status_code == 404:
        return True
    if exc.response.status_code != 400:
        return False
    try:
        body = exc.response.text.lower()
    except Exception:
        return False
    return any(sig in body for sig in _PERMANENT_400_SIGNATURES)


class CircuitBreaker:
    """Per-provider failure counter with a cooldown window.

    After `threshold` consecutive failures the provider is skipped for
    `cooldown_sec` instead of being retried on every call.
    """

    def __init__(self, threshold: int = 3, cooldown_sec: float = 60.0):
        self.threshold = threshold
        self.cooldown_sec = cooldown_sec
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if (time.monotonic() - self._opened_at) >= self.cooldown_sec:
            # Cooldown elapsed — allow one probe attempt through.
            self._reset()
            return False
        return True

    def record_success(self) -> None:
        self._reset()

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.threshold and self._opened_at is None:
            self._opened_at = time.monotonic()
            logger.warning(
                "[LLM] circuit breaker OPEN after %d consecutive failures (cooldown %.0fs)",
                self._consecutive_failures,
                self.cooldown_sec,
            )

    def record_timeout(self) -> None:
        """Open immediately on a single per-attempt timeout. A timeout
        already burned llm_timeout_sec (25s) of the 90s research budget,
        so waiting for `threshold` failures like fast errors do would let
        one slow provider eat the whole run — including re-runs on every
        outer generate_json retry. Cooldown still re-probes afterwards."""
        self._consecutive_failures += 1
        if self._opened_at is None:
            self._opened_at = time.monotonic()
            logger.warning(
                "[LLM] circuit breaker OPEN after timeout (cooldown %.0fs)",
                self.cooldown_sec,
            )

    def _reset(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None


def _attempt_timeout(base: float) -> httpx.Timeout:
    """Per-attempt read-gap timeout (httpx 0.28 has no total cap)."""
    return httpx.Timeout(base)


async def _with_total_cap(coro_factory: Callable[[], Awaitable[Any]], base: float) -> Any:
    """Run one provider attempt under a HARD total wall-clock cap.

    read=base bounds the inter-chunk gap only. Without a total, a
    drip-feeding proxy (bytes every few seconds, never a 60s gap) holds a
    call open far past the intended budget — measured live: a 16-MINUTE
    planner call against a stalled proxy whose read gaps never tripped the
    read timeout. The wall-clock cap is base * 1.5 at asyncio level; the
    raised TimeoutError is classified as a timeout everywhere (fail-fast,
    breaker-recorded as a timeout)."""
    return await asyncio.wait_for(coro_factory(), timeout=base * 1.5)


def _real_key(value: Any) -> str:
    """Settings may carry example placeholders (your_...); treat those as
    unset so we skip the provider instead of burning calls on 401s."""
    text = str(value or "").strip()
    if not text or text.lower().startswith("your_"):
        return ""
    return text


def _is_fail_fast_error(exc: BaseException) -> bool:
    """401/402/403 must fail fast to fallback — retrying bad credentials or
    an empty wallet only burns time and worsens rate limiting."""
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response is not None
        and exc.response.status_code in (401, 402, 403)
    )


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, PromptTooLargeError):
        # The identical payload can only 413 again; the caller shrinks it.
        return False
    if _permanent_client_error(exc):
        # Empty wallet / unknown model / bad key: deterministic failure.
        return False
    if _is_fail_fast_error(exc):
        return False
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        # A per-attempt timeout means this provider cannot serve the prompt
        # inside llm_timeout_sec; retrying the same slow call would burn the
        # 90s research budget before the fallback chain is ever reached.
        # Fail fast to the next provider instead. Fast failures (connect
        # errors, 5xx, 429) stay retryable below.
        return False
    return isinstance(exc, (httpx.HTTPError, RuntimeError))


def _retry_after_hint(exc: BaseException | None) -> float:
    """The provider's own retry hint, in seconds, or 0 when none given.

    Groq puts a numeric header OR a 'Please try again in X.XXs' body line
    on TPM-limit 429s; honoring the provider's number beats guessing."""
    if not isinstance(exc, httpx.HTTPStatusError) or exc.response is None:
        return 0.0
    raw = exc.response.headers.get("retry-after") or ""
    try:
        return float(raw)
    except (TypeError, ValueError):
        pass
    match = re.search(r"try again in ([0-9.]+)s", exc.response.text.lower())
    return float(match.group(1)) if match else 0.0


def _wait_with_retry_after(retry_state) -> float:
    """Honor the provider's Retry-After on 429s; exponential jitter otherwise.

    The first retry of a call may wait up to 20s when the provider says so:
    free-tier TPM windows are the difference between an LLM-written report
    and an extractive fallback, and a rolling per-minute window usually
    clears within one patient wait. Later retries cap at 3s so a sustained
    wall fails fast to the next provider instead of stalling the run."""
    outcome = getattr(retry_state, "outcome", None)
    exc = outcome.exception() if outcome is not None else None
    if (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response is not None
        and exc.response.status_code == 429
    ):
        delay = _retry_after_hint(exc)
        if delay:
            cap = 20.0 if getattr(retry_state, "attempt_number", 1) <= 1 else 3.0
            return min(max(delay, 1.0), cap)
    return wait_exponential_jitter(initial=0.4, max=3)(retry_state)


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Seed the response cache (idempotent; disabled flag respected).
        try:
            llm_cache.configure(settings)
        except Exception as exc:  # pragma: no cover - cache must never block startup
            logger.warning("llm cache configure failed: %s", exc)
        # Breaker state lives on the client instance (created once at app
        # startup), is bounded, and resets on success — not per-request state.
        self.groq_breaker = CircuitBreaker(threshold=3, cooldown_sec=60.0)
        self.custom_breaker = CircuitBreaker(threshold=3, cooldown_sec=60.0)
        # Concurrency cap across ALL LLM calls (planner, N summarizer workers,
        # critic, synthesizer). Without it, one expansion pass fires up to
        # max_parallel_agents simultaneous prompts — the observed cause of
        # Groq 429/TPD exhaustion on free tiers. Starts low per AGENTS.md §5.
        self._llm_semaphore = asyncio.Semaphore(max(1, int(getattr(settings, "max_parallel_llm", 2) or 2)))
        # Last failure per provider, surfaced when the whole chain fails.
        self._last_errors: Dict[str, str] = {}
        # The custom/active-provider leg shares one breaker keyed to the
        # resolved (endpoint, model) identity. Switching providers (Providers
        # tab) must NOT inherit the previous provider's failure state — a
        # dead provider A timed out once would otherwise block healthy
        # provider B for the full cooldown.
        self._custom_identity: tuple | None = None
        # Pre-flight probe cache: successes hold 30s, failures 10s. Without
        # it every stream request re-pinged every provider — free-tier
        # rate meters ticked for pings, and users waited for them.
        self._probe_cache: tuple[float, bool, str] | None = None
        # Negative cache for the provider-store lookup. When the DB has no
        # `llm_providers` table (bench/scripts, or a route that runs before
        # init_db), every generation used to re-query and log a full
        # traceback — hundreds of identical warnings per benchmark run. A
        # short hold (10s) collapses the spam while still retrying soon
        # enough that a startup-ordering issue self-heals.
        self._provider_store_down_until: float = 0.0

    async def probe_targets(self) -> List[Dict[str, str]]:
        """Providers a pre-flight probe should ping. Exclusive active
        provider → that one only; otherwise the full env chain."""
        custom, exclusive = await self._resolve_custom()
        fallback_ok = bool(getattr(self.settings, "active_provider_fallback", False))
        targets: List[Dict[str, str]] = []
        if custom:
            targets.append({
                "name": str(custom.get("name", "custom")),
                "endpoint": custom["endpoint"],
                "api_key": custom["api_key"],
                "model": custom["model"],
                "style": "openai",
            })
        if not exclusive or fallback_ok:
            groq_key = _real_key(self.settings.groq_api_key)
            if groq_key:
                targets.append({
                    "name": "groq",
                    "endpoint": "https://api.groq.com/openai/v1/chat/completions",
                    "api_key": groq_key,
                    "model": self.settings.groq_model,
                    "style": "openai",
                })
            hf_key = _real_key(self.settings.huggingface_api_key)
            if hf_key:
                targets.append({
                    "name": "huggingface",
                    "endpoint": f"https://api-inference.huggingface.co/models/{self.settings.huggingface_model}",
                    "api_key": hf_key,
                    "model": self.settings.huggingface_model,
                    "style": "hf",
                })
        return targets

    async def probe_all(self, timeout: float = 10.0) -> tuple[bool, str]:
        """Pre-flight check for the research stream: one tiny parallel ping
        per configured provider, each capped at `timeout` seconds.

        Results are cached briefly (30s success / 10s failure) so a burst of
        queued requests doesn't re-ping providers — the probes themselves
        consume free-tier rate budget.

        Returns (any_ok, failure_detail). Fail-open on internal errors —
        a probe bug must never block research; the pipeline's own
        degradation handling still applies. When every provider fails the
        run is doomed (it would degrade to extraction), so the caller can
        fail fast with the per-provider reasons instead of wasting the
        research budget.
        """
        now = time.monotonic()
        if self._probe_cache is not None and now < self._probe_cache[0]:
            return self._probe_cache[1], self._probe_cache[2]
        try:
            targets = await self.probe_targets()
        except Exception as exc:
            logger.warning("probe target resolution failed, skipping pre-flight: %s", exc, exc_info=exc)
            return True, ""
        if not targets:
            return False, (
                "no LLM provider is configured "
                "(set GROQ_API_KEY / CUSTOM_LLM_* in .env, or add and select one in the Providers tab)"
            )

        async def _ping(target: Dict[str, str]) -> tuple[bool, str, str]:
            try:
                if target["style"] == "hf":
                    payload: Dict[str, Any] = {
                        "inputs": "Reply with exactly: ok",
                        "parameters": {"max_new_tokens": 4, "return_full_text": False},
                    }
                else:
                    payload = {
                        "model": target["model"],
                        "max_tokens": 8,
                        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
                    }
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        target["endpoint"],
                        headers={"Authorization": f"Bearer {target['api_key']}", "Content-Type": "application/json"},
                        json=payload,
                    )
                    response.raise_for_status()
                return True, target["name"], ""
            except Exception as exc:
                detail = str(exc).strip() or type(exc).__name__
                return False, target["name"], f"{type(exc).__name__}: {detail[:140]}"

        results = await asyncio.gather(*(_ping(t) for t in targets))
        any_ok = any(ok for ok, _, _ in results)
        if any_ok:
            # Cache successes only: a healthy provider stays healthy for the
            # next 30s, but failures must re-probe (the test suite's recovery
            # case, and users retrying after a quota reset, need fresh state).
            self._probe_cache = (time.monotonic() + 30.0, True, "")
            return True, ""
        detail = "; ".join(f"{name}: {err}" for _, name, err in results)
        self._probe_cache = None
        return False, detail

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        retries: int = 3,
        response_model: Type[BaseModel] | None = None,
    ) -> Dict[str, Any]:
        """Call an LLM and return the parsed JSON as a dict.

        When `response_model` is provided, the parsed payload is validated
        against the Pydantic model; validation failures are treated like any
        other failed attempt and trigger a retry instead of returning garbage.

        Timeouts are never retried at this level: every provider already had
        its single budgeted chance inside _generate_with_fallback, so
        re-running the whole chain would multiply slow-provider time past
        the research timeout. The caller falls back immediately instead.
        AllProvidersFailedError is likewise never retried: providers did not
        become healthy 0.7s later inside the same request.
        """
        for attempt in range(retries):
            try:
                text = await self._generate_with_fallback(system_prompt, user_prompt)
                payload = self._extract_json(text)
                if response_model is not None:
                    validated = response_model.model_validate(payload)
                    return validated.model_dump()
                return payload
            except (httpx.TimeoutException, TimeoutError, AllProvidersFailedError, PromptTooLargeError):
                raise
            except Exception as exc:
                if attempt == retries - 1:
                    raise
                logger.warning(
                    "[LLM] attempt %d/%d failed, retrying: %s", attempt + 1, retries, exc, exc_info=exc
                )
                await asyncio.sleep(0.7 * (attempt + 1))
        return {}

    async def _resolve_custom(self) -> tuple[Dict[str, str] | None, bool]:
        """(config, exclusive). A DB-selected active provider wins and is
        EXCLUSIVE — the user's explicit choice, so no other key is spent.
        Otherwise the env CUSTOM_LLM_* trio (non-exclusive, legacy chain)."""
        now = time.monotonic()
        if now >= self._provider_store_down_until:
            try:
                from app.core.providers import get_active_provider

                active = await get_active_provider(self.settings.database_url)
            except Exception as exc:
                # Warn once, then stay quiet for the hold window: the same
                # missing table would otherwise log a traceback on every LLM
                # call. exc_info is included only on the first failure.
                logger.warning(
                    "[LLM] provider store unreadable, using env config: %s", exc,
                    exc_info=exc,
                )
                self._provider_store_down_until = now + 10.0
                active = None
            else:
                self._provider_store_down_until = 0.0
        else:
            active = None
        if active:
            base = str(active.get("base_url", "") or "").strip().rstrip("/")
            endpoint = base if base.lower().endswith("/chat/completions") else base + "/chat/completions"
            return {
                "api_key": str(active.get("api_key", "") or ""),
                "endpoint": endpoint,
                "model": str(active.get("model", "") or ""),
                "name": str(active.get("name", "") or "active provider"),
            }, True
        return self._custom_config(), False

    async def _generate_with_fallback(self, system_prompt: str, user_prompt: str) -> str:
        groq_key = _real_key(self.settings.groq_api_key)
        hf_key = _real_key(self.settings.huggingface_api_key)
        custom, exclusive = await self._resolve_custom()
        # Strict exclusivity (default) zeroes the env keys: a failing active
        # provider degrades the run instead of silently spending another
        # provider's key. active_provider_fallback relaxes exactly that —
        # built for flaky free proxies: primary when healthy, rescued by
        # Groq/HF when it stalls.
        fallback_ok = exclusive and bool(getattr(self.settings, "active_provider_fallback", False))
        if exclusive and not fallback_ok:
            groq_key = ""
            hf_key = ""
        # Identity-keyed breaker reset (see __init__): switching to a
        # DIFFERENT endpoint+model gets a fresh breaker, never the previous
        # provider's failure count. The first resolution (None → identity)
        # keeps any pre-existing state — only a real switch resets.
        identity = (custom["endpoint"], custom["model"]) if custom else None
        if identity is not None and self._custom_identity is not None and identity != self._custom_identity:
            self.custom_breaker = CircuitBreaker(threshold=3, cooldown_sec=60.0)
        self._custom_identity = identity

        # ---- Response cache: exact-prompt hits skip providers entirely ----
        cache_endpoint = (custom or {}).get("endpoint", "https://api.groq.com/openai/v1/chat/completions")
        cache_model = (custom or {}).get("model", self.settings.groq_model)
        cached = llm_cache.get(cache_endpoint, cache_model, system_prompt, user_prompt)
        if cached is not None:
            self._record_usage(
                cached["text"], provider="cache", model=cache_model,
                input_tokens=cached.get("input_tokens") or None,
                output_tokens=cached.get("output_tokens") or None,
                cached=True,
            )
            return cached["text"]

        attempted = 0
        size_failures = 0
        async with self._llm_semaphore:
            if custom and not self.custom_breaker.is_open():
                attempted += 1
                try:
                    result = await self._call_custom(system_prompt, user_prompt, custom)
                    self.custom_breaker.record_success()
                    self._cache_and_record(result, system_prompt, user_prompt)
                    return result.text
                except Exception as exc:
                    self._last_errors["custom"] = f"{type(exc).__name__}: {exc}"
                    size_failures += 1 if isinstance(exc, PromptTooLargeError) else 0
                    self._record_provider_failure(self.custom_breaker, exc)
                    logger.warning(
                        "[LLM] Custom provider call failed (breaker failures=%d), falling back: %s",
                        self.custom_breaker._consecutive_failures,
                        exc,
                        exc_info=exc,
                    )
                    if exclusive and not fallback_ok:
                        raise AllProvidersFailedError(
                            f"Active provider '{custom.get('name', 'custom')}' failed: "
                            f"{type(exc).__name__}. No fallback providers run while "
                            "one is selected — the run continues on deterministic fallbacks."
                        ) from exc
            if groq_key and not self.groq_breaker.is_open():
                attempted += 1
                try:
                    result = await self._call_groq(system_prompt, user_prompt)
                    self.groq_breaker.record_success()
                    self._cache_and_record(result, system_prompt, user_prompt)
                    return result.text
                except Exception as exc:
                    self._last_errors["groq"] = f"{type(exc).__name__}: {exc}"
                    size_failures += 1 if isinstance(exc, PromptTooLargeError) else 0
                    self._record_provider_failure(self.groq_breaker, exc)
                    logger.warning(
                        "[LLM] Groq call failed (breaker failures=%d), falling back: %s",
                        self.groq_breaker._consecutive_failures,
                        exc,
                        exc_info=exc,
                    )

            if hf_key:
                attempted += 1
                try:
                    result = await self._call_huggingface(system_prompt, user_prompt)
                    self._cache_and_record(result, system_prompt, user_prompt)
                    return result.text
                except Exception as exc:
                    self._last_errors["huggingface"] = f"{type(exc).__name__}: {exc}"
                    size_failures += 1 if isinstance(exc, PromptTooLargeError) else 0
                    raise

        # Every attempted provider rejected the REQUEST SIZE: the providers
        # are healthy — the payload is not. Raise the sizing signal so the
        # caller's ladder can shrink and retry instead of degrading.
        if attempted > 0 and size_failures == attempted:
            raise PromptTooLargeError(
                f"all {attempted} provider(s) rejected the request size as too large"
            )

        if attempted == 0:
            # Keys existed but every provider's breaker was open — or nothing
            # was configured at all. The nothing-configured case keeps the
            # legacy message (routes.py matches it for a friendly NDJSON
            # error); breaker-open is a different failure with its own error.
            if not (custom or groq_key or hf_key):
                raise RuntimeError("No LLM provider configured. Set GROQ_API_KEY, HUGGINGFACE_API_KEY, or the CUSTOM_LLM_* trio.")
            configured = [name for name, ok in (
                ("custom", bool(custom)), ("groq", bool(groq_key)), ("huggingface", bool(hf_key)),
            ) if ok]
            if exclusive and custom and not fallback_ok:
                configured = [f"active provider '{custom.get('name', 'custom')}' (exclusive, no fallbacks)"]
            raise AllProvidersFailedError(
                "All LLM providers skipped this attempt — circuit breakers open "
                f"for: {', '.join(configured)}. Wait for the breaker cooldown "
                "(60s) or check provider quotas."
            )
        def _brief(msg: str) -> str:
            first = (msg or "").strip().splitlines()[0] if (msg or "").strip() else "unknown error"
            return first[:160]

        detail = "; ".join(f"{name}: {_brief(msg)}" for name, msg in self._last_errors.items())
        raise AllProvidersFailedError(
            f"All {attempted} configured LLM provider(s) failed — {detail or 'unknown errors'}. "
            "The run will continue on deterministic fallbacks."
        )

    @staticmethod
    def _record_usage(
        text: str,
        *,
        provider: str,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
        cached: bool = False,
        system_prompt: str = "",
        user_prompt: str = "",
    ) -> None:
        """Feed the per-run ledger if one is active (research runs only).

        Outside a run (provider probes, tests) there is no ledger and
        nothing is recorded — the call still works exactly as before."""
        usage = get_run_usage()
        if usage is None:
            return
        stage = usage.stage_hint or "llm"
        try:
            # `cached` is threaded into the budget so a cache hit records $0
            # at write time — no post-hoc "refund the last record" step that
            # could subtract the wrong amount if records interleave.
            usage.record_llm(
                stage,
                prompt=f"{system_prompt}\n{user_prompt}"[:20000],
                completion=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model or None,
                cached=cached,
            )
        except Exception as exc:  # accounting must never break generation
            logger.warning("usage recording failed: %s", exc)

    def _cache_and_record(
        self, result: "CompletionResult", system_prompt: str, user_prompt: str
    ) -> None:
        """Persist a successful completion and feed the run ledger."""
        llm_cache.put(
            result.endpoint,
            result.model,
            system_prompt,
            user_prompt,
            result.text,
            input_tokens=result.input_tokens or None,
            output_tokens=result.output_tokens or None,
        )
        tin, tout = result.input_tokens, result.output_tokens
        if not tin:
            from app.agents.budget import estimate_tokens
            tin = estimate_tokens(f"{system_prompt}\n{user_prompt}")
        if not tout:
            from app.agents.budget import estimate_tokens
            tout = estimate_tokens(result.text)
        self._record_usage(
            result.text,
            provider=result.provider,
            model=result.model,
            input_tokens=tin,
            output_tokens=tout,
            cached=False,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    @staticmethod
    def _record_provider_failure(breaker: CircuitBreaker, exc: Exception) -> None:
        """Timeouts open the breaker immediately (one 25s stall is enough
        signal inside a 90s budget); fast failures use the normal
        threshold counter. PromptTooLargeError records NOTHING: the provider
        is healthy — our payload was too big — and sizing retries must not
        trip the breaker into skipping that provider."""
        if isinstance(exc, PromptTooLargeError):
            return
        if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
            breaker.record_timeout()
        else:
            breaker.record_failure()

    def _custom_config(self) -> Dict[str, str] | None:
        """Validated custom-provider trio, or None when not configured."""
        key = _real_key(self.settings.custom_llm_api_key)
        base = str(self.settings.custom_llm_base_url or "").strip().rstrip("/")
        model = str(self.settings.custom_llm_model or "").strip()
        if not (key and base and model):
            return None
        if base.lower().endswith("/chat/completions"):
            endpoint = base
        else:
            endpoint = base + "/chat/completions"
        return {"api_key": key, "endpoint": endpoint, "model": model}

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=_wait_with_retry_after,
        retry=retry_if_exception(_is_retryable),
    )
    async def _call_custom(self, system_prompt: str, user_prompt: str, custom: Dict[str, str]) -> CompletionResult:
        """Generic OpenAI-compatible chat completions call.

        Uses custom_llm_timeout_sec (not the shared llm_timeout_sec):
        slower third-party providers routinely need 30-60s on planner-sized
        prompts, and a premature ReadTimeout opens the breaker and degrades
        the whole run. The whole attempt sits under a hard wall-clock cap
        (_with_total_cap) so a drip-feeding server cannot hang the stage.
        """
        timeout_base = self.settings.custom_llm_timeout_sec

        async def _do() -> CompletionResult:
            return await self._post_custom(system_prompt, user_prompt, custom, timeout_base)

        return await _with_total_cap(_do, timeout_base)

    async def _post_custom(self, system_prompt: str, user_prompt: str,
                           custom: Dict[str, str], timeout_base: float) -> CompletionResult:
        async with httpx.AsyncClient(timeout=_attempt_timeout(timeout_base)) as client:
            response = await client.post(
                custom["endpoint"],
                headers={
                    "Authorization": f"Bearer {custom['api_key']}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": custom["model"],
                    "temperature": 0.1,
                    # generate_json always JSON-parses the reply, so request
                    # JSON mode instead of hoping the model obeys the prompt.
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            if response.status_code == 413:
                raise PromptTooLargeError(
                    "provider rejected request size (custom provider)"
                )
            response.raise_for_status()
            data = response.json()
            tin, tout = _parse_usage(data)
            return CompletionResult(
                text=data["choices"][0]["message"]["content"],
                provider="custom",
                model=custom["model"],
                endpoint=custom["endpoint"],
                input_tokens=tin,
                output_tokens=tout,
            )

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=_wait_with_retry_after,
        retry=retry_if_exception(_is_retryable),
    )
    async def _call_groq(self, system_prompt: str, user_prompt: str) -> CompletionResult:
        return await _with_total_cap(
            lambda: self._post_groq(system_prompt, user_prompt),
            self.settings.llm_timeout_sec,
        )

    async def _post_groq(self, system_prompt: str, user_prompt: str) -> CompletionResult:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.groq_model,
            "temperature": 0.1,
            # generate_json always JSON-parses the reply, so request JSON
            # mode instead of hoping the model obeys the prompt.
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        # Per-attempt timeout is llm_timeout_sec; the decorator bounds retries
        # so total wait stays a small multiple of one attempt.
        async with httpx.AsyncClient(
            timeout=_attempt_timeout(self.settings.llm_timeout_sec)
        ) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 413:
                raise PromptTooLargeError("provider rejected request size (groq)")
            response.raise_for_status()
            data = response.json()
            tin, tout = _parse_usage(data)
            return CompletionResult(
                text=data["choices"][0]["message"]["content"],
                provider="groq",
                model=self.settings.groq_model,
                endpoint=url,
                input_tokens=tin,
                output_tokens=tout,
            )

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=_wait_with_retry_after,
        retry=retry_if_exception(_is_retryable),
    )
    async def _call_huggingface(self, system_prompt: str, user_prompt: str) -> CompletionResult:
        return await _with_total_cap(
            lambda: self._post_huggingface(system_prompt, user_prompt),
            self.settings.llm_timeout_sec,
        )

    async def _post_huggingface(self, system_prompt: str, user_prompt: str) -> CompletionResult:
        headers = {
            "Authorization": f"Bearer {self.settings.huggingface_api_key}",
            "Content-Type": "application/json",
        }
        prompt = (
            "You are a precise research assistant. Return valid JSON only.\n\n"
            f"System: {system_prompt}\n\n"
            f"User: {user_prompt}"
        )
        payload: Dict[str, Any] = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.2,
                "return_full_text": False,
            },
        }

        model_candidates = [self.settings.huggingface_model, *[m for m in HF_FALLBACK_MODELS if m != self.settings.huggingface_model]]
        not_available_errors: List[str] = []

        async with httpx.AsyncClient(
            timeout=_attempt_timeout(self.settings.llm_timeout_sec)
        ) as client:
            for model_name in model_candidates:
                url = f"https://api-inference.huggingface.co/models/{model_name}"
                response = await client.post(url, headers=headers, json=payload)

                if response.status_code in (404, 410):
                    not_available_errors.append(f"{model_name} -> {response.status_code}")
                    continue

                response.raise_for_status()
                data = response.json()

                text = None
                if isinstance(data, list) and data and "generated_text" in data[0]:
                    text = data[0]["generated_text"]
                elif isinstance(data, dict) and "generated_text" in data:
                    text = data["generated_text"]
                elif isinstance(data, dict) and "error" in data:
                    # Model can be valid but unavailable due to provider-side load.
                    raise RuntimeError(f"HuggingFace model '{model_name}' error: {data.get('error')}")
                if text is not None:
                    # HF inference returns bare text: estimate token usage.
                    from app.agents.budget import estimate_tokens
                    return CompletionResult(
                        text=text,
                        provider="huggingface",
                        model=model_name,
                        endpoint=url,
                        input_tokens=estimate_tokens(prompt),
                        output_tokens=estimate_tokens(text),
                    )

        tried = ", ".join(not_available_errors) if not_available_errors else "no models tried"
        raise RuntimeError(
            "HuggingFace inference models are unavailable (404/410). "
            "Set HUGGINGFACE_MODEL in .env to an available model. "
            f"Tried: {tried}"
        )

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in model output")

        return json.loads(match.group(0))


def clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))
