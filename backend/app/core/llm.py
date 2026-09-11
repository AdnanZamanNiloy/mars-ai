import asyncio
import json
from app.core.logging import get_logger
import re
import time
from typing import Any, Dict, List, Type

import httpx
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.config import Settings

logger = get_logger(__name__)

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


def _real_key(value: Any) -> str:
    """Settings may carry example placeholders (your_...); treat those as
    unset so we skip the provider instead of burning calls on 401s."""
    text = str(value or "").strip()
    if not text or text.lower().startswith("your_"):
        return ""
    return text


def _is_fail_fast_error(exc: BaseException) -> bool:
    """401/403/402 must fail fast to fallback — retrying bad credentials or
    an empty wallet only burns time and worsens rate limiting."""
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response is not None
        and exc.response.status_code in (401, 402, 403)
    )


def _is_retryable(exc: BaseException) -> bool:
    if _is_fail_fast_error(exc):
        return False
    if isinstance(exc, httpx.TimeoutException):
        # A per-attempt timeout means this provider cannot serve the prompt
        # inside llm_timeout_sec; retrying the same slow call would burn the
        # 90s research budget before the fallback chain is ever reached.
        # Fail fast to the next provider instead. Fast failures (connect
        # errors, 5xx, 429) stay retryable below.
        return False
    return isinstance(exc, (httpx.HTTPError, RuntimeError))


def _wait_with_retry_after(retry_state) -> float:
    """Honor the provider's Retry-After on 429s; exponential jitter otherwise."""
    outcome = getattr(retry_state, "outcome", None)
    exc = outcome.exception() if outcome is not None else None
    if (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response is not None
        and exc.response.status_code == 429
    ):
        try:
            delay = float(exc.response.headers.get("retry-after", ""))
            # Capped far below the research timeout: honoring a 10s+
            # Retry-After per attempt (up to ~30s per call) would convert
            # every rate-limit burst into a run timeout. Short blips are
            # still honored; long walls fail fast to the next provider or
            # the deterministic fallback instead of stalling the run.
            return min(max(delay, 1.0), 3.0)
        except (TypeError, ValueError):
            pass
    return wait_exponential_jitter(initial=0.4, max=3)(retry_state)


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
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

    async def probe_targets(self) -> List[Dict[str, str]]:
        """Providers a pre-flight probe should ping. Exclusive active
        provider → that one only; otherwise the full env chain."""
        custom, exclusive = await self._resolve_custom()
        targets: List[Dict[str, str]] = []
        if custom:
            targets.append({
                "name": str(custom.get("name", "custom")),
                "endpoint": custom["endpoint"],
                "api_key": custom["api_key"],
                "model": custom["model"],
                "style": "openai",
            })
        if not exclusive:
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

        Returns (any_ok, failure_detail). Fail-open on internal errors —
        a probe bug must never block research; the pipeline's own
        degradation handling still applies. When every provider fails the
        run is doomed (it would degrade to extraction), so the caller can
        fail fast with the per-provider reasons instead of wasting the
        research budget.
        """
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
            return True, ""
        detail = "; ".join(f"{name}: {err}" for _, name, err in results)
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
            except (httpx.TimeoutException, AllProvidersFailedError) as exc:
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
        try:
            from app.core.providers import get_active_provider

            active = await get_active_provider(self.settings.database_url)
        except Exception as exc:
            logger.warning(
                "[LLM] provider store unreadable, using env config: %s", exc, exc_info=exc
            )
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
        if exclusive:
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
        attempted = 0
        async with self._llm_semaphore:
            if custom and not self.custom_breaker.is_open():
                attempted += 1
                try:
                    text = await self._call_custom(system_prompt, user_prompt, custom)
                    self.custom_breaker.record_success()
                    return text
                except Exception as exc:
                    self._last_errors["custom"] = f"{type(exc).__name__}: {exc}"
                    self._record_provider_failure(self.custom_breaker, exc)
                    logger.warning(
                        "[LLM] Custom provider call failed (breaker failures=%d), falling back: %s",
                        self.custom_breaker._consecutive_failures,
                        exc,
                        exc_info=exc,
                    )
                    if exclusive:
                        raise AllProvidersFailedError(
                            f"Active provider '{custom.get('name', 'custom')}' failed: "
                            f"{type(exc).__name__}. No fallback providers run while "
                            "one is selected — the run continues on deterministic fallbacks."
                        ) from exc
            if groq_key and not self.groq_breaker.is_open():
                attempted += 1
                try:
                    text = await self._call_groq(system_prompt, user_prompt)
                    self.groq_breaker.record_success()
                    return text
                except Exception as exc:
                    self._last_errors["groq"] = f"{type(exc).__name__}: {exc}"
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
                    return await self._call_huggingface(system_prompt, user_prompt)
                except Exception as exc:
                    self._last_errors["huggingface"] = f"{type(exc).__name__}: {exc}"
                    raise

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
            if exclusive and custom:
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
    def _record_provider_failure(breaker: CircuitBreaker, exc: Exception) -> None:
        """Timeouts open the breaker immediately (one 25s stall is enough
        signal inside a 90s budget); fast failures use the normal
        threshold counter."""
        if isinstance(exc, httpx.TimeoutException):
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
    async def _call_custom(self, system_prompt: str, user_prompt: str, custom: Dict[str, str]) -> str:
        """Generic OpenAI-compatible chat completions call.

        Uses custom_llm_timeout_sec (not the shared llm_timeout_sec):
        slower third-party providers routinely need 30-60s on planner-sized
        prompts, and a premature ReadTimeout opens the breaker and degrades
        the whole run.
        """
        async with httpx.AsyncClient(timeout=self.settings.custom_llm_timeout_sec) as client:
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
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=_wait_with_retry_after,
        retry=retry_if_exception(_is_retryable),
    )
    async def _call_groq(self, system_prompt: str, user_prompt: str) -> str:
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
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_sec) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    @retry(
        reraise=True,
        stop=stop_after_attempt(2),
        wait=_wait_with_retry_after,
        retry=retry_if_exception(_is_retryable),
    )
    async def _call_huggingface(self, system_prompt: str, user_prompt: str) -> str:
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

        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_sec) as client:
            for model_name in model_candidates:
                url = f"https://api-inference.huggingface.co/models/{model_name}"
                response = await client.post(url, headers=headers, json=payload)

                if response.status_code in (404, 410):
                    not_available_errors.append(f"{model_name} -> {response.status_code}")
                    continue

                response.raise_for_status()
                data = response.json()

                if isinstance(data, list) and data and "generated_text" in data[0]:
                    return data[0]["generated_text"]
                if isinstance(data, dict) and "generated_text" in data:
                    return data["generated_text"]
                if isinstance(data, dict) and "error" in data:
                    # Model can be valid but unavailable due to provider-side load.
                    raise RuntimeError(f"HuggingFace model '{model_name}' error: {data.get('error')}")

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
