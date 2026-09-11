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


def _is_non_retryable_error(exc: BaseException) -> bool:
    """Client errors fail fast to fallback: 401/403 (bad credentials) and
    400/404/405/422 (malformed endpoint, model, or payload) will not resolve
    by retrying — retrying only burns time and worsens rate limiting."""
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response is not None
        and exc.response.status_code in (400, 401, 403, 404, 405, 422)
    )


def _is_retryable(exc: BaseException) -> bool:
    if _is_non_retryable_error(exc):
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
            # Capped well below the research timeout: honoring a 30s+
            # Retry-After inside a 90s research budget would convert every
            # rate-limit burst into a run timeout.
            return min(max(delay, 1.0), 10.0)
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
        """
        for attempt in range(retries):
            try:
                text = await self._generate_with_fallback(system_prompt, user_prompt)
                payload = self._extract_json(text)
                if response_model is not None:
                    validated = response_model.model_validate(payload)
                    return validated.model_dump()
                return payload
            except Exception as exc:
                if attempt == retries - 1:
                    raise
                logger.warning(
                    "[LLM] attempt %d/%d failed, retrying: %s", attempt + 1, retries, exc, exc_info=exc
                )
                await asyncio.sleep(0.7 * (attempt + 1))
        return {}

    async def _generate_with_fallback(self, system_prompt: str, user_prompt: str) -> str:
        groq_key = _real_key(self.settings.groq_api_key)
        hf_key = _real_key(self.settings.huggingface_api_key)
        custom = self._custom_config()
        if custom and not self.custom_breaker.is_open():
            try:
                text = await self._call_custom(system_prompt, user_prompt, custom)
                self.custom_breaker.record_success()
                return text
            except Exception as exc:
                self.custom_breaker.record_failure()
                logger.warning(
                    "[LLM] Custom provider call failed (breaker failures=%d), falling back: %s",
                    self.custom_breaker._consecutive_failures,
                    exc,
                    exc_info=exc,
                )
        if groq_key and not self.groq_breaker.is_open():
            try:
                text = await self._call_groq(system_prompt, user_prompt)
                self.groq_breaker.record_success()
                return text
            except Exception as exc:
                self.groq_breaker.record_failure()
                logger.warning(
                    "[LLM] Groq call failed (breaker failures=%d), falling back: %s",
                    self.groq_breaker._consecutive_failures,
                    exc,
                    exc_info=exc,
                )

        if hf_key:
            return await self._call_huggingface(system_prompt, user_prompt)

        raise RuntimeError("No LLM provider configured. Set GROQ_API_KEY, HUGGINGFACE_API_KEY, or the CUSTOM_LLM_* trio.")

    def _custom_config(self) -> Dict[str, str] | None:
        """Validated custom-provider trio, or None when not configured."""
        key = _real_key(self.settings.custom_llm_api_key)
        # Defensive: strip URL fragments/params users paste from docs
        # ("https://host/v1# comment" must not become the endpoint).
        base = str(self.settings.custom_llm_base_url or "").split("#")[0].strip().rstrip("/")
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
        """Generic OpenAI-compatible chat completions call."""
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_sec) as client:
            response = await client.post(
                custom["endpoint"],
                headers={
                    "Authorization": f"Bearer {custom['api_key']}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": custom["model"],
                    "temperature": 0.1,
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
