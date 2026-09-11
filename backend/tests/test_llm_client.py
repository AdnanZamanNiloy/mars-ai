"""Retry and circuit-breaker behavior of LLMClient (Phase 1.1)."""
import json

import httpx
import pytest
import respx

from app.core.llm import CircuitBreaker, LLMClient

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
HF_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"


def _groq_response(content: dict) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}]})


@pytest.fixture
def client(test_settings) -> LLMClient:
    return LLMClient(test_settings)


@pytest.fixture
def hf_client() -> LLMClient:
    """Client with real-looking keys for both providers, so the HF fallback
    path is exercised (placeholder keys are skipped by design)."""
    from app.core.config import Settings

    return LLMClient(Settings(groq_api_key="test-key", huggingface_api_key="test-hf-key",
                              _env_file=None))


async def test_retry_recovers_after_transient_failures(client):
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(GROQ_URL).mock(
            side_effect=[
                httpx.ConnectError("transient 1"),
                httpx.ConnectError("transient 2"),
                _groq_response({"ok": True}),
            ]
        )
        result = await client.generate_json("sp", "up")
        assert result == {"ok": True}
        assert route.call_count == 3


async def test_persistent_groq_failure_falls_back_to_hf(hf_client):
    with respx.mock(assert_all_called=False) as mock:
        mock.post(GROQ_URL).mock(return_value=httpx.Response(500, json={"error": "down"}))
        mock.post(HF_URL).mock(return_value=httpx.Response(200, json=[{"generated_text": json.dumps({"via": "hf"})}]))
        result = await hf_client.generate_json("sp", "up")
        assert result == {"via": "hf"}
        assert hf_client.groq_breaker._consecutive_failures >= 1


async def test_open_breaker_skips_groq_entirely(hf_client):
    client = hf_client
    # Threshold 1 so a single failure trips the breaker for this test.
    client.groq_breaker = CircuitBreaker(threshold=1, cooldown_sec=60.0)
    with respx.mock(assert_all_called=False) as mock:
        groq_route = mock.post(GROQ_URL).mock(return_value=httpx.Response(500, json={"error": "down"}))
        mock.post(HF_URL).mock(return_value=httpx.Response(200, json=[{"generated_text": json.dumps({"via": "hf"})}]))

        first = await client.generate_json("sp", "up")
        assert first == {"via": "hf"}
        assert client.groq_breaker.is_open()

        calls_at_breaker_open = groq_route.call_count
        second = await client.generate_json("sp", "up")
        assert second == {"via": "hf"}
        assert groq_route.call_count == calls_at_breaker_open, "Groq must be skipped while breaker is open"


async def test_breaker_resets_on_success(client):
    client.groq_breaker.record_failure()
    client.groq_breaker.record_failure()
    client.groq_breaker.record_failure()
    assert client.groq_breaker.is_open()
    client.groq_breaker.record_success()
    assert not client.groq_breaker.is_open()


async def test_malformed_output_rejected_and_retried(hf_client):
    """Phase 1.2: payload failing response_model validation triggers retry, not silent accept."""
    from app.core.schemas import PlannerOutputModel

    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(GROQ_URL).mock(
            side_effect=[
                _groq_response(json.dumps({"no_sub_questions": True})),  # missing required key
                _groq_response(
                    json.dumps(
                        {
                            "sub_questions": [
                                {"id": 1, "question": "what is retrieval augmented generation", "axis": "definition"}
                            ]
                        }
                    )
                ),
            ]
        )
        # After Groq's two side effects (both invalid) are consumed, later
        # attempts route through the HF fallback — which returns the valid
        # payload. This asserts the malformed response was *rejected*, and
        # the pipeline recovered on a later attempt.
        mock.post(HF_URL).mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "generated_text": json.dumps(
                            {
                                "sub_questions": [
                                    {"id": 1, "question": "what is retrieval augmented generation", "axis": "definition"}
                                ]
                            }
                        )
                    }
                ],
            )
        )
        result = await hf_client.generate_json("sp", "up", response_model=PlannerOutputModel)
        assert result["sub_questions"][0]["question"].startswith("what is")
        assert route.call_count == 2, "malformed response must be retried"


async def test_validation_failure_after_all_retries_raises(client):
    from app.core.schemas import PlannerOutputModel

    with respx.mock(assert_all_called=False) as mock:
        mock.post(GROQ_URL).mock(return_value=_groq_response(json.dumps({"no_sub_questions": True})))
        with pytest.raises(Exception):
            await client.generate_json("sp", "up", response_model=PlannerOutputModel)


async def test_facts_schema_aliases_accepted(client):
    """Fast models return near-miss shapes ({"claims": [...]}, bare lists).
    The facts model coerces them instead of burning retries + quota."""
    from app.core.schemas import SummarizerFactsModel

    good = [{"claim": "Transformers use attention", "source": "https://a.com", "confidence": 0.9}]
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(GROQ_URL).mock(
            side_effect=[
                _groq_response({"sub_question": "q", "claims": good}),
                _groq_response(good),
            ])
        first = await client.generate_json("sp", "up", response_model=SummarizerFactsModel)
        assert first["facts"][0]["claim"] == "Transformers use attention"
        second = await client.generate_json("sp", "up", response_model=SummarizerFactsModel)
        assert second["facts"][0]["source"] == "https://a.com"
        assert route.call_count == 2, "near-miss shapes must not trigger retries"


def test_env_file_precedence_real_key_beats_placeholder():
    """.env must override .env.example placeholders (regression: settings
    loaded 'your_groq_api_key_here' and every LLM call 401'd silently)."""
    from pathlib import Path

    from app.core.config import Settings

    project = Path(__file__).resolve().parents[1]
    env_keys = {}
    for name in (".env.example", ".env"):
        path = project / name
        if path.exists():
            for line in path.read_text().splitlines():
                if line.startswith("GROQ_API_KEY="):
                    env_keys[name] = line.split("=", 1)[1]
    if ".env" in env_keys and not env_keys[".env"].startswith("your_"):
        settings = Settings()
        assert not settings.groq_api_key.startswith("your_"), (
            "placeholder from .env.example is overriding the real .env key"
        )


async def test_auth_error_fails_fast_without_retries(hf_client):
    hf_ok = httpx.Response(200, json=[{"generated_text": json.dumps({"via": "hf"})}])
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(GROQ_URL).mock(
            return_value=httpx.Response(401, json={"error": "invalid key"}))
        mock.post(HF_URL).mock(return_value=hf_ok)
        result = await hf_client.generate_json("sp", "up")
        assert result == {"via": "hf"}
        assert route.call_count == 1


async def test_payment_error_fails_fast_without_retries(hf_client):
    """402 (empty provider wallet) is fail-fast like 401/403: retrying a
    billing wall only burns the research budget."""
    hf_ok = httpx.Response(200, json=[{"generated_text": json.dumps({"via": "hf"})}])
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(GROQ_URL).mock(
            return_value=httpx.Response(402, json={"error": "payment required"}))
        mock.post(HF_URL).mock(return_value=hf_ok)
        result = await hf_client.generate_json("sp", "up")
        assert result == {"via": "hf"}
        assert route.call_count == 1


async def test_rate_limit_retries_then_recovers(hf_client):
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(GROQ_URL).mock(side_effect=[
            httpx.Response(429, headers={"retry-after": "0"}, json={"error": "slow down"}),
            httpx.Response(429, headers={"retry-after": "0"}, json={"error": "slow down"}),
            _groq_response({"ok": True}),
        ])
        result = await hf_client.generate_json("sp", "up")
        assert result == {"ok": True}
        assert route.call_count == 3


async def test_placeholder_keys_are_skipped():
    from app.core.config import Settings

    settings = Settings(groq_api_key="your_groq_key_here",
                        huggingface_api_key="your_hf_key_here", _env_file=None)
    client = LLMClient(settings)
    with respx.mock(assert_all_called=False) as mock:
        groq_route = mock.post(GROQ_URL).mock(return_value=_groq_response({"x": 1}))
        hf_route = mock.post(HF_URL).mock(return_value=_groq_response({"x": 1}))
        with pytest.raises(RuntimeError, match="No LLM provider configured"):
            await client.generate_json("sp", "up")
        assert groq_route.call_count == 0
        assert hf_route.call_count == 0


CUSTOM_URL = "https://llm.example.com/v1/chat/completions"


def _custom_settings(**over):
    from app.core.config import Settings

    base = {"groq_api_key": "test-key", "huggingface_api_key": "test-hf-key",
            "custom_llm_api_key": "test-custom-key",
            "custom_llm_base_url": "https://llm.example.com/v1",
            "custom_llm_model": "test-model", "_env_file": None}
    base.update(over)
    return Settings(**base)


async def test_custom_provider_leads_chain():
    client = LLMClient(_custom_settings())
    with respx.mock(assert_all_called=False) as mock:
        custom_route = mock.post(CUSTOM_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"via": "custom"})}}]}))
        groq_route = mock.post(GROQ_URL).mock(return_value=_groq_response({"via": "groq"}))
        result = await client.generate_json("sp", "up")
        assert result == {"via": "custom"}
        assert custom_route.call_count == 1
        assert groq_route.call_count == 0


async def test_custom_failure_falls_back_to_groq():
    client = LLMClient(_custom_settings())
    with respx.mock(assert_all_called=False) as mock:
        mock.post(CUSTOM_URL).mock(return_value=httpx.Response(500, json={"error": "down"}))
        mock.post(GROQ_URL).mock(return_value=_groq_response({"via": "groq"}))
        result = await client.generate_json("sp", "up")
        assert result == {"via": "groq"}
        assert client.custom_breaker._consecutive_failures >= 1


async def test_partial_custom_trio_skipped():
    client = LLMClient(_custom_settings(custom_llm_api_key=""))
    with respx.mock(assert_all_called=False) as mock:
        custom_route = mock.post(CUSTOM_URL).mock(return_value=_groq_response({"via": "custom"}))
        mock.post(GROQ_URL).mock(return_value=_groq_response({"via": "groq"}))
        result = await client.generate_json("sp", "up")
        assert result == {"via": "groq"}
        assert custom_route.call_count == 0


async def test_full_endpoint_url_accepted():
    client = LLMClient(_custom_settings(
        custom_llm_base_url="https://llm.example.com/v1/chat/completions"))
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(CUSTOM_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"ok": True})}}]}))
        result = await client.generate_json("sp", "up")
        assert result == {"ok": True}
        assert route.call_count == 1


def test_validator_accepts_custom_only():
    from app.core.config import Settings

    settings = Settings(custom_llm_api_key="k", custom_llm_base_url="https://x/v1",
                        custom_llm_model="m", _env_file=None)
    assert settings.custom_llm_model == "m"


async def test_timeout_fails_fast_to_next_provider():
    """A provider timeout must not be retried in-place: one slow attempt,
    then the chain moves on (regression: 4x25s tenacity retries on the
    planner prompt blew past the 90s research timeout before Groq/fallback
    was ever reached)."""
    client = LLMClient(_custom_settings())
    with respx.mock(assert_all_called=False) as mock:
        custom_route = mock.post(CUSTOM_URL).mock(side_effect=httpx.ReadTimeout("slow model"))
        groq_route = mock.post(GROQ_URL).mock(return_value=_groq_response({"via": "groq"}))
        result = await client.generate_json("sp", "up")
        assert result == {"via": "groq"}
        assert custom_route.call_count == 1, "timed-out provider must be tried once, not retried"
        assert groq_route.call_count == 1


async def test_timeout_skips_outer_retry(hf_client):
    """When every provider times out, generate_json raises after one pass
    over the chain instead of re-running slow providers `retries` times."""
    with respx.mock(assert_all_called=False) as mock:
        groq_route = mock.post(GROQ_URL).mock(side_effect=httpx.ReadTimeout("groq slow"))
        hf_route = mock.post(HF_URL).mock(side_effect=httpx.ReadTimeout("hf slow"))
        with pytest.raises(httpx.TimeoutException):
            await hf_client.generate_json("sp", "up")
        assert groq_route.call_count == 1
        assert hf_route.call_count == 1


async def test_timeout_opens_breaker_immediately():
    """One provider timeout trips the breaker at once, so later calls in
    the burst skip the slow provider instead of burning 25s per attempt."""
    client = LLMClient(_custom_settings())
    with respx.mock(assert_all_called=False) as mock:
        custom_route = mock.post(CUSTOM_URL).mock(side_effect=httpx.ReadTimeout("slow model"))
        mock.post(GROQ_URL).mock(return_value=_groq_response({"via": "groq"}))
        assert await client.generate_json("sp", "up") == {"via": "groq"}
        assert client.custom_breaker.is_open()
        assert await client.generate_json("sp", "up") == {"via": "groq"}
        assert custom_route.call_count == 1, "slow provider must be skipped after one timeout"


async def test_groq_timeout_opens_groq_breaker(hf_client):
    with respx.mock(assert_all_called=False) as mock:
        groq_route = mock.post(GROQ_URL).mock(side_effect=httpx.ReadTimeout("groq slow"))
        mock.post(HF_URL).mock(
            return_value=httpx.Response(200, json=[{"generated_text": json.dumps({"via": "hf"})}]))
        assert await hf_client.generate_json("sp", "up") == {"via": "hf"}
        assert hf_client.groq_breaker.is_open()
        assert await hf_client.generate_json("sp", "up") == {"via": "hf"}
        assert groq_route.call_count == 1, "timed-out Groq must be skipped while breaker is open"


async def test_json_mode_requested(client):
    """generate_json always JSON-parses, so both providers must be asked
    for JSON mode instead of hoping the prompt is obeyed."""
    custom_client = LLMClient(_custom_settings())
    with respx.mock(assert_all_called=False) as mock:
        custom_route = mock.post(CUSTOM_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"via": "custom"})}}]}))
        groq_route = mock.post(GROQ_URL).mock(return_value=_groq_response({"via": "groq"}))
        assert await custom_client.generate_json("sp", "up") == {"via": "custom"}
        assert await client.generate_json("sp", "up") == {"via": "groq"}
        for route in (custom_route, groq_route):
            body = json.loads(route.calls[0].request.content.decode("utf-8"))
            assert body.get("response_format") == {"type": "json_object"}


async def test_all_providers_failed_raises_fail_fast():
    """When keys are configured but every provider refuses the call, the
    client raises AllProvidersFailedError and generate_json must NOT re-run
    the chain `retries` times (regression: the full chain was retried 3x on
    a deterministic outage, burning seconds per agent call per run)."""
    client = LLMClient(_custom_settings(huggingface_api_key=""))
    from app.core.llm import AllProvidersFailedError

    with respx.mock(assert_all_called=False) as mock:
        custom_route = mock.post(CUSTOM_URL).mock(
            return_value=httpx.Response(402, json={"error": "payment required"}))
        groq_route = mock.post(GROQ_URL).mock(
            return_value=httpx.Response(402, json={"error": "payment required"}))
        with pytest.raises(AllProvidersFailedError) as exc_info:
            await client.generate_json("sp", "up")
        # Provider detail surfaces so the operator sees WHY everything failed.
        assert "custom" in str(exc_info.value) and "groq" in str(exc_info.value)
        assert custom_route.call_count == 1, "402 must fail fast once per provider"
        assert groq_route.call_count == 1


async def test_open_breakers_skip_all_providers():
    """With every breaker open, the chain reports the skip condition instead
    of pretending no keys exist."""
    from app.core.llm import AllProvidersFailedError

    client = LLMClient(_custom_settings(huggingface_api_key=""))
    client.custom_breaker.record_timeout()
    client.groq_breaker.record_timeout()
    with respx.mock(assert_all_called=False) as mock:
        custom_route = mock.post(CUSTOM_URL).mock(return_value=_groq_response({"x": 1}))
        groq_route = mock.post(GROQ_URL).mock(return_value=_groq_response({"x": 1}))
        with pytest.raises(AllProvidersFailedError, match="circuit breakers open"):
            await client.generate_json("sp", "up")
        assert custom_route.call_count == 0 and groq_route.call_count == 0


async def test_llm_semaphore_bounds_concurrency():
    """No more than max_parallel_llm provider calls may be in flight — the
    concurrent summarizer burst was what exhausted free-tier quotas."""
    import asyncio

    client = LLMClient(_custom_settings(huggingface_api_key="", max_parallel_llm=1))
    in_flight = 0
    peak = 0

    async def _slow_handler(request):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"ok": True})}}]})

    with respx.mock(assert_all_called=False) as mock:
        mock.post_route = mock.post(GROQ_URL).mock(side_effect=_slow_handler)
        await asyncio.gather(*(client.generate_json("sp", f"up-{i}") for i in range(4)))
        assert peak == 1, f"concurrent LLM calls exceeded the semaphore cap (peak={peak})"


async def test_all_failed_error_survives_empty_exception_messages():
    """str(httpx.ReadTimeout()) is '' — the per-provider detail formatter
    used to crash with IndexError *inside* its own raise, replacing the
    fail-fast AllProvidersFailedError with a retryable IndexError and
    re-running the whole dead chain 3x (observed live on the synthesizer)."""
    from app.core.llm import AllProvidersFailedError

    client = LLMClient(_custom_settings(huggingface_api_key=""))
    with respx.mock(assert_all_called=False) as mock:
        mock.post(CUSTOM_URL).mock(side_effect=httpx.ReadTimeout(""))  # empty message
        mock.post(GROQ_URL).mock(return_value=httpx.Response(402, json={"error": "payment required"}))
        with pytest.raises(AllProvidersFailedError, match="ReadTimeout"):
            await client.generate_json("sp", "up")
