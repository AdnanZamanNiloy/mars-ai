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
