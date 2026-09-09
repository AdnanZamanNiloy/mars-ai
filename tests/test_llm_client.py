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


async def test_persistent_groq_failure_falls_back_to_hf(client):
    with respx.mock(assert_all_called=False) as mock:
        mock.post(GROQ_URL).mock(return_value=httpx.Response(500, json={"error": "down"}))
        mock.post(HF_URL).mock(return_value=httpx.Response(200, json=[{"generated_text": json.dumps({"via": "hf"})}]))
        result = await client.generate_json("sp", "up")
        assert result == {"via": "hf"}
        assert client.groq_breaker._consecutive_failures >= 1


async def test_open_breaker_skips_groq_entirely(test_settings):
    client = LLMClient(test_settings)
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
