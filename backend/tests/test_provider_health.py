"""Probe accuracy for the active provider and active_provider_fallback.

Regression focus: a healthy env provider must never mask a failing active
provider's pre-flight health (a run would start optimistically against a
rate-limited primary and degrade). And ACTIVE_PROVIDER_FALLBACK must be the
only thing that lets the env chain rescue a failing active provider.
"""
import pytest
from cryptography.fernet import Fernet

from app.core.llm import AllProvidersFailedError, LLMClient

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
ACTIVE_URL = "https://llm.example.com/v1/chat/completions"


@pytest.fixture
def _secret(monkeypatch):
    monkeypatch.setenv("MARS_SECRET_KEY", Fernet.generate_key().decode("utf-8"))


@pytest.fixture
def db_path(tmp_path, _secret):
    import asyncio

    from app.db.sqlite import init_db

    path = str(tmp_path / "provider_health.db")
    asyncio.run(init_db(path))
    return path


def _settings(**over):
    from app.core.config import Settings

    base = {"groq_api_key": "test-key", "huggingface_api_key": "test-hf-key",
            "_env_file": None}
    base.update(over)
    return Settings(**base)


async def _activate_provider(db_path):
    from app.core import providers as store

    row = await store.save_provider(
        db_path, name="alpha", base_url="https://llm.example.com/v1",
        model="m-1", api_key="sk-test-key-1234",
    )
    await store.set_active_provider(db_path, row["id"])
    return row


async def test_probe_all_reports_active_failure_despite_healthy_env(db_path):
    """Exclusive active provider 429s while Groq answers 200: probe_all must
    return ok=False and name the active provider — the env 200 must NOT mask
    the primary's failure (default strict exclusivity)."""
    import httpx
    import respx

    await _activate_provider(db_path)
    client = LLMClient(_settings(database_url=db_path))

    with respx.mock(assert_all_called=False) as mock:
        active_route = mock.post(ACTIVE_URL).mock(
            return_value=httpx.Response(429, json={"error": "rate limited"}))
        groq_route = mock.post(GROQ_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
        ok, detail = await client.probe_all(timeout=5.0)

    assert ok is False
    assert "alpha" in detail, "detail must name the active provider"
    assert "429" in detail or "rate limited" in detail or "HTTPStatusError" in detail
    assert active_route.call_count == 1
    # Strict exclusivity: env providers are not even probed, let alone allowed
    # to mask the active provider's failure.
    assert groq_route.call_count == 0


async def test_probe_all_env_chain_any_ok(db_path):
    """No active provider: the existing any-ok semantics hold — one healthy
    env provider is enough for the pre-flight to pass."""
    import httpx
    import respx

    from app.core import providers as store

    client = LLMClient(_settings(database_url=db_path))
    assert await store.get_active_provider(db_path) is None

    with respx.mock(assert_all_called=False) as mock:
        mock.post(GROQ_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
        mock.post("https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct").mock(
            return_value=httpx.Response(503, json={"error": "loading"}))
        ok, detail = await client.probe_all(timeout=5.0)

    assert ok is True and detail == ""


async def test_active_fallback_true_rescues_with_env_provider(db_path):
    """Active provider fails + active_provider_fallback=True: the env chain
    is attempted (not zeroed) and a real model answer is returned."""
    import httpx
    import respx

    await _activate_provider(db_path)
    client = LLMClient(_settings(database_url=db_path, active_provider_fallback=True))

    with respx.mock(assert_all_called=False) as mock:
        active_route = mock.post(ACTIVE_URL).mock(
            return_value=httpx.Response(429, json={"error": "rate limited"}))
        groq_route = mock.post(GROQ_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]}))
        result = await client.generate_json("sp", "up")

    assert result == {"ok": True}
    assert active_route.call_count >= 1
    assert groq_route.call_count == 1


async def test_active_fallback_false_is_strict_all_providers_failed(db_path):
    """Active provider fails + active_provider_fallback=False (default): the
    env chain is zeroed — Groq is never spent and the chain fails fast."""
    import httpx
    import respx

    await _activate_provider(db_path)
    client = LLMClient(_settings(database_url=db_path, active_provider_fallback=False))

    with respx.mock(assert_all_called=False) as mock:
        active_route = mock.post(ACTIVE_URL).mock(
            return_value=httpx.Response(429, json={"error": "rate limited"}))
        groq_route = mock.post(GROQ_URL).mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]}))
        with pytest.raises(AllProvidersFailedError, match="Active provider"):
            await client.generate_json("sp", "up")

    assert active_route.call_count >= 1
    assert groq_route.call_count == 0, "strict exclusivity must not spend the env key"
