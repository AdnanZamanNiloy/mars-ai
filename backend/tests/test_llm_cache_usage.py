"""LLM response cache + per-run usage ledger."""

import json

import httpx
import respx

from app.core import llm_cache
from app.core.config import Settings
from app.core.llm import LLMClient
from app.core.usage import clear_run_usage, get_run_usage, start_run_usage

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _settings(tmp_path, **kw) -> Settings:
    base = dict(
        groq_api_key="test-key",
        database_url=str(tmp_path / "ledger.db"),
        _env_file=None,
    )
    base.update(kw)
    return Settings(**base)


def _groq_response(content: str, usage: dict | None = None) -> httpx.Response:
    body = {"choices": [{"message": {"content": content}}]}
    if usage:
        body["usage"] = usage
    return httpx.Response(200, json=body)


async def test_cache_roundtrip_and_hit_skips_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_cache, "_force_disabled", False)
    llm_cache.set_enabled(True)
    llm_cache.configure(_settings(tmp_path))
    llm_cache.clear()
    client = LLMClient(_settings(tmp_path))

    with respx.mock(assert_all_called=False) as mock:
        mock.post(GROQ_URL).mock(return_value=_groq_response(
            json.dumps({"answer": 1}), usage={"prompt_tokens": 10, "completion_tokens": 5}))

        first = await client.generate_json("sys", "what is the answer?")
        assert first == {"answer": 1}
        assert mock.post(GROQ_URL).call_count == 1

        # Second identical call must NOT hit the provider.
        second = await client.generate_json("sys", "what is the answer?")
        assert second == {"answer": 1}
        assert mock.post(GROQ_URL).call_count == 1  # unchanged

    llm_cache.clear()


async def test_cache_miss_on_different_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_cache, "_force_disabled", False)
    llm_cache.set_enabled(True)
    settings = _settings(tmp_path)
    llm_cache.configure(settings)
    llm_cache.clear()
    client = LLMClient(settings)

    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(GROQ_URL).mock(return_value=_groq_response(json.dumps({"a": 1})))
        await client.generate_json("sys", "question one")
        await client.generate_json("sys", "question two — different")
        assert route.call_count == 2

    llm_cache.clear()


async def test_usage_ledger_records_tokens_and_cost(tmp_path):
    settings = _settings(tmp_path)
    client = LLMClient(settings)

    with respx.mock(assert_all_called=False) as mock:
        mock.post(GROQ_URL).mock(return_value=_groq_response(
            json.dumps({"ok": True}),
            usage={"prompt_tokens": 120, "completion_tokens": 30}))

        usage = start_run_usage("req-1", settings, mode="standard")
        from app.core.usage import set_stage_hint
        set_stage_hint("planner")
        await client.generate_json("sys", "plan this")
        clear_run_usage()

    snap = usage.snapshot()
    assert snap["llm_calls"] == 1
    assert snap["spent_tokens"] == 150
    assert snap["spent_usd"] > 0
    assert snap["cache_hits"] == 0
    assert snap["by_stage"]["planner"]["tokens"] == 150


async def test_cache_hits_refund_usd_but_count_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_cache, "_force_disabled", False)
    llm_cache.set_enabled(True)
    settings = _settings(tmp_path)
    llm_cache.configure(settings)
    llm_cache.clear()
    client = LLMClient(settings)

    with respx.mock(assert_all_called=False) as mock:
        mock.post(GROQ_URL).mock(return_value=_groq_response(
            json.dumps({"v": 7}), usage={"prompt_tokens": 100, "completion_tokens": 50}))

        usage = start_run_usage("req-2", settings)
        await client.generate_json("sys", "same prompt")
        spent_after_real_call = usage.budget.spent_usd
        assert spent_after_real_call > 0

        await client.generate_json("sys", "same prompt")  # cache hit
        clear_run_usage()

    snap = usage.snapshot()
    assert snap["cache_hits"] == 1
    assert snap["cache_misses"] == 1
    # USD refunded on the hit; tokens still counted.
    assert snap["spent_usd"] == round(spent_after_real_call, 6)
    assert snap["spent_tokens"] == 300
    assert snap["llm_calls"] == 2

    llm_cache.clear()


async def test_no_ledger_outside_run(tmp_path):
    """Calls outside a research request (probes, scripts) record nothing."""
    settings = _settings(tmp_path)
    client = LLMClient(settings)
    assert get_run_usage() is None
    with respx.mock(assert_all_called=False) as mock:
        mock.post(GROQ_URL).mock(return_value=_groq_response(json.dumps({"x": 1})))
        await client.generate_json("sys", "probe")
    assert get_run_usage() is None


async def test_ledger_feeds_budget_exhaustion(tmp_path):
    settings = _settings(tmp_path, max_llm_calls=2)
    client = LLMClient(settings)

    with respx.mock(assert_all_called=False) as mock:
        mock.post(GROQ_URL).mock(return_value=_groq_response(
            json.dumps({"n": 1}), usage={"prompt_tokens": 5, "completion_tokens": 5}))

        usage = start_run_usage("req-3", settings)
        await client.generate_json("sys", "call one")
        await client.generate_json("sys", "call two")
        clear_run_usage()

    assert usage.budget.llm_calls == 2
    assert usage.exhausted  # call ceiling reached
    assert not usage.budget.can_afford(calls=1)
