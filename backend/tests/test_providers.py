"""User-managed LLM providers: encrypted store, active exclusivity, routes."""
import pytest
from cryptography.fernet import Fernet

from app.core.llm import AllProvidersFailedError, LLMClient


@pytest.fixture
def _secret(monkeypatch):
    monkeypatch.setenv("MARS_SECRET_KEY", Fernet.generate_key().decode("utf-8"))


@pytest.fixture
def db_path(tmp_path, _secret):
    import asyncio

    from app.db.sqlite import init_db

    path = str(tmp_path / "providers.db")
    asyncio.run(init_db(path))
    return path


def _settings(**over):
    from app.core.config import Settings

    base = {"groq_api_key": "test-key", "_env_file": None}
    base.update(over)
    return Settings(**base)


async def _seed(db_path, name="alpha", url="https://llm.example.com/v1", model="m-1", key="sk-test-key-1234"):
    from app.core import providers as store

    return await store.save_provider(db_path, name=name, base_url=url, model=model, api_key=key)


async def test_crud_masks_key(db_path):
    from app.core import providers as store

    row = await _seed(db_path)
    assert row["key_hint"] == "••••1234"
    listed = await store.list_providers(db_path)
    assert len(listed) == 1
    assert "api_key" not in listed[0] and "api_key_enc" not in listed[0]
    assert listed[0]["has_key"] is True and listed[0]["is_active"] is False

    updated = await store.save_provider(db_path, provider_id=row["id"], name="alpha",
                                        base_url="https://llm.example.com/v2", model="m-2")
    assert updated["base_url"].endswith("/v2") and updated["key_hint"] == "••••1234"

    with pytest.raises(ValueError):
        await _seed(db_path, name="alpha")
    with pytest.raises(ValueError):
        await _seed(db_path, name="bad", url="not-a-url")
    assert await store.delete_provider(db_path, row["id"]) is True
    assert await store.delete_provider(db_path, row["id"]) is False


async def test_active_exclusivity_and_decrypt(db_path):
    from app.core import providers as store

    a = await _seed(db_path, name="alpha")
    b = await _seed(db_path, name="beta", key="sk-other-key-9999")
    assert await store.get_active_provider(db_path) is None
    await store.set_active_provider(db_path, a["id"])
    assert (await store.get_active_provider(db_path))["name"] == "alpha"
    await store.set_active_provider(db_path, b["id"])
    rows = {r["name"]: r["is_active"] for r in await store.list_providers(db_path)}
    assert rows == {"alpha": False, "beta": True}
    active = await store.get_active_provider(db_path)
    assert active["api_key"] == "sk-other-key-9999"
    await store.delete_provider(db_path, b["id"])
    assert await store.get_active_provider(db_path) is None
    with pytest.raises(LookupError):
        await store.set_active_provider(db_path, 424242)


async def test_wrong_secret_cannot_decrypt(db_path, monkeypatch):
    from app.core import providers as store

    row = await _seed(db_path)
    await store.set_active_provider(db_path, row["id"])
    monkeypatch.setenv("MARS_SECRET_KEY", Fernet.generate_key().decode("utf-8"))
    with pytest.raises(ValueError, match="cannot be decrypted"):
        await store.get_active_provider(db_path)


async def test_exclusive_chain_skips_groq(db_path):
    """Active provider set + failing: Groq must not be touched (exclusive)."""
    import httpx
    import respx

    from app.core import providers as store

    await _seed(db_path)
    providers = await store.list_providers(db_path)
    await store.set_active_provider(db_path, providers[0]["id"])
    client = LLMClient(_settings(database_url=db_path))
    with respx.mock(assert_all_called=False) as mock:
        custom_route = mock.post("https://llm.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": "down"}))
        groq_route = mock.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]}))
        with pytest.raises(AllProvidersFailedError, match="Active provider"):
            await client.generate_json("sp", "up")
        assert custom_route.call_count == 4  # tenacity still retries fast 5xx in place
        assert groq_route.call_count == 0


async def test_provider_routes_crud_and_test(tmp_path, _secret):
    """Endpoints: create/list/set-active/test/delete round trip."""
    import httpx
    import respx
    from fastapi import FastAPI

    from app.api.routes import limiter, router as api_router

    from app.db.sqlite import init_db

    db_path = str(tmp_path / "routes.db")
    await init_db(db_path)
    app = FastAPI()
    app.state.settings = _settings(database_url=db_path)
    app.state.limiter = limiter
    app.include_router(api_router, prefix="/api")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = (await client.post("/api/providers", json={
            "name": "pro", "base_url": "https://llm.example.com/v1",
            "api_key": "sk-live-1234", "model": "m-1"})).json()
        assert created["provider"]["key_hint"] == "••••1234"
        assert "api_key" not in created["provider"]
        pid = created["provider"]["id"]

        listed = (await client.get("/api/providers")).json()
        assert listed["active_id"] is None and len(listed["providers"]) == 1

        with respx.mock(assert_all_called=False) as mock:
            mock.post("https://llm.example.com/v1/chat/completions").mock(
                return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
            probed = (await client.post(f"/api/providers/{pid}/test")).json()
        assert probed["ok"] is True and probed["latency_ms"] >= 0

        activated = (await client.post(f"/api/providers/{pid}/active")).json()
        assert activated["active"]["is_active"] is True
        assert (await client.get("/api/providers")).json()["active_id"] == pid

        renamed = (await client.put(f"/api/providers/{pid}", json={"name": "pro2"})).json()
        assert renamed["provider"]["name"] == "pro2"
        assert renamed["provider"]["key_hint"] == "••••1234"

        assert (await client.delete("/api/providers/424242")).status_code == 404
        assert (await client.delete(f"/api/providers/{pid}")).json() == {"deleted": True}
        assert (await client.get("/api/providers")).json() == {"providers": [], "active_id": None}


async def test_probe_timeout_clamped_and_optional(tmp_path, _secret):
    """timeout_sec is optional (default 15s) and clamped to 5..120s."""
    import httpx
    import respx
    from fastapi import FastAPI

    from app.api.routes import limiter, router as api_router

    from app.db.sqlite import init_db

    db_path = str(tmp_path / "timeout.db")
    await init_db(db_path)
    app = FastAPI()
    app.state.settings = _settings(database_url=db_path)
    app.state.limiter = limiter
    app.include_router(api_router, prefix="/api")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        pid = (await client.post("/api/providers", json={
            "name": "slow", "base_url": "https://slow.example.com/v1",
            "api_key": "sk-slow-key", "model": "m-slow"})).json()["provider"]["id"]
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post("https://slow.example.com/v1/chat/completions").mock(
                return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
            defaulted = (await client.post(f"/api/providers/{pid}/test")).json()
            assert defaulted["ok"] is True
            huge = (await client.post(f"/api/providers/{pid}/test", json={"timeout_sec": 9999})).json()
            assert huge["ok"] is True
            assert route.call_count == 2


async def test_breaker_resets_when_active_provider_switches(db_path):
    """Switching the active provider must not inherit the previous
    provider's breaker state: a provider that timed out once would
    otherwise block the freshly-selected healthy one for the cooldown.
    Exercises the real reset inside _generate_with_fallback."""
    import respx
    import httpx

    from app.core import providers as provider_store
    from app.core.llm import LLMClient

    row_a = await provider_store.save_provider(
        db_path, name="prov-a", base_url="https://a.example.com/v1",
        model="model-a", api_key="key-a",
    )
    await provider_store.set_active_provider(db_path, row_a["id"])
    settings = _settings(database_url=db_path)
    client = LLMClient(settings)

    with respx.mock(assert_all_called=False) as mock:
        route_a = mock.post("https://a.example.com/v1/chat/completions").mock(
            side_effect=httpx.ReadTimeout(""))  # empty message: also covers the formatter
        # A times out: exclusive selection fails fast (no env fallback),
        # and the timeout opens A's breaker.
        with pytest.raises(AllProvidersFailedError, match="prov-a"):
            await client.generate_json("sp", "up")
        assert client.custom_breaker.is_open()

        # User switches to provider B. Despite A's open breaker, B must be
        # attempted — the identity switch resets the breaker.
        await provider_store.set_active_provider(db_path, row_a["id"])  # same row replaced below
        await provider_store.save_provider(
            db_path, name="prov-b", base_url="https://b.example.com/v1",
            model="model-b", api_key="key-b",
        )
        rows = await provider_store.list_providers(db_path)
        row_b = next(r for r in rows if r["name"] == "prov-b")
        await provider_store.set_active_provider(db_path, row_b["id"])

        route_b = mock.post("https://b.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "{\"ok\": true}"}}]}))
        result = await client.generate_json("sp", "up")
        assert result == {"ok": True}
        assert route_b.call_count == 1, "switched provider must be attempted despite stale breaker"
        assert route_a.call_count == 1


async def test_probe_all_reports_dead_providers(db_path):
    """Pre-flight probe, env chain (no active provider): all providers
    failing -> (False, per-provider detail); one succeeding -> (True, '')."""
    import respx
    import httpx

    from app.core import providers as provider_store
    from app.core.llm import LLMClient

    settings = _settings(groq_api_key="k", huggingface_api_key="hf", database_url=db_path)
    client = LLMClient(settings)
    assert await provider_store.get_active_provider(db_path) is None

    with respx.mock(assert_all_called=False) as mock:
        mock.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=httpx.Response(429, json={"error": "quota"}))
        mock.post("https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct").mock(
            return_value=httpx.Response(503, json={"error": "loading"}))
        ok, detail = await client.probe_all(timeout=5.0)
    assert ok is False
    assert "groq" in detail and "huggingface" in detail

    with respx.mock(assert_all_called=False) as mock:
        mock.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
        mock.post("https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct").mock(
            return_value=httpx.Response(503, json={"error": "loading"}))
        ok, detail = await client.probe_all(timeout=5.0)
    assert ok is True and detail == ""


async def test_exclusive_probe_pings_active_only(db_path):
    """While a provider is active, the probe must NOT ping Groq/HF —
    selection is exclusive, and pinging irrelevant keys wastes quota."""
    import respx
    import httpx

    from app.core import providers as provider_store
    from app.core.llm import LLMClient

    row = await provider_store.save_provider(
        db_path, name="only", base_url="https://only.example.com/v1",
        model="m", api_key="key",
    )
    await provider_store.set_active_provider(db_path, row["id"])
    settings = _settings(groq_api_key="k", huggingface_api_key="hf", database_url=db_path)
    client = LLMClient(settings)

    with respx.mock(assert_all_called=False) as mock:
        route = mock.post("https://only.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
        groq_route = mock.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
        ok, _ = await client.probe_all(timeout=5.0)
    assert ok is True
    assert route.call_count == 1
    assert groq_route.call_count == 0, "exclusive selection must not ping the env chain"
