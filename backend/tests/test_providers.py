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


async def test_active_provider_fallback_rescues_the_run(db_path, monkeypatch):
    """ACTIVE_PROVIDER_FALLBACK=true: the UI-selected provider stays primary,
    but when it fails the env chain (Groq) serves the call instead of the run
    degrading to deterministic extraction."""
    import httpx
    import respx

    from app.core import providers as store
    from app.core.config import Settings
    from app.core.llm import LLMClient

    await _seed(db_path)
    providers = await store.list_providers(db_path)
    await store.set_active_provider(db_path, providers[0]["id"])
    settings = Settings(groq_api_key="test-key", database_url=db_path,
                        active_provider_fallback=True, _env_file=None)
    client = LLMClient(settings)

    with respx.mock(assert_all_called=False) as mock:
        custom_route = mock.post("https://llm.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": "down"}))
        groq_route = mock.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]}))
        result = await client.generate_json("sp", "up")
        assert result == {"ok": True}
        assert custom_route.call_count >= 1
        assert groq_route.call_count == 1


async def test_exclusive_probe_covers_env_fallbacks_when_enabled(db_path):
    """With active_provider_fallback on, the pre-flight probe must ping the
    active provider AND the env fallbacks — the probe's job is to detect a
    doomed chain, and the chain now includes Groq/HF."""
    import httpx
    import respx

    from app.core.config import Settings
    from app.core import providers as store
    from app.core.llm import LLMClient

    row = await store.save_provider(
        db_path, name="only2", base_url="https://only2.example.com/v1",
        model="m", api_key="key",
    )
    await store.set_active_provider(db_path, row["id"])
    settings = Settings(
        groq_api_key="k", huggingface_api_key="hf",
        active_provider_fallback=True, database_url=db_path, _env_file=None,
    )
    client = LLMClient(settings)
    with respx.mock(assert_all_called=False) as mock:
        mock.post("https://only2.example.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
        mock.post("https://api.groq.com/openai/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}))
        ok, _ = await client.probe_all(timeout=5.0)
    assert ok is True


# ---------------------------------------------------------------------------
# Regression: the orphaned `model_name` column broke /api/providers with 500.
#
# The persisted table had a `model_name` column and every row stored '' for it,
# but `_public()` never selected it and no migration/INSERT/UPDATE mentioned it.
# The store resolves at boot against whatever column set the on-disk database
# actually has, so a database written by the build that added the column could
# never be read back by the build that did not know about it.
# ---------------------------------------------------------------------------


async def test_reads_provider_rows_written_with_model_name_column(db_path):
    """A row carrying model_name must serialize, not raise. `model_name` blank
    resolves to the model id so the wire shape is always a usable label."""
    import aiosqlite

    from app.core import providers as store

    await _seed(db_path, name="alpha")
    async with aiosqlite.connect(db_path) as raw:
        await raw.execute(
            "UPDATE llm_providers SET model_name = ? WHERE name = ?",
            ("GPT-4o mini", "alpha"),
        )
        await raw.commit()

    rows = await store.list_providers(db_path)
    row = next(r for r in rows if r["name"] == "alpha")
    assert row["model_name"] == "GPT-4o mini"

    # The column is present in the fresh schema, so a blank label is the
    # documented "no separate label" state and must fall back to the id.
    async with aiosqlite.connect(db_path) as raw:
        await raw.execute("UPDATE llm_providers SET model_name = '' WHERE name = ?", ("alpha",))
        await raw.commit()
    row = next(r for r in await store.list_providers(db_path) if r["name"] == "alpha")
    assert row["model_name"] == row["model"]


async def test_fresh_schema_carries_the_model_name_column(db_path):
    """init_db must create the column outright, and the ALTER path must still
    converge an older database onto the same shape."""
    import aiosqlite

    from app.db.sqlite import init_db

    async with aiosqlite.connect(db_path) as db:
        cols = {r[1] for r in await (await db.execute("PRAGMA table_info(llm_providers)")).fetchall()}
    assert "model_name" in cols, "fresh databases must include model_name"

    # Simulate a database created before the column existed by rebuilding the
    # table without it, then re-running the migration.
    async with aiosqlite.connect(db_path) as db:
        await db.execute("ALTER TABLE llm_providers RENAME TO _old")
        await db.execute(
            """CREATE TABLE llm_providers (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE,
                base_url TEXT NOT NULL, api_key_enc TEXT NOT NULL,
                key_hint TEXT NOT NULL DEFAULT '', model TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"""
        )
        await db.execute(
            "INSERT INTO llm_providers (name, base_url, api_key_enc, key_hint, model, is_active, created_at, updated_at)"
            " SELECT name, base_url, api_key_enc, key_hint, model, is_active, created_at, updated_at FROM _old"
        )
        await db.execute("DROP TABLE _old")
        await db.commit()
        cols = {r[1] for r in await (await db.execute("PRAGMA table_info(llm_providers)")).fetchall()}
        assert "model_name" not in cols, "precondition: the legacy shape has no model_name"

    await init_db(db_path)

    async with aiosqlite.connect(db_path) as db:
        cols = {r[1] for r in await (await db.execute("PRAGMA table_info(llm_providers)")).fetchall()}
    assert "model_name" in cols, "init_db must migrate a legacy table forward"


async def test_model_name_round_trips_and_is_independent_of_model(db_path):
    """Model name is a display label; Model ID is what goes to the provider.
    They must be independently settable, and an omitted label on partial
    update must preserve the stored one rather than blanking it."""
    from app.core import providers as store

    created = await store.save_provider(
        db_path, name="labelled", base_url="https://llm.example.com/v1",
        model="gpt-4o-mini", api_key="sk-live-1234", model_name="GPT-4o mini",
    )
    assert created["model"] == "gpt-4o-mini"
    assert created["model_name"] == "GPT-4o mini"

    # Omitting model_name (None) keeps the stored label.
    updated = await store.save_provider(
        db_path, provider_id=created["id"], name="labelled",
        base_url="https://llm.example.com/v1", model="gpt-4o-mini",
    )
    assert updated["model_name"] == "GPT-4o mini", "an omitted label must be preserved"

    # An explicit blank clears the label back to the model id.
    cleared = await store.save_provider(
        db_path, provider_id=created["id"], name="labelled",
        base_url="https://llm.example.com/v1", model="gpt-4o-mini", model_name="",
    )
    assert cleared["model_name"] == "gpt-4o-mini"


async def test_provider_routes_accept_model_name(db_path):
    """/api/providers must round-trip model_name end to end: the 500 this
    guards against surfaced on exactly this route."""
    import httpx
    from fastapi import FastAPI

    from app.api.routes import limiter, router as api_router

    app = FastAPI()
    app.state.settings = _settings(database_url=db_path)
    app.state.limiter = limiter
    app.include_router(api_router, prefix="/api")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/providers", json={
            "name": "seeded", "base_url": "https://seeded.example.com/v1",
            "api_key": "sk-seeded-7777", "model": "seeded-model",
        })
        listed = await client.get("/api/providers")
        assert listed.status_code == 200, "GET /api/providers must not 500"
        assert listed.json()["providers"], "the seeded provider must be readable"

        created = await client.post("/api/providers", json={
            "name": "routed", "base_url": "https://routed.example.com/v1",
            "api_key": "sk-routed-9999", "model": "routed-model",
            "model_name": "Routed Model Label",
        })
        assert created.status_code == 201, created.text
        assert created.json()["provider"]["model_name"] == "Routed Model Label"

        # A legacy client that omits the field entirely still succeeds.
        legacy = await client.post("/api/providers", json={
            "name": "legacy", "base_url": "https://legacy.example.com/v1",
            "api_key": "sk-legacy-8888", "model": "legacy-model",
        })
        assert legacy.status_code == 201, legacy.text
        assert legacy.json()["provider"]["model_name"] == "legacy-model"

        renamed = await client.put(
            f"/api/providers/{created.json()['provider']['id']}",
            json={"model_name": "Renamed Label"},
        )
        assert renamed.status_code == 200, renamed.text
        assert renamed.json()["provider"]["model_name"] == "Renamed Label"
