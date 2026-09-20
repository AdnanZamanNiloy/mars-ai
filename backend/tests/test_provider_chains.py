"""Provider fallback chains: ordering, failover, classification, persistence.

The chain feature layers an ordered primary→fallback list of EXISTING saved
providers on top of single-provider selection. These tests are the contract:

  1. execution order        — members are tried strictly in configured order
  2. failover               — a transient failure advances to the next member
  3. failure classification — a non-retryable/user-invalid error must NOT
                              advance; provider-availability errors must
  4. state preservation     — the SAME prompt is sent to the fallback member
  5. success accounting     — which provider succeeded + why earlier ones failed
  6. loop/duplicate guard   — a provider cannot appear twice
  7. all-failed             — AllProvidersFailedError names every member
  8. persistence            — chains survive re-reads (reload) in order
  9. single-provider compat — no enabled chain leaves legacy behavior intact

All network is mocked with respx; no real provider is contacted.
"""
import json

import httpx
import pytest
import respx
from cryptography.fernet import Fernet

from app.core import providers as store
from app.core.config import Settings
from app.core.llm import (
    AllProvidersFailedError,
    LLMClient,
    PromptTooLargeError,
    chain_failover_eligible,
)

A_URL = "https://a.example.com/v1/chat/completions"
B_URL = "https://b.example.com/v1/chat/completions"
C_URL = "https://c.example.com/v1/chat/completions"


@pytest.fixture
def _secret(monkeypatch):
    monkeypatch.setenv("MARS_SECRET_KEY", Fernet.generate_key().decode("utf-8"))


@pytest.fixture
def db_path(tmp_path, _secret):
    import asyncio

    from app.db.sqlite import init_db

    path = str(tmp_path / "chains.db")
    asyncio.run(init_db(path))
    return path


def _settings(db_path, **over):
    base = {"groq_api_key": "test-key", "database_url": db_path, "_env_file": None}
    base.update(over)
    return Settings(**base)


def _ok(payload: dict) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(payload)}}]})


async def _provider(db_path, name, url, model=None, key=None):
    return await store.save_provider(
        db_path, name=name, base_url=url, model=model or f"m-{name}",
        api_key=key or f"key-{name}",
    )


async def _chain(db_path, name, providers, enabled=True):
    chain = await store.save_chain(db_path, name=name)
    await store.set_chain_members(db_path, chain["id"], [p["id"] for p in providers])
    if enabled:
        await store.set_chain_enabled(db_path, chain["id"], True)
    return await store.get_chain(db_path, chain["id"])


# ---------------------------------------------------------------------------
# 1. Execution order
# ---------------------------------------------------------------------------

async def test_chain_tried_strictly_in_configured_order(db_path):
    """Member #1 answers → no later member is contacted at all."""
    a = await _provider(db_path, "a", "https://a.example.com/v1")
    b = await _provider(db_path, "b", "https://b.example.com/v1")
    await _chain(db_path, "main", [b, a])  # b is primary despite a being created first
    client = LLMClient(_settings(db_path))

    with respx.mock(assert_all_called=False) as mock:
        rb = mock.post(B_URL).mock(return_value=_ok({"via": "b"}))
        ra = mock.post(A_URL).mock(return_value=_ok({"via": "a"}))
        result = await client.generate_json("sp", "up")
    assert result == {"via": "b"}
    assert rb.call_count == 1
    assert ra.call_count == 0, "later members must not run when the primary succeeds"


async def test_reordering_changes_which_provider_is_primary(db_path):
    a = await _provider(db_path, "a", "https://a.example.com/v1")
    b = await _provider(db_path, "b", "https://b.example.com/v1")
    chain = await _chain(db_path, "main", [a, b])
    # Reorder to b-first and confirm the primary changes.
    await store.reorder_chain_members(db_path, chain["id"], [b["id"], a["id"]])
    client = LLMClient(_settings(db_path))

    with respx.mock(assert_all_called=False) as mock:
        rb = mock.post(B_URL).mock(return_value=_ok({"via": "b"}))
        ra = mock.post(A_URL).mock(return_value=_ok({"via": "a"}))
        result = await client.generate_json("sp", "up")
    assert result == {"via": "b"}
    assert rb.call_count == 1 and ra.call_count == 0


# ---------------------------------------------------------------------------
# 2. Failover on transient/provider failures
# ---------------------------------------------------------------------------

async def test_transient_failure_advances_to_next_provider(db_path):
    a = await _provider(db_path, "a", "https://a.example.com/v1")
    b = await _provider(db_path, "b", "https://b.example.com/v1")
    await _chain(db_path, "main", [a, b])
    client = LLMClient(_settings(db_path))

    with respx.mock(assert_all_called=False) as mock:
        mock.post(A_URL).mock(return_value=httpx.Response(500, json={"error": "down"}))
        rb = mock.post(B_URL).mock(return_value=_ok({"via": "b"}))
        result = await client.generate_json("sp", "up")
    assert result == {"via": "b"}
    assert rb.call_count == 1


async def test_timeout_advances_to_next_provider_once(db_path):
    """A timeout on the primary must fail over after ONE attempt, not retry
    the slow provider (the documented research-budget trap)."""
    a = await _provider(db_path, "a", "https://a.example.com/v1")
    b = await _provider(db_path, "b", "https://b.example.com/v1")
    await _chain(db_path, "main", [a, b])
    client = LLMClient(_settings(db_path))

    with respx.mock(assert_all_called=False) as mock:
        ra = mock.post(A_URL).mock(side_effect=httpx.ReadTimeout("slow"))
        rb = mock.post(B_URL).mock(return_value=_ok({"via": "b"}))
        result = await client.generate_json("sp", "up")
    assert result == {"via": "b"}
    assert ra.call_count == 1, "timed-out chain member must be tried once"
    assert rb.call_count == 1


async def test_three_member_chain_falls_through_two_failures(db_path):
    a = await _provider(db_path, "a", "https://a.example.com/v1")
    b = await _provider(db_path, "b", "https://b.example.com/v1")
    c = await _provider(db_path, "c", "https://c.example.com/v1")
    await _chain(db_path, "main", [a, b, c])
    client = LLMClient(_settings(db_path))

    with respx.mock(assert_all_called=False) as mock:
        mock.post(A_URL).mock(return_value=httpx.Response(503, json={"error": "down"}))
        mock.post(B_URL).mock(return_value=httpx.Response(429, headers={"retry-after": "0"}, json={"e": 1}))
        rc = mock.post(C_URL).mock(return_value=_ok({"via": "c"}))
        result = await client.generate_json("sp", "up")
    assert result == {"via": "c"}
    assert rc.call_count == 1


async def test_hard_provider_error_still_fails_over(db_path):
    """401 on the primary is provider-specific (bad key), so a different
    provider with its own key can serve the call: it must fail over."""
    a = await _provider(db_path, "a", "https://a.example.com/v1")
    b = await _provider(db_path, "b", "https://b.example.com/v1")
    await _chain(db_path, "main", [a, b])
    client = LLMClient(_settings(db_path))

    with respx.mock(assert_all_called=False) as mock:
        ra = mock.post(A_URL).mock(return_value=httpx.Response(401, json={"error": "bad key"}))
        rb = mock.post(B_URL).mock(return_value=_ok({"via": "b"}))
        result = await client.generate_json("sp", "up")
    assert result == {"via": "b"}
    assert ra.call_count == 1, "401 must fail fast, not retry"
    assert rb.call_count == 1


# ---------------------------------------------------------------------------
# 3. Failure classification — non-retryable/user-invalid must NOT advance
# ---------------------------------------------------------------------------

def test_failover_eligible_classification():
    req = httpx.Request("POST", A_URL)

    def status(code):
        return httpx.HTTPStatusError("e", request=req, response=httpx.Response(code, json={}))

    # Provider-availability / transient: eligible.
    assert chain_failover_eligible(httpx.ReadTimeout("slow")) is True
    assert chain_failover_eligible(httpx.ConnectError("down")) is True
    assert chain_failover_eligible(status(429)) is True
    assert chain_failover_eligible(status(500)) is True
    assert chain_failover_eligible(status(401)) is True
    assert chain_failover_eligible(status(402)) is True
    assert chain_failover_eligible(status(404)) is True
    assert chain_failover_eligible(PromptTooLargeError("big")) is True
    # Generic client/user error: NOT eligible — the same payload fails everywhere.
    assert chain_failover_eligible(status(400)) is False
    assert chain_failover_eligible(status(422)) is False
    assert chain_failover_eligible(status(418)) is False
    # Unknown error: conservative — do not burn the rest of the chain.
    assert chain_failover_eligible(ValueError("weird")) is False


async def test_invalid_request_does_not_fail_over(db_path):
    """A generic 400 (malformed user request) must STOP the chain: the same
    payload would be rejected by every member, so advancing only spends quota
    and hides the real cause."""
    a = await _provider(db_path, "a", "https://a.example.com/v1")
    b = await _provider(db_path, "b", "https://b.example.com/v1")
    await _chain(db_path, "main", [a, b])
    client = LLMClient(_settings(db_path))

    with respx.mock(assert_all_called=False) as mock:
        ra = mock.post(A_URL).mock(
            return_value=httpx.Response(400, json={"error": "malformed request body"})
        )
        rb = mock.post(B_URL).mock(return_value=_ok({"via": "b"}))
        with pytest.raises(AllProvidersFailedError, match="not eligible for fallback"):
            await client.generate_json("sp", "up")
    assert ra.call_count >= 1
    assert rb.call_count == 0, "a non-retryable request must not try later providers"


# ---------------------------------------------------------------------------
# 4. State preservation across attempts
# ---------------------------------------------------------------------------

async def test_fallback_attempt_preserves_the_same_request(db_path):
    """The fallback member receives the identical system+user prompt — the
    failover is the same logical request on a different provider."""
    a = await _provider(db_path, "a", "https://a.example.com/v1")
    b = await _provider(db_path, "b", "https://b.example.com/v1")
    await _chain(db_path, "main", [a, b])
    client = LLMClient(_settings(db_path))

    with respx.mock(assert_all_called=False) as mock:
        mock.post(A_URL).mock(return_value=httpx.Response(500, json={"error": "down"}))
        rb = mock.post(B_URL).mock(return_value=_ok({"via": "b"}))
        await client.generate_json("SYS-PROMPT", "USER-PROMPT")
    body = json.loads(rb.calls[0].request.content.decode("utf-8"))
    messages = body["messages"]
    assert {"role": "system", "content": "SYS-PROMPT"} in messages
    assert {"role": "user", "content": "USER-PROMPT"} in messages


# ---------------------------------------------------------------------------
# 5. Success accounting
# ---------------------------------------------------------------------------

async def test_last_chain_run_records_success_and_failure_reasons(db_path):
    a = await _provider(db_path, "a", "https://a.example.com/v1")
    b = await _provider(db_path, "b", "https://b.example.com/v1")
    await _chain(db_path, "main", [a, b])
    client = LLMClient(_settings(db_path))

    with respx.mock(assert_all_called=False) as mock:
        mock.post(A_URL).mock(return_value=httpx.Response(500, json={"error": "down"}))
        mock.post(B_URL).mock(return_value=_ok({"via": "b"}))
        await client.generate_json("sp", "up")
    run = client._last_chain_run
    assert run["succeeded"] == "b"
    assert run["succeeded_index"] == 1
    assert run["total"] == 2
    assert run["failures"][0]["provider"] == "a"
    assert run["failures"][0]["failover"] is True


# ---------------------------------------------------------------------------
# 6. Loop / duplicate protection
# ---------------------------------------------------------------------------

async def test_duplicate_provider_in_chain_is_rejected(db_path):
    a = await _provider(db_path, "a", "https://a.example.com/v1")
    chain = await store.save_chain(db_path, name="dup")
    with pytest.raises(ValueError, match="more than once"):
        await store.set_chain_members(db_path, chain["id"], [a["id"], a["id"]])


async def test_duplicate_member_row_is_rejected_at_the_db_level(db_path):
    """The unique index on (chain_id, provider_id) is the second line of
    defense behind the store's validation: a duplicate insert can never
    reach the runtime, so a chain can never call the same provider twice."""
    import aiosqlite

    a = await _provider(db_path, "a", "https://a.example.com/v1")
    chain = await store.save_chain(db_path, name="legacy")
    await store.set_chain_members(db_path, chain["id"], [a["id"]])
    with pytest.raises(aiosqlite.IntegrityError):
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "INSERT INTO provider_chain_members (chain_id, provider_id, position, created_at)"
                " VALUES (?, ?, 1, 'x')",
                (chain["id"], a["id"]),
            )
            await db.commit()


async def test_resolution_dedupes_member_list(db_path, monkeypatch):
    """Defense in depth: even if a duplicate ever reached get_enabled_chain
    (e.g. a future query change), get_chain_providers must dedupe by provider
    id so the runtime chain cannot loop."""
    a = await _provider(db_path, "a", "https://a.example.com/v1")
    chain = await store.save_chain(db_path, name="legacy")
    await store.set_chain_members(db_path, chain["id"], [a["id"]])
    await store.set_chain_enabled(db_path, chain["id"], True)

    real_get_enabled = store.get_enabled_chain

    async def _duplicated(path):
        found = await real_get_enabled(path)
        # Simulate a corrupt read: the same member listed twice.
        found["members"] = found["members"] + [dict(found["members"][0])]
        return found

    monkeypatch.setattr(store, "get_enabled_chain", _duplicated)
    resolved = await store.get_chain_providers(db_path)
    assert [r["id"] for r in resolved] == [a["id"]], "duplicate members must collapse to one"

    client = LLMClient(_settings(db_path))
    with respx.mock(assert_all_called=False) as mock:
        ra = mock.post(A_URL).mock(return_value=_ok({"ok": True}))
        assert await client.generate_json("sp", "up") == {"ok": True}
    assert ra.call_count == 1, "a deduped chain must not call the same provider twice"


async def test_unknown_provider_id_rejected(db_path):
    chain = await store.save_chain(db_path, name="c1")
    with pytest.raises(LookupError, match="Unknown provider"):
        await store.set_chain_members(db_path, chain["id"], [424242])


# ---------------------------------------------------------------------------
# 7. All providers failed
# ---------------------------------------------------------------------------

async def test_all_chain_members_failed_names_every_provider(db_path):
    a = await _provider(db_path, "a", "https://a.example.com/v1")
    b = await _provider(db_path, "b", "https://b.example.com/v1")
    c = await _provider(db_path, "c", "https://c.example.com/v1")
    await _chain(db_path, "main", [a, b, c])
    client = LLMClient(_settings(db_path))

    with respx.mock(assert_all_called=False) as mock:
        mock.post(A_URL).mock(return_value=httpx.Response(500, json={"error": "down"}))
        mock.post(B_URL).mock(return_value=httpx.Response(503, json={"error": "down"}))
        mock.post(C_URL).mock(return_value=httpx.Response(504, json={"error": "down"}))
        with pytest.raises(AllProvidersFailedError) as exc:
            await client.generate_json("sp", "up")
    message = str(exc.value)
    assert "a:" in message and "b:" in message and "c:" in message
    assert client._last_chain_run["succeeded"] is None
    assert len(client._last_chain_run["failures"]) == 3


async def test_all_members_413_aggregates_to_prompt_too_large(db_path):
    a = await _provider(db_path, "a", "https://a.example.com/v1")
    b = await _provider(db_path, "b", "https://b.example.com/v1")
    await _chain(db_path, "main", [a, b])
    client = LLMClient(_settings(db_path))
    with respx.mock(assert_all_called=False) as mock:
        mock.post(A_URL).mock(return_value=httpx.Response(413, json={"error": "too large"}))
        mock.post(B_URL).mock(return_value=httpx.Response(413, json={"error": "too large"}))
        with pytest.raises(PromptTooLargeError):
            await client.generate_json("sp", "up")


# ---------------------------------------------------------------------------
# 8. Persistence
# ---------------------------------------------------------------------------

async def test_chain_persists_order_and_enabled_state_across_reads(db_path):
    a = await _provider(db_path, "a", "https://a.example.com/v1")
    b = await _provider(db_path, "b", "https://b.example.com/v1")
    c = await _provider(db_path, "c", "https://c.example.com/v1")
    await _chain(db_path, "main", [c, a, b])

    loaded = await store.list_chains(db_path)
    assert len(loaded) == 1
    assert loaded[0]["name"] == "main"
    assert loaded[0]["is_enabled"] is True
    assert [m["name"] for m in loaded[0]["members"]] == ["c", "a", "b"], "order must persist"

    # A fresh read (simulating reload) resolves the same order.
    resolved = await store.get_chain_providers(db_path)
    assert [r["name"] for r in resolved] == ["c", "a", "b"]
    assert await store.get_enabled_chain(db_path) is not None


async def test_only_one_chain_can_be_enabled(db_path):
    a = await _provider(db_path, "a", "https://a.example.com/v1")
    c1 = await _chain(db_path, "one", [a])
    c2 = await _chain(db_path, "two", [a])
    # c2 was enabled last; enabling c1 must disable c2.
    await store.set_chain_enabled(db_path, c1["id"], True)
    chain_one = await store.get_chain(db_path, c1["id"])
    chain_two = await store.get_chain(db_path, c2["id"])
    assert chain_one["is_enabled"] is True
    assert chain_two["is_enabled"] is False
    listing = await store.list_chains(db_path)
    assert sum(1 for c in listing if c["is_enabled"]) == 1


async def test_disabling_chain_restores_single_provider_mode(db_path):
    a = await _provider(db_path, "a", "https://a.example.com/v1")
    b = await _provider(db_path, "b", "https://b.example.com/v1")
    chain = await _chain(db_path, "main", [a, b])
    await store.set_chain_enabled(db_path, chain["id"], False)
    assert await store.get_enabled_chain(db_path) is None
    assert await store.get_chain_providers(db_path) == []


async def test_deleting_provider_cascades_out_of_chain(db_path):
    a = await _provider(db_path, "a", "https://a.example.com/v1")
    b = await _provider(db_path, "b", "https://b.example.com/v1")
    chain = await _chain(db_path, "main", [a, b])
    assert await store.delete_provider(db_path, a["id"]) is True
    await store.remove_provider_from_chains(db_path, a["id"])
    reloaded = await store.get_chain(db_path, chain["id"])
    assert [m["name"] for m in reloaded["members"]] == ["b"]


async def test_chain_rename_and_delete(db_path):
    a = await _provider(db_path, "a", "https://a.example.com/v1")
    chain = await _chain(db_path, "old", [a])
    renamed = await store.save_chain(db_path, chain_id=chain["id"], name="new")
    assert renamed["name"] == "new"
    assert await store.delete_chain(db_path, chain["id"]) is True
    assert await store.get_chain(db_path, chain["id"]) is None


async def test_reorder_requires_exact_current_member_set(db_path):
    a = await _provider(db_path, "a", "https://a.example.com/v1")
    b = await _provider(db_path, "b", "https://b.example.com/v1")
    chain = await _chain(db_path, "main", [a, b])
    # Removing a member through reorder is rejected (use set members).
    with pytest.raises(ValueError, match="exactly the chain's current provider ids"):
        await store.reorder_chain_members(db_path, chain["id"], [a["id"]])


# ---------------------------------------------------------------------------
# 9. Single-provider compatibility
# ---------------------------------------------------------------------------

async def test_no_enabled_chain_uses_legacy_single_provider_path(db_path):
    """With a saved (but disabled) chain and an active provider set, the
    legacy exclusive-active-provider behavior must be unchanged."""
    from app.core import providers as provider_store

    a = await _provider(db_path, "a", "https://a.example.com/v1")
    await _chain(db_path, "main", [a], enabled=False)
    await provider_store.set_active_provider(db_path, a["id"])
    client = LLMClient(_settings(db_path))
    with respx.mock(assert_all_called=False) as mock:
        ra = mock.post(A_URL).mock(return_value=_ok({"ok": True}))
        result = await client.generate_json("sp", "up")
    assert result == {"ok": True} and ra.call_count == 1
    assert client._last_chain_run is None, "no chain ran, so no chain accounting"


async def test_provider_chains_enabled_flag_is_a_kill_switch(db_path):
    """provider_chains_enabled=False ignores a stored+enabled chain and uses
    the active provider instead (the chain data is left intact)."""
    from app.core import providers as provider_store

    a = await _provider(db_path, "a", "https://a.example.com/v1")
    b = await _provider(db_path, "b", "https://b.example.com/v1")
    await _chain(db_path, "main", [b, a], enabled=True)
    await provider_store.set_active_provider(db_path, a["id"])
    client = LLMClient(_settings(db_path, provider_chains_enabled=False))
    with respx.mock(assert_all_called=False) as mock:
        ra = mock.post(A_URL).mock(return_value=_ok({"via": "a"}))
        rb = mock.post(B_URL).mock(return_value=_ok({"via": "b"}))
        result = await client.generate_json("sp", "up")
    assert result == {"via": "a"}
    assert rb.call_count == 0, "disabled chain must not run"
    # Data survives the flag being flipped back on.
    assert await store.get_enabled_chain(db_path) is not None


# ---------------------------------------------------------------------------
# 10. Routes
# ---------------------------------------------------------------------------

async def test_chain_routes_round_trip(tmp_path, _secret):
    import httpx as _httpx
    from fastapi import FastAPI

    from app.api.routes import limiter, router as api_router
    from app.db.sqlite import init_db

    db_path = str(tmp_path / "routes.db")
    await init_db(db_path)
    app = FastAPI()
    app.state.settings = _settings(db_path)
    app.state.limiter = limiter
    app.include_router(api_router, prefix="/api")
    transport = _httpx.ASGITransport(app=app)
    async with _httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        a = (await client.post("/api/providers", json={
            "name": "a", "base_url": "https://a.example.com/v1",
            "api_key": "k-a", "model": "m-a"})).json()["provider"]
        b = (await client.post("/api/providers", json={
            "name": "b", "base_url": "https://b.example.com/v1",
            "api_key": "k-b", "model": "m-b"})).json()["provider"]

        created = (await client.post("/api/provider-chains", json={"name": "prod"})).json()["chain"]
        cid = created["id"]
        assert created["is_enabled"] is False and created["members"] == []

        members = (await client.put(f"/api/provider-chains/{cid}/members",
                                    json={"provider_ids": [a["id"], b["id"]]})).json()["chain"]
        assert [m["provider_id"] for m in members["members"]] == [a["id"], b["id"]]

        reordered = (await client.post(f"/api/provider-chains/{cid}/reorder",
                                       json={"provider_ids": [b["id"], a["id"]]})).json()["chain"]
        assert [m["provider_id"] for m in reordered["members"]] == [b["id"], a["id"]]

        enabled = (await client.post(f"/api/provider-chains/{cid}/enabled", json={"enabled": True})).json()["chain"]
        assert enabled["is_enabled"] is True
        assert (await client.get("/api/provider-chains")).json()["enabled_id"] == cid

        renamed = (await client.put(f"/api/provider-chains/{cid}", json={"name": "prod2"})).json()["chain"]
        assert renamed["name"] == "prod2"

        # Duplicate members rejected with 422.
        dup = await client.put(f"/api/provider-chains/{cid}/members",
                               json={"provider_ids": [a["id"], a["id"]]})
        assert dup.status_code == 422

        # Clear returns to single-provider mode.
        await client.post("/api/provider-chains/clear")
        assert (await client.get("/api/provider-chains")).json()["enabled_id"] is None

        assert (await client.delete("/api/provider-chains/424242")).status_code == 404
        assert (await client.delete(f"/api/provider-chains/{cid}")).json() == {"deleted": True}


async def test_provider_delete_route_cascades_chain_membership(tmp_path, _secret):
    import httpx as _httpx
    from fastapi import FastAPI

    from app.api.routes import limiter, router as api_router
    from app.db.sqlite import init_db

    db_path = str(tmp_path / "cascade.db")
    await init_db(db_path)
    app = FastAPI()
    app.state.settings = _settings(db_path)
    app.state.limiter = limiter
    app.include_router(api_router, prefix="/api")
    transport = _httpx.ASGITransport(app=app)
    async with _httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        a = (await client.post("/api/providers", json={
            "name": "a", "base_url": "https://a.example.com/v1",
            "api_key": "k-a", "model": "m-a"})).json()["provider"]
        b = (await client.post("/api/providers", json={
            "name": "b", "base_url": "https://b.example.com/v1",
            "api_key": "k-b", "model": "m-b"})).json()["provider"]
        cid = (await client.post("/api/provider-chains", json={"name": "c"})).json()["chain"]["id"]
        await client.put(f"/api/provider-chains/{cid}/members", json={"provider_ids": [a["id"], b["id"]]})

        await client.delete(f"/api/providers/{a['id']}")
        chain = next(c for c in (await client.get("/api/provider-chains")).json()["chains"] if c["id"] == cid)
        assert [m["provider_id"] for m in chain["members"]] == [b["id"]], (
            "deleting a provider must remove it from every chain"
        )
