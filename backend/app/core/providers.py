"""User-managed LLM providers (sidebar Providers tab).

Multiple OpenAI-compatible providers can be stored; exactly one may be
active at a time. The active provider is the ONLY model the research
pipeline uses — no Groq/HuggingFace fallback runs while one is selected.
That exclusivity is deliberate: selection is an explicit user choice, so a
failing active provider degrades the run instead of silently spending a
different key. With no active provider, the legacy chain applies
(env CUSTOM_LLM_* trio, then Groq, then HuggingFace).

Provider fallback chains (optional, additive) layer an ORDERED list of
saved providers on top of that single-selection behavior: chain member #1
is the primary, members #2..N are fallbacks tried in order. A chain
references existing providers by id; it never duplicates provider config or
keys. At most one chain is enabled at a time (enabling one disables the
others). With no enabled chain, resolution is byte-for-byte the legacy
single-active-provider behavior — single-provider selection stays compatible.

API keys are encrypted at rest (Fernet). Key resolution order:
MARS_SECRET_KEY env var, else backend/.mars_secret (0600, auto-created on
first encrypt). The plaintext key only ever leaves this module inside
get_active_provider()/get_chain_providers()/get_provider_secret() (all
server-side) — list and CRUD responses carry a last-4 hint, never the key.
"""

from __future__ import annotations

import base64
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite
from cryptography.fernet import Fernet, InvalidToken


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _secret_file_path() -> Path:
    return Path(__file__).resolve().parent.parent / ".mars_secret"


def _fernet_key() -> bytes:
    """Fernet key: real env var first, then Settings (.env file), else the
    local key file (created once, 0600). An arbitrary secret string is
    hashed into key shape; a proper Fernet key is used as-is. Tests pin
    MARS_SECRET_KEY via env to stay hermetic (never touches the file).
    """
    raw = (os.environ.get("MARS_SECRET_KEY") or "").strip()
    if not raw:
        try:
            from app.core.config import get_settings  # no cycle: config is leaf

            raw = (get_settings().mars_secret_key or "").strip()
        except Exception:
            raw = ""
    if raw:
        try:
            Fernet(raw.encode("utf-8"))
            return raw.encode("utf-8")
        except Exception:
            pass
        digest = hashlib.sha256(raw.encode("utf-8")).digest()
        return base64.urlsafe_b64encode(digest)
    path = _secret_file_path()
    if path.exists():
        return path.read_bytes().strip()
    key = Fernet.generate_key()
    path.write_bytes(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key


def encrypt_api_key(plaintext: str) -> str:
    return Fernet(_fernet_key()).encrypt((plaintext or "").encode("utf-8")).decode("utf-8")


def decrypt_api_key(token: str) -> str:
    try:
        return Fernet(_fernet_key()).decrypt((token or "").encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError(
            "Stored provider key cannot be decrypted with the current secret. "
            "If MARS_SECRET_KEY changed, re-enter the key."
        ) from exc


def _validate(name: str, base_url: str, model: str) -> tuple[str, str, str]:
    name = str(name or "").strip()[:60]
    base_url = str(base_url or "").strip().rstrip("/")[:500]
    model = str(model or "").strip()[:200]
    if not name:
        raise ValueError("Provider name is required.")
    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        raise ValueError("Base URL must start with http:// or https://.")
    if not model:
        raise ValueError("Model ID is required.")
    return name, base_url, model


def _public(row: Any) -> Dict[str, Any]:
    """Wire shape: everything except the key (hint only)."""
    return {
        "id": row["id"],
        "name": row["name"],
        "base_url": row["base_url"],
        "model": row["model"],
        "is_active": bool(row["is_active"]),
        "has_key": bool(row["api_key_enc"]),
        "key_hint": row["key_hint"] or "",
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


async def list_providers(database_path: str) -> List[Dict[str, Any]]:
    async with aiosqlite.connect(database_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM llm_providers ORDER BY name")
        return [_public(dict(r)) for r in await cur.fetchall()]


async def get_provider(database_path: str, provider_id: int) -> Optional[Dict[str, Any]]:
    async with aiosqlite.connect(database_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM llm_providers WHERE id = ?", (int(provider_id),))
        row = await cur.fetchone()
        return _public(dict(row)) if row else None


async def save_provider(
    database_path: str,
    *,
    provider_id: Optional[int] = None,
    name: str,
    base_url: str,
    model: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Insert (provider_id None) or update. api_key None on update keeps the
    stored key; empty string on insert is rejected. Name must stay unique."""
    name, base_url, model = _validate(name, base_url, model)
    async with aiosqlite.connect(database_path) as db:
        db.row_factory = aiosqlite.Row
        if provider_id is None:
            key = str(api_key or "").strip()
            if not key:
                raise ValueError("API key is required for a new provider.")
            if len(key) > 2000:
                raise ValueError("API key is too long.")
            hint = f"••••{key[-4:]}"
            try:
                cur = await db.execute(
                    "INSERT INTO llm_providers (name, base_url, api_key_enc, key_hint, model, is_active, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
                    (name, base_url, encrypt_api_key(key), hint, model, _utcnow(), _utcnow()),
                )
            except Exception as exc:
                if "UNIQUE" in str(exc).upper():
                    raise ValueError(f"A provider named '{name}' already exists.") from exc
                raise
            new_id = cur.lastrowid
        else:
            cur = await db.execute("SELECT * FROM llm_providers WHERE id = ?", (int(provider_id),))
            existing = await cur.fetchone()
            if not existing:
                raise LookupError(f"Unknown provider id: {provider_id}")
            clash = await db.execute(
                "SELECT id FROM llm_providers WHERE name = ? AND id != ?",
                (name, int(provider_id)),
            )
            if await clash.fetchone():
                raise ValueError(f"A provider named '{name}' already exists.")
            if api_key is None:
                enc, hint = existing["api_key_enc"], existing["key_hint"]
            else:
                key = str(api_key).strip()
                if not key:
                    raise ValueError("API key must not be blank (omit it to keep the stored key).")
                if len(key) > 2000:
                    raise ValueError("API key is too long.")
                enc, hint = encrypt_api_key(key), f"••••{key[-4:]}"
            await db.execute(
                "UPDATE llm_providers SET name = ?, base_url = ?, api_key_enc = ?, key_hint = ?, model = ?, updated_at = ?"
                " WHERE id = ?",
                (name, base_url, enc, hint, model, _utcnow(), int(provider_id)),
            )
            new_id = int(provider_id)
        await db.commit()
        cur = await db.execute("SELECT * FROM llm_providers WHERE id = ?", (new_id,))
        return _public(dict(await cur.fetchone()))


async def delete_provider(database_path: str, provider_id: int) -> bool:
    async with aiosqlite.connect(database_path) as db:
        cur = await db.execute("DELETE FROM llm_providers WHERE id = ?", (int(provider_id),))
        await db.commit()
        return (cur.rowcount or 0) > 0


async def set_active_provider(database_path: str, provider_id: int) -> Dict[str, Any]:
    """Serving mode = Single Model. Exactly one active provider, and this is
    mutually exclusive with an enabled fallback chain: setting a single model
    disables any enabled chain in the SAME transaction. Enforced here (not
    only in the route) so no code path can leave both modes active."""
    async with aiosqlite.connect(database_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM llm_providers WHERE id = ?", (int(provider_id),))
        row = await cur.fetchone()
        if not row:
            raise LookupError(f"Unknown provider id: {provider_id}")
        await db.execute("UPDATE llm_providers SET is_active = 0")
        await db.execute(
            "UPDATE llm_providers SET is_active = 1, updated_at = ? WHERE id = ?",
            (_utcnow(), int(provider_id)),
        )
        # Mutual exclusion: a single active model turns off every chain.
        await db.execute("UPDATE provider_chains SET is_enabled = 0")
        await db.commit()
        cur = await db.execute("SELECT * FROM llm_providers WHERE id = ?", (int(provider_id),))
        return _public(dict(await cur.fetchone()))


async def clear_active_provider(database_path: str) -> None:
    async with aiosqlite.connect(database_path) as db:
        await db.execute("UPDATE llm_providers SET is_active = 0")
        await db.commit()


async def get_active_provider(database_path: str) -> Optional[Dict[str, Any]]:
    """Server-side only: full secret row for the LLM chain. Never serialize
    this to the UI — list/get deliberately exclude the key."""
    async with aiosqlite.connect(database_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM llm_providers WHERE is_active = 1 ORDER BY id LIMIT 1")
        row = await cur.fetchone()
        if not row:
            return None
        data = dict(row)
    return {
        "id": data["id"],
        "name": data["name"],
        "base_url": data["base_url"],
        "api_key": decrypt_api_key(data["api_key_enc"]),
        "model": data["model"],
    }


async def get_provider_secret(database_path: str, provider_id: int) -> Optional[Dict[str, Any]]:
    """Full secret row for one provider (connection-test probe). Same
    never-serialize rule as get_active_provider."""
    async with aiosqlite.connect(database_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM llm_providers WHERE id = ?", (int(provider_id),))
        row = await cur.fetchone()
        if not row:
            return None
        data = dict(row)
    return {
        "id": data["id"],
        "name": data["name"],
        "base_url": data["base_url"],
        "api_key": decrypt_api_key(data["api_key_enc"]),
        "model": data["model"],
    }


# ======================================================================
# Provider fallback chains
# ======================================================================

# Bounded, like every other user-authored list in this project (AGENTS.md
# §4.3, §5): an unbounded count is a slower way to hit the same OOM a
# module-level dict once did.
MAX_PROVIDER_CHAINS = 20
MAX_CHAIN_MEMBERS = 12


def _validate_chain_name(name: str) -> str:
    name = str(name or "").strip()[:60]
    if not name:
        raise ValueError("Chain name is required.")
    return name


def _chain_public(row: Any, members: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "is_enabled": bool(row["is_enabled"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "members": members,
    }


async def _load_chain_members(db: Any, chain_id: int) -> List[Dict[str, Any]]:
    """Ordered members (position asc, id tiebreak) joined to provider
    metadata. Provider rows are read with `*` but only masked fields are
    returned — the key never reaches the wire."""
    cur = await db.execute(
        "SELECT m.id AS member_id, m.position AS position, p.* "
        "FROM provider_chain_members m JOIN llm_providers p ON p.id = m.provider_id "
        "WHERE m.chain_id = ? ORDER BY m.position ASC, m.id ASC",
        (int(chain_id),),
    )
    members: List[Dict[str, Any]] = []
    for r in await cur.fetchall():
        data = dict(r)
        members.append({
            "member_id": data["member_id"],
            "position": int(data["position"]),
            "provider_id": data["id"],
            "name": data["name"],
            "base_url": data["base_url"],
            "model": data["model"],
            "has_key": bool(data["api_key_enc"]),
            "key_hint": data["key_hint"] or "",
        })
    return members


async def list_chains(database_path: str) -> List[Dict[str, Any]]:
    """All chains with ordered members. Enabled chain first, then by name."""
    async with aiosqlite.connect(database_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM provider_chains ORDER BY is_enabled DESC, name ASC, id ASC"
        )
        rows = [dict(r) for r in await cur.fetchall()]
        return [_chain_public(r, await _load_chain_members(db, r["id"])) for r in rows]


async def get_chain(database_path: str, chain_id: int) -> Optional[Dict[str, Any]]:
    async with aiosqlite.connect(database_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM provider_chains WHERE id = ?", (int(chain_id),))
        row = await cur.fetchone()
        if not row:
            return None
        return _chain_public(dict(row), await _load_chain_members(db, int(chain_id)))


async def save_chain(
    database_path: str,
    *,
    chain_id: Optional[int] = None,
    name: str,
) -> Dict[str, Any]:
    """Insert (chain_id None) or rename an existing chain. Name is unique."""
    name = _validate_chain_name(name)
    async with aiosqlite.connect(database_path) as db:
        db.row_factory = aiosqlite.Row
        if chain_id is None:
            count_cur = await db.execute("SELECT COUNT(*) FROM provider_chains")
            (count,) = await count_cur.fetchone()
            if int(count) >= MAX_PROVIDER_CHAINS:
                raise ValueError(f"At most {MAX_PROVIDER_CHAINS} chains are allowed.")
            try:
                cur = await db.execute(
                    "INSERT INTO provider_chains (name, is_enabled, created_at, updated_at)"
                    " VALUES (?, 0, ?, ?)",
                    (name, _utcnow(), _utcnow()),
                )
            except Exception as exc:
                if "UNIQUE" in str(exc).upper():
                    raise ValueError(f"A chain named '{name}' already exists.") from exc
                raise
            new_id = cur.lastrowid
        else:
            cur = await db.execute("SELECT * FROM provider_chains WHERE id = ?", (int(chain_id),))
            if not await cur.fetchone():
                raise LookupError(f"Unknown chain id: {chain_id}")
            clash = await db.execute(
                "SELECT id FROM provider_chains WHERE name = ? AND id != ?",
                (name, int(chain_id)),
            )
            if await clash.fetchone():
                raise ValueError(f"A chain named '{name}' already exists.")
            await db.execute(
                "UPDATE provider_chains SET name = ?, updated_at = ? WHERE id = ?",
                (name, _utcnow(), int(chain_id)),
            )
            new_id = int(chain_id)
        await db.commit()
        cur = await db.execute("SELECT * FROM provider_chains WHERE id = ?", (new_id,))
        row = dict(await cur.fetchone())
        return _chain_public(row, await _load_chain_members(db, new_id))


async def delete_chain(database_path: str, chain_id: int) -> bool:
    async with aiosqlite.connect(database_path) as db:
        await db.execute("DELETE FROM provider_chain_members WHERE chain_id = ?", (int(chain_id),))
        cur = await db.execute("DELETE FROM provider_chains WHERE id = ?", (int(chain_id),))
        await db.commit()
        return (cur.rowcount or 0) > 0


async def set_chain_enabled(database_path: str, chain_id: int, enabled: bool) -> Dict[str, Any]:
    """Enable exactly one chain (enabling disables the others) or disable one.

    Single-enabled invariant mirrors the single-active-provider rule: the
    runtime can only serve one ordered chain per request, and leaving two
    'enabled' would make resolution order-dependent. Enabling the already-
    enabled chain is idempotent."""
    async with aiosqlite.connect(database_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM provider_chains WHERE id = ?", (int(chain_id),))
        if not await cur.fetchone():
            raise LookupError(f"Unknown chain id: {chain_id}")
        if enabled:
            await db.execute("UPDATE provider_chains SET is_enabled = 0")
            await db.execute(
                "UPDATE provider_chains SET is_enabled = 1, updated_at = ? WHERE id = ?",
                (_utcnow(), int(chain_id)),
            )
            # Mutual exclusion: an enabled chain turns off every single model.
            await db.execute("UPDATE llm_providers SET is_active = 0")
        else:
            await db.execute(
                "UPDATE provider_chains SET is_enabled = 0, updated_at = ? WHERE id = ?",
                (_utcnow(), int(chain_id)),
            )
        await db.commit()
        cur = await db.execute("SELECT * FROM provider_chains WHERE id = ?", (int(chain_id),))
        row = dict(await cur.fetchone())
        return _chain_public(row, await _load_chain_members(db, int(chain_id)))


async def set_chain_members(database_path: str, chain_id: int, provider_ids: List[int]) -> Dict[str, Any]:
    """Replace a chain's ordered members. `provider_ids` is the desired order
    (index 0 = primary). Rejects unknown providers and duplicate ids; the
    uniqueness index is a second line of defense. The same provider may not
    appear twice — that is the loop/duplicate guard at the storage layer."""
    ids: List[int] = []
    seen: set[int] = set()
    for raw in provider_ids or []:
        try:
            pid = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid provider id: {raw!r}")
        if pid in seen:
            raise ValueError(f"Provider id {pid} appears more than once in the chain.")
        seen.add(pid)
        ids.append(pid)
    if len(ids) > MAX_CHAIN_MEMBERS:
        raise ValueError(f"A chain can hold at most {MAX_CHAIN_MEMBERS} providers.")

    async with aiosqlite.connect(database_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM provider_chains WHERE id = ?", (int(chain_id),))
        if not await cur.fetchone():
            raise LookupError(f"Unknown chain id: {chain_id}")
        if ids:
            placeholders = ",".join("?" for _ in ids)
            cur = await db.execute(
                f"SELECT id FROM llm_providers WHERE id IN ({placeholders})", tuple(ids)
            )
            found = {int(r["id"]) for r in await cur.fetchall()}
            missing = [pid for pid in ids if pid not in found]
            if missing:
                raise LookupError(f"Unknown provider id(s): {', '.join(map(str, missing))}")
        await db.execute(
            "DELETE FROM provider_chain_members WHERE chain_id = ?", (int(chain_id),)
        )
        now = _utcnow()
        for position, pid in enumerate(ids):
            await db.execute(
                "INSERT INTO provider_chain_members (chain_id, provider_id, position, created_at)"
                " VALUES (?, ?, ?, ?)",
                (int(chain_id), pid, position, now),
            )
        await db.execute(
            "UPDATE provider_chains SET updated_at = ? WHERE id = ?", (_utcnow(), int(chain_id))
        )
        await db.commit()
        cur = await db.execute("SELECT * FROM provider_chains WHERE id = ?", (int(chain_id),))
        row = dict(await cur.fetchone())
        return _chain_public(row, await _load_chain_members(db, int(chain_id)))


async def reorder_chain_members(
    database_path: str, chain_id: int, provider_ids: List[int]
) -> Dict[str, Any]:
    """Reorder by providing the same member ids in a new order. Validates
    that the set matches exactly the current members (no additions/removals
    through this path — use set_chain_members for that)."""
    current = await get_chain(database_path, chain_id)
    if current is None:
        raise LookupError(f"Unknown chain id: {chain_id}")
    current_ids = [int(m["provider_id"]) for m in current["members"]]
    proposed = [int(p) for p in (provider_ids or [])]
    if sorted(proposed) != sorted(current_ids):
        raise ValueError(
            "reorder must supply exactly the chain's current provider ids "
            "(use PUT members to add or remove providers)"
        )
    return await set_chain_members(database_path, chain_id, proposed)


async def remove_provider_from_chains(database_path: str, provider_id: int) -> None:
    """Cascade a provider deletion through chain membership so no chain keeps
    a dangling member (the FK ON DELETE CASCADE covers a DELETE, but this
    keeps membership cleanup explicit and testable across engines)."""
    async with aiosqlite.connect(database_path) as db:
        await db.execute(
            "DELETE FROM provider_chain_members WHERE provider_id = ?", (int(provider_id),)
        )
        await db.commit()


async def get_enabled_chain(database_path: str) -> Optional[Dict[str, Any]]:
    """The one enabled chain (public shape), or None. Deterministic: only one
    row can carry is_enabled=1 (set_chain_enabled enforces it); the ORDER BY
    is a defensive tiebreak."""
    async with aiosqlite.connect(database_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM provider_chains WHERE is_enabled = 1 ORDER BY id LIMIT 1"
        )
        row = await cur.fetchone()
        if not row:
            return None
        return _chain_public(dict(row), await _load_chain_members(db, int(row["id"])))


async def get_chain_providers(database_path: str) -> List[Dict[str, Any]]:
    """Server-side execution view: the enabled chain's members in order, each
    with the DECRYPTED key, deduped by provider id. Never serialize this to
    the UI. Empty list when no chain is enabled.

    Loop/duplicate protection: the storage layer forbids duplicates, but a
    legacy/hand-edited DB could still carry one; dedup here guarantees the
    runtime chain can never loop on the same provider."""
    chain = await get_enabled_chain(database_path)
    if not chain or not chain["members"]:
        return []
    out: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for member in chain["members"]:
        pid = int(member["provider_id"])
        if pid in seen:
            continue
        seen.add(pid)
        secret = await get_provider_secret(database_path, pid)
        if secret is None:
            continue  # deleted between reads: skip rather than crash the run
        out.append({**secret, "chain_id": chain["id"], "position": int(member["position"])})
    return out
