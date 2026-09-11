"""User-managed LLM providers (sidebar Providers tab).

Multiple OpenAI-compatible providers can be stored; exactly one may be
active at a time. The active provider is the ONLY model the research
pipeline uses — no Groq/HuggingFace fallback runs while one is selected.
That exclusivity is deliberate: selection is an explicit user choice, so a
failing active provider degrades the run instead of silently spending a
different key. With no active provider, the legacy chain applies
(env CUSTOM_LLM_* trio, then Groq, then HuggingFace).

API keys are encrypted at rest (Fernet). Key resolution order:
MARS_SECRET_KEY env var, else backend/.mars_secret (0600, auto-created on
first encrypt). The plaintext key only ever leaves this module inside
get_active_provider() (server-side chain) and the /{id}/test probe — list
and CRUD responses carry a last-4 hint, never the key.
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
    """Exactly-one invariant: one transaction clears all flags, sets one."""
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
