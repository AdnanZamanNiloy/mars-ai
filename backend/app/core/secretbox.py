"""Key encryption at rest for user-managed LLM providers.

The Providers UI promises "stored encrypted, never displayed": API keys are
Fernet-encrypted before they touch SQLite. The Fernet key lives in a
machine-local secret file next to the database, created on first use with
owner-only permissions. Losing that file means stored keys are undecryptable
(the provider row must be re-entered) — that is stated here, not hidden.
"""
from __future__ import annotations

import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from app.core.logging import get_logger

logger = get_logger(__name__)

_secret_file: Path | None = None
_fernet: Fernet | None = None


def _key_path(database_path: str) -> Path:
    db = Path(database_path)
    directory = db.parent if str(db.parent) not in ("", ".") else Path(".")
    return directory / ".provider_secret"


def _get_fernet(database_path: str) -> Fernet:
    global _secret_file, _fernet
    if _fernet is not None and _secret_file == _key_path(database_path):
        return _fernet
    path = _key_path(database_path)
    if path.exists():
        key = path.read_bytes().strip()
    else:
        key = Fernet.generate_key()
        path.write_bytes(key)
        try:
            os.chmod(path, 0o600)
        except OSError:
            logger.warning("provider secret file permissions not set (non-POSIX fs?)")
        logger.info("provider secret key created", path=str(path))
    _secret_file = path
    _fernet = Fernet(key)
    return _fernet


def encrypt_secret(value: str, database_path: str) -> str:
    """Encrypt an API key for storage. Empty input stays empty."""
    text = str(value or "").strip()
    if not text:
        return ""
    token = _get_fernet(database_path).encrypt(text.encode("utf-8"))
    return token.decode("ascii")


def decrypt_secret(value: str, database_path: str) -> str:
    """Decrypt a stored key. Undecryptable rows (missing/rotated secret file)
    return "" — the provider then behaves like it has no key and the /test
    probe surfaces the auth failure instead of leaking anything."""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return _get_fernet(database_path).decrypt(text.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError) as exc:
        logger.warning("provider key undecryptable (secret file missing or rotated): %s", exc)
        return ""


def key_hint(value: str) -> str:
    """Last-4 display hint. Empty/undecryptable keys show no characters."""
    text = str(value or "").strip()
    if len(text) < 4:
        return ""
    return f"…{text[-4:]}"
