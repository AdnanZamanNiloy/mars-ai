"""LLM response cache: disk-backed, keyed by exact prompt identity.

Research pipelines re-issue the same prompts far more often than chatbots:
the critic re-evaluates unchanged evidence after a partial expansion pass,
re-runs of the same query rebuild identical planner prompts, and the
probe-free deterministic prompts (summarizer preambles) repeat across every
run. Each repeat re-bills free-tier tokens and adds seconds of latency.

Design:

* Key = sha256(provider endpoint, model, system prompt, user prompt). No
  semantic/fuzzy matching — an exact-hit cache can never return a subtly
  wrong answer to a slightly different question.
* Value = {"text": ..., "input_tokens": ..., "output_tokens": ...} so cache
  hits still feed the token ledger (free-tier TPD budgets are consumed even
  when the dollar cost is zero... conceptually the tokens were already
  spent when the entry was written; recording them keeps utilization honest
  without double-billing USD).
* TTL + hard size cap via diskcache, in a subdirectory of the shared cache
  root. Entries are evicted least-recently-used first.
* Disabled by default in tests (conftest fixture) so respx-mocked HTTP
  assertions never get bypassed by a warm disk cache.

The module-level ``enabled`` flag lets tests flip the cache off without
reconstructing the client; the Settings field is the persistent default.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any, Dict, Optional

from diskcache import Cache

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_CACHE_VERSION = "v1"
_cache: Optional[Cache] = None
_cache_dir: str = ""
_lock = threading.Lock()

# Two switches: the settings-seeded default and a hard test override.
# configure() may freely rewrite the first; tests monkeypatch the second so
# LLMClient construction mid-test can NEVER re-enable the cache.
_settings_enabled: bool = True
_force_disabled: bool = False


def is_enabled() -> bool:
    """Effective state: settings say on AND tests haven't forced it off."""
    return _settings_enabled and not _force_disabled


def set_enabled(value: bool) -> None:
    """Explicit runtime switch (benchmarks/tests)."""
    global _settings_enabled, _force_disabled
    _settings_enabled = bool(value)
    _force_disabled = False


ttl_sec: int = 21600  # 6h default — research prompts age out with sources


def configure(settings: Settings) -> None:
    """Seed the process-level cache + switches from settings. Idempotent.

    Only touches the settings-seeded switch — a forced test override stays
    in force even when a new LLMClient is constructed mid-test."""
    global _cache, _cache_dir, _settings_enabled, ttl_sec
    with _lock:
        _settings_enabled = bool(getattr(settings, "llm_cache_enabled", True))
        ttl_sec = int(getattr(settings, "llm_cache_ttl_sec", 21600) or 21600)
        root = (
            settings.database_url.rsplit("/", 1)[0] + "/.cache"
            if "/" in str(settings.database_url)
            else ".cache"
        )
        directory = f"{root}/llm"
        if _cache is None or _cache_dir != directory:
            try:
                _cache = Cache(
                    directory=directory,
                    size_limit=int(getattr(settings, "cache_size_limit_bytes", 250_000_000) or 250_000_000),
                )
                _cache_dir = directory
            except Exception as exc:  # pragma: no cover - unwritable disk
                logger.warning("llm cache unavailable, running uncached: %s", exc)
                _cache = None
                _cache_dir = ""


def _get_cache() -> Optional[Cache]:
    if not is_enabled():
        return None
    if _cache is None:
        return None
    return _cache


def cache_key(endpoint: str, model: str, system_prompt: str, user_prompt: str) -> str:
    digest = hashlib.sha256()
    for part in (_CACHE_VERSION, endpoint, model, system_prompt, user_prompt):
        digest.update(part.encode("utf-8", errors="replace"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def get(endpoint: str, model: str, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
    """Cached completion payload, or None on miss/disabled/unavailable."""
    store = _get_cache()
    if store is None:
        return None
    try:
        value = store.get(cache_key(endpoint, model, system_prompt, user_prompt))
        if isinstance(value, dict) and isinstance(value.get("text"), str):
            return value
        return None
    except Exception as exc:  # pragma: no cover - corrupt entry, read-only disk
        logger.warning("llm cache read failed (%s); treating as miss", exc)
        return None


def put(
    endpoint: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    text: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> None:
    """Store a successful completion. Failures are logged, never raised —
    a cache write problem must not fail a research run."""
    store = _get_cache()
    if store is None:
        return
    try:
        store.set(
            cache_key(endpoint, model, system_prompt, user_prompt),
            {
                "text": text,
                "input_tokens": int(input_tokens or 0),
                "output_tokens": int(output_tokens or 0),
            },
            expire=ttl_sec,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("llm cache write failed: %s", exc)


def clear() -> None:
    """Drop every entry (benchmarks call this between scenarios)."""
    store = _get_cache()
    if store is None:
        return
    try:
        store.clear()
    except Exception as exc:  # pragma: no cover
        logger.warning("llm cache clear failed: %s", exc)


def stats() -> Dict[str, Any]:
    """Cache vitals for diagnostics + benchmarks."""
    store = _cache
    if store is None:
        return {"enabled": is_enabled(), "available": False}
    try:
        return {
            "enabled": is_enabled(),
            "available": True,
            "directory": _cache_dir,
            "size_bytes": int(store.volume()) if hasattr(store, "volume") else None,
            "count": len(store),
            "ttl_sec": ttl_sec,
        }
    except Exception:  # pragma: no cover
        return {"enabled": is_enabled(), "available": False}
