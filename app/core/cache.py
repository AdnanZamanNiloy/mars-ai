"""Bounded disk-backed cache (Phase 1.4).

Any cache in this project MUST have an explicit size cap (AGENTS.md Section 5)
— an unbounded cache is a slower way to hit the same OOM RUNTIME_STATE caused.
"""
import hashlib
import json
import logging
from typing import Any, Optional

from diskcache import Cache

from app.core.config import Settings

_cache: Optional[Cache] = None


def get_cache(settings: Settings) -> Cache:
    """Lazily create the process-wide cache with a hard size limit."""
    global _cache
    if _cache is None:
        _cache = Cache(
            directory=settings.database_url.rsplit("/", 1)[0] + "/.cache"
            if "/" in settings.database_url
            else ".cache",
            size_limit=settings.cache_size_limit_bytes,
        )
    return _cache


def cache_key(*parts: Any) -> str:
    """Stable hash key from arbitrary JSON-serializable parts."""
    serialized = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
