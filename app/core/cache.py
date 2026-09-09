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

logger = logging.getLogger(__name__)

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


async def cached(key: str, factory, settings: Settings, ttl_sec: Optional[int] = None):
    """Get-or-set with TTL. `factory` must be an awaitable-returning callable.

    Returns the cached value on hit, or the factory's value on miss (and
    stores it). Cache errors never break the pipeline — they log and bypass.
    """
    cache = get_cache(settings)
    try:
        hit = cache.get(key)
        if hit is not None:
            logger.debug("[Cache] hit key=%s", key[:12])
            return hit
    except Exception as exc:
        logger.warning("[Cache] get failed, bypassing: %s", exc, exc_info=exc)

    value = await factory()
    try:
        cache.set(key, value, expire=(ttl_sec if ttl_sec is not None else settings.cache_ttl_sec))
    except Exception as exc:
        logger.warning("[Cache] set failed, continuing uncached: %s", exc, exc_info=exc)
    return value
