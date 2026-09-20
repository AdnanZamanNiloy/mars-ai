"""Shared fixtures for MARS test suite."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def test_settings():
    from app.core.config import get_settings

    return get_settings()


@pytest.fixture
def fake_groq_json():
    """Returns a factory building a Groq-shaped 200 response with given content."""

    def _build(content: str):
        import httpx

        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return _build


@pytest.fixture(autouse=True)
def _disable_llm_cache(monkeypatch):
    """The LLM response cache sits ABOVE httpx: a warm disk entry would
    bypass respx mocks entirely and make HTTP-level assertions flaky
    (and order-dependent across pytest invocations). Every test runs
    uncached; dedicated cache tests re-enable it explicitly."""
    from app.core import llm_cache

    monkeypatch.setattr(llm_cache, "_force_disabled", True)


@pytest.fixture(autouse=True)
def _disable_live_citation_check(monkeypatch):
    """Unit tests must never fire real HTTP at cited URLs. The synthesizer
    node imports check_citations at call time, so patching the module
    attribute covers every path; dedicated citation tests re-enable it."""
    from app.agents import citation_check

    async def _noop(answer, answer_support, **kw):
        return {"checked": 0, "sources": [], "summary": {}, "enabled": False}

    monkeypatch.setattr(citation_check, "check_citations", _noop)


@pytest.fixture(autouse=True)
def _isolate_provider_store(monkeypatch):
    """Unit tests must never read the developer's real research.db: an
    active provider OR an enabled fallback chain selected in the Providers
    tab would hijack every LLMClient call in tests that don't override
    database_url. Both lookups return empty for the default path; tmp-DB
    tests keep the real behavior."""
    from app.core import providers as store

    real_active = store.get_active_provider
    real_chain = store.get_chain_providers

    async def _patched_active(database_path):
        if str(database_path) in ("", "./research.db"):
            return None
        return await real_active(database_path)

    async def _patched_chain(database_path):
        if str(database_path) in ("", "./research.db"):
            return []
        return await real_chain(database_path)

    monkeypatch.setattr(store, "get_active_provider", _patched_active)
    monkeypatch.setattr(store, "get_chain_providers", _patched_chain)
