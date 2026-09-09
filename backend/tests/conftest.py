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
