"""Tavily provider, scraper hygiene, and PDF extraction (web-research upgrade)."""

import httpx
import pytest
import respx

from app.agents.search import (
    SearchClient,
    _extract_pdf_text,
    _has_tavily_key,
    _looks_like_block_page,
    _tavily_to_results,
)
from app.core.config import Settings

TAVILY_URL = "https://api.tavily.com/search"


def _settings(**over):
    base = {"groq_api_key": "k", "_env_file": None}
    base.update(over)
    return Settings(**base)


def test_tavily_key_gate():
    assert _has_tavily_key(_settings(tavily_api_key="tvly-real-key")) is True
    assert _has_tavily_key(_settings(tavily_api_key="")) is False
    assert _has_tavily_key(_settings(tavily_api_key="your_tavily_api_key_here")) is False


def test_tavily_payload_mapping():
    payload = {
        "results": [
            {"title": "RAG survey", "url": "https://arxiv.org/a",
             "content": "Long survey content here", "score": 0.9,
             "published_date": "2026-03-01"},
            {"title": "No URL row"},
            "garbage",
        ]
    }
    results = _tavily_to_results(payload, "what is RAG")
    assert len(results) == 1
    row = results[0]
    assert row.url == "https://arxiv.org/a"
    assert row.provider == "tavily"
    assert row.published_at == "2026-03-01"
    assert row.snippet == "Long survey content here"
    assert _tavily_to_results({}, "q") == []
    assert _tavily_to_results(None, "q") == []


async def test_tavily_used_when_configured(monkeypatch, tmp_path):
    settings = _settings(tavily_api_key="tvly-real-key",
                         database_url=str(tmp_path / "t.db"))
    client = SearchClient(settings)
    seen = {}

    async def fake_tavily(self, query):
        seen["q"] = query
        return []

    async def boom_ddg(self, query):
        raise AssertionError("DDG must not run when Tavily is configured")

    async def empty_wiki(self, query):
        return []

    monkeypatch.setattr(SearchClient, "_tavily_search", fake_tavily)
    monkeypatch.setattr(SearchClient, "_ddg_text", boom_ddg)
    monkeypatch.setattr(SearchClient, "_ddg_news", boom_ddg)
    monkeypatch.setattr(SearchClient, "_wiki", empty_wiki)
    await client._search("tavily routing probe query")
    assert seen["q"] == "tavily routing probe query"


async def test_tavily_failure_falls_back_to_ddg(monkeypatch, tmp_path):
    settings = _settings(tavily_api_key="tvly-real-key",
                         database_url=str(tmp_path / "t.db"))
    client = SearchClient(settings)

    async def fake_tavily(self, query):
        # Mirror the real method's contract: fall back internally.
        results = await self._ddg_text(query)
        return results

    async def fake_ddg(self, query):
        from app.agents.search import SearchResult
        return [SearchResult(title="t", url="https://a.com", snippet="s", provider="ddg_text")]

    monkeypatch.setattr(SearchClient, "_tavily_search", fake_tavily)
    monkeypatch.setattr(SearchClient, "_ddg_text", fake_ddg)
    monkeypatch.setattr(SearchClient, "_ddg_news", fake_ddg)
    monkeypatch.setattr(SearchClient, "_wiki", fake_ddg)
    ranked = await client._search("tavily fallback probe query")
    assert len(ranked) >= 1


async def test_tavily_http_error_falls_back_live(monkeypatch):
    """The real _tavily_search converts transport errors into DDG results."""
    settings = _settings(tavily_api_key="tvly-real-key")
    client = SearchClient(settings)

    async def fake_ddg(self, query):
        from app.agents.search import SearchResult
        return [SearchResult(title="t", url="https://a.com", snippet="s", provider="ddg_text")]

    monkeypatch.setattr(SearchClient, "_ddg_text", fake_ddg)
    monkeypatch.setattr(SearchClient, "_ddg_news", fake_ddg)
    with respx.mock(assert_all_called=False) as mock:
        mock.post(TAVILY_URL).mock(side_effect=httpx.ConnectError("down"))
        results = await client._tavily_search("what is RAG")
    assert [r.provider for r in results] == ["ddg_text", "ddg_text"]


def test_block_page_detection():
    assert _looks_like_block_page("Access denied. Please verify you are human to continue.") is True
    assert _looks_like_block_page("Enable JavaScript to view this content") is True
    # Long articles mentioning captchas are NOT block pages.
    long_article = ("This paper surveys captcha designs across two decades of research. " * 20
                    + "Access denied patterns are compared in section four.")
    assert _looks_like_block_page(long_article) is False
    assert _looks_like_block_page("") is False


def test_pdf_extraction_reads_pages(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Transfer learning reuses pretrained models for new tasks.")
    pdf_bytes = doc.tobytes()
    doc.close()
    text = _extract_pdf_text(pdf_bytes, "https://x.com/paper.pdf")
    assert "Transfer learning reuses pretrained models" in text


def test_pdf_missing_lib_skips(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("pymupdf", "fitz"):
            raise ImportError("no pdf lib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert _extract_pdf_text(b"%PDF-1.4 junk", "https://x.com/a.pdf") == ""
