"""Tavily provider, scraper hygiene, and PDF extraction (web-research upgrade)."""

import httpx
import json
import pytest
import respx
import time

from app.agents.search import (
    SearchClient,
    _clean_html,
    _extract_pdf_text,
    _has_tavily_key,
    _looks_like_block_page,
    _prepare_tavily_query,
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

    async def fake_tavily(self, query, query_domains=None, topic="general"):
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

    async def fake_tavily(self, query, query_domains=None, topic="general"):
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


async def test_tavily_content_skips_refetch(monkeypatch, tmp_path):
    """Tavily content counts as fetched: no second HTTP fetch per URL."""
    from app.agents import search as search_mod

    settings = _settings(tavily_api_key="tvly-real-key",
                         database_url=str(tmp_path / "t.db"))
    client = SearchClient(settings)
    calls = []

    async def fake_tavily(self, query, query_domains=None, topic="general"):
        return _tavily_to_results(
            {"results": [{"title": "t", "url": "https://a.com",
                           "content": "already have full text here"}]},
            query)

    async def boom_fetch(url, client=None):
        calls.append(url)
        raise AssertionError("must not refetch Tavily content")

    async def empty_wiki(self, query):
        return []

    monkeypatch.setattr(SearchClient, "_tavily_search", fake_tavily)
    monkeypatch.setattr(SearchClient, "_wiki", empty_wiki)
    monkeypatch.setattr(search_mod, "_fetch_content", boom_fetch)
    ranked = await client._search("tavily skip probe query")
    assert len(ranked) == 1
    assert ranked[0].is_content_fetched is True
    assert calls == []


def test_prepare_tavily_query_translates_site_operators():
    query, domains = _prepare_tavily_query(
        "retrieval benchmarks site:arxiv.org,foo/bar comparisons",
        ["example.com", "arxiv.org"],
    )

    assert "site:" not in query
    assert domains == ["example.com", "arxiv.org", "foo"]
    assert len(query) <= 400


def test_prepare_tavily_query_truncates_long_queries():
    query, domains = _prepare_tavily_query("word " * 200)

    assert len(query) == 400
    assert domains == []


def test_tavily_payload_content_fallbacks():
    payload = {
        "results": [
            {"title": "Raw", "url": "https://example.com/raw",
             "raw_content": "Raw cleaned body text here"},
            {"title": "Snippet", "url": "https://example.com/snippet",
             "snippet": "Snippet fallback text here", "date": "2026-09-10"},
            {"title": "Missing URL"},
        ]
    }
    results = _tavily_to_results(payload, "probe")

    assert [(row.url, row.content, row.snippet) for row in results] == [
        ("https://example.com/raw", "Raw cleaned body text here", "Raw cleaned body text here"),
        ("https://example.com/snippet", "", "Snippet fallback text here"),
    ]
    assert [row.published_at for row in results] == ["", "2026-09-10"]


async def test_tavily_request_uses_clean_query_and_domains():
    settings = _settings(tavily_api_key="tvly-real-key")
    client = SearchClient(settings)
    watched = {}

    async def fake_ddg(self, query):
        raise AssertionError("DDG fallback must not run on HTTP 200")

    async def empty_wiki(self, query):
        return []

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(SearchClient, "_ddg_text", fake_ddg)
    monkeypatch.setattr(SearchClient, "_ddg_news", fake_ddg)
    monkeypatch.setattr(SearchClient, "_wiki", empty_wiki)
    try:
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(TAVILY_URL).mock(return_value=httpx.Response(200, json={
                "results": [{
                    "title": "Scoped hit",
                    "url": "https://arxiv.org/scoped",
                    "content": "Scoped cleaned body",
                }],
            }))
            results = await client._tavily_search(
                "scoped retrieval query site:arxiv.org/extra",
                query_domains=["example.com"],
            )
    finally:
        monkeypatch.undo()

    assert route.called
    payload = json.loads(route.calls[0].request.content.decode("utf-8"))
    assert payload["include_domains"] == ["example.com", "arxiv.org"]
    assert "site:" not in payload["query"]
    assert len(payload["query"]) <= 400
    assert [(row.url, row.provider) for row in results] == [("https://arxiv.org/scoped", "tavily")]


def test_clean_html_drops_nonvisible_markup():
    html = (
        "<html><head><title>Ignored title</title>"
        "<script>var tracker = 1;</script>"
        "<style>.hidden { display: none; }</style></head>"
        "<body><nav>Home | Search</nav>"
        "<article><h1>Visible finding</h1><p>First paragraph.</p>"
        "<p>Second paragraph.</p></article></body></html>"
    )
    text = _clean_html(html)

    assert "Visible finding" in text
    assert "First paragraph." in text
    assert "Second paragraph." in text
    assert "tracker" not in text
    assert "display: none" not in text
    assert "<" not in text
    assert len(text) <= 6000


async def test_wiki_sends_user_agent():
    """Wikipedia 403s script default UAs; _wiki must identify itself or
    every call degrades to [] (regression: all wiki queries failed live
    with a JSON decode error on the 403 page)."""
    settings = _settings()
    client = SearchClient(settings)
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get("https://en.wikipedia.org/w/api.php").mock(
            return_value=httpx.Response(200, json={
                "query": {"search": [{
                    "title": "Transformer",
                    "snippet": "A <span>transformer</span> is a device",
                    "timestamp": "2026-01-01T00:00:00Z",
                }]},
            }))
        results = await client._wiki("transformer")
    assert route.called
    assert "user-agent" in route.calls[0].request.headers
    assert "python-httpx" not in route.calls[0].request.headers["user-agent"]
    assert [(row.title, row.provider) for row in results] == [("Transformer", "wikipedia")]


async def test_tavily_empty_results_fall_back_to_ddg(monkeypatch):
    """Zero usable Tavily hits must trigger the DDG fallback (upstream
    treats empty as failure) instead of leaving wiki-only coverage."""
    from app.agents.search import SearchResult

    settings = _settings(tavily_api_key="tvly-real-key")
    client = SearchClient(settings)

    async def fake_text(self, query):
        return [SearchResult(title="t", url="https://a.com", snippet="s", provider="ddg_text")]

    async def fake_news(self, query):
        return []

    async def empty_wiki(self, query):
        return []

    monkeypatch.setattr(SearchClient, "_ddg_text", fake_text)
    monkeypatch.setattr(SearchClient, "_ddg_news", fake_news)
    monkeypatch.setattr(SearchClient, "_wiki", empty_wiki)
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(TAVILY_URL).mock(
            return_value=httpx.Response(200, json={"results": []}))
        results = await client._tavily_search("query with no usable hits")
    assert route.called
    assert json.loads(route.calls[0].request.content.decode("utf-8")).get("use_cache") is True
    assert [(row.url, row.provider) for row in results] == [("https://a.com", "ddg_text")]


async def test_fetch_content_sends_browser_ua():
    """Generic page fetch identifies as a browser (upstream scraper parity):
    publishers that block script UAs would otherwise yield block pages."""
    from app.agents import search as search_mod

    with respx.mock(assert_all_called=False) as mock:
        route = mock.get("https://example.com/article").mock(
            return_value=httpx.Response(
                200, headers={"content-type": "text/html"},
                text="<html><body><p>Article body text.</p></body></html>"))
        text, _ = await search_mod._fetch_content("https://example.com/article")
    assert route.called
    assert route.calls[0].request.headers["user-agent"].startswith("Mozilla/")
    assert "Article body text." in text


async def test_ddg_hang_degrades_fast(monkeypatch):
    """A stalled DDG call (blocking lib, no own timeout) must degrade to []
    within search_timeout_sec instead of hanging the run."""
    from app.agents import search as search_mod

    class HangingDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def text(self, query, max_results=8):
            time.sleep(5)
            return []

        def news(self, query, max_results=6):
            time.sleep(5)
            return []

    monkeypatch.setattr(search_mod, "DDGS", HangingDDGS)
    settings = _settings(search_timeout_sec=0.05)
    client = SearchClient(settings)
    start = time.monotonic()
    assert await client._ddg_text("hang probe") == []
    assert await client._ddg_news("hang probe") == []
    assert time.monotonic() - start < 4


def test_split_query_shapes():
    from app.agents.search import _split_query

    assert _split_query("plain text") == ("plain text", "")
    assert _split_query(("pair text", "news")) == ("pair text", "news")
    assert _split_query({"question": "  contract q  ", "search_type": "news"}) == ("contract q", "news")
    assert _split_query({"question": "", "search_type": "news"}) == ("", "news")
    assert _split_query("") == ("", "")


async def test_news_contract_uses_news_topic(monkeypatch, tmp_path):
    """Delegation search_type must steer retrieval: news contracts hit
    Tavily's news topic with the clean question text (never str(dict))."""
    settings = _settings(tavily_api_key="tvly-real-key",
                         database_url=str(tmp_path / "t.db"))
    client = SearchClient(settings)
    seen = {}

    async def fake_tavily(self, query, query_domains=None, topic="general"):
        seen["query"] = query
        seen["topic"] = topic
        return []

    async def empty_wiki(self, query):
        seen["wiki_query"] = query
        return []

    monkeypatch.setattr(SearchClient, "_tavily_search", fake_tavily)
    monkeypatch.setattr(SearchClient, "_wiki", empty_wiki)
    await client._search({"question": "central bank rates today", "search_type": "news"})
    assert seen["query"] == "central bank rates today"
    assert "central bank" in seen["wiki_query"]
    assert seen["topic"] == "news"


async def test_general_pair_keeps_general_topic(monkeypatch, tmp_path):
    settings = _settings(tavily_api_key="tvly-real-key",
                         database_url=str(tmp_path / "t.db"))
    client = SearchClient(settings)
    seen = {}

    async def fake_tavily(self, query, query_domains=None, topic="general"):
        seen["topic"] = topic
        return []

    async def empty_wiki(self, query):
        return []

    monkeypatch.setattr(SearchClient, "_tavily_search", fake_tavily)
    monkeypatch.setattr(SearchClient, "_wiki", empty_wiki)
    await client._search(("encyclopedia probe", "encyclopedia"))
    assert seen["topic"] == "general"


async def test_tavily_topic_in_request_body():
    """The topic parameter reaches the Tavily wire payload."""
    settings = _settings(tavily_api_key="tvly-real-key")
    client = SearchClient(settings)
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(TAVILY_URL).mock(
            return_value=httpx.Response(200, json={"results": []}))
        occasion = {"n": 0}

        async def fake_text(self, query):
            occasion["n"] += 1
            return []

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(SearchClient, "_ddg_text", fake_text)
        monkeypatch.setattr(SearchClient, "_ddg_news", fake_text)
        try:
            await client._tavily_search("markets today", topic="news")
        finally:
            monkeypatch.undo()
    assert route.called
    assert json.loads(route.calls[0].request.content.decode("utf-8"))["topic"] == "news"


async def test_empty_question_searches_nothing(monkeypatch, tmp_path):
    settings = _settings(tavily_api_key="tvly-real-key",
                         database_url=str(tmp_path / "t.db"))
    client = SearchClient(settings)
    calls = []

    async def boom_tavily(self, *a, **k):
        calls.append(1)
        return []

    monkeypatch.setattr(SearchClient, "_tavily_search", boom_tavily)
    assert await client._search({"question": "   ", "search_type": "news"}) == []
    assert calls == []
