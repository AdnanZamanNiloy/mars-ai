"""Retrieval depth: content-fetch breadth follows search_fetch_top_n."""

import asyncio

from app.agents import search as search_mod
from app.agents.search import SearchClient, SearchResult
from app.core.config import Settings


TOPICS = [
    "transformer attention architectures benchmarks",
    "vector database indexing strategies comparison",
    "quantization techniques memory footprint analysis",
    "retrieval chunking methods evaluation study",
    "embedding model leaderboard multilingual results",
    "reranking algorithms latency quality tradeoffs",
    "knowledge graph construction pipelines survey",
    "agent orchestration frameworks design patterns",
]


def _result(i: int, tag: str) -> SearchResult:
    topic = TOPICS[i % len(TOPICS)]
    return SearchResult(
        title=f"Result {i}: {topic}",
        url=f"https://example-{i}.com/{tag}",
        snippet=f"Article {i} covering {topic} with measurements",
        provider="test",
    )


def _run_fetch_count(monkeypatch, tmp_path, top_n: int, pool: int = 8) -> int:
    tag = f"depthprobe{top_n}"
    settings = Settings(
        groq_api_key="k",
        search_fetch_top_n=top_n,
        database_url=str(tmp_path / "t.db"),
        _env_file=None,
    )
    client = SearchClient(settings)
    calls = []

    async def fake_provider(self, query):
        return [_result(i, tag) for i in range(pool)]

    async def fake_fetch(url: str):
        calls.append(url)
        return "full page content here", ""

    monkeypatch.setattr(SearchClient, "_ddg_text", fake_provider)
    monkeypatch.setattr(SearchClient, "_ddg_news", fake_provider)
    monkeypatch.setattr(SearchClient, "_wiki", fake_provider)
    monkeypatch.setattr(search_mod, "_fetch_content", fake_fetch)

    ranked = asyncio.run(client._search(f"retrieval study {tag}"))
    assert ranked, "expected ranked results"
    return len(calls)


def test_fetch_breadth_follows_setting(monkeypatch, tmp_path):
    assert _run_fetch_count(monkeypatch, tmp_path, 5) == 5


def test_fetch_breadth_small_setting(monkeypatch, tmp_path):
    assert _run_fetch_count(monkeypatch, tmp_path, 2) == 2


def test_fetch_top_n_default_is_five():
    settings = Settings(groq_api_key="k", _env_file=None)
    assert settings.search_fetch_top_n == 5
