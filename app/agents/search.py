from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlparse

import httpx
from duckduckgo_search import DDGS

from app.core.config import Settings
from app.agents.evidence_utils import source_reliability_score

logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURE
# =============================================================================

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    content: str = ""
    sub_question: str = ""
    provider: str = "unknown"
    search_type: str = "general"
    reliability_score: float = 0.0
    content_length: int = 0
    fetched_at: float = field(default_factory=time.time)
    is_content_fetched: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "content": self.content or self.snippet,
            "sub_question": self.sub_question,
            "provider": self.provider,
            "search_type": self.search_type,
            "reliability_score": round(self.reliability_score, 3),
            "content_length": self.content_length,
        }


# =============================================================================
# UTILITIES (🔥 NEW)
# =============================================================================

def _normalize_text(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _semantic_overlap(a: str, b: str) -> float:
    a_words = set(_normalize_text(a).split())
    b_words = set(_normalize_text(b).split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


def _is_semantic_duplicate(a: str, b: str, threshold: float = 0.6) -> bool:
    return _semantic_overlap(a, b) >= threshold


def _relevance_score(query: str, text: str) -> float:
    q_words = set(_normalize_text(query).split())
    t_words = set(_normalize_text(text).split())
    if not q_words:
        return 0.0
    return len(q_words & t_words) / len(q_words)


def _domain(url: str) -> str:
    return urlparse(url).netloc.replace("www.", "")


# =============================================================================
# CONFIG
# =============================================================================

BLOCKED_DOMAINS = {
    "pinterest.com", "instagram.com", "facebook.com",
    "twitter.com", "x.com", "tiktok.com", "youtube.com",
    "amazon.com", "ebay.com", "quora.com"
}


# =============================================================================
# SCORING (🔥 UPGRADED)
# =============================================================================

def _score_result(result: SearchResult, query: str) -> float:
    base = source_reliability_score(result.url)

    snippet = result.snippet or ""
    content = result.content or ""

    richness = min(0.10, len(snippet) / 1500)
    relevance = _relevance_score(query, snippet + " " + content)
    content_bonus = 0.06 if result.is_content_fetched else 0.0
    wiki_penalty = 0.15 if "wikipedia.org" in result.url else 0.0

    return base + richness + (relevance * 0.25) + content_bonus - wiki_penalty


# =============================================================================
# RANK + DEDUP (🔥 CORE FIX)
# =============================================================================

def _deduplicate_and_rank(results, query, max_results=10):
    seen = set()
    filtered = []

    # URL dedup + block
    for r in results:
        if not r.url or r.url in seen:
            continue
        if any(b in r.url for b in BLOCKED_DOMAINS):
            continue
        seen.add(r.url)
        filtered.append(r)

    # scoring
    for r in filtered:
        r.reliability_score = _score_result(r, query)

    ranked = sorted(filtered, key=lambda r: r.reliability_score, reverse=True)

    # semantic dedup (🔥)
    diverse = []
    for r in ranked:
        if not any(_is_semantic_duplicate(r.snippet, e.snippet) for e in diverse):
            diverse.append(r)

    # domain diversity
    selected = []
    domain_count = {}

    for r in diverse:
        d = _domain(r.url)
        cap = 1 if "wikipedia.org" in d else 2

        if domain_count.get(d, 0) >= cap:
            continue

        selected.append(r)
        domain_count[d] = domain_count.get(d, 0) + 1

        if len(selected) >= max_results:
            break

    return selected


# =============================================================================
# CONTENT FETCH
# =============================================================================

def _clean_html(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:6000]


async def _fetch_content(url: str):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            if r.status_code == 200:
                return _clean_html(r.text)
    except Exception:
        return ""
    return ""


# =============================================================================
# MAIN CLIENT
# =============================================================================

class SearchClient:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.semaphore = asyncio.Semaphore(max(1, settings.max_parallel_search))

    async def run_search(self, sub_questions):
        tasks = [self._search(q) for q in sub_questions]
        batches = await asyncio.gather(*tasks)

        results = []
        for q, batch in zip(sub_questions, batches):
            for r in batch:
                r.sub_question = q
                results.append(r.to_dict())

        return results

    async def _search(self, query):
        collected = []

        async with self.semaphore:
            collected.extend(await self._ddg_text(query))
            collected.extend(await self._ddg_news(query))
            collected.extend(await self._wiki(query))

        if not collected:
            return []

        ranked = _deduplicate_and_rank(collected, query)

        # fetch content
        for r in ranked[:3]:
            content = await _fetch_content(r.url)
            if content:
                r.content = content
                r.content_length = len(content)
                r.is_content_fetched = True

        return ranked

    # =========================
    # PROVIDERS
    # =========================

    async def _ddg_text(self, query):
        def _search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=8))

        rows = await asyncio.to_thread(_search)

        return [
            SearchResult(
                title=r["title"],
                url=r["href"],
                snippet=r["body"],
                provider="ddg_text",
            )
            for r in rows if r.get("href")
        ]

    async def _ddg_news(self, query):
        def _search():
            with DDGS() as ddgs:
                return list(ddgs.news(query, max_results=6))

        rows = await asyncio.to_thread(_search)

        return [
            SearchResult(
                title=r["title"],
                url=r["url"],
                snippet=r["body"],
                provider="ddg_news",
                search_type="news"
            )
            for r in rows if r.get("url")
        ]

    async def _wiki(self, query):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": query,
                        "format": "json"
                    }
                )
                data = r.json()

            results = []
            for item in data.get("query", {}).get("search", []):
                title = item["title"]
                results.append(SearchResult(
                    title=title,
                    url=f"https://en.wikipedia.org/wiki/{quote(title)}",
                    snippet=re.sub(r"<.*?>", "", item.get("snippet", "")),
                    provider="wikipedia"
                ))

            return results
        except:
            return []