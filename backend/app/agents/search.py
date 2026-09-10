from __future__ import annotations

import asyncio
from app.core.logging import get_logger
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
from urllib.parse import quote, urlparse

import httpx
from ddgs import DDGS

from app.core.config import Settings
from app.core.llm import _real_key
from app.agents.evidence_utils import source_reliability_score
from app.agents.planner import SubQuestion
from app.core.cache import cache_key, get_cache

logger = get_logger(__name__)

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
    # Publish date when the provider supplies one (DDG news `date`,
    # Wikipedia revision `timestamp`) or a fetch Last-Modified header.
    # "" means unknown — never synthesized.
    published_at: str = ""

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
            "published_at": self.published_at,
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


# Phrases that mark bot-block / JS-gate pages. Matched only when the cleaned
# text is short — long articles may legitimately mention captchas.
BLOCK_PAGE_PHRASES = (
    "access denied",
    "verify you are human",
    "enable javascript",
    "please complete the security check",
    "unusual traffic from your computer network",
)


def _looks_like_block_page(text: str) -> bool:
    lowered = (text or "").lower()
    return len(lowered) < 600 and any(p in lowered for p in BLOCK_PAGE_PHRASES)


def _extract_pdf_text(content: bytes, url: str) -> str:
    """Best-effort first pages of a PDF. PyMuPDF is optional: without it,
    PDFs stay unfetched rather than crashing the run."""
    try:
        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf
    except ImportError:
        logger.warning("[Search] PyMuPDF missing, skipping PDF: %s", url[:80])
        return ""
    try:
        pages = []
        with pymupdf.open(stream=content, filetype="pdf") as doc:
            for page in doc[:5]:
                pages.append(page.get_text())
        return re.sub(r"\s+", " ", "\n".join(pages)).strip()[:6000]
    except Exception as exc:
        logger.warning("[Search] PDF extract failed for %s: %s", url[:80], exc, exc_info=exc)
        return ""


async def _fetch_content(url: str):
    """Fetch + clean page text. Returns (text, last_modified_header_or_empty)."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            if r.status_code == 200:
                content_type = str(r.headers.get("content-type", "") or "").lower()
                if "application/pdf" in content_type or url.lower().split("?")[0].endswith(".pdf"):
                    return _extract_pdf_text(r.content, url), str(r.headers.get("last-modified", "") or "")
                text = _clean_html(r.text)
                if _looks_like_block_page(text):
                    logger.warning("[Search] block page detected, dropping: %s", url[:80])
                    return "", ""
                return text, str(r.headers.get("last-modified", "") or "")
    except Exception as exc:
        logger.warning("[Search] content fetch failed for %s: %s", url[:80], exc, exc_info=exc)
    return "", ""


def _has_tavily_key(settings: Settings) -> bool:
    """A real Tavily key (not empty, not an example placeholder)."""
    return bool(_real_key(getattr(settings, "tavily_api_key", "")))


def _tavily_to_results(payload: Any, query: str) -> List["SearchResult"]:
    """Map a Tavily /search response to ranked SearchResults. Pure — the
    network call stays in _tavily_search so this is unit-testable offline."""
    results = []
    items = payload.get("results", []) if isinstance(payload, dict) else []
    for row in items:
        if not isinstance(row, dict) or not row.get("url"):
            continue
        # Tavily already returns cleaned content: keep it in `content` so
        # the fetch loop below recognizes it as fetched (no double fetch).
        content = str(row.get("content", "") or "")
        results.append(SearchResult(
            title=str(row.get("title", "") or ""),
            url=str(row.get("url") or ""),
            snippet=content[:1500],
            content=content[:6000],
            provider="tavily",
            published_at=str(row.get("published_date", "") or ""),
        ))
    return results


# =============================================================================
# MAIN CLIENT
# =============================================================================

class SearchClient:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.semaphore = asyncio.Semaphore(max(1, settings.max_parallel_search))

    async def run_search(self, sub_questions: List[Union[SubQuestion, str]]) -> List[Dict[str, Any]]:
        """Search a batch of delegation-contract sub-questions (or raw strings)."""
        tasks = [self._search(q) for q in sub_questions]
        batches = await asyncio.gather(*tasks)

        results = []
        for contract, batch in zip(sub_questions, batches):
            if isinstance(contract, str):
                question_text = contract
            else:
                # The contract's question is what flows to the summarizer;
                # minimum_sources/stop_condition stay with the contract owner.
                question_text = str(contract.get("question", "")).strip() or str(contract)
            for r in batch:
                r.sub_question = question_text
                results.append(r.to_dict())

        return results

    async def _search(self, query):
        settings = self.settings

        async def _uncached_search():
            collected = []

            async with self.semaphore:
                # Tavily replaces DDG text/news when configured: one paid call
                # with clean markdown beats free snippets. Wikipedia always
                # runs (free, high-trust encyclopedia). Any Tavily failure
                # falls back to DDG inside _tavily_search.
                if _has_tavily_key(settings):
                    collected.extend(await self._tavily_search(query))
                else:
                    collected.extend(await self._ddg_text(query))
                    collected.extend(await self._ddg_news(query))
                collected.extend(await self._wiki(query))

            if not collected:
                return []

            ranked = _deduplicate_and_rank(collected, query)

            # Fetch content for the top results concurrently (independent I/O).
            # Depth is setting-driven (search_fetch_top_n); fetched text is
            # consumed by the summarizer, which releases it afterwards.
            async def _attach(r):
                # Tavily results arrive with content: never re-fetch them.
                if r.content:
                    r.content_length = len(r.content)
                    r.is_content_fetched = True
                    return
                content, last_modified = await _fetch_content(r.url)
                if content:
                    r.content = content
                    r.content_length = len(content)
                    r.is_content_fetched = True
                    if not r.published_at:
                        r.published_at = last_modified

            fetch_n = max(1, int(getattr(settings, "search_fetch_top_n", 3) or 3))
            await asyncio.gather(*(_attach(r) for r in ranked[:fetch_n]))
            return ranked

        key = cache_key("search_query", _normalize_text(query))
        try:
            cached = get_cache(settings).get(key)
        except Exception as exc:
            logger.warning("[Search] cache read failed: %s", exc, exc_info=exc)
            cached = None

        if cached is not None:
            logger.info("[Search] cache hit for query: %s", query[:60])
            return cached

        logger.info("[Search] cache miss for query: %s", query[:60])
        ranked = await _uncached_search()
        try:
            get_cache(settings).set(key, ranked, expire=settings.cache_ttl_sec)
        except Exception as exc:
            logger.warning("[Search] cache write failed, continuing uncached: %s", exc, exc_info=exc)
        return ranked

    # =========================
    # PROVIDERS
    # =========================

    async def _tavily_search(self, query):
        """Tavily web search (paid, free tier available). Falls back to DDG
        on any failure — a bad paid call must never be worse than free."""
        api_key = _real_key(self.settings.tavily_api_key)
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": api_key,
                        "query": query if isinstance(query, str) else str(query),
                        "search_depth": "basic",
                        "max_results": 8,
                        "include_answer": False,
                    },
                )
                r.raise_for_status()
                payload = r.json()
        except Exception as exc:
            logger.warning("[Search] Tavily failed, falling back to DDG: %s", exc, exc_info=exc)
            results = await self._ddg_text(query)
            results.extend(await self._ddg_news(query))
            return results
        return _tavily_to_results(payload, query)

    async def _ddg_text(self, query):
        def _search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=8))

        try:
            rows = await asyncio.to_thread(_search)
        except Exception as exc:
            logger.warning("[Search] DDG text failed for query %r: %s", query[:60], exc, exc_info=exc)
            return []

        return [
            SearchResult(
                title=r["title"],
                url=r["href"],
                snippet=r["body"],
                provider="ddg_text",
                published_at=str(r.get("date", "") or ""),
            )
            for r in rows if r.get("href")
        ]

    async def _ddg_news(self, query):
        def _search():
            with DDGS() as ddgs:
                return list(ddgs.news(query, max_results=6))

        try:
            rows = await asyncio.to_thread(_search)
        except Exception as exc:
            logger.warning("[Search] DDG news failed for query %r: %s", query[:60], exc, exc_info=exc)
            return []

        return [
            SearchResult(
                title=r["title"],
                url=r["url"],
                snippet=r["body"],
                provider="ddg_news",
                search_type="news",
                published_at=str(r.get("date", "") or ""),
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
                    provider="wikipedia",
                    published_at=str(item.get("timestamp", "") or ""),
                ))

            return results
        except Exception as exc:
            logger.warning("[Search] Wikipedia failed for query %r: %s", query[:60], exc, exc_info=exc)
            return []