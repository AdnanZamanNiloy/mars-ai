from __future__ import annotations

import asyncio
from app.core.logging import get_logger
import html
from html.parser import HTMLParser
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

# Wikimedia (and many publishers) 403 script default UAs. Contact-style
# string per https://meta.wikimedia.org/wiki/User-Agent_policy.
_WIKI_USER_AGENT = "MARS-research/1.0 (personal research assistant)"

# Generic page fetch uses a browser UA like upstream's scraper does: many
# publishers block identifying/script UAs on article pages (the MediaWiki
# API above is the exception — it explicitly wants an identifying UA).
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)


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

class _VisibleTextExtractor(HTMLParser):
    """Extract readable text while dropping non-visible markup.

    Dependency-free counterpart to soup-based cleaners: script/style-like
    content is discarded, block structure becomes line breaks, and the rest
    collapses to plain text.
    """

    _SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg"})
    _BLOCK_TAGS = frozenset({
        "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5",
        "h6", "section", "article", "header", "footer", "nav", "aside",
        "main", "figure", "figcaption", "table", "tr", "td", "th",
        "blockquote", "pre", "hr",
    })

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        name = (tag or "").lower()
        if name in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if name in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        name = (tag or "").lower()
        if name in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if name in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not data or not data.strip():
            return
        self._chunks.append(data.strip() + " ")

    def text(self) -> str:
        return re.sub(r"[ \t\f\v]+", " ", "".join(self._chunks)).strip()


def _clean_html(raw: str) -> str:
    try:
        extractor = _VisibleTextExtractor()
        extractor.feed(raw or "")
        extractor.close()
        text = extractor.text()
    except Exception as exc:
        # HTMLParser is tolerant, but never let cleaning break a search.
        # Regex fallback is lossy (may keep script text); log so it stays visible.
        logger.warning("[Search] HTML clean failed, using regex fallback: %s", exc, exc_info=exc)
        text = re.sub(r"<[^>]+>", " ", raw or "")
    # Decode entities (&quot without semicolon is not a valid charref, so
    # the parser leaves it literal — seen live as 'Concepts&quot').
    text = html.unescape(text)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)
    text = re.sub(r"[ \t\f\v]{2,}", " ", text).strip()
    return text[:6000]


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


async def _fetch_content(url: str, client: httpx.AsyncClient | None = None):
    """Fetch + clean page text. Returns (text, last_modified_header_or_empty).

    A shared client (connection pooling) is passed on the hot path; when
    None, a throwaway client is created so unit tests and one-off callers
    keep working unchanged.
    """
    try:
        if client is None:
            async with httpx.AsyncClient(timeout=10, headers={"User-Agent": _BROWSER_USER_AGENT}) as owned:
                return await _fetch_with_client(owned, url)
        return await _fetch_with_client(client, url)
    except Exception as exc:
        logger.warning("[Search] content fetch failed for %s: %s", url[:80], exc, exc_info=exc)
    return "", ""


async def _fetch_with_client(client: httpx.AsyncClient, url: str):
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
    return "", ""


_SITE_OPERATOR_RE = re.compile(r"site:(\S+)", re.IGNORECASE)
_TAVILY_MAX_QUERY_CHARS = 400


def _prepare_tavily_query(query: Any, query_domains: list[str] | None = None) -> tuple[str, list[str]]:
    """Normalize a query for Tavily: translate Google-style site: operators
    into include_domains (which Tavily rejects inline), collapse whitespace,
    and cap length at Tavily's 400-character limit."""
    text = query if isinstance(query, str) else str(query or "")
    domains: list[str] = []

    def _add_domain(candidate: Any) -> None:
        for part in str(candidate or "").replace(",", " ").split():
            domain = part.strip().strip(",").split("/")[0]
            if domain and domain not in domains:
                domains.append(domain)

    for candidate in list(query_domains or []):
        _add_domain(candidate)
    for match in _SITE_OPERATOR_RE.findall(text):
        _add_domain(match)
    text = _SITE_OPERATOR_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_TAVILY_MAX_QUERY_CHARS], domains


def _split_query(query: Any) -> tuple[str, str]:
    """Split any accepted query shape into (question_text, search_type).

    run_search takes raw strings, delegation contracts (SubQuestion dicts),
    or (text, search_type) pairs — the workflow fans contracts out to bare
    strings with types attached. Previously _search received whatever came
    in and handed dicts straight to the providers (str(dict) as a query);
    the workflow only ever sent strings, so this never fired live — but
    every contract's search_type was silently dropped at the boundary and
    any dict input searched garbage. Normalize once, here.
    """
    if isinstance(query, dict):
        return (
            str(query.get("question", "") or "").strip(),
            str(query.get("search_type", "") or "").strip(),
        )
    if isinstance(query, (tuple, list)) and len(query) == 2:
        return (str(query[0] or "").strip(), str(query[1] or "").strip())
    return (str(query or "").strip(), "")


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
        # Tavily may return cleaned content, raw_content, or only a snippet.
        # Keep cleaned content in `content` so the fetch loop below recognizes
        # it as fetched (no double fetch); rows without usable text fall back
        # to direct fetch in _attach.
        content = str(
            row.get("content", "") or row.get("raw_content", "") or ""
        )
        snippet_source = content or str(row.get("snippet", "") or "")
        published = (
            str(row.get("published_date", "") or "")
            or str(row.get("publishedDate", "") or "")
            or str(row.get("date", "") or "")
        )
        results.append(SearchResult(
            title=str(row.get("title", "") or ""),
            url=str(row.get("url") or ""),
            snippet=snippet_source[:1500],
            content=content[:6000],
            provider="tavily",
            published_at=published,
        ))
    return results


# =============================================================================
# MAIN CLIENT
# =============================================================================

class SearchClient:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.semaphore = asyncio.Semaphore(max(1, settings.max_parallel_search))

    async def run_search(self, sub_questions: List[Union[SubQuestion, str, tuple]]) -> List[Dict[str, Any]]:
        """Search a batch of delegation-contract sub-questions (or raw strings)."""
        tasks = [self._search(q) for q in sub_questions]
        batches = await asyncio.gather(*tasks)

        results = []
        for contract, batch in zip(sub_questions, batches):
            if isinstance(contract, str):
                question_text = contract
            elif isinstance(contract, dict):
                # The contract's question is what flows to the summarizer;
                # minimum_sources/stop_condition stay with the contract owner.
                question_text = str(contract.get("question", "")).strip() or str(contract)
            else:
                # (question_text, search_type) pair from the workflow fan-out.
                question_text, _ = _split_query(contract)
            for r in batch:
                r.sub_question = question_text
                results.append(r.to_dict())

        return results

    async def _search(self, query):
        settings = self.settings
        question_text, search_type = _split_query(query)
        if not question_text:
            return []

        async def _uncached_search():
            collected = []

            async with self.semaphore:
                # Tavily replaces DDG text/news when configured: one paid call
                # with clean markdown beats free snippets. Wikipedia always
                # runs (free, high-trust encyclopedia). Any Tavily failure
                # falls back to DDG inside _tavily_search. News-typed
                # contracts search Tavily's news topic; everything else uses
                # the general topic.
                topic = "news" if search_type.strip().lower() == "news" else "general"
                if _has_tavily_key(settings):
                    collected.extend(await self._tavily_search(question_text, topic=topic))
                else:
                    collected.extend(await self._ddg_text(question_text))
                    collected.extend(await self._ddg_news(question_text))
                collected.extend(await self._wiki(question_text))

            if not collected:
                return []

            ranked = _deduplicate_and_rank(collected, question_text)

            # Fetch content for the top results concurrently (independent I/O).
            # Depth is setting-driven (search_fetch_top_n); fetched text is
            # consumed by the summarizer, which releases it afterwards. One
            # shared client per pass: connection pooling instead of a fresh
            # TLS handshake per page (up to fetch_top_n per sub-question).
            async def _attach(r, client):
                # Tavily results arrive with content: never re-fetch them.
                if r.content:
                    r.content_length = len(r.content)
                    r.is_content_fetched = True
                    return
                content, last_modified = await _fetch_content(r.url, client)
                if content:
                    r.content = content
                    r.content_length = len(content)
                    r.is_content_fetched = True
                    if not r.published_at:
                        r.published_at = last_modified

            fetch_n = max(1, int(getattr(settings, "search_fetch_top_n", 3) or 3))
            async with httpx.AsyncClient(
                timeout=10, headers={"User-Agent": _BROWSER_USER_AGENT}
            ) as fetch_client:
                await asyncio.gather(*(_attach(r, fetch_client) for r in ranked[:fetch_n]))
            return ranked

        key = cache_key("search_query", _normalize_text(question_text))
        try:
            cached = get_cache(settings).get(key)
        except Exception as exc:
            logger.warning("[Search] cache read failed: %s", exc, exc_info=exc)
            cached = None

        if cached is not None:
            logger.info("[Search] cache hit for query: %s", question_text[:60])
            return cached

        logger.info("[Search] cache miss for query: %s", question_text[:60])
        ranked = await _uncached_search()
        try:
            get_cache(settings).set(key, ranked, expire=settings.cache_ttl_sec)
        except Exception as exc:
            logger.warning("[Search] cache write failed, continuing uncached: %s", exc, exc_info=exc)
        return ranked

    # =========================
    # PROVIDERS
    # =========================

    async def _tavily_search(self, query, query_domains: list[str] | None = None,
                             topic: str = "general"):
        """Tavily web search (paid, free tier available). Falls back to DDG
        on any failure — a bad paid call must never be worse than free."""
        api_key = _real_key(self.settings.tavily_api_key)
        clean_query, include_domains = _prepare_tavily_query(query, query_domains)
        topic = str(topic or "general").strip().lower() or "general"
        if topic not in ("general", "news"):
            topic = "general"
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": api_key,
                        "query": clean_query,
                        "search_depth": "basic",
                        "topic": topic,
                        "max_results": 8,
                        "include_answer": False,
                        "include_domains": include_domains or None,
                        # Tavily-side cache: repeated queries cost no credits.
                        "use_cache": True,
                    },
                )
                r.raise_for_status()
                payload = r.json()
        except Exception as exc:
            logger.warning("[Search] Tavily failed, falling back to DDG: %s", exc, exc_info=exc)
            results = await self._ddg_text(query)
            results.extend(await self._ddg_news(query))
            return results
        if not isinstance(payload, dict):
            logger.warning("[Search] Tavily returned non-JSON-object payload; falling back to DDG")
            results = await self._ddg_text(query)
            results.extend(await self._ddg_news(query))
            return results
        mapped = _tavily_to_results(payload, clean_query)
        if not mapped:
            # Zero usable hits (e.g. rejected query) is a failure, not an
            # answer — upstream treats it the same and the caller must still
            # get the free DDG fallback instead of wiki-only coverage.
            logger.warning("[Search] Tavily returned no usable results, falling back to DDG")
            results = await self._ddg_text(query)
            results.extend(await self._ddg_news(query))
            return results
        return mapped

    async def _ddg_text(self, query):
        def _search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=8))

        try:
            # ddgs is blocking with no timeout of its own: bound the wait so
            # a stalled DDG call degrades to [] instead of hanging the run.
            # Uses the existing (previously unwired) search_timeout_sec.
            rows = await asyncio.wait_for(
                asyncio.to_thread(_search),
                timeout=float(getattr(self.settings, "search_timeout_sec", 20) or 20),
            )
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
            # Same bound as _ddg_text: blocking ddgs must never hang the run.
            rows = await asyncio.wait_for(
                asyncio.to_thread(_search),
                timeout=float(getattr(self.settings, "search_timeout_sec", 20) or 20),
            )
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
            # Wikipedia 403s script default UAs ("Please set a user-agent");
            # without this header every _wiki call degrades to [].
            async with httpx.AsyncClient(
                timeout=10, headers={"User-Agent": _WIKI_USER_AGENT}
            ) as client:
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