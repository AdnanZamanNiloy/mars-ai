"""Search & retrieval layer.

Correctness fixes (each was silently costing evidence)
------------------------------------------------------
1. REDIRECTS WERE NOT FOLLOWED. `httpx` does not follow redirects by default and
   `_fetch_with_client` only accepted status 200, so every page behind a 301/302
   — which is most canonical URLs, every http->https upgrade, and nearly all
   news sites — returned empty text. The result looked like "content fetch
   failed" but was really "we never asked for the real page".

2. BLOCKLIST MATCHED SUBSTRINGS OF THE WHOLE URL. `any(b in r.url ...)` blocked
   any URL containing "x.com" anywhere, including paths and unrelated hosts like
   "matrix.com", while missing nothing it was meant to catch. Matching is now
   host-based.

3. PLANNED VARIANTS WERE NEVER SEARCHED. The planner is instructed to emit 1-2
   alternate phrasings per sub-question specifically because different phrasings
   retrieve different sources — and `run_search` only ever searched
   `question`. Variants and the primary-source query are now searched and their
   results merged before ranking.

4. CACHE KEY IGNORED search_type. A "news" search and an "encyclopedia" search
   for the same words shared one cache entry, so whichever ran first defined
   both for the whole TTL.

5. DUPLICATE URLS COUNTED AS DISTINCT SOURCES. Dedup compared raw URL strings,
   so `?utm_source=`, `#section`, `http://` and trailing-slash variants each
   consumed a fetch slot and each inflated the "distinct sources" count that
   gates the critic.

Capability upgrades
-------------------
* Primary-source providers (arXiv, Crossref, Wikipedia extracts) — free, no key,
  and they return the documents that secondary sources quote.
* Retries + per-provider circuit breakers, so a rate-limited provider costs one
  timeout per cooldown instead of one per sub-question per pass.
* Ranking that accounts for recency and primary-source status rather than
  authority plus snippet length.
* Response size caps: an unbounded `r.text` on a large document is a real
  memory risk on an 8GB host.
"""
from __future__ import annotations

import asyncio
import html
import re
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import quote, urlparse
from xml.etree import ElementTree

import httpx

from app.core.cache import cache_key, get_cache
from app.core.config import Settings
from app.core.llm import _real_key
from app.core.logging import get_logger

from app.agents.planner import SubQuestion
from app.agents.reliability import (
    RetryPolicy,
    call_protected,
    gather_bounded,
    get_breaker,
)
from app.agents.sources import (
    canonical_url,
    classify_source,
    extract_domain as _host,
    freshness_score,
    is_primary_source,
)

logger = get_logger(__name__)

# Bump when the shape of a cached search payload changes; stale entries would
# otherwise serve pre-fix results for a full TTL.
SEARCH_CACHE_VERSION = "search-v3"

# ddgs is optional. A missing dependency should degrade one provider, not stop
# the module from importing (which would take the whole pipeline down).
try:  # pragma: no cover - import shape depends on environment
    from ddgs import DDGS  # type: ignore
except Exception:  # noqa: BLE001
    DDGS = None  # type: ignore
    logger.warning("[Search] ddgs unavailable; DuckDuckGo providers disabled")


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
    # Publish date when the provider supplies one (DDG news `date`, Wikipedia
    # revision `timestamp`, arXiv `published`, Crossref `issued`) or a fetch
    # Last-Modified header. "" means unknown — never synthesized.
    published_at: str = ""
    # Which planned query actually produced this hit: the base question, a
    # variant, or the primary-source-scoped variant. Kept so the trace can show
    # which phrasings are earning their cost.
    matched_query: str = ""
    is_primary: bool = False

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
            "matched_query": self.matched_query,
            "is_primary": self.is_primary,
        }


# =============================================================================
# UTILITIES
# =============================================================================

def _normalize_text(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


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
    return _host(url)


# =============================================================================
# CONFIG
# =============================================================================

BLOCKED_DOMAINS = {
    "pinterest.com", "instagram.com", "facebook.com",
    "twitter.com", "x.com", "tiktok.com", "youtube.com",
    "amazon.com", "ebay.com", "quora.com",
}


def _is_blocked(url: str) -> bool:
    """Host-based blocklist check.

    The previous substring test (`"x.com" in url`) blocked any URL whose path or
    host merely contained a blocked string — "matrix.com", "netflix.com/x.com/",
    a query parameter mentioning youtube.com — and was therefore both too
    aggressive and unpredictable.
    """
    host = _host(url)
    if not host:
        return True
    return any(host == b or host.endswith(f".{b}") for b in BLOCKED_DOMAINS)


# Wikimedia (and many publishers) 403 script default UAs. Contact-style string
# per https://meta.wikimedia.org/wiki/User-Agent_policy.
_WIKI_USER_AGENT = "MARS-research/1.0 (personal research assistant)"

# Generic page fetch uses a browser UA: many publishers block identifying/script
# UAs on article pages (the MediaWiki API above is the exception — it explicitly
# wants an identifying UA).
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)

# Hard ceiling on a single fetched document. `r.text` on an unbounded response
# decodes the whole body into a Python str; one 200MB PDF or misconfigured
# endpoint is enough to OOM an 8GB host mid-run.
MAX_FETCH_BYTES = 3_000_000

# Content types worth reading. Anything else (video, images, archives) costs
# bandwidth and yields nothing.
_READABLE_TYPES = ("text/html", "text/plain", "application/xhtml", "application/pdf",
                   "application/json", "text/xml", "application/xml")


# =============================================================================
# SCORING
# =============================================================================

def _score_result(result: SearchResult, query: str) -> float:
    """Rank a result before any content is fetched.

    Changes from the previous formula:
      * primary sources get an explicit boost — the point of reaching them
      * recency is scored by decay against the result's own search_type, so a
        2019 news hit sinks while a 2019 paper does not
      * the flat 0.15 Wikipedia penalty is kept but no longer applies when
        Wikipedia is the only high-authority hit for an encyclopedia-typed
        question, which is exactly when it is the right answer
    """
    profile = classify_source(result.url)
    base = profile.authority

    snippet = result.snippet or ""
    content = result.content or ""

    richness = min(0.10, len(snippet) / 1500)
    relevance = _relevance_score(query, snippet + " " + content)
    content_bonus = 0.06 if result.is_content_fetched else 0.0
    primary_bonus = 0.10 if profile.is_primary else 0.0
    recency = freshness_score(result.published_at, result.search_type or "default")
    recency_weight = 0.12 if (result.search_type or "").lower() == "news" else 0.06
    wiki_penalty = 0.15 if "wikipedia.org" in (result.url or "") else 0.0

    return (
        base
        + richness
        + (relevance * 0.25)
        + content_bonus
        + primary_bonus
        + (recency * recency_weight)
        - wiki_penalty
    )


# =============================================================================
# RANK + DEDUP
# =============================================================================

def _deduplicate_and_rank(results, query, max_results=10, search_type: str = ""):
    """Canonical-URL dedup, scoring, near-duplicate removal, domain diversity."""
    seen: set = set()
    filtered: List[SearchResult] = []

    for r in results:
        if not r.url:
            continue
        key = canonical_url(r.url)
        if not key or key in seen:
            continue
        if _is_blocked(r.url):
            continue
        seen.add(key)
        if search_type and (not r.search_type or r.search_type == "general"):
            r.search_type = search_type
        r.is_primary = is_primary_source(r.url)
        filtered.append(r)

    for r in filtered:
        r.reliability_score = _score_result(r, query)

    ranked = sorted(filtered, key=lambda r: r.reliability_score, reverse=True)

    # Near-duplicate snippets. Restricted to same-domain pairs plus very high
    # overlap across domains: two independent publishers describing the same
    # fact in similar words is CORROBORATION, and dropping the second copy is
    # how the pipeline used to destroy its own cross-source agreement signal
    # before it was ever measured.
    diverse: List[SearchResult] = []
    for r in ranked:
        duplicate = False
        for kept in diverse:
            same_host = _domain(r.url) == _domain(kept.url)
            threshold = 0.6 if same_host else 0.85
            if _is_semantic_duplicate(r.snippet, kept.snippet, threshold):
                duplicate = True
                break
        if not duplicate:
            diverse.append(r)

    selected: List[SearchResult] = []
    domain_count: Dict[str, int] = {}
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
    """Extract readable text while dropping non-visible markup."""

    _SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg",
                            "nav", "footer", "form", "aside"})
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


def _clean_html(raw: str, max_chars: int = 12000) -> str:
    """HTML -> readable text.

    The character cap rose from 6000 to 12000: the summarizer now chunks long
    documents, so truncating the source at 6000 characters was discarding
    material the extractor could use. Fetch-side caps still bound memory.
    """
    try:
        extractor = _VisibleTextExtractor()
        extractor.feed(raw or "")
        extractor.close()
        text = extractor.text()
    except Exception as exc:
        logger.warning("[Search] HTML clean failed, using regex fallback: %s", exc, exc_info=exc)
        text = re.sub(r"<[^>]+>", " ", raw or "")
    text = html.unescape(text)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)
    text = re.sub(r"[ \t\f\v]{2,}", " ", text).strip()
    return text[:max_chars]


BLOCK_PAGE_PHRASES = (
    "access denied",
    "verify you are human",
    "enable javascript",
    "please complete the security check",
    "unusual traffic from your computer network",
    "are you a robot",
    "request blocked",
)


def _looks_like_block_page(text: str) -> bool:
    lowered = (text or "").lower()
    return len(lowered) < 600 and any(p in lowered for p in BLOCK_PAGE_PHRASES)


def _extract_pdf_text(content: bytes, url: str) -> str:
    """Best-effort first pages of a PDF. PyMuPDF is optional."""
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
            for page in doc[:8]:
                pages.append(page.get_text())
        return re.sub(r"\s+", " ", "\n".join(pages)).strip()[:12000]
    except Exception as exc:
        logger.warning("[Search] PDF extract failed for %s: %s", url[:80], exc, exc_info=exc)
        return ""


async def _fetch_content(url: str, client: httpx.AsyncClient | None = None):
    """Fetch + clean page text. Returns (text, last_modified_header_or_empty).

    A shared client (connection pooling) is passed on the hot path; when None, a
    throwaway client is created so unit tests and one-off callers keep working.
    """
    try:
        if client is None:
            async with httpx.AsyncClient(
                timeout=12,
                follow_redirects=True,
                headers={"User-Agent": _BROWSER_USER_AGENT},
            ) as owned:
                return await _fetch_with_client(owned, url)
        return await _fetch_with_client(client, url)
    except Exception as exc:
        logger.warning("[Search] content fetch failed for %s: %s", url[:80], exc, exc_info=exc)
    return "", ""


async def _fetch_with_client(client: httpx.AsyncClient, url: str):
    """One page fetch with redirects, type filtering and a size ceiling.

    Accepting only status 200 previously discarded every redirected page,
    because the default httpx client does not follow redirects. Both are fixed
    here: redirects are followed and the check is `is_success`.
    """
    r = await client.get(url, follow_redirects=True)
    if not r.is_success:
        logger.debug("[Search] fetch %s returned %s", url[:80], r.status_code)
        return "", ""

    content_type = str(r.headers.get("content-type", "") or "").lower()
    last_modified = str(r.headers.get("last-modified", "") or "")

    is_pdf = "application/pdf" in content_type or url.lower().split("?")[0].endswith(".pdf")
    if not is_pdf and content_type and not any(t in content_type for t in _READABLE_TYPES):
        logger.debug("[Search] skipping unreadable content-type %s for %s", content_type, url[:60])
        return "", ""

    body = r.content
    if len(body) > MAX_FETCH_BYTES:
        logger.warning(
            "[Search] truncating oversized response (%d bytes) from %s",
            len(body), url[:80],
        )
        body = body[:MAX_FETCH_BYTES]

    if is_pdf:
        return _extract_pdf_text(body, url), last_modified

    try:
        raw = body.decode(r.encoding or "utf-8", errors="replace")
    except (LookupError, UnicodeDecodeError):
        raw = body.decode("utf-8", errors="replace")

    text = _clean_html(raw)
    if _looks_like_block_page(text):
        logger.warning("[Search] block page detected, dropping: %s", url[:80])
        return "", ""
    return text, last_modified


# =============================================================================
# QUERY PREPARATION
# =============================================================================

_SITE_OPERATOR_RE = re.compile(r"site:(\S+)", re.IGNORECASE)
_TAVILY_MAX_QUERY_CHARS = 400


def _prepare_tavily_query(query: Any, query_domains: list[str] | None = None) -> tuple[str, list[str]]:
    """Normalize a query for Tavily: translate Google-style site: operators into
    include_domains (which Tavily rejects inline), collapse whitespace, and cap
    length at Tavily's 400-character limit."""
    text = query if isinstance(query, str) else str(query or "")
    domains: list[str] = []

    def _add_domain(candidate: Any) -> None:
        for part in str(candidate or "").replace(",", " ").split():
            if part.upper() == "OR":
                continue
            domain = part.strip().strip(",").split("/")[0]
            if domain and domain not in domains:
                domains.append(domain)

    for candidate in list(query_domains or []):
        _add_domain(candidate)
    for match in _SITE_OPERATOR_RE.findall(text):
        _add_domain(match)
    text = _SITE_OPERATOR_RE.sub("", text)
    text = re.sub(r"\s+\bOR\b\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_TAVILY_MAX_QUERY_CHARS], domains


def _split_query(query: Any) -> tuple[str, str]:
    """Split any accepted query shape into (question_text, search_type)."""
    if isinstance(query, dict):
        return (
            str(query.get("question", "") or "").strip(),
            str(query.get("search_type", "") or "").strip(),
        )
    if isinstance(query, (tuple, list)) and len(query) == 2:
        return (str(query[0] or "").strip(), str(query[1] or "").strip())
    return (str(query or "").strip(), "")


def contract_queries(contract: Any, max_queries: int = 3) -> List[str]:
    """Every query one delegation contract should actually run.

    Base question, then the planner's variants, then the primary-source-scoped
    variant. Variants exist precisely because different phrasings retrieve
    different documents, and until now they were parsed, validated, stored and
    never used. Capped so a 5-contract plan cannot fan out to 20 searches.
    """
    question, _ = _split_query(contract)
    queries: List[str] = []
    if question:
        queries.append(question)
    if isinstance(contract, dict):
        for variant in contract.get("variants") or ():
            text = re.sub(r"\s+", " ", str(variant or "")).strip()
            if text and text.lower() != question.lower():
                queries.append(text)
        primary = re.sub(r"\s+", " ", str(contract.get("primary_source_query", "") or "")).strip()
        if primary:
            queries.append(primary)
    deduped: List[str] = []
    for q in queries:
        if not any(_semantic_overlap(q, kept) >= 0.92 for kept in deduped):
            deduped.append(q)
        if len(deduped) >= max(1, max_queries):
            break
    return deduped


def _has_tavily_key(settings: Settings) -> bool:
    """A real Tavily key (not empty, not an example placeholder)."""
    return bool(_real_key(getattr(settings, "tavily_api_key", "")))


def _tavily_to_results(payload: Any, query: str) -> List["SearchResult"]:
    """Map a Tavily /search response to SearchResults. Pure — the network call
    stays in _tavily_search so this is unit-testable offline."""
    results: List[SearchResult] = []
    items = payload.get("results", []) if isinstance(payload, dict) else []
    for row in items:
        if not isinstance(row, dict) or not row.get("url"):
            continue
        content = str(row.get("content", "") or row.get("raw_content", "") or "")
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
            content=content[:12000],
            provider="tavily",
            published_at=published,
            matched_query=query,
        ))
    return results


# =============================================================================
# PRIMARY-SOURCE PROVIDERS (free, no API key, return the actual documents)
# =============================================================================

def _arxiv_to_results(xml_text: str, query: str) -> List[SearchResult]:
    """Parse an arXiv Atom feed. Pure, so it is testable without network."""
    out: List[SearchResult] = []
    try:
        root = ElementTree.fromstring(xml_text or "")
    except ElementTree.ParseError as exc:
        logger.warning("[Search] arXiv XML parse failed: %s", exc)
        return out
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("a:entry", ns):
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("a:published", default="", namespaces=ns) or "").strip()
        link = ""
        for candidate in entry.findall("a:link", ns):
            if candidate.get("rel") in (None, "alternate"):
                link = candidate.get("href", "") or ""
                break
        if not link:
            link = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
        if not link or not title:
            continue
        clean_summary = re.sub(r"\s+", " ", summary)
        out.append(SearchResult(
            title=re.sub(r"\s+", " ", title),
            url=link,
            snippet=clean_summary[:1200],
            content=clean_summary[:6000],
            provider="arxiv",
            search_type="academic",
            published_at=published,
            matched_query=query,
        ))
    return out


def _crossref_to_results(payload: Any, query: str) -> List[SearchResult]:
    """Map a Crossref /works response to SearchResults.

    Crossref indexes the DOI record itself: title, venue, date and abstract come
    from the publisher, not from a page that mentions the paper. That is the
    definition of a primary bibliographic source.
    """
    out: List[SearchResult] = []
    items = ((payload or {}).get("message") or {}).get("items") or []
    for row in items:
        if not isinstance(row, dict):
            continue
        doi = str(row.get("DOI", "") or "")
        url = str(row.get("URL", "") or (f"https://doi.org/{doi}" if doi else ""))
        titles = row.get("title") or []
        title = str(titles[0]) if titles else ""
        if not url or not title:
            continue
        abstract = re.sub(r"<[^>]+>", " ", str(row.get("abstract", "") or ""))
        abstract = re.sub(r"\s+", " ", html.unescape(abstract)).strip()
        container = row.get("container-title") or []
        venue = str(container[0]) if container else ""
        parts = ((row.get("issued") or {}).get("date-parts") or [[]])[0]
        published = "-".join(f"{p:02d}" if i else str(p) for i, p in enumerate(parts[:3])) if parts else ""
        summary = abstract or f"{title}. {venue}".strip()
        out.append(SearchResult(
            title=re.sub(r"\s+", " ", title),
            url=url,
            snippet=summary[:1200],
            content=abstract[:6000],
            provider="crossref",
            search_type="academic",
            published_at=published,
            matched_query=query,
        ))
    return out


def _wiki_search_to_results(data: Any, query: str) -> List[SearchResult]:
    results: List[SearchResult] = []
    for item in ((data or {}).get("query") or {}).get("search") or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "") or "")
        if not title:
            continue
        results.append(SearchResult(
            title=title,
            url=f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
            snippet=html.unescape(re.sub(r"<.*?>", "", str(item.get("snippet", "") or ""))),
            provider="wikipedia",
            search_type="encyclopedia",
            published_at=str(item.get("timestamp", "") or ""),
            matched_query=query,
        ))
    return results


# =============================================================================
# MAIN CLIENT
# =============================================================================

class SearchClient:
    """Multi-provider search with per-provider fault isolation.

    Concurrency is bounded twice: `semaphore` limits whole sub-question searches
    (as before) and a separate fetch limit bounds page downloads, so a
    contract that ranks 10 fetchable pages cannot monopolize the event loop or
    the socket pool.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._search_limit = max(1, int(getattr(settings, "max_parallel_search", 3) or 3))
        self.semaphore = asyncio.Semaphore(self._search_limit)
        self._fetch_limit = max(1, int(getattr(settings, "max_parallel_fetch", 4) or 4))
        self._timeout = float(getattr(settings, "search_timeout_sec", 20) or 20)
        self._retry = RetryPolicy(
            attempts=int(getattr(settings, "search_retry_attempts", 3) or 3),
            base_delay=0.5,
            max_delay=6.0,
            timeout=self._timeout,
        )
        self.provider_stats: Dict[str, Dict[str, int]] = {}

    # -- public API --------------------------------------------------------

    async def run_search(
        self, sub_questions: List[Union[SubQuestion, str, tuple]]
    ) -> List[Dict[str, Any]]:
        """Search a batch of delegation contracts (or raw strings).

        Bounded by `max_parallel_search` through `gather_bounded`, which does not
        allocate every coroutine up front — the previous `asyncio.gather` over
        all contracts created every provider client immediately and relied on an
        inner semaphore to throttle them.
        """
        contracts = list(sub_questions or [])
        if not contracts:
            return []

        factories = [(lambda c=c: self._search(c)) for c in contracts]
        batches = await gather_bounded(factories, self._search_limit)

        results: List[Dict[str, Any]] = []
        for contract, batch in zip(contracts, batches):
            if isinstance(batch, BaseException):
                logger.warning(
                    "[Search] contract failed: %s", type(batch).__name__, exc_info=batch
                )
                continue
            if isinstance(contract, str):
                question_text = contract
            elif isinstance(contract, dict):
                question_text = str(contract.get("question", "")).strip() or str(contract)
            else:
                question_text, _ = _split_query(contract)
            for r in batch or []:
                r.sub_question = question_text
                results.append(r.to_dict())

        return results

    # -- per-contract search ----------------------------------------------

    async def _search(self, query) -> List[SearchResult]:
        settings = self.settings
        question_text, search_type = _split_query(query)
        if not question_text:
            return []

        queries = contract_queries(
            query,
            max_queries=int(getattr(settings, "max_queries_per_contract", 3) or 3),
        )
        max_results = int(getattr(settings, "search_max_results", 10) or 10)

        # Cache key now includes the search_type and a version tag. Without the
        # type, a news contract and an encyclopedia contract for the same words
        # shared one entry; without the version, a shape change served stale
        # payloads for a full TTL.
        key = cache_key(
            SEARCH_CACHE_VERSION,
            "search_query",
            (search_type or "general").lower(),
            _normalize_text(" | ".join(queries)),
        )
        cache = None
        try:
            cache = get_cache(settings)
            cached = cache.get(key)
        except Exception as exc:
            logger.warning("[Search] cache read failed: %s", exc, exc_info=exc)
            cached = None

        if cached is not None:
            logger.info("[Search] cache hit for query: %s", question_text[:60])
            return self._decode_cached(cached)

        logger.info(
            "[Search] cache miss; %d query variant(s) for: %s",
            len(queries), question_text[:60],
        )

        async with self.semaphore:
            collected: List[SearchResult] = []
            for q in queries:
                collected.extend(await self._providers_for(q, search_type))

        if not collected:
            return []

        ranked = _deduplicate_and_rank(collected, question_text, max_results, search_type)
        await self._attach_content(ranked)

        try:
            if cache is not None:
                cache.set(
                    key,
                    [self._to_cache(r) for r in ranked],
                    expire=getattr(settings, "cache_ttl_sec", 3600),
                )
        except Exception as exc:
            logger.warning("[Search] cache write failed, continuing uncached: %s", exc, exc_info=exc)
        return ranked

    @classmethod
    def _decode_cached(cls, cached: Any) -> List[SearchResult]:
        """Cache entries are plain dicts (portable across processes and
        pickle-safe). Entries written by an older build stored SearchResult
        objects directly, so both shapes are accepted for one TTL."""
        if not isinstance(cached, list):
            return []
        out: List[SearchResult] = []
        for row in cached:
            if isinstance(row, SearchResult):
                out.append(row)
            elif isinstance(row, dict):
                out.append(cls._from_cache(row))
        return out

    @staticmethod
    def _to_cache(result: SearchResult) -> Dict[str, Any]:
        payload = result.to_dict()
        payload["is_content_fetched"] = result.is_content_fetched
        payload["reliability_score"] = result.reliability_score
        return payload

    @staticmethod
    def _from_cache(row: Dict[str, Any]) -> SearchResult:
        result = SearchResult(
            title=str(row.get("title", "")),
            url=str(row.get("url", "")),
            snippet=str(row.get("snippet", "")),
            content=str(row.get("content", "")),
            provider=str(row.get("provider", "cache")),
            search_type=str(row.get("search_type", "general")),
            published_at=str(row.get("published_at", "")),
            matched_query=str(row.get("matched_query", "")),
        )
        result.reliability_score = float(row.get("reliability_score", 0.0) or 0.0)
        result.content_length = int(row.get("content_length", 0) or 0)
        result.is_content_fetched = bool(row.get("is_content_fetched", bool(result.content)))
        result.is_primary = bool(row.get("is_primary", is_primary_source(result.url)))
        return result

    async def _providers_for(self, query: str, search_type: str) -> List[SearchResult]:
        """Fan out one query across the providers that suit its search_type.

        Provider choice is now type-driven instead of one-size-fits-all: an
        academic contract queries arXiv and Crossref (which return the papers
        themselves), a statistical contract stays on general web search where
        agency pages live, and Wikipedia always runs because it is free and
        high-trust for background.
        """
        stype = (search_type or "").strip().lower()
        tasks: List[Any] = []

        if _has_tavily_key(self.settings):
            topic = "news" if stype == "news" else "general"
            tasks.append(self._tavily_search(query, topic=topic))
        else:
            tasks.append(self._ddg_text(query))
            if stype in ("news", "", "general"):
                tasks.append(self._ddg_news(query))

        if stype == "academic":
            tasks.append(self._arxiv(query))
            tasks.append(self._crossref(query))
        elif stype in ("encyclopedia", "", "general"):
            tasks.append(self._wiki(query))
        elif stype == "statistical":
            tasks.append(self._wiki(query))

        batches = await asyncio.gather(*tasks, return_exceptions=True)
        collected: List[SearchResult] = []
        for batch in batches:
            if isinstance(batch, BaseException):
                logger.warning("[Search] provider error: %s", type(batch).__name__)
                continue
            collected.extend(batch or [])
        return collected

    async def _attach_content(self, ranked: List[SearchResult]) -> None:
        """Download the top N pages concurrently under a fetch bulkhead."""
        fetch_n = max(1, int(getattr(self.settings, "search_fetch_top_n", 3) or 3))
        targets = ranked[:fetch_n]
        if not targets:
            return

        async with httpx.AsyncClient(
            timeout=12,
            follow_redirects=True,
            headers={"User-Agent": _BROWSER_USER_AGENT},
            limits=httpx.Limits(max_connections=self._fetch_limit),
        ) as client:

            async def _attach(r: SearchResult) -> None:
                if r.content:
                    # Providers that already returned text (Tavily, arXiv,
                    # Crossref) must never be re-fetched.
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

            await gather_bounded(
                [(lambda r=r: _attach(r)) for r in targets], self._fetch_limit
            )

    def _count(self, provider: str, outcome: str) -> None:
        bucket = self.provider_stats.setdefault(provider, {"ok": 0, "fail": 0, "results": 0})
        if outcome in bucket:
            bucket[outcome] += 1

    # =========================
    # PROVIDERS
    # =========================

    async def _tavily_search(
        self, query, query_domains: list[str] | None = None, topic: str = "general"
    ) -> List[SearchResult]:
        """Tavily web search. Retries transient failures, opens a circuit after
        repeated ones, and falls back to DDG — a paid call must never leave the
        run worse off than the free path."""
        api_key = _real_key(self.settings.tavily_api_key)
        clean_query, include_domains = _prepare_tavily_query(query, query_domains)
        topic = str(topic or "general").strip().lower() or "general"
        if topic not in ("general", "news"):
            topic = "general"

        async def _call() -> List[SearchResult]:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                r = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": api_key,
                        "query": clean_query,
                        "search_depth": str(getattr(self.settings, "tavily_search_depth", "basic") or "basic"),
                        "topic": topic,
                        "max_results": 8,
                        "include_answer": False,
                        "include_domains": include_domains or None,
                        "use_cache": True,
                    },
                )
                r.raise_for_status()
                payload = r.json()
            if not isinstance(payload, dict):
                raise ValueError("Tavily returned a non-JSON-object payload")
            mapped = _tavily_to_results(payload, clean_query)
            if not mapped:
                # Zero usable hits is a failure, not an answer: the caller must
                # still get the free fallback rather than wiki-only coverage.
                raise ValueError("Tavily returned no usable results")
            self._count("tavily", "ok")
            return mapped

        async def _fallback() -> List[SearchResult]:
            self._count("tavily", "fail")
            results = await self._ddg_text(query)
            results.extend(await self._ddg_news(query))
            return results

        return await call_protected(
            _call,
            name="tavily",
            policy=self._retry,
            breaker=get_breaker("tavily", failure_threshold=3, cooldown=60.0),
            fallback=_fallback,
        )

    async def _ddg_run(self, query: str, kind: str, max_results: int) -> List[dict]:
        if DDGS is None:
            return []

        def _search() -> List[dict]:
            with DDGS() as ddgs:
                method = getattr(ddgs, kind)
                return list(method(query, max_results=max_results))

        # ddgs is blocking with no timeout of its own; bound the wait so a
        # stalled call degrades instead of hanging the run.
        async def _call() -> List[dict]:
            return await asyncio.wait_for(
                asyncio.to_thread(_search), timeout=self._timeout
            )

        async def _empty() -> List[dict]:
            self._count(f"ddg_{kind}", "fail")
            return []

        return await call_protected(
            _call,
            name=f"ddg_{kind}",
            policy=RetryPolicy(attempts=2, base_delay=1.0, max_delay=4.0, timeout=self._timeout),
            breaker=get_breaker(f"ddg_{kind}", failure_threshold=4, cooldown=45.0),
            fallback=_empty,
        )

    async def _ddg_text(self, query) -> List[SearchResult]:
        rows = await self._ddg_run(str(query), "text", 8)
        return [
            SearchResult(
                title=str(r.get("title", "") or ""),
                url=str(r.get("href", "") or ""),
                snippet=str(r.get("body", "") or ""),
                provider="ddg_text",
                published_at=str(r.get("date", "") or ""),
                matched_query=str(query),
            )
            for r in rows if r.get("href")
        ]

    async def _ddg_news(self, query) -> List[SearchResult]:
        rows = await self._ddg_run(str(query), "news", 6)
        return [
            SearchResult(
                title=str(r.get("title", "") or ""),
                url=str(r.get("url", "") or ""),
                snippet=str(r.get("body", "") or ""),
                provider="ddg_news",
                search_type="news",
                published_at=str(r.get("date", "") or ""),
                matched_query=str(query),
            )
            for r in rows if r.get("url")
        ]

    async def _wiki(self, query) -> List[SearchResult]:
        async def _call() -> List[SearchResult]:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": _WIKI_USER_AGENT},
            ) as client:
                r = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": str(query),
                        "srlimit": 5,
                        "format": "json",
                    },
                )
                r.raise_for_status()
                data = r.json()
            self._count("wikipedia", "ok")
            return _wiki_search_to_results(data, str(query))

        async def _empty() -> List[SearchResult]:
            self._count("wikipedia", "fail")
            return []

        return await call_protected(
            _call,
            name="wikipedia",
            policy=RetryPolicy(attempts=2, base_delay=0.5, max_delay=3.0, timeout=self._timeout),
            breaker=get_breaker("wikipedia", failure_threshold=4, cooldown=45.0),
            fallback=_empty,
        )

    async def _arxiv(self, query) -> List[SearchResult]:
        """arXiv Atom API — preprints, free, no key.

        Worth a provider slot because an academic contract that lands on a blog
        summarizing a paper is strictly worse evidence than the paper, and the
        general web search reliably prefers the blog.
        """
        text = re.sub(r"\s+", " ", _SITE_OPERATOR_RE.sub("", str(query))).strip()
        if not text:
            return []

        async def _call() -> List[SearchResult]:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": _WIKI_USER_AGENT},
            ) as client:
                r = await client.get(
                    "https://export.arxiv.org/api/query",
                    params={
                        "search_query": f"all:{text[:200]}",
                        "start": 0,
                        "max_results": 6,
                        "sortBy": "relevance",
                    },
                )
                r.raise_for_status()
                payload = r.text
            self._count("arxiv", "ok")
            return _arxiv_to_results(payload, text)

        async def _empty() -> List[SearchResult]:
            self._count("arxiv", "fail")
            return []

        return await call_protected(
            _call,
            name="arxiv",
            policy=RetryPolicy(attempts=2, base_delay=1.0, max_delay=4.0, timeout=self._timeout),
            breaker=get_breaker("arxiv", failure_threshold=3, cooldown=90.0),
            fallback=_empty,
        )

    async def _crossref(self, query) -> List[SearchResult]:
        """Crossref works API — DOI metadata and abstracts, free, no key."""
        text = re.sub(r"\s+", " ", _SITE_OPERATOR_RE.sub("", str(query))).strip()
        if not text:
            return []

        async def _call() -> List[SearchResult]:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": _WIKI_USER_AGENT},
            ) as client:
                r = await client.get(
                    "https://api.crossref.org/works",
                    params={
                        "query.bibliographic": text[:300],
                        "rows": 5,
                        "select": "DOI,URL,title,abstract,container-title,issued",
                        "sort": "relevance",
                    },
                )
                r.raise_for_status()
                payload = r.json()
            self._count("crossref", "ok")
            return _crossref_to_results(payload, text)

        async def _empty() -> List[SearchResult]:
            self._count("crossref", "fail")
            return []

        return await call_protected(
            _call,
            name="crossref",
            policy=RetryPolicy(attempts=2, base_delay=1.0, max_delay=4.0, timeout=self._timeout),
            breaker=get_breaker("crossref", failure_threshold=3, cooldown=90.0),
            fallback=_empty,
        )
