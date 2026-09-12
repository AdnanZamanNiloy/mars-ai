"""Source intelligence: primary-source registry, tiering, and freshness.

Why this module exists
----------------------
Before this, source quality was a five-branch TLD guess (`.gov` -> 0.92,
`.com` -> 0.58) plus a 19-entry authority set. That conflates two different
things a research system must keep separate:

  authority  — how much weight a domain's word carries
  primacy    — whether the domain PUBLISHED the fact or REPORTED on it

A statistics agency releasing a number and a blog quoting that number can
score identically on authority heuristics, yet only one is citable evidence.
This module classifies every URL on both axes, so the planner can steer
searches at primary publishers, the ranker can prefer them, and the
confidence engine can report what share of a report rests on primary
evidence.

Everything here is pure, offline, and dependency-free: no network, no LLM.
`evidence_utils.source_reliability_score` delegates to `authority_score`,
and every score the old function returned for a given URL is preserved
(see tests_mars_v3/test_sources.py::test_legacy_scores_preserved) so
existing thresholds elsewhere in the pipeline keep their meaning.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------

TIER_OFFICIAL = "official"        # statute/statistic/filing publisher; the record itself
TIER_PEER_REVIEWED = "peer_reviewed"  # journals, indexed proceedings
TIER_PREPRINT = "preprint"        # arXiv/bioRxiv/SSRN: primary but unrefereed
TIER_REFERENCE = "reference"      # encyclopedias, standards references
TIER_INDUSTRY = "industry"        # trade bodies, vendor docs, research firms
TIER_MEDIA = "media"              # journalism
TIER_SECONDARY = "secondary"      # aggregators, general web
TIER_LOW = "low"                  # UGC, SEO farms, social

TIER_ORDER: Tuple[str, ...] = (
    TIER_OFFICIAL,
    TIER_PEER_REVIEWED,
    TIER_PREPRINT,
    TIER_REFERENCE,
    TIER_INDUSTRY,
    TIER_MEDIA,
    TIER_SECONDARY,
    TIER_LOW,
)

# Authority ceilings per tier. Deliberately aligned with the legacy scale
# so downstream thresholds (0.55 keep, 0.60 strong, 0.62 facts) still mean
# what they meant before this module existed.
TIER_AUTHORITY: Dict[str, float] = {
    TIER_OFFICIAL: 0.95,
    TIER_PEER_REVIEWED: 0.95,
    TIER_PREPRINT: 0.88,
    TIER_REFERENCE: 0.86,
    TIER_INDUSTRY: 0.70,
    TIER_MEDIA: 0.68,
    TIER_SECONDARY: 0.58,
    TIER_LOW: 0.0,
}

# Tiers whose documents ARE the evidence rather than commentary on it.
PRIMARY_TIERS: frozenset[str] = frozenset(
    {TIER_OFFICIAL, TIER_PEER_REVIEWED, TIER_PREPRINT}
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Statistical / regulatory / treaty publishers. These publish the numbers
# everyone else quotes, which is exactly why a research system should reach
# them directly.
OFFICIAL_DOMAINS: Set[str] = {
    # Inter-governmental & multilateral
    "who.int", "un.org", "unctad.org", "unesco.org", "unep.org", "undp.org",
    "iaea.org", "irena.org", "iea.org", "oecd.org", "worldbank.org",
    "data.worldbank.org", "imf.org", "bis.org", "wto.org", "ilo.org",
    "fao.org", "ipcc.ch", "itu.int", "wipo.int", "eia.gov",
    # Statistics agencies & central banks
    "ec.europa.eu", "eurostat.ec.europa.eu", "ecb.europa.eu",
    "federalreserve.gov", "bls.gov", "census.gov", "bea.gov", "cbo.gov",
    "gao.gov", "sec.gov", "cdc.gov", "nih.gov", "fda.gov", "epa.gov",
    "energy.gov", "nrel.gov", "nasa.gov", "noaa.gov", "usgs.gov",
    "ons.gov.uk", "bankofengland.co.uk", "gov.uk", "statcan.gc.ca",
    "abs.gov.au", "rbi.org.in", "mospi.gov.in", "stat.go.jp", "boj.or.jp",
    "destatis.de", "bundesbank.de", "insee.fr", "istat.it",
    "bbs.gov.bd", "bb.org.bd",
    # Standards & registries
    "iso.org", "ietf.org", "rfc-editor.org", "w3.org", "ieee.org",
    "nist.gov", "iec.ch", "unicode.org", "ecma-international.org",
    "clinicaltrials.gov", "eur-lex.europa.eu", "congress.gov",
    "federalregister.gov", "supremecourt.gov", "courtlistener.com",
}

# Suffix-matched official patterns: a host ending in any of these is treated
# as an official publisher even when the exact host is not registered.
OFFICIAL_SUFFIXES: Tuple[str, ...] = (
    ".gov", ".gov.uk", ".gov.au", ".gov.in", ".gov.bd", ".gov.ca",
    ".gc.ca", ".gouv.fr", ".go.jp", ".go.kr", ".govt.nz", ".gov.sg",
    ".europa.eu", ".int", ".mil",
)

PEER_REVIEWED_DOMAINS: Set[str] = {
    "nature.com", "science.org", "sciencemag.org", "cell.com",
    "thelancet.com", "nejm.org", "bmj.com", "jamanetwork.com",
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "pnas.org",
    "sciencedirect.com", "springer.com", "link.springer.com",
    "wiley.com", "onlinelibrary.wiley.com", "tandfonline.com",
    "sagepub.com", "journals.sagepub.com", "cambridge.org",
    "academic.oup.com", "oup.com", "acm.org", "dl.acm.org",
    "ieeexplore.ieee.org", "aps.org", "journals.aps.org", "iopscience.iop.org",
    "aeaweb.org", "jstor.org", "plos.org", "journals.plos.org",
    "frontiersin.org", "mdpi.com", "elifesciences.org", "aclanthology.org",
    "jmlr.org", "neurips.cc", "proceedings.mlr.press", "openreview.net",
    "doi.org", "crossref.org", "semanticscholar.org", "nber.org",
    "royalsocietypublishing.org", "annualreviews.org",
}

PREPRINT_DOMAINS: Set[str] = {
    "arxiv.org", "biorxiv.org", "medrxiv.org", "chemrxiv.org",
    "ssrn.com", "papers.ssrn.com", "osf.io", "preprints.org",
    "researchsquare.com", "hal.science", "econpapers.repec.org", "repec.org",
}

REFERENCE_DOMAINS: Set[str] = {
    "britannica.com", "plato.stanford.edu", "iep.utm.edu",
    "encyclopedia.com", "oxfordreference.com", "merriam-webster.com",
    "en.wikipedia.org", "wikipedia.org", "wikidata.org", "ourworldindata.org",
    "mathworld.wolfram.com", "routledge.com",
}

INDUSTRY_DOMAINS: Set[str] = {
    "gartner.com", "forrester.com", "idc.com", "mckinsey.com",
    "bcg.com", "deloitte.com", "pwc.com", "kpmg.com", "ey.com",
    "statista.com", "spglobal.com", "moodys.com", "fitchratings.com",
    "bloomberg.com", "woodmac.com", "bnef.com", "rystadenergy.com",
    "seia.org", "iea-pvps.org", "epri.com", "lazard.com",
    "huggingface.co", "paperswithcode.com", "github.com", "openml.org",
    "mlcommons.org", "kaggle.com", "developer.mozilla.org",
    "docs.python.org", "python.org", "pytorch.org", "tensorflow.org",
    "kubernetes.io", "postgresql.org", "sqlite.org", "openai.com",
    "anthropic.com", "deepmind.google", "research.google",
}

MEDIA_DOMAINS: Set[str] = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "ft.com",
    "economist.com", "wsj.com", "nytimes.com", "washingtonpost.com",
    "theguardian.com", "npr.org", "cnbc.com", "axios.com", "politico.com",
    "arstechnica.com", "theverge.com", "wired.com", "technologyreview.com",
    "ieee-spectrum.org", "spectrum.ieee.org", "nikkei.com", "scmp.com",
    "aljazeera.com", "dw.com", "france24.com", "thedailystar.net",
}

# User-generated / SEO / social. Superset of the legacy LOW_QUALITY_DOMAINS
# list, which stays exported from evidence_utils for compatibility.
LOW_TRUST_DOMAINS: Set[str] = {
    "reddit.com", "quora.com", "zhihu.com", "baidu.com", "sohu.com",
    "csdn.net", "medium.com", "blogspot.com", "substack.com",
    "wordpress.com", "youtube.com", "youtu.be", "tiktok.com",
    "pinterest.com", "whatfix.com", "facebook.com", "instagram.com",
    "twitter.com", "x.com", "linkedin.com", "tumblr.com", "wattpad.com",
    "answers.com", "ask.com", "wikihow.com", "ehow.com", "geeksforgeeks.org",
    "w3schools.com", "tutorialspoint.com", "javatpoint.com",
    "simplilearn.com", "coursera.org", "udemy.com", "scribd.com",
    "slideshare.net", "academia.edu", "chegg.com", "coursehero.com",
    "studocu.com", "brainly.com", "amazon.com", "ebay.com", "alibaba.com",
    "pinterest.co.uk", "vk.com", "weibo.com", "telegra.ph",
}

# Content-farm shapes that no domain list can keep up with. Matched against
# the full URL, not the host, so "/blog/top-10-best-..." style paths are
# recognized on otherwise-neutral domains.
SEO_PATH_RE = re.compile(
    r"/(?:top-?\d+|best-\d+|\d+-best|listicle|coupon|deals?|"
    r"casino|betting|essay-?writing|buy-?now)/",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceProfile:
    """Everything the pipeline knows about a URL before reading its text."""

    url: str
    domain: str
    tier: str
    authority: float
    is_primary: bool
    reasons: Tuple[str, ...] = ()

    @property
    def is_usable(self) -> bool:
        return self.authority > 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "url": self.url,
            "domain": self.domain,
            "tier": self.tier,
            "authority": round(self.authority, 3),
            "is_primary": self.is_primary,
            "reasons": list(self.reasons),
        }


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------

_TRACKING_PARAM_RE = re.compile(
    r"^(?:utm_|ga_|gclid$|fbclid$|mc_|ref$|ref_src$|referrer$|source$|"
    r"spm$|_hsenc$|_hsmi$|igshid$|si$|amp$)",
    re.IGNORECASE,
)


def extract_domain(url: str) -> str:
    """Lowercase host with a leading `www.` removed, or "" when unparseable."""
    try:
        parsed = urlparse((url or "").strip())
    except ValueError:
        return ""
    host = (parsed.netloc or "").lower().split("@")[-1]
    if ":" in host:
        host = host.split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def canonical_url(url: str) -> str:
    """Collapse the many URLs that name one document into a single key.

    Strips tracking parameters, fragments, default ports, `amp` suffixes,
    duplicate slashes and a trailing slash, and normalizes the scheme to
    https. Without this, `?utm_source=...`, `#section`, and `http://` copies
    of the same page each consumed a separate fetch, a separate cache entry,
    and a separate slot in the "distinct sources" counts that gate the
    critic — inflating perceived source diversity with duplicates.
    """
    raw = (url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
    except ValueError:
        return raw
    if not parsed.netloc:
        return raw

    host = (parsed.netloc or "").lower().split("@")[-1]
    if host.endswith(":80") or host.endswith(":443"):
        host = host.rsplit(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]

    path = re.sub(r"/{2,}", "/", parsed.path or "")
    if path.endswith("/amp"):
        path = path[:-4]
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]

    query_pairs = [
        (k, v)
        for k, v in parse_qsl(parsed.query or "", keep_blank_values=True)
        if not _TRACKING_PARAM_RE.match(k)
    ]
    query_pairs.sort()

    return urlunparse(("https", host, path, "", urlencode(query_pairs), ""))


def same_document(url_a: str, url_b: str) -> bool:
    a, b = canonical_url(url_a), canonical_url(url_b)
    return bool(a) and a == b


def _matches(domain: str, registry: Iterable[str]) -> bool:
    for entry in registry:
        entry = entry.lower().strip()
        if not entry:
            continue
        if domain == entry or domain.endswith(f".{entry}"):
            return True
    return False


def _has_suffix(domain: str, suffixes: Sequence[str]) -> bool:
    return any(domain == s.lstrip(".") or domain.endswith(s) for s in suffixes)


def classify_source(url: str) -> SourceProfile:
    """Assign tier, authority and primacy to a URL. Pure and deterministic."""
    domain = extract_domain(url)
    if not domain:
        return SourceProfile(url or "", "", TIER_LOW, 0.0, False, ("unparseable url",))

    reasons: List[str] = []

    if _matches(domain, LOW_TRUST_DOMAINS):
        return SourceProfile(url, domain, TIER_LOW, 0.0, False, ("low-trust domain",))
    if SEO_PATH_RE.search(url or ""):
        return SourceProfile(url, domain, TIER_LOW, 0.0, False, ("seo content-farm path",))

    tier: Optional[str] = None
    if _matches(domain, OFFICIAL_DOMAINS) or _has_suffix(domain, OFFICIAL_SUFFIXES):
        tier, why = TIER_OFFICIAL, "official publisher"
    elif _matches(domain, PEER_REVIEWED_DOMAINS):
        tier, why = TIER_PEER_REVIEWED, "peer-reviewed venue"
    elif _matches(domain, PREPRINT_DOMAINS):
        tier, why = TIER_PREPRINT, "preprint server"
    elif _matches(domain, REFERENCE_DOMAINS):
        tier, why = TIER_REFERENCE, "reference work"
    elif _matches(domain, INDUSTRY_DOMAINS):
        tier, why = TIER_INDUSTRY, "industry/vendor authority"
    elif _matches(domain, MEDIA_DOMAINS):
        tier, why = TIER_MEDIA, "established journalism"
    elif domain.endswith(".edu") or domain.endswith(".ac.uk") or ".edu." in domain:
        tier, why = TIER_PEER_REVIEWED, "academic institution"
    else:
        tier, why = TIER_SECONDARY, "unregistered domain"
    reasons.append(why)

    authority = TIER_AUTHORITY[tier]

    # Legacy TLD nudges, retained so unregistered domains keep their old
    # relative ordering (.org above .co above .com above the rest).
    if tier == TIER_SECONDARY:
        if domain.endswith(".org"):
            authority = 0.75
            reasons.append(".org nonprofit")
        elif domain.endswith(".co"):
            authority = 0.62
        elif domain.endswith(".com"):
            authority = 0.58
        else:
            authority = 0.55

    # Institutional floor: a university-hosted reference work (the Stanford
    # Encyclopedia, IEP) is not a mere tertiary source. Applied after tiering
    # so it lifts, never lowers.
    if domain.endswith(".edu") or domain.endswith(".ac.uk") or ".edu." in domain:
        if authority < 0.92:
            authority = 0.92
            reasons.append("academic institution floor")

    return SourceProfile(
        url=url,
        domain=domain,
        tier=tier,
        authority=round(authority, 3),
        is_primary=tier in PRIMARY_TIERS,
        reasons=tuple(reasons),
    )


def authority_score(url: str) -> float:
    """0.0-1.0 authority. 0.0 means "do not cite" (blocked/low-trust)."""
    return classify_source(url).authority


def is_primary_source(url: str) -> bool:
    return classify_source(url).is_primary


def primary_source_share(urls: Iterable[str]) -> float:
    """Fraction of DISTINCT documents that are primary sources."""
    seen: Dict[str, bool] = {}
    for url in urls or []:
        key = canonical_url(url)
        if key and key not in seen:
            seen[key] = classify_source(url).is_primary
    if not seen:
        return 0.0
    return sum(1 for v in seen.values() if v) / len(seen)


def tier_distribution(urls: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    seen: Set[str] = set()
    for url in urls or []:
        key = canonical_url(url)
        if not key or key in seen:
            continue
        seen.add(key)
        counts[classify_source(url).tier] = counts.get(classify_source(url).tier, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Search steering: which publishers to aim a query at
# ---------------------------------------------------------------------------

# Domain hints per (search_type, domain) used to build `site:`-scoped query
# variants. Two or three hosts per bucket, never a wall of operators: an
# over-constrained query returns nothing, which is worse than a broad one.
PRIMARY_SOURCE_HINTS: Dict[str, Tuple[str, ...]] = {
    "statistical:economics": ("worldbank.org", "imf.org", "oecd.org"),
    "statistical:policy": ("oecd.org", "europa.eu", "gov.uk"),
    "statistical:science": ("nasa.gov", "noaa.gov", "who.int"),
    "statistical:general": ("worldbank.org", "oecd.org", "census.gov"),
    "statistical:machine_learning": ("paperswithcode.com", "mlcommons.org"),
    "statistical:software": ("stackoverflow.blog", "github.blog"),
    "academic:machine_learning": ("arxiv.org", "aclanthology.org", "openreview.net"),
    "academic:science": ("nature.com", "science.org", "pubmed.ncbi.nlm.nih.gov"),
    "academic:economics": ("nber.org", "repec.org", "aeaweb.org"),
    "academic:philosophy": ("plato.stanford.edu", "iep.utm.edu"),
    "academic:general": ("arxiv.org", "doi.org", "semanticscholar.org"),
    "academic:legal": ("eur-lex.europa.eu", "courtlistener.com"),
    "academic:policy": ("oecd.org", "nber.org"),
    "news:general": ("reuters.com", "apnews.com", "ft.com"),
    "comparison:general": (),
    "encyclopedia:general": ("britannica.com",),
    "encyclopedia:philosophy": ("plato.stanford.edu",),
}

# Terms that steer a general web search toward primary documents even when
# no site: operator applies.
PRIMARY_INTENT_TERMS: Dict[str, Tuple[str, ...]] = {
    "statistical": ("official statistics", "dataset", "annual report"),
    "academic": ("peer-reviewed study", "paper", "doi"),
    "news": ("press release", "official announcement"),
    "comparison": ("benchmark results", "side-by-side"),
    "encyclopedia": ("definition", "overview"),
}


def primary_source_hints(search_type: str, domain: str = "general") -> Tuple[str, ...]:
    """Preferred publisher hosts for a (search_type, domain) pair."""
    st = (search_type or "").strip().lower() or "encyclopedia"
    dm = (domain or "").strip().lower() or "general"
    return (
        PRIMARY_SOURCE_HINTS.get(f"{st}:{dm}")
        or PRIMARY_SOURCE_HINTS.get(f"{st}:general")
        or ()
    )


def primary_intent_terms(search_type: str) -> Tuple[str, ...]:
    return PRIMARY_INTENT_TERMS.get((search_type or "").strip().lower(), ())


def build_primary_source_query(
    question: str, search_type: str, domain: str = "general", max_sites: int = 2
) -> str:
    """A `site:`-scoped variant of `question` aimed at primary publishers.

    Returns "" when no hint applies, so callers can skip the extra search
    instead of firing a duplicate of the plain query.
    """
    hints = primary_source_hints(search_type, domain)[: max(0, max_sites)]
    text = re.sub(r"\s+", " ", (question or "")).strip()
    if not text or not hints:
        return ""
    return f"{text} " + " OR ".join(f"site:{h}" for h in hints)


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------

# Half-lives in days: how fast a claim of this kind goes stale. Used to turn
# a publish date into a 0-1 recency weight instead of the previous
# all-or-nothing "has a date / has no date".
FRESHNESS_HALF_LIFE: Dict[str, float] = {
    "news": 120.0,
    "statistical": 550.0,
    "comparison": 730.0,
    "academic": 1460.0,
    "encyclopedia": 2200.0,
    "default": 730.0,
}


def _as_date(value: str) -> Optional[date]:
    text = (value or "").strip()
    if not text:
        return None
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate).date()
        except (TypeError, ValueError):
            continue
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except (TypeError, ValueError):
            continue
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(text).date()
    except (TypeError, ValueError, IndexError):
        return None


def freshness_score(
    published_at: str,
    search_type: str = "default",
    today: Optional[date] = None,
    unknown_score: float = 0.45,
) -> float:
    """Exponential-decay recency weight in [0, 1].

    `unknown_score` is deliberately mid-scale, not 0: an undated page is
    unknown, not stale, and penalizing it as stale would systematically
    demote primary PDFs (which rarely expose dates) in favour of blogs
    (which always do).
    """
    parsed = _as_date(published_at)
    if parsed is None:
        return unknown_score
    ref = today or datetime.now(timezone.utc).date()
    age_days = max(0.0, (ref - parsed).days)
    half_life = FRESHNESS_HALF_LIFE.get(
        (search_type or "default").strip().lower(), FRESHNESS_HALF_LIFE["default"]
    )
    return round(0.5 ** (age_days / half_life), 4)


def evidence_freshness(
    items: Sequence[Dict[str, object]],
    search_type_key: str = "search_type",
    date_key: str = "published_at",
    today: Optional[date] = None,
) -> float:
    """Mean freshness across items; 0.45 (unknown) when nothing is dated."""
    scores = [
        freshness_score(
            str(item.get(date_key, "") or ""),
            str(item.get(search_type_key, "default") or "default"),
            today=today,
        )
        for item in items or []
    ]
    return round(sum(scores) / len(scores), 4) if scores else 0.45
