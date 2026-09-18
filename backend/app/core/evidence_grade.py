"""Claim-level evidence grading — the evidence-quality spine of the pipeline.

Why this module exists
----------------------
The pipeline used to treat a claim as a claim: once it had a URL, a confidence
float and a `verified` flag, everything downstream (confidence, synthesis,
"64 sources" badges) read equally well whether the claim rested on one blog
post or on three independent primary sources. That is source-count thinking,
not evidence thinking.

This module turns every extracted claim into an explicit, deterministic
EvidenceRecord:

    claim -> source tier -> freshness -> verification -> corroboration
          (independence measured by registrable domain, not URL count)
          -> numeric support -> contradiction status
          -> grade A|B|C|D  +  needs_corroboration flag

It is pure and LLM-free by design (same rule as the verifier): evidence
quality must be cheap enough to compute on every claim every pass, or it
won't be computed at all. Every function is additive and total — an unknown
input yields a conservative grade, never an exception.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.agents.evidence_utils import (
    extract_domain,
    extract_numbers,
    numbers_grounded,
    numeric_conflict,
    rare_content_tokens,
    _significant_quantities,
)
from app.agents.sources import classify_source

# --- Grade vocabulary -------------------------------------------------------
# A: primary/peer-reviewed, verified, independently corroborated, numerically
#    grounded — the strongest evidence the system can assemble.
# B: verified and either corroborated OR primary/authoritative; solid.
# C: verified but single-source and secondary, or strong source unverified.
# D: unverified / unattributable / contradicted / unsupported numbers.
GRADE_A = "A"
GRADE_B = "B"
GRADE_C = "C"
GRADE_D = "D"

GRADE_RANK: Dict[str, int] = {GRADE_A: 3, GRADE_B: 2, GRADE_C: 1, GRADE_D: 0}

# Source tiers (from sources.py) that count as primary/authoritative for the
# purpose of corroboration: independent instances of these strongly corroborate.
_PRIMARY_LIKE_TIERS = frozenset({
    "official", "peer_reviewed", "preprint", "reference",
})

# A claim needs at least this many INDEPENDENT (distinct registrable-domain)
# sources before it can grade A. Important claims must reach at least B.
MIN_INDEPENDENT_CORROBORATION = 2

# Two hosts that share a registrable domain (blog.example.com and
# www.example.com) are the same publisher, not independent corroboration.
# Coarse but conservative eTLD+1 handling for the common multi-part suffixes.
_MULTI_PART_TLDS = frozenset({"co.uk", "org.uk", "gov.uk", "ac.uk", "com.au",
                              "co.jp", "com.br", "co.in", "com.cn"})


def registrable_domain(url_or_domain: str) -> str:
    """eTLD+1-ish key: 'news.bbc.co.uk' -> 'bbc.co.uk', 'x.com' -> 'x.com'.

    Corroboration independence must be measured at the PUBLISHER level. Two
    URLs on the same registrable domain (a homepage and a deep article) are
    one source; the old corroboration_count counted them separately, which is
    exactly the false-independence the brief forbids.
    """
    host = extract_domain(url_or_domain) if "/" in str(url_or_domain) else str(url_or_domain or "").strip().lower()
    if not host:
        return ""
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    last2 = ".".join(parts[-2:])
    if last2 in _MULTI_PART_TLDS and len(parts) >= 3:
        return ".".join(parts[-3:])
    return last2


def _claim_is_quantitative(claim: str) -> bool:
    """Any extracted number/percentage/currency/date makes a claim
    quantitative — and therefore subject to the corroboration requirement.
    `extract_numbers` is the single source of truth for what counts."""
    if extract_numbers(claim, limit=1):
        return True
    return any(ch.isdigit() for ch in claim)


def _looks_definitional(claim: str) -> bool:
    low = f" {claim.lower()} "
    return " is " in low or " are " in low or " refers to " in low or " means " in low


@dataclass
class EvidenceRecord:
    """The structured evidence object for one claim (the brief's requirement 5)."""

    claim: str
    source: str = ""
    domain: str = ""
    tier: str = ""
    authority: float = 0.0
    is_primary: bool = False
    verified: bool = False
    verification_score: float = 0.0
    corroborating_domains: List[str] = field(default_factory=list)
    corroboration_count: int = 1            # independent registrable domains
    contradiction_count: int = 0
    numeric_conflict: bool = False
    has_numbers: bool = False
    numbers_supported: bool = True
    freshness: float = 0.0
    is_stale: bool = False
    grade: str = GRADE_D
    needs_corroboration: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "source": self.source,
            "domain": self.domain,
            "tier": self.tier,
            "authority": round(self.authority, 3),
            "is_primary": self.is_primary,
            "verified": self.verified,
            "verification_score": round(self.verification_score, 3),
            "corroborating_domains": list(self.corroborating_domains),
            "corroboration_count": self.corroboration_count,
            "contradiction_count": self.contradiction_count,
            "numeric_conflict": self.numeric_conflict,
            "has_numbers": self.has_numbers,
            "numbers_supported": self.numbers_supported,
            "freshness": round(self.freshness, 3),
            "is_stale": self.is_stale,
            "grade": self.grade,
            "needs_corroboration": self.needs_corroboration,
        }


def independent_corroboration(
    sources: Iterable[str], primary_source: str = ""
) -> Tuple[List[str], int]:
    """Distinct registrable domains among a claim's sources.

    Returns (domains, count). The primary/own source is always included; a
    URL that adds no new publisher does not raise the count — that is the
    whole point of independence.

    Independence here is a HARD invariant, not a convention: a second URL on
    the same registrable domain (a deep article and a homepage, or the same
    publisher republished) can never raise the count. Every source string
    flows through `registrable_domain`, so there is no path that counts two
    URLs from one publisher twice.
    """
    domains: List[str] = []
    seen: Set[str] = set()
    seen_domains: Set[str] = set()

    def _add(url: str) -> None:
        key = _independence_key(url)
        domain = registrable_domain(url)
        if not key:
            return
        # Both axes: one publisher, and one document identity. Same domain is
        # never independent even when slugs differ; a mirror across hosts is
        # never independent even when domains differ.
        if domain and domain in seen_domains:
            return
        if key in seen:
            return
        seen.add(key)
        if domain:
            seen_domains.add(domain)
        domains.append(domain or key)

    if primary_source:
        _add(primary_source)
    for url in sources or ():
        _add(str(url))
    return domains, len(domains)


_ARXIV_ID_RE = re.compile(r"(?:arxiv[:/]|abs/|pdf/)(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)
_DOI_RE = re.compile(r"(?:doi\.org/|doi[:/])(10\.\d{4,9}/[^\s?#]+)", re.IGNORECASE)


def document_fingerprint(url: str) -> str:
    """A stable identity for the DOCUMENT behind a URL, not the host.

    Independence is about distinct WORKS, not distinct domains: the same paper
    hosted on arxiv.org and on papers.neurips.cc (and mirrored at doi.org) is
    ONE source, and counting its mirrors as corroboration is a false
    independence signal. This returns a work identity from, in order:

      1. DOI (10.xxxx/...)              — the canonical work id
      2. arXiv id (2401.12345)          — primary preprint id
      3. a normalized title/tail slug   — best-effort for non-DOI pages

    Returns "" when no identity can be extracted (callers then fall back to
    domain-only independence, preserving prior behavior).
    """
    raw = str(url or "").strip()
    if not raw:
        return ""
    # arXiv first: a DOI-form arXiv id ("10.48550/arXiv.1706.03762") and the
    # native "arxiv.org/abs/1706.03762" both carry the same work id, so they
    # must fingerprint identically.
    arxiv = _ARXIV_ID_RE.search(raw)
    if arxiv:
        return "arxiv:" + arxiv.group(1).lower()
    doi = _DOI_RE.search(raw)
    if doi:
        doi_val = doi.group(1).lower().rstrip(".")
        embedded_arxiv = re.search(r"arxiv\.(\d{4}\.\d{4,5})", doi_val)
        if embedded_arxiv:
            return "arxiv:" + embedded_arxiv.group(1)
        return "doi:" + doi_val
    # Slug identity: the last path segment, stripped of extension and query.
    # Catches proceedings/lecture mirrors like ".../7181-attention-is-all-you-need.pdf".
    path = re.sub(r"[?#].*$", "", raw).rstrip("/")
    segment = path.rsplit("/", 1)[-1].lower()
    segment = re.sub(r"\.(pdf|html?|md|txt|php|aspx)$", "", segment)
    segment = re.sub(r"[^a-z0-9]+", "-", segment).strip("-")
    if len(segment) >= 8:
        return "slug:" + segment
    return ""


def _independence_key(url: str) -> str:
    """The unit of independence: the document if identifiable, else domain.

    A documented work identity always wins over the host, so a mirror of the
    same paper cannot corroborate itself. When no identity is extractable the
    registrable domain is used, exactly as before.
    """
    fp = document_fingerprint(url)
    if fp:
        return fp
    return registrable_domain(url or "")


def _shares_domain(a: str, b: str) -> bool:
    da, db = registrable_domain(a or ""), registrable_domain(b or "")
    return bool(da) and da == db


def is_new_publisher(url: str, existing_urls: Iterable[str]) -> bool:
    """True when `url` adds a source the claim does not already have.

    Independence requires BOTH axes to differ: a different registrable domain
    AND a different document identity. Two URLs on one domain are never
    independent, and mirrors of one document across hosts are never
    independent. Empty/unparseable URLs are never new publishers.
    """
    key = _independence_key(url)
    if not key:
        return False
    for existing in existing_urls or ():
        existing = str(existing or "")
        if not existing:
            continue
        if _shares_domain(url, existing) or _independence_key(existing) == key:
            return False
    return True


def distinct_publisher_count(sources: Iterable[str]) -> int:
    """Number of DISTINCT independent sources among source URLs.

    The one true corroboration count. Any module that reports a numeric
    corroboration figure (dedupe, synthesis badges) must derive it here, not
    from `len(urls)`: two URLs from one publisher are one source, and mirrors
    of one document are one source regardless of host.

    Deduplicates on the document identity when one exists (so arXiv + DOI of
    one paper count once) and on the registrable domain otherwise (so two
    pages of one publisher count once). Distinct works on a shared host (two
    different arXiv papers) are genuinely distinct sources and count twice.
    """
    seen: Set[str] = set()
    for url in sources or ():
        key = _independence_key(str(url or ""))
        if key:
            seen.add(key)
    return len(seen)


# Similarity band at/above which a NEW publisher's page text is treated as
# supporting an existing claim (independent corroboration). Deliberately below
# the 0.86 dedup-merge threshold: a corroborating page rarely restates the
# claim verbatim, and requiring near-identity is exactly why the live deep run
# reached two publishers zero times.
CORROBORATION_SIMILARITY = 0.55

# --- Anchor path (numeric + rare-term grounding) ----------------------------
# Full-claim similarity is blind to paraphrase: a live corroborating page said
# "Spending on AI infrastructure hit about $200 billion last year" against the
# claim "Global AI capital expenditure reached $200 billion in 2025" and scored
# 0.30 — below any usable semantic band. The distinctive part of a
# quantitative claim is its numbers and rare content terms, so a deterministic
# anchor matcher is the second, independent path.

# Minimum overlap of rare content anchors for the "numerically grounded"
# branch. A claim's significant numbers must FIRST all be grounded in the
# candidate (value match with the shared 2% tolerance) — precision over
# recall, so a same-topic page quoting a DIFFERENT figure cannot corroborate a
# quantitative claim (a separate conflict guard also rejects that on the
# semantic path). Kept at 2: the realistic paraphrase shares only
# "200"/"billion" once anchors are counted, and requiring a third distinctive
# word would re-break exactly the corroboration it exists to catch. Numbers
# and named entities are anchors, not free passes.
CORROBORATION_ANCHOR_MIN_CONTENT_OVERLAP = 2

# The title/snippet branch fires only on a STRONG overlap of the claim's rare
# anchors — a third of them, at least three. This is the conservative backstop
# for headline-style matches ("AI capex $200 billion record") where the page
# body is thin; a loosely related article will not clear it.
CORROBORATION_ANCHOR_TITLE_MIN_SHARED = 3
CORROBORATION_ANCHOR_TITLE_MIN_FRACTION = 0.34

# Bounded retained page text used for matching after verification blanks the
# full content (memory-release rule). Short enough to keep thousands of
# results cheap.
CORROBORATION_EXCERPT_CHARS = 800


def _rare_anchors(text: str) -> Set[str]:
    return rare_content_tokens(text)


def _is_numeric_anchor(token: str) -> bool:
    """True for the number-carrying anchors (digits) — unit words like
    'billion' are content but not the discriminating quantity itself."""
    return any(ch.isdigit() for ch in token)


def _anchor_support(claim: str, candidate_text: str) -> bool:
    """Deterministic anchor match: numbers AND rare-term overlap.

    Branch (b): every significant quantity in the claim is numerically present
    in the candidate (via the shared `numbers_grounded` tolerance), AND the
    two texts share at least `CORROBORATION_ANCHOR_MIN_CONTENT_OVERLAP` rare
    content terms. Claims with no significant numbers cannot use this branch —
    they fall back to semantic similarity.
    """
    claim_numbers = _significant_quantities(claim)
    if not claim_numbers:
        return False
    if not numbers_grounded(claim, candidate_text):
        return False
    shared = _rare_anchors(claim) & _rare_anchors(candidate_text)
    if len(shared) < CORROBORATION_ANCHOR_MIN_CONTENT_OVERLAP:
        return False
    # At least one shared anchor must be a number: a page that merely repeats
    # the claim's vocabulary without its quantity is not corroboration of a
    # quantitative claim.
    return any(_is_numeric_anchor(tok) for tok in shared)


def _numeric_conflict_guard(claim: str, candidate_text: str) -> bool:
    """True when the candidate states a CONFLICTING significant quantity.

    A quantitatively similar page with the wrong figure ("$300 billion" for a
    "$200 billion" claim) scores high on lexical similarity and would sail
    through the semantic band — but stating a different number is
    contradiction, not corroboration. Applies only when the claim itself is
    quantitative and the candidate carries a comparable quantity.
    """
    if not _significant_quantities(claim):
        return False
    return numeric_conflict(claim, candidate_text) is not None


def _title_snippet_support(claim: str, title: str, snippet: str) -> bool:
    """Branch (c): strong rare-anchor overlap in title+snippet alone.

    For headline-shaped evidence where the page body is unavailable. Requires
    an absolute floor AND a fraction of the claim's anchors, so long claims
    cannot be satisfied by incidental vocabulary.
    """
    anchors = _rare_anchors(claim)
    if not anchors:
        return False
    surface = _rare_anchors(f"{title} {snippet}")
    shared = anchors & surface
    if len(shared) < CORROBORATION_ANCHOR_TITLE_MIN_SHARED:
        return False
    return len(shared) / len(anchors) >= CORROBORATION_ANCHOR_TITLE_MIN_FRACTION


def _candidate_text(candidate: Dict[str, Any]) -> str:
    """Title + snippet + retained excerpt + best sentence, whitespace-normalized.

    Built from every text field a search result carries, because real
    corroboration is scattered across them: the title names the figure, the
    snippet paraphrases it and the retained excerpt (surviving the verifier's
    content blank) holds the surrounding sentence. Snippet-only matching was
    root cause 1 of the zero-corroboration live failure.
    """
    parts = [
        str(candidate.get("title", "") or ""),
        str(candidate.get("snippet", "") or ""),
        str(candidate.get("corroboration_excerpt", "") or ""),
        str(candidate.get("content", "") or "")[:2000],
    ]
    text = " ".join(p for p in parts if p)
    return re.sub(r"\s+", " ", text).strip()


def apply_corroboration(fact: Dict[str, Any], url: str) -> bool:
    """Attach `url` as a corroborating source to `fact`, enforcing independence.

    The ONLY sanctioned write path for acquired corroboration: it rejects a
    URL whose registrable domain the fact already holds (via `is_new_publisher`)
    and recomputes `corroboration_count` with `distinct_publisher_count`, so a
    same-publisher page can never raise the count. Returns True when the count
    actually increased, False otherwise. Additive: mutates `fact` in place
    (callers own the dict), and is total — a bad URL is a no-op.
    """
    if not isinstance(fact, dict):
        return False
    candidate = str(url or "").strip()
    if not candidate:
        return False
    existing: List[str] = [
        str(u) for u in (fact.get("corroborating_sources") or []) if str(u).strip()
    ]
    source = str(fact.get("source", "") or "").strip()
    if source and not any(
        registrable_domain(source) == registrable_domain(u) for u in existing
    ):
        existing.append(source)
    if not is_new_publisher(candidate, existing):
        return False
    try:
        before = int(fact.get("corroboration_count", 1) or 1)
    except (TypeError, ValueError):
        before = 1
    seen = {registrable_domain(u) for u in existing}
    seen.add(registrable_domain(candidate))
    seen.discard("")
    after = len(seen)
    if after <= before:
        return False
    if candidate not in existing:
        existing.append(candidate)
    fact["corroborating_sources"] = existing
    fact["corroboration_count"] = max(after, before + 1)
    return True


def find_corroborating_sources(
    claim: str,
    candidates: Sequence[Dict[str, Any]],
    existing_urls: Iterable[str],
    *,
    threshold: float = CORROBORATION_SIMILARITY,
) -> List[str]:
    """New-publisher URLs among `candidates` whose text supports `claim`.

    `candidates` are search-result dicts (`url`, `title`, `snippet`, `content`,
    `corroboration_excerpt`). A URL is returned when its registrable domain is
    absent from `existing_urls` (`is_new_publisher` — the sole independence
    gate) AND EITHER:

      (a) the candidate's combined title/snippet/excerpt text scores at or
          above `threshold` on the shared hybrid semantic engine, OR
      (b) the claim's significant numbers are all numerically grounded in the
          candidate AND the two share enough rare content anchors
          (`_anchor_support`), OR
      (c) the title+snippet alone share a strong fraction of the claim's rare
          anchors (`_title_snippet_support`).

    Paths (b)/(c) exist because a real corroborating page paraphrases the
    claim; near-verbatim similarity was the sole test and it never fired on
    live evidence. (b)/(c) are deliberately conservative (numbers must all
    match; anchors must overlap) so a same-topic-different-claim page cannot
    corroborate. Deterministic and LLM-free.
    """
    text = str(claim or "").strip()
    if not text:
        return []
    out: List[str] = []
    seen_domains: Set[str] = set()
    for candidate in candidates or ():
        if not isinstance(candidate, dict):
            continue
        url = str(candidate.get("url", "") or "").strip()
        if not url or not is_new_publisher(url, [*existing_urls, *out]):
            continue
        domain = registrable_domain(url)
        if not domain or domain in seen_domains:
            continue
        combined = _candidate_text(candidate)
        title = str(candidate.get("title", "") or "")
        snippet = str(candidate.get("snippet", "") or "")
        if _numeric_conflict_guard(text, combined):
            continue  # contradictory figure is never corroboration
        supported = (
            _best_text_support(text, combined) >= threshold
            or _anchor_support(text, combined)
            or _title_snippet_support(text, title, snippet)
        )
        if supported:
            seen_domains.add(domain)
            out.append(url)
    return out


def _best_text_support(claim: str, text: str) -> float:
    """Max hybrid similarity of `claim` to combined candidate text / its best
    sentence. Sentence-level scoring keeps a long page from diluting a single
    claim-bearing passage."""
    from app.core.semantic import cross_similarity, pair_similarity

    compact = re.sub(r"\s+", " ", text or "").strip()
    if not compact:
        return 0.0
    best = pair_similarity(claim, compact)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", compact) if s.strip()]
    if sentences:
        row = cross_similarity([claim], sentences[:80])[0]
        if len(row):
            best = max(best, float(max(row)))
    return best


def grade_claim(
    fact: Dict[str, Any],
    *,
    contradictions: Optional[Sequence[Dict[str, Any]]] = None,
) -> EvidenceRecord:
    """Compute the EvidenceRecord for one fact dict. Total and deterministic."""
    claim = str(fact.get("claim", "") or "").strip()
    source = str(fact.get("source", "") or "").strip()
    profile = classify_source(source) if source else None

    record = EvidenceRecord(
        claim=claim,
        source=source,
        domain=registrable_domain(source),
        tier=(profile.tier if profile else ""),
        authority=float(profile.authority) if profile else 0.0,
        is_primary=bool(profile.is_primary) if profile else False,
        verified=bool(fact.get("verified", False)),
        verification_score=float(fact.get("verification_score", 0.0) or 0.0),
    )
    record.is_stale = bool(fact.get("is_stale", False))
    # Freshness is only present on graded runs; absent means unmeasured (0.0),
    # which is treated as neutral, never as "stale".
    record.freshness = float(fact.get("freshness", 0.0) or 0.0)

    # --- corroboration, measured by publisher independence ---
    corroborating = [
        str(u) for u in (fact.get("corroborating_sources") or []) if str(u).strip()
    ]
    domains, count = independent_corroboration(corroborating, primary_source=source)
    record.corroborating_domains = domains
    record.corroboration_count = max(1, count)

    # --- numeric support ---
    record.has_numbers = _claim_is_quantitative(claim)
    # A number is considered supported unless the verifier explicitly said the
    # claim's numbers were not grounded (verified=False with numbers present).
    record.numbers_supported = not (
        record.has_numbers and not record.verified
    )

    # --- contradictions ---
    # Fix C: only UNRESOLVED contradictions disqualify a claim. A conflict the
    # resolution pass explained (different period/scope/metric) is a recorded
    # spread, not a disagreement, and must not cap the grade or flag the claim.
    contradictions = contradictions or []
    for c in contradictions:
        if not isinstance(c, dict) or c.get("resolved"):
            continue
        a = str(c.get("claim_a", "") or "")
        b = str(c.get("claim_b", "") or "")
        if a == claim or b == claim:
            record.contradiction_count += 1
            if str(c.get("kind", "")) == "numeric":
                record.numeric_conflict = True

    record.grade, record.needs_corroboration = _assign_grade(record)
    return record


def _assign_grade(r: EvidenceRecord) -> Tuple[str, bool]:
    """Grade A-D and the corroboration requirement from the record's signals."""
    important = r.has_numbers or _looks_definitional(r.claim)
    needs_corroboration = important and r.corroboration_count < MIN_INDEPENDENT_CORROBORATION

    # A contradiction is disqualifying for a top grade regardless of sourcing:
    # holding a conflicted claim up as "established" is the false-confidence
    # failure the brief explicitly forbids.
    if r.contradiction_count > 0:
        return (GRADE_C if r.verified else GRADE_D), needs_corroboration

    if not r.verified:
        # Unverified but strongly sourced is still only C — never A/B.
        return (GRADE_C if r.is_primary or r.authority >= 0.7 else GRADE_D), needs_corroboration

    # Verified from here on.
    primary_ok = r.is_primary or r.tier in _PRIMARY_LIKE_TIERS or r.authority >= 0.7
    corroborated = r.corroboration_count >= MIN_INDEPENDENT_CORROBORATION
    numbers_ok = not r.has_numbers or r.numbers_supported

    if numbers_ok and primary_ok and corroborated:
        return GRADE_A, needs_corroboration
    if numbers_ok and (corroborated or primary_ok):
        return GRADE_B, needs_corroboration
    return GRADE_C, needs_corroboration


def grade_facts(
    facts: Sequence[Dict[str, Any]],
    contradictions: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Attach an `evidence` EvidenceRecord dict to each fact (additive).

    Returns NEW dicts; the originals are not mutated, so callers decide
    whether to adopt the annotations (the verifier keeps its own shape).
    """
    out: List[Dict[str, Any]] = []
    for fact in facts or []:
        if not isinstance(fact, dict):
            continue
        record = grade_claim(fact, contradictions=contradictions)
        enriched = dict(fact)
        enriched["evidence"] = record.to_dict()
        enriched["evidence_grade"] = record.grade
        out.append(enriched)
    return out


def grade_distribution(records: Sequence[EvidenceRecord]) -> Dict[str, int]:
    dist = {GRADE_A: 0, GRADE_B: 0, GRADE_C: 0, GRADE_D: 0}
    for r in records or ():
        dist[r.grade] = dist.get(r.grade, 0) + 1
    return dist


def evidence_quality_score(records: Sequence[EvidenceRecord]) -> float:
    """0-1 signal: share of the pool that is at least B, weighted by grade.

    A pool of three A-grade claims outscores a pool of thirty D-grade claims
    — which is the entire thesis of evidence-driven research. Empty pool = 0.
    """
    records = list(records or ())
    if not records:
        return 0.0
    total = 0.0
    for r in records:
        total += {GRADE_A: 1.0, GRADE_B: 0.75, GRADE_C: 0.4, GRADE_D: 0.1}.get(r.grade, 0.1)
    return round(total / len(records), 3)


def coverage_gaps_from_records(records: Sequence[EvidenceRecord]) -> List[str]:
    """Human-readable evidence deficiencies, for the critic/report.

    These are the concrete "what is missing" strings the brief asks for —
    each names a claim and the specific missing property.
    """
    gaps: List[str] = []
    for r in records or ():
        if r.contradiction_count > 0:
            gaps.append(
                f"'{_clip(r.claim)}' is contradicted by {r.contradiction_count} source(s) "
                "and needs resolution before it can be treated as established."
            )
        elif r.needs_corroboration and r.corroboration_count < MIN_INDEPENDENT_CORROBORATION:
            kind = "quantitative" if r.has_numbers else "definitional"
            gaps.append(
                f"'{_clip(r.claim)}' is a single-source {kind} claim — "
                "needs independent corroboration from another publisher."
            )
        elif r.grade == GRADE_D:
            gaps.append(
                f"'{_clip(r.claim)}' is unverified or rests on weak provenance."
            )
    return gaps[:12]


def _clip(text: str, limit: int = 120) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
