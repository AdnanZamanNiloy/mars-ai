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

    def _add(url: str) -> None:
        d = registrable_domain(url)
        if d and d not in seen:
            seen.add(d)
            domains.append(d)

    if primary_source:
        _add(primary_source)
    for url in sources or ():
        _add(str(url))
    return domains, len(domains)


def is_new_publisher(url: str, existing_urls: Iterable[str]) -> bool:
    """True when `url`'s registrable domain is absent from `existing_urls`.

    The procurement side of independence: before treating a search result as
    corroboration, callers ask whether it comes from a publisher the claim
    already has. It reuses `registrable_domain`, so it agrees with
    `independent_corroboration` by construction. Empty/unparseable URLs are
    never new publishers (nothing to corroborate with).
    """
    domain = registrable_domain(url or "")
    if not domain:
        return False
    for existing in existing_urls or ():
        if registrable_domain(str(existing or "")) == domain:
            return False
    return True


def distinct_publisher_count(sources: Iterable[str]) -> int:
    """Number of DISTINCT registrable domains among source URLs.

    The one true corroboration count. Any module that reports a numeric
    corroboration figure (dedupe, synthesis badges) must derive it here, not
    from `len(urls)`: two URLs from one publisher are one source.
    """
    seen: Set[str] = set()
    for url in sources or ():
        d = registrable_domain(str(url or ""))
        if d:
            seen.add(d)
    return len(seen)


# Similarity band at/above which a NEW publisher's page text is treated as
# supporting an existing claim (independent corroboration). Deliberately below
# the 0.86 dedup-merge threshold: a corroborating page rarely restates the
# claim verbatim, and requiring near-identity is exactly why the live deep run
# reached two publishers zero times.
CORROBORATION_SIMILARITY = 0.55


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

    `candidates` are search-result dicts (`url`, `snippet`, `content`). Text
    support is measured with the shared hybrid semantic engine against the
    result's snippet and its best-matching sentence — the claim-bearing
    passage, not the whole page. A URL is only returned when its registrable
    domain is absent from `existing_urls` (`is_new_publisher`), so ambiguity
    about which publisher a hit belongs to is always resolved conservatively.
    Deterministic and LLM-free.
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
        snippet = str(candidate.get("snippet", "") or "")
        content = str(candidate.get("content", "") or "")
        score = _best_text_support(text, snippet, content)
        if score >= threshold:
            seen_domains.add(domain)
            out.append(url)
    return out


def _best_text_support(claim: str, snippet: str, content: str) -> float:
    """Max hybrid similarity of `claim` to a result's snippet / best sentence."""
    from app.core.semantic import cross_similarity, pair_similarity

    best = 0.0
    for text in (snippet, content[:2000]):
        compact = re.sub(r"\s+", " ", text or "").strip()
        if not compact:
            continue
        best = max(best, pair_similarity(claim, compact))
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
