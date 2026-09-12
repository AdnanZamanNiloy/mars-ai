"""Verification Agent — does the cited source actually support the claim?

Deterministic and LLM-free by design: verification runs over every claim in the
pool (often 100+), so a model call per claim is unaffordable, and the checks
that catch real failures are mechanical anyway.

What the previous version could not catch
-----------------------------------------
It computed one signal: the fraction of a claim's content words that appear in
the cited source. That passes three dangerous cases.

1. NUMBER HALLUCINATION. "Capacity grew 40% in 2024" against a source saying
   "capacity grew 4% in 2024" shares every word except the digit, so it scored
   ~0.95 and verified. Fabricated or mis-transcribed statistics are the highest
   -impact error a research report can contain, and lexical overlap is blind to
   them by construction.

2. POLARITY INVERSION. "Nuclear power does not reduce emissions" against a
   source arguing it does shares nearly all vocabulary. The negation is one
   token in a bag of twenty.

3. QUOTE FABRICATION. A `direct_quote` that appears nowhere in the source was
   never checked because the field was never read.

All three are now hard failures. Overlap remains, but rare terms count for more
than common ones — a claim built from generic vocabulary should not verify on
generic vocabulary alone.

Output contract is a superset of the previous one: `verified`,
`verification_score` and `verification_reason` keep their names and meanings, so
existing consumers are unaffected.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.core.logging import get_logger

from app.agents.evidence_utils import (
    claim_polarity,
    looks_truncated,
    numbers_grounded,
    source_reliability_score,
)
from app.agents.sources import canonical_url, classify_source, freshness_score

logger = get_logger(__name__)

# Weighted overlap at or above this ratio means the claim's key terms are
# present in the cited source's text.
MIN_OVERLAP_RATIO = 0.35

# Source must independently pass the reliability bar.
MIN_SOURCE_SCORE = 0.55

# Below this freshness the claim is still verified but flagged stale, so the
# confidence engine can discount it rather than the verifier silently dropping
# evidence that may be the only evidence available.
STALE_FRESHNESS = 0.25

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "that", "this",
    "these", "those", "it", "its", "as", "by", "at", "from", "which",
    "who", "what", "when", "where", "why", "how", "can", "could", "will",
    "would", "has", "have", "had", "not", "no", "but", "into", "about",
    "over", "after", "before", "between", "more", "most", "also", "than",
    "then", "their", "there", "such", "may", "might", "must", "per",
}

# Very common in research prose; matching on these proves nothing.
_LOW_INFORMATION = {
    "research", "study", "studies", "data", "report", "reports", "analysis",
    "results", "based", "used", "using", "including", "important", "significant",
    "different", "various", "many", "several", "often", "generally", "however",
    "example", "examples", "information", "system", "systems", "process",
    "new", "high", "low", "large", "small", "year", "years", "time",
}


def _tokens(text: str) -> List[str]:
    return [
        t for t in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
        if t not in _STOPWORDS
    ]


def _term_weight(term: str) -> float:
    """Rarity proxy: longer, non-boilerplate terms carry more evidentiary load.

    A real IDF table would need a corpus the pipeline does not have. Length plus
    a stop-list of research boilerplate is a crude but effective substitute, and
    it fixes the concrete failure of the unweighted version: long claims full of
    generic words verified as easily as short claims full of specific ones.
    """
    if term in _LOW_INFORMATION:
        return 0.35
    if term.isdigit():
        return 1.6
    if len(term) >= 10:
        return 1.5
    if len(term) >= 7:
        return 1.2
    return 1.0


def _source_text_map(search_results: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    """Map source URL -> best available text (content, then snippet).

    Keyed by BOTH the raw and canonical URL: the claim's stored source string may
    differ cosmetically from the search result's (trailing slash, tracking
    parameter), and a lookup miss used to produce "source content unavailable to
    verify claim" — an unverified claim caused purely by URL formatting.
    """
    mapping: Dict[str, str] = {}
    for item in search_results or []:
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        content = str(item.get("content", "") or "")
        snippet = str(item.get("snippet", "") or "")
        text = content if content.strip() else snippet
        if not text.strip():
            continue
        for key in {url, canonical_url(url)}:
            if key:
                existing = mapping.get(key, "")
                if len(text) > len(existing):
                    mapping[key] = text
    return mapping


def _weighted_overlap(claim_terms: Sequence[str], source_text: str) -> float:
    if not claim_terms:
        return 0.0
    source_tokens: Set[str] = set(_tokens(source_text))
    total = 0.0
    hit = 0.0
    for term in claim_terms:
        weight = _term_weight(term)
        total += weight
        if term in source_tokens:
            hit += weight
    return (hit / total) if total else 0.0


def _overlap_ratio(claim_terms: List[str], source_text: str) -> float:
    """Kept for compatibility with callers/tests that used the unweighted form."""
    if not claim_terms:
        return 0.0
    source_tokens = set(_tokens(source_text))
    hits = sum(1 for term in claim_terms if term in source_tokens)
    return hits / len(claim_terms)


def _lookup_text(source: str, sources: Dict[str, str]) -> str:
    return sources.get(source) or sources.get(canonical_url(source)) or ""


def verify_facts(
    facts: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach verification metadata to each fact.

    Adds, per fact:
      verified              bool   — passed every hard check
      verification_score    float  — weighted lexical overlap, 0-1
      verification_reason   str    — human-readable verdict
      verification_checks   dict   — per-check results, for the Evidence Explorer
      is_stale              bool   — verified but old for its content type
    """
    sources = _source_text_map(search_results)

    verified: List[Dict[str, Any]] = []
    for fact in facts or []:
        if not isinstance(fact, dict):
            continue
        claim = str(fact.get("claim", ""))
        # Last-chance fragment guard: no truncated claim may persist or be
        # synthesized, regardless of which upstream path (or cache) let it in.
        if looks_truncated(claim):
            continue

        source = str(fact.get("source", "")).strip()
        source_text = _lookup_text(source, sources)
        claim_terms = _tokens(claim)
        source_score = source_reliability_score(source)

        checks: Dict[str, Any] = {
            "source_reachable": bool(source_text),
            "source_score": round(source_score, 3),
        }

        if not source_text:
            out = dict(fact)
            out["verified"] = False
            out["verification_score"] = 0.0
            out["verification_reason"] = "source content unavailable to verify claim"
            out["verification_checks"] = checks
            out["is_stale"] = False
            verified.append(out)
            continue

        overlap = _weighted_overlap(claim_terms, source_text)
        checks["lexical_overlap"] = round(overlap, 3)

        numeric_ok = numbers_grounded(claim, source_text)
        checks["numbers_grounded"] = numeric_ok

        # Polarity: compare the claim's direction against the source passage most
        # similar to it, not the whole document — a long article contains both
        # directions somewhere, so whole-document polarity is meaningless.
        polarity_ok, polarity_detail = _polarity_check(claim, source_text)
        checks["polarity_consistent"] = polarity_ok

        quote_state = fact.get("quote_verified", None)
        if quote_state is None and fact.get("direct_quote"):
            from app.agents.summarizer import _quote_supported

            quote_state = _quote_supported(str(fact.get("direct_quote", "")), source_text)
        checks["quote_verified"] = quote_state

        fresh = freshness_score(
            str(fact.get("published_at", "") or ""),
            str(fact.get("search_type", "default") or "default"),
        )
        checks["freshness"] = fresh
        is_stale = fresh < STALE_FRESHNESS

        failures: List[str] = []
        if overlap < MIN_OVERLAP_RATIO:
            failures.append(f"weighted overlap {overlap:.2f} < {MIN_OVERLAP_RATIO}")
        if source_score < MIN_SOURCE_SCORE:
            failures.append(f"source score {source_score:.2f} < {MIN_SOURCE_SCORE}")
        if not numeric_ok:
            failures.append("a figure in the claim does not appear in the source")
        if not polarity_ok:
            failures.append(f"claim direction contradicts the source ({polarity_detail})")
        if quote_state is False:
            failures.append("direct quote not found in the source")

        is_verified = not failures
        reason = (
            f"verified: overlap {overlap:.2f}, source {source_score:.2f}"
            + (", numbers grounded" if _has_numbers(claim) else "")
            + (", quote located" if quote_state is True else "")
            if is_verified
            else "; ".join(failures)
        )

        out = dict(fact)
        out["verified"] = is_verified
        out["verification_score"] = round(overlap, 3)
        out["verification_reason"] = reason
        out["verification_checks"] = checks
        out["is_stale"] = is_stale
        verified.append(out)

    if facts:
        passed = sum(1 for f in verified if f.get("verified"))
        logger.info(
            "[Verifier] %d/%d claims verified (%d dropped as truncated)",
            passed, len(verified), len(facts) - len(verified),
        )
    return verified


def _has_numbers(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _polarity_check(claim: str, source_text: str) -> Tuple[bool, str]:
    """Compare the claim's polarity to the most similar source sentence.

    Returns (ok, detail). Neutral claims and claims with no comparable sentence
    pass: this check must only fire on a clear inversion, because a false
    positive here silently deletes good evidence.
    """
    claim_polarity_value = claim_polarity(claim)
    if claim_polarity_value == 0:
        return True, "claim is directionally neutral"

    claim_tokens = set(_tokens(claim))
    if not claim_tokens:
        return True, "no comparable terms"

    best_sentence = ""
    best_overlap = 0.0
    for sentence in _SENTENCE_SPLIT.split(source_text or "")[:400]:
        tokens = set(_tokens(sentence))
        if not tokens:
            continue
        overlap = len(claim_tokens & tokens) / len(claim_tokens)
        if overlap > best_overlap:
            best_overlap, best_sentence = overlap, sentence

    # Require a genuinely matching passage before judging direction.
    if best_overlap < 0.55 or not best_sentence:
        return True, "no closely matching passage to compare direction"

    source_polarity = claim_polarity(best_sentence)
    if source_polarity == 0 or source_polarity == claim_polarity_value:
        return True, "direction consistent"
    return False, f"source passage polarity {source_polarity} vs claim {claim_polarity_value}"


def verification_summary(facts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate verification outcome — what the Evidence Explorer shows and
    what the confidence engine discounts on."""
    items = [f for f in (facts or []) if isinstance(f, dict)]
    if not items:
        return {
            "total": 0, "verified": 0, "rate": 0.0, "failure_reasons": {},
            "stale": 0, "numeric_failures": 0, "polarity_failures": 0,
            "unreachable_sources": 0,
        }
    verified = [f for f in items if f.get("verified")]
    reasons: Dict[str, int] = {}
    numeric_failures = polarity_failures = unreachable = 0
    for fact in items:
        if fact.get("verified"):
            continue
        reason = str(fact.get("verification_reason", "unknown"))
        head = reason.split(";")[0].strip()[:80]
        reasons[head] = reasons.get(head, 0) + 1
        checks = fact.get("verification_checks") or {}
        if checks.get("numbers_grounded") is False:
            numeric_failures += 1
        if checks.get("polarity_consistent") is False:
            polarity_failures += 1
        if checks.get("source_reachable") is False:
            unreachable += 1
    return {
        "total": len(items),
        "verified": len(verified),
        "rate": round(len(verified) / len(items), 4),
        "failure_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:6]),
        "stale": sum(1 for f in items if f.get("is_stale")),
        "numeric_failures": numeric_failures,
        "polarity_failures": polarity_failures,
        "unreachable_sources": unreachable,
    }
