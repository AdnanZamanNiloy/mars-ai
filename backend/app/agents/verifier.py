"""Verification Agent (Phase 2.3).

Deterministic, LLM-free check: does each extracted fact's key vocabulary
actually appear in the content/snippet of the source it cites, and is that
source reliable? Facts that fail stay in state for transparency but are
excluded from the Synthesizer's usable evidence.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from app.agents.evidence_utils import looks_truncated, source_reliability_score

# Overlap at or above this ratio means the claim's key terms are present
# in the cited source's text.
MIN_OVERLAP_RATIO = 0.35

# Source must independently pass the reliability bar.
MIN_SOURCE_SCORE = 0.55

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "that", "this",
    "these", "those", "it", "its", "as", "by", "at", "from", "which",
    "who", "what", "when", "where", "why", "how", "can", "could", "will",
    "would", "has", "have", "had", "not", "no", "but", "into", "about",
    "over", "after", "before", "between", "more", "most", "also", "than",
    "then", "their", "there", "such", "may", "might", "must", "per",
}


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]{3,}", text.lower()) if t not in _STOPWORDS]


def _source_text_map(search_results: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    """Map normalized source URL -> best available text (content, then snippet)."""
    mapping: Dict[str, str] = {}
    for item in search_results or []:
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        content = str(item.get("content", "") or "")
        snippet = str(item.get("snippet", "") or "")
        text = content if content.strip() else snippet
        if text.strip():
            mapping[url] = text
    return mapping


def _overlap_ratio(claim_terms: List[str], source_text: str) -> float:
    if not claim_terms:
        return 0.0
    source_tokens = set(_tokens(source_text))
    hits = sum(1 for term in claim_terms if term in source_tokens)
    return hits / len(claim_terms)


def verify_facts(
    facts: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach `verified`, `verification_score`, `verification_reason` to each fact."""
    sources = _source_text_map(search_results)

    verified: List[Dict[str, Any]] = []
    for fact in facts:
        claim = str(fact.get("claim", ""))
        # Last-chance fragment guard: no truncated claim may persist or be
        # synthesized, regardless of which upstream path (or cache) let it
        # through. See AGENTS.md stale-fact-cache entry.
        if looks_truncated(claim):
            continue
        source = str(fact.get("source", "")).strip()
        source_text = sources.get(source, "")
        claim_terms = _tokens(claim)

        if not source_text:
            reason = "source content unavailable to verify claim"
            score = 0.0
            is_verified = False
        else:
            score = _overlap_ratio(claim_terms, source_text)
            source_score = source_reliability_score(source)
            is_verified = score >= MIN_OVERLAP_RATIO and source_score >= MIN_SOURCE_SCORE
            reason = (
                f"lexical overlap {score:.2f}, source score {source_score:.2f}"
            )

        out = dict(fact)
        out["verified"] = is_verified
        out["verification_score"] = round(score, 3)
        out["verification_reason"] = reason
        verified.append(out)

    return verified
