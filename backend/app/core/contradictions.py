"""Contradiction Engine (Phase 3.2, Feature 09).

After dedupe_semantic_facts collapses near-identical claims, this pass
flags claims that are TOPICALLY similar but NUMERICALLY different — e.g.
two claims about "market growth rate" citing different percentages.

Heuristic (MVP per manual): claims whose similarity sits in a band below
the dedup merge threshold (similar enough to be about the same thing, not
similar enough to be the same claim), both carrying numbers whose values
differ significantly, from DIFFERENT sources.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from app.agents.evidence_utils import _semantic_similarity

# Just below the dedup merge threshold (0.86): same topic, not same claim.
SIMILARITY_LOW = 0.50
SIMILARITY_HIGH = 0.86

# Relative numeric difference that counts as a genuine conflict.
SIGNIFICANT_DIFF = 0.05

MAX_CONTRADICTIONS = 5

_NUMBER_RE = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<suffix>%|percent|thousand|million|billion|bn|k\b|m\b|b\b)?",
    re.IGNORECASE,
)

_SUFFIX_SCALE = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "million": 1e6,
    "bn": 1e9, "b": 1e9, "billion": 1e9,
    "%": None, "percent": None,  # percent keeps its own scale
}


def extract_numbers(text: str) -> List[float]:
    """Extract scale-normalized numbers. Percents stay as-is (20% -> 20.0)."""
    values: List[float] = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group("value").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        suffix = (match.group("suffix") or "").lower()
        scale = _SUFFIX_SCALE.get(suffix, 1.0)
        if scale is not None:
            value *= scale
        values.append(value)
    return values


def _numbers_conflict(a: List[float], b: List[float]) -> bool:
    """True if any shared-position values differ by more than SIGNIFICANT_DIFF."""
    for va in a:
        for vb in b:
            if va == vb:
                continue
            bigger = max(abs(va), abs(vb))
            if bigger == 0:
                continue
            if abs(va - vb) / bigger >= SIGNIFICANT_DIFF:
                return True
    return False


def find_contradictions(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pairwise scan for numeric conflicts between topically-similar claims
    from different sources. Returns at most MAX_CONTRADICTIONS entries."""
    contradictions: List[Dict[str, Any]] = []

    def _claim_key(fact: Dict[str, Any]) -> str:
        return re.sub(r"\W+", " ", str(fact.get("claim", "")).lower()).strip()

    cleaned = [
        f for f in facts
        if str(f.get("claim", "")).strip() and str(f.get("source", "")).strip()
    ]

    for i in range(len(cleaned)):
        for j in range(i + 1, len(cleaned)):
            a, b = cleaned[i], cleaned[j]
            if str(a.get("source", "")).strip() == str(b.get("source", "")).strip():
                continue  # same source restating itself isn't a contradiction

            similarity = _semantic_similarity(str(a.get("claim", "")), str(b.get("claim", "")))
            if not (SIMILARITY_LOW <= similarity < SIMILARITY_HIGH):
                continue

            numbers_a = extract_numbers(str(a.get("claim", "")))
            numbers_b = extract_numbers(str(b.get("claim", "")))
            if not numbers_a or not numbers_b:
                continue
            if not _numbers_conflict(numbers_a, numbers_b):
                continue

            # The same conflict (e.g. via a near-duplicate restatement of an
            # already-flagged claim) shouldn't produce a second entry.
            is_duplicate = any(
                _semantic_similarity(_claim_key(a), existing_side) >= SIMILARITY_HIGH
                or _semantic_similarity(_claim_key(b), existing_side) >= SIMILARITY_HIGH
                for existing in contradictions
                for existing_side in (
                    _claim_key({"claim": existing["claim_a"]}),
                    _claim_key({"claim": existing["claim_b"]}),
                )
            )
            if is_duplicate:
                continue

            contradictions.append({
                "topic_similarity": round(similarity, 3),
                "claim_a": str(a.get("claim", "")),
                "source_a": str(a.get("source", "")),
                "value_a": numbers_a[:3],
                "claim_b": str(b.get("claim", "")),
                "source_b": str(b.get("source", "")),
                "value_b": numbers_b[:3],
                "note": "topically similar claims cite significantly different figures",
            })
            if len(contradictions) >= MAX_CONTRADICTIONS:
                return contradictions

    return contradictions
