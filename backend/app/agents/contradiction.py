"""Contradiction engine (vision Feature 09).

Disagreement between sources is the most valuable signal a research system can
surface, and the easiest to destroy. The previous pipeline destroyed it twice
over: `dedupe_semantic_facts` merged near-identical claims and kept only the
higher-confidence copy, and nothing anywhere compared the *numbers* inside
claims. A run where three sources said 18%, 11% and 7% produced one claim, one
number, and a confident report.

This module finds real conflicts without an LLM call, which matters because a
contradiction check that costs a model call per claim pair is unaffordable at
100+ claims (4,950 pairs). Detection is deterministic:

  numeric     same subject and unit, values diverging beyond a threshold
  polarity    same subject, opposite assertion ("X reduces Y" / "X does not")
  temporal    same subject, incompatible dates for the same event

Conflicts between two claims from the SAME domain are reported as
`intra_source` and down-weighted: a publisher restating itself imprecisely is
an editing artifact, not a genuine dispute between sources.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.agents.evidence_utils import (
    claim_polarity,
    extract_domain,
    extract_numbers,
    numeric_conflict,
    semantic_similarity,
)

# Two claims must be about the same thing before their difference means
# anything. 0.34 was chosen so paraphrases of one measurement stay together
# while different measurements of the same market ("size" vs "growth rate")
# fall apart — the failure mode that produces phantom contradictions.
SUBJECT_SIMILARITY = 0.34

# Relative divergence at which two numbers stop being rounding variants.
NUMERIC_DIVERGENCE = 0.20

# Above this, a numeric gap is not a nuance but a range that must be reported
# as a range.
SEVERE_DIVERGENCE = 0.60

_MEASURE_WORDS = {
    "growth", "cagr", "rate", "share", "size", "cost", "price", "capex",
    "opex", "revenue", "capacity", "output", "emissions", "efficiency",
    "accuracy", "latency", "throughput", "population", "gdp", "inflation",
    "yield", "lifetime", "duration", "temperature", "probability",
}

_STOP = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "by", "at", "from", "that", "this",
    "it", "its", "as", "than", "about", "over", "per", "will", "has", "have",
}


def _content_tokens(text: str) -> Set[str]:
    return {t for t in re.findall(r"[a-z][a-z0-9\-]{2,}", (text or "").lower()) if t not in _STOP}


def _subject_key(text: str) -> Tuple[str, ...]:
    """A coarse subject fingerprint: the measure words plus the rarest nouns.

    Used only to bucket claims so the O(n^2) pair comparison runs inside small
    buckets rather than across the whole pool. Recall matters more than
    precision here: a wrong bucket loses a conflict, a loose bucket only costs
    a few extra comparisons.
    """
    tokens = _content_tokens(text)
    measures = sorted(tokens & _MEASURE_WORDS)
    rest = sorted(tokens - _MEASURE_WORDS, key=lambda t: (-len(t), t))[:3]
    return tuple(measures[:2] + rest)


@dataclass
class Contradiction:
    """One detected conflict.

    Field names `claim_a/claim_b/source_a/source_b` are kept exactly as the
    critic and synthesizer already consume them; everything else is additive.
    """

    claim_a: str
    claim_b: str
    source_a: str
    source_b: str
    kind: str                     # numeric | polarity | temporal
    severity: float               # 0-1
    detail: str
    values: Dict[str, Any] = field(default_factory=dict)
    intra_source: bool = False
    sub_question_a: str = ""
    sub_question_b: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_a": self.claim_a,
            "claim_b": self.claim_b,
            "source_a": self.source_a,
            "source_b": self.source_b,
            "kind": self.kind,
            "severity": round(self.severity, 3),
            "detail": self.detail,
            "values": self.values,
            "intra_source": self.intra_source,
            "sub_question_a": self.sub_question_a,
            "sub_question_b": self.sub_question_b,
        }


_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
_EVENT_WORDS = {
    "founded", "established", "launched", "published", "released", "signed",
    "enacted", "introduced", "announced", "began", "started", "completed",
    "discovered", "invented", "adopted", "approved", "banned",
}


def _temporal_conflict(a: str, b: str) -> Optional[Dict[str, Any]]:
    """Same dated event asserted with different years."""
    tokens_a, tokens_b = _content_tokens(a), _content_tokens(b)
    if not (tokens_a & _EVENT_WORDS) or not (tokens_b & _EVENT_WORDS):
        return None
    years_a = {int(y) for y in _YEAR_RE.findall(a or "")}
    years_b = {int(y) for y in _YEAR_RE.findall(b or "")}
    if not years_a or not years_b or (years_a & years_b):
        return None
    return {
        "years_a": sorted(years_a),
        "years_b": sorted(years_b),
        "gap": min(abs(x - y) for x in years_a for y in years_b),
    }


def detect_contradictions(
    facts: Sequence[Dict[str, Any]],
    *,
    subject_similarity: float = SUBJECT_SIMILARITY,
    divergence: float = NUMERIC_DIVERGENCE,
    max_pairs: int = 20_000,
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """Find conflicts across the evidence pool. Pure, deterministic, no LLM.

    Returns dicts (not dataclasses) so the result drops straight into the
    existing `contradictions` context the critic and synthesizer read.
    Ordered by severity, cross-source conflicts first.
    """
    items = [
        f for f in (facts or [])
        if isinstance(f, dict) and str(f.get("claim", "")).strip()
    ]
    if len(items) < 2:
        return []

    # Bucket by subject fingerprint; compare within buckets and, for numeric
    # claims, within unit groups across buckets (a divergent number about the
    # same unit is worth a look even if the wording drifted).
    buckets: Dict[Tuple[str, ...], List[int]] = {}
    for idx, fact in enumerate(items):
        key = _subject_key(str(fact.get("claim", "")))
        buckets.setdefault(key, []).append(idx)
        # Also index by each individual token so partial-overlap subjects meet.
        for token in key[:2]:
            buckets.setdefault((token,), []).append(idx)

    seen_pairs: Set[Tuple[int, int]] = set()
    found: List[Contradiction] = []
    comparisons = 0

    for indices in buckets.values():
        if len(indices) < 2:
            continue
        for i_pos in range(len(indices)):
            for j_pos in range(i_pos + 1, len(indices)):
                i, j = indices[i_pos], indices[j_pos]
                if i == j:
                    continue
                pair = (min(i, j), max(i, j))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                comparisons += 1
                if comparisons > max_pairs:
                    return _finalize(found, limit)

                fa, fb = items[i], items[j]
                claim_a = str(fa.get("claim", ""))
                claim_b = str(fb.get("claim", ""))
                source_a = str(fa.get("source", "") or "")
                source_b = str(fb.get("source", "") or "")

                similarity = semantic_similarity(claim_a, claim_b)
                if similarity < subject_similarity:
                    continue
                # Identical restatements are duplicates, not disputes.
                if similarity >= 0.95:
                    continue

                intra = bool(source_a and source_b) and extract_domain(source_a) == extract_domain(source_b)
                contradiction: Optional[Contradiction] = None

                numeric = numeric_conflict(claim_a, claim_b, divergence=divergence)
                if numeric:
                    rel = float(numeric["relative_divergence"])
                    severity = min(1.0, 0.35 + rel)
                    unit = numeric["unit"]
                    contradiction = Contradiction(
                        claim_a=claim_a, claim_b=claim_b,
                        source_a=source_a, source_b=source_b,
                        kind="numeric",
                        severity=severity * (0.5 if intra else 1.0),
                        detail=(
                            f"Sources disagree on the same {unit} measure: "
                            f"{numeric['raw_a']} vs {numeric['raw_b']} "
                            f"({rel:.0%} apart)."
                        ),
                        values=numeric,
                        intra_source=intra,
                    )
                elif claim_polarity(claim_a) and claim_polarity(claim_b) and \
                        claim_polarity(claim_a) != claim_polarity(claim_b) and similarity >= 0.45:
                    contradiction = Contradiction(
                        claim_a=claim_a, claim_b=claim_b,
                        source_a=source_a, source_b=source_b,
                        kind="polarity",
                        severity=(0.55 + 0.35 * similarity) * (0.5 if intra else 1.0),
                        detail=(
                            "Sources assert opposite directions for the same "
                            "relationship."
                        ),
                        values={
                            "polarity_a": claim_polarity(claim_a),
                            "polarity_b": claim_polarity(claim_b),
                            "similarity": round(similarity, 3),
                        },
                        intra_source=intra,
                    )
                else:
                    temporal = _temporal_conflict(claim_a, claim_b)
                    if temporal:
                        contradiction = Contradiction(
                            claim_a=claim_a, claim_b=claim_b,
                            source_a=source_a, source_b=source_b,
                            kind="temporal",
                            severity=min(1.0, 0.4 + temporal["gap"] / 50.0) * (0.5 if intra else 1.0),
                            detail=(
                                f"Sources date the same event differently: "
                                f"{temporal['years_a']} vs {temporal['years_b']}."
                            ),
                            values=temporal,
                            intra_source=intra,
                        )

                if contradiction:
                    contradiction.sub_question_a = str(fa.get("sub_question", "") or "")
                    contradiction.sub_question_b = str(fb.get("sub_question", "") or "")
                    found.append(contradiction)

    return _finalize(found, limit)


def _finalize(found: List[Contradiction], limit: int) -> List[Dict[str, Any]]:
    found.sort(key=lambda c: (c.intra_source, -c.severity))
    return [c.to_dict() for c in found[: max(1, limit)]]


# ---------------------------------------------------------------------------
# Resolution guidance
# ---------------------------------------------------------------------------

def summarize_contradictions(contradictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate view the report and the confidence engine both need."""
    items = [c for c in (contradictions or []) if isinstance(c, dict)]
    cross = [c for c in items if not c.get("intra_source")]
    severe = [c for c in cross if float(c.get("severity", 0.0) or 0.0) >= SEVERE_DIVERGENCE]
    by_kind: Dict[str, int] = {}
    for c in items:
        by_kind[str(c.get("kind", "unknown"))] = by_kind.get(str(c.get("kind", "unknown")), 0) + 1
    return {
        "total": len(items),
        "cross_source": len(cross),
        "severe": len(severe),
        "by_kind": by_kind,
        "max_severity": round(max((float(c.get("severity", 0.0) or 0.0) for c in items), default=0.0), 3),
    }


def numeric_ranges(contradictions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn numeric conflicts into reportable ranges.

    This is the whole point of detecting them: the correct output for
    "7% / 11% / 18%" is a 7-18% range with the reason for the spread, never an
    average and never a silently chosen winner.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for c in contradictions or []:
        if c.get("kind") != "numeric":
            continue
        values = c.get("values") or {}
        unit = str(values.get("unit", "") or "dimensionless")
        grouped.setdefault(unit, []).append(c)

    out: List[Dict[str, Any]] = []
    for unit, group in grouped.items():
        numbers: List[float] = []
        sources: List[str] = []
        for c in group:
            values = c.get("values") or {}
            for key, src in (("value_a", "source_a"), ("value_b", "source_b")):
                try:
                    numbers.append(float(values.get(key)))
                except (TypeError, ValueError):
                    continue
                source = str(c.get(src, "") or "")
                if source and source not in sources:
                    sources.append(source)
        if len(numbers) < 2:
            continue
        out.append({
            "unit": unit,
            "low": min(numbers),
            "high": max(numbers),
            "spread": round(max(numbers) - min(numbers), 4),
            "sources": sources[:8],
            "guidance": (
                "Report as a range with the assumptions behind each end, not a "
                "point estimate; the spread is the finding."
            ),
        })
    out.sort(key=lambda r: r["spread"], reverse=True)
    return out


def contradiction_followups(
    contradictions: Sequence[Dict[str, Any]], limit: int = 3
) -> List[str]:
    """Search queries that would RESOLVE the biggest conflicts.

    Feeding these back into the next pass is what turns contradiction
    detection from a report footnote into an actual research loop: the system
    goes looking for the methodology that explains the disagreement.
    """
    queries: List[str] = []
    for c in sorted(
        (c for c in contradictions or [] if not c.get("intra_source")),
        key=lambda c: -float(c.get("severity", 0.0) or 0.0),
    ):
        claim = str(c.get("claim_a", ""))
        subject = " ".join(sorted(_content_tokens(claim) - _MEASURE_WORDS)[:4])
        if not subject:
            continue
        kind = c.get("kind")
        if kind == "numeric":
            unit = str((c.get("values") or {}).get("unit", "")) or "figure"
            query = f"{subject} {unit} methodology assumptions why estimates differ"
        elif kind == "polarity":
            query = f"{subject} evidence for and against systematic review"
        else:
            query = f"{subject} authoritative timeline primary source date"
        query = re.sub(r"\s+", " ", query).strip()
        if query and query not in queries:
            queries.append(query)
        if len(queries) >= max(1, limit):
            break
    return queries
