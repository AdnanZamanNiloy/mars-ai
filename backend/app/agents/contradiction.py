"""Contradiction engine (vision Feature 09) — v3 API surface.

One detector, one name. The live detector is
`app.core.contradictions.find_contradictions` — the numeric-band scan the host
workflow runs. This module no longer carries a second detector.
`detect_contradictions` here is a thin adapter over the live engine that keeps
the v3 result shape (kind/severity dicts, the `Contradiction` record, and the
`summarize_contradictions` / `numeric_ranges` / `contradiction_followups`
helpers the critic, synthesizer and red-team already consume) so the deferred
mission port imports a single name.

The live detector currently reports numeric conflicts only; polarity and
temporal detection return with the mission-port adapter commit, where the v3
detection rules are re-decided against the mission's own tests.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

# Two claims must be about the same thing before their difference means
# anything. The live engine enforces its own similarity band
# (SIMILARITY_LOW/HIGH in app.core.contradictions); this constant remains the
# documented v3 default for the `subject_similarity` kwarg.
SUBJECT_SIMILARITY = 0.34

# Relative divergence at which two numbers stop being rounding variants.
# Applied by this adapter as a post-filter on live-engine findings, whose own
# threshold is lower.
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


def detect_contradictions(
    facts: Sequence[Dict[str, Any]],
    *,
    subject_similarity: float = SUBJECT_SIMILARITY,
    divergence: float = NUMERIC_DIVERGENCE,
    max_pairs: int = 20_000,
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """Find conflicts across the evidence pool by delegating to the live
    engine (`app.core.contradictions.find_contradictions`), reshaped into the
    v3 dict form. Pure, deterministic, no LLM.

    `divergence` post-filters the live engine's findings (its own threshold
    is lower). `subject_similarity` and `max_pairs` are accepted for
    call-site compatibility; the live engine's similarity band and finding
    cap stand until the mission port revisits them.
    """
    from app.core.contradictions import find_contradictions

    found: List[Contradiction] = []
    for raw in find_contradictions([f for f in (facts or []) if isinstance(f, dict)]):
        pair = _report_pair(raw, divergence)
        if pair is None:
            continue
        rel = pair["relative_divergence"]
        found.append(
            Contradiction(
                claim_a=str(raw.get("claim_a", "")),
                claim_b=str(raw.get("claim_b", "")),
                source_a=str(raw.get("source_a", "") or ""),
                source_b=str(raw.get("source_b", "") or ""),
                kind="numeric",
                severity=min(1.0, 0.35 + rel),
                detail=str(raw.get("note", "")),
                values={
                    # The live engine normalizes scales but does not extract
                    # units yet; numeric_ranges groups these under the
                    # dimensionless bucket until it does.
                    "unit": "dimensionless",
                    "value_a": pair["value_a"],
                    "value_b": pair["value_b"],
                    "relative_divergence": round(rel, 4),
                    "topic_similarity": float(raw.get("topic_similarity", 0.0)),
                },
                # The live engine already skips same-source pairs.
                intra_source=False,
            )
        )

    return _finalize(found, limit)


def _report_pair(raw: Dict[str, Any], divergence: float) -> Optional[Dict[str, Any]]:
    """The value pair to report for a live-engine finding.

    Claims often carry incidental numbers (years, counts) alongside the
    disputed measure, and the largest raw divergence is usually one of those,
    not the real conflict. The tightest pair that still passes `divergence`
    is the most conservative genuine conflict; incidental numbers are
    reported only when nothing better qualifies.
    """
    best: Optional[Dict[str, Any]] = None
    for va in raw.get("value_a") or []:
        for vb in raw.get("value_b") or []:
            bigger = max(abs(va), abs(vb))
            if bigger == 0:
                continue
            rel = abs(va - vb) / bigger
            if rel < divergence:
                continue
            if best is None or rel < best["relative_divergence"]:
                best = {"value_a": va, "value_b": vb, "relative_divergence": rel}
    return best


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
