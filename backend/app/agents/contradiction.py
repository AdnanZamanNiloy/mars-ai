"""Contradiction resolution helpers (vision Feature 09).

Detection lives in `app.core.contradictions.find_contradictions` — the live
engine the host workflow runs (numeric, polarity and temporal conflicts).
This module keeps the deterministic helpers the critic, synthesizer and
red-team consume: aggregate summaries, reportable numeric ranges, and
follow-up search queries for the biggest unresolved conflicts.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Set

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


# ---------------------------------------------------------------------------
# Resolution guidance
# ---------------------------------------------------------------------------

def summarize_contradictions(contradictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate view the report and the confidence engine both need.

    Fix C: a contradiction whose `resolved` flag is set (different
    period/scope/metric explains the spread) is recorded but never counted as
    a cross-source or severe conflict — it must not make the red-team or the
    confidence engine treat an explained spread as a live disagreement.
    """
    items = [c for c in (contradictions or []) if isinstance(c, dict)]
    unresolved = [c for c in items if not c.get("resolved")]
    cross = [c for c in unresolved if not c.get("intra_source")]
    severe = [c for c in cross if float(c.get("severity", 0.0) or 0.0) >= SEVERE_DIVERGENCE]
    by_kind: Dict[str, int] = {}
    for c in items:
        by_kind[str(c.get("kind", "unknown"))] = by_kind.get(str(c.get("kind", "unknown")), 0) + 1
    return {
        "total": len(items),
        "resolved": len(items) - len(unresolved),
        "unresolved": len(unresolved),
        "cross_source": len(cross),
        "severe": len(severe),
        "by_kind": by_kind,
        "max_severity": round(max((float(c.get("severity", 0.0) or 0.0) for c in unresolved), default=0.0), 3),
    }


def numeric_ranges(contradictions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn numeric conflicts into reportable ranges.

    This is the whole point of detecting them: the correct output for
    "7% / 11% / 18%" is a 7-18% range with the reason for the spread, never an
    average and never a silently chosen winner.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for c in contradictions or []:
        if c.get("kind") != "numeric" or c.get("resolved"):
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
