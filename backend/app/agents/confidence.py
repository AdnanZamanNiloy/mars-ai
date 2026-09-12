"""Evidence-based confidence engine (vision Feature 10).

The old pipeline's confidence came from the critic's own `confidence` field —
a number the model chose about its own judgement, clamped and then reported to
the user as "87%". A model's self-assessment is not a measurement: it does not
know how many distinct domains were consulted, whether any claim was
independently corroborated, how stale the evidence is, or that two sources
disagreed by 60%.

This module computes confidence from things that were actually observed. Eight
signals, each in [0, 1], each independently inspectable, combined by fixed
weights into an overall score. The model's own opinion is one input among
eight, capped at its weight, and it can no longer override the evidence.

The scoring must be conservative in a specific way: it should be *hard* to
reach high confidence and *easy* to lose it. So there are explicit ceilings —
a single-domain evidence pool cannot exceed 0.55 no matter how good it looks,
and a severe unresolved contradiction caps the run below the sufficiency line.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.agents.contradiction import summarize_contradictions
from app.agents.evidence_utils import evidence_stats

# Weights sum to 1.0. Ordering reflects what actually predicts a correct
# report: verification and corroboration first, model opinion last.
SIGNAL_WEIGHTS: Dict[str, float] = {
    "claim_verification": 0.22,
    "cross_source_agreement": 0.16,
    "source_quality": 0.14,
    "source_diversity": 0.12,
    "evidence_coverage": 0.12,
    "primary_source_share": 0.10,
    "freshness": 0.07,
    "critic_survival": 0.07,
}

SUFFICIENCY_THRESHOLD = 0.75


@dataclass
class ConfidenceReport:
    overall: float
    level: str
    signals: Dict[str, float] = field(default_factory=dict)
    caps_applied: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_sufficient(self) -> bool:
        return self.overall >= SUFFICIENCY_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": round(self.overall, 4),
            "level": self.level,
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "weights": dict(SIGNAL_WEIGHTS),
            "caps_applied": list(self.caps_applied),
            "notes": list(self.notes),
            "stats": self.stats,
            "is_sufficient": self.is_sufficient,
        }

    def render(self) -> str:
        """The Feature-10 panel, as text the report can embed verbatim."""
        rows = [
            f"{name.replace('_', ' ').title():<24}{value * 100:>4.0f}%"
            for name, value in sorted(
                self.signals.items(), key=lambda kv: -SIGNAL_WEIGHTS.get(kv[0], 0)
            )
        ]
        body = "\n".join(rows)
        return (
            "RESEARCH CONFIDENCE\n\n"
            f"{body}\n"
            f"{'-' * 28}\n"
            f"{'OVERALL CONFIDENCE':<24}{self.overall * 100:>4.0f}%"
        )


def _level(score: float) -> str:
    if score >= 0.85:
        return "very_high"
    if score >= SUFFICIENCY_THRESHOLD:
        return "high"
    if score >= 0.55:
        return "medium"
    if score >= 0.35:
        return "low"
    return "very_low"


def _saturating(count: float, target: float) -> float:
    """Diminishing-returns ramp: `target` occurrences score ~0.86, more helps
    less. Prevents "we found 40 sources" from reading as certainty."""
    if target <= 0:
        return 0.0
    return round(1.0 - pow(2.718281828, -2.0 * (max(0.0, count) / target)), 4)


def compute_confidence(
    facts: Sequence[Dict[str, Any]],
    *,
    contradictions: Optional[Sequence[Dict[str, Any]]] = None,
    critic_confidence: Optional[float] = None,
    citation_support: Optional[Dict[str, Any]] = None,
    planned_axes: int = 0,
    redteam_survival: Optional[float] = None,
    degraded_stages: Sequence[str] = (),
    target_domains: int = 5,
) -> ConfidenceReport:
    """Score a finished (or in-progress) evidence pool.

    Every argument is optional so this can be called mid-run for the stopping
    decision and again at the end for the report, with the same weights.
    """
    stats = evidence_stats(facts)
    total = int(stats["total"])
    signals: Dict[str, float] = {}
    notes: List[str] = []
    caps: List[str] = []

    if total == 0:
        return ConfidenceReport(
            overall=0.0, level="very_low",
            signals={k: 0.0 for k in SIGNAL_WEIGHTS},
            caps_applied=["no evidence"],
            notes=["No usable evidence was extracted."],
            stats=stats,
        )

    # 1. Verification: share of claims whose numbers and vocabulary were found
    #    in the source they cite.
    signals["claim_verification"] = round(stats["verified"] / total, 4)

    # 2. Cross-source agreement: share of claims independently corroborated.
    #    This is the signal the old dedup silently deleted.
    signals["cross_source_agreement"] = round(stats["corroborated"] / total, 4)

    # 3. Source quality: mean authority of the DOCUMENTS behind the evidence,
    #    not of the claims — one page cited ten times is one source.
    signals["source_quality"] = _authority_mean(facts)

    # 4. Diversity: distinct domains, saturating.
    signals["source_diversity"] = _saturating(stats["distinct_domains"], max(2, target_domains))

    # 5. Coverage: how many planned research angles produced evidence.
    if planned_axes > 0:
        signals["evidence_coverage"] = round(min(1.0, stats["axes_covered"] / planned_axes), 4)
    else:
        signals["evidence_coverage"] = _saturating(stats["axes_covered"], 4)

    # 6. Primary sources: share of documents that published the fact rather
    #    than reporting it.
    signals["primary_source_share"] = float(stats["primary_share"])

    # 7. Freshness: decay-weighted recency.
    signals["freshness"] = float(stats["freshness"])

    # 8. Critic / red-team survival. Prefer a measured survival rate; fall back
    #    to the critic's own number, which is the only place model opinion
    #    enters the score.
    if redteam_survival is not None:
        signals["critic_survival"] = round(max(0.0, min(1.0, float(redteam_survival))), 4)
    elif critic_confidence is not None:
        signals["critic_survival"] = round(max(0.0, min(1.0, float(critic_confidence))), 4)
        notes.append("Critic survival falls back to the critic's self-reported confidence.")
    else:
        signals["critic_survival"] = 0.5

    overall = sum(signals[name] * weight for name, weight in SIGNAL_WEIGHTS.items())

    # ------------------------------------------------------------------
    # Hard ceilings. A weighted average can be dragged up by six mediocre
    # signals; these encode failures that must dominate the average.
    # ------------------------------------------------------------------
    if stats["distinct_domains"] < 2:
        overall = min(overall, 0.55)
        caps.append("single-domain evidence pool capped at 0.55")
    if stats["verified"] == 0:
        overall = min(overall, 0.45)
        caps.append("zero verified claims capped at 0.45")
    if total < 4:
        overall = min(overall, 0.60)
        caps.append(f"thin pool ({total} claims) capped at 0.60")

    summary = summarize_contradictions(contradictions or [])
    if summary["severe"]:
        overall = min(overall, 0.70)
        caps.append(f"{summary['severe']} severe unresolved contradiction(s) capped at 0.70")
    elif summary["cross_source"] >= 3:
        overall = min(overall, 0.80)
        caps.append(f"{summary['cross_source']} cross-source conflicts capped at 0.80")

    if citation_support:
        rate = citation_support.get("rate")
        if isinstance(rate, (int, float)) and citation_support.get("cited"):
            # A report whose own sentences do not trace to its cited sources
            # cannot be trusted above that trace rate.
            overall = min(overall, 0.35 + 0.65 * float(rate))
            caps.append(f"citation support rate {float(rate):.0%} bounds confidence")
        numeric_rate = citation_support.get("numeric_rate")
        if isinstance(numeric_rate, (int, float)) and numeric_rate < 0.8:
            overall = min(overall, 0.65)
            caps.append(f"numeric grounding {float(numeric_rate):.0%} capped at 0.65")

    degraded = [str(d) for d in (degraded_stages or []) if d]
    if degraded:
        penalty = min(0.15, 0.05 * len(degraded))
        overall -= penalty
        notes.append(
            f"Deterministic fallback covered {', '.join(degraded)}; "
            f"confidence reduced by {penalty:.2f}."
        )

    overall = round(max(0.0, min(1.0, overall)), 4)

    if signals["primary_source_share"] < 0.2:
        notes.append("Few primary sources: the report rests mainly on secondary reporting.")
    if signals["cross_source_agreement"] < 0.15 and total >= 6:
        notes.append("Almost no claim was independently corroborated by a second domain.")
    if signals["freshness"] < 0.35:
        notes.append("Evidence skews old for a time-sensitive question.")

    return ConfidenceReport(
        overall=overall,
        level=_level(overall),
        signals=signals,
        caps_applied=caps,
        notes=notes,
        stats=stats,
    )


def _authority_mean(facts: Sequence[Dict[str, Any]]) -> float:
    from app.agents.sources import authority_score, canonical_url

    seen: Dict[str, float] = {}
    for fact in facts or []:
        url = str(fact.get("source", "") or "")
        key = canonical_url(url)
        if key and key not in seen:
            seen[key] = authority_score(url)
    if not seen:
        return 0.0
    return round(sum(seen.values()) / len(seen), 4)


def confidence_delta(before: Optional[ConfidenceReport], after: ConfidenceReport) -> float:
    if before is None:
        return after.overall
    return round(after.overall - before.overall, 4)
