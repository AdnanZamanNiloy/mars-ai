"""Evidence-based confidence (vision Feature 10) — v3 API surface.

One engine, one name. The live confidence engine is
`app.core.confidence.compute_confidence`: it is what the host workflow reports
to the user, and its weights are what historical score comparability depends
on. This module no longer carries a second engine. `compute_confidence` here
is a thin adapter over the live engine that keeps the v3 call surface —
`ConfidenceReport`, `SUFFICIENCY_THRESHOLD`, the keyword arguments mission.py
and stopping.py are specified against — so the deferred mission/decision ports
import a single name and cannot silently reintroduce divergent scoring.

Epistemic adjustments (new)
---------------------------
The live engine scores the POOL — how much evidence, how well sourced, how
verified. Four signals it does not consume were previously accepted and
recorded as a note saying they were "not wired in yet", which meant a report
could detect five severe source conflicts and still publish "High (0.82)".
The headline confidence number is the one thing a reader actually trusts, so
a detected conflict that does not move it is worse than not detecting it.

`apply_epistemic_adjustments` now applies them as explicit, ordered, NAMED
caps and penalties on top of the live engine's score:

  * unresolved source conflicts       — cap, scaled by how many
  * unmet claim-type evidence standards — cap (a causal claim on one blog
    cannot support a high-confidence report even if the pool is large)
  * missing planned axes              — proportional penalty
  * weak citation support             — proportional penalty
  * one-sided evidence on a comparison — cap
  * stale evidence on a time-sensitive question — cap

The live engine's weights are untouched, so historical score comparability
survives: adjustments are applied after it and every one of them is recorded
in `caps_applied` with its name and magnitude. Nothing is ever silently
fudged — a score that moved can always be explained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# The sufficiency line the critic gate and stopping logic share. The live
# engine's DEGRADED_CAP (app.core.confidence) keeps degraded runs below it.
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
            "weights": dict(self.stats.get("weights", {})),
            "caps_applied": list(self.caps_applied),
            "notes": list(self.notes),
            "stats": self.stats,
            "is_sufficient": self.is_sufficient,
        }

    def render(self) -> str:
        """The Feature-10 panel, as text the report can embed verbatim."""
        weights = self.stats.get("weights", {})
        rows = [
            f"{name.replace('_', ' ').title():<24}{value * 100:>4.0f}%"
            for name, value in sorted(
                self.signals.items(), key=lambda kv: -weights.get(kv[0], 0)
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
    covered_axes: int = 0,
    epistemics: Any = None,
    query: str = "",
) -> ConfidenceReport:
    """Score an evidence pool through the live engine (app.core.confidence).

    v3 keyword mapping: `redteam_survival` (preferred) or `critic_confidence`
    becomes the engine's critic-survival signal — sufficient at or above
    SUFFICIENCY_THRESHOLD, else the engine's stopped-early value;
    `degraded_stages` maps to the engine's `degraded` cap.

    `contradictions`, `citation_support`, `planned_axes` and `target_domains`
    are applied AFTER the live engine by `apply_epistemic_adjustments`, as
    named caps and penalties rather than as engine inputs — the engine's
    weights stay fixed so scores remain comparable over time, and every
    adjustment is recorded in `caps_applied` and `stats`.

    Pass `epistemics` (an `app.agents.epistemics.EpistemicReport`) whenever
    one is available: it has already separated genuine disagreements from
    time-series and scope artifacts, so the conflict penalty lands on real
    conflicts instead of on every numeric pair the detector fired on.
    """
    if not [f for f in (facts or []) if isinstance(f, dict)]:
        return ConfidenceReport(
            overall=0.0,
            level="very_low",
            signals={},
            caps_applied=["no evidence"],
            notes=["No usable evidence was extracted."],
            stats={"source": "app.core.confidence"},
        )

    from app.core.confidence import compute_confidence as _live_engine

    survival = (
        redteam_survival if redteam_survival is not None else critic_confidence
    )
    critique: Dict[str, Any] = {}
    if survival is not None:
        critique["is_sufficient"] = float(survival) >= SUFFICIENCY_THRESHOLD

    live = _live_engine(
        list(facts),
        critique,
        0,
        1,
        degraded=[str(d) for d in (degraded_stages or [])],
    )

    overall = float(live.get("overall", 0.0))
    notes = list(live.get("notes", []))
    caps = [n for n in notes if "capped" in n]
    plain = [n for n in notes if "capped" not in n]

    report = ConfidenceReport(
        overall=overall,
        level=_level(overall),
        signals={k: float(v) for k, v in live.get("signals", {}).items()},
        caps_applied=caps,
        notes=plain,
        stats={
            "weights": dict(live.get("weights", {})),
            "source": "app.core.confidence",
            "engine_overall": round(overall, 4),
        },
    )

    return apply_epistemic_adjustments(
        report,
        facts=facts,
        contradictions=contradictions,
        citation_support=citation_support,
        planned_axes=planned_axes,
        covered_axes=covered_axes,
        target_domains=target_domains,
        epistemics=epistemics,
        query=query,
    )


# ---------------------------------------------------------------------------
# Epistemic adjustments: the signals the pool score cannot see
# ---------------------------------------------------------------------------

# Caps for unresolved GENUINE conflicts, by count. A report that cannot say
# which of two contradictory figures is correct is not a high-confidence
# report, however much evidence it gathered — the disagreement is precisely
# about the thing the reader wants to know.
_CONFLICT_CAPS: Dict[int, float] = {1: 0.72, 2: 0.65, 3: 0.55}
_CONFLICT_CAP_FLOOR = 0.50

# One-sided evidence on a comparison: the report looks balanced and is not.
_ASYMMETRY_CAP = 0.70

# Time-sensitive question answered from dated evidence.
_STALE_CAP = 0.68

# A missing planned angle is a hole in the answer, not just less evidence.
_AXIS_PENALTY_PER_GAP = 0.06
_AXIS_PENALTY_MAX = 0.24


def apply_epistemic_adjustments(
    report: ConfidenceReport,
    *,
    facts: Sequence[Dict[str, Any]] = (),
    contradictions: Optional[Sequence[Dict[str, Any]]] = None,
    citation_support: Optional[Dict[str, Any]] = None,
    planned_axes: int = 0,
    covered_axes: int = 0,
    target_domains: int = 5,
    epistemics: Any = None,
    query: str = "",
) -> ConfidenceReport:
    """Apply the caps and penalties the live pool engine cannot compute.

    Deliberately a separate, pure function: the live engine's weights stay
    untouched (so historical scores remain comparable) and every adjustment
    made here is named and recorded, so a score that moved can be explained
    to the person reading the report.

    Adjustments compose as: penalties subtract, caps clamp, and the lowest
    cap wins. Order does not matter to the result, which makes the whole
    thing testable one signal at a time.
    """
    overall = float(report.overall)
    caps: List[str] = list(report.caps_applied)
    notes: List[str] = list(report.notes)
    adjustments: Dict[str, float] = {}
    before = overall

    # --- epistemic layer: adjudicated conflicts + claim standards ----------
    # Preferred input: a full EpistemicReport, which has already separated
    # real disagreements from time-series and scope artifacts. Without it we
    # fall back to the raw contradiction count, which overcounts — so the
    # fallback is deliberately gentler than the informed path.
    ceiling: Optional[float] = None
    if epistemics is not None and hasattr(epistemics, "confidence_ceiling"):
        ceiling = _as_float(getattr(epistemics, "confidence_ceiling", None))
        for note in _as_list(getattr(epistemics, "notes", lambda: [])()):
            notes.append(str(note))
        if ceiling is not None and ceiling < overall:
            caps.append(f"epistemics ceiling {ceiling:.2f} (conflicts / claim standards)")
            overall = ceiling
    elif contradictions:
        unresolved = sum(
            1 for c in contradictions
            if isinstance(c, dict) and not c.get("resolved")
        )
        if unresolved:
            cap = _CONFLICT_CAPS.get(min(unresolved, 3), _CONFLICT_CAP_FLOOR)
            if unresolved > 3:
                cap = _CONFLICT_CAP_FLOOR
            if cap < overall:
                caps.append(
                    f"unresolved source conflicts x{unresolved} capped confidence at {cap:.2f}"
                )
                overall = cap
            notes.append(
                f"{unresolved} source conflict(s) remain unresolved; confidence is "
                "capped until one side is adjudicated or the range is accepted."
            )

    # --- planned axes actually covered -------------------------------------
    if planned_axes > 0:
        covered = max(0, int(covered_axes))
        if covered == 0:
            covered = _covered_axes_from_facts(facts)
        missing = max(0, int(planned_axes) - covered)
        if missing:
            penalty = min(_AXIS_PENALTY_MAX, missing * _AXIS_PENALTY_PER_GAP)
            adjustments["missing_axes"] = -penalty
            overall -= penalty
            notes.append(
                f"{missing} of {planned_axes} planned research angles produced no "
                f"evidence; confidence reduced by {penalty:.2f}."
            )

    # --- citation support of the written answer ----------------------------
    support = _support_ratio(citation_support)
    if support is not None and support < 0.85:
        penalty = min(0.20, (0.85 - support) * 0.5)
        adjustments["weak_citation_support"] = -penalty
        overall -= penalty
        notes.append(
            f"Only {support:.0%} of the answer's factual sentences trace to a "
            f"cited source; confidence reduced by {penalty:.2f}."
        )

    # --- source breadth against the mode's target --------------------------
    domains = _distinct_domains(facts)
    if target_domains > 0 and domains and domains < target_domains:
        shortfall = (target_domains - domains) / max(1, target_domains)
        penalty = min(0.15, round(shortfall * 0.15, 4))
        if penalty >= 0.01:
            adjustments["narrow_source_base"] = -penalty
            overall -= penalty
            notes.append(
                f"Evidence spans {domains} domain(s) against a target of "
                f"{target_domains}; confidence reduced by {penalty:.2f}."
            )

    # --- one-sided comparison ----------------------------------------------
    asymmetry = getattr(epistemics, "asymmetry", None) if epistemics is not None else None
    if asymmetry is not None and not getattr(asymmetry, "balanced", True):
        if _ASYMMETRY_CAP < overall:
            caps.append(f"one-sided evidence capped confidence at {_ASYMMETRY_CAP:.2f}")
            overall = _ASYMMETRY_CAP

    # --- staleness on a time-sensitive question ----------------------------
    temporal = _temporal_for(facts, query)
    if temporal is not None and getattr(temporal, "stale", False):
        if _STALE_CAP < overall:
            caps.append(
                f"evidence too dated for a time-sensitive question, capped at {_STALE_CAP:.2f}"
            )
            overall = _STALE_CAP

    overall = max(0.0, min(1.0, round(overall, 4)))
    stats = dict(report.stats)
    stats["epistemic_adjustments"] = adjustments
    stats["engine_overall"] = stats.get("engine_overall", round(before, 4))
    stats["adjusted_delta"] = round(overall - before, 4)

    return ConfidenceReport(
        overall=overall,
        level=_level(overall),
        signals=dict(report.signals),
        caps_applied=caps,
        notes=notes,
        stats=stats,
    )


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _support_ratio(citation_support: Optional[Dict[str, Any]]) -> Optional[float]:
    """Pull a 0-1 support ratio out of whatever shape the caller passed.

    Callers hand this in as a `verify_answer_support` result, a
    `CitationAudit.to_dict()`, or a bare ratio, so all three are accepted
    rather than making every call site normalize first.
    """
    if citation_support is None:
        return None
    if isinstance(citation_support, (int, float)):
        return max(0.0, min(1.0, float(citation_support)))
    if not isinstance(citation_support, dict):
        return None
    for key in ("support_ratio", "citation_density", "supported_ratio", "ratio"):
        value = _as_float(citation_support.get(key))
        if value is not None:
            return max(0.0, min(1.0, value))
    supported = _as_float(citation_support.get("supported"))
    total = _as_float(citation_support.get("total"))
    if supported is not None and total:
        return max(0.0, min(1.0, supported / total))
    cited = _as_float(citation_support.get("cited_sentences"))
    sentences = _as_float(citation_support.get("total_sentences"))
    if cited is not None and sentences:
        return max(0.0, min(1.0, cited / sentences))
    return None


def _covered_axes_from_facts(facts: Sequence[Dict[str, Any]]) -> int:
    axes = {
        str(f.get("sub_question", "") or "").strip()
        for f in (facts or ())
        if isinstance(f, dict) and str(f.get("sub_question", "") or "").strip()
    }
    return len(axes)


def _distinct_domains(facts: Sequence[Dict[str, Any]]) -> int:
    try:
        from app.agents.evidence_utils import extract_domain
    except Exception:  # noqa: BLE001
        return 0
    domains = {
        extract_domain(str(f.get("source", "") or ""))
        for f in (facts or ())
        if isinstance(f, dict) and f.get("source")
    }
    domains.discard("")
    return len(domains)


def _temporal_for(facts: Sequence[Dict[str, Any]], query: str) -> Any:
    """Recency profile for these facts, or None when unavailable."""
    if not facts:
        return None
    try:
        from app.agents.research_quality import temporal_profile
    except Exception:  # noqa: BLE001
        return None
    try:
        from app.agents.orchestrator import classify_query_type

        query_type = classify_query_type(query) if query else ""
    except Exception:  # noqa: BLE001
        query_type = ""
    try:
        return temporal_profile(facts, query_type=query_type)
    except Exception:  # noqa: BLE001
        return None


def confidence_delta(before: Optional[ConfidenceReport], after: ConfidenceReport) -> float:
    if before is None:
        return after.overall
    return round(after.overall - before.overall, 4)
