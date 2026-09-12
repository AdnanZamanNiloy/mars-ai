"""Evidence-based confidence (vision Feature 10) — v3 API surface.

One engine, one name. The live confidence engine is
`app.core.confidence.compute_confidence`: it is what the host workflow reports
to the user, and its weights are what historical score comparability depends
on. This module no longer carries a second engine. `compute_confidence` here
is a thin adapter over the live engine that keeps the v3 call surface —
`ConfidenceReport`, `SUFFICIENCY_THRESHOLD`, the keyword arguments mission.py
and stopping.py are specified against — so the deferred mission/decision ports
import a single name and cannot silently reintroduce divergent scoring.

Inputs the live engine does not consume yet (contradictions, citation
support, planned axes, target domains) are accepted and recorded as notes,
never silently dropped, until the mission port wires them for real.
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
) -> ConfidenceReport:
    """Score an evidence pool through the live engine (app.core.confidence).

    v3 keyword mapping: `redteam_survival` (preferred) or `critic_confidence`
    becomes the engine's critic-survival signal — sufficient at or above
    SUFFICIENCY_THRESHOLD, else the engine's stopped-early value;
    `degraded_stages` maps to the engine's `degraded` cap. `contradictions`,
    `citation_support`, `planned_axes` and `target_domains` have no
    live-engine input yet and are recorded as notes.
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
    for name, value in (
        ("contradictions", contradictions),
        ("citation_support", citation_support),
        ("planned_axes", planned_axes),
        ("target_domains", target_domains),
    ):
        if value:
            plain.append(f"v3 input {name!r} recorded but not wired into the live engine yet.")

    return ConfidenceReport(
        overall=overall,
        level=_level(overall),
        signals={k: float(v) for k, v in live.get("signals", {}).items()},
        caps_applied=caps,
        notes=plain,
        stats={
            "weights": dict(live.get("weights", {})),
            "source": "app.core.confidence",
        },
    )


def confidence_delta(before: Optional[ConfidenceReport], after: ConfidenceReport) -> float:
    if before is None:
        return after.overall
    return round(after.overall - before.overall, 4)
