"""Dynamic research depth and intelligent stopping (vision Feature 11).

The previous loop had exactly one stop rule — "the critic said yes, or we hit
max_iterations" — plus deterministic gates so strict (avg_confidence >= 0.74,
a definitional claim required, four facts minimum) that most runs went to the
ceiling. Two failure modes fell out of that:

  over-research  a question already answered in pass 1 kept searching, because
                 one arbitrary threshold sat a hundredth below its line
  under-research a genuinely hard question stopped at the ceiling with visible
                 gaps and no record of what was still missing

Both are the same missing concept: *marginal information gain*. A pass that
adds no new domains, no new verified claims and no confidence is not research,
it is spend. This module measures each pass and decides.

It is deliberately not an LLM. A stopping rule that costs a model call is a
stopping rule you cannot afford to consult often, and the inputs it needs are
all counts the pipeline already has.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from app.agents.confidence import SUFFICIENCY_THRESHOLD, ConfidenceReport
from app.agents.evidence_utils import canonical_url, extract_domain

# A pass must clear at least one of these to have earned its cost.
MIN_NEW_VERIFIED = 1
MIN_NEW_DOMAINS = 1
MIN_CONFIDENCE_GAIN = 0.02

# Composite gain below this counts as a stall; two consecutive stalls stop.
STALL_GAIN = 0.08


@dataclass
class PassMetrics:
    """What one research pass actually produced."""

    iteration: int
    total_claims: int
    verified_claims: int
    domains: int
    documents: int
    axes_covered: int
    confidence: float
    new_claims: int = 0
    new_verified: int = 0
    new_domains: int = 0
    new_documents: int = 0
    confidence_gain: float = 0.0
    marginal_gain: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "total_claims": self.total_claims,
            "verified_claims": self.verified_claims,
            "domains": self.domains,
            "documents": self.documents,
            "axes_covered": self.axes_covered,
            "confidence": round(self.confidence, 4),
            "new_claims": self.new_claims,
            "new_verified": self.new_verified,
            "new_domains": self.new_domains,
            "new_documents": self.new_documents,
            "confidence_gain": round(self.confidence_gain, 4),
            "marginal_gain": round(self.marginal_gain, 4),
        }


@dataclass
class StopDecision:
    stop: bool
    reason: str
    code: str
    marginal_gain: float = 0.0
    gaps: List[str] = field(default_factory=list)
    followups: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stop": self.stop,
            "reason": self.reason,
            "code": self.code,
            "marginal_gain": round(self.marginal_gain, 4),
            "gaps": list(self.gaps),
            "followups": list(self.followups),
        }


class DepthController:
    """Tracks pass-over-pass progress and decides whether to keep going.

    One instance per mission. `observe()` after each pass, `decide()` to get
    the verdict. Both are pure functions of recorded state, so the whole
    stopping policy is unit-testable without a network or a model.
    """

    def __init__(
        self,
        *,
        max_iterations: int = 3,
        min_iterations: int = 1,
        confidence_target: float = SUFFICIENCY_THRESHOLD,
        planned_axes: int = 0,
    ) -> None:
        self.max_iterations = max(1, int(max_iterations))
        self.min_iterations = max(1, min(int(min_iterations), self.max_iterations))
        self.confidence_target = float(confidence_target)
        self.planned_axes = max(0, int(planned_axes))
        self.history: List[PassMetrics] = []
        self._seen_claims: Set[str] = set()
        self._seen_domains: Set[str] = set()
        self._seen_documents: Set[str] = set()
        self._searched_queries: Set[str] = set()
        self._stalls = 0

    # -- query memory: stops the loop re-running searches it already ran ----

    def register_queries(self, queries: Sequence[str]) -> None:
        for q in queries or ():
            key = _query_key(q)
            if key:
                self._searched_queries.add(key)

    def novel_queries(self, queries: Sequence[str], limit: int = 5) -> List[str]:
        """Filter follow-up queries down to ones not already searched.

        The critic used to emit `improved_queries` with no memory of previous
        passes, so a stubborn gap produced the same three searches every
        iteration — full cost, zero new evidence, and the cache made it
        instant enough that nobody noticed.
        """
        out: List[str] = []
        for q in queries or ():
            text = str(q or "").strip()
            key = _query_key(text)
            if not key or key in self._searched_queries:
                continue
            if any(_similar_query(key, _query_key(existing)) for existing in out):
                continue
            out.append(text)
            if len(out) >= max(1, limit):
                break
        return out

    # -- measurement -------------------------------------------------------

    def observe(
        self,
        iteration: int,
        facts: Sequence[Dict[str, Any]],
        confidence: ConfidenceReport | float,
    ) -> PassMetrics:
        score = (
            confidence.overall
            if isinstance(confidence, ConfidenceReport)
            else float(confidence or 0.0)
        )
        claims: Set[str] = set()
        domains: Set[str] = set()
        documents: Set[str] = set()
        axes: Set[str] = set()
        verified = 0
        for fact in facts or []:
            if not isinstance(fact, dict):
                continue
            claim = str(fact.get("claim", "") or "").strip().lower()
            if not claim:
                continue
            claims.add(claim)
            if fact.get("verified") is True:
                verified += 1
            url = str(fact.get("source", "") or "")
            if url:
                domains.add(extract_domain(url))
                documents.add(canonical_url(url))
            axis = str(fact.get("sub_question", "") or "").strip()
            if axis:
                axes.add(axis)

        new_claims = len(claims - self._seen_claims)
        new_domains = len(domains - self._seen_domains)
        new_documents = len(documents - self._seen_documents)
        previous = self.history[-1] if self.history else None
        new_verified = max(0, verified - (previous.verified_claims if previous else 0))
        gain = round(score - (previous.confidence if previous else 0.0), 4)

        metrics = PassMetrics(
            iteration=int(iteration),
            total_claims=len(claims),
            verified_claims=verified,
            domains=len(domains),
            documents=len(documents),
            axes_covered=len(axes),
            confidence=score,
            new_claims=new_claims,
            new_verified=new_verified,
            new_domains=new_domains,
            new_documents=new_documents,
            confidence_gain=gain,
        )
        metrics.marginal_gain = self._marginal_gain(metrics)

        self._seen_claims |= claims
        self._seen_domains |= domains
        self._seen_documents |= documents
        self.history.append(metrics)

        if metrics.marginal_gain < STALL_GAIN:
            self._stalls += 1
        else:
            self._stalls = 0
        return metrics

    @staticmethod
    def _marginal_gain(m: PassMetrics) -> float:
        """Composite information gain of one pass, in [0, 1].

        Weighted so that *new verified evidence from a new domain* dominates:
        twenty more claims restating what a single site already said is close
        to worthless, and the score reflects that.
        """
        verified_term = min(1.0, m.new_verified / 4.0) * 0.40
        domain_term = min(1.0, m.new_domains / 3.0) * 0.25
        claim_term = min(1.0, m.new_claims / 8.0) * 0.15
        confidence_term = min(1.0, max(0.0, m.confidence_gain) / 0.10) * 0.20
        return round(verified_term + domain_term + claim_term + confidence_term, 4)

    # -- decision ----------------------------------------------------------

    def decide(
        self,
        *,
        iteration: int,
        confidence: ConfidenceReport | float,
        critic_sufficient: bool = False,
        budget_exhausted: bool = False,
        can_afford_next_pass: bool = True,
        open_gaps: Sequence[str] = (),
        followups: Sequence[str] = (),
    ) -> StopDecision:
        """Decide whether the mission continues after `iteration`.

        Order matters: hard limits first (they are not negotiable), then
        quality-satisfied, then diminishing returns, then keep going. Each
        branch names its code so the trace and the UI can show *why* a run
        ended, which was previously unknowable.
        """
        score = (
            confidence.overall
            if isinstance(confidence, ConfidenceReport)
            else float(confidence or 0.0)
        )
        gaps = [str(g) for g in (open_gaps or ()) if g]
        novel = self.novel_queries(followups)
        latest = self.history[-1] if self.history else None
        gain = latest.marginal_gain if latest else 0.0

        if iteration >= self.max_iterations:
            return StopDecision(
                True,
                f"Maximum depth reached ({self.max_iterations} passes).",
                "max_depth", gain, gaps, novel,
            )
        if budget_exhausted or not can_afford_next_pass:
            return StopDecision(
                True,
                "Research budget exhausted; finalizing with the evidence gathered.",
                "budget", gain, gaps, novel,
            )

        coverage_complete = (
            self.planned_axes == 0
            or (latest is not None and latest.axes_covered >= self.planned_axes)
        )

        if iteration >= self.min_iterations and score >= self.confidence_target and coverage_complete and not gaps:
            return StopDecision(
                True,
                f"Confidence {score:.2f} at or above target {self.confidence_target:.2f} "
                "with every planned angle covered and no open gaps.",
                "confidence_target", gain, gaps, novel,
            )

        if iteration >= self.min_iterations and critic_sufficient and score >= self.confidence_target - 0.05:
            return StopDecision(
                True,
                f"Critic passed the evidence and measured confidence is {score:.2f}.",
                "critic_pass", gain, gaps, novel,
            )

        if not novel and gaps:
            return StopDecision(
                True,
                "Remaining gaps have no new searchable angle left "
                f"({len(gaps)} gap(s) recorded as limitations).",
                "no_novel_queries", gain, gaps, novel,
            )

        if self._stalls >= 2:
            return StopDecision(
                True,
                f"Two consecutive passes added almost nothing "
                f"(marginal gain {gain:.2f} < {STALL_GAIN}); further search is not paying for itself.",
                "diminishing_returns", gain, gaps, novel,
            )

        if iteration < self.min_iterations:
            return StopDecision(
                False, "Minimum depth not yet reached.", "min_depth", gain, gaps, novel
            )

        return StopDecision(
            False,
            f"Confidence {score:.2f} below target {self.confidence_target:.2f}"
            + (f"; {len(gaps)} open gap(s)" if gaps else "")
            + f"; last pass gain {gain:.2f}.",
            "continue", gain, gaps, novel,
        )

    # -- reporting ---------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "passes": [m.to_dict() for m in self.history],
            "total_claims": len(self._seen_claims),
            "total_domains": len(self._seen_domains),
            "total_documents": len(self._seen_documents),
            "queries_searched": len(self._searched_queries),
            "stalls": self._stalls,
        }


def _query_key(query: str) -> str:
    import re

    tokens = sorted(set(re.findall(r"[a-z0-9]{3,}", (query or "").lower())))
    return " ".join(tokens)


def _similar_query(a: str, b: str, threshold: float = 0.8) -> bool:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def coverage_gaps(
    planned: Sequence[Dict[str, Any]], facts: Sequence[Dict[str, Any]]
) -> List[str]:
    """Planned sub-questions that produced no evidence, plus unmet source floors.

    This is what makes "dynamic depth" targeted instead of just "loop again":
    the next pass knows which contract failed, not merely that confidence is
    low. Contracts carry `minimum_sources`; until now nothing ever checked it.
    """
    by_axis: Dict[str, Set[str]] = {}
    for fact in facts or []:
        if not isinstance(fact, dict):
            continue
        axis = str(fact.get("sub_question", "") or "").strip()
        url = str(fact.get("source", "") or "")
        if axis and url:
            by_axis.setdefault(axis, set()).add(canonical_url(url))

    gaps: List[str] = []
    for contract in planned or ():
        if not isinstance(contract, dict):
            continue
        question = str(contract.get("question", "") or "").strip()
        if not question:
            continue
        found = by_axis.get(question, set())
        required = max(1, int(contract.get("minimum_sources", 2) or 2))
        if not found:
            gaps.append(f"no evidence for angle: {question}")
        elif len(found) < required:
            gaps.append(
                f"angle under-sourced ({len(found)}/{required}): {question}"
            )
    return gaps
