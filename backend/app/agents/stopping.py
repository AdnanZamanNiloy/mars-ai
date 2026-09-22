"""Planned-coverage gaps for dynamic research depth (vision Feature 11).

`coverage_gaps` names the planned sub-questions that produced no evidence,
plus any contract whose `minimum_sources` floor went unmet, so the next pass
knows which contract failed rather than merely that confidence is low. The
critic consumes it to target expansion. Deterministic — the inputs are counts
the pipeline already has, never a model call.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set

from app.agents.evidence_utils import canonical_url


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
