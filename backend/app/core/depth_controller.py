"""Dynamic Research Depth / Adaptive Expansion (Phase 2.8, Feature 11).

Decides, after every critic iteration, whether to expand (another
planner→search→summarize pass) or finalize — based on evidence signals
(axis coverage, verified-fact sufficiency, marginal confidence gain,
iteration ceiling) rather than iteration count alone.

This is deliberately distinct from route_after_critic's old two-condition
check (is_sufficient / ceiling).

`decide(state)` is called by workflow.route_after_critic and returns
"expand" | "finalize". `stop_reason(state)` recomputes the same signals
deterministically at report time so limitations can name the stop cause.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal

from app.core.config import Settings, get_settings

DECISION = Literal["expand", "finalize"]

# Axis coverage is a gap when an axis has fewer than this many verified facts.
DEFAULT_MINIMUM_SOURCES = 2
# A single axis holding more than this fraction of all verified facts is imbalance.
AXIS_DOMINANCE_THRESHOLD = 0.60
# Fewer than this many distinct axes covered means poor spread.
MIN_AXES_COVERED = 2


def _planned_axes(state: Dict[str, Any]) -> List[str]:
    return sorted({
        str(q.get("axis", "general"))
        for q in state.get("sub_questions", [])
        if isinstance(q, dict) and str(q.get("axis", "")).strip()
    })


def _verified_facts(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    facts = state.get("facts", [])
    if not any("verified" in f for f in facts):
        return list(facts)  # verification never ran; treat all as evidence
    return [f for f in facts if f.get("verified")]


def _url_to_axis(state: Dict[str, Any]) -> Dict[str, str]:
    """Map source URL -> axis via the search result's sub_question text and
    the planner's delegation contracts (sub_question -> axis)."""
    contract_axis: Dict[str, str] = {}
    for q in state.get("sub_questions", []):
        if isinstance(q, dict):
            question = str(q.get("question", "")).strip()
            axis = str(q.get("axis", "")).strip()
            if question and axis:
                contract_axis[question] = axis

    url_axis: Dict[str, str] = {}
    for result in state.get("search_results", []) or []:
        url = str(result.get("url", "")).strip()
        sub_q = str(result.get("sub_question", "")).strip()
        if url and sub_q in contract_axis:
            url_axis[url] = contract_axis[sub_q]
    return url_axis


def _axis_coverage(state: Dict[str, Any], minimum_sources: int) -> Dict[str, int]:
    """Verified-fact count per planned axis, attributed via the fact's source URL."""
    verified = _verified_facts(state)
    url_axis = _url_to_axis(state)
    counts: Dict[str, int] = {axis: 0 for axis in _planned_axes(state)}
    for fact in verified:
        url = str(fact.get("source", "")).strip()
        axis = url_axis.get(url, "")
        if axis in counts:
            counts[axis] += 1
    return counts


def _axes_below_threshold(state: Dict[str, Any], minimum_sources: int) -> List[str]:
    counts = _axis_coverage(state, minimum_sources)
    return [axis for axis, n in counts.items() if n < minimum_sources]


def _axes_covered(state: Dict[str, Any], minimum_sources: int) -> List[str]:
    """Planned axes that have at least `minimum_sources` verified facts."""
    counts = _axis_coverage(state, minimum_sources)
    return [axis for axis, n in counts.items() if n >= minimum_sources]


def _axis_imbalance(state: Dict[str, Any]) -> bool:
    verified = _verified_facts(state)
    if not verified:
        return False
    url_axis = _url_to_axis(state)
    counts: Dict[str, int] = {}
    for f in verified:
        axis = url_axis.get(str(f.get("source", "")).strip(), "unattributed")
        counts[axis] = counts.get(axis, 0) + 1
    total = len(verified)
    return any(n / total > AXIS_DOMINANCE_THRESHOLD for n in counts.values())


def _marginal_gain(history: List[float]) -> List[float]:
    return [history[i] - history[i - 1] for i in range(1, len(history))]


def _two_consecutive_stalls(history: List[float], min_gain: float) -> bool:
    deltas = _marginal_gain(history)
    if len(deltas) < 2:
        return False
    return deltas[-1] < min_gain and deltas[-2] < min_gain


def evaluate(state: Dict[str, Any], settings: Settings | None = None) -> Dict[str, Any]:
    """Return the decision inputs and which rules fired — pure and inspectable."""
    settings = settings or get_settings()
    critique = state.get("critique", {})
    iteration = int(state.get("iteration", 0))
    max_iterations = int(state.get("max_iterations", 3))
    max_depth = int(settings.max_research_depth or 0) or max_iterations
    ceiling = max(max_iterations, max_depth)
    confidence = float(state.get("confidence", 0.0))
    history = list(state.get("confidence_history", []))
    improved = critique.get("improved_queries") or []

    minimum_sources = DEFAULT_MINIMUM_SOURCES
    for q in state.get("sub_questions", []):
        if isinstance(q, dict) and q.get("minimum_sources"):
            minimum_sources = max(1, int(q["minimum_sources"]))
            break

    axes_below = _axes_below_threshold(state, minimum_sources)
    axes_covered = _axes_covered(state, minimum_sources)
    sufficiency_met = (
        confidence >= settings.sufficiency_threshold
        and not axes_below
    )

    checks = {
        "sufficiency_met": sufficiency_met,
        "sufficiency_stop": bool(critique.get("is_sufficient", False)) or sufficiency_met,
        "marginal_gain_stop": _two_consecutive_stalls(history, settings.min_marginal_gain),
        "ceiling_reached": iteration >= ceiling,
        # Coverage-gap detection (manual 2.8): compare COVERED axes (verified
        # facts via source attribution) against the axes the planner scoped.
        "coverage_gap": bool(improved) and (
            len(axes_covered) < MIN_AXES_COVERED
            or _axis_imbalance(state)
            or bool(axes_below)
        ),
        "axes_below_threshold": axes_below,
        "axes_covered": axes_covered,
    }
    return checks


def decide(state: Dict[str, Any], settings: Settings | None = None) -> DECISION:
    checks = evaluate(state, settings)

    # Stop conditions, in priority order.
    if checks["sufficiency_stop"]:
        return "finalize"
    if checks["marginal_gain_stop"]:
        return "finalize"
    if checks["ceiling_reached"]:
        return "finalize"

    # Expansion trigger: critic sees a specific gap AND axis coverage is poor.
    if checks["coverage_gap"]:
        return "expand"

    # Default: trust the critic's loop decision (it returned insufficient
    # with improved_queries even if axis data couldn't confirm a gap).
    if state.get("critique", {}).get("improved_queries"):
        return "expand"

    return "finalize"


def stop_reason(state: Dict[str, Any], settings: Settings | None = None) -> str | None:
    """Deterministic explanation of why the pipeline stopped early — for the
    report's Limitations section. Returns None when nothing unusual fired."""
    checks = evaluate(state, settings)
    if checks["marginal_gain_stop"] and not checks["sufficiency_stop"]:
        return (
            "Stopped early on marginal information gain: confidence improved by "
            "less than the minimum gain threshold for two consecutive iterations."
        )
    return None
