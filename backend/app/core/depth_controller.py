"""Dynamic Research Depth / Adaptive Expansion (Phase 2.8 + Feature 11).

Decides, after every critic iteration, whether to expand (another
planner→search→summarize pass) or finalize — based on evidence signals
(axis coverage, verified-fact sufficiency, marginal confidence gain,
novel-query availability, budget headroom, iteration ceiling) rather than
iteration count alone.

v2 (production upgrade) merges the Feature-11 DepthController's stopping
policy into this stateless module so the live graph benefits without a
per-mission object:

* budget stop        — the run ledger (app/core/usage.py) knows dollars,
                       tokens, calls and wall-clock; a pass that cannot be
                       paid for never starts.
* no-novel-queries   — cross-checks the critic's proposed follow-ups
                       against every query already searched (question texts,
                       variants, prior improved_queries). Re-running the
                       same searches is spend, not research.
* mode targets       — each mode carries a confidence target
                       (MODE_CONFIDENCE_TARGET); audit stops later than
                       quick by design, and the stopping rule now honors it.
* min_iterations     — a mode may demand a minimum depth before any early
                       stop fires (guards against stopping on pass 1 noise).

`decide(state)` is called by workflow.route_after_critic and returns
"expand" | "finalize". `stop_reason(state)` recomputes the same signals
deterministically at report time so limitations can name the stop cause.
`last_decision` exposes the full check dict for the trace/UI.
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
# Modes that must complete at least this many passes before early stops.
MODE_MIN_ITERATIONS = {"quick": 1, "audit": 2, "redteam": 1}

_last_decision: Dict[str, Any] = {}


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


# ---------------------------------------------------------------------------
# Query memory (Feature 11): which searches already ran
# ---------------------------------------------------------------------------

def _query_key(query: str) -> str:
    import re

    tokens = sorted(set(re.findall(r"[a-z0-9]{3,}", (query or "").lower())))
    return " ".join(tokens)


def _similar_query(a: str, b: str, threshold: float = 0.8) -> bool:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def _searched_queries(state: Dict[str, Any]) -> set:
    """Every query this run has already issued: sub-question texts that
    produced results, their variants, and prior critic follow-ups."""
    searched: set = set()
    for r in state.get("search_results", []) or []:
        if isinstance(r, dict):
            key = _query_key(str(r.get("sub_question", "")))
            if key:
                searched.add(key)
    for q in state.get("sub_questions", []) or []:
        if isinstance(q, dict):
            for text in [q.get("question", ""), *(q.get("variants", []) or [])]:
                key = _query_key(str(text or ""))
                if key:
                    searched.add(key)
    for c in state.get("coverage_searched", []) or []:
        key = _query_key(str(c or ""))
        if key:
            searched.add(key)
    return searched


def _novel_followups(state: Dict[str, Any]) -> List[str]:
    """Critic follow-ups that are NOT near-duplicates of already-run searches."""
    from app.agents.planner import normalize_text

    searched = _searched_queries(state)
    if not searched:
        return [str(q) for q in state.get("critique", {}).get("improved_queries", []) or []]
    out: List[str] = []
    for q in state.get("critique", {}).get("improved_queries", []) or []:
        text = str(q or "").strip()
        key = _query_key(text)
        if not text or not key:
            continue
        if any(_similar_query(key, seen) for seen in searched):
            continue
        if any(_similar_query(key, _query_key(other)) for other in out):
            continue
        out.append(normalize_text(text))
    return out


# ---------------------------------------------------------------------------
# Budget signals (app/core/usage.py ledger)
# ---------------------------------------------------------------------------

def _budget_checks(state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from app.core.usage import get_run_usage

        usage = get_run_usage()
    except Exception:
        usage = None
    if usage is None:
        return {"active": False, "exhausted": False, "can_afford_pass": True, "utilization": 0.0}
    pending = max(1, len(_axes_below_threshold(state, DEFAULT_MINIMUM_SOURCES)))
    return {
        "active": True,
        "exhausted": usage.exhausted,
        "can_afford_pass": usage.can_afford_pass(pending),
        "utilization": usage.budget.utilization(),
        "snapshot": usage.budget.snapshot(),
    }


def _confidence_target(state: Dict[str, Any], settings: Settings) -> float:
    """Mode-aware target: audit demands more proof than quick by design."""
    try:
        from app.agents.orchestrator import MODE_CONFIDENCE_TARGET

        mode = str(state.get("mode", "") or "standard").lower()
        if mode in MODE_CONFIDENCE_TARGET:
            return float(MODE_CONFIDENCE_TARGET[mode])
    except Exception:
        pass
    return float(settings.sufficiency_threshold)


def _min_iterations(state: Dict[str, Any]) -> int:
    mode = str(state.get("mode", "") or "standard").lower()
    return int(MODE_MIN_ITERATIONS.get(mode, 1))


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
    target = _confidence_target(state, settings)
    budget = _budget_checks(state)
    novel = _novel_followups(state)
    min_iters = _min_iterations(state)

    sufficiency_met = (
        confidence >= target
        and not axes_below
    )

    checks = {
        "sufficiency_met": sufficiency_met,
        "sufficiency_stop": bool(critique.get("is_sufficient", False)) or sufficiency_met,
        "marginal_gain_stop": _two_consecutive_stalls(history, settings.min_marginal_gain),
        "ceiling_reached": iteration >= ceiling,
        "min_iterations_not_reached": iteration < min_iters,
        # Coverage-gap detection (manual 2.8): compare COVERED axes (verified
        # facts via source attribution) against the axes the planner scoped.
        "coverage_gap": bool(improved) and (
            len(axes_covered) < MIN_AXES_COVERED
            or _axis_imbalance(state)
            or bool(axes_below)
        ),
        "axes_below_threshold": axes_below,
        "axes_covered": axes_covered,
        # Feature-11 signals now live:
        "confidence_target": target,
        "novel_followups": novel,
        "no_novel_queries": bool(improved) and not novel,
        "budget": budget,
        "budget_stop": bool(budget["exhausted"] or not budget["can_afford_pass"]),
    }
    return checks


def decide(state: Dict[str, Any], settings: Settings | None = None) -> DECISION:
    global _last_decision
    checks = evaluate(state, settings)
    _last_decision = dict(checks)

    # Stop conditions, in priority order.
    if checks["sufficiency_stop"] and not checks["min_iterations_not_reached"]:
        return "finalize"
    if checks["budget_stop"]:
        # Budget is a hard wall: never expand into a pass we cannot pay for.
        return "finalize"
    if checks["marginal_gain_stop"] and not checks["min_iterations_not_reached"]:
        return "finalize"
    if checks["ceiling_reached"]:
        return "finalize"
    if checks["no_novel_queries"]:
        # Every proposed follow-up duplicates a search we already ran —
        # expanding would burn a pass to re-find the same pages.
        return "finalize"

    # Mode demands a minimum depth (audit re-scopes even a sufficient-looking
    # pass 1): the only stops that may preempt this are the hard walls above.
    if checks["min_iterations_not_reached"]:
        return "expand"

    # Expansion trigger: critic sees a specific gap AND axis coverage is poor.
    if checks["coverage_gap"]:
        return "expand"

    # Default: trust the critic's loop decision (it returned insufficient
    # with improved_queries even if axis data couldn't confirm a gap).
    if state.get("critique", {}).get("improved_queries"):
        return "expand"

    return "finalize"


def last_decision() -> Dict[str, Any]:
    """The checks behind the most recent decide() call (trace/UI/benchmarks)."""
    return dict(_last_decision)


def stop_reason(state: Dict[str, Any], settings: Settings | None = None) -> str | None:
    """Deterministic explanation of why the pipeline stopped early — for the
    report's Limitations section. Returns None when nothing unusual fired."""
    checks = evaluate(state, settings)
    if checks["budget_stop"] and not checks["sufficiency_stop"]:
        budget = checks["budget"]
        util = float(budget.get("utilization") or 0.0)
        return (
            "Stopped early on the research budget (dollars/tokens/calls/time "
            f"utilization {util:.0%}); finalizing with the evidence gathered so far."
        )
    if checks["marginal_gain_stop"] and not checks["sufficiency_stop"]:
        return (
            "Stopped early on marginal information gain: confidence improved by "
            "less than the minimum gain threshold for two consecutive iterations."
        )
    if checks["no_novel_queries"] and not checks["sufficiency_stop"]:
        return (
            "Stopped early because every remaining question duplicated a search "
            "already run; the gaps are recorded below as limitations instead."
        )
    return None
