"""Dynamic Research Depth / Adaptive Expansion (Phase 2.8 + Feature 11).

Decides, after every critic iteration, whether to expand (another
planner→search→summarize pass) or finalize — based on evidence signals
(axis coverage, verified-fact sufficiency, marginal confidence gain,
novel-query availability, budget headroom, iteration ceiling) rather than
iteration count alone.

v2 (production upgrade) merges the Feature-11 dynamic-depth stopping
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
`decide_with_checks(state)` returns the decision plus the full check dict.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

DECISION = Literal["expand", "finalize"]

# Axis coverage is a gap when an axis has fewer than this many verified facts.
DEFAULT_MINIMUM_SOURCES = 2
# A single axis holding more than this fraction of all verified facts is imbalance.
AXIS_DOMINANCE_THRESHOLD = 0.60
# Fewer than this many distinct axes covered means poor spread.
MIN_AXES_COVERED = 2
# Modes that must complete at least this many passes before early stops.
MODE_MIN_ITERATIONS = {"quick": 1, "audit": 2, "redteam": 1}

# Contradiction severity at or above which a conflict blocks a confident stop
# (mirrors app.core.contradictions.SEVERE_SEVERITY; kept local to avoid a
# contradiction-engine import in the hot stopping path).
SEVERE_SEVERITY = 0.60


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
    """Verified-fact count per planned axis, attributed via the fact's source URL.

    Attribution order (the second is the fix for false "uncovered axis"
    reports): a fact is credited to the planner axis of its source URL's
    sub-question; when the URL mapping is unavailable (the search results that
    would supply it have been trimmed, or the fact was re-sourced during
    corroboration), the fact's OWN `axis` field is used. Without the fallback a
    fully-researched pool whose URL→axis map was empty reported every axis as a
    hard hole, and those false holes flowed on as "unknown" noise.
    """
    verified = _verified_facts(state)
    url_axis = _url_to_axis(state)
    counts: Dict[str, int] = {axis: 0 for axis in _planned_axes(state)}
    if not counts:
        return counts
    for fact in verified:
        url = str(fact.get("source", "")).strip()
        axis = url_axis.get(url, "")
        if axis not in counts:
            # Fall back to the fact's own axis (stamped by the summarizer from
            # its contract) before treating the fact as unattributed.
            axis = str(fact.get("axis", "") or "").strip().lower()
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


def _uncovered_axes(state: Dict[str, Any]) -> List[str]:
    """Planned axes with ZERO verified facts attributed — hard coverage holes.

    Distinct from `_axes_below_threshold` (which uses the per-contract source
    floor): an axis nobody has any evidence for is a hole in the research, and
    must block a soft stop regardless of how confident the pool looks overall.
    """
    counts = _axis_coverage(state, DEFAULT_MINIMUM_SOURCES)
    return [axis for axis, n in counts.items() if n <= 0]


def _severe_contradictions(state: Dict[str, Any]) -> int:
    """Unresolved contradictions strong enough to block a confident finish.

    A contradiction resolved by the Fix C pass (different period/scope/metric)
    is an EXPLAINED spread, not a disagreement — it must not keep driving
    expansion or blocking a stop.
    """
    total = 0
    for c in state.get("contradictions", []) or []:
        if not isinstance(c, dict):
            continue
        if c.get("resolved"):
            continue
        kind = str(c.get("kind", "") or "")
        try:
            severity = float(c.get("severity", 0.0) or 0.0)
        except (TypeError, ValueError):
            severity = 0.0
        if severity >= SEVERE_SEVERITY or kind in ("numeric", "polarity"):
            total += 1
    return total


def _needs_corroboration_count(state: Dict[str, Any]) -> int:
    """Important (quantitative/definitional) claims still resting on one
    publisher. Pure helper over the evidence spine; grading failure is neutral."""
    facts = [f for f in state.get("facts", []) or [] if isinstance(f, dict)]
    if not facts:
        return 0
    try:
        from app.core.evidence_grade import grade_facts

        graded = grade_facts(facts, contradictions=state.get("contradictions") or [])
    except Exception as exc:  # grading must never change routing
        logger.warning("depth_corroboration_grading_failed", error=str(exc), exc_info=exc)
        return 0
    count = 0
    for g in graded:
        ev = g.get("evidence") if isinstance(g, dict) else None
        if isinstance(ev, dict) and ev.get("needs_corroboration"):
            count += 1
    return count


def _summary_claims(state: Dict[str, Any]) -> List[str]:
    """Claim texts the executive summary / key findings will draw on.

    Mirrors `workflow._summary_claim_texts` (kept local to avoid importing the
    graph module into the stopping hot path). Sourced from explicit state keys
    when present; deterministic, bounded, and empty when synthesis has not run
    yet — callers then rank on the other impact signals.
    """
    texts: List[str] = []
    for key in ("summary_claims", "key_findings", "headline_claims"):
        for item in state.get(key) or []:
            if isinstance(item, dict):
                text = str(item.get("claim", "") or "").strip()
            else:
                text = str(item or "").strip()
            if text:
                texts.append(text)
    answer = str(state.get("synthesized_answer", "") or "")
    if answer:
        for raw in answer.splitlines():
            line = raw.strip().lstrip("-*# ").strip()
            if 20 <= len(line) <= 200:
                texts.append(line)
            if len(texts) >= 12:
                break
    return texts[:12]


def _high_impact_uncorroborated(state: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    """High-impact single-publisher claims, ranked, with their impact score.

    Impact is the same deterministic policy the corroboration procurement uses
    (`evidence_completion.rank_completion_targets`): quantitative claims and
    claims the executive summary / key findings uses lead. `needs_corroboration`
    alone is a broad net (a peripheral definitional remark qualifies); the
    adaptive loop should spend its remaining passes on the claims that would
    change the ANSWER. Returns [] on empty/ungradeable input — never raises, so
    grading can never change routing on failure.
    """
    facts = [f for f in state.get("facts", []) or [] if isinstance(f, dict)]
    if not facts:
        return []
    try:
        from app.core.evidence_completion import rank_completion_targets

        ranked = rank_completion_targets(
            facts,
            state.get("contradictions") or [],
            summary_claims=_summary_claims(state),
            limit=max(1, int(limit)),
        )
    except Exception as exc:  # ranking must never change routing
        logger.warning("depth_completion_ranking_failed", error=str(exc), exc_info=exc)
        return []
    out: List[Dict[str, Any]] = []
    for record in ranked:
        if not isinstance(record, dict):
            continue
        claim = str(record.get("claim", "") or "").strip()
        if not claim:
            continue
        out.append({
            "claim": claim[:200],
            "impact": int(record.get("impact", 0) or 0),
            "has_numbers": bool(record.get("has_numbers", False)),
        })
    return out


def _exhausted_claim_keys(state: Dict[str, Any]) -> set:
    """Normalized keys of claims recorded as EXHAUSTED in investigation state.

    An exhausted claim was already targeted for corroboration, its attempt
    budget is spent and it is STILL single-source — it is an acknowledged
    limitation, not an actionable gap. The depth controller must not keep
    expanding for it. Keys reuse `investigation_state.investigation_key`
    (= `planner.normalize_text`) so they join the high-impact claim list
    without a parallel key scheme.

    Total/fail-safe: a missing, None, garbage or empty state yields an empty
    set; a keying/import failure is logged and also yields an empty set — so
    routing can never be changed by a malformed state (AGENTS.md 4.4).
    """
    raw = state.get("investigation_state")
    if not raw:
        return set()
    try:
        from app.core.investigation_state import (
            STATUS_EXHAUSTED,
            investigation_key,
            sanitize_investigation_state,
        )

        inv = sanitize_investigation_state(raw)
        keys = set()
        for key, entry in inv.items():
            if not isinstance(entry, dict) or entry.get("status") != STATUS_EXHAUSTED:
                continue
            claim = str(entry.get("claim", "") or "")
            keys.add(key)
            if claim:
                # The stored key is already the normalized claim; adding the
                # re-derived key makes a hand-built state with a drifted key
                # still match the claim text.
                derived = investigation_key(claim)
                if derived:
                    keys.add(derived)
        return keys
    except Exception as exc:
        logger.warning("depth_investigation_lookup_failed", error=str(exc), exc_info=exc)
        return set()


def _norm_claim_key(claim: str) -> str:
    try:
        from app.agents.planner import normalize_text

        return normalize_text(claim)
    except Exception:
        return " ".join(str(claim or "").lower().split())


def _thin_dimensions(state: Dict[str, Any], min_facts: int = 1) -> List[str]:
    """Planned research dimensions whose evidence is too thin to finalize on.

    A dimension is thin when:
      * it is a planned axis/sub-question (from `sub_questions`) with fewer than
        `min_facts` verified facts attributed to it (source-URL → axis), OR
      * it has facts but a primary-source share at/below
        `evidence_completion.PRIMARY_THIN_THRESHOLD` (the same threshold the
        primary-source follow-up channel uses — no parallel system).

    Deterministic and total: a grading/import failure yields [] so a bug can
    never wedge the loop. Unattributed facts (no sub_question) never create a
    thin dimension; a dimension with no facts at all is already an uncovered
    axis and is reported separately.
    """
    facts = [f for f in state.get("facts", []) or [] if isinstance(f, dict)]
    if not facts:
        return []
    planned = {
        str(q.get("question", "") or "").strip()
        for q in state.get("sub_questions", []) or []
        if isinstance(q, dict) and str(q.get("question", "") or "").strip()
    }
    if not planned:
        return []
    # Dimension attribution needs the summarizer's `sub_question` stamp. When
    # NO fact carries it (hand-built states, legacy runs, resume rebuilds), the
    # input is absent — thinness is unmeasurable, so report none rather than
    # falsely flag every planned dimension (backwards-compatible rule).
    stamped = [
        f for f in facts if str(f.get("sub_question", "") or "").strip()
    ]
    if not stamped:
        return []
    thin: List[str] = []
    # (a) planned dimensions with no verified fact attributed at all.
    have: Dict[str, int] = {}
    for f in stamped:
        dim = str(f.get("sub_question", "") or "").strip()
        have[dim] = have.get(dim, 0) + 1
    for dim in sorted(planned):
        if have.get(dim, 0) < min_facts:
            thin.append(dim)
    # (b) planned dimensions that have facts but are primary-source thin.
    try:
        from app.core.evidence_completion import (
            PRIMARY_THIN_THRESHOLD,
            dimension_primary_share,
        )

        shares = dimension_primary_share(stamped)
        for dim in sorted(planned):
            if dim in thin:
                continue
            if dim in shares and shares[dim] <= PRIMARY_THIN_THRESHOLD:
                thin.append(dim)
    except Exception as exc:  # primary-share is an ADDITIONAL signal only
        logger.warning("depth_thin_dimension_share_failed", error=str(exc), exc_info=exc)
    return thin


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
    uncovered = _uncovered_axes(state)
    target = _confidence_target(state, settings)
    budget = _budget_checks(state)
    novel = _novel_followups(state)
    min_iters = _min_iterations(state)
    needs_corroboration = _needs_corroboration_count(state)
    severe_contradictions = _severe_contradictions(state)
    high_impact = _high_impact_uncorroborated(state)
    thin = _thin_dimensions(state)

    # Exhausted claims are acknowledged limitations, not actionable gaps. Only
    # the ACTIVE (non-exhausted) high-impact gaps may drive expansion or block
    # a sufficiency stop; an exhausted-only gap set does neither. Claims the
    # investigation state has no entry for, or a non-exhausted entry for,
    # remain active — an un-attempted or attempted-with-budget gap is real.
    exhausted_keys = _exhausted_claim_keys(state)
    active_high_impact = [
        h for h in high_impact if _norm_claim_key(str(h.get("claim", ""))) not in exhausted_keys
    ]
    active_uncorroborated = len(active_high_impact)

    sufficiency_met = (
        confidence >= target
        and not axes_below
    )

    # Holistic evidence-sufficiency — the Step-3 signal: confidence at target,
    # axes covered, no thin dimension, no uncorroborated important claim and no
    # unresolved severe contradiction. `sufficiency_met` keeps its historical
    # (confidence + per-axis source floor) meaning for callers that depend on
    # it; this is the stricter measure the adaptive stop reads. Exhausted gaps
    # are excluded from the corroboration term so a run whose only remaining
    # gaps are exhausted can reach a genuine sufficiency stop.
    evidence_sufficient = (
        sufficiency_met
        and not uncovered
        and not thin
        and active_uncorroborated == 0
        and severe_contradictions == 0
    )

    # Concrete, human-readable triggers — the WHY behind expand/finalize, so
    # the trace/UI can explain the decision instead of showing a bare verdict.
    # Deterministic ordering (sorted/ranked) so the reason string is stable.
    reasons: List[str] = []
    if severe_contradictions:
        reasons.append(
            f"{severe_contradictions} unresolved severe contradiction"
            + ("s" if severe_contradictions != 1 else "")
        )
    if thin:
        reasons.append(
            f"{len(thin)} thin dimension" + ("s" if len(thin) != 1 else "")
        )
    if active_high_impact:
        reasons.append(
            f"{active_uncorroborated} high-impact claim"
            + ("s" if active_uncorroborated != 1 else "")
            + " uncorroborated"
        )
    if uncovered:
        reasons.append(
            f"{len(uncovered)} uncovered planned axe"
            + ("s" if len(uncovered) != 1 else "")
        )

    checks = {
        "sufficiency_met": sufficiency_met,
        "evidence_sufficient": evidence_sufficient,
        "sufficiency_stop": bool(critique.get("is_sufficient", False)) or sufficiency_met,
        "critic_sufficient": bool(critique.get("is_sufficient", False)),
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
        # Evidence-first stopping signals (research-loop fix):
        "uncovered_axes": uncovered,
        # `needs_corroboration_count` is the RAW count (all single-source
        # important claims) and stays for backwards compatibility. The ACTIVE
        # count excludes exhausted claims (acknowledged limitations); it is
        # what evidence_sufficient and the expansion gate read.
        "needs_corroboration_count": needs_corroboration,
        "exhausted_gap_count": len(exhausted_keys),
        "active_high_impact_uncorroborated_count": active_uncorroborated,
        "severe_contradictions": severe_contradictions,
        # Step 3 adaptive depth: impact-ranked corroboration gaps + thin
        # dimensions. `high_impact_uncorroborated` is the RAW ranked list for
        # trace/UI; `active_high_impact_uncorroborated` is what still drives
        # expansion.
        "high_impact_uncorroborated": high_impact,
        "high_impact_uncorroborated_count": len(high_impact),
        "active_high_impact_uncorroborated": active_high_impact,
        "thin_dimensions": thin,
        "counter_evidence_attempted": bool(state.get("counter_evidence_attempted", False)),
        # Feature-11 signals now live:
        "confidence_target": target,
        "novel_followups": novel,
        "no_novel_queries": bool(improved) and not novel,
        "budget": budget,
        "budget_stop": bool(budget["exhausted"] or not budget["can_afford_pass"]),
        # Explainability: the ordered trigger list behind the decision.
        "decision_reasons": reasons,
    }
    checks["decision_reason"] = "; ".join(reasons) if reasons else "evidence sufficient"
    return checks


def decide(state: Dict[str, Any], settings: Settings | None = None) -> DECISION:
    """Route decision only. For the checks behind it, use
    `decide_with_checks` — there is deliberately no module-level cache, so
    two concurrent runs can never read each other's decision."""
    decision, _ = decide_with_checks(state, settings)
    return decision


def decide_with_checks(
    state: Dict[str, Any], settings: Settings | None = None
) -> tuple[DECISION, Dict[str, Any]]:
    """Return (decision, checks) for a single evaluation.

    The checks live only for this call. Earlier versions stashed them in a
    module-level dict, which concurrent runs clobbered and which forced
    tests to call `decide()` then read global state out of band; callers
    now get an explicit value they own.
    """
    checks = evaluate(state, settings)

    def _with_reason(decision: DECISION, reason: str) -> tuple[DECISION, Dict[str, Any]]:
        """Stamp the concrete reason for this call (checks are per-call)."""
        checks["decision"] = decision
        checks["decision_reason"] = reason
        return decision, checks

    # ------------------------------------------------------------------
    # Hard walls are ABSOLUTE — nothing below can preempt them. Budget and the
    # iteration/depth ceiling are the anti-infinite-loop guarantee. The
    # ceiling is checked BEFORE every evidence-completeness block so an
    # uncovered axis, an uncorroborated claim or a severe contradiction can
    # only ever trigger expansion while iterations remain — never past the
    # ceiling. `hard_wall_reached` (used by route_after_critic) mirrors both.
    # ------------------------------------------------------------------
    if checks["budget_stop"]:
        # Budget is a hard wall: never expand into a pass we cannot pay for.
        return _with_reason(
            "finalize",
            "hard wall: research budget exhausted (dollars/tokens/calls/time)",
        )
    if checks["ceiling_reached"]:
        # The iteration/depth ceiling is a hard wall alongside budget: at the
        # limit the run finalizes even if gaps remain (they become limitations).
        reason = (
            "hard wall: iteration/depth ceiling reached; remaining gaps "
            "recorded as limitations"
        )
        if checks["decision_reasons"]:
            reason += f" ({checks['decision_reason']})"
        return _with_reason("finalize", reason)

    # Mode demands a minimum depth (audit re-scopes even a sufficient-looking
    # pass 1): only the hard walls above may preempt this.
    if checks["min_iterations_not_reached"]:
        return _with_reason(
            "expand",
            f"mode minimum depth not reached (iteration < {_min_iterations(state)})",
        )

    # ------------------------------------------------------------------
    # Evidence-completeness hard-blocks. These preempt every SOFT stop
    # (sufficiency, marginal gain, no-novel-queries): a run that still has an
    # unsourced planned angle, an uncorroborated important claim, a severe open
    # contradiction, or a thinly-evidenced planned dimension must not finalize
    # while a useful pass can still run.
    # ------------------------------------------------------------------
    if checks["uncovered_axes"]:
        # A planned angle with zero verified facts is a hole, not a rounding
        # error. Expanding is the only way to fill it.
        return _with_reason(
            "expand",
            f"{len(checks['uncovered_axes'])} uncovered planned axe"
            + ("s" if len(checks["uncovered_axes"]) != 1 else "")
            + ": " + ", ".join(checks["uncovered_axes"][:3]),
        )

    # Only ACTIVE (non-exhausted) corroboration gaps block a stop or force a
    # pass. An exhausted claim is an acknowledged limitation — re-finding the
    # same dead end is spend, not research; the workflow already excludes it
    # from the next corroboration pass.
    gaps_present = (
        checks["active_high_impact_uncorroborated_count"] > 0
        or checks["severe_contradictions"] > 0
        or bool(checks["thin_dimensions"])
    )
    if gaps_present:
        if checks["novel_followups"]:
            return _with_reason("expand", checks["decision_reason"])
        # Nothing new left to search: record as limitations rather than burn a
        # pass re-finding the same pages (prevents an unbounded loop).
        reason = checks["decision_reason"] or "evidence gaps remain"
        return _with_reason(
            "finalize",
            f"evidence gaps remain but nothing novel is left to search ({reason}); "
            "recorded as limitations",
        )

    # A critic that explicitly said "insufficient" and proposed actionable new
    # queries forces a pass — the model verdict is not waivable by a measured
    # sufficiency that ignores what the critic saw.
    if not checks["critic_sufficient"] and checks["novel_followups"]:
        return _with_reason(
            "expand",
            "critic reported insufficient evidence and proposed novel follow-up queries",
        )

    # ------------------------------------------------------------------
    # Soft stops. Reached only when the evidence base is complete.
    # ------------------------------------------------------------------
    if checks["sufficiency_stop"]:
        reason = "evidence sufficient: confidence at target and planned axes covered"
        if checks["exhausted_gap_count"] and not checks["active_high_impact_uncorroborated_count"]:
            reason += (
                f"; {checks['exhausted_gap_count']} exhausted gap"
                + ("s" if checks["exhausted_gap_count"] != 1 else "")
                + " recorded as acknowledged limitations"
            )
        return _with_reason("finalize", reason)
    if checks["marginal_gain_stop"]:
        return _with_reason(
            "finalize",
            "no meaningful marginal confidence gain for two consecutive iterations",
        )
    if checks["no_novel_queries"]:
        # Every proposed follow-up duplicates a search we already ran —
        # expanding would burn a pass to re-find the same pages.
        return _with_reason(
            "finalize",
            "no novel queries left: every follow-up duplicates an already-run search",
        )

    # Expansion trigger: critic sees a specific gap AND axis coverage is poor.
    if checks["coverage_gap"]:
        return _with_reason(
            "expand",
            "critic-reported coverage gap: axis spread/imbalance or a below-floor axis",
        )

    # Default: trust the critic's loop decision (it returned insufficient
    # with improved_queries even if axis data couldn't confirm a gap).
    if state.get("critique", {}).get("improved_queries"):
        return _with_reason(
            "expand",
            "critic returned insufficient with actionable follow-up queries",
        )

    return _with_reason("finalize", "no expansion trigger and no gaps detected")


def hard_wall_reached(state: Dict[str, Any], settings: Settings | None = None) -> bool:
    """True when only a hard wall (budget/time or iteration ceiling) can stop.

    Public so `route_after_critic` can tell an evidence gate it cannot act on
    (hard wall) from one it should honour — the ordering bug that let a soft
    depth-controller stop defeat the evidence gate.
    """
    checks = evaluate(state, settings)
    return bool(checks["budget_stop"] or checks["ceiling_reached"])


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
    # Evidence-driven early stop: the run hit a hard wall while measurable
    # evidence gaps remained. Name them so the Limitations section reflects why
    # the report is not deeper, not just that it stopped.
    if (checks["ceiling_reached"] or checks["budget_stop"]) and checks["decision_reasons"]:
        return (
            "Stopped at the research depth limit with outstanding evidence gaps "
            f"({checks['decision_reason']}); they are recorded below as limitations."
        )
    return None
