"""Central investigation-intelligence allocator.

Why this module exists
----------------------
By the time the adaptive loop reaches an expansion pass the pipeline already
DETECTS several kinds of evidence deficiency, and each one owns a channel that
can seek the missing evidence:

  * single-source important claims  -> corroboration channel
    (``evidence_grade`` + ``sources.build_corroboration_query``)
  * contradictions / disagreements  -> counter-evidence channel
    (``workflow._counter_evidence_queries`` / ``contradiction.contradiction_followups``)
  * primary-source-thin dimensions  -> primary-source channel
    (``evidence_completion.primary_source_followups``)

What was missing is an ALLOCATOR: the query set for a pass was a fixed-order
concatenation of those channels, so a peripheral corroboration target could
displace a central contradicted claim simply because of channel order. With a
bounded per-pass query cap, that ordering IS the research-budget policy.

This module ranks every candidate investigation across every channel by a single
deterministic expected-value score and returns the top-K within the pass budget.
It adds NO new detectors, NO new LLM calls and NO new evidence/depth logic — it
is a thin, pure ranking layer over the existing candidate generators and
signals (impact ranking, evidence grades, depth dimensions, investigation
state). A candidate the investigation state records as exhausted or
corroborated has ZERO expected gain and can never be selected.

Scoring model (all weights are named constants below)
-----------------------------------------------------
    score = expected_gain * ( W_IMPACT * impact
                              + W_UNCERTAINTY * uncertainty
                              + W_GAP * gap )

  impact       quantitative / summary / corroboration-need claims carry the
               most weight (reuses ``evidence_completion.claim_impact``).
  uncertainty  single-source, unresolved contradiction, low grade, staleness.
  gap          needs_corroboration, primary-thin / uncovered dimension.
  expected_gain novelty: full for an un-attempted (``open``) target, reduced
               for ``attempted`` (scaled by attempts remaining), ZERO for
               ``exhausted`` / ``corroborated``.

Ties are broken deterministically by (score desc, kind, normalized target text,
query text), so identical inputs always produce an identical ranking.

Everything is total and fail-safe: a failure in any signal source skips that
candidate (logged) and never raises. There is NO module-level mutable state.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.core.logging import get_logger

logger = get_logger(__name__)

# --- Candidate kinds (one per EXISTING channel) -----------------------------
KIND_CORROBORATION = "corroboration"
KIND_PRIMARY_SOURCE = "primary_source"
KIND_COUNTER_EVIDENCE = "counter_evidence"
KIND_CONTRADICTION_RESOLUTION = "contradiction_resolution"

# Deterministic kind ordering used as the first tie-break. Declared as a tuple
# (not a set) so the order is stable and documented.
KIND_ORDER: Tuple[str, ...] = (
    KIND_CONTRADICTION_RESOLUTION,
    KIND_COUNTER_EVIDENCE,
    KIND_CORROBORATION,
    KIND_PRIMARY_SOURCE,
)

# --- Score weights (the allocation policy, in the open) ---------------------
W_IMPACT = 1.0
W_UNCERTAINTY = 0.6
W_GAP = 0.6

# Raw ``claim_impact`` scores run roughly 0..10. This normalizes them onto the
# same 0..1 axis as uncertainty/gap so the weights are comparable and the
# documented policy is meaningful.
IMPACT_NORMALIZER = 8.0

# Uncertainty contributions.
UNCERTAINTY_SINGLE_SOURCE = 1.0
UNCERTAINTY_UNRESOLVED_CONTRADICTION = 1.0
UNCERTAINTY_LOW_GRADE = 0.7         # grade C or D
UNCERTAINTY_STALE = 0.4
UNCERTAINTY_THIN_DIMENSION = 0.5

# Gap contributions.
GAP_NEEDS_CORROBORATION = 1.0
GAP_PRIMARY_THIN = 0.8
GAP_UNRESOLVED_CONTRADICTION = 0.5

# Expected-gain (novelty) multipliers.
EXPECTED_GAIN_OPEN = 1.0
EXPECTED_GAIN_ATTEMPTED_MIN = 0.4   # floor for an attempted target with budget left

# A candidate-scored float is rounded to this many decimals before ranking so
# floating-point noise can never reorder two otherwise-equal candidates.
SCORE_PRECISION = 4


def _normalize(text: Any) -> str:
    """Normalized text key, reusing the planner's canonical formatter.

    Falls back to a local whitespace/lowercase formatter if the planner import
    fails, so keying can never raise into the loop (AGENTS.md 4.4).
    """
    try:
        from app.agents.planner import normalize_text

        return normalize_text(str(text or ""))
    except Exception as exc:
        logger.warning("investigation_planner_normalize_failed", error=str(exc), exc_info=exc)
        return " ".join(str(text or "").lower().split())


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _impact_score(record: Dict[str, Any]) -> float:
    """Normalized 0..1 impact from the existing completion-impact policy."""
    raw = _as_int(record.get("impact", 0))
    if raw <= 0:
        try:
            from app.core.evidence_completion import claim_impact

            raw = _as_int(claim_impact(record))
        except Exception as exc:
            logger.warning("investigation_planner_impact_failed", error=str(exc), exc_info=exc)
            raw = 0
    return max(0.0, min(1.0, raw / IMPACT_NORMALIZER))


def _uncertainty_for_record(record: Dict[str, Any]) -> float:
    """Evidence-uncertainty score from the graded record's own signals."""
    score = 0.0
    if record.get("needs_corroboration"):
        score += UNCERTAINTY_SINGLE_SOURCE
    if _as_int(record.get("contradiction_count", 0)) > 0:
        score += UNCERTAINTY_UNRESOLVED_CONTRADICTION
    if str(record.get("grade", "") or "").upper() in ("C", "D"):
        score += UNCERTAINTY_LOW_GRADE
    if record.get("is_stale"):
        score += UNCERTAINTY_STALE
    return score


def _expected_gain(entry: Optional[Dict[str, Any]]) -> float:
    """Novelty multiplier for a claim target from its investigation-state entry.

    ``open`` (or no entry at all) is full gain; ``attempted`` is scaled by the
    fraction of its attempt budget still unspent (floored so it stays
    selectable but ranks below an untouched gap); ``exhausted`` / ``corroborated``
    is ZERO — a known dead end or a finished gap gets no budget.
    """
    if not isinstance(entry, dict):
        return EXPECTED_GAIN_OPEN
    status = str(entry.get("status", "") or "")
    if status in ("exhausted", "corroborated"):
        return 0.0
    if status == "attempted":
        max_attempts = max(1, _as_int(entry.get("max_attempts", 1), 1))
        attempts = max(0, _as_int(entry.get("attempts", 0)))
        remaining = max(0, max_attempts - attempts)
        fraction = remaining / max_attempts
        return max(EXPECTED_GAIN_ATTEMPTED_MIN, min(1.0, fraction))
    return EXPECTED_GAIN_OPEN


def _final_score(impact: float, uncertainty: float, gap: float, gain: float) -> float:
    base = W_IMPACT * impact + W_UNCERTAINTY * uncertainty + W_GAP * gap
    return round(base * gain, SCORE_PRECISION)


def _make_candidate(
    *,
    kind: str,
    target: str,
    query: str,
    impact: float,
    uncertainty: float,
    gap: float,
    gain: float,
    reason: str,
) -> Dict[str, Any]:
    score = _final_score(impact, uncertainty, gap, gain)
    return {
        "kind": kind,
        "target": target,
        "query": query,
        "impact": round(impact, SCORE_PRECISION),
        "uncertainty": round(uncertainty, SCORE_PRECISION),
        "gap": round(gap, SCORE_PRECISION),
        "expected_gain": round(gain, SCORE_PRECISION),
        "score": score,
        "reason": reason,
    }


def _sort_key(candidate: Dict[str, Any]) -> Tuple[Any, ...]:
    """Deterministic ranking key: score desc, then kind, target, query."""
    kind = str(candidate.get("kind", "") or "")
    try:
        kind_rank = KIND_ORDER.index(kind)
    except ValueError:
        kind_rank = len(KIND_ORDER)
    return (
        -_as_float(candidate.get("score", 0.0)),
        kind_rank,
        kind,
        _normalize(candidate.get("target", "")),
        _normalize(candidate.get("query", "")),
    )


# ---------------------------------------------------------------------------
# Candidate generation — one function per existing channel. Each is total: a
# failure logs and yields [], never raising.
# ---------------------------------------------------------------------------

def _investigation_entries(
    state: Dict[str, Any], max_attempts: int
) -> Dict[str, Dict[str, Any]]:
    try:
        from app.core.investigation_state import sanitize_investigation_state

        return sanitize_investigation_state(
            state.get("investigation_state"), max_attempts=max_attempts
        )
    except Exception as exc:
        logger.warning("investigation_planner_state_failed", error=str(exc), exc_info=exc)
        return {}


def _summary_claims(state: Dict[str, Any]) -> List[str]:
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
    return texts[:12]


def _corroboration_candidates(
    state: Dict[str, Any],
    entries: Dict[str, Dict[str, Any]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    """Claim-specific independent-publisher queries for single-source claims.

    Reuses the existing completion ranking (impact) and the existing
    ``build_corroboration_query`` builder — the same query the corroboration
    channel already generates, so nothing about retrieval changes.
    """
    facts = [f for f in state.get("facts", []) or [] if isinstance(f, dict)]
    if not facts:
        return []
    try:
        from app.core.evidence_completion import rank_completion_targets
        from app.core.evidence_grade import registrable_domain
        from app.agents.sources import build_corroboration_query
        from app.graph.workflow import _claim_terms

        ranked = rank_completion_targets(
            facts,
            state.get("contradictions") or [],
            summary_claims=_summary_claims(state),
            limit=max(1, limit),
        )
    except Exception as exc:
        logger.warning("investigation_planner_corroboration_failed", error=str(exc), exc_info=exc)
        return []

    out: List[Dict[str, Any]] = []
    for record in ranked:
        if not isinstance(record, dict):
            continue
        claim = str(record.get("claim", "") or "").strip()
        if not claim:
            continue
        key = _normalize(claim)
        entry = entries.get(key)
        gain = _expected_gain(entry)
        attempts = _as_int(entry.get("attempts", 0)) if entry else 0
        domain = registrable_domain(
            str(record.get("domain", "") or record.get("source", "") or "")
        )
        try:
            terms = _claim_terms(claim)
            if not terms:
                continue
            query = build_corroboration_query(
                terms,
                exclude_domain=domain,
                quantitative=bool(record.get("has_numbers")),
                attempt=attempts,
            )
        except Exception as exc:
            logger.warning("investigation_planner_corroboration_query_failed", error=str(exc), exc_info=exc)
            continue
        if not query:
            continue
        out.append(
            _make_candidate(
                kind=KIND_CORROBORATION,
                target=claim,
                query=query,
                impact=_impact_score(record),
                uncertainty=_uncertainty_for_record(record),
                gap=GAP_NEEDS_CORROBORATION if record.get("needs_corroboration") else 0.0,
                gain=gain,
                reason=(
                    "single-source important claim needs an independent publisher"
                    + ("; previously attempted" if attempts else "")
                ),
            )
        )
    return out


def _primary_source_candidates(
    state: Dict[str, Any], *, limit: int
) -> List[Dict[str, Any]]:
    """Targeted primary-source queries for primary-thin / uncovered dimensions.

    Reuses ``dimension_primary_share`` + the authoritative query builder, i.e.
    the exact generator ``evidence_completion.primary_source_followups`` uses.
    Dimensions are not claims, so investigation state does not apply them.
    """
    facts = [f for f in state.get("facts", []) or [] if isinstance(f, dict)]
    if not facts:
        return []
    try:
        from app.core.evidence_completion import (
            PRIMARY_THIN_THRESHOLD,
            dimension_primary_share,
        )
        from app.agents.sources import build_dimension_primary_query

        shares = dimension_primary_share(facts)
    except Exception as exc:
        logger.warning("investigation_planner_primary_failed", error=str(exc), exc_info=exc)
        return []

    by_dim: Dict[str, Dict[str, Any]] = {}
    for item in state.get("sub_questions", []) or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("question", "") or "").strip()
        if text:
            by_dim.setdefault(text, item)

    # Exactly the thinness signal `primary_source_followups` uses: a dimension
    # is a target only when it HAS facts but its primary share is at/below the
    # threshold. A dimension with no researched facts is not detected here (the
    # existing primary channel does not detect it either) — adding it would be
    # a new detector, which this allocator must not introduce.
    thin: List[Tuple[float, str]] = [
        (share, dim)
        for dim, share in shares.items()
        if share <= PRIMARY_THIN_THRESHOLD and dim != "__unattributed__" and dim in by_dim
    ]
    thin.sort(key=lambda pair: (pair[0], pair[1]))

    attempt = max(0, _as_int(state.get("iteration", 0)))
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for share, dim in thin:
        if dim in seen:
            continue
        seen.add(dim)
        item = by_dim.get(dim)
        if not isinstance(item, dict):
            continue
        try:
            query = build_dimension_primary_query(
                str(item.get("question", "") or ""),
                str(item.get("search_type", "") or ""),
                str(item.get("domain", "") or ""),
                attempt=attempt,
            )
        except Exception as exc:
            logger.warning("investigation_planner_primary_query_failed", error=str(exc), exc_info=exc)
            continue
        if not query:
            continue
        out.append(
            _make_candidate(
                kind=KIND_PRIMARY_SOURCE,
                target=dim,
                query=query,
                impact=max(0.2, min(1.0, 1.0 - share)),
                uncertainty=UNCERTAINTY_THIN_DIMENSION,
                gap=GAP_PRIMARY_THIN,
                gain=EXPECTED_GAIN_OPEN,
                reason="planned dimension is thin on primary/official publishers",
            )
        )
        if len(out) >= max(1, limit):
            break
    return out


def _counter_evidence_candidates(
    state: Dict[str, Any],
    entries: Dict[str, Dict[str, Any]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    """Disagreement-seeking queries for contradicted claims.

    Uses the existing counter-evidence channel shape (the same query text
    ``workflow._counter_evidence_queries`` emits) and the graded records for
    impact / uncertainty. Claim-targeted, so investigation state applies.
    """
    facts = [f for f in state.get("facts", []) or [] if isinstance(f, dict)]
    if not facts:
        return []
    try:
        from app.core.evidence_grade import grade_facts

        graded = grade_facts(facts, contradictions=state.get("contradictions") or [])
    except Exception as exc:
        logger.warning("investigation_planner_counter_failed", error=str(exc), exc_info=exc)
        return []

    out: List[Dict[str, Any]] = []
    for g in graded:
        ev = g.get("evidence") if isinstance(g, dict) else None
        if not isinstance(ev, dict):
            continue
        if _as_int(ev.get("contradiction_count", 0)) <= 0:
            continue
        claim = str(ev.get("claim", "") or "").strip()
        if not claim:
            continue
        key = _normalize(claim)
        entry = entries.get(key)
        gain = _expected_gain(entry)
        if gain <= 0:
            continue
        query = f"{claim[:140]} conflicting evidence OR disagreement"
        out.append(
            _make_candidate(
                kind=KIND_COUNTER_EVIDENCE,
                target=claim,
                query=query,
                impact=_impact_score(ev),
                uncertainty=_uncertainty_for_record(ev),
                gap=GAP_UNRESOLVED_CONTRADICTION,
                gain=gain,
                reason="claim is contradicted; seek the strongest opposing evidence",
            )
        )
        if len(out) >= max(1, limit):
            break
    return out


def _contradiction_resolution_candidates(
    state: Dict[str, Any],
    entries: Dict[str, Dict[str, Any]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    """Resolution queries for unresolved contradictions (existing generator).

    Reuses ``contradiction.contradiction_followups`` — the same channel the
    critic already feeds into ``improved_queries`` — one contradiction at a
    time so each query keeps its originating claim as its target.
    """
    contradictions = [
        c
        for c in (state.get("contradictions") or [])
        if isinstance(c, dict) and not c.get("resolved") and not c.get("intra_source")
    ]
    if not contradictions:
        return []
    try:
        from app.agents.contradiction import contradiction_followups
    except Exception as exc:
        logger.warning("investigation_planner_resolution_failed", error=str(exc), exc_info=exc)
        return []

    ordered = sorted(contradictions, key=lambda c: -_as_float(c.get("severity", 0.0)))
    out: List[Dict[str, Any]] = []
    for c in ordered:
        claim = str(c.get("claim_a", "") or "").strip()
        if not claim:
            continue
        key = _normalize(claim)
        entry = entries.get(key)
        gain = _expected_gain(entry)
        if gain <= 0:
            continue
        try:
            queries = contradiction_followups([c], limit=1)
        except Exception as exc:
            logger.warning("investigation_planner_resolution_query_failed", error=str(exc), exc_info=exc)
            continue
        if not queries:
            continue
        severity = max(0.0, min(1.0, _as_float(c.get("severity", 0.0))))
        out.append(
            _make_candidate(
                kind=KIND_CONTRADICTION_RESOLUTION,
                target=claim,
                query=str(queries[0]),
                impact=max(0.3, severity),
                uncertainty=UNCERTAINTY_UNRESOLVED_CONTRADICTION,
                gap=GAP_UNRESOLVED_CONTRADICTION,
                gain=gain,
                reason="unresolved contradiction; seek the methodology that explains it",
            )
        )
        if len(out) >= max(1, limit):
            break
    return out


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _executed_query_keys(state: Dict[str, Any]) -> Set[str]:
    """Normalized query texts this run has already issued (never re-issue)."""
    out: Set[str] = set()
    for key in ("executed_queries", "coverage_searched"):
        for q in state.get(key) or []:
            norm = _normalize(q)
            if norm:
                out.add(norm)
    return out


def select_investigations(
    state: Any,
    budget_cap: int,
    *,
    max_attempts: int = 2,
) -> Dict[str, Any]:
    """Rank every candidate investigation and select the top-K within budget.

    Returns ``{"selected": [...], "ranked": [...], "summary": {...}}`` where
    each entry carries ``kind, target, query, score, reason`` (plus the
    component signals for explainability). ``ranked`` is the full deterministic
    ordering; ``selected`` is the deduped, executed-query-filtered top-K.

    Deterministic and total: empty/garbage ``state`` yields an empty selection
    and never raises — the caller then falls back to the historical channel
    behavior (AGENTS.md 4.7).
    """
    empty: Dict[str, Any] = {"selected": [], "ranked": [], "summary": {}}
    if not isinstance(state, dict):
        return empty
    try:
        cap = max(0, _as_int(budget_cap, 0))
        if cap <= 0:
            return {**empty, "summary": {"budget_cap": 0, "candidates": 0, "selected": 0}}

        entries = _investigation_entries(state, max_attempts)
        generation_limit = max(cap * 4, 8)

        candidates: List[Dict[str, Any]] = []
        for generator in (
            lambda: _contradiction_resolution_candidates(
                state, entries, limit=generation_limit
            ),
            lambda: _counter_evidence_candidates(state, entries, limit=generation_limit),
            lambda: _corroboration_candidates(state, entries, limit=generation_limit),
            lambda: _primary_source_candidates(state, limit=generation_limit),
        ):
            try:
                candidates.extend(generator() or [])
            except Exception as exc:  # a broken channel skips, never raises
                logger.warning("investigation_planner_channel_failed", error=str(exc), exc_info=exc)

        candidates.sort(key=_sort_key)
        # A zero-gain candidate must never be selectable, regardless of channel.
        with_gain = [c for c in candidates if _as_float(c.get("expected_gain", 0.0)) > 0]

        executed = _executed_query_keys(state)
        selected: List[Dict[str, Any]] = []
        seen_queries: Set[str] = set()
        for c in with_gain:
            query_key = _normalize(c.get("query", ""))
            if not query_key or query_key in seen_queries or query_key in executed:
                continue
            seen_queries.add(query_key)
            selected.append(c)
            if len(selected) >= cap:
                break

        counts: Dict[str, int] = {}
        for c in candidates:
            counts[str(c.get("kind", ""))] = counts.get(str(c.get("kind", "")), 0) + 1
        summary = {
            "budget_cap": cap,
            "candidates": len(candidates),
            "selectable": len(with_gain),
            "selected": len(selected),
            "excluded_no_gain": len(candidates) - len(with_gain),
            "excluded_executed": sum(
                1
                for c in with_gain
                if _normalize(c.get("query", "")) in executed
            ),
            "by_kind": counts,
        }
        try:
            logger.info(
                "investigation_selection",
                budget_cap=cap,
                candidates=len(candidates),
                selected=len(selected),
                kinds=counts,
            )
        except Exception:  # logging must never break selection
            pass
        return {"selected": selected, "ranked": candidates, "summary": summary}
    except Exception as exc:  # allocator failure must never break the run
        logger.warning("investigation_selection_failed", error=str(exc), exc_info=exc)
        return empty


def explain(selected: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compact, JSON-friendly explanation rows for a list of candidates."""
    out: List[Dict[str, Any]] = []
    for c in selected or []:
        if not isinstance(c, dict):
            continue
        out.append(
            {
                "kind": str(c.get("kind", "")),
                "target": str(c.get("target", "")),
                "query": str(c.get("query", "")),
                "score": _as_float(c.get("score", 0.0)),
                "reason": str(c.get("reason", "")),
            }
        )
    return out
