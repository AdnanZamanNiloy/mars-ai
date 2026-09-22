"""Per-claim investigation state — closing the adaptive-investigation loop.

The failure this module exists for
----------------------------------
The pipeline already DETECTS and TARGETS evidence gaps:

  * `evidence_completion.rank_completion_targets` ranks the uncorroborated
    high-impact claims and the expansion routing issues claim-specific
    follow-up queries for them (workflow._corroboration_queries), and
  * `evidence_grade.find_corroborating_sources` matches NEW-publisher results
    back to those claims when the queries return.

But nothing tracked whether a targeted attempt WORKED. A live deep run left
`needs_corroboration` stuck at 20-50 claims per query and every iteration
"still not sufficient": the same dead-end gap could be re-funded pass after
pass, and an exhausted gap could not be told apart from an un-attempted one.
Detect-and-target without outcome memory is an open loop.

This module is the missing memory. It is run-scoped and additive:

  * a claim's entry records attempts made, the (deduped) queries issued for
    it, and the outcome — still single-source, corroborated, or exhausted;
  * claims that became corroborated leave the open set;
  * claims whose attempt budget is spent and that are STILL single-source are
    marked exhausted and become an acknowledged limitation instead of being
    silently re-chased;

The state is a plain dict threaded through LangGraph state (`investigation_state`)
— there is NO module-level mutable store, so concurrent runs cannot see each
other's state and there is no cleanup path to forget (AGENTS.md 4.3). Every
function is pure, deterministic and total: a grading/lookup failure is logged
and degrades to a neutral result, never raises.

This module deliberately does not change any stopping policy — it only exposes
the investigation state the policy can consume.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from app.core.logging import get_logger

logger = get_logger(__name__)

# Attempt budget: how many targeted follow-up queries one claim may receive
# before it is acknowledged as an exhausted (still single-source) limitation.
# Mirrors the default of Settings.max_corroboration_attempts so the two budget
# notions cannot drift; callers pass the Settings value in production.
DEFAULT_MAX_ATTEMPTS = 2

# Bounded per-claim query memory. A long run can accumulate queries; this caps
# the stored list so state cannot grow without bound (AGENTS.md 5).
MAX_QUERIES_PER_CLAIM = 8

# Status vocabulary.
STATUS_OPEN = "open"                    # known gap, no targeted attempt yet
STATUS_ATTEMPTED = "attempted"          # targeted, still single-source, budget left
STATUS_EXHAUSTED = "exhausted"          # targeted, still single-source, budget spent
STATUS_CORROBORATED = "corroborated"    # a second independent publisher landed


def investigation_key(claim: str) -> str:
    """Stable, run-consistent key for a claim.

    Reuses `app.agents.planner.normalize_text` — the same normalized-claim key
    the corroboration registry already uses — so an investigation entry and a
    registry entry for the same claim can be joined without a parallel keying
    system. Empty/blank input yields "" (never a synthesized key).
    """
    try:
        from app.agents.planner import normalize_text

        return normalize_text(str(claim or ""))
    except Exception as exc:  # a keying bug must never raise into the loop
        logger.warning("investigation_key_failed", error=str(exc), exc_info=exc)
        return " ".join(str(claim or "").lower().split())


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _new_entry(claim: str, max_attempts: int) -> Dict[str, Any]:
    return {
        "claim": str(claim or "").strip(),
        "attempts": 0,
        "queries": [],
        "status": STATUS_OPEN,
        "last_outcome": "",
        "max_attempts": max(1, int(max_attempts or DEFAULT_MAX_ATTEMPTS)),
    }


def _sanitize_entry(entry: Any, key: str, max_attempts: int) -> Dict[str, Any]:
    """Coerce a possibly hand-built/legacy entry into the canonical shape."""
    if not isinstance(entry, dict):
        return _new_entry(key, max_attempts)
    queries = [
        str(q) for q in (entry.get("queries") or []) if str(q or "").strip()
    ][-MAX_QUERIES_PER_CLAIM:]
    status = str(entry.get("status", "") or STATUS_OPEN)
    if status not in (STATUS_OPEN, STATUS_ATTEMPTED, STATUS_EXHAUSTED, STATUS_CORROBORATED):
        status = STATUS_OPEN
    return {
        "claim": str(entry.get("claim", "") or key).strip(),
        "attempts": max(0, _as_int(entry.get("attempts", 0))),
        "queries": queries,
        "status": status,
        "last_outcome": str(entry.get("last_outcome", "") or ""),
        "max_attempts": max(1, _as_int(entry.get("max_attempts", max_attempts), max_attempts)),
    }


def sanitize_investigation_state(
    raw: Any, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS
) -> Dict[str, Dict[str, Any]]:
    """Normalize an arbitrary value into a well-formed investigation state.

    Total: a non-dict, a None or a dict with malformed entries yields a valid
    (possibly empty) dict, so a resumed/legacy run can never crash the loop.
    """
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, entry in raw.items():
        key_s = str(key or "").strip()
        if not key_s:
            continue
        out[key_s] = _sanitize_entry(entry, key_s, max_attempts)
    return out


def record_attempts(
    state: Any,
    targets: Sequence[Any],
    *,
    queries_by_claim: Optional[Dict[str, Sequence[str]]] = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Dict[str, Dict[str, Any]]:
    """Record that a targeted query was issued for each claim in `targets`.

    `targets` are the ranked completion-target dicts (each carrying `claim`);
    `queries_by_claim` maps an investigation key to the query texts issued for
    it this pass. Returns a NEW sanitized state dict (the input is not
    mutated). Deduplicates queries per claim and bumps `attempts` once per
    target that had at least one query issued. Total: malformed targets are
    skipped, never raised on.
    """
    out = sanitize_investigation_state(state, max_attempts=max_attempts)
    qmap = queries_by_claim if isinstance(queries_by_claim, dict) else {}
    for target in targets or []:
        if not isinstance(target, dict):
            continue
        claim = str(target.get("claim", "") or "").strip()
        if not claim:
            continue
        key = investigation_key(claim)
        if not key:
            continue
        entry = out.get(key) or _new_entry(claim, max_attempts)
        entry["claim"] = entry.get("claim") or claim
        issued = [
            str(q) for q in (qmap.get(key) or []) if str(q or "").strip()
        ]
        # Deduplicate against what this claim already issued. Query memory is
        # bounded (AGENTS.md 5) — keep the most recent MAX_QUERIES_PER_CLAIM.
        new_queries = [q for q in issued if q not in entry["queries"]]
        if new_queries:
            entry["queries"] = [*entry["queries"], *new_queries][-MAX_QUERIES_PER_CLAIM:]
            entry["attempts"] = entry["attempts"] + 1
        if entry["status"] == STATUS_OPEN and entry["attempts"] > 0:
            entry["status"] = STATUS_ATTEMPTED
        if entry["attempts"] >= entry["max_attempts"]:
            entry["status"] = STATUS_EXHAUSTED
        out[key] = entry
    return out


def reconcile_outcomes(
    state: Any,
    facts: Sequence[Any],
    *,
    needs_corroboration: Optional[Dict[str, bool]] = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Dict[str, Dict[str, Any]]:
    """Update each tracked claim's outcome from the current fact pool.

    A claim that is no longer `needs_corroboration` (a second independent
    publisher landed) is marked `corroborated` and leaves the open set. A claim
    still single-source with its budget spent is marked `exhausted`. A claim
    still single-source with budget left is `attempted` (or `open` if never
    attempted). Returns a NEW state dict.

    `needs_corroboration` is an optional precomputed map (investigation key ->
    bool) so callers that already graded the pool do not pay for grading twice;
    when absent the module grades the facts itself. A grading failure logs and
    leaves statuses unchanged (never raises).
    """
    out = sanitize_investigation_state(state, max_attempts=max_attempts)
    if not out:
        return out
    nc = needs_corroboration
    if nc is None:
        nc = _needs_corroboration_map(facts)
    if nc is None:
        return out  # grading failed: leave recorded state intact
    for key, entry in out.items():
        still_needs = nc.get(key)
        if still_needs is None:
            # The claim is not in the pool at all (dropped/rebuilt). Do not
            # invent an outcome; its recorded attempts stay as they are.
            continue
        if not still_needs:
            entry["status"] = STATUS_CORROBORATED
            entry["last_outcome"] = "corroborated"
            continue
        entry["last_outcome"] = "still_single_source"
        if entry["attempts"] >= entry["max_attempts"]:
            entry["status"] = STATUS_EXHAUSTED
        elif entry["attempts"] > 0:
            entry["status"] = STATUS_ATTEMPTED
        else:
            entry["status"] = STATUS_OPEN
    return out


def _needs_corroboration_map(facts: Sequence[Any]) -> Optional[Dict[str, bool]]:
    """Investigation key -> needs_corroboration for every fact, or None on failure.

    Grading is the same deterministic spine the rest of the pipeline uses; when
    it fails the caller leaves state unchanged rather than guessing.
    """
    pool = [f for f in (facts or []) if isinstance(f, dict) and str(f.get("claim", "")).strip()]
    if not pool:
        return {}
    try:
        from app.core.evidence_grade import grade_claim

        out: Dict[str, bool] = {}
        for fact in pool:
            claim = str(fact.get("claim", "") or "").strip()
            key = investigation_key(claim)
            if not key:
                continue
            record = grade_claim(fact)
            out[key] = bool(record.needs_corroboration)
        return out
    except Exception as exc:  # grading must never break the loop
        logger.warning("investigation_grading_failed", error=str(exc), exc_info=exc)
        return None


def exhausted_limitations(
    state: Any, *, max_attempts: int = DEFAULT_MAX_ATTEMPTS, limit: int = 5
) -> List[str]:
    """Human-readable limitations for exhausted, still-single-source claims.

    This is the point of the whole module: an exhausted gap is ACKNOWLEDGED in
    the report instead of being silently re-chased. Deterministic, bounded and
    empty-safe.
    """
    out = sanitize_investigation_state(state, max_attempts=max_attempts)
    lines: List[str] = []
    for entry in sorted(out.values(), key=lambda e: str(e.get("claim", ""))):
        if entry["status"] != STATUS_EXHAUSTED:
            continue
        claim = " ".join(str(entry.get("claim", "")).split())
        if not claim:
            continue
        if len(claim) > 140:
            claim = claim[:139].rstrip() + "…"
        lines.append(
            f"'{claim}' remains single-source after "
            f"{entry['attempts']} targeted corroboration attempt"
            + ("s" if entry["attempts"] != 1 else "")
            + "; recorded as an acknowledged limitation."
        )
        if len(lines) >= max(1, int(limit)):
            break
    return lines
