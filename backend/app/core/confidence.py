"""Confidence Engine v2 (Phase 2.4 + Feature 10 wiring).

Multi-signal confidence replacing the inline formula in critic_node.
Returns the overall score AND the per-signal breakdown so the frontend can
render the confidence breakdown UI.

v2 wires the three signals the v3 adapter recorded but never fed live:

* contradictions  — a PENALTY, not a weighted term: severe cross-source
                    conflicts subtract from the final score (a weighted
                    average would let strong sources mask direct lies)
* citation_support — the previous iteration's answer-support rate, blended
                    in once a synthesis exists (10% carved from citation
                    coverage — the two measure claim-level vs sentence-level
                    grounding of the same thing)
* axis_coverage   — planned-vs-covered research angles (5% carved from
                    source diversity; an unanswered axis means the number is
                    incomplete no matter how good the covered sources are)

Signals only contribute when their inputs exist, and absent inputs keep
the historical weights exactly (the WEIGHTS_FRESH pattern), so scores stay
comparable across runs and the existing test contracts hold.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List

from app.agents.evidence_utils import extract_domain, source_reliability_score
from app.core.logging import get_logger

logger = get_logger(__name__)

# Corroboration band: similar enough to be about the same thing, below the
# dedupe merge threshold (0.86) so identical claims were already collapsed.
CORROBORATION_SIMILARITY = 0.55

# Minimum verified facts a planned axis needs before it counts as covered.
AXIS_MIN_FACTS = 2

WEIGHTS: Dict[str, float] = {
    "source_quality": 0.20,
    "source_diversity": 0.15,
    "citation_coverage": 0.25,
    "claim_verification_strength": 0.15,
    "cross_source_agreement": 0.15,
    "critic_survival": 0.10,
    "freshness": 0.0,  # not yet measured — no publish dates captured
}

# Freshness takes 0.05 from citation coverage ONLY when publish dates are
# actually present; undated runs keep the exact legacy weights (WEIGHTS),
# so historical scores stay comparable.
WEIGHTS_FRESH: Dict[str, float] = {
    **WEIGHTS,
    "citation_coverage": 0.20,
    "freshness": 0.05,
}

# Contradiction penalty: severe cross-source conflicts subtract from the
# weighted sum (not a weighted term — a weighted average would let strong
# sources mask direct conflicts). Capped so a noisy pool can't zero out a
# otherwise-solid run.
CONTRADICTION_PENALTY_SEVERE = 0.06   # per severe conflict (severity >= 0.60)
CONTRADICTION_PENALTY_MODERATE = 0.03  # per moderate conflict
CONTRADICTION_PENALTY_CAP = 0.18
SEVERE_CONTRADICTION_SEVERITY = 0.60


def _freshness(
    source_dates: List[str] | None,
    facts: List[Dict[str, Any]] | None = None,
) -> tuple[float, bool]:
    """Mean recency over parseable dates. (0.0, False) when none parse —
    unknown stays unmeasured, never faked.

    Dates are taken from the FACTS actually in the pool when available (each
    fact carries `published_at`), not from the raw search-result list. The old
    behavior averaged every dated search result, including ones that never
    produced a fact, so a stale irrelevant page dragged down (or a fresh
    irrelevant page flattered) a signal that is supposed to describe the
    evidence. Falls back to `source_dates` for legacy callers.

    Uses the SAME exponential recency curve as the verifier
    (`sources.freshness_score`, per search_type), removing the prior split
    where the confidence engine's linear 730-day curve scored a source 0.0
    while the verifier scored the identical source ~0.5-0.7.
    """
    from app.agents.sources import freshness_score

    items: List[tuple[str, str]] = []
    pool = [f for f in (facts or []) if isinstance(f, dict)]
    if pool:
        for f in pool:
            published = str(f.get("published_at", "") or "")
            if published.strip():
                items.append((published, str(f.get("search_type", "default") or "default")))
    else:
        items = [(str(raw), "default") for raw in (source_dates or []) if str(raw).strip()]

    scores: List[float] = []
    for published, search_type in items:
        from app.agents.evidence_utils import parse_published_date

        if not parse_published_date(published):
            continue  # unparseable stays unmeasured
        scores.append(freshness_score(published, search_type))
    if not scores:
        return 0.0, False
    return round(sum(scores) / len(scores), 3), True


def _safe_conf(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any, default: int = 1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _similarity(a: str, b: str) -> float:
    a_norm = " ".join(sorted(a.lower().split()))
    b_norm = " ".join(sorted(b.lower().split()))
    if not a_norm or not b_norm:
        return 0.0
    # Cheap prefilter before the O(n*m) char diff (this is the hot path of
    # _cross_source_agreement's O(n^2) pair loop). Pairs whose word sets
    # barely overlap and share no rare/numeric anchors cannot reach the
    # corroboration band (0.55) on the char ratio: the matched characters
    # are bounded by the shared tokens' length. Short claims skip the gate —
    # their diffs are cheap anyway.
    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())
    if len(a_tokens) >= 4 and len(b_tokens) >= 4:
        inter = a_tokens & b_tokens
        union = a_tokens | b_tokens
        rare_shared = sum(
            1 for t in inter if len(t) >= 7 or any(c.isdigit() for c in t)
        )
        if len(inter) / len(union) < 0.10 and rare_shared < 2:
            return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def _source_quality(facts: List[Dict[str, Any]]) -> float:
    scores = [source_reliability_score(str(f.get("source", ""))) for f in facts if f.get("source")]
    return sum(scores) / len(scores) if scores else 0.0


def _source_diversity(facts: List[Dict[str, Any]]) -> float:
    """Distinct PUBLISHERS (registrable domains) relative to fact count.

    Uses registrable domains, not raw hosts, matching the evidence spine:
    `arxiv.org` and `ar5iv.labs.arxiv.org` are one publisher, as are
    `www.britannica.com` and `kids.britannica.com`. The previous raw-host
    version counted subdomains as independent sources, so a pool dominated by
    one organisation's subdomains could read 1.0. Returns 0.0 when there is
    only one fact or one publisher — a single publisher is never diverse.
    """
    from app.core.evidence_grade import registrable_domain

    domains: set[str] = set()
    for f in facts:
        source = str(f.get("source", ""))
        if not source:
            continue
        domain = registrable_domain(source) or extract_domain(source)
        if domain:
            domains.add(domain)
    if len(facts) < 2 or len(domains) < 2:
        return 0.0
    return min(1.0, len(domains) / max(2.0, min(len(facts), 5)))


def _citation_coverage(facts: List[Dict[str, Any]]) -> float:
    """Fraction of facts that passed verification (Phase 2.3)."""
    if not facts:
        return 0.0
    if not any("verified" in f for f in facts):
        return 0.0  # verification never ran — don't award coverage credit
    verified = sum(1 for f in facts if f.get("verified"))
    return verified / len(facts)


def _claim_verification_strength(facts: List[Dict[str, Any]]) -> float:
    """Mean lexical-overlap score among facts that were checked (Phase 2.3)."""
    checked = [f for f in facts if isinstance(f.get("verification_score"), (int, float))]
    if not checked:
        return 0.0
    return sum(_safe_conf(f.get("verification_score", 0.0)) for f in checked) / len(checked)


def _cross_source_agreement(facts: List[Dict[str, Any]]) -> float:
    """Fraction of claims independently corroborated by a DIFFERENT publisher.

    Two ways to measure this, tried in order:

    1. MEASURED CORROBORATION (preferred). Dedup and the corroboration
       acquisition pass already record, per fact, how many distinct
       registrable-domain publishers assert it (`corroboration_count`, written
       by `dedupe_semantic_facts` / `apply_corroboration`). That count is the
       signal a human means by "cross-source agreement", and the O(n^2)
       pairwise re-derivation below is blind to it — it asks whether two
       *different* claims are similar, which by construction they usually are
       not (verified live: median 0.077, 34% exactly zero). When any fact
       carries the measured count, this is authoritative.
    2. PAIRWISE FALLBACK. Only for legacy callers whose facts predate the
       corroboration fields: fraction of claims with a similar claim from a
       different registrable domain. Independence is at the registrable-domain
       level (blog.example.com and www.example.com are one publisher).

    Neither path can manufacture agreement: a claim with one publisher stays
    at 1 corroborator, and the fallback still requires a genuinely similar
    second claim.
    """
    if not facts:
        return 0.0

    measured = [
        f for f in facts
        if isinstance(f, dict) and "corroboration_count" in f
    ]
    if measured:
        # Agreement is scored at the ANGLE (sub_question) level, not per claim.
        # A specific, useful claim ("BLEU 28.4 on WMT 2014") is rarely restated
        # verbatim elsewhere, so a per-claim 0/1 average penalizes precision by
        # construction and floors the signal. What a researcher needs to know
        # is whether each research angle is backed by 2+ independent sources,
        # which is what this measures. A claim with one publisher still scores
        # 0; a single ungrouped pool falls back to per-claim scoring.
        by_angle: Dict[str, List[Dict[str, Any]]] = {}
        for f in measured:
            angle = str(f.get("sub_question", "") or "").strip()
            by_angle.setdefault(angle, []).append(f)

        def _angle_score(group: List[Dict[str, Any]]) -> float:
            # The angle's best-supported claim: if any fact for this angle has
            # 2+ independent publishers, the angle is independently corroborated.
            corroborated = sum(
                1 for f in group
                if _safe_int(f.get("corroboration_count", 1)) >= 2
            )
            return corroborated / len(group)

        if len(by_angle) > 1 or "" not in by_angle:
            return sum(_angle_score(g) for g in by_angle.values()) / len(by_angle)

        # No usable angle attribution: fall back to per-claim scoring.
        return sum(
            min(1.0, max(0, _safe_int(f.get("corroboration_count", 1)) - 1))
            for f in measured
        ) / len(measured)

    if len(facts) < 2:
        return 0.0
    from app.core.evidence_grade import registrable_domain

    domains = [
        registrable_domain(str(f.get("source", ""))) or extract_domain(str(f.get("source", "")))
        for f in facts
    ]
    claims = [str(f.get("claim", "")) for f in facts]
    corroborated = 0
    for i, claim in enumerate(claims):
        for j, other in enumerate(claims):
            if i == j or domains[i] == domains[j]:
                continue
            if _similarity(claim, other) >= CORROBORATION_SIMILARITY:
                corroborated += 1
                break
    return corroborated / len(claims)


def _critic_survival(
    critique: Dict[str, Any],
    iteration: int,
    max_iterations: int,
    signals: Dict[str, Any] | None = None,
) -> float:
    """How much of the evidence base survives the critic's scrutiny.

    Two prior versions were wrong in the same direction. The first returned a
    constant (1.0/0.6/0.4) keyed on `is_sufficient` and the ceiling. The second
    averaged the measured coverage/corroboration signals but still hard-capped
    at 0.4/0.6 on any fail, so a run whose critic named one minor gap scored
    identically to one with three severe gaps — and the cap made "hit the
    iteration limit" the dominant term. Live evidence: 91% of 250 reviews at
    0.4/0.6.

    Survival here measures how much of the critic's *specific criticism* the
    evidence base already answers. The critic emits `gaps` (uncovered angles)
    and `gate_failures` (objective failures). A fail with zero named gaps is a
    model hedge, not an evidence weakness; a fail with many named gaps is a
    real shortfall. So:

      * critic passed                         -> 1.0
      * fail, no named gaps                   -> 0.6 (hedge, not weakness)
      * fail, named gaps                      -> 1 - (named/applicable_criteria)

    `is_sufficient` still dominates (a pass is a pass), and the value is never
    above 0.6 on a fail, so a FAIL can never outrank a PASS. The ceiling no
    longer appears in the formula at all.
    """
    if bool(critique.get("is_sufficient", False)):
        return 1.0

    gaps = [g for g in (critique.get("gaps") or []) if str(g).strip()]
    gate_failures = [f for f in (critique.get("gate_failures") or []) if str(f).strip()]
    named = len(gaps) + len(gate_failures)
    if named == 0:
        return 0.6  # a hedge, not a demonstrated evidence weakness

    # Normalizer: the number of objective criteria the critic can name. A pool
    # failing 1 of many criteria survives most of its scrutiny; failing all of
    # them survives little. Floor of 3 keeps a single failure from reading as
    # catastrophic, and matches the observable gate vocabulary.
    denominator = max(3, named)
    survival = 1.0 - (named / denominator)
    return round(max(0.0, min(0.6, survival)), 3)


# When the summarizer ran on its deterministic fallback, the fact pool is
# extractive rather than model-written: every claim lexically overlaps its
# own source by construction, so source/verification signals read high no
# matter how good the evidence actually is. Capping below the sufficiency
# threshold (0.75) keeps a degraded run from finalizing as "High" confidence
# and keeps the honest signals (degraded list) attached to the breakdown.
DEGRADED_CAP = 0.55
EXTRACTIVE_FALLBACK_AGENTS = frozenset({"summarizer", "synthesizer"})

# A summarizer that fell back to heuristic extraction but whose pool is still
# well-verified and independently corroborated is NOT a degraded evidence base:
# the fallback is extractive, but the evidence pipeline (verification,
# corroboration, grading) succeeded on it. Capping such a run to 0.55 punishes
# good evidence for a formatting/parse failure. The cap therefore depends on
# the evidence quality: it applies to the summarizer only when the pool is
# actually weak. The synthesizer is different — an extractive REPORT is a
# degraded product regardless of the pool, so it always caps.
SUMMARIZER_CAP_MIN_CORROBORATED_RATIO = 0.30
SUMMARIZER_CAP_MIN_VERIFIED_RATIO = 0.60


def _summarizer_fallback_is_degraded(
    facts: List[Dict[str, Any]],
    contradictions: List[Dict[str, Any]] | None,
) -> bool:
    """Should a summarizer fallback cap confidence?

    Returns True (cap) when the evidence pool is genuinely weak — no pool, no
    verification, or fewer than a third of verified claims independently
    corroborated by a second publisher. Returns False (do not cap) only when the
    extractive pool is still verified AND independently corroborated: the
    summarizer fell back to extraction, but the evidence pipeline (verification
    + corroboration) succeeded on it, so the report is not a false-confidence
    risk of the kind the cap exists for. Deterministic and total.
    """
    pool = [f for f in (facts or []) if isinstance(f, dict) and str(f.get("claim", "")).strip()]
    if not pool:
        return True
    if not any("verified" in f for f in pool):
        # Verification never ran: we cannot claim the extractive pool is good.
        return False
    verified = [f for f in pool if f.get("verified")]
    if not verified:
        return True
    verified_ratio = len(verified) / len(pool)
    if verified_ratio < SUMMARIZER_CAP_MIN_VERIFIED_RATIO:
        return True
    corroborated_ratio = sum(
        1 for f in verified if int(f.get("corroboration_count", 1) or 1) >= 2
    ) / len(verified)
    if corroborated_ratio < SUMMARIZER_CAP_MIN_CORROBORATED_RATIO:
        return True
    return False


def _axis_coverage(sub_questions: List[Dict[str, Any]] | None, facts: List[Dict[str, Any]]) -> float:
    """Covered research axes / planned axes.

    Facts attribute to a contract's axis via the fact's own `sub_question`
    (stamped by the summarizer from the source record and preserved by dedup).
    Only verified facts count — unless verification never ran, in which case
    the pool is judged as-is.

    An axis counts as covered only when at least AXIS_MIN_FACTS verified facts
    support it. The previous "any single fact" rule let every axis reach 1.0
    with one claim each, so a plan whose angles were each touched once — but
    not actually investigated — reported full coverage. A single fact about an
    angle is not coverage of that angle.
    """
    axis_by_question: Dict[str, str] = {}
    for q in sub_questions or []:
        if isinstance(q, dict):
            question = str(q.get("question", "")).strip()
            axis = str(q.get("axis", "")).strip()
            if question and axis:
                axis_by_question[question] = axis
    if not axis_by_question:
        return 0.0

    pool = [f for f in facts or [] if isinstance(f, dict)]
    if any("verified" in f for f in pool):
        pool = [f for f in pool if f.get("verified")]

    facts_per_axis: Dict[str, int] = {}
    for f in pool:
        axis = axis_by_question.get(str(f.get("sub_question", "")).strip())
        if axis:
            facts_per_axis[axis] = facts_per_axis.get(axis, 0) + 1

    covered = sum(1 for n in facts_per_axis.values() if n >= AXIS_MIN_FACTS)
    return min(1.0, covered / len(set(axis_by_question.values())))


def _grade_records(facts: List[Dict[str, Any]], contradictions: List[Dict[str, Any]] | None) -> float | None:
    """Evidence-grade quality of the fact pool, or None when ungradeable.

    Grades come from measured per-claim signals (independent corroboration,
    verification, numeric support, contradiction) — never an LLM opinion.
    Returns None (so the confidence weights stay historical) when there is no
    usable pool or grading fails; grading must never break confidence.
    """
    pool = [f for f in (facts or []) if isinstance(f, dict) and str(f.get("claim", "")).strip()]
    if not pool:
        return None
    try:
        from app.core.evidence_grade import grade_facts

        graded = grade_facts(pool, contradictions=contradictions)
        records = [g["evidence"] for g in graded if isinstance(g.get("evidence"), dict)]
        if not records:
            return None
        # grade_facts already emitted serialized records; score them directly
        # (EvidenceRecord reconstruction from partial dicts is fragile).
        weights = {  # same scale as evidence_grade.evidence_quality_score
            "A": 1.0, "B": 0.75, "C": 0.4, "D": 0.1,
        }
        return round(sum(weights.get(str(r.get("grade", "D")), 0.1) for r in records) / len(records), 3)
    except Exception as exc:  # grading must never break confidence
        logger.warning("evidence grading failed, skipping signal: %s", exc)
        return None


def _contradiction_penalty(
    contradictions: List[Dict[str, Any]] | None,
    fact_count: int = 0,
) -> float:
    """Total subtraction for cross-source conflicts (0 .. CONTRADICTION_PENALTY_CAP).

    Pool-size scaled: one severe conflict among 30 facts is noise (or one
    stale page); the same conflict among 3 facts means a third of the
    evidence base disagrees with itself. Small pools pay up to 2x the per-
    conflict penalty, capped as before so a noisy pool still can't zero out
    an otherwise-solid run.
    """
    scale = min(2.0, max(1.0, 6.0 / max(1, fact_count)))
    penalty = 0.0
    for c in contradictions or []:
        if not isinstance(c, dict) or c.get("intra_source"):
            continue
        # Fix C: a contradiction explained by a different period/scope/metric
        # is resolved — recorded for the report, but it must not penalize an
        # otherwise-sound run. Only genuine same-unit/scope/period conflicts
        # subtract.
        if c.get("resolved"):
            continue
        severity = _safe_conf(c.get("severity", 0.5))
        if severity >= SEVERE_CONTRADICTION_SEVERITY:
            penalty += CONTRADICTION_PENALTY_SEVERE * scale
        else:
            penalty += CONTRADICTION_PENALTY_MODERATE * scale
    return min(CONTRADICTION_PENALTY_CAP, penalty)


def compute_confidence(
    facts: List[Dict[str, Any]],
    critique: Dict[str, Any],
    iteration: int,
    max_iterations: int,
    source_dates: List[str] | None = None,
    degraded: List[str] | None = None,
    contradictions: List[Dict[str, Any]] | None = None,
    answer_support: Dict[str, Any] | None = None,
    sub_questions: List[Dict[str, Any]] | None = None,
    provider_degraded: bool = False,
) -> Dict[str, Any]:
    """Return {"overall": float, "signals": {...}} with per-signal values.

    `degraded` is the per-request fallback list (degradation.take_fallbacks):
    when an evidence-producing stage ran deterministically, the overall score
    is capped — a degraded run must never look more confident than a healthy
    one that was forced through the ceiling.

    `contradictions` (from the live engine), `answer_support` (previous
    iteration's verify_answer_support) and `sub_questions` (the plan) wire
    the three v3 signals; each contributes only when present so historical
    scores stay comparable.

    `provider_degraded` is True when a provider failed for TRANSPORT reasons
    during the run. A provider outage preserves no evidence signal — the
    extractive fallback that replaces it self-verifies — so a transport
    failure must not inflate confidence even when the resulting pool looks
    well-corroborated. It applies the cap unconditionally (unlike a
    summarizer fallback, whose cap is evidence-conditional), while the
    degradation reasons still report provider-transient vs weak-evidence
    separately so the two causes are never conflated.
    """
    freshness_value, measured = _freshness(source_dates, facts)
    weights = dict(WEIGHTS_FRESH if measured else WEIGHTS)

    support_rate: float | None = None
    if isinstance(answer_support, dict):
        raw = answer_support.get("rate")
        if raw is not None:
            support_rate = max(0.0, min(1.0, _safe_conf(raw)))

    planned_axes_count = sum(
        1 for q in (sub_questions or []) if isinstance(q, dict) and str(q.get("axis", "")).strip()
    )

    signals: Dict[str, float] = {
        "source_quality": round(_source_quality(facts), 3),
        "source_diversity": round(_source_diversity(facts), 3),
        "citation_coverage": round(_citation_coverage(facts), 3),
        "claim_verification_strength": round(_claim_verification_strength(facts), 3),
        "cross_source_agreement": round(_cross_source_agreement(facts), 3),
        "freshness": freshness_value,
    }
    # Critic survival reads the measured dimensions above, so it is computed
    # after they exist (it was a 3-valued constant when computed inline).
    signals["critic_survival"] = round(
        _critic_survival(critique, iteration, max_iterations, signals), 3
    )

    # --- v3 signal wiring (only when the inputs exist) --------------------
    if support_rate is not None:
        weights["citation_coverage"] = round(weights.get("citation_coverage", 0.25) - 0.10, 4)
        weights["citation_support"] = 0.10
        signals["citation_support"] = round(support_rate, 3)
    if planned_axes_count > 0:
        weights["source_diversity"] = round(weights.get("source_diversity", 0.15) - 0.05, 4)
        weights["axis_coverage"] = 0.05
        signals["axis_coverage"] = round(_axis_coverage(sub_questions, facts), 3)

    # --- evidence-grade signal (Step 2): how good is the evidence itself? ---
    # Grades are computed from measured, per-claim signals (independence,
    # verification, numeric support, contradiction) — never from an LLM's
    # opinion. Carved 0.05 from source_quality so the historical weights
    # stay intact when grades are absent (older callers/tests).
    grade_records = _grade_records(facts, contradictions)
    if grade_records is not None:
        weights["source_quality"] = round(weights.get("source_quality", 0.20) - 0.05, 4)
        weights["claim_evidence_quality"] = 0.05
        signals["claim_evidence_quality"] = round(grade_records, 3)

    overall = sum(weights.get(name, 0.0) * value for name, value in signals.items())
    overall = round(max(0.0, min(1.0, overall)), 3)

    notes = []
    if not measured:
        notes.append("freshness not yet measured (no publish dates captured) — weighted 0")

    penalty = _contradiction_penalty(contradictions, fact_count=len(facts or []))
    if penalty > 0:
        overall = round(max(0.0, overall - penalty), 3)
        unresolved = sum(
            1 for c in (contradictions or [])
            if isinstance(c, dict) and not c.get("resolved") and not c.get("intra_source")
        )
        notes.append(
            f"confidence penalized {penalty:.2f} for {unresolved} "
            "unresolved cross-source contradiction(s)"
        )

    degraded_agents = {str(a) for a in (degraded or []) if a}
    extractive = sorted(degraded_agents & EXTRACTIVE_FALLBACK_AGENTS)
    # A synthesizer fallback is always a degraded PRODUCT; a summarizer fallback
    # only caps when the evidence pool it produced is actually weak (see
    # _summarizer_fallback_is_degraded). This keeps an extractive-but-well-
    # corroborated run from being falsely capped at 0.55 while preserving the
    # guard against genuinely degraded runs.
    capping = []
    for agent_name in extractive:
        if agent_name == "synthesizer":
            capping.append(agent_name)
        elif _summarizer_fallback_is_degraded(facts, contradictions):
            capping.append(agent_name)
        else:
            notes.append(
                "summarizer ran on deterministic extraction, but the evidence "
                "pool is verified and independently corroborated — no cap applied"
            )
    if capping:
        notes.append(
            "confidence capped: " + ", ".join(capping)
            + " ran on deterministic extraction — claims are unrewritten source text"
        )
        overall = round(min(overall, DEGRADED_CAP), 3)
    elif provider_degraded:
        # A provider failed for transport reasons. Even if the extractive pool
        # happens to look verified/corroborated, it was produced under an
        # outage — the same reason the extractive cap exists. Applied only when
        # no explicit extraction cap already fired, so the note names the real
        # cause. This is NOT weak evidence: the reasons object distinguishes
        # provider-transient from evidence weakness downstream.
        notes.append(
            "confidence capped: an LLM provider failed during the run "
            "(provider-transient) — evidence was produced under degradation"
        )
        overall = round(min(overall, DEGRADED_CAP), 3)
    return {
        "overall": overall,
        "signals": signals,
        "weights": weights,
        "notes": notes,
    }
