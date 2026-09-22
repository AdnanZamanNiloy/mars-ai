"""Embedding-ranked per-section context selection (GPT Researcher adaptation).

The failure this module exists for
----------------------------------
`build_outline` groups the report's evidence by AXIS only: every fact whose
`axis` field says "evidence" lands in the Evidence section, whether it is
about the section's actual question or not. A broad query's sections are
written from pools that can contain any claim sharing a coarse axis label,
which is why a comparison section can be padded with a definitional fact and
a definition section can receive a market figure. GPT Researcher avoids this
by ranking each subtopic's context against that subtopic's *query* with
embeddings and handing the writer only the top chunks (`ContextCompressor`,
`gpt_researcher/context/compression.py`).

MARS has no embeddings endpoint (see AGENTS.md: "no local embedding models",
remote LLM APIs only), so the ranking here is built on the EXISTING TF-IDF
hybrid engine (`app.core.semantic`) — the same engine the dedup, contradiction
and citation-support paths already trust. That is deliberate: no new model,
no new dependency, no network, no extra RAM.

Ranking, not filtering
----------------------
Pure similarity is the wrong objective for research evidence. A unique,
well-corroborated quantitative claim often scores mid-range against its
section's label while three near-identical restatements of a tangential claim
score high and crowd it out. The score therefore COMBINES, with bounded
weights:

  * semantic relevance to the section (title + question + coverage goal),
  * evidence quality (A-D grade, `evidence_quality`),
  * independent corroboration (`corroboration_count`, `needs_corroboration`),
  * source authority / primary preference (`app.agents.sources`),
  * a bounded diversity bonus for a publisher not yet represented.

Two guarantees keep the selector from doing harm:

  1. HIGH-IMPACT FLOOR — quantitative and definitional claims (the figures and
     definitions a report is judged on, and the ones `_assign_grade` already
     treats as "important") are reserved a slot even when their similarity is
     mid-range. They are only displaced by other high-impact claims.
  2. DETERMINISTIC FALLBACK — any error or empty result for a section returns
     the original axis-grouped facts, so a section is never shipped with no
     evidence (AGENTS.md 4.7, 4.4).

The selector never mutates its input: it returns shallow copies of the full
claim dicts (claim, source, evidence record, corroborating_sources, numbers,
verification flags), so citation traceability survives unchanged. Callers that
want thematic compression run `_compress_to_themes` AFTER selection.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.agents.evidence_utils import extract_numbers
from app.agents.sources import classify_source
from app.core.logging import get_logger

logger = get_logger(__name__)

# Per-section output cap, consistent with the writer budget: the section-wise
# path numbers every section's facts into ONE shared legend capped at
# MAX_LEGEND_SOURCES (14) in the synthesizer, so a larger per-section pool
# cannot be cited anyway and would starve later sections of legend numbers.
DEFAULT_MAX_FACTS = 12

# How many global-pool facts a section may import (the cross-section recovery
# allowance). Small by design: recovery targets the few off-axis facts that are
# genuinely relevant to this section, not a wholesale copy of the pool. Keeping
# it bounded preserves the per-section partitioning and the shared legend.
DEFAULT_MAX_IMPORT = 4

# How many additional high-impact facts above the reserved floor may be kept
# when the pool is large. Kept small so the floor cannot become "everything".
DEFAULT_IMPACT_RESERVE = 6

# The impact floor only protects high-impact claims that are at least this
# relevant to the section, relative to the section's best candidate. Without
# this, a unique quantitative figure in ANOTHER topic's evidence would be
# reserved into every section ahead of that section's own on-topic claims —
# the off-topic-evidence failure this module exists to fix, inverted.
IMPACT_RELEVANCE_FRACTION = 0.25

# Absolute relevance floor for the impact reservation. A purely semantic 0.0
# (no shared content with the section at all) is never reserved, even when the
# section's own best candidate also scores near zero.
IMPACT_RELEVANCE_MIN = 0.03

# Combined-score weights. Relevance dominates (this is a relevance selector)
# but never so far that a mid-relevance, A-grade, corroborated, primary claim
# loses to a high-relevance unverified restatement.
_W_RELEVANCE = 1.0
_W_GRADE = 0.55
_W_CORROBORATION = 0.30
_W_AUTHORITY = 0.25
_W_PRIMARY = 0.20
_W_DIVERSITY = 0.12
_W_VERIFIED = 0.15

# Bounded bonus for a fact already assigned to this section by axis grouping.
# Small by design: it keeps the section's own evidence from being displaced by
# cross-section recovery without letting the coarse axis label override a
# genuinely more relevant claim from elsewhere.
_W_OWN = 0.10

# Grade -> 0-1 quality. Mirrors evidence_grade.evidence_quality_score's weights
# so a "good" claim means the same thing here as in the confidence engine.
_GRADE_VALUE: Dict[str, float] = {"A": 1.0, "B": 0.75, "C": 0.4, "D": 0.1}

_AXIS_LABELS: Dict[str, str] = {
    "definition": "what it is definition meaning concept",
    "mechanism": "how it works mechanism process cause",
    "application": "applications uses example real world",
    "evidence": "evidence data statistics numbers figures",
    "comparison": "compare comparison versus versus alternatives trade-off",
    "criticism": "criticism limitations risks problems drawbacks",
    "history": "history background origin timeline",
    "outlook": "outlook future trends forecast projection",
}


def _section_query_text(section: Any) -> str:
    """The text a section's facts are ranked against.

    Combines the section title, its question and its coverage goal, plus the
    axis's own vocabulary. A section's title alone ("How It Compares") carries
    little content; the axis vocabulary restores the dimension's intent
    ("compare comparison versus alternatives trade-off") without an LLM.
    """
    axis = str(getattr(section, "axis", "") or "").strip().lower()
    parts = [
        str(getattr(section, "title", "") or ""),
        str(getattr(section, "question", "") or ""),
        str(getattr(section, "coverage_goal", "") or ""),
        _AXIS_LABELS.get(axis, ""),
    ]
    return re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()


def _evidence_record(fact: Dict[str, Any]) -> Dict[str, Any]:
    record = fact.get("evidence")
    return record if isinstance(record, dict) else {}


def _grade_value(fact: Dict[str, Any]) -> str:
    """The fact's grade string, preferring the annotated record."""
    grade = str(_evidence_record(fact).get("grade", "") or "").strip().upper()
    if grade:
        return grade
    return str(fact.get("evidence_grade", "") or "").strip().upper()


def _is_high_impact(fact: Dict[str, Any]) -> bool:
    """Quantitative or definitional — the classes `_assign_grade` treats as
    important. These are the claims a reader judges the report on, so they are
    reserved a slot rather than left to a similarity cut."""
    claim = str(fact.get("claim", "") or "")
    if not claim:
        return False
    if extract_numbers(claim, limit=1):
        return True
    low = f" {claim.lower()} "
    return " is " in low or " are " in low or " refers to " in low or " means " in low


def _authority(fact: Dict[str, Any]) -> float:
    record = _evidence_record(fact)
    try:
        authority = float(record.get("authority", 0.0) or 0.0)
    except (TypeError, ValueError):
        authority = 0.0
    if authority > 0.0:
        return authority
    source = str(fact.get("source", "") or "").strip()
    if not source:
        return 0.0
    try:
        return float(classify_source(source).authority)
    except Exception as exc:  # noqa: BLE001 - scoring must never break a run
        logger.warning("[SectionContext] source classification failed", exc_info=exc)
        return 0.0


def _is_primary(fact: Dict[str, Any]) -> bool:
    record = _evidence_record(fact)
    if "is_primary" in record:
        return bool(record.get("is_primary"))
    if fact.get("is_primary") is not None:
        return bool(fact.get("is_primary"))
    source = str(fact.get("source", "") or "").strip()
    if not source:
        return False
    try:
        return bool(classify_source(source).is_primary)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SectionContext] primary check failed", exc_info=exc)
        return False


def _corroboration(fact: Dict[str, Any]) -> float:
    """0-1 corroboration signal: log-scaled independent source count, with a
    small bonus when the claim is not flagged as needing corroboration."""
    record = _evidence_record(fact)
    try:
        count = int(
            record.get("corroboration_count", fact.get("corroboration_count", 1)) or 1
        )
    except (TypeError, ValueError):
        count = 1
    count = max(1, count)
    # 1 -> 0.0, 2 -> 0.5, 3 -> 0.67, 4+ -> ~0.75-0.9. Bounded to 1.
    base = min(1.0, (count - 1) / 2.0)
    if record.get("needs_corroboration") and count < 2:
        base = max(0.0, base - 0.15)
    return base


def _verified_signal(fact: Dict[str, Any]) -> float:
    if fact.get("verified") is True:
        return 1.0
    record = _evidence_record(fact)
    if record.get("verified") is True:
        return 1.0
    return 0.0


def _publisher_key(fact: Dict[str, Any]) -> str:
    """Registrable-domain-ish publisher key for diversity accounting."""
    record = _evidence_record(fact)
    domain = str(record.get("domain", "") or "").strip().lower()
    if domain:
        return domain
    source = str(fact.get("source", "") or "").strip()
    if not source:
        return ""
    try:
        return classify_source(source).domain.lower()
    except Exception:  # noqa: BLE001 - diversity is best-effort
        return ""


def _claim_text(fact: Dict[str, Any]) -> str:
    return re.sub(r"\s+", " ", str(fact.get("claim", "") or "")).strip()


def rank_section_facts(
    section: Any,
    candidates: Sequence[Dict[str, Any]],
    *,
    max_facts: int = DEFAULT_MAX_FACTS,
    impact_reserve: int = DEFAULT_IMPACT_RESERVE,
    reserved_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Rank a section's candidate facts by the combined score and cut to cap.

    Deterministic, LLM-free. Always re-ranks (the most relevant fact must lead
    the prompt even when the pool already fits), never mutates `candidates`,
    and returns shallow copies so downstream annotations stay isolated. Facts
    whose `id()` is in `reserved_ids` (the section's own axis group) are given
    a guaranteed slot. On any error, or when no candidate survives, returns the
    candidates unchanged (up to `max_facts`) — the axis-grouped pool, never an
    empty section.
    """
    facts = [f for f in (candidates or []) if isinstance(f, dict) and _claim_text(f)]
    if not facts:
        return []
    limit = max(1, int(max_facts or DEFAULT_MAX_FACTS))

    try:
        return _rank_impl(
            section,
            facts,
            limit=limit,
            impact_reserve=impact_reserve,
            reserved_ids=reserved_ids or set(),
        )
    except Exception as exc:  # noqa: BLE001 - deterministic fallback below
        logger.warning(
            "[SectionContext] ranking failed for section '%s'; "
            "falling back to axis-grouped order",
            str(getattr(section, "title", "") or getattr(section, "axis", "") or "?"),
            exc_info=exc,
        )
        return [dict(f) for f in facts[:limit]]


def _rank_impl(
    section: Any,
    facts: List[Dict[str, Any]],
    *,
    limit: int,
    impact_reserve: int,
    reserved_ids: set,
) -> List[Dict[str, Any]]:
    from app.core.semantic import rank_by_similarity

    query = _section_query_text(section)
    claims = [_claim_text(f) for f in facts]
    relevance = rank_by_similarity(query, claims) if query else [0.0] * len(facts)

    scored: List[Tuple[float, int, Dict[str, Any]]] = []
    for i, fact in enumerate(facts):
        rel = float(relevance[i]) if i < len(relevance) else 0.0
        grade = _grade_value(fact)
        score = (
            _W_RELEVANCE * rel
            + _W_GRADE * _GRADE_VALUE.get(grade, 0.1)
            + _W_CORROBORATION * _corroboration(fact)
            + _W_AUTHORITY * min(1.0, max(0.0, _authority(fact)))
            + _W_PRIMARY * (1.0 if _is_primary(fact) else 0.0)
            + _W_VERIFIED * _verified_signal(fact)
        )
        scored.append((score, i, fact))

    # Deterministic ordering: score desc, then original index asc.
    scored.sort(key=lambda item: (-item[0], item[1]))

    selected: List[Dict[str, Any]] = []
    selected_ids: set[int] = set()

    # Phase 0 — impact floor: reserve up to `impact_reserve` slots for the
    # highest-scoring high-impact (quantitative/definitional) claims, but ONLY
    # those that clear a relevance bar to THIS section. A high-impact claim
    # belonging to another topic must not be imported into every section.
    if impact_reserve > 0:
        max_rel = max(relevance) if relevance else 0.0
        reserved = 0
        for score, i, fact in scored:
            if reserved >= impact_reserve or len(selected) >= limit:
                break
            if not _is_high_impact(fact):
                continue
            if relevance[i] < IMPACT_RELEVANCE_MIN:
                continue
            if relevance[i] < IMPACT_RELEVANCE_FRACTION * max_rel and relevance[i] < 0.15:
                continue
            selected.append(dict(fact))
            selected_ids.add(i)
            reserved += 1

    # Phase 1 — general selection with bounded bonuses. Re-scan the score order
    # each pick so a claim from an unseen publisher, or one the section was
    # already assigned by axis grouping, can overtake a marginally higher-scoring
    # fact. Bonuses are bounded so relevance + evidence quality still dominate.
    seen_publishers = {_publisher_key(f) for f in selected if _publisher_key(f)}
    selected_claims = {_claim_text(f) for f in selected}
    remaining = [item for item in scored if item[1] not in selected_ids]
    while remaining and len(selected) < limit:
        best_index = 0
        best_value = None
        for pos, (score, i, fact) in enumerate(remaining):
            publisher = _publisher_key(fact)
            bonus = _W_DIVERSITY if publisher and publisher not in seen_publishers else 0.0
            if id(fact) in reserved_ids:
                bonus += _W_OWN
            value = score + bonus
            if best_value is None or value > best_value:
                best_value = value
                best_index = pos
        _, i, fact = remaining.pop(best_index)
        claim = _claim_text(fact)
        if claim in selected_claims:
            continue  # same claim already picked (copies share text, not id)
        selected.append(dict(fact))
        selected_claims.add(claim)
        selected_ids.add(i)
        publisher = _publisher_key(fact)
        if publisher:
            seen_publishers.add(publisher)

    # Own-group guarantee: a section that had its own axis-grouped evidence must
    # never end up with none of it. Swap the weakest pick for the best own fact,
    # matching by CLAIM TEXT (selected entries are copies, so ids differ).
    if reserved_ids:
        own_claims = {_claim_text(f) for f in facts if id(f) in reserved_ids}
        if own_claims and not (own_claims & selected_claims):
            own_best = next(
                ((score, i, fact) for score, i, fact in scored if id(fact) in reserved_ids),
                None,
            )
            if own_best is not None:
                if len(selected) >= limit and selected:
                    selected.pop()
                selected.append(dict(own_best[2]))

    if not selected:
        return [dict(f) for f in facts[:limit]]
    return selected[:limit]


def build_section_candidate_pool(
    section: Any,
    own: Sequence[Dict[str, Any]],
    global_pool: Sequence[Dict[str, Any]],
    *,
    max_candidates: int = DEFAULT_MAX_FACTS,
    max_import: int = DEFAULT_MAX_IMPORT,
) -> Tuple[List[Dict[str, Any]], set]:
    """Bounded candidate pool for one section: its own facts + top global ones.

    Widening a section's pool with the ENTIRE global pool made every section
    see every fact (under the cap), which destroys the per-section partitioning
    the section-wise writer relies on and floods the shared citation legend.
    The global supplement is therefore bounded by BOTH the per-section budget
    and `max_import`: only the most relevant few global facts join the pool,
    pre-ranked by the same hybrid engine. Returns (candidates, reserved_ids)
    where `reserved_ids` are the section's own facts, which the selector
    guarantees a slot.
    """
    own_list = [f for f in (own or []) if isinstance(f, dict) and _claim_text(f)]
    own_ids = {id(f) for f in own_list}
    budget = min(
        max(0, int(max_import)),
        max(0, int(max_candidates) - len(own_list)),
    )
    supplement: List[Dict[str, Any]] = []
    if budget > 0:
        candidates = [
            f
            for f in (global_pool or [])
            if isinstance(f, dict) and _claim_text(f) and id(f) not in own_ids
        ]
        if len(candidates) > budget:
            try:
                from app.core.semantic import rank_by_similarity

                query = _section_query_text(section)
                scores = rank_by_similarity(query, [_claim_text(f) for f in candidates])
                order = sorted(range(len(candidates)), key=lambda i: -scores[i])
                candidates = [candidates[i] for i in order[:budget]]
            except Exception as exc:  # noqa: BLE001 - bounded fallback below
                logger.warning(
                    "[SectionContext] global supplement ranking failed; "
                    "taking the first %d facts",
                    budget,
                    exc_info=exc,
                )
                candidates = candidates[:budget]
        supplement = candidates
    return own_list + supplement, own_ids


def select_section_facts(
    section: Any,
    candidates: Sequence[Dict[str, Any]],
    *,
    max_facts: int = DEFAULT_MAX_FACTS,
    impact_reserve: int = DEFAULT_IMPACT_RESERVE,
    reserved_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Public entry point: ranked, quality-weighted facts for one section.

    `reserved_ids` are `id()`s of facts that MUST keep a slot (the section's
    own axis-grouped evidence when the candidate pool was widened with the
    global pool) — they cannot be displaced by cross-section recovery.

    Guarantees:
      * never mutates `candidates`,
      * full claim dicts are preserved (evidence record, corroborating_sources,
        numbers, verification flags, citation fields),
      * empty input -> empty output; any failure -> the axis-grouped pool
        (so a section is never starved of evidence).
    """
    return rank_section_facts(
        section,
        candidates,
        max_facts=max_facts,
        impact_reserve=impact_reserve,
        reserved_ids=reserved_ids,
    )


def select_context_for_outline(
    outline: Any,
    facts: Sequence[Dict[str, Any]] | None = None,
    *,
    max_facts: int = DEFAULT_MAX_FACTS,
    impact_reserve: int = DEFAULT_IMPACT_RESERVE,
) -> List[Tuple[Any, List[Dict[str, Any]]]]:
    """Pair each outline section with its ranked facts.

    Drop-in replacement for `outline.group_facts_by_section` when the caller
    wants per-section ranking instead of raw axis grouping. When `facts` is
    given it is used as the shared candidate pool for every section (each
    section's own axis group is reserved a slot); otherwise each section's own
    `facts` are used (the axis-grouped default).
    """
    pairs: List[Tuple[Any, List[Dict[str, Any]]]] = []
    for section in getattr(outline, "sections", []) or []:
        own = list(getattr(section, "facts", []) or [])
        if facts is not None:
            own_ids = {id(f) for f in own}
            candidates = own + [
                f for f in facts if isinstance(f, dict) and id(f) not in own_ids
            ]
            reserved = {id(f) for f in own}
        else:
            candidates = own
            reserved = None
        if not candidates:
            pairs.append((section, []))
            continue
        selected = select_section_facts(
            section,
            candidates,
            max_facts=max_facts,
            impact_reserve=impact_reserve,
            reserved_ids=reserved,
        )
        pairs.append((section, selected))
    return pairs
