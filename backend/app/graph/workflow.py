from __future__ import annotations

import asyncio
import datetime
import re
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.evidence_utils import dedupe_semantic_facts, filter_facts_by_domain, verify_answer_support
from app.agents.answer_quality import evaluate_answer
from app.agents.critic import critic_agent
from app.agents.intent import classify_intent, heuristic_intent
from app.agents.orchestrator import MODE_CONFIDENCE_TARGET, orchestrate
from app.agents.planner import normalize_text, planner_agent
from app.agents.redteam import redteam_agent
from app.agents.search import SearchClient
from app.agents.summarizer import summarizer_agent
from app.agents.synthesizer import synthesizer_agent
from app.agents.verifier import verify_facts
from app.core.llm import LLMClient
from app.core.confidence import compute_confidence
from app.core.contradictions import find_contradictions
from app.core.degradation import take_fallbacks
from app.core.decision import build_decision_layer
from app.core import depth_controller
from app.core.isolation import AgentContext, build_contexts
from app.core.logging import get_logger

logger = get_logger(__name__)


class ResearchState(TypedDict, total=False):
    query: str
    sub_questions: List[Any]
    search_results: List[Dict[str, str]]
    facts: List[Dict[str, Any]]
    critique: Dict[str, Any]
    critique_feedback: str
    iteration: int
    max_iterations: int
    final_report: str
    synthesized_answer: str
    confidence: float
    orchestration: Dict[str, Any]
    deep_research: bool
    verification_stats: Dict[str, Any]
    answer_support: Dict[str, Any]
    confidence_breakdown: Dict[str, Any]
    confidence_history: List[float]
    contradictions: List[Dict[str, Any]]
    mode: str
    decision_options: List[Dict[str, Any]]
    redteam: Dict[str, Any]
    # Wave execution (Feature 03): plan shape + per-pass wave results.
    execution_waves: List[List[str]]
    wave_report: List[Dict[str, Any]]
    # Citation validation v2: live URL health of the emitted answer's legend.
    citation_health: Dict[str, Any]
    # Intent classification (understand-before-searching): ambiguity, senses,
    # domain and explanation level, resolved before planning. Plus the raw
    # grounding search snippets, shared by intent and the planner.
    intent: Dict[str, Any]
    context_snippets: List[str]
    # Answer quality gate: five-axis 0-100 score of the delivered report.
    quality: Dict[str, Any]
    # Answer-first outline (GPT Researcher adaptation): the section shape the
    # writer targeted (broad flag + section list), and whether section-wise
    # synthesis was used. Declared on state so LangGraph carries them to the
    # route for streaming.
    outline: Dict[str, Any]
    section_wise: bool
    # Evidence grades (Step 5): A/B/C/D counts over the verified fact pool.
    evidence_distribution: Dict[str, int]
    # Research-loop: whether a counter-evidence query has actually been issued.
    counter_evidence_attempted: bool
    # Fix A: claim-specific queries that seek a NEW publisher for uncorroborated
    # claims. Computed in critic_node, executed directly in search_node (they
    # do not depend on the planner model rephrasing them).
    corroboration_queries: List[str]
    # Corroboration ACQUISITION: per-claim attempt state (normalized claim ->
    # {claim, domains_queried, query_keys, attempts}), threaded through state so
    # no module-level mutable registry is needed. A claim stops being re-queried
    # once its attempt budget is spent (it stays in the gap set as a limitation).
    corroboration_registry: Dict[str, Dict[str, Any]]
    # Fix B: hard per-run cap on expansion search passes actually issued.
    expansion_passes: int


class PlannerUpdate(TypedDict):
    sub_questions: List[str]
    execution_waves: List[List[str]]


class IntentUpdate(TypedDict):
    intent: Dict[str, Any]
    context_snippets: List[str]


class SearchUpdate(TypedDict, total=False):
    search_results: List[Dict[str, str]]
    counter_evidence_attempted: bool
    expansion_passes: int
    facts: List[Dict[str, Any]]


class SummarizerUpdate(TypedDict):
    facts: List[Dict[str, Any]]
    wave_report: List[Dict[str, Any]]


class VerifierUpdate(TypedDict):
    facts: List[Dict[str, Any]]
    verification_stats: Dict[str, Any]


class CriticUpdate(TypedDict):
    critique: Dict[str, Any]
    iteration: int
    confidence: float
    critique_feedback: str
    confidence_breakdown: Dict[str, Any]
    contradictions: List[Dict[str, Any]]
    redteam: Dict[str, Any]
    facts: List[Dict[str, Any]]
    counter_evidence_attempted: bool
    corroboration_registry: Dict[str, Dict[str, Any]]


class SynthesizerUpdate(TypedDict):
    synthesized_answer: str
    answer_support: Dict[str, Any]
    citation_health: Dict[str, Any]
    quality: Dict[str, Any]
    evidence_distribution: Dict[str, int]
    # Answer-first outline: the section shape the writer targeted (broad flag
    # + section list) and whether the section-wise path was used. Rides in
    # state so the existing final_report event can surface it.
    outline: Dict[str, Any]
    section_wise: bool


class FinalizeUpdate(TypedDict):
    final_report: str
    decision_options: List[Dict[str, Any]]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_question_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(item.get("question", "")).strip()
    return ""


def _merge_questions(existing: List[Any], new: List[Any]) -> List[Any]:
    """Append-only plan growth for expansion passes: keep every researched
    question, add genuinely new ones with continuing ids. Prevents the
    full-replan pattern where expansion discards the working plan."""
    seen = {normalize_text(_extract_question_text(q)) for q in existing or []} - {""}
    merged = list(existing or [])
    used_ids = [int(q.get("id", 0)) for q in merged if isinstance(q, dict)]
    next_id = max(used_ids) if used_ids else 0
    for item in new or []:
        text = normalize_text(_extract_question_text(item))
        if not text or text in seen:
            continue
        seen.add(text)
        next_id += 1
        merged.append({**item, "id": next_id} if isinstance(item, dict) else item)
    return merged


def _unanswered_questions(sub_questions: List[Any], search_results: List[Any]) -> List[str]:
    """Question texts with no results yet — expansion passes search only
    these instead of re-running the whole plan. NOTE: resume-rebuilt
    results lack sub_question keys, so a post-resume expansion re-searches
    once (safe fallback, not a loop — fresh results carry the key)."""
    answered = set()
    for r in search_results or []:
        if isinstance(r, dict):
            q = normalize_text(str(r.get("sub_question", "")))
            if q:
                answered.add(q)
    return [
        text for text in (_extract_question_text(i) for i in sub_questions or [])
        if text and normalize_text(text) not in answered
    ]


# Nodes traversed per research pass after the first (planner→search→
# summarizer→verifier→critic) plus the intent/planner/search/summarizer/
# verifier/critic entry and the synthesizer→finalize tail. A pass budget is
# converted to a LangGraph superstep budget so a legitimate deep run cannot
# trip LangGraph's default recursion limit before route_after_critic's
# ceiling is ever reached.
_NODES_PER_PASS = 5
_GRAPH_ENTRY_AND_TAIL = 8


def graph_recursion_limit(state: ResearchState, extra: int = 4) -> int:
    """LangGraph superstep budget for a run.

    LangGraph's default recursion limit is 25, which a 5-pass deep run
    exceeds (intent + 5×(planner/search/summarizer/verifier/critic) +
    synthesizer/finalize ≈ 26+). Without this, the run aborts with
    GraphRecursionError before the routing-level iteration ceiling can
    finalize it. The budget is derived from the run's own ceiling plus a
    small margin, so a misconfigured `max_iterations` still cannot loop
    unbounded — route_after_critic is the actual stop, this is headroom.
    """
    max_iterations = max(1, int(state.get("max_iterations", 3) or 3))
    return _GRAPH_ENTRY_AND_TAIL + _NODES_PER_PASS * max_iterations + max(0, int(extra))


def build_initial_state(
    query: str,
    max_iterations: int,
    deep_research: bool = False,
    max_parallel_agents: int = 3,
    mode: str = "standard",
) -> ResearchState:
    """Build the run's initial state.

    When `mode` is a valid preset (3.7), it overrides the raw parameters
    with its (max_agents, max_iterations, deep_research) tuple.
    """
    from app.agents.orchestrator import MODE_PRESETS, scaled_max_iterations

    preset = MODE_PRESETS.get(mode)
    if preset is not None:
        # A mode preset sets iterations explicitly — do NOT apply the
        # max(3, ...) floor (GAP-8) or quick mode would be no quicker.
        max_iterations = preset["max_iterations"]
        deep_research = preset["deep_research"]
        # The preset's agent cap REPLACES the setting default: deep and
        # executive are the only modes allowed to exceed MAX_PARALLEL_AGENTS
        # (vision §28: explicit opt-in via mode selection + governor check).
        max_parallel_agents = preset["max_agents"]
        effective_max_iterations = int(max_iterations)
    else:
        effective_max_iterations = max(3, int(max_iterations))
    plan = orchestrate(query, max_parallel_agents=max_parallel_agents,
                       deep_research=deep_research, mode=mode)
    # Fix B.1 — scale the deep/executive iteration budget to the map size now
    # that `orchestrate` has set target_agents. Bounded by scaled_max_iterations
    # (one pass per ~2 contracts, floored at 5); quick/standard unchanged.
    if preset is not None:
        effective_max_iterations = scaled_max_iterations(
            mode, plan.target_agents
        )
    targets = plan.targets.to_dict() if plan.targets is not None else {}
    return {
        "query": query,
        "sub_questions": [],
        "search_results": [],
        "facts": [],
        "critique": {},
        "critique_feedback": "",
        "iteration": 0,
        "max_iterations": effective_max_iterations,
        "final_report": "",
        "synthesized_answer": "",
        "confidence": 0.0,
        "orchestration": {
            "complexity_score": plan.complexity.score,
            "complexity_level": plan.complexity.level,
            "query_type": plan.complexity.query_type,
            "target_agents": plan.target_agents,
            "max_parallel_agents": plan.max_parallel_agents,
            "clamped": plan.clamped,
            "deep_research": plan.deep_research,
            "notes": plan.notes,
            # v3 plan targets: hard requirements the planner must honour.
            "required_axes": list(targets.get("required_axes", []) or []),
            "target_sub_questions": int(targets.get("sub_questions", 0) or 0),
            "min_sources_per_axis": int(targets.get("min_sources_per_axis", 0) or 0),
        },
        "deep_research": plan.deep_research,
        "confidence_history": [],
        "mode": mode if preset is not None else "standard",
        "intent": {},
        "context_snippets": [],
    }


def _prepare_supporting_evidence(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    quality = filter_facts_by_domain(facts)
    deduped = dedupe_semantic_facts(quality)
    ranked = sorted(deduped, key=lambda x: _safe_float(x.get("confidence", 0.0)), reverse=True)

    # Keep evidence diverse by preferring distinct source domains.
    seen_domains = set()
    diverse: List[Dict[str, Any]] = []
    for item in ranked:
        source = str(item.get("source", ""))
        domain = source.split("/")[2].lower() if source.startswith("http") and "/" in source else source.lower()
        if domain in seen_domains:
            continue
        seen_domains.add(domain)
        diverse.append(item)
        if len(diverse) >= 10:
            break

    return diverse if diverse else ranked[:10]


def _verified_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Facts that passed verification; if verification never ran, treat all as usable."""
    if not facts:
        return []
    if not any("verified" in f for f in facts):
        return facts
    return [f for f in facts if f.get("verified")]


def _evidence_gaps_remain(state: ResearchState) -> bool:
    """Measured evidence deficiencies that should outrank a critic's "enough".

    Returns True when the fact pool still contains a single-source
    quantitative/definitional claim that needs independent corroboration, or
    an unresolved contradiction. Returns False (critic wins, behavior
    unchanged) when the pool is empty/ungradeable or grading fails — the gate
    only ever ADDS research, never blocks an otherwise-healthy finalize.
    """
    facts = [f for f in state.get("facts", []) or [] if isinstance(f, dict)]
    if not facts:
        return False
    try:
        from app.core.evidence_grade import grade_facts

        graded = grade_facts(facts, contradictions=state.get("contradictions") or [])
    except Exception as exc:  # a grading bug must never change routing
        logger.warning("evidence_gate_grading_failed", error=str(exc), exc_info=exc)
        return False
    for g in graded:
        ev = g.get("evidence") if isinstance(g, dict) else None
        if not isinstance(ev, dict):
            continue
        if int(ev.get("contradiction_count", 0) or 0) > 0:
            return True
        if ev.get("needs_corroboration"):
            return True
    return False


def _claim_terms(claim: str, limit: int = 8) -> str:
    """Content tokens from a claim, most informative first — the subject the
    corroboration query should target. Deterministic and LLM-free."""
    import re

    stop = {
        "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with",
        "is", "are", "was", "were", "be", "by", "at", "from", "that", "this",
        "it", "its", "as", "than", "about", "over", "per", "will", "has",
        "have", "had", "as", "which", "there", "their", "they", "been",
    }
    tokens = re.findall(r"[A-Za-z0-9%$][A-Za-z0-9%$.\-]*", claim or "")
    out: List[str] = []
    for tok in tokens:
        low = tok.lower()
        if low in stop or len(low) < 3:
            continue
        if tok not in out:
            out.append(tok)
        if len(out) >= limit:
            break
    return " ".join(out)


def _summary_claim_texts(state: ResearchState) -> List[str]:
    """Claim texts the executive summary / key findings will draw on.

    Budget targeting prefers these claims because they are the report's
    headline: corroborating a fact the summary states changes the answer, while
    corroborating a peripheral remark does not. Sourced from the state's own
    synthesized answer / critique when present; empty when synthesis has not run
    yet (expansions before the first synthesis simply rank on the other
    signals). Deterministic and total.
    """
    texts: List[str] = []
    answer = str(state.get("synthesized_answer", "") or "")
    if not answer:
        return texts
    # Best-effort: the summary's own sentences are its claims. Kept bounded so
    # matching stays cheap; never parsed with an LLM.
    for raw in re.split(r"[\n.!?]+", answer):
        line = raw.strip().lstrip("-*# ").strip()
        if 20 <= len(line) <= 200:
            texts.append(line)
        if len(texts) >= 12:
            break
    return texts


def _corroboration_queries(
    state: ResearchState, limit: int = 3, settings: Any | None = None
) -> tuple[List[str], Dict[str, Dict[str, Any]]]:
    """CLAIM-SPECIFIC procurement queries that seek a DIFFERENT publisher.

    The live deep run measured corroboration perfectly and then never went
    looking for it: 0 claims reached two independent registrable domains while
    121 needed corroboration. Measurement without procurement is a dead end.

    For each important single-publisher claim this builds a query that
    (a) targets the claim's own terms, (b) explicitly EXCLUDES the current
    publisher with `-site:<registrable-domain>`, and (c) targets the
    authoritative publisher registry (site:gov/edu/int + named agencies,
    journals and datasets) where an independent corroborating source is most
    likely to live.

    Per-claim attempt state is threaded through `corroboration_registry`
    (keyed by normalized claim) so a claim is never re-queried identically
    beyond `max_corroboration_attempts` and so the caller can see which
    domains were already targeted. Deterministic fallback: if grading fails,
    returns ([], {}) — never invent queries.
    """
    facts = [f for f in state.get("facts", []) or [] if isinstance(f, dict)]
    if not facts:
        return [], {}
    try:
        from app.core.evidence_grade import grade_facts, registrable_domain
        from app.agents.sources import build_corroboration_query
    except Exception as exc:
        logger.warning("corroboration_grading_failed", error=str(exc), exc_info=exc)
        return [], {}

    try:
        graded = grade_facts(facts, contradictions=state.get("contradictions") or [])
    except Exception as exc:
        logger.warning("corroboration_grading_failed", error=str(exc), exc_info=exc)
        return [], {}

    # Budget allocation (workstream B): when there are more single-source
    # claims than the pass can query, IMPACT decides which ones get the scarce
    # queries — quantitative claims, claims the executive summary uses, and
    # high-corroboration-need claims first. `rank_completion_targets` is
    # deterministic and returns [] on any grading failure, so the loop below
    # keeps its historical order in that case.
    target_order: Dict[str, Dict[str, Any]] = {}
    try:
        from app.core.evidence_completion import rank_completion_targets

        summary_claims = _summary_claim_texts(state)
        for passed in rank_completion_targets(
            facts, state.get("contradictions") or [], summary_claims=summary_claims
        ):
            target_order[normalize_text(str(passed.get("claim", "")))] = passed
    except Exception as exc:
        logger.warning("completion_ranking_failed", error=str(exc), exc_info=exc)
        target_order = {}

    settings = settings if settings is not None else get_settings_safe()
    max_attempts = max(1, int(getattr(settings, "max_corroboration_attempts", 2) or 2))
    registry: Dict[str, Dict[str, Any]] = {
        str(k): dict(v)
        for k, v in (state.get("corroboration_registry") or {}).items()
        if isinstance(v, dict)
    }

    # Highest-impact claims first; any graded claim not in the ranked set keeps
    # its original relative order after the ranked targets.
    ordered = sorted(
        graded,
        key=lambda g: (
            0
            if normalize_text(
                str(((g.get("evidence") or {}) if isinstance(g, dict) else {}).get("claim", ""))
            )
            in target_order
            else 1
        ),
    )

    queries: List[str] = []
    for g in ordered:
        ev = g.get("evidence") if isinstance(g, dict) else None
        if not isinstance(ev, dict) or not ev.get("needs_corroboration"):
            continue
        claim = str(ev.get("claim", "")).strip()
        if not claim:
            continue
        key = normalize_text(claim)
        entry = registry.setdefault(
            key, {"claim": claim, "domains_queried": [], "query_keys": [], "attempts": 0}
        )
        # Hard wall: a claim whose attempt budget is spent stays in the gap set
        # (so it is recorded as a limitation) but is no longer re-queried.
        try:
            attempts = int(entry.get("attempts", 0) or 0)
        except (TypeError, ValueError):
            attempts = 0
        if attempts >= max_attempts:
            continue
        domain = registrable_domain(str(ev.get("domain", "") or ev.get("source", "")))
        terms = _claim_terms(claim)
        if not terms:
            continue
        # Rotate targeted hosts per attempt so a second pass is genuinely new
        # procurement, not a repeat of the first scoped query.
        query = build_corroboration_query(
            terms,
            exclude_domain=domain,
            quantitative=bool(ev.get("has_numbers")),
            attempt=attempts,
        )
        if not query:
            continue
        query_key = normalize_text(query)
        if query_key in (entry.get("query_keys") or []):
            # Identical query text is never re-issued — even with a larger
            # attempt budget, a rotation collision must not double-spend.
            entry["attempts"] = max_attempts
            continue
        queries.append(query)
        entry["attempts"] = attempts + 1
        entry["query_keys"] = [*(entry.get("query_keys") or []), query_key]
        if domain and domain not in (entry.get("domains_queried") or []):
            entry["domains_queried"] = [*(entry.get("domains_queried") or []), domain]
        if len(queries) >= max(1, limit):
            break
    return queries, registry


def get_settings_safe():
    """Settings for deterministic-fallback helpers; never raises."""
    try:
        from app.core.config import get_settings

        return get_settings()
    except Exception as exc:  # settings failure must not break query generation
        logger.warning("settings_lookup_failed", error=str(exc), exc_info=exc)
        return None


def _acquire_corroboration(
    facts: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
    contradictions: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Attach NEW-publisher supporting pages to single-source claims.

    For every fact whose graded record still needs corroboration, find search
    results from a registrable domain the fact does not already hold whose text
    supports the claim at/above the corroboration similarity band, and attach
    them through `apply_corroboration` (the only sanctioned write path, which
    rejects same-publisher URLs). Facts are mutated in place and the same list
    is returned; deterministic, LLM-free, and total — any grading failure
    leaves the pool untouched rather than breaking the run.
    """
    items = [f for f in facts or [] if isinstance(f, dict)]
    candidates = [r for r in search_results or [] if isinstance(r, dict)]
    if not items or not candidates:
        return list(facts or [])
    try:
        from app.core.evidence_grade import (
            apply_corroboration,
            find_corroborating_sources,
            grade_facts,
        )
    except Exception as exc:
        logger.warning("corroboration_acquisition_import_failed", error=str(exc), exc_info=exc)
        return list(facts or [])

    try:
        graded = grade_facts(items, contradictions=contradictions or [])
    except Exception as exc:
        logger.warning("corroboration_acquisition_grading_failed", error=str(exc), exc_info=exc)
        return list(facts or [])

    by_claim = {
        normalize_text(str(g.get("claim", ""))): g
        for g in graded
        if isinstance(g, dict)
    }
    attached = 0
    settings = get_settings_safe()
    threshold = float(getattr(settings, "corroboration_similarity", 0.55) or 0.55)
    for fact in items:
        claim = str(fact.get("claim", "") or "").strip()
        if not claim:
            continue
        record = by_claim.get(normalize_text(claim))
        ev = record.get("evidence") if isinstance(record, dict) else None
        if not isinstance(ev, dict) or not ev.get("needs_corroboration"):
            continue
        existing = [
            str(u) for u in (fact.get("corroborating_sources") or []) if str(u).strip()
        ]
        if fact.get("source"):
            existing.append(str(fact["source"]))
        matches = find_corroborating_sources(
            claim, candidates, existing, threshold=threshold
        )
        for url in matches:
            if apply_corroboration(fact, url):
                attached += 1
    if attached:
        logger.info("corroboration_acquired", attachments=attached)
    return list(facts or [])


def _counter_evidence_queries(state: ResearchState, limit: int = 2) -> List[str]:
    """Targeted disagreement searches for the weakest claims in the pool.

    The brief is explicit: do not only search for support. For every
    single-source or contradicted claim, propose a query that seeks
    counter-evidence, limitations, or credible opposing views. These ride the
    existing critic `improved_queries` channel, so the normal expansion loop
    researches them — no new pipeline stage.
    """
    facts = [f for f in state.get("facts", []) or [] if isinstance(f, dict)]
    if not facts:
        return []
    try:
        from app.core.evidence_grade import grade_facts

        graded = grade_facts(facts, contradictions=state.get("contradictions") or [])
    except Exception as exc:
        logger.warning("counter_evidence_grading_failed", error=str(exc), exc_info=exc)
        return []
    queries: List[str] = []
    for g in graded:
        ev = g.get("evidence") if isinstance(g, dict) else None
        if not isinstance(ev, dict):
            continue
        claim = str(ev.get("claim", "")).strip()
        if not claim:
            continue
        if int(ev.get("contradiction_count", 0) or 0) > 0:
            queries.append(f"{claim[:140]} conflicting evidence OR disagreement")
        elif ev.get("needs_corroboration"):
            # Independent corroboration, aimed at PRIMARY publishers rather
            # than more commentary: the whole deficiency is that only one
            # publisher stands behind an important claim.
            queries.append(
                f"{claim[:120]} independent corroboration official data "
                "OR government report OR peer-reviewed study"
            )
        if len(queries) >= max(1, limit):
            break
    return queries


def _attach_evidence(
    facts: List[Dict[str, Any]],
    contradictions: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Annotate state facts in place with the claim-level EvidenceRecord.

    Requirement 6: a graded fact reaching synthesis must carry
    claim→source→verification→independence→corroboration→contradiction→
    confidence. `grade_facts` produces those records; without this the records
    lived only inside the confidence/synthesizer computations and the final
    report could not rely on them. Additive: existing fact keys are preserved
    and the record rides under `evidence` / `evidence_grade`.
    """
    try:
        from app.core.evidence_grade import grade_facts

        graded = grade_facts(facts, contradictions=contradictions or [])
    except Exception as exc:  # a grading bug must never break a run
        logger.warning("attach_evidence_failed", error=str(exc), exc_info=exc)
        return list(facts or [])
    by_claim = {
        str(g.get("claim", "")): g.get("evidence")
        for g in graded
        if isinstance(g, dict)
    }
    out: List[Dict[str, Any]] = []
    for fact in facts or []:
        if not isinstance(fact, dict):
            continue
        ev = by_claim.get(str(fact.get("claim", "")))
        if ev and "evidence" not in fact:
            enriched = dict(fact)
            enriched["evidence"] = ev
            enriched["evidence_grade"] = str(ev.get("grade", ""))
            out.append(enriched)
        else:
            out.append(dict(fact))
    return out



# Minimum semantic similarity for a fact to count as part of the query's own
# evidence pool. Calibrated on live off-topic bleed: on-topic claims for a
# short query scored 0.07-0.24, unrelated domains (Iran nuclear, NBER
# clientelism) scored 0.00-0.02. Well below the on-topic band and above the
# noise floor.
IN_SCOPE_SIMILARITY = 0.04


def _query_inscope_facts(
    facts: List[Dict[str, Any]], query: str, senses: Sequence[str] = ()
) -> List[Dict[str, Any]]:
    """The query's OWN evidence pool: facts topically about the query.

    Live reports leaked unrelated claims into the Limitations list (Iran's
    nuclear programme, an NBER paper on clientelism — on a "What is a
    transformer?" query) because this pool was the raw `state["facts"]`: every
    sub-question's evidence, including off-domain material a search pass
    happened to fetch. Limitations and the evidence scoring must describe THIS
    query's evidence, not every domain the run touched.

    Reuses the EXISTING relevance machinery rather than adding a new system:
    word overlap (`claim_query_overlap` + `MIN_QUERY_OVERLAP`, the summarizer's
    own filter) OR the shared TF-IDF hybrid engine's similarity
    (`app.core.semantic`, the same engine section ranking and corroboration
    trust). A fact is in scope when it clears either bar against the query or
    any of the query's disambiguation senses (an ambiguous term's second
    meaning is genuinely in scope). The filter is all-or-nothing-safe: if it
    would leave nothing, the original pool is returned unchanged so a
    limitations section is never starved (AGENTS.md 4.7).
    """
    if not facts:
        return []
    try:
        from app.agents.evidence_utils import MIN_QUERY_OVERLAP, claim_query_overlap

        scope_texts = [str(t) for t in (query, *senses) if str(t or "").strip()]
        if not scope_texts:
            return facts
        claims = [str(f.get("claim", "") or "") for f in facts]
        # Semantic similarity is the stronger signal for a SHORT query, where
        # word overlap is stopword noise ("What is a transformer?" -> "a").
        try:
            from app.core.semantic import rank_by_similarity

            semantic: List[float] = [0.0] * len(claims)
            for text in scope_texts:
                scores = rank_by_similarity(text, claims)
                semantic = [max(a, float(b)) for a, b in zip(semantic, scores)]
            inscope = [f for i, f in enumerate(facts) if semantic[i] >= IN_SCOPE_SIMILARITY]
        except Exception as exc:  # noqa: BLE001 - fall back to word overlap
            logger.warning("query_inscope_semantic_failed", error=str(exc), exc_info=exc)
            inscope = [
                f
                for i, f in enumerate(facts)
                if max(
                    (claim_query_overlap(t, claims[i]) for t in scope_texts),
                    default=0.0,
                )
                >= MIN_QUERY_OVERLAP
            ]
        return inscope or facts
    except Exception as exc:  # noqa: BLE001 - isolation must never break a run
        logger.warning("query_inscope_filter_failed", error=str(exc), exc_info=exc)
        return facts


def _measured_coverage_gaps(state: ResearchState) -> List[str]:
    """Human-readable evidence gaps for the mandatory Limitations section.

    Deterministic, from the graded pool (single-source/contradicted claims) and
    the corroboration registry (claims whose procurement budget is spent and
    which remain uncorroborated). Only the query's OWN evidence pool is graded,
    so an unrelated domain's facts cannot surface as this report's limitations.
    Empty on any failure — a limitations section with no measured gap is honest,
    a crash is not.
    """
    facts = [f for f in state.get("facts", []) or [] if isinstance(f, dict)]
    if facts:
        intent = state.get("intent") or {}
        senses = [
            str(s.get("label", "") or "")
            for s in (intent.get("senses") or [])
            if isinstance(s, dict)
        ]
        facts = _query_inscope_facts(facts, str(state.get("query", "") or ""), senses)
    gaps: List[str] = []
    if facts:
        try:
            from app.core.evidence_grade import coverage_gaps_from_records, grade_claim

            records = [grade_claim(f, contradictions=state.get("contradictions") or []) for f in facts]
            gaps.extend(coverage_gaps_from_records(records))
        except Exception as exc:
            logger.warning("coverage_gaps_failed", error=str(exc), exc_info=exc)
    return gaps[:8]


def build_markdown_report(
    state: ResearchState,
    decision_options: List[Dict[str, Any]] | None = None,
) -> str:
    facts = _prepare_supporting_evidence(state.get("facts", []))
    critique = state.get("critique", {})
    confidence = float(state.get("confidence", 0.0))
    synthesized = str(state.get("synthesized_answer", "")).strip()

    evidence_lines = [
        f"- {item.get('claim', '').strip()} ({item.get('source', '').strip()})"
        for item in facts[:10]
        if item.get("claim") and item.get("source")
    ]
    if not evidence_lines:
        evidence_lines = ["- No robust findings were extracted from available high-quality sources."]

    final_answer = synthesized or "A confident synthesis could not be generated from the available evidence."

    is_sufficient = bool(critique.get("is_sufficient", False))
    critique_reason = str(critique.get("reason", "Limited evidence quality or coverage."))
    limitations: List[str] = []
    if not is_sufficient:
        limitations.append(critique_reason)
    limitations.extend(
        [
            "Free-tier APIs may rate limit or return shallow snippets.",
            "No paywalled or private databases were accessed.",
        ]
    )

    improved_queries = critique.get("improved_queries", [])
    if not is_sufficient and isinstance(improved_queries, list):
        for item in improved_queries[:2]:
            if isinstance(item, str) and item.strip():
                limitations.append(f"Potential follow-up search: {item.strip()}")

    # Dynamic Research Depth (2.8): name an early stop on marginal gain.
    early_stop_note = depth_controller.stop_reason(state)
    if early_stop_note:
        limitations.append(early_stop_note)

    # Citation validation v2: dead or partially-unsupported cited sources.
    try:
        from app.agents.citation_check import citation_health_note

        health_note = citation_health_note(state.get("citation_health"))
        if health_note:
            limitations.append(health_note)
    except Exception as exc:
        logger.warning("citation_health_note_failed", error=str(exc), exc_info=exc)

    limitations = [
        *limitations,
        "Confidence is estimated from evidence quality and critic assessment, not formal verification.",
    ]

    lines = [
        "# Final Answer",
        final_answer,
        "",
        "# Supporting Evidence",
        *evidence_lines,
        "",
    ]

    # Contradiction Engine (3.2): surface source conflicts explicitly. Fix C:
    # a resolved conflict (period/scope/metric difference) is recorded with its
    # explanation so the report is honest about the spread, while only
    # unresolved conflicts read as open disagreements.
    contradictions = state.get("contradictions", [])
    if contradictions:
        contradiction_lines = []
        for c in contradictions[:5]:
            contradiction_lines.append(
                f"- \"{c.get('claim_a', '')[:140]}\" ({c.get('source_a', '')})"
            )
            contradiction_lines.append(
                f"  conflicts with \"{c.get('claim_b', '')[:140]}\" ({c.get('source_b', '')})"
            )
            if c.get("resolved"):
                contradiction_lines.append(
                    f"  RESOLVED: {c.get('resolution', '')}"
                )
        lines.extend(["# Contradictions", *contradiction_lines, ""])

    # Decision Intelligence Layer (3.5, Feature 18): options → recommendation
    # → rationale, structurally SEPARATE from the findings above. Factual
    # queries produce no options — no section rather than a placeholder.
    if decision_options is None:
        decision_options = build_decision_layer(state)
    if decision_options:
        decision_lines = []
        for o in decision_options:
            marker = " (RECOMMENDED)" if o.get("is_recommended") else ""
            decision_lines.append(f"- Option {o.get('option_label', '?')}{marker}: {o.get('description', '')}")
            if o.get("rationale"):
                decision_lines.append(f"  Rationale: {o['rationale']}")
            if o.get("risk_note"):
                decision_lines.append(f"  Risk: {o['risk_note']}")
        lines.extend(["# Decision Layer", *decision_lines, ""])

    # Answer Quality (final editor): the measured five-axis score of THIS
    # report, appended from state so the disclosure cannot be skipped.
    quality = state.get("quality") or {}
    if isinstance(quality, dict) and quality.get("overall") is not None:
        quality_lines = [
            f"Accuracy {quality.get('accuracy', 0)}/100 · "
            f"Relevance {quality.get('relevance', 0)}/100 · "
            f"Evidence {quality.get('evidence', 0)}/100 · "
            f"Clarity {quality.get('clarity', 0)}/100 · "
            f"Reasoning {quality.get('reasoning', 0)}/100",
            f"Overall: {quality.get('overall', 0)}/100 — "
            + ("passed the quality gate." if quality.get("passed")
               else "BELOW THRESHOLD — treat with additional caution."),
        ]
        if not quality.get("passed") and quality.get("failures"):
            quality_lines.append("Gate findings:")
            quality_lines.extend(f"- {f}" for f in quality.get("failures", [])[:5])
        lines.extend(["# Answer Quality", *quality_lines, ""])

    lines.extend([
        "# Limitations",
        *[f"- {item}" for item in limitations],
        "",
        "# Confidence Score",
        f"{confidence:.2f}",
    ])
    return "\n".join(lines)


def create_workflow(llm: LLMClient, search_client: SearchClient, entry_node: str | None = None):
    """Compile the research graph.

    entry_node=None (default): START → intent → planner (full pipeline).
    entry_node="critic": START → critic — used by the resume endpoint (3.3)
    so a failed/timeout run continues from persisted evidence instead of
    re-running intent/planner/search.
    """
    if entry_node is not None and entry_node != "critic":
        raise ValueError(f"unsupported entry_node={entry_node!r} (only 'critic' is supported)")
    graph = StateGraph(ResearchState)

    async def intent_node(state: ResearchState) -> IntentUpdate:
        """Understand the question BEFORE shaping research (vision step 1).

        One grounding search on the raw query serves double duty: it gives the
        intent classifier real-world sense evidence, and the planner its
        terminology grounding (previously the planner ran this search itself
        on the raw query — the exact mechanism that pulled electrical-
        transformer statistics into an ML question's plan). Ambiguity is
        resolved here, never after the evidence is in.
        """
        # Latency: the grounding search and the intent LLM call are
        # independent — run them CONCURRENTLY. The classifier reads the
        # query's own phrasing (the primary signal; the forced-both policy
        # covers definitional ambiguity); the search results still ground
        # the planner, which was their original job.
        async def _context_search() -> List[str]:
            try:
                raw_results = await search_client.run_search([state["query"]])
                return [
                    f"{str(r.get('title', '') or '').strip()}: {str(r.get('snippet', '') or '').strip()[:220]}"
                    for r in (raw_results or [])[:6]
                    if isinstance(r, dict) and (r.get("title") or r.get("snippet"))
                ]
            except Exception as exc:
                logger.warning("planner_context_search_failed", error=str(exc), exc_info=exc)
                return []

        intent_enabled = bool(getattr(llm.settings, "intent_enabled", True))

        async def _classify() -> Dict[str, Any]:
            if intent_enabled:
                return (await classify_intent(llm, state["query"])).to_dict()
            return heuristic_intent(state["query"]).to_dict()

        context_snippets, intent_dict = await asyncio.gather(_context_search(), _classify())
        return {"intent": intent_dict, "context_snippets": context_snippets}

    async def planner_node(state: ResearchState) -> PlannerUpdate:
        existing = state.get("sub_questions", []) or []
        feedback = state.get("critique_feedback", "")
        expanding = bool(existing) and int(state.get("iteration", 0)) > 0
        if expanding:
            already = "; ".join(
                _extract_question_text(q) for q in existing if _extract_question_text(q)
            )
            feedback = (
                feedback
                + "\nAlready researched (do not repeat — only add gap-closing questions): "
                + already
            ).strip()

        # Search-informed planning (gpt-researcher parity): the grounding
        # search on the raw query ran once in intent_node; expansion passes
        # already have critique feedback to aim at and skip the context.
        context_snippets: List[str] = []
        if not expanding:
            context_snippets = [
                str(s) for s in (state.get("context_snippets") or []) if str(s).strip()
            ]

        # Intent (understand-before-searching): senses, domain and explanation
        # level resolved before planning. The planner targets the user's
        # likely meaning instead of whatever the raw query string retrieves.
        intent = state.get("intent") or {}

        # v3 plan targets live on the orchestration dict (build_initial_state).
        orchestration = state.get("orchestration", {})
        if expanding:
            # Per-axis expansion: the axes already under research ARE the
            # required contract. Seeding from the orchestration's static axis
            # list here re-injected the generic canonical axes (definition/
            # evidence/criticism/mechanism) on every expansion pass, which put
            # the boilerplate back alongside the dynamic-planning dimensions the
            # first pass derived. Existing contracts carry those dimensions.
            required_axes = []
            existing_axes = {
                str(item.get("axis", "") or "").strip().lower()
                for item in existing
                if isinstance(item, dict) and str(item.get("axis", "") or "").strip()
            }
            for axis in existing_axes:
                if axis and axis not in required_axes:
                    required_axes.append(axis)
        else:
            required_axes = list(orchestration.get("required_axes", []) or [])
        sub_questions = await planner_agent(
            llm=llm,
            query=state["query"],
            critique_feedback=feedback,
            today=datetime.date.today().isoformat(),
            context_snippets=context_snippets or None,
            intent=intent or None,
            # v3 plan targets from orchestration: the plan is sized and
            # axis-shaped here; the hardware cap below stays as backstop.
            target_count=int(orchestration.get("target_sub_questions", 0) or 0) or None,
            required_axes=required_axes,
            minimum_sources=int(orchestration.get("min_sources_per_axis", 0) or 0) or 2,
        )
        # Hardware guardrail: cap the plan at the orchestrated target agents.
        target = int(orchestration.get("target_agents", 5) or 5)
        if expanding:
            # Per-axis expansion: keep researched history, cap only the NEW
            # additions at target so per-pass load stays bounded.
            merged = _merge_questions(existing, sub_questions)
            added = merged[len(existing):][: max(1, target)]
            sub_questions = [*existing, *added]
        else:
            sub_questions = sub_questions[: max(1, target)]
        # Wave structure (Feature 03): dependency-ordered groups the
        # summarizer executes sequentially, passing earlier-wave findings to
        # dependent contracts. Exposed on state so the UI can show the plan's
        # shape and the benchmark suite can verify wave execution.
        try:
            from app.agents.planner import execution_waves
            waves = execution_waves(sub_questions)
            wave_shape = [
                [q.get("question", "") if isinstance(q, dict) else str(q) for q in wave]
                for wave in waves
            ]
        except Exception:
            wave_shape = [[_extract_question_text(q) for q in sub_questions]]
        logger.info("planner_done", sub_questions=len(sub_questions), expanding=expanding,
                    waves=len(wave_shape))
        return {"sub_questions": sub_questions, "execution_waves": wave_shape}

    async def search_node(state: ResearchState) -> SearchUpdate:
        previous = [r for r in state.get("search_results", []) or [] if isinstance(r, dict)]
        # Per-axis expansion: search only questions with no results yet.
        # Each unanswered parent fans out to its alternate phrasings
        # (variants ride the parent contract, so nothing orphans).
        # Accumulation is explicitly capped (SEARCH_MAX_RESULTS_RETAINED) and
        # the verifier blanks raw content after each pass.
        settings = getattr(search_client, "settings", None)
        fresh: List[Any] = []
        answered = {
            normalize_text(str(r.get("sub_question", ""))) for r in previous
        } - {""}
        by_text = {}
        for item in state.get("sub_questions", []) or []:
            text = _extract_question_text(item)
            if text and text not in by_text:
                by_text[text] = item

        def _question_search_type(text: str) -> str:
            """The contract's search_type steers retrieval (news topic vs
            general). Variants inherit their parent's type — they are
            rephrasings, not new contracts."""
            item = by_text.get(text)
            if isinstance(item, dict):
                return str(item.get("search_type", "") or "")
            return ""

        for text in _unanswered_questions(state.get("sub_questions", []), previous):
            fresh.append((text, _question_search_type(text)))
            item = by_text.get(text)
            if isinstance(item, dict):
                parent_type = _question_search_type(text)
                for v in item.get("variants", []) or []:
                    vs = str(v or "").strip()
                    if vs and normalize_text(vs) not in answered:
                        fresh.append((vs, parent_type))

        # Fix A.3 — corroboration procurement must actually be EXECUTED, not
        # merely stored on the critique. These claim-specific, publisher-
        # excluding queries are injected straight into this pass's search set on
        # expansion passes, independent of whether the planner model chose to
        # turn critique feedback into a contract. Deduped against everything
        # already searched.
        corroboration_to_run: List[str] = []
        if int(state.get("iteration", 0)) > 0:
            for q in state.get("corroboration_queries", []) or []:
                text = str(q or "").strip()
                key = normalize_text(text)
                if not text or not key or key in answered:
                    continue
                corroboration_to_run.append(text)
                answered.add(key)

        cap = max(1, int(getattr(settings, "search_max_queries_per_pass", 8) or 8))
        # Corroboration queries get priority within the per-pass query cap:
        # they are the pass's reason for existing when a claim needs a new
        # publisher, so an oversized plan cannot crowd them out.
        room = max(0, cap - len(corroboration_to_run))
        fresh = fresh[:room]
        fresh.extend((q, "general") for q in corroboration_to_run)
        if not fresh:
            if previous:
                return {"search_results": previous}
            fallback = state.get("query", "").strip()
            if not fallback:
                return {"search_results": previous}
            fresh = [(fallback, "")]
        # Fix B.3 — hard per-run expansion wall. Count the expansion passes and
        # the extra searches they issue; once either cap is hit, stop issuing
        # NEW expansion searches (the evidence already gathered still flows on).
        prior_passes = int(state.get("expansion_passes", 0) or 0)
        is_expansion = int(state.get("iteration", 0)) > 0
        max_passes = max(1, int(getattr(settings, "max_expansion_passes", 12) or 12))
        max_searches = max(1, int(getattr(settings, "max_expansion_searches", 48) or 48))
        expansion_passes = prior_passes + (1 if is_expansion else 0)
        budget_stop = is_expansion and (
            expansion_passes > max_passes
            or prior_passes * cap + len(fresh) > max_searches
        )
        if budget_stop:
            logger.info(
                "expansion_wall_reached",
                expansion_passes=expansion_passes,
                max_expansion_passes=max_passes,
                iteration=int(state.get("iteration", 0)),
            )
            return {
                "search_results": previous,
                "expansion_passes": prior_passes,
            }
        results = await search_client.run_search(fresh)
        # Track whether a disagreement-seeking/corroboration query was actually
        # issued — the stopping redesign requires counter-evidence to have been
        # attempted, and this is the only point that knows what really ran.
        counter_markers = (
            "conflicting evidence", "disagreement", "independent corroboration",
            "counter-evidence", "counter evidence", "verification",
            "independent source", "official report", "-site:",
        )
        issued_counter = any(
            any(marker in str(text).lower() for marker in counter_markers)
            for text, _ in fresh
        )
        seen_urls = {r.get("url") for r in previous if r.get("url")}
        merged = [*previous, *(r for r in results if r.get("url") not in seen_urls)]
        # Hard memory bound: expansion passes append results forever, so a
        # long deep run could otherwise accumulate hundreds of result dicts
        # in state (raw content is blanked after verification, but the list
        # and its metadata still grow). Keep the NEWEST results — the ones
        # the current pass's summarizer needs — and drop the oldest beyond
        # the cap. Verification already ran on older passes, so nothing
        # downstream loses content it still needs.
        cap_results = max(10, int(getattr(settings, "search_max_results_retained", 80) or 80))
        if len(merged) > cap_results:
            dropped = len(merged) - cap_results
            merged = merged[dropped:]
            logger.info("search_results_capped", dropped=dropped, retained=len(merged))
        logger.info("search_done", results=len(merged), fresh=len(fresh))
        # Corroboration LINKING: fresh expansion results are matched back to
        # pending needs_corroboration facts HERE, at the moment the new pages
        # exist. Previously the results were merely merged into state (and the
        # measurement primitives existed) but nothing ever matched a fresh
        # result to the claim that needed it, so corroborating_sources stayed
        # at one publisher and corroborated_ge2 was always 0. Annotate copies
        # of the facts so no fact is dropped and no same-publisher URL can
        # raise the count (find_corroborating_sources/apply_corroboration
        # enforce registrable-domain independence).
        updated_facts: List[Dict[str, Any]] = [
            f for f in (state.get("facts", []) or []) if isinstance(f, dict)
        ]
        matched_corroboration = 0
        try:
            from app.core.evidence_grade import (
                apply_corroboration,
                find_corroborating_sources,
                grade_claim,
                registrable_domain,
            )

            existing_domains = {
                registrable_domain(str(f.get("source", "") or ""))
                for f in updated_facts
            }
            new_publishers = sum(
                1
                for r in results
                if isinstance(r, dict)
                and registrable_domain(str(r.get("url", "") or ""))
                and registrable_domain(str(r.get("url", "") or ""))
                not in existing_domains
            )
            settings = getattr(search_client, "settings", None)
            threshold = float(getattr(settings, "corroboration_similarity", 0.55) or 0.55)
            annotated: List[Dict[str, Any]] = []
            for fact in updated_facts:
                claim = str(fact.get("claim", "") or "").strip()
                source = str(fact.get("source", "") or "").strip()
                if not claim or not source:
                    annotated.append(fact)
                    continue
                record = grade_claim(fact)
                if not (record.needs_corroboration or record.corroboration_count < 2):
                    annotated.append(fact)
                    continue
                existing_urls = [
                    str(u) for u in (fact.get("corroborating_sources") or [])
                    if str(u).strip()
                ]
                existing_urls.append(source)
                matches = find_corroborating_sources(
                    claim, results, existing_urls, threshold=threshold
                )
                gained = 0
                copy = dict(fact)
                for url in matches:
                    if apply_corroboration(copy, url):
                        gained += 1
                if gained:
                    matched_corroboration += 1
                    annotated.append(copy)
                else:
                    annotated.append(fact)
            updated_facts = annotated
            logger.info(
                "corroboration_funnel",
                queries_issued=len(fresh),
                results_returned=len(results),
                new_publishers=new_publishers,
                matched_corroboration=matched_corroboration,
            )
        except Exception as exc:
            logger.warning(
                "corroboration_linking_failed", error=str(exc), exc_info=exc
            )
            updated_facts = [
                f for f in (state.get("facts", []) or []) if isinstance(f, dict)
            ]

        update: SearchUpdate = {
            "search_results": merged,
            "counter_evidence_attempted": bool(
                state.get("counter_evidence_attempted") or issued_counter
            ),
            "expansion_passes": expansion_passes,
        }
        if matched_corroboration:
            update["facts"] = updated_facts
        return update

    async def summarizer_node(state: ResearchState) -> SummarizerUpdate:
        # Agent Context Isolation (2.9): each sub-question worker sees only
        # its own AgentContext — own contract + own results, never the full
        # ResearchState or another sub-question's raw content.
        contexts = build_contexts(
            sub_questions=state.get("sub_questions", []),
            search_results=state.get("search_results", []),
        )

        # Wave execution (Feature 03): the planner already computes
        # dependency waves — run them in order so dependent contracts
        # ("compare X vs Y" depending on "what is X") extract with the
        # earlier wave's findings as grounding context. Independent members
        # of one wave still run concurrently under the LLM semaphore.
        by_wave: Dict[int, List[AgentContext]] = {}
        for ctx in contexts:
            wave = int(ctx.contract.get("wave", 0) or 0) if isinstance(ctx.contract, dict) else 0
            by_wave.setdefault(max(0, wave), []).append(ctx)
        wave_numbers = sorted(by_wave)

        async def _summarize_context(ctx: AgentContext, prior: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
            if not ctx.own_results:
                return []
            # Specialist routing (3.1): role comes from THIS context's
            # delegation contract — each specialist sees only its own
            # scoped context, never the shared ResearchState (2.9).
            # `prior_findings` is passed ONLY when a dependent wave has
            # prerequisite context — wave-0 calls keep the historical
            # signature shape so duck-typed fakes keep working. Same for
            # `sense` (intent disambiguation): passed only when the plan
            # actually carries one.
            sense = ctx.sense()
            if prior and sense:
                return await summarizer_agent(
                    llm=llm,
                    query=state["query"],
                    search_results=ctx.own_results,
                    specialist_role=ctx.specialist_role(),
                    prior_findings=prior,
                    sense=sense,
                )
            if prior:
                return await summarizer_agent(
                    llm=llm,
                    query=state["query"],
                    search_results=ctx.own_results,
                    specialist_role=ctx.specialist_role(),
                    prior_findings=prior,
                )
            if sense:
                return await summarizer_agent(
                    llm=llm,
                    query=state["query"],
                    search_results=ctx.own_results,
                    specialist_role=ctx.specialist_role(),
                    sense=sense,
                )
            return await summarizer_agent(
                llm=llm,
                query=state["query"],
                search_results=ctx.own_results,
                specialist_role=ctx.specialist_role(),
            )

        wave_report: List[Dict[str, Any]] = []
        accumulated: List[Dict[str, Any]] = []
        for wave in wave_numbers:
            wave_contexts = [c for c in by_wave[wave] if c.own_results]
            results = await asyncio.gather(*(
                _summarize_context(ctx, accumulated if wave > 0 else None)
                for ctx in wave_contexts
            ))
            wave_facts = [fact for facts in results for fact in facts]
            accumulated = [*accumulated, *wave_facts]
            wave_report.append({
                "wave": wave,
                "contracts": len(wave_contexts),
                "facts_extracted": len(wave_facts),
                "with_prerequisites": bool(wave > 0 and accumulated),
            })
            logger.info("wave_done", wave=wave, contracts=len(wave_contexts),
                        facts=len(wave_facts))

        fresh_facts = accumulated
        merged = dedupe_semantic_facts([*state.get("facts", []), *fresh_facts])
        # Corroboration ACQUISITION: attach a NEW publisher's supporting page
        # text to pending single-source claims, deterministically. Measurement
        # (`independent_corroboration`) already existed; without this pass the
        # corroboration searches ran and their results were never matched back
        # to the claims that needed them.
        merged = _acquire_corroboration(merged, state.get("search_results", []))
        logger.info("summarizer_done", fresh_facts=len(fresh_facts), total_facts=len(merged),
                    waves=len(wave_report))
        return {"facts": merged, "wave_report": wave_report}

    async def verifier_node(state: ResearchState) -> VerifierUpdate:
        verified = verify_facts(
            facts=state.get("facts", []),
            search_results=state.get("search_results", []),
        )
        stats = {
            "total": len(verified),
            "verified": sum(1 for f in verified if f.get("verified")),
        }

        # Raw content is discarded once verification completes — verification
        # is the last consumer of full page text (it checks claims against
        # content, not just snippets); nothing downstream (critic, synthesizer,
        # finalize, later iterations) needs the full page. Blank in place: these
        # dict objects are shared with the contexts built in summarizer_node.
        # Before blanking, retain a BOUNDED excerpt: the expansion-pass
        # corroboration linker runs after verification and otherwise has only
        # short snippets to match a pending claim against. The excerpt is a
        # few hundred chars (memory-release rule still holds — the full page is
        # released), falling back to the snippet when content was already empty.
        from app.core.evidence_grade import CORROBORATION_EXCERPT_CHARS

        for result in state.get("search_results", []):
            if not isinstance(result, dict) or "content" not in result:
                continue
            raw = str(result.get("content", "") or "")
            if not raw:
                raw = str(result.get("snippet", "") or "")
            if raw and not result.get("corroboration_excerpt"):
                result["corroboration_excerpt"] = re.sub(
                    r"\s+", " ", raw
                ).strip()[:CORROBORATION_EXCERPT_CHARS]
            result["content"] = ""

        logger.info("verifier_done", **stats)
        return {"facts": verified, "verification_stats": stats}

    async def critic_node(state: ResearchState) -> CriticUpdate:
        next_iteration = int(state.get("iteration", 0)) + 1

        # Contradiction Engine (3.2) computed HERE (not in verifier) so the
        # resume path — which re-enters at critic with persisted facts —
        # still feeds contradictions to the critic and the report.
        contradictions = find_contradictions(state.get("facts", []))
        # Fix C — resolution pass. A temporal or scope difference is an
        # EXPLAINED spread, not a disagreement; it is recorded for the report
        # but must not penalize confidence or drive further expansion. Only
        # genuinely conflicting (same unit/scope/period/metric, different
        # values) entries stay `resolved: false`.
        try:
            from app.core.contradiction_resolution import resolve_contradictions

            contradictions = resolve_contradictions(contradictions)
        except Exception as exc:  # resolution must never break a run
            logger.warning("contradiction_resolution_failed", error=str(exc), exc_info=exc)
        if contradictions:
            logger.info(
                "contradictions_found",
                count=len(contradictions),
                unresolved=sum(1 for c in contradictions if not c.get("resolved")),
            )

        # Red-team review (v3, heuristics only: deterministic, zero LLM
        # cost). Attacks the evidence base every pass; the survival score
        # feeds the critic gate and the findings render in the report.
        # Never fatal: heuristics must not break a run.
        try:
            redteam_state = (
                await redteam_agent(
                    None,
                    state["query"],
                    state.get("facts", []),
                    contradictions=contradictions,
                    use_llm=False,
                )
            ).to_dict()
        except Exception as exc:
            logger.warning("redteam_heuristics_failed", error=str(exc), exc_info=exc)
            redteam_state = {
                "findings": [], "survival_score": 0.6, "survives": True,
                "blocking": [], "targeted_queries": [], "summary": "",
            }

        # The critic's optional gate inputs, wired (they were built and tested
        # but never passed, so the coverage-gap gate could never fire, the
        # prompt never knew which searches already ran, and mode confidence
        # targets were honored only by the depth controller, not the gate).
        searched: List[str] = []
        for q in state.get("sub_questions", []) or []:
            if isinstance(q, dict):
                text = str(q.get("question", "")).strip()
                if text:
                    searched.append(text)
                for v in q.get("variants", []) or []:
                    vs = str(v or "").strip()
                    if vs:
                        searched.append(vs)
        mode_target = MODE_CONFIDENCE_TARGET.get(str(state.get("mode", "") or "standard"))

        critique = await critic_agent(
            llm=llm,
            query=state["query"],
            facts=state.get("facts", []),
            iteration=next_iteration,
            max_iterations=int(state.get("max_iterations", 3)),
            contradictions=contradictions,
            query_type=str(state.get("orchestration", {}).get("query_type", "")),
            redteam_survival=float(redteam_state.get("survival_score", 0.6) or 0.6),
            plan=state.get("sub_questions", []),
            searched_queries=searched,
            confidence_target=mode_target,
            # Quick mode: the iteration ceiling makes the verdict
            # routing-neutral, so the critic runs gates-only (no LLM call)
            # and measured evidence stats stand in for the model verdict —
            # one fewer serial LLM call on the latency-sensitive mode.
            use_llm=str(state.get("mode", "standard")) != "quick",
        )

        # Confidence Engine (Phase 2.4) replaces the inline weighted formula.
        # Degraded stages cap the score: a run whose evidence came from the
        # extractive fallback must not finalize as "High" confidence.
        source_dates = [
            r.get("published_at", "") for r in state.get("search_results", []) or []
            if isinstance(r, dict) and r.get("published_at")
        ]
        breakdown = compute_confidence(
            facts=state.get("facts", []),
            critique=critique,
            iteration=next_iteration,
            max_iterations=int(state.get("max_iterations", 3)),
            source_dates=source_dates,
            degraded=take_fallbacks(),
            # v3 signal wiring: conflicts penalize, the previous synthesis's
            # support rate blends in, and unanswered plan axes cap the score.
            contradictions=contradictions,
            answer_support=state.get("answer_support"),
            sub_questions=state.get("sub_questions", []),
        )
        overall_conf = breakdown["overall"]

        improved = list(critique.get("improved_queries", []) or [])
        # Evidence-first: when the pool has uncorroborated or contradicted
        # claims, add disagreement-seeking queries so the expansion loop
        # researches the weakest evidence, not just more supporting pages.
        for q in _counter_evidence_queries(state):
            if q not in improved:
                improved.append(q)
        # Fix A — corroboration PROCUREMENT. Independent corroboration was
        # measured but never sought; these claim-specific, publisher-excluding
        # queries are executed directly by search_node (not left to the planner
        # model to rephrase). They also ride improved_queries so the stopping
        # policy can see them.
        corroboration_queries, corroboration_registry = _corroboration_queries(
            state, settings=getattr(llm, "settings", None)
        )
        # Primary-source completion (workstream A): a dimension whose evidence
        # is thin on primary/official publishers gets a targeted primary query
        # on the next pass. These reuse the same per-pass search channel as the
        # corroboration queries (never a new research loop) and are appended
        # after them so claim-specific procurement keeps priority.
        try:
            from app.core.evidence_completion import primary_source_followups

            attempt = max(0, int(state.get("iteration", 0)))
            for q in primary_source_followups(
                state.get("facts", []) or [],
                state.get("sub_questions", []) or [],
                limit=2,
                attempt=attempt,
            ):
                if q not in corroboration_queries:
                    corroboration_queries.append(q)
        except Exception as exc:
            logger.warning("primary_followup_generation_failed", error=str(exc), exc_info=exc)
        for q in corroboration_queries:
            if q not in improved:
                improved.append(q)
        # Freeze the augmented follow-ups on the critique so the depth
        # controller's novel-query check sees the counter-evidence queries too;
        # previously they existed only in critique_feedback and were invisible
        # to the stopping policy.
        critique["improved_queries"] = improved
        critique_feedback = critique.get("reason", "")
        if improved:
            critique_feedback = f"{critique_feedback} Improved search focus: {'; '.join(improved)}"

        # Claim-level evidence spine (requirement 6): attach the graded record
        # to the state facts so synthesis and the report can rely on
        # claim→source→verification→independence→corroboration→contradiction.
        enriched_facts = _attach_evidence(state.get("facts", []), contradictions)

        return {
            "critique": critique,
            "iteration": next_iteration,
            "confidence": overall_conf,
            "critique_feedback": critique_feedback,
            "confidence_breakdown": breakdown,
            "confidence_history": [*state.get("confidence_history", []), overall_conf],
            "contradictions": contradictions,
            "redteam": redteam_state,
            "facts": enriched_facts,
            "corroboration_queries": corroboration_queries,
            "corroboration_registry": corroboration_registry,
        }

    async def synthesizer_node(state: ResearchState) -> SynthesizerUpdate:
        # Only verification-passed facts are usable evidence (Phase 2.3).
        usable = _verified_facts(state.get("facts", []))
        all_facts = state.get("facts", [])
        intent = state.get("intent") or {}
        base_context = {
            "contradictions": state.get("contradictions", []),
            "confidence": state.get("confidence", None),
            "degraded": take_fallbacks(),
            "total_facts": len(all_facts),
            "verified_count": sum(1 for f in all_facts if f.get("verified")),
            "mode": state.get("mode", "standard"),
            "redteam_findings": (state.get("redteam", {}) or {}).get("findings", []),
            # Intent: the synthesis must answer the user's likely meaning
            # and disambiguate up front when the query was ambiguous.
            "intent": intent,
            # Answer-first outline inputs: the plan's axes are the query's
            # dimensions; the synthesizer turns them into the report's shape.
            "sub_questions": state.get("sub_questions", []),
            # Mandatory-section inputs: measured coverage gaps so the
            # limitations section is populated from real deficiencies.
            "coverage_gaps": _measured_coverage_gaps(state),
        }
        # Evidence grades (Step 4): the measured quality distribution drives
        # the writer's epistemic labeling. Computed once, failure-safe.
        evidence_distribution: Dict[str, int] = {}
        try:
            from app.core.evidence_grade import grade_facts

            graded = grade_facts(usable, contradictions=state.get("contradictions") or [])
            dist = {"A": 0, "B": 0, "C": 0, "D": 0}
            for g in graded:
                grade = str((g.get("evidence") or {}).get("grade", "D"))
                dist[grade] = dist.get(grade, 0) + 1
            base_context["evidence_distribution"] = dist
            evidence_distribution = dist
        except Exception as exc:
            logger.warning("evidence_distribution_failed", error=str(exc), exc_info=exc)
        gate_enabled = bool(getattr(llm.settings, "quality_gate_enabled", True))
        revision_enabled = bool(getattr(llm.settings, "synthesis_revision_enabled", True))
        threshold = float(getattr(llm.settings, "quality_threshold", 70.0) or 70.0)

        # Answer-first outline + section-wise synthesis (GPT Researcher
        # adaptation). Section-wise is reserved for genuinely broad questions
        # in deeper modes: it costs one LLM call per section, and the
        # synthesizer itself falls back to a single pass when a section fails.
        outline_enabled = bool(getattr(llm.settings, "synthesis_outline_enabled", True))
        section_wise_enabled = bool(
            getattr(llm.settings, "synthesis_section_wise_enabled", True)
        ) and outline_enabled
        compress_context = bool(
            getattr(llm.settings, "synthesis_context_compression", True)
        )
        compress_threshold = float(
            getattr(llm.settings, "synthesis_compression_threshold", 0.72) or 0.72
        )
        answer_outline = None
        if outline_enabled:
            try:
                from app.agents.outline import build_outline

                answer_outline = build_outline(
                    state["query"],
                    usable,
                    state.get("sub_questions", []),
                    intent=intent,
                )
            except Exception as exc:
                logger.warning("outline_build_failed", error=str(exc), exc_info=exc)
                answer_outline = None
        section_wise = bool(
            section_wise_enabled
            and answer_outline is not None
            and answer_outline.broad
            and str(state.get("mode", "standard") or "standard") in ("deep", "executive", "standard", "audit")
        )

        async def _synthesize_and_score(ctx: Dict[str, Any]):
            # Outline / section-wise / compression options ride in `context` so
            # the synthesizer entry point keeps its original signature (test
            # doubles patch it with that signature).
            answer = await synthesizer_agent(
                llm=llm,
                query=state["query"],
                facts=usable,
                context={
                    **ctx,
                    "outline": answer_outline,
                    "section_wise": section_wise,
                    "compress_context": compress_context,
                    "compress_threshold": compress_threshold,
                },
            )
            # Report-contract verification: check the emitted answer's citations
            # against the evidence (never the reverse). Observational only —
            # it scores honesty, it does not rewrite.
            support = verify_answer_support(answer, state.get("facts", []))
            try:
                from app.agents.citation_check import check_citations

                health = await check_citations(
                    answer,
                    support,
                    enabled=bool(getattr(llm.settings, "citation_check_enabled", True)),
                    timeout=float(getattr(llm.settings, "citation_check_timeout_sec", 5.0) or 5.0),
                    max_sources=int(getattr(llm.settings, "citation_check_max", 10) or 10),
                )
            except Exception as exc:
                logger.warning("citation_health_check_failed", error=str(exc), exc_info=exc)
                health = {"checked": 0, "sources": [], "summary": {}, "enabled": False}
            quality = evaluate_answer(
                state["query"],
                intent=intent,
                answer=answer,
                facts=state.get("facts", []),
                answer_support=support,
                citation_health=health,
                contradictions=state.get("contradictions", []),
                redteam_findings=(state.get("redteam", {}) or {}).get("findings", []),
                mode=str(state.get("mode", "standard") or "standard"),
                threshold=threshold,
            )
            return answer, support, health, quality

        answer, support, citation_health, quality = await _synthesize_and_score(base_context)

        # Answer revision pass (the quality optimizer's LLM half): the writer
        # rewrites its draft ONCE — fed the measured failures when the gate
        # failed, a polish mandate when it passed — and the better-scoring
        # draft ships. Never a loop (budget rule). Skipped when the gate is
        # disabled or the synthesizer itself is on deterministic fallback
        # (extraction cannot act on feedback).
        # Quick mode trades the unconditional polish pass for latency; the
        # gate still repairs a FAILING draft there.
        quick_mode = str(state.get("mode", "standard")) == "quick"
        run_revision = (
            gate_enabled and revision_enabled
            and ("synthesizer" not in take_fallbacks())
            and (not quick_mode or not quality.passed)
        )
        if run_revision:
            try:
                answer2, support2, health2, quality2 = await _synthesize_and_score({
                    **base_context,
                    "quality_feedback": quality.failures,
                    "revision": True,
                })
            except Exception as exc:
                logger.warning("synthesis_revision_failed", error=str(exc), exc_info=exc)
            else:
                if quality2.overall >= quality.overall:
                    answer, support, citation_health, quality = (
                        answer2, support2, health2, quality2,
                    )

        logger.info("synthesizer_done", answer_chars=len(answer), usable_facts=len(usable),
                    support_rate=round(support_rate_val, 2) if (support_rate_val := support.get("rate")) is not None else None,
                    unsupported=len(support["unsupported"]),
                    citation_summary=citation_health.get("summary", {}),
                    quality=quality.overall, quality_passed=quality.passed)
        return {
            "synthesized_answer": answer,
            "answer_support": support,
            "citation_health": citation_health,
            "quality": quality.to_dict(),
            "evidence_distribution": evidence_distribution,
            "outline": answer_outline.to_dict() if answer_outline is not None else {},
            "section_wise": bool(section_wise),
        }

    async def finalize_node(state: ResearchState) -> FinalizeUpdate:
        # Decision options computed once and shared with the report builder.
        options = build_decision_layer(state)
        report = build_markdown_report(state, decision_options=options)
        # Decision options ride in state so the route can persist them (3.5).
        return {
            "final_report": report,
            "decision_options": options,
        }

    def route_after_critic(state: ResearchState) -> str:
        critique = state.get("critique", {})
        is_sufficient = bool(critique.get("is_sufficient", False))

        # Hard walls (budget/time, iteration ceiling) are absolute — an
        # evidence gap cannot be acted on if no pass can run. Checked here,
        # before ANY expand branch, so no evidence-gap override or
        # critic-insufficient expansion can route past the ceiling and run
        # LangGraph into its recursion limit.
        if depth_controller.hard_wall_reached(state):
            logger.info(
                "route_hard_wall_finalize",
                iteration=int(state.get("iteration", 0)),
                max_iterations=int(state.get("max_iterations", 0)),
            )
            return "synthesizer"

        # Evidence-first sufficiency (Step 3): a critic saying "enough" is an
        # OPINION, not proof. Before trusting it, check the measured evidence
        # base — uncovered axis gaps, uncorroborated quantitative claims, and
        # unresolved contradictions are grounds to keep researching even when
        # the model is satisfied. When the evidence gate is clean, the critic
        # wins exactly as before (no extra iteration, no score change).
        if is_sufficient and _evidence_gaps_remain(state):
            logger.info("evidence_gate_overrides_critic", iteration=int(state.get("iteration", 0)))
            is_sufficient = False

        if is_sufficient:
            return "synthesizer"

        # Dynamic Research Depth (2.8) decides expand vs finalize from axis
        # coverage, corroboration, contradictions, marginal gain and the
        # hard walls. The evidence gate is now baked into `decide`'s priority
        # order, so a soft stop (marginal gain / no-novel-queries) can no
        # longer defeat an outstanding coverage or corroboration gap.
        decision = depth_controller.decide(state)
        logger.info("depth_decision", decision=decision, iteration=int(state.get("iteration", 0)))
        if decision == "expand":
            return "planner"
        return "synthesizer"

    graph.add_node("intent", intent_node)
    graph.add_node("planner", planner_node)
    graph.add_node("search", search_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("critic", critic_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, entry_node if entry_node else "intent")
    if entry_node is None:
        # Understand-before-searching: intent runs before any plan is shaped.
        graph.add_edge("intent", "planner")
    graph.add_edge("planner", "search")
    graph.add_edge("search", "summarizer")
    graph.add_edge("summarizer", "verifier")
    graph.add_edge("verifier", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"planner": "planner", "synthesizer": "synthesizer"})
    graph.add_edge("synthesizer", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
