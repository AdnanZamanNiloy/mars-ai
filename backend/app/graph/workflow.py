from __future__ import annotations

import asyncio
import datetime
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.evidence_utils import dedupe_semantic_facts, filter_facts_by_domain, verify_answer_support
from app.agents.critic import critic_agent
from app.agents.orchestrator import orchestrate
from app.agents.planner import normalize_text, planner_agent
from app.agents.search import SearchClient
from app.agents.summarizer import summarizer_agent
from app.agents.synthesizer import synthesizer_agent
from app.agents.verifier import verify_facts
from app.core.llm import LLMClient
from app.core.confidence import compute_confidence
from app.core.contradictions import find_contradictions
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
    budget_tracker: Any
    verification_stats: Dict[str, Any]
    answer_support: Dict[str, Any]
    confidence_breakdown: Dict[str, Any]
    confidence_history: List[float]
    contradictions: List[Dict[str, Any]]
    mode: str
    decision_options: List[Dict[str, Any]]


class PlannerUpdate(TypedDict):
    sub_questions: List[str]


class SearchUpdate(TypedDict):
    search_results: List[Dict[str, str]]


class SummarizerUpdate(TypedDict):
    facts: List[Dict[str, Any]]


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


class SynthesizerUpdate(TypedDict):
    synthesized_answer: str
    answer_support: Dict[str, Any]


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
    from app.agents.orchestrator import MODE_PRESETS

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
    plan = orchestrate(query, max_parallel_agents=max_parallel_agents, deep_research=deep_research)
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
        },
        "deep_research": plan.deep_research,
        "confidence_history": [],
        "mode": mode if preset is not None else "standard",
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


def build_markdown_report(state: ResearchState) -> str:
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

    tracker = state.get("budget_tracker")
    if tracker is not None and tracker.over_budget:
        limitations.append(tracker.limitation_note())

    # Dynamic Research Depth (2.8): name an early stop on marginal gain.
    early_stop_note = depth_controller.stop_reason(state)
    if early_stop_note:
        limitations.append(early_stop_note)

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

    # Contradiction Engine (3.2): surface source conflicts explicitly.
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
        lines.extend(["# Contradictions", *contradiction_lines, ""])

    # Decision Intelligence Layer (3.5, Feature 18): options → recommendation
    # → rationale, structurally SEPARATE from the findings above.
    decision_options = build_decision_layer(state)
    decision_lines = []
    for o in decision_options:
        marker = " (RECOMMENDED)" if o.get("is_recommended") else ""
        decision_lines.append(f"- Option {o.get('option_label', '?')}{marker}: {o.get('description', '')}")
        if o.get("rationale"):
            decision_lines.append(f"  Rationale: {o['rationale']}")
        if o.get("risk_note"):
            decision_lines.append(f"  Risk: {o['risk_note']}")
    lines.extend(["# Decision Layer", *decision_lines, ""])

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

    entry_node=None (default): START → planner (full pipeline).
    entry_node="critic": START → critic — used by the resume endpoint (3.3)
    so a failed/timeout run continues from persisted evidence instead of
    re-running planner/search.
    """
    if entry_node is not None and entry_node != "critic":
        raise ValueError(f"unsupported entry_node={entry_node!r} (only 'critic' is supported)")
    graph = StateGraph(ResearchState)

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
        sub_questions = await planner_agent(
            llm=llm,
            query=state["query"],
            critique_feedback=feedback,
            today=datetime.date.today().isoformat(),
        )
        # Hardware guardrail: cap the plan at the orchestrated target agents.
        orchestration = state.get("orchestration", {})
        target = int(orchestration.get("target_agents", 5) or 5)
        if expanding:
            # Per-axis expansion: keep researched history, cap only the NEW
            # additions at target so per-pass load stays bounded.
            merged = _merge_questions(existing, sub_questions)
            added = merged[len(existing):][: max(1, target)]
            sub_questions = [*existing, *added]
        else:
            sub_questions = sub_questions[: max(1, target)]
        logger.info("planner_done", sub_questions=len(sub_questions), expanding=expanding)
        return {"sub_questions": sub_questions}

    async def search_node(state: ResearchState) -> SearchUpdate:
        previous = [r for r in state.get("search_results", []) or [] if isinstance(r, dict)]
        # Per-axis expansion: search only questions with no results yet.
        # Each unanswered parent fans out to its alternate phrasings
        # (variants ride the parent contract, so nothing orphans).
        # Accumulated results stay bounded (passes × ~10 snippets) and the
        # verifier still releases raw content after each pass.
        fresh: List[str] = []
        answered = {
            normalize_text(str(r.get("sub_question", ""))) for r in previous
        } - {""}
        by_text = {}
        for item in state.get("sub_questions", []) or []:
            text = _extract_question_text(item)
            if text and text not in by_text:
                by_text[text] = item
        for text in _unanswered_questions(state.get("sub_questions", []), previous):
            fresh.append(text)
            item = by_text.get(text)
            if isinstance(item, dict):
                for v in item.get("variants", []) or []:
                    vs = str(v or "").strip()
                    if vs and normalize_text(vs) not in answered:
                        fresh.append(vs)
        cap = max(1, int(getattr(getattr(search_client, "settings", None),
                               "search_max_queries_per_pass", 8) or 8))
        fresh = fresh[:cap]
        if not fresh:
            if previous:
                return {"search_results": previous}
            fallback = state.get("query", "").strip()
            if not fallback:
                return {"search_results": previous}
            fresh = [fallback]
        results = await search_client.run_search(fresh)
        seen_urls = {r.get("url") for r in previous if r.get("url")}
        merged = [*previous, *(r for r in results if r.get("url") not in seen_urls)]
        logger.info("search_done", results=len(merged), fresh=len(fresh))
        return {"search_results": merged}

    async def summarizer_node(state: ResearchState) -> SummarizerUpdate:
        # Agent Context Isolation (2.9): each sub-question worker sees only
        # its own AgentContext — own contract + own results, never the full
        # ResearchState or another sub-question's raw content.
        contexts = build_contexts(
            sub_questions=state.get("sub_questions", []),
            search_results=state.get("search_results", []),
        )

        async def _summarize_context(ctx: AgentContext) -> List[Dict[str, Any]]:
            if not ctx.own_results:
                return []
            # Specialist routing (3.1): role comes from THIS context's
            # delegation contract — each specialist sees only its own
            # scoped context, never the shared ResearchState (2.9).
            return await summarizer_agent(
                llm=llm,
                query=state["query"],
                search_results=ctx.own_results,
                specialist_role=ctx.specialist_role(),
            )

        results = await asyncio.gather(*(_summarize_context(ctx) for ctx in contexts if ctx.own_results))

        fresh_facts = [fact for facts in results for fact in facts]
        merged = dedupe_semantic_facts([*state.get("facts", []), *fresh_facts])
        logger.info("summarizer_done", fresh_facts=len(fresh_facts), total_facts=len(merged))
        return {"facts": merged}

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
        # finalize, later iterations) needs it. Blank in place: these dict
        # objects are shared with the contexts built in summarizer_node.
        for result in state.get("search_results", []):
            if isinstance(result, dict) and "content" in result:
                result["content"] = ""

        logger.info("verifier_done", **stats)
        return {"facts": verified, "verification_stats": stats}

    async def critic_node(state: ResearchState) -> CriticUpdate:
        next_iteration = int(state.get("iteration", 0)) + 1

        # Contradiction Engine (3.2) computed HERE (not in verifier) so the
        # resume path — which re-enters at critic with persisted facts —
        # still feeds contradictions to the critic and the report.
        contradictions = find_contradictions(state.get("facts", []))
        if contradictions:
            logger.info("contradictions_found", count=len(contradictions))

        critique = await critic_agent(
            llm=llm,
            query=state["query"],
            facts=state.get("facts", []),
            iteration=next_iteration,
            max_iterations=int(state.get("max_iterations", 3)),
            contradictions=contradictions,
        )

        # Confidence Engine (Phase 2.4) replaces the inline weighted formula.
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
        )
        overall_conf = breakdown["overall"]

        improved = critique.get("improved_queries", [])
        critique_feedback = critique.get("reason", "")
        if improved:
            critique_feedback = f"{critique_feedback} Improved search focus: {'; '.join(improved)}"

        return {
            "critique": critique,
            "iteration": next_iteration,
            "confidence": overall_conf,
            "critique_feedback": critique_feedback,
            "confidence_breakdown": breakdown,
            "confidence_history": [*state.get("confidence_history", []), overall_conf],
            "contradictions": contradictions,
        }

    async def synthesizer_node(state: ResearchState) -> SynthesizerUpdate:
        # Only verification-passed facts are usable evidence (Phase 2.3).
        usable = _verified_facts(state.get("facts", []))
        answer = await synthesizer_agent(
            llm=llm,
            query=state["query"],
            facts=usable,
        )
        # Report-contract verification: check the emitted answer's citations
        # against the evidence (never the reverse). Observational only —
        # it scores honesty, it does not rewrite.
        support = verify_answer_support(answer, state.get("facts", []))
        support_rate = support["rate"]
        logger.info("synthesizer_done", answer_chars=len(answer), usable_facts=len(usable),
                    support_rate=round(support_rate, 2) if support_rate is not None else None,
                    unsupported=len(support["unsupported"]))
        return {"synthesized_answer": answer, "answer_support": support}

    async def finalize_node(state: ResearchState) -> FinalizeUpdate:
        report = build_markdown_report(state)
        # Decision options ride in state so the route can persist them (3.5).
        return {
            "final_report": report,
            "decision_options": build_decision_layer(state),
        }

    def route_after_critic(state: ResearchState) -> str:
        critique = state.get("critique", {})
        is_sufficient = bool(critique.get("is_sufficient", False))

        if is_sufficient:
            return "synthesizer"

        # Dynamic Research Depth (2.8) replaces the old two-condition check:
        # decides expand vs finalize from axis coverage, marginal confidence
        # gain, remaining budget, and the iteration/depth ceiling.
        decision = depth_controller.decide(state)
        logger.info("depth_decision", decision=decision, iteration=int(state.get("iteration", 0)))
        if decision == "expand":
            return "planner"
        return "synthesizer"

    graph.add_node("planner", planner_node)
    graph.add_node("search", search_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("critic", critic_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, entry_node if entry_node else "planner")
    graph.add_edge("planner", "search")
    graph.add_edge("search", "summarizer")
    graph.add_edge("summarizer", "verifier")
    graph.add_edge("verifier", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"planner": "planner", "synthesizer": "synthesizer"})
    graph.add_edge("synthesizer", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
