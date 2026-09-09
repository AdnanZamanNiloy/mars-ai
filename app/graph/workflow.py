from __future__ import annotations

import asyncio
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.evidence_utils import dedupe_semantic_facts, filter_facts_by_domain, source_reliability_score
from app.agents.critic import critic_agent
from app.agents.orchestrator import orchestrate
from app.agents.planner import planner_agent
from app.agents.search import SearchClient
from app.agents.summarizer import summarizer_agent
from app.agents.synthesizer import synthesizer_agent
from app.agents.verifier import verify_facts
from app.core.llm import LLMClient
from app.core.confidence import compute_confidence
from app.core.contradictions import find_contradictions
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
    confidence_breakdown: Dict[str, Any]
    confidence_history: List[float]
    contradictions: List[Dict[str, Any]]
    mode: str


class PlannerUpdate(TypedDict):
    sub_questions: List[str]


class SearchUpdate(TypedDict):
    search_results: List[Dict[str, str]]


class SummarizerUpdate(TypedDict):
    facts: List[Dict[str, Any]]


class VerifierUpdate(TypedDict):
    facts: List[Dict[str, Any]]
    verification_stats: Dict[str, Any]
    contradictions: List[Dict[str, Any]]


class CriticUpdate(TypedDict):
    critique: Dict[str, Any]
    iteration: int
    confidence: float
    critique_feedback: str
    confidence_breakdown: Dict[str, Any]


class SynthesizerUpdate(TypedDict):
    synthesized_answer: str


class FinalizeUpdate(TypedDict):
    final_report: str


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
        # The preset's agent cap REPLACES the setting default: deep is the
        # only mode allowed to exceed MAX_PARALLEL_AGENTS (manual 3.7 DoD).
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

    lines.extend([
        "# Limitations",
        *[f"- {item}" for item in limitations],
        "",
        "# Confidence Score",
        f"{confidence:.2f}",
    ])
    return "\n".join(lines)


def create_workflow(llm: LLMClient, search_client: SearchClient):
    graph = StateGraph(ResearchState)

    async def planner_node(state: ResearchState) -> PlannerUpdate:
        sub_questions = await planner_agent(
            llm=llm,
            query=state["query"],
            critique_feedback=state.get("critique_feedback", ""),
        )
        # Hardware guardrail: cap the plan at the orchestrated target agents.
        orchestration = state.get("orchestration", {})
        target = int(orchestration.get("target_agents", 5) or 5)
        sub_questions = sub_questions[: max(1, target)]
        logger.info("planner_done", sub_questions=len(sub_questions))
        return {"sub_questions": sub_questions}

    async def search_node(state: ResearchState) -> SearchUpdate:
        raw_questions = state.get("sub_questions", [])[:5]
        questions = [_extract_question_text(item) for item in raw_questions]
        questions = [q for q in questions if q]
        if not questions:
            questions = [state.get("query", "").strip()]
        results = await search_client.run_search(questions)
        logger.info("search_done", results=len(results), sub_questions=len(questions))
        return {"search_results": results}

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

        # Contradiction Engine (3.2): flag topically-similar claims with
        # conflicting figures from different sources.
        contradictions = find_contradictions(verified)
        if contradictions:
            logger.info("contradictions_found", count=len(contradictions))

        return {"facts": verified, "verification_stats": stats, "contradictions": contradictions}

    async def critic_node(state: ResearchState) -> CriticUpdate:
        next_iteration = int(state.get("iteration", 0)) + 1
        critique = await critic_agent(
            llm=llm,
            query=state["query"],
            facts=state.get("facts", []),
            iteration=next_iteration,
            max_iterations=int(state.get("max_iterations", 3)),
            contradictions=state.get("contradictions", []),
        )

        # Confidence Engine (Phase 2.4) replaces the inline weighted formula.
        breakdown = compute_confidence(
            facts=state.get("facts", []),
            critique=critique,
            iteration=next_iteration,
            max_iterations=int(state.get("max_iterations", 3)),
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
        }

    async def synthesizer_node(state: ResearchState) -> SynthesizerUpdate:
        # Only verification-passed facts are usable evidence (Phase 2.3).
        usable = _verified_facts(state.get("facts", []))
        answer = await synthesizer_agent(
            llm=llm,
            query=state["query"],
            facts=usable,
        )
        logger.info("synthesizer_done", answer_chars=len(answer), usable_facts=len(usable))
        return {"synthesized_answer": answer}

    async def finalize_node(state: ResearchState) -> FinalizeUpdate:
        return {"final_report": build_markdown_report(state)}

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

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "search")
    graph.add_edge("search", "summarizer")
    graph.add_edge("summarizer", "verifier")
    graph.add_edge("verifier", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"planner": "planner", "synthesizer": "synthesizer"})
    graph.add_edge("synthesizer", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
