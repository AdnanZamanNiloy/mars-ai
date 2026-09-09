from __future__ import annotations

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
) -> ResearchState:
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
        "# Limitations",
        *[f"- {item}" for item in limitations],
        "",
        "# Confidence Score",
        f"{confidence:.2f}",
    ]
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
        fresh_facts = await summarizer_agent(
            llm=llm,
            query=state["query"],
            search_results=state.get("search_results", []),
        )
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
        logger.info("verifier_done", **stats)
        return {"facts": verified, "verification_stats": stats}

    async def critic_node(state: ResearchState) -> CriticUpdate:
        next_iteration = int(state.get("iteration", 0)) + 1
        critique = await critic_agent(
            llm=llm,
            query=state["query"],
            facts=state.get("facts", []),
            iteration=next_iteration,
            max_iterations=int(state.get("max_iterations", 3)),
        )

        current_facts = state.get("facts", [])
        fact_scores = [_safe_float(item.get("confidence", 0.0)) for item in current_facts if item.get("confidence") is not None]
        source_scores = [source_reliability_score(str(item.get("source", ""))) for item in current_facts if item.get("source")]
        fact_conf = sum(fact_scores) / len(fact_scores) if fact_scores else 0.0
        source_conf = sum(source_scores) / len(source_scores) if source_scores else 0.0
        critic_conf = _safe_float(critique.get("confidence", 0.0))
        overall_conf = max(0.0, min(1.0, (0.45 * fact_conf) + (0.35 * critic_conf) + (0.20 * source_conf)))

        improved = critique.get("improved_queries", [])
        critique_feedback = critique.get("reason", "")
        if improved:
            critique_feedback = f"{critique_feedback} Improved search focus: {'; '.join(improved)}"

        return {
            "critique": critique,
            "iteration": next_iteration,
            "confidence": overall_conf,
            "critique_feedback": critique_feedback,
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
        min_quality_iterations = 3
        max_iterations = int(state.get("max_iterations", 3))
        iteration = int(state.get("iteration", 0))

        if is_sufficient:
            return "synthesizer"

        # Cost Governor (2.2): over budget → stop expanding, finalize with
        # a limitations note instead of exceeding the cap.
        tracker = state.get("budget_tracker")
        if tracker is not None and tracker.over_budget:
            logger.info("budget_cutoff", cost=tracker.estimated_cost_usd, limit=tracker.limit_usd)
            return "synthesizer"

        quality_ceiling = max(max_iterations, min_quality_iterations)
        if iteration >= quality_ceiling:
            return "synthesizer"
        return "planner"

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
