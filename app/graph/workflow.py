from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.evidence_utils import dedupe_semantic_facts, filter_facts_by_domain, source_reliability_score
from app.agents.critic import critic_agent
from app.agents.planner import planner_agent
from app.agents.search import SearchClient
from app.agents.summarizer import summarizer_agent
from app.agents.synthesizer import synthesizer_agent
from app.core.llm import LLMClient


class ResearchState(TypedDict, total=False):
    query: str
    sub_questions: List[str]
    search_results: List[Dict[str, str]]
    facts: List[Dict[str, Any]]
    critique: Dict[str, Any]
    critique_feedback: str
    iteration: int
    max_iterations: int
    final_report: str
    synthesized_answer: str
    confidence: float


class PlannerUpdate(TypedDict):
    sub_questions: List[str]


class SearchUpdate(TypedDict):
    search_results: List[Dict[str, str]]


class SummarizerUpdate(TypedDict):
    facts: List[Dict[str, Any]]


class CriticUpdate(TypedDict):
    critique: Dict[str, Any]
    iteration: int
    confidence: float
    critique_feedback: str


class SynthesizerUpdate(TypedDict):
    synthesized_answer: str


class FinalizeUpdate(TypedDict):
    final_report: str


# Lightweight in-memory runtime state store (per request id).
RUNTIME_STATE: Dict[str, Dict[str, Any]] = {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_initial_state(query: str, max_iterations: int) -> ResearchState:
    effective_max_iterations = max(3, int(max_iterations))
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
        return {"sub_questions": sub_questions}

    async def search_node(state: ResearchState) -> SearchUpdate:
        questions = state.get("sub_questions", [])[:5]
        results = await search_client.run_search(questions)
        return {"search_results": results}

    async def summarizer_node(state: ResearchState) -> SummarizerUpdate:
        fresh_facts = await summarizer_agent(
            llm=llm,
            query=state["query"],
            search_results=state.get("search_results", []),
        )
        merged = dedupe_semantic_facts([*state.get("facts", []), *fresh_facts])
        return {"facts": merged}

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
        answer = await synthesizer_agent(
            llm=llm,
            query=state["query"],
            facts=state.get("facts", []),
        )
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
        quality_ceiling = max(max_iterations, min_quality_iterations)
        if iteration >= quality_ceiling:
            return "synthesizer"
        return "planner"

    graph.add_node("planner", planner_node)
    graph.add_node("search", search_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "search")
    graph.add_edge("search", "summarizer")
    graph.add_edge("summarizer", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"planner": "planner", "synthesizer": "synthesizer"})
    graph.add_edge("synthesizer", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
