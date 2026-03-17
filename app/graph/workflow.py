from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.critic import critic_agent
from app.agents.planner import planner_agent
from app.agents.search import SearchClient
from app.agents.summarizer import summarizer_agent
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
    confidence: float


# Lightweight in-memory runtime state store (per request id).
RUNTIME_STATE: Dict[str, Dict[str, Any]] = {}


def build_initial_state(query: str, max_iterations: int) -> ResearchState:
    return {
        "query": query,
        "sub_questions": [],
        "search_results": [],
        "facts": [],
        "critique": {},
        "critique_feedback": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "final_report": "",
        "confidence": 0.0,
    }


def _dedupe_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    cleaned: List[Dict[str, Any]] = []
    for fact in facts:
        key = (fact.get("claim", "").strip().lower(), fact.get("source", "").strip().lower())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(fact)
    return cleaned


def build_markdown_report(state: ResearchState) -> str:
    facts = state.get("facts", [])
    critique = state.get("critique", {})
    confidence = float(state.get("confidence", 0.0))

    finding_lines = [
        f"- {item.get('claim', '').strip()} ({item.get('source', '').strip()})"
        for item in facts[:10]
        if item.get("claim") and item.get("source")
    ]
    if not finding_lines:
        finding_lines = ["- No robust findings were extracted from available search results."]

    analysis_text = critique.get("reason", "Analysis was generated from limited open-web evidence.")
    limitations = [
        "Free-tier APIs may rate limit or return shallow snippets.",
        "No paywalled or private databases were accessed.",
        "Confidence is estimated from retrieved evidence quality, not formal verification.",
    ]

    lines = [
        "# Research Report",
        "",
        "## Key Findings",
        *finding_lines,
        "",
        "## Analysis",
        str(analysis_text),
        "",
        "## Limitations",
        *[f"- {item}" for item in limitations],
        "",
        "## Confidence Score",
        f"{confidence:.2f}",
    ]
    return "\n".join(lines)


def create_workflow(llm: LLMClient, search_client: SearchClient):
    graph = StateGraph(ResearchState)

    async def planner_node(state: ResearchState) -> Dict[str, Any]:
        sub_questions = await planner_agent(
            llm=llm,
            query=state["query"],
            critique_feedback=state.get("critique_feedback", ""),
        )
        return {"sub_questions": sub_questions}

    async def search_node(state: ResearchState) -> Dict[str, Any]:
        questions = state.get("sub_questions", [])[:5]
        results = await search_client.run_search(questions)
        return {"search_results": results}

    async def summarizer_node(state: ResearchState) -> Dict[str, Any]:
        fresh_facts = await summarizer_agent(
            llm=llm,
            query=state["query"],
            search_results=state.get("search_results", []),
        )
        merged = _dedupe_facts([*state.get("facts", []), *fresh_facts])
        return {"facts": merged}

    async def critic_node(state: ResearchState) -> Dict[str, Any]:
        next_iteration = int(state.get("iteration", 0)) + 1
        critique = await critic_agent(
            llm=llm,
            query=state["query"],
            facts=state.get("facts", []),
            iteration=next_iteration,
            max_iterations=int(state.get("max_iterations", 3)),
        )

        fact_scores = [float(item.get("confidence", 0.0)) for item in state.get("facts", []) if item.get("confidence") is not None]
        fact_conf = sum(fact_scores) / len(fact_scores) if fact_scores else 0.0
        overall_conf = max(0.0, min(1.0, (fact_conf + float(critique.get("confidence", 0.0))) / 2))

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

    async def finalize_node(state: ResearchState) -> Dict[str, Any]:
        return {"final_report": build_markdown_report(state)}

    def route_after_critic(state: ResearchState) -> str:
        critique = state.get("critique", {})
        is_sufficient = bool(critique.get("is_sufficient", False))
        if is_sufficient:
            return "finalize"
        if int(state.get("iteration", 0)) >= int(state.get("max_iterations", 3)):
            return "finalize"
        return "planner"

    graph.add_node("planner", planner_node)
    graph.add_node("search", search_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "search")
    graph.add_edge("search", "summarizer")
    graph.add_edge("summarizer", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"planner": "planner", "finalize": "finalize"})
    graph.add_edge("finalize", END)

    return graph.compile()
