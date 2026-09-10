"""Agent Context Isolation (Phase 2.9, Feature 04).

An AgentContext is the ONLY thing specialist/search/summarizer worker
functions receive — never the full ResearchState, never another
sub-question's raw content. Cross-agent information enters only through
the structured facts list after summarization + verification.

Raw fetched content (SearchResult.content) is dropped from the context
once that sub-question's summarization is complete (see release_raw_content).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.agents.planner import SubQuestion

ALLOWED_TOOL_PERMISSIONS = frozenset({"web_search", "wikipedia", "content_fetch"})

# Specialist domain routing (Phase 3.1): the summarizer gets a domain
# prompt variant based on the contract's `domain` field.
SPECIALIST_DOMAINS = {
    "economics": "financial",
    "machine_learning": "technical",
    "software": "technical",
    "science": "scientific",
    "legal": "legal",
    "policy": "policy",
    "academic": "academic",
}
SPECIALIST_ROLES = frozenset({"financial", "technical", "market", "general",
                              "legal", "scientific", "policy", "academic"})


def specialist_role_for_domain(domain: str) -> str:
    """Map a delegation contract's domain to a specialist role.

    Market covers news/statistical comparison work across domains;
    everything unrouted stays general.
    """
    d = (domain or "").strip().lower()
    if d in SPECIALIST_DOMAINS:
        return SPECIALIST_DOMAINS[d]
    if d == "general":
        return "general"
    return "general"


@dataclass
class AgentContext:
    """The scoped contract one agent invocation is allowed to see."""

    contract: SubQuestion
    # This sub-question's own search results only (never other sub-questions').
    own_results: List[Dict[str, Any]] = field(default_factory=list)
    # Explicit tool scope: a specialist may be restricted further (3.1).
    tool_permissions: frozenset = ALLOWED_TOOL_PERMISSIONS
    # Optional domain allowlist enforced for the specialist (3.1 hook).
    allowed_source_domains: Optional[Tuple[str, ...]] = None

    def release_raw_content(self) -> None:
        """Drop raw fetched content once summarization for this context is done.

        The snippet stays (small, useful for transparency); the full page
        text must not persist in shared state past summarization.
        """
        for item in self.own_results:
            if isinstance(item, dict) and "content" in item:
                item["content"] = ""

    def question(self) -> str:
        return str(self.contract.get("question", "")).strip()

    def axis(self) -> str:
        return str(self.contract.get("axis", "")).strip()

    def minimum_sources(self) -> int:
        return int(self.contract.get("minimum_sources", 2) or 2)

    def stop_condition(self) -> str:
        return str(self.contract.get("stop_condition", "")).strip()

    def specialist_role(self) -> str:
        """Role selected by this contract's domain (3.1)."""
        return specialist_role_for_domain(str(self.contract.get("domain", "general")))

    def domain(self) -> str:
        return str(self.contract.get("domain", "general")).strip()


def build_contexts(
    sub_questions: List[SubQuestion],
    search_results: List[Dict[str, Any]],
) -> List[AgentContext]:
    """One AgentContext per sub-question, each holding only its own results."""
    by_question: Dict[str, List[Dict[str, Any]]] = {}
    for result in search_results or []:
        sub_q = str(result.get("sub_question", "")).strip()
        by_question.setdefault(sub_q, []).append(result)

    contexts: List[AgentContext] = []
    for contract in sub_questions:
        if not isinstance(contract, dict):
            continue
        question = str(contract.get("question", "")).strip()
        # Variant phrasings belong to the parent contract's context —
        # otherwise their results orphan and the summarizer drops them.
        wanted = {question}
        variants = contract.get("variants", [])
        if isinstance(variants, list):
            wanted |= {str(v).strip() for v in variants if str(v).strip()}
        own = []
        for text, results in by_question.items():
            if text in wanted:
                own.extend(results)
        contexts.append(
            AgentContext(
                contract=contract,
                own_results=own,
            )
        )
    return contexts
