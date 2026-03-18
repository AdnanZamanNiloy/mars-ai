from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, TypedDict

from app.core.llm import LLMClient

logger = logging.getLogger(__name__)


# =========================
# Structured Types
# =========================

class SubQuestion(TypedDict, total=False):
    id: int
    question: str
    axis: str
    search_type: str
    priority: int
    depends_on: List[int]
    coverage_goal: str
    domain: str


class PlannerOutput(TypedDict, total=False):
    query_type: str
    query_scope: str
    dominant_domain: str
    sub_questions: List[SubQuestion]
    coverage_note: str


# =========================
# Constants
# =========================

VALID_SEARCH_TYPES = {
    "encyclopedia",
    "academic",
    "statistical",
    "news",
    "comparison",
}

VALID_DOMAINS = {
    "machine_learning",
    "software",
    "philosophy",
    "economics",
    "science",
    "general",
}


# =========================
# Utilities
# =========================

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def normalize_domain(domain: str) -> str:
    d = normalize_text(domain)
    return d if d in VALID_DOMAINS else "general"


def is_valid_question(q: str) -> bool:
    return len(q.split()) >= 3


def deduplicate_semantic(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    result = []

    for item in items:
        key = normalize_text(item["question"])
        key = re.sub(r"(definition|overview|introduction)", "", key)

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


# =========================
# Fallback Planner (CRITICAL)
# =========================

def fallback_plan(query: str) -> List[Dict[str, Any]]:
    concept = re.sub(r"^(what is|define|explain)\s+", "", query.lower()).strip()

    return [
        {
            "id": 1,
            "question": f"{concept} definition explanation overview",
            "axis": "definition",
            "search_type": "encyclopedia",
            "priority": 1,
            "depends_on": [],
            "coverage_goal": "core meaning",
            "domain": "general",
        },
        {
            "id": 2,
            "question": f"{concept} mechanism how it works components",
            "axis": "mechanism",
            "search_type": "academic",
            "priority": 2,
            "depends_on": [1],
            "coverage_goal": "internal working",
            "domain": "general",
        },
        {
            "id": 3,
            "question": f"{concept} applications real world examples",
            "axis": "application",
            "search_type": "comparison",
            "priority": 2,
            "depends_on": [1],
            "coverage_goal": "practical usage",
            "domain": "general",
        },
        {
            "id": 4,
            "question": f"{concept} limitations challenges drawbacks",
            "axis": "criticism",
            "search_type": "academic",
            "priority": 3,
            "depends_on": [2],
            "coverage_goal": "weaknesses",
            "domain": "general",
        },
    ]


# =========================
# Planner Agent
# =========================

async def planner_agent(
    llm: LLMClient,
    query: str,
    critique_feedback: str = ""
) -> List[Dict[str, Any]]:

    normalized = normalize_text(query)

    # =========================
    # Fast Path
    # =========================
    if normalized.startswith(("what is", "define", "explain")) and not critique_feedback:
        logger.info("[Planner] Using fast-path")
        return fallback_plan(normalized)

    # =========================
    # LLM Planning
    # =========================

    feedback_block = f"\nCritique feedback: {critique_feedback}" if critique_feedback else ""

    user_prompt = f"""
Query: {query}
{feedback_block}

Generate a structured research plan.
Return JSON only.
"""

    try:
        payload: PlannerOutput = await llm.generate_json(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except Exception as e:
        logger.error(f"[Planner] LLM failed: {e}")
        return fallback_plan(query)

    sub_questions = payload.get("sub_questions", [])

    if not sub_questions:
        logger.warning("[Planner] Empty LLM output, using fallback")
        return fallback_plan(query)

    # =========================
    # Post-processing
    # =========================

    cleaned: List[Dict[str, Any]] = []

    for i, item in enumerate(sub_questions):
        q = item.get("question", "").strip()

        if not q or not is_valid_question(q):
            continue

        cleaned.append({
            "id": item.get("id", i + 1),
            "question": q,
            "axis": item.get("axis", "general"),
            "search_type": item.get("search_type", "encyclopedia")
                if item.get("search_type") in VALID_SEARCH_TYPES else "encyclopedia",
            "priority": int(item.get("priority", 2)),
            "depends_on": item.get("depends_on", []),
            "coverage_goal": item.get("coverage_goal", ""),
            "domain": normalize_domain(item.get("domain", "general")),
        })

    # Deduplicate (semantic-ish)
    cleaned = deduplicate_semantic(cleaned)

    # Sort by priority
    cleaned.sort(key=lambda x: x["priority"])

    # Limit results
    final = cleaned[:5]

    if not final:
        logger.warning("[Planner] All filtered out, fallback used")
        return fallback_plan(query)

    return final