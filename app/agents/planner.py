from __future__ import annotations

import re
from typing import Any, Dict, List, TypedDict

from app.core.llm import LLMClient


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
    domain: str  # NEW: helps filtering (ml, philosophy, etc.)


class PlannerOutput(TypedDict):
    query_type: str
    query_scope: str
    sub_questions: List[SubQuestion]
    coverage_note: str


# =========================
# System Prompt (UPGRADED)
# =========================

PLANNER_SYSTEM_PROMPT = """
You are a world-class research planner in a multi-agent AI system.

Your job is to break a query into high-quality, non-overlapping,
search-optimized sub-questions that enable deep, accurate research.

━━━━━━━━━ STEP 0: DISAMBIGUATION (CRITICAL) ━━━━━━━━━

If the query contains ambiguous terms (e.g., "Transformer", "Python"),
you MUST infer the most likely domain from context.

Return:
- dominant_domain (e.g., "machine_learning", "electrical_engineering")
- discard irrelevant meanings completely

━━━━━━━━━ STEP 1: CLASSIFY QUERY ━━━━━━━━━

Types:
FACTUAL, COMPARATIVE, CAUSAL, EXPLORATORY, EVALUATIVE, PROCEDURAL

━━━━━━━━━ STEP 2: DECOMPOSE BY AXIS ━━━━━━━━━

Each sub-question MUST cover a unique axis:

- definition
- mechanism
- evidence
- comparison
- application
- criticism

NO duplication allowed.

━━━━━━━━━ STEP 3: SEARCH-OPTIMIZED QUERIES ━━━━━━━━━

Write queries like search engine inputs (NOT natural language questions).

Example:
BAD: "What is machine learning?"
GOOD: "machine learning definition supervised unsupervised reinforcement overview"

━━━━━━━━━ STEP 4: SEARCH TYPE + DOMAIN ━━━━━━━━━

Assign:
- search_type: encyclopedia | academic | statistical | news | comparison
- domain: one of (machine_learning, philosophy, software, economics, etc.)

━━━━━━━━━ STEP 5: PRIORITY ━━━━━━━━━

- 1 = core
- 2 = important
- 3 = optional depth

━━━━━━━━━ STEP 6: SELF-CHECK ━━━━━━━━━

Ensure:
- No redundancy
- No mixed domains
- Full coverage

━━━━━━━━━ OUTPUT ━━━━━━━━━

Return ONLY JSON:

{
  "query_type": "...",
  "query_scope": "...",
  "dominant_domain": "...",
  "sub_questions": [
    {
      "id": 1,
      "question": "...",
      "axis": "...",
      "search_type": "...",
      "priority": 1,
      "depends_on": [],
      "coverage_goal": "...",
      "domain": "..."
    }
  ],
  "coverage_note": "..."
}
""".strip()


# =========================
# Planner Agent
# =========================

async def planner_agent(
    llm: LLMClient,
    query: str,
    critique_feedback: str = ""
) -> List[Dict[str, Any]]:

    normalized = query.strip().lower()

    # =========================
    # Fast Path (optimized)
    # =========================
    if normalized.startswith(("what is", "define")) and not critique_feedback:
        concept = re.sub(r"^(what is|define)\s+", "", normalized).strip()

        return [
            {
                "id": 1,
                "question": f"{concept} definition formal explanation",
                "axis": "definition",
                "search_type": "encyclopedia",
                "priority": 1,
                "depends_on": [],
                "coverage_goal": "formal definition and meaning",
                "domain": "general",
            },
            {
                "id": 2,
                "question": f"{concept} how it works mechanism components",
                "axis": "mechanism",
                "search_type": "academic",
                "priority": 2,
                "depends_on": [1],
                "coverage_goal": "internal working and structure",
                "domain": "general",
            },
            {
                "id": 3,
                "question": f"{concept} real world applications examples",
                "axis": "application",
                "search_type": "comparison",
                "priority": 2,
                "depends_on": [1],
                "coverage_goal": "practical usage",
                "domain": "general",
            },
            {
                "id": 4,
                "question": f"{concept} limitations drawbacks challenges",
                "axis": "criticism",
                "search_type": "academic",
                "priority": 3,
                "depends_on": [2],
                "coverage_goal": "weaknesses and limitations",
                "domain": "general",
            },
        ]

    # =========================
    # Full LLM Planning
    # =========================

    feedback_block = f"\nCritique feedback: {critique_feedback}" if critique_feedback else ""

    user_prompt = f"""
Query: {query}
{feedback_block}

Generate a high-quality structured research plan.
Return JSON only.
"""

    payload: PlannerOutput = await llm.generate_json(
        PLANNER_SYSTEM_PROMPT,
        user_prompt
    )

    sub_questions = payload.get("sub_questions", [])

    # =========================
    # Post-processing (ELITE TOUCH)
    # =========================

    cleaned: List[Dict[str, Any]] = []
    seen = set()

    for item in sub_questions:
        q = item.get("question", "").strip().lower()

        if not q or q in seen:
            continue

        seen.add(q)

        cleaned.append({
            "id": item.get("id"),
            "question": item.get("question"),
            "axis": item.get("axis"),
            "search_type": item.get("search_type", "encyclopedia"),
            "priority": item.get("priority", 2),
            "depends_on": item.get("depends_on", []),
            "coverage_goal": item.get("coverage_goal", ""),
            "domain": item.get("domain", "general"),
        })

    return cleaned[:5]