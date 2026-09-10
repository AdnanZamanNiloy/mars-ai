from __future__ import annotations

from app.core.logging import get_logger
import re
from typing import Any, Dict, List, TypedDict

from app.core.llm import LLMClient
from app.core.schemas import PlannerOutputModel

logger = get_logger(__name__)


# =========================
# Structured Types
# =========================

class SubQuestion(TypedDict, total=False):
    """Delegation Contract (Phase 2.6) — the single typed shape every
    sub-question flowing through the pipeline must have. Validated by
    app.core.schemas.PlannerOutputModel before agents ever see it."""

    id: int
    question: str
    axis: str
    search_type: str
    priority: int
    depends_on: List[int]
    coverage_goal: str
    domain: str
    minimum_sources: int
    stop_condition: str


DEFAULT_MINIMUM_SOURCES = 2
DEFAULT_STOP_CONDITION = "sufficient evidence for this axis"


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


def diversity_coverage(sub_questions: List[Dict[str, Any]]) -> set:
    """Distinct valid search_types in a plan — its diversity contract."""
    return {
        q.get("search_type") for q in sub_questions
        if isinstance(q, dict) and q.get("search_type") in VALID_SEARCH_TYPES
    }


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


PLANNER_SYSTEM_PROMPT = """
You are the Planner Agent in a multi-agent research pipeline.
Your job is to decompose a research query into a set of specific,
search-ready sub-questions. The plan you produce determines what the
Searcher fetches and what the rest of the pipeline can ever know.

━━━ YOUR RULES ━━━

RULE 1 — QUESTIONS, NOT TOPIC LABELS
  Each sub_question must be a specific, search-ready question or
  keyword query a search engine can act on.
  BAD  → "energy"
  GOOD → "cost per megawatt-hour of nuclear vs solar energy 2024"

RULE 2 — COVER IN PHASES, NOT JUST AXES
  Plan as a researcher would: survey first, then drill down.
  Phase A (survey): 1 question establishing landscape/definition.
  Phase B (dimensions): 1-2 questions on the distinct angles the survey
  would reveal (mechanism, application, history, outlook).
  Phase C (evidence): at least 1 statistical question hunting numbers,
  measurements, or official data — never leave a plan without one.
  Phase D (challenge): at least 1 criticism question seeking limitations,
  counter-evidence, or risks — this feeds the contradiction engine.
  Cover at least 2 distinct axes overall:
  definition | mechanism | application | criticism | comparison |
  evidence | history | outlook
  Do not restate the same angle twice.

  DIVERSITY TABLE — spread the plan across information types by using at
  least 3 distinct search_types (they are the plan's diversity contract):
  facts/data      → statistical   (numbers, market data, official stats)
  cases/examples  → comparison    (A-vs-B, implementations in the wild)
  expert views    → academic      (papers, studies, technical depth)
  trends/outlook  → news          (recent developments, forecasts)
  background      → encyclopedia  (definitions, established facts)
  Name the information type each question serves in its coverage_goal.

RULE 3 — SEARCH TYPE PER QUESTION
  Assign exactly one search_type:
  encyclopedia  → background, definitions, established facts
  academic      → papers, studies, technical depth
  statistical   → numbers, market data, official statistics
  news          → recent developments, current events
  comparison    → direct A-vs-B comparisons

RULE 4 — PRIORITY AND DEPENDENCIES
  priority 1 → essential, must be searched first
  priority 2 → important, strengthens the answer
  priority 3 → nice to have
  Priority is survival: the plan may be truncated to the top priorities
  (small runs keep only 2-3 questions). Assign priority 1 to the survey
  question AND the statistical question, priority 2 to criticism and key
  dimensions, priority 3 to the rest — so truncation keeps evidence and
  challenge angles, never just background.
  Use depends_on to list ids of sub-questions this one builds on.

RULE 5 — RESPECT CRITIQUE FEEDBACK
  If critique_feedback is provided, generate sub_questions that
  specifically close the gaps it describes rather than repeating
  the original plan.

RULE 6 — CLASSIFY THE QUERY
  query_type:    factual | comparative | analytical | exploratory
  query_scope:   narrow | broad
  domain must be one of: machine_learning | software | philosophy |
  economics | science | general

RULE 7 — CONCRETE QUESTIONS ONLY
  Every question must name a searchable noun AND the evidence it seeks.
  BAD  → "overview of solar energy"
  GOOD → "utility-scale solar installation costs per MW 2023-2025"
  A question that could be answered from general knowledge alone is a
  bad question — rewrite it to demand external evidence.

RULE 8 — COVERAGE NOTE WITH TEETH
  coverage_note must name the single most important angle the plan
  does NOT cover and why it was deprioritized — or state "full
  coverage: no major angle omitted" if that is genuinely true.

━━━ OUTPUT FORMAT ━━━

Return ONLY valid JSON. No markdown fences. No text outside JSON.

{
  "query_type": "<factual|comparative|analytical|exploratory>",
  "query_scope": "<narrow|broad>",
  "dominant_domain": "<machine_learning|software|philosophy|economics|science|general>",
  "sub_questions": [
    {
      "id": 1,
      "question": "<specific, search-ready sub-question>",
      "axis": "<definition|mechanism|application|criticism|comparison|...>",
      "search_type": "<encyclopedia|academic|statistical|news|comparison>",
      "priority": 1,
      "depends_on": [],
      "coverage_goal": "<what this sub-question should establish>",
      "domain": "<same enum as dominant_domain>",
      "minimum_sources": 2,
      "stop_condition": "<when this sub-question's search can stop, e.g. 'sufficient evidence for this axis'>"
    }
  ],
  "coverage_note": "<one sentence: what would full coverage of this query require>"
}
""".strip()


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
            "minimum_sources": DEFAULT_MINIMUM_SOURCES,
            "stop_condition": DEFAULT_STOP_CONDITION,
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
            "minimum_sources": DEFAULT_MINIMUM_SOURCES,
            "stop_condition": DEFAULT_STOP_CONDITION,
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
            "minimum_sources": DEFAULT_MINIMUM_SOURCES,
            "stop_condition": DEFAULT_STOP_CONDITION,
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
            "minimum_sources": DEFAULT_MINIMUM_SOURCES,
            "stop_condition": DEFAULT_STOP_CONDITION,
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

    # No fast-path bypass: every query goes through LLM planning with the
    # methodology prompt above. The per-run cost governor caps spend, so the
    # old "what is" shortcut only saved pennies while guaranteeing generic
    # plans on the most common query shape. fallback_plan remains the
    # failure fallback below — determinism on errors, never on bypass.

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
            response_model=PlannerOutputModel,
        )
    except Exception as e:
        logger.error(f"[Planner] LLM failed: {e}", exc_info=e)
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
            "minimum_sources": max(1, int(item.get("minimum_sources", DEFAULT_MINIMUM_SOURCES))),
            "stop_condition": str(item.get("stop_condition", "")).strip() or DEFAULT_STOP_CONDITION,
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