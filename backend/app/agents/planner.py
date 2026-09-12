"""Planner Agent — research strategy as enforceable delegation contracts.

Fixes and upgrades in this version
----------------------------------
1. THE PLAN NOW OBEYS THE ORCHESTRATOR. `final = cleaned[:5]` hard-capped every
   plan at five sub-questions regardless of what the orchestrator computed, so
   the adaptive agent count had no effect on the actual work done. The planner
   takes `target_count` and honours it.

2. REQUIRED AXES ARE ENFORCED, NOT REQUESTED. The prompt has always demanded a
   statistical angle (Phase C) and a criticism angle (Phase D) because those two
   are what separate research from recall. Nothing verified compliance, and
   models routinely returned four background questions. `enforce_axis_coverage`
   deterministically injects any missing required axis.

3. PRIORITY-AWARE TRUNCATION KEEPS DIVERSITY. Trimming a plan by priority alone
   discards whole information types when the model assigns priority 1 to
   everything. Truncation now guarantees axis and search_type spread first, then
   fills by priority.

4. DEPENDENCIES ARE VALIDATED AND USED. `depends_on` was parsed and then
   ignored, including self-references and cycles. It is now sanitized and turned
   into execution waves, so dependent angles can be researched with their
   prerequisite's findings in hand instead of blind.

5. CONTRACTS CARRY SOURCE PREFERENCES. Each contract gains `preferred_domains`
   from the primary-source registry and a `specialist` role, so search can aim
   at publishers and the summarizer can load the right domain overlay.

6. SEMANTIC DEDUP IS ACTUALLY SEMANTIC. The old version normalized text,
   deleted three specific words, and compared for exact equality — which meant
   "solar panel costs 2025" and "cost of solar panels in 2025" both survived as
   distinct angles, wasting a whole agent on a duplicate search.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypedDict

from app.core.degradation import record_fallback
from app.core.llm import LLMClient
from app.core.logging import get_logger
from app.core.schemas import PlannerOutputModel

from app.agents.sources import build_primary_source_query, primary_source_hints

logger = get_logger(__name__)


# =========================
# Structured Types
# =========================

class SubQuestion(TypedDict, total=False):
    """Delegation Contract — the typed shape every sub-question must have.
    Validated by app.core.schemas.PlannerOutputModel before agents see it."""

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
    variants: List[str]
    agent: str
    tools: List[str]
    scope: List[str]
    output_format: str
    # --- additive contract fields ---
    specialist: str            # summarizer prompt overlay to load
    preferred_domains: List[str]
    primary_source_query: str  # site:-scoped variant aimed at publishers
    wave: int                  # execution wave from dependency order


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
    "legal",
    "policy",
    "academic",
    "general",
}

VALID_TOOLS = ["web_search", "fetch_content"]

VALID_AXES = {
    "definition", "mechanism", "application", "criticism", "comparison",
    "evidence", "history", "outlook", "risk", "cost", "regulation",
}

# Which search_type serves each axis when the planner has to synthesize a
# missing angle itself.
AXIS_SEARCH_TYPE: Dict[str, str] = {
    "definition": "encyclopedia",
    "mechanism": "academic",
    "application": "comparison",
    "criticism": "academic",
    "comparison": "comparison",
    "evidence": "statistical",
    "history": "encyclopedia",
    "outlook": "news",
    "risk": "academic",
    "cost": "statistical",
    "regulation": "news",
}

# Domain -> specialist overlay in summarizer.SPECIALIST_PROMPT_ADDITIONS.
DOMAIN_TO_SPECIALIST: Dict[str, str] = {
    "economics": "financial",
    "machine_learning": "technical",
    "software": "technical",
    "science": "scientific",
    "legal": "legal",
    "policy": "policy",
    "academic": "academic",
    "philosophy": "academic",
    "general": "general",
}


# =========================
# Utilities
# =========================

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def normalize_domain(domain: str) -> str:
    d = normalize_text(domain)
    return d if d in VALID_DOMAINS else "general"


def normalize_axis(axis: str) -> str:
    a = normalize_text(axis).replace(" ", "_")
    return a if a in VALID_AXES else "general"


def is_valid_question(q: str) -> bool:
    return len((q or "").split()) >= 3


def _clean_str_list(values: Any, limit: int = 5) -> List[str]:
    """String-list parser guard: keeps short non-empty strings, drops junk."""
    if not isinstance(values, list):
        return []
    cleaned: List[str] = []
    for v in values:
        text = str(v or "").strip()
        if text and len(text) <= 120 and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def diversity_coverage(sub_questions: List[Dict[str, Any]]) -> set:
    """Distinct valid search_types in a plan — its diversity contract."""
    return {
        q.get("search_type") for q in sub_questions
        if isinstance(q, dict) and q.get("search_type") in VALID_SEARCH_TYPES
    }


_QUESTION_STOPWORDS = {
    "what", "which", "how", "why", "when", "where", "who", "is", "are",
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "do",
    "does", "did", "with", "about", "definition", "overview",
    "introduction", "explanation", "explain", "examples", "example",
}


def _question_signature(question: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", (question or "").lower())
        if token not in _QUESTION_STOPWORDS
    }


def _question_similarity(a: str, b: str) -> float:
    sa, sb = _question_signature(a), _question_signature(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


def deduplicate_semantic(
    items: List[Dict[str, Any]], threshold: float = 0.75
) -> List[Dict[str, Any]]:
    """Drop sub-questions that would search for the same thing.

    Uses content-token overlap (coefficient, not Jaccard, so a short question
    subsumed by a longer one is caught) instead of exact string equality after
    deleting three words. Two questions on the same axis are held to a stricter
    bar than two on different axes: "cost of X" as evidence and "cost of X" as
    criticism are genuinely different research jobs.
    """
    result: List[Dict[str, Any]] = []
    for item in items:
        question = str(item.get("question", ""))
        axis = str(item.get("axis", ""))
        duplicate = False
        for kept in result:
            same_axis = str(kept.get("axis", "")) == axis
            bar = threshold if same_axis else threshold + 0.15
            if _question_similarity(question, str(kept.get("question", ""))) >= bar:
                duplicate = True
                break
        if not duplicate:
            result.append(item)
    return result


# =========================
# Prompt
# =========================

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

  AMBIGUITY — if the query has multiple distinct senses ("transformer":
  electrical device vs ML architecture; "apple": fruit vs company), emit
  one sense-disambiguating question PER sense ("transformer electrical
  device definition", "transformer machine learning model architecture")
  instead of letting evidence for different senses mash together.

  VARIANTS — every sub_question MUST include 1-2 variants: alternate
  phrasings with different keywords but the same intent. Different
  phrasings retrieve different sources; a question without variants
  leaves evidence on the table.
  question → "utility-scale solar installation costs per MW 2024"
  variants → ["residential solar price per watt 2024 SEIA",
              "solar farm capital expenditure benchmarks"]

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
  Priority is survival: the plan may be truncated to the top priorities.
  Assign priority 1 to the survey question AND the statistical question,
  priority 2 to criticism and key dimensions, priority 3 to the rest — so
  truncation keeps evidence and challenge angles, never just background.
  Use depends_on to list ids of sub-questions this one builds on. A
  dependency means the dependent question genuinely needs the earlier
  findings to be worth asking; do NOT chain every question linearly, that
  serializes research that could run in parallel.

  CONTRACT FIELDS — each sub-question is a delegation contract:
  agent: short role label, e.g. "financial_researcher" (derived from domain)
  tools: subset of [web_search, fetch_content] — the only tools that exist
  scope: 2-5 noun phrases bounding the sub-question
  output_format: always "structured_findings" (the pipeline's only consumer)

RULE 5 — RESPECT CRITIQUE FEEDBACK
  If critique_feedback is provided, generate sub_questions that
  specifically close the gaps it describes rather than repeating
  the original plan.

RULE 6 — CLASSIFY THE QUERY
  query_type:    factual | comparative | analytical | exploratory
  query_scope:   narrow | broad
  domain must be one of: machine_learning | software | philosophy |
  economics | science | legal | policy | academic | general

RULE 7 — CONCRETE QUESTIONS ONLY
  Every question must name a searchable noun AND the evidence it seeks.
  BAD  → "overview of solar energy"
  GOOD → "utility-scale solar installation costs per MW 2023-2025"
  A question that could be answered from general knowledge alone is a
  bad question — rewrite it to demand external evidence.

  PRIMARY SOURCES — prefer questions that would land on the organisation
  that PUBLISHED the fact (statistics agency, regulator, standards body,
  peer-reviewed venue) over ones that land on commentary about it. Name
  the publisher in the question when you know it ("IEA electricity
  capacity additions 2024", "SEC 10-K risk factors").

  TEMPORAL AWARENESS — time-sensitive questions (news, trends, outlook,
  statistics, "latest"/"recent") MUST carry the actual current year from
  the request date, never a hardcoded past year and never no year.

RULE 8 — COVERAGE NOTE WITH TEETH
  coverage_note must name the single most important angle the plan
  does NOT cover and why it was deprioritized — or state "full
  coverage: no major angle omitted" if that is genuinely true.

━━━ OUTPUT FORMAT ━━━

Return ONLY valid JSON. No markdown fences. No text outside JSON.

{
  "query_type": "<factual|comparative|analytical|exploratory>",
  "query_scope": "<narrow|broad>",
  "dominant_domain": "<machine_learning|software|philosophy|economics|science|legal|policy|academic|general>",
  "sub_questions": [
    {
      "id": 1,
      "question": "<specific, search-ready sub-question>",
      "axis": "<definition|mechanism|application|criticism|comparison|evidence|history|outlook>",
      "search_type": "<encyclopedia|academic|statistical|news|comparison>",
      "priority": 1,
      "depends_on": [],
      "coverage_goal": "<what this sub-question should establish>",
      "domain": "<same enum as dominant_domain>",
      "minimum_sources": 2,
      "stop_condition": "<when this sub-question's search can stop>",
      "variants": ["<1-2 alternate phrasings with different keywords, same intent>"],
      "agent": "<short role label, e.g. financial_researcher>",
      "tools": ["web_search"],
      "scope": ["<2-5 bounding noun phrases>"],
      "output_format": "structured_findings"
    }
  ],
  "coverage_note": "<one sentence: what would full coverage of this query require>"
}
""".strip()


# =========================
# Contract construction
# =========================

def _contract(
    *,
    index: int,
    question: str,
    axis: str,
    search_type: str,
    priority: int,
    domain: str,
    coverage_goal: str = "",
    minimum_sources: int = DEFAULT_MINIMUM_SOURCES,
    stop_condition: str = DEFAULT_STOP_CONDITION,
    variants: Optional[Sequence[str]] = None,
    depends_on: Optional[Sequence[int]] = None,
    agent: str = "",
    tools: Optional[Sequence[str]] = None,
    scope: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build one fully-populated delegation contract.

    Single construction point so every field (including the new source
    preferences) is present on contracts from the LLM path, the axis-repair
    path and the fallback path alike. Downstream code can rely on the shape
    instead of defaulting per call site.
    """
    axis = normalize_axis(axis)
    search_type = search_type if search_type in VALID_SEARCH_TYPES else AXIS_SEARCH_TYPE.get(axis, "encyclopedia")
    domain = normalize_domain(domain)
    specialist = DOMAIN_TO_SPECIALIST.get(domain, "general")
    hints = list(primary_source_hints(search_type, domain))
    return {
        "id": index,
        "question": question.strip(),
        "axis": axis,
        "search_type": search_type,
        "priority": max(1, min(3, int(priority or 2))),
        "depends_on": list(depends_on or []),
        "coverage_goal": coverage_goal,
        "domain": domain,
        "minimum_sources": max(1, int(minimum_sources)),
        "stop_condition": stop_condition or DEFAULT_STOP_CONDITION,
        "variants": list(variants or [])[:2],
        "agent": (agent or f"{specialist}_researcher")[:60],
        "tools": list(tools or ["web_search"]),
        "scope": list(scope or [])[:5],
        "output_format": "structured_findings",
        "specialist": specialist,
        "preferred_domains": hints,
        "primary_source_query": build_primary_source_query(question, search_type, domain),
        "wave": 0,
    }


# =========================
# Plan repair and shaping
# =========================

def sanitize_dependencies(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop dangling, self- and cyclic dependencies, then assign waves.

    A cycle or a self-reference silently disabled dependency-aware execution
    before (the field was never read, so nothing broke visibly — it just never
    worked). Waves are the payoff: wave 0 contracts run in parallel
    immediately, wave 1 contracts run after their prerequisites and can be
    given the earlier findings as context.
    """
    ids = {int(item["id"]) for item in plan if isinstance(item.get("id"), int)}
    for item in plan:
        raw = item.get("depends_on") or []
        if not isinstance(raw, (list, tuple)):
            raw = []
        deps: List[int] = []
        for value in raw:
            try:
                dep = int(value)
            except (TypeError, ValueError):
                continue
            if dep in ids and dep != int(item["id"]) and dep not in deps:
                deps.append(dep)
        item["depends_on"] = deps[:3]

    by_id = {int(item["id"]): item for item in plan}
    waves: Dict[int, int] = {}

    def _wave(node_id: int, seen: Set[int]) -> int:
        if node_id in waves:
            return waves[node_id]
        if node_id in seen:          # cycle: break it by treating as root
            by_id[node_id]["depends_on"] = []
            waves[node_id] = 0
            return 0
        seen = seen | {node_id}
        deps = by_id[node_id].get("depends_on") or []
        depth = 0 if not deps else 1 + max(_wave(d, seen) for d in deps)
        depth = min(depth, 2)        # never more than 3 waves: latency floor
        waves[node_id] = depth
        return depth

    for node_id in list(by_id):
        by_id[node_id]["wave"] = _wave(node_id, set())

    # A cycle-broken node had its depends_on cleared mid-recursion, but its
    # outer _wave frame still computed a depth from the OLD deps and
    # overwrote the memo — stranding the node in a later wave even though it
    # is now a root. Roots must run in wave 0.
    for item in by_id.values():
        if not item.get("depends_on") and int(item.get("wave", 0) or 0) > 0:
            item["wave"] = 0
    return plan


def enforce_axis_coverage(
    plan: List[Dict[str, Any]],
    query: str,
    required_axes: Sequence[str],
    *,
    domain: str = "general",
    minimum_sources: int = DEFAULT_MINIMUM_SOURCES,
    today: str = "",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Add a contract for every required axis the model omitted.

    Returns (plan, injected_axes). The injected questions are deliberately
    keyword-shaped rather than prose: they are search strings, and the whole
    reason they exist is that the model already failed to produce one.
    """
    present = {str(item.get("axis", "")) for item in plan}
    concept = _query_concept(query)
    year = _year_from(today)
    injected: List[str] = []
    next_id = max((int(item.get("id", 0)) for item in plan), default=0) + 1

    templates: Dict[str, Tuple[str, str, int]] = {
        "evidence": (
            f"{concept} statistics official data figures{year}", "statistical", 1,
        ),
        "criticism": (
            f"{concept} limitations criticism counter-evidence risks", "academic", 2,
        ),
        "comparison": (
            f"{concept} compared alternatives side by side{year}", "comparison", 2,
        ),
        "definition": (
            f"{concept} definition explanation overview", "encyclopedia", 1,
        ),
        "outlook": (
            f"{concept} outlook forecast recent developments{year}", "news", 2,
        ),
    }

    for axis in required_axes or ():
        axis = normalize_axis(axis)
        if axis in present or axis not in templates:
            continue
        question, search_type, priority = templates[axis]
        plan.append(
            _contract(
                index=next_id,
                question=question,
                axis=axis,
                search_type=search_type,
                priority=priority,
                domain=domain,
                coverage_goal=f"required {axis} coverage (injected by axis enforcement)",
                minimum_sources=minimum_sources,
            )
        )
        present.add(axis)
        injected.append(axis)
        next_id += 1

    if injected:
        logger.info("[Planner] injected required axes: %s", ", ".join(injected))
    return plan, injected


def select_plan(
    plan: List[Dict[str, Any]], target_count: int, required_axes: Sequence[str] = ()
) -> List[Dict[str, Any]]:
    """Trim to `target_count` while protecting axis and search_type diversity.

    Pure priority sorting was the bug: a model that marks everything priority 1
    made truncation arbitrary, and a plan could end up as three encyclopedia
    questions with no evidence angle. Selection order is:
      1. every required axis (one contract each, best priority)
      2. one contract per remaining unseen search_type
      3. the rest by (priority, id)
    """
    if target_count <= 0:
        return []
    ranked = sorted(
        plan, key=lambda item: (int(item.get("priority", 2)), int(item.get("id", 0)))
    )
    required = [normalize_axis(a) for a in (required_axes or ())]
    picked: List[Dict[str, Any]] = []
    picked_ids: Set[int] = set()

    def _take(item: Dict[str, Any]) -> None:
        picked.append(item)
        picked_ids.add(int(item.get("id", -1)))

    for axis in required:
        if len(picked) >= target_count:
            break
        for item in ranked:
            if int(item.get("id", -1)) in picked_ids:
                continue
            if str(item.get("axis", "")) == axis:
                _take(item)
                break

    seen_types = {str(item.get("search_type", "")) for item in picked}
    for item in ranked:
        if len(picked) >= target_count:
            break
        if int(item.get("id", -1)) in picked_ids:
            continue
        stype = str(item.get("search_type", ""))
        if stype not in seen_types:
            _take(item)
            seen_types.add(stype)

    for item in ranked:
        if len(picked) >= target_count:
            break
        if int(item.get("id", -1)) not in picked_ids:
            _take(item)

    picked.sort(key=lambda item: (int(item.get("priority", 2)), int(item.get("id", 0))))
    return picked


def _query_concept(query: str) -> str:
    text = re.sub(
        r"^\s*(what\s+is|what\s+are|who\s+is|define|explain|how\s+do(?:es)?|should\s+\w+)\s+",
        "",
        (query or "").strip(),
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip(" ?.!") or (query or "").strip()


def _year_from(today: str) -> str:
    match = re.search(r"(20\d{2})", today or "")
    return f" {match.group(1)}" if match else ""


# =========================
# Fallback Planner
# =========================

def fallback_plan(
    query: str,
    target_count: int = 4,
    required_axes: Sequence[str] = ("definition", "evidence", "criticism"),
    today: str = "",
) -> List[Dict[str, Any]]:
    """Deterministic plan for when the LLM call fails.

    Now shaped by the same axis contract as a real plan (an evidence angle and a
    criticism angle, not four definition variants) and sized to the
    orchestrator's target, so a degraded run is a smaller research plan rather
    than a different, weaker kind of plan.
    """
    concept = _query_concept(query)
    year = _year_from(today)
    blueprint: List[Tuple[str, str, str, int]] = [
        (f"{concept} definition explanation overview", "definition", "encyclopedia", 1),
        (f"{concept} statistics official data figures{year}", "evidence", "statistical", 1),
        (f"{concept} limitations criticism counter-evidence risks", "criticism", "academic", 2),
        (f"{concept} mechanism how it works components", "mechanism", "academic", 2),
        (f"{concept} real world applications examples compared", "application", "comparison", 2),
        (f"{concept} recent developments outlook{year}", "outlook", "news", 3),
    ]
    ordered = [b for b in blueprint if b[1] in set(required_axes or ())] + [
        b for b in blueprint if b[1] not in set(required_axes or ())
    ]
    plan = [
        _contract(
            index=i + 1,
            question=question,
            axis=axis,
            search_type=search_type,
            priority=priority,
            domain="general",
            coverage_goal=f"fallback {axis} coverage",
        )
        for i, (question, axis, search_type, priority) in enumerate(
            ordered[: max(1, target_count)]
        )
    ]
    return sanitize_dependencies(plan)


# =========================
# Planner Agent
# =========================

async def planner_agent(
    llm: LLMClient,
    query: str,
    critique_feedback: str = "",
    today: str = "",
    context_snippets: List[str] | None = None,
    target_count: Optional[int] = None,
    required_axes: Sequence[str] = (),
    minimum_sources: int = DEFAULT_MINIMUM_SOURCES,
) -> List[Dict[str, Any]]:
    """Produce a validated, axis-complete, dependency-ordered research plan.

    New optional arguments come from `OrchestrationPlan.targets`; omitting them
    reproduces the previous behaviour (five sub-questions, no axis enforcement),
    so existing call sites keep working while the workflow migrates.
    """
    target = int(target_count) if target_count else 5
    target = max(1, min(8, target))

    feedback_block = (
        f"\nCritique feedback: {critique_feedback}" if critique_feedback else ""
    )
    date_block = (
        f"\nToday is {today.strip()} — use this year in time-sensitive questions."
        if today.strip() else ""
    )
    axis_block = ""
    if required_axes:
        axis_block = (
            "\nMandatory axes — the plan MUST contain one sub-question for each "
            f"of: {', '.join(normalize_axis(a) for a in required_axes)}. A plan "
            "missing any of them will be repaired automatically, and the "
            "repaired question will be cruder than one you write yourself."
        )
    budget_block = (
        f"\nProduce at most {target} sub-questions — this is a hard budget, so "
        "spend it on the highest-value angles rather than listing everything."
    )
    context_block = ""
    snippets = [s for s in (context_snippets or []) if str(s).strip()]
    if snippets:
        context_block = (
            "\nWeb context — top search results for the raw query. Use it to "
            "ground and disambiguate the sub-questions (real terminology, "
            "entities, and numbers the plan should target), never to answer "
            "the query itself:\n"
            + "\n".join(f"- {s}" for s in snippets[:6])
        )

    user_prompt = f"""
Query: {query}
{feedback_block}
{date_block}
{axis_block}
{budget_block}
{context_block}

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
        record_fallback("planner")
        return fallback_plan(query, target, required_axes or ("definition", "evidence", "criticism"), today)

    sub_questions = payload.get("sub_questions", []) if isinstance(payload, dict) else []
    if not sub_questions:
        logger.warning("[Planner] Empty LLM output, using fallback")
        record_fallback("planner")
        return fallback_plan(query, target, required_axes or ("definition", "evidence", "criticism"), today)

    dominant_domain = normalize_domain(
        str(payload.get("dominant_domain", "general")) if isinstance(payload, dict) else "general"
    )

    # ---------------- post-processing ----------------

    cleaned: List[Dict[str, Any]] = []
    for i, item in enumerate(sub_questions):
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "") or "").strip()
        if not q or not is_valid_question(q):
            continue

        raw_variants = item.get("variants", [])
        variants: List[str] = []
        if isinstance(raw_variants, list):
            for v in raw_variants[:2]:
                vs = str(v or "").strip()
                if vs and is_valid_question(vs) and normalize_text(vs) != normalize_text(q):
                    variants.append(vs)

        try:
            declared_id = int(item.get("id", i + 1))
        except (TypeError, ValueError):
            declared_id = i + 1

        cleaned.append(
            _contract(
                index=declared_id,
                question=q,
                axis=str(item.get("axis", "general") or "general"),
                search_type=str(item.get("search_type", "") or ""),
                priority=int(item.get("priority", 2) or 2)
                if str(item.get("priority", "2")).strip().isdigit() else 2,
                domain=str(item.get("domain", dominant_domain) or dominant_domain),
                coverage_goal=str(item.get("coverage_goal", "") or ""),
                minimum_sources=max(
                    minimum_sources,
                    int(item.get("minimum_sources", minimum_sources) or minimum_sources)
                    if str(item.get("minimum_sources", "")).strip().isdigit()
                    else minimum_sources,
                ),
                stop_condition=str(item.get("stop_condition", "") or "").strip(),
                variants=variants,
                depends_on=item.get("depends_on", []),
                agent=str(item.get("agent", "") or "").strip(),
                tools=[
                    t for t in _clean_str_list(item.get("tools", ["web_search"]))
                    if t in VALID_TOOLS
                ] or ["web_search"],
                scope=_clean_str_list(item.get("scope", [])),
            )
        )

    # Deduplicate ids so dependency resolution and selection stay coherent.
    seen_ids: Set[int] = set()
    for position, item in enumerate(cleaned, start=1):
        if item["id"] in seen_ids or item["id"] <= 0:
            item["id"] = max(seen_ids, default=0) + 1
        seen_ids.add(item["id"])

    cleaned = deduplicate_semantic(cleaned)
    cleaned, injected = enforce_axis_coverage(
        cleaned, query, required_axes,
        domain=dominant_domain, minimum_sources=minimum_sources, today=today,
    )

    # Axis enforcement may push the plan over budget; selection decides what
    # survives, and required axes are protected inside it.
    final = select_plan(cleaned, target, required_axes)
    final = sanitize_dependencies(final)

    if not final:
        logger.warning("[Planner] All filtered out, fallback used")
        record_fallback("planner")
        return fallback_plan(query, target, required_axes or ("definition", "evidence", "criticism"), today)

    logger.info(
        "[Planner] %d contract(s), axes=%s, search_types=%s, injected=%s",
        len(final),
        sorted({item["axis"] for item in final}),
        sorted(diversity_coverage(final)),
        injected or "none",
    )
    return final


def execution_waves(plan: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group contracts into dependency-ordered waves for parallel dispatch.

    Wave members are independent and may run concurrently; a later wave starts
    only after the previous one finishes, and its contracts can be handed the
    earlier findings. This is the mechanism that makes "adaptive orchestration"
    more than a synonym for "fan out everything at once".
    """
    waves: Dict[int, List[Dict[str, Any]]] = {}
    for contract in plan or ():
        waves.setdefault(int(contract.get("wave", 0) or 0), []).append(contract)
    return [waves[key] for key in sorted(waves)]
