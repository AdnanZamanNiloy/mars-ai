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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, TypedDict

from app.core.degradation import record_fallback
from app.core.llm import LLMClient
from app.core.logging import get_logger
from app.core.schemas import PlannerOutputModel, PlanningDirectiveModel
from app.core.usage import set_stage_hint

from app.agents.sources import build_dimension_primary_query, primary_source_hints

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
    sense: str                 # intent sense label this contract researches ("" = unambiguous)


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
    "engineering",
    "general",
}

VALID_TOOLS = ["web_search", "fetch_content"]

VALID_AXES = {
    "definition", "mechanism", "application", "criticism", "comparison",
    "evidence", "history", "outlook", "risk", "cost", "regulation",
    # --- frontier six-axis research tracks (equal weight) ---
    "capability", "infrastructure", "economics", "adoption", "safety",
    "counter_evidence",
}

# The six mandatory frontier-research tracks for "current state and trends of
# AI" style queries. Equal weight: every plan must carry at least one contract
# per axis, regardless of how many the model volunteers. Counter-evidence is
# forced on EVERY query, not only ambiguous ones.
FRONTIER_AXES: Tuple[str, ...] = (
    "capability",
    "infrastructure",
    "economics",
    "adoption",
    "regulation",
    "safety",
)
# Always-on adversarial sub-track. This is the axis that keeps a report from
# being a press release: it searches specifically for over-hype, plateau,
# ROI-negative and capability-overstated arguments.
COUNTER_EVIDENCE_AXIS = "counter_evidence"

# Membership set for the front-half of the frontier taxonomy; the counter-
# evidence track is added separately because it is always-on.
FRONTIER_AXIS_SET = frozenset((*FRONTIER_AXES, COUNTER_EVIDENCE_AXIS))

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
    # frontier tracks: capability/benchmarks are academic, infrastructure and
    # economics are statistical, adoption news, safety academic, and the
    # counter-evidence track is deliberately academic+news (skeptical essays,
    # replication failures, plateau analyses live in both).
    "capability": "academic",
    "infrastructure": "statistical",
    "economics": "statistical",
    "adoption": "news",
    "safety": "academic",
    "counter_evidence": "academic",
}

# Literal search questions for each frontier track. Used both to synthesize a
# missing contract and as the canonical wording the axis-coverage gate matches
# against. Each is phrased to pull PRIMARY and non-Western material where it
# exists, per the source-integrity requirement.
FRONTIER_AXIS_QUESTIONS: Dict[str, str] = {
    "capability": (
        "frontier AI model capability trajectory and benchmark results "
        "(reasoning, coding, multimodal, agentic, long-context, "
        "inference-time scaling) primary technical reports"
    ),
    "infrastructure": (
        "AI compute infrastructure and constraints: chips, datacenters, "
        "energy and power demand, supply chain, manufacturing capacity "
        "official reports and statistics"
    ),
    "economics": (
        "AI economics and investment: capex, funding, valuations, ROI, "
        "enterprise pilot success and failure rates, bubble indicators "
        "financial filings and investor reports"
    ),
    "adoption": (
        "AI adoption and diffusion: enterprise, consumer, education, "
        "geographic and demographic unevenness official surveys and statistics"
    ),
    "regulation": (
        "AI governance and regulation as reactive context: US, EU, China and "
        "global rules, enforcement and compliance official regulatory texts"
    ),
    "safety": (
        "AI safety, alignment and risk: progress versus capability, "
        "incidents, evaluations and open debates primary research"
    ),
    "counter_evidence": (
        "skeptical AI analysis: over-hype, capability plateau, scaling limits, "
        "ROI-negative enterprise results, replication failures and strongest "
        "counterarguments from independent researchers"
    ),
}


# Domain -> specialist overlay in summarizer.SPECIALIST_PROMPT_ADDITIONS.
DOMAIN_TO_SPECIALIST: Dict[str, str] = {
    "economics": "financial",
    "machine_learning": "technical",
    "software": "technical",
    "engineering": "technical",
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


# Ordered (label-fragment, canonical-axis) aliases. Common wordings for the
# canonical retrieval categories are recognized so a model-chosen dimension
# ("head-to-head comparison") still maps onto the retrieval/section machinery,
# while genuinely query-specific dimensions keep their own label. Order
# matters: more specific fragments are matched first.
_AXIS_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("counter-evidence", "criticism"),
    ("counterevidence", "criticism"),
    ("criticism", "criticism"),
    ("critique", "criticism"),
    ("limitation", "criticism"),
    ("drawback", "criticism"),
    ("downside", "criticism"),
    ("failure mode", "risk"),
    ("risk", "risk"),
    ("hazard", "risk"),
    ("safety", "risk"),
    ("head-to-head", "comparison"),
    ("comparison", "comparison"),
    ("compared", "comparison"),
    ("versus", "comparison"),
    ("trade-off", "comparison"),
    ("tradeoff", "comparison"),
    ("cost", "cost"),
    ("price", "cost"),
    ("financing", "cost"),
    ("financial", "cost"),
    ("funding", "cost"),
    ("budget", "cost"),
    ("efficiency", "cost"),
    ("mechanism", "mechanism"),
    ("how it works", "mechanism"),
    ("causal", "mechanism"),
    ("cause", "mechanism"),
    ("driver", "mechanism"),
    ("root cause", "mechanism"),
    ("post-mortem", "mechanism"),
    ("definition", "definition"),
    ("what is", "definition"),
    ("overview", "definition"),
    ("background", "definition"),
    ("outlook", "outlook"),
    ("forecast", "outlook"),
    ("projection", "outlook"),
    ("trend", "outlook"),
    ("future", "outlook"),
    ("evidence", "evidence"),
    ("data", "evidence"),
    ("statistic", "evidence"),
    ("quantitative", "evidence"),
    ("measurement", "evidence"),
    ("application", "application"),
    ("use case", "application"),
    ("implementation", "application"),
    ("history", "history"),
    ("origin", "history"),
    ("timeline", "history"),
    ("regulation", "regulation"),
    ("regulatory", "regulation"),
    ("policy", "regulation"),
    ("governance", "regulation"),
    ("legal", "regulation"),
)


def dimension_to_axis(dimension: str, search_type: str = "") -> str:
    """Canonical retrieval axis for a model-chosen dimension label.

    The dynamic dimensions are free-form text; this maps the ones that clearly
    name a canonical category onto that category's retrieval/section behavior,
    preferring a `search_type` implied by the dimension when it is explicit.
    A genuinely query-specific dimension ("policy options", "institutional
    failure") returns its normalized slug unchanged — downstream code holds
    unknown axes as first-class string keys, so it still gets its own section.
    """
    text = normalize_text(dimension)
    if not text:
        return "general"
    if search_type == "statistical" and not any(
        c in text for c in ("risk", "failure", "criticism", "counter")
    ):
        return "evidence"
    for fragment, axis in _AXIS_ALIASES:
        if fragment in text:
            return axis
    slug = text.replace(" ", "_")
    return slug if slug in VALID_AXES else slug


def axis_search_type(axis: str, *, default: str = "academic") -> str:
    """Retrieval type for an axis — canonical map first, else a text hint."""
    if axis in AXIS_SEARCH_TYPE:
        return AXIS_SEARCH_TYPE[axis]
    text = axis.replace("_", " ")
    if any(c in text for c in ("cost", "price", "financ", "evidence", "stat", "number", "data")):
        return "statistical"
    if any(c in text for c in ("comparison", "compared", "trade", "versus", "option", "altern")):
        return "comparison"
    if any(c in text for c in ("outlook", "forecast", "trend", "projection", "future", "recent")):
        return "news"
    if any(c in text for c in ("definition", "background", "overview", "history")):
        return "encyclopedia"
    return default



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
  Cover at least 3 distinct axes overall:
  definition | mechanism | application | criticism | comparison |
  evidence | history | outlook
  Do not restate the same angle twice.

  NON-OVERLAP IS MANDATORY. Two sub-questions overlap when a single source
  could answer both, or when they differ only in phrasing. If you cannot
  name the DISTINCT evidence each question would retrieve, you have written
  the same question twice — replace one with a genuinely different angle.
  BAD  → "AI trends 2026" and "current AI developments" (same retrieval)
  GOOD → "enterprise agentic AI adoption rate 2026" and "scaling-law
          diminishing returns evidence 2025-2026" (different sources)

  WHY ANGLE — for any analytical, trend, or comparative query, at least one
  sub-question must target MECHANISM/CAUSATION: why the trend is happening,
  what drives it, how the mechanism works, or what trade-off explains it.
  A plan of only "what is X" and "X statistics" answers what, never why.

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
      "axis": "<the research dimension this sub-question serves — a short lowercase label. Prefer the required-dimension labels supplied in the request when they apply; otherwise name the dimension yourself (e.g. 'cost and financing', 'mechanism of action', 'counter-evidence'). Do not use 'general'.>",
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
# Dynamic planning (Phase 2)
# =========================

# The old pipeline asked the plan model for an enum axis and then injected a
# hard-coded template (mechanism/outlook/risk/cost/history/regulation) whenever
# a required axis was missing. Measured across eight diverse live queries, that
# made different query TYPES converge: every plan carried
# ["evidence", "criticism"], with mechanism/definition/outlook bolted on
# regardless of whether the question was a definition, a decision, a mechanism
# explanation, a causal post-mortem or a quantitative forecast. This stage runs
# FIRST and asks a model which dimensions the SPECIFIC query needs; the plan
# below must cover whatever it returns, so the required-axis machinery enforces
# the model's dimensions instead of a generic template. See
# test_dynamic_planning.py for the diversity invariant.
PLANNING_DIRECTIVE_PROMPT = """
You are the Planning Directive stage of a multi-agent research pipeline.
Your ONLY job: decide which research dimensions THIS SPECIFIC query needs.

You are not writing a report and not choosing from a fixed menu. You are
writing the dimension list that the research plan will be measured against.

━━━ METHOD ━━━
1. Read the query and identify what KIND of answer it demands:
   definition / mechanism / comparison / decision / causal explanation /
   quantitative forecast / controversy / trend / feasibility / cost.
2. Name the dimensions that would have to be researched to answer it WELL.
   A dimension is a distinct kind of evidence — not a topic.
   GOOD dimension → "mechanism of action", "cost per unit", "failure modes"
   BAD  dimension → "information", "background", "details"   (topic labels)

━━━ MANDATORY RULES ━━━
R1 — DERIVE THE DIMENSIONS FROM THE QUERY. Do not emit a stock list.
   A "what is X?" question needs fewer, more definitional dimensions than a
   "should we do X over 20 years?" decision question, which needs cost,
   feasibility and risk dimensions. A "why did X happen?" question needs
   causal/post-mortem dimensions, not outlook. Different query types MUST
   produce different dimension sets. Copy-pasting the same set onto every
   query is the failure this stage exists to prevent.

R2 — A WHY/MECHANISM dimension is required ONLY when the query asks how
   something works, why something happens, or how a mechanism produces its
   effect. A pure "what is X" definition or a pure "how much" figure question
   does NOT require one.

R3 — A QUANTITATIVE dimension (numbers, statistics, measurements, forecasts)
   is required ONLY when the query asks for amounts, trends, comparisons of
   magnitude, or projections. Do not force numbers onto a conceptual query.

R4 — If the query is a DECISION ("should we", "should X"), include a
   cost/trade-off dimension AND a risk-or-feasibility dimension.
   If it is COMPARATIVE ("X vs Y"), include a head-to-head comparison
   dimension. If it is CAUSAL ("why did", "what caused"), include a
   causal-mechanism dimension and a counterfactual/alternative-explanation
   dimension. If it is CONTESTED, include a strongest-counter-evidence
   dimension.

R5 — 3 to 6 dimensions. More is not better: each one becomes a research
   contract and a search budget. Name ONLY what this query needs.

R6 — `must_cover` is the 1-3 dimensions whose absence would make the answer
   fail the question. They may repeat entries from `dimensions` exactly.

━━━ OUTPUT FORMAT ━━━

Return ONLY valid JSON. No markdown fences. No text outside JSON.

{
  "query_type": "<factual|comparative|analytical|exploratory>",
  "dominant_domain": "<machine_learning|software|philosophy|economics|science|legal|policy|academic|general>",
  "reasoning": "<one sentence: what kind of answer this query wants>",
  "dimensions": [
    "<specific dimension this query needs, 2-4 words>",
    "<...>"
  ],
  "must_cover": ["<1-3 of the dimensions above that are indispensable>"],
  "preferred_search_types": ["<optional: statistical|academic|news|comparison|encyclopedia the dimensions would use>"],
  "coverage_note": "<what full coverage would require for THIS query>"
}
""".strip()


# Deterministic dimensions by coarse query shape. This is the fallback tuple
# for `_heuristic_dimensions` when the query text carries no usable signal at
# all — kept deliberately coarse so the LLM directive, not a template, is what
# normally drives the plan.
_HEURISTIC_DIMENSION_DEFAULTS: Tuple[str, ...] = ("background", "evidence", "criticism")


def _heuristic_dimensions(query: str, intent: Optional[Dict[str, Any]] = None) -> List[str]:
    """Deterministic, query-type-aware dimension list for the fallback path.

    Mirrors the LLM directive's rules with lexical signals only (AGENTS.md
    4.7): a decision query gets cost/risk dimensions, a comparative query gets
    a head-to-head dimension, a causal query gets post-mortem dimensions —
    genuinely different sets per query type, never the old fixed template.
    """
    intent = intent or {}
    text = f" {(query or '').lower().strip()} "
    qtype = str(intent.get("query_type", "") or "").strip().lower()
    if qtype not in ("factual", "comparative", "analytical", "exploratory"):
        # Local import: orchestrator is a leaf module that must not import the
        # planner (planner -> orchestrator would close a cycle at import time).
        from app.agents.orchestrator import classify_query_type

        qtype = classify_query_type(query)
    dims: List[str] = []

    def _add(name: str) -> None:
        if name not in dims:
            dims.append(name)

    is_decision = any(
        m in text
        for m in ("should we", "should i", "should our", "should the",
                  "should ", "invest", "worth it", "recommend", "decide")
    )
    is_comparative = qtype == "comparative" or any(
        m in text for m in (" vs ", " versus ", "compare", "compared", "comparison")
    )
    is_causal = any(
        m in text
        for m in ("what caused", "why did", "why does", "why is", "cause of",
                  "cause ", "reasons for", "root cause", "how did", "led to")
    )
    is_mechanism = qtype == "analytical" or any(
        m in text
        for m in ("how does", "how do", "how is", "how it works", "mechanism",
                  "work and", "works")
    )
    is_quant = any(
        m in text
        for m in ("how much", "how many", "cost", "price", "market size",
                  "growth", "rate", "percentage", "percent", "share",
                  "forecast", "projection", "statistics", "by 20")
    )
    is_contested = any(
        m in text
        for m in ("harmful", "controversial", "debate", "criticism", "myth",
                  "safe", "danger", "ethical", "controversial", "bias")
    )

    if is_decision:
        _add("policy options")
        _add("cost and financing")
        _add("risk and feasibility")
        _add("evidence and projections")
    elif is_causal:
        _add("causal mechanism")
        _add("alternative explanations")
        _add("institutional and regulatory failures")
    elif is_comparative:
        _add("head-to-head comparison")
        _add("cost per unit")
        _add("operational trade-offs")
        _add("evidence and data")
    else:
        _add("definition")
        if is_mechanism:
            _add("mechanism of action")
        _add("evidence and data")

    if is_quant and not any("evidence" in d for d in dims):
        _add("evidence and projections")
    if is_contested:
        _add("counter-evidence and criticism")
    if not any("criticism" in d or "counter" in d for d in dims):
        _add("criticism and limitations")

    # Preserve the historical WHY dimension for analytical/trend queries even
    # when the lexical shape above produced only broad dimensions.
    if qtype in ("analytical", "exploratory") and not any(
        "mechanism" in d or "cause" in d for d in dims
    ):
        _add("underlying drivers")

    return dims[:6] or list(_HEURISTIC_DIMENSION_DEFAULTS)


def _coarse_query_type(query: str) -> str:
    """query_type classification for the directive stage (leaf import)."""
    from app.agents.orchestrator import classify_query_type

    return classify_query_type(query)


# =========================
# Deterministic plan validation
# =========================

# A required angle is "covered" when any existing dimension or must_cover entry
# carries one of these vocabulary fragments. These are the SAME words the
# directive prompt (R3/R4) and the orchestrator classifiers already use, so the
# validator speaks the pipeline's existing vocabulary instead of inventing a
# parallel one. Reuse `dimension_to_axis` for the semantic half of matching.
_REQUIRED_ANGLE_TOKENS: Dict[str, Tuple[str, ...]] = {
    "quantitative": (
        "quantit", "statistic", "data", "metric", "number", "figure",
        "cost", "price", "growth", "projection", "forecast",
    ),
    "decision": (
        "decision", "tradeoff", "trade-off", "option", "policy",
        "risk", "comparison", "scenario",
    ),
    "contested": (
        "critic", "counter", "limitation", "risk", "controvers",
        "oppos", "alternative",
    ),
    "comparison": (
        "comparison", "compar", "head-to-head", "versus", "vs", "tradeoff",
    ),
}

# Injected requirement label -> the canonical axis its synthesized contract
# takes. Used to prove a newly-appended angle is NOT already served by an
# existing contract (`dimension_to_axis` maps the covered label the same way),
# so the validator never adds a dimension a plan already covers.
_REQUIRED_ANGLE_AXIS: Dict[str, str] = {
    "quantitative": "evidence",
    "decision": "comparison",
    "contested": "criticism",
    "comparison": "comparison",
}


def _angle_token_covered(label: str, angle: str) -> bool:
    """Lexical coverage test or()ed with the angle's canonical-axis mapping.

    "cost per unit" is a quantitative angle by vocabulary; "head-to-head
    comparison" is served by the canonical `comparison` axis even though it
    carries no literal "compar"/"versus" marker in every wording.
    """
    tokens = _REQUIRED_ANGLE_TOKENS[angle]
    text = normalize_text(label)
    if any(tok in text for tok in tokens):
        return True
    return dimension_to_axis(label) == _REQUIRED_ANGLE_AXIS[angle]


def _dedupe_exact_dimensions(dims: Sequence[str]) -> Tuple[List[str], bool]:
    """Collapse exactly-normalized duplicate dimensions, preserving order.

    Only exact duplicates (whitespace/case-insensitive) are dropped: near
    duplicates are left to `select_plan`/`deduplicate_semantic`, which own
    semantic overlap and must not be second-guessed here. Order and the first
    occurrence of each label are always preserved.
    """
    result: List[str] = []
    seen: Set[str] = set()
    duplicate_flag = False
    for raw in dims or ():
        name = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not name:
            continue
        key = normalize_text(name)
        if key in seen:
            duplicate_flag = True
            continue
        seen.add(key)
        result.append(name)
    return result, duplicate_flag


def validate_plan_dimensions(
    query: str,
    dimensions: Sequence[str],
    must_cover: Sequence[str],
    *,
    complexity: Any = None,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Validate planned dimensions against what the query actually requires.

    Fully deterministic, no LLM calls (AGENTS.md 4.7): reuses the orchestrator's
    `score_complexity` classifiers and the planner's own `dimension_to_axis`.
    Missing required angles are APPENDED (never replace/reorder the model's
    plan) and added to `must_cover` so `enforce_axis_coverage` turns each into a
    real delegation contract. Non-required angles are never dropped, so a valid
    plan comes back byte-identical. Any classifier failure skips validation
    rather than raising — an empty plan is returned unchanged.
    """
    dims, duplicate_flag = _dedupe_exact_dimensions(dimensions)
    cover, _ = _dedupe_exact_dimensions(must_cover)
    # must_cover entries that do not name a planned dimension are dropped: every
    # must_cover entry must correspond to a dimension that becomes a contract.
    dim_keys = {normalize_text(d) for d in dims}
    cover = [c for c in cover if normalize_text(c) in dim_keys]
    if not dims:
        return dims, cover, {"applied": False, "reason": "no_dimensions"}

    try:
        if complexity is None:
            from app.agents.orchestrator import score_complexity

            complexity = score_complexity(query)
        required_angles: List[str] = []
        if complexity.needs_quantitative:
            required_angles.append("quantitative")
        if complexity.is_decision:
            required_angles.append("decision")
        if complexity.is_contested:
            required_angles.append("contested")
        if complexity.query_type == "comparative":
            required_angles.append("comparison")
    except Exception as exc:
        # Fail-safe: a classifier failure must never break planning.
        logger.warning("[Planner] plan validation skipped (classifier failed)", exc_info=exc)
        return dims, cover, {"applied": False, "reason": "classifier_error"}

    pool = dims + cover
    added: List[str] = []
    for angle in required_angles:
        if not any(_angle_token_covered(label, angle) for label in pool):
            injected = f"{angle} evidence" if angle == "quantitative" else f"{angle} angle"
            dims.append(injected)
            cover.append(injected)
            pool = dims + cover
            added.append(injected)

    meta = {
        "applied": True,
        "query_type": complexity.query_type,
        "required_angles": required_angles,
        "added": added,
        "duplicate_dimensions": duplicate_flag,
    }
    if added:
        logger.info("[Planner] plan validation added required dimensions: %s", ", ".join(added))
    if duplicate_flag:
        logger.warning("[Planner] plan contains overlapping dimensions (flagged, not merged)")
    return dims, cover, meta


async def plan_dimensions(
    llm: LLMClient,
    query: str,
    *,
    today: str = "",
    intent: Optional[Dict[str, Any]] = None,
    context_snippets: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Ask the model which research dimensions THIS query needs.

    Returns (dimensions, must_cover, meta). On any LLM failure the deterministic
    query-type-aware fallback is returned so a degraded run still gets genuinely
    different dimensions per query type rather than a generic template
    (AGENTS.md 4.7). `meta` carries the model's query_type/domain plus a
    `planned_by` marker ("llm" | "heuristic") for the trace.
    """
    intent = intent or {}
    date_block = f"\nToday is {today.strip()}." if today.strip() else ""
    intent_note = ""
    if intent.get("ambiguity") and intent.get("senses"):
        labels = [
            str(s.get("label", "")).strip()
            for s in intent["senses"]
            if isinstance(s, dict) and str(s.get("label", "")).strip()
        ]
        if labels:
            intent_note = (
                "\nThis query is AMBIGUOUS between: " + "; ".join(labels[:2])
                + ". Include a disambiguation dimension for each meaning."
            )
    snippet_block = ""
    snippets = [s for s in (context_snippets or []) if str(s).strip()]
    if snippets:
        snippet_block = (
            "\nTop web results for the raw query (ground your dimensions in "
            "this real terminology; do not answer the query):\n"
            + "\n".join(f"- {s}" for s in snippets[:5])
        )
    user_prompt = (
        f"Query: {query}{date_block}{intent_note}{snippet_block}\n\n"
        "Which research dimensions does THIS query need? Return JSON only."
    )

    try:
        set_stage_hint("planner")
        payload = await llm.generate_json(
            system_prompt=PLANNING_DIRECTIVE_PROMPT,
            user_prompt=user_prompt,
            response_model=PlanningDirectiveModel,
        )
    except Exception as exc:
        logger.warning("[Planner] dimension directive LLM failed, using heuristic", exc_info=exc)
        record_fallback("planner_dimensions")
        return _validated_heuristic_dimensions(query, intent)

    if not isinstance(payload, dict):
        logger.warning("[Planner] dimension directive returned non-dict, using heuristic")
        record_fallback("planner_dimensions")
        return _validated_heuristic_dimensions(query, intent)

    dims: List[str] = []
    for raw in payload.get("dimensions") or []:
        name = re.sub(r"\s+", " ", str(raw or "")).strip()
        if name and len(name) <= 60 and name.lower() not in {d.lower() for d in dims}:
            dims.append(name)
    if not dims:
        logger.warning("[Planner] dimension directive returned no dimensions, using heuristic")
        record_fallback("planner_dimensions")
        return _validated_heuristic_dimensions(query, intent)

    must_cover: List[str] = []
    for raw in payload.get("must_cover") or []:
        name = re.sub(r"\s+", " ", str(raw or "")).strip()
        if name and name not in must_cover:
            must_cover.append(name)
    if not must_cover:
        must_cover = dims[:2]

    # Validate the model's dimensions against what the query actually requires
    # (quantitative / decision / contested / comparative). Missing angles are
    # APPENDED and added to must_cover, so the prompt's R3/R4 requirements are
    # enforced deterministically instead of trusted. Reordering never happens,
    # so an already-valid plan is returned unchanged.
    dims, must_cover, validation_meta = validate_plan_dimensions(query, dims, must_cover)

    meta = {
        "planned_by": "llm",
        "query_type": str(payload.get("query_type", "") or _coarse_query_type(query)),
        "dominant_domain": str(payload.get("dominant_domain", "general") or "general"),
        "reasoning": str(payload.get("reasoning", "") or ""),
        "coverage_note": str(payload.get("coverage_note", "") or ""),
        "preferred_search_types": [
            str(s) for s in (payload.get("preferred_search_types") or [])
            if str(s) in VALID_SEARCH_TYPES
        ],
        "plan_validation": validation_meta,
    }
    logger.info("[Planner] dynamic dimensions: %s (must_cover=%s)", dims, must_cover)
    return dims, must_cover, meta


def _validated_heuristic_dimensions(
    query: str, intent: Optional[Dict[str, Any]] = None
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Heuristic dimensions, also run through the deterministic validator.

    The second return path of `plan_dimensions`: when the directive yields no
    dimensions the heuristic list is only a coarse starting point, so it is
    validated (and missing required angles appended) exactly like the LLM
    path. `must_cover` defaults to the first two dimensions and gains every
    appended required angle, so each maps to a real contract downstream.
    """
    dims = _heuristic_dimensions(query, intent)
    dims, must_cover, validation_meta = validate_plan_dimensions(query, dims, dims[:2])
    return dims, must_cover, {
        "planned_by": "heuristic",
        "query_type": _coarse_query_type(query),
        "plan_validation": validation_meta,
    }


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
    sense: str = "",
) -> Dict[str, Any]:
    """Build one fully-populated delegation contract.

    Single construction point so every field (including the new source
    preferences) is present on contracts from the LLM path, the axis-repair
    path and the fallback path alike. Downstream code can rely on the shape
    instead of defaulting per call site.
    Dynamic-planning note: `axis` may be a model-chosen dimension label, not a
    canonical enum value. Canonical labels and common aliases are mapped via
    `dimension_to_axis`; a genuinely query-specific label is kept as a slug so
    it still gets its own report section (downstream keyed by axis string).
    """
    # Frontier tracks are first-class axes: never fold them onto the legacy
    # canonical alias ("safety" must not become "risk", "counter_evidence"
    # must not become "criticism"). Those aliases exist for free-form model
    # labels; the mandatory tracks are their own report sections and their
    # coverage is checked by exact name.
    requested = str(axis).strip()
    axis = requested if requested in FRONTIER_AXIS_SET else dimension_to_axis(axis)
    search_type = (
        search_type if search_type in VALID_SEARCH_TYPES
        else axis_search_type(axis, default="encyclopedia")
    )
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
        "primary_source_query": build_dimension_primary_query(question, search_type, domain),
        "wave": 0,
        "sense": str(sense or "").strip(),
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


def _intent_research_senses(intent: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """(sense_label, domain) pairs the intent stage says research should target.

    Empty when the query is unambiguous (or intent is absent): nothing in the
    plan is sense-tagged and behaviour is exactly the pre-intent one.
    """
    if not intent or not intent.get("ambiguity"):
        return []
    senses = [
        s for s in (intent.get("senses") or [])
        if isinstance(s, dict) and str(s.get("label", "")).strip()
    ]
    if not senses:
        return []
    chosen = senses[:2] if intent.get("recommended_action") == "research_both" else senses[:1]
    return [
        (str(s["label"]).strip(), normalize_domain(str(s.get("domain", "")) or "general"))
        for s in chosen
    ]


def _sense_concept(label: str) -> str:
    """Lowercased, parenthetical-stripped sense label — a search-ready phrase."""
    return re.sub(r"\s*\([^)]*\)", "", (label or "")).strip().lower()


def _assign_intent_senses(
    plan: List[Dict[str, Any]],
    intent: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Stamp the intent's sense labels onto contracts (deterministic).

    The model's own sense tags win when they name a researched sense; the
    rest are assigned by round-robin (research_both) or the single dominant
    sense. A sense's domain upgrades a `general` contract so the right
    specialist prompt loads downstream.
    """
    research = _intent_research_senses(intent)
    if not research:
        return plan
    domains = dict(research)
    valid = {label for label, _ in research}

    unassigned: List[Dict[str, Any]] = []
    for item in plan:
        sense = str(item.get("sense", "") or "").strip()
        if sense in valid:
            if item.get("domain", "general") == "general" and domains[sense] != "general":
                item["domain"] = domains[sense]
                item["specialist"] = DOMAIN_TO_SPECIALIST.get(item["domain"], "general")
        else:
            item["sense"] = ""
            unassigned.append(item)

    if len(valid) == 1:
        only = next(iter(valid))
        for item in unassigned:
            item["sense"] = only
            if item.get("domain", "general") == "general" and domains[only] != "general":
                item["domain"] = domains[only]
                item["specialist"] = DOMAIN_TO_SPECIALIST.get(item["domain"], "general")
    else:
        labels = [label for label, _ in research]
        for i, item in enumerate(unassigned):
            sense = labels[i % len(labels)]
            item["sense"] = sense
            if item.get("domain", "general") == "general" and domains[sense] != "general":
                item["domain"] = domains[sense]
                item["specialist"] = DOMAIN_TO_SPECIALIST.get(item["domain"], "general")
    return plan


def synthesize_dimension_contract(
    *,
    index: int,
    dimension: str,
    query: str,
    domain: str = "general",
    minimum_sources: int = DEFAULT_MINIMUM_SOURCES,
    today: str = "",
    coverage_goal: str = "",
) -> Dict[str, Any]:
    """Build a search-ready contract for a model-chosen dimension the plan omitted.

    The dynamic-planning replacement for the old fixed axis templates: the
    dimension label is the model's own, so the question is derived from that
    label + the query's concept rather than from a stock sentence. Keyword-
    shaped (not prose) because its whole job is to be a search string.
    """
    axis = dimension_to_axis(dimension)
    search_type = axis_search_type(axis)
    concept = _query_concept(query)
    year = _year_from(today)
    label = dimension.replace("_", " ").strip()
    question = f"{concept} {label}{year}".strip()
    return _contract(
        index=index,
        question=question,
        axis=axis,
        search_type=search_type,
        priority=1,
        domain=domain,
        coverage_goal=coverage_goal or f"required dimension: {label}",
        minimum_sources=minimum_sources,
    )


# Legacy axis->keyword template. Retained ONLY for the no-dimension fallback
# (a caller that requires canonical axes but supplied no dynamic dimension
# labels). New planning flows should never reach it — required axes now come
# from `plan_dimensions` and are synthesized by `synthesize_dimension_contract`.
_AXIS_FALLBACK_TEMPLATES: Dict[str, Tuple[str, str, int]] = {
    "evidence": ("statistics official data figures", "statistical", 1),
    "criticism": ("limitations criticism counter-evidence risks", "academic", 2),
    "comparison": ("compared alternatives side by side", "comparison", 2),
    "definition": ("definition explanation overview", "encyclopedia", 1),
    "outlook": ("outlook forecast recent developments", "news", 2),
    "mechanism": ("how it works mechanism why causes drivers", "academic", 1),
    "risk": ("risks failure modes downsides", "academic", 2),
    "cost": ("cost price economics figures", "statistical", 2),
    "history": ("history origin development timeline", "encyclopedia", 3),
    "regulation": ("regulation policy law governance", "news", 3),
}


def enforce_axis_coverage(
    plan: List[Dict[str, Any]],
    query: str,
    required_axes: Sequence[str],
    *,
    domain: str = "general",
    minimum_sources: int = DEFAULT_MINIMUM_SOURCES,
    today: str = "",
    required_questions: Optional[Mapping[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Add a contract for every required dimension the model omitted.

    `required_axes` are the dynamic dimensions from `plan_dimensions`; when a
    `required_questions` mapping is supplied (as `axis -> literal question`),
    the injected contract uses that question verbatim instead of synthesizing
    one. Returns (plan, injected_axes).
    """
    present = {str(item.get("axis", "")) for item in plan}
    concept = _query_concept(query)
    year = _year_from(today)
    injected: List[str] = []
    next_id = max((int(item.get("id", 0)) for item in plan), default=0) + 1
    overrides = {
        dimension_to_axis(k): str(v).strip()
        for k, v in (required_questions or {}).items()
        if str(v).strip()
    }

    for raw_axis in required_axes or ():
        # `required_axes` may carry dynamic dimension labels ("cost and
        # financing"): map each to its canonical axis first so an existing
        # contract that already serves it counts as covering it.
        axis = dimension_to_axis(raw_axis)
        if axis in present or not str(raw_axis).strip():
            continue
        label = str(raw_axis).replace("_", " ").strip()
        if axis in overrides:
            question = overrides[axis]
            search_type = axis_search_type(axis)
            priority = 1
        elif axis in _AXIS_FALLBACK_TEMPLATES:
            suffix, search_type, priority = _AXIS_FALLBACK_TEMPLATES[axis]
            question = f"{concept} {suffix}{year}".strip()
        else:
            # A dynamic, non-canonical dimension with no supplied question.
            plan.append(
                synthesize_dimension_contract(
                    index=next_id,
                    dimension=label,
                    query=query,
                    domain=domain,
                    minimum_sources=minimum_sources,
                    today=today,
                )
            )
            present.add(axis)
            injected.append(axis)
            next_id += 1
            continue
        plan.append(
            _contract(
                index=next_id,
                question=question,
                axis=axis,
                search_type=search_type,
                priority=priority,
                domain=domain,
                coverage_goal=f"required dimension: {label}",
                minimum_sources=minimum_sources,
            )
        )
        present.add(axis)
        injected.append(axis)
        next_id += 1

    if injected:
        logger.info("[Planner] injected required dimensions: %s", ", ".join(injected))
    return plan, injected


def enforce_frontier_axes(
    plan: List[Dict[str, Any]],
    query: str,
    *,
    domain: str = "general",
    minimum_sources: int = DEFAULT_MINIMUM_SOURCES,
    today: str = "",
    axes: Sequence[str] = FRONTIER_AXES,
    include_counter_evidence: bool = True,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Guarantee one contract per mandatory frontier axis, plus counter-evidence.

    The six frontier tracks are equal-weight and never optional. This runs AFTER
    the model's plan and `enforce_axis_coverage`, appending a literal, primary-
    source-seeking contract for any missing track. It never reorders or removes
    the model's contracts, so a plan that already covers all six is unchanged.

    Counter-evidence is forced even when the model planned a `criticism` angle:
    the two are not the same search. `criticism` asks for limitations of the
    subject; the counter-evidence track asks for arguments that the whole
    prevailing narrative is wrong (over-hype, plateau, negative ROI), which is
    the one thing a hype-heavy corpus will never surface on its own.
    """
    present = {str(item.get("axis", "")) for item in plan}
    # Also honour contracts the model labelled with a frontier synonym:
    # dimension_to_axis maps "ai safety" -> risk and "counter argument" ->
    # criticism, so check each contract's raw label AND its canonical mapping
    # against the frontier axis. Exact frontier axis names always count.
    normalized_present = set(present)
    for item in plan:
        raw = str(item.get("axis", ""))
        normalized_present.add(dimension_to_axis(raw))
        normalized_present.add(raw.replace(" ", "_"))
    wanted = list(axes or ())
    if include_counter_evidence:
        wanted.append(COUNTER_EVIDENCE_AXIS)

    injected: List[str] = []
    next_id = max((int(item.get("id", 0)) for item in plan), default=0) + 1
    for axis in wanted:
        if axis in normalized_present:
            continue
        question = FRONTIER_AXIS_QUESTIONS.get(axis, "").strip()
        if not question:
            continue
        plan.append(
            _contract(
                index=next_id,
                question=question,
                axis=axis,
                search_type=axis_search_type(axis),
                priority=1,
                domain=domain,
                coverage_goal=f"mandatory frontier track: {axis}",
                minimum_sources=minimum_sources,
            )
        )
        normalized_present.add(axis)
        injected.append(axis)
        next_id += 1

    if injected:
        logger.info("[Planner] injected frontier tracks: %s", ", ".join(injected))
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
    # Required dimensions arrive as model labels ("cost and financing"): map
    # each to the canonical axis its contract carries so the protection below
    # actually matches (enforce_axis_coverage applies the same mapping).
    required = [dimension_to_axis(a) for a in (required_axes or ())]
    picked: List[Dict[str, Any]] = []
    picked_ids: Set[int] = set()

    def _take(item: Dict[str, Any]) -> None:
        picked.append(item)
        picked_ids.add(int(item.get("id", -1)))

    for axis in required:
        # Required axes are a CONTRACT, not a preference: they must survive
        # truncation. The budget cap below is honoured for optional angles, but
        # a required axis is only skipped when no contract serves it at all.
        # Previously `len(picked) >= target_count` broke this loop, so with
        # target=3 and axes [definition, evidence, criticism, mechanism,
        # outlook] the injected mechanism/outlook angles were selected and then
        # immediately discarded — the same 3-axis plan every pass, which is the
        # decomposition gap the WHY angle exists to close.
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
    intent: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Deterministic plan for when the LLM call fails.

    Now shaped by the same axis contract as a real plan (an evidence angle and a
    criticism angle, not four definition variants) and sized to the
    orchestrator's target, so a degraded run is a smaller research plan rather
    than a different, weaker kind of plan. When the intent stage flagged the
    query ambiguous, contracts are sense-scoped and split across the researched
    senses — a degraded plan still targets the user's likely meaning.
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

    research = _intent_research_senses(intent)
    specs: List[Tuple[str, str, str, int, str, str]] = []
    if research:
        # Interleave senses so each researched meaning gets the top axes.
        for bi, (question, axis, search_type, priority) in enumerate(ordered):
            for si, (label, domain) in enumerate(research):
                if len(specs) >= max(1, target_count):
                    break
                sense_question = f"{_sense_concept(label)}{question[len(concept):]}"
                specs.append((sense_question, axis, search_type, priority, label, domain))
            if len(specs) >= max(1, target_count):
                break
    else:
        specs = [
            (question, axis, search_type, priority, "", "general")
            for (question, axis, search_type, priority) in ordered[: max(1, target_count)]
        ]

    plan = [
        _contract(
            index=i + 1,
            question=question,
            axis=axis,
            search_type=search_type,
            priority=priority,
            domain=domain,
            coverage_goal=f"fallback {axis} coverage",
            sense=sense,
        )
        for i, (question, axis, search_type, priority, sense, domain) in enumerate(specs)
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
    intent: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Produce a validated, axis-complete, dependency-ordered research plan.

    New optional arguments come from `OrchestrationPlan.targets`; omitting them
    reproduces the previous behaviour (five sub-questions, no axis enforcement),
    so existing call sites keep working while the workflow migrates. `intent`
    (when provided) steers the plan at the user's likely meaning and tags
    contracts with sense labels for ambiguous queries.
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

    intent = intent or {}

    # -------- Dynamic planning directive (Phase 2) --------
    # The plan must cover the dimensions THIS query needs. On the first pass we
    # ask the directive stage which dimensions those are and use its answer as
    # the required set, replacing the generic axis templates. On expansion
    # passes the workflow has already folded in the axes under research, so we
    # do NOT re-plan dimensions (that would undo per-axis expansion) — the
    # caller's required_axes is authoritative and any missing one is filled by
    # the dimension synthesizer below.
    planned_dimensions: List[str] = []
    required_questions: Dict[str, str] = {}
    directive_meta: Dict[str, Any] = {}
    # `critique_feedback` is non-empty only on expansion passes (the workflow
    # passes the critic's gap list). First-pass planning is the only time the
    # dimension directive runs; expansion must preserve the axes already under
    # research instead of re-deriving a fresh dimension set.
    if not critique_feedback:
        planned_dimensions, must_cover, directive_meta = await plan_dimensions(
            llm, query, today=today, intent=intent, context_snippets=context_snippets,
        )
        # The directive's dimensions become the plan's hard requirements.
        # Caller-supplied required_axes (orchestration budget axes) are folded
        # in ONLY when the directive actually planned (planned_by=llm): on the
        # deterministic fallback the caller's explicit axes are the contract
        # (a caller that passed required_axes gets exactly those, not the
        # heuristic set added on top — otherwise required axes balloon past the
        # plan budget).
        if directive_meta.get("planned_by") == "llm":
            # The directive's dimensions ARE the contract: they are specific to
            # this query and already encode whatever evidence/criticism/mechanism
            # angles it needs. Re-merging the orchestration's generic axes here
            # would re-introduce exactly the boilerplate this stage replaces
            # (measured: every plan then carried definition/evidence/criticism/
            # mechanism alongside its real dimensions). The orchestration axes
            # remain the fallback contract when the directive cannot plan.
            required_axes = tuple(
                str(d or "").strip() for d in planned_dimensions if str(d or "").strip()
            )
        # On the deterministic fallback (directive LLM failed) the caller's
        # required_axes, if any, stay authoritative; if the caller passed none,
        # nothing is force-injected here — a plan-LLM failure already routes to
        # fallback_plan(), which guarantees the canonical safety-net axes.
        # must_cover dimensions are non-negotiable: if the plan model omits one,
        # the coverage enforcement below synthesizes a contract from the
        # dimension label itself. (No literal question is supplied — the
        # synthesizer derives one from the label + query concept.)
        if directive_meta.get("coverage_note"):
            logger.info("[Planner] directive coverage note: %s", directive_meta["coverage_note"])

    # Intent block: the resolved understanding of the question. It overrides
    # the model's own reading — that is the whole point of resolving intent
    # BEFORE research is shaped.
    intent_senses = [
        s for s in (intent.get("senses") or [])
        if isinstance(s, dict) and str(s.get("label", "")).strip()
    ]
    intent_parts: List[str] = []
    if intent.get("ambiguity") and intent_senses:
        listed = "\n".join(
            f"  {i + 1}. {str(s.get('label')).strip()} "
            f"({s.get('domain', 'general')}, p={float(s.get('probability', 0) or 0):.2f})"
            for i, s in enumerate(intent_senses[:3])
        )
        if intent.get("recommended_action") == "research_both":
            stance = (
                "Research BOTH leading senses — split the plan's budget across them, "
                "and never mix the two meanings inside one sub-question."
            )
        else:
            stance = (
                f"Research ONLY the most likely sense ('{intent_senses[0].get('label')}'); "
                "the report will cover the other meaning(s) in a brief disambiguation "
                "paragraph, so do not spend sub-questions on them."
            )
        intent_parts.append(
            "AMBIGUOUS QUERY — the user's term has distinct meanings:\n"
            f"{listed}\n{stance}\n"
            "Every sub-question MUST carry a \"sense\" field set to the exact label "
            "of the meaning it researches."
        )
    elif intent_senses:
        intent_parts.append(
            f"Likely meaning: {intent_senses[0].get('label')} — target the plan at this sense."
        )
    # Under-specified (not homonymous) query: the term has multiple useful
    # readings. Plan a sub-question per materially useful reading so the answer
    # can address both instead of explaining the ambiguity.
    interpretations = [
        i for i in (intent.get("interpretations") or [])
        if isinstance(i, dict) and str(i.get("label", "")).strip()
    ]
    if len(interpretations) >= 2:
        listed = "; ".join(str(i.get("label")).strip() for i in interpretations[:3])
        intent_parts.append(
            "UNDER-SPECIFIED QUERY — it can be read in more than one useful way: "
            f"{listed}. Plan a sub-question for each materially useful reading so "
            "the answer addresses both; do not spend research on the fact that the "
            "term is ambiguous."
        )
    if intent.get("domain"):
        intent_parts.append(
            f"Research domain: {intent.get('domain')} (overrides your own classification)."
        )
    if intent.get("explanation_level") == "basic":
        intent_parts.append(
            "Explanation level: basic — prefer one clear definitional sub-question "
            "over many technical angles."
        )
    intent_block = (
        "\nUser intent (resolved before research — obey it):\n" + "\n".join(intent_parts) + "\n"
        if intent_parts
        else ""
    )
    axis_block = ""
    if required_axes:
        axis_block = (
            "\nRequired research dimensions — the plan MUST contain one "
            "sub-question for each of: "
            f"{', '.join(str(a).replace('_', ' ') for a in required_axes)}. "
            "These were derived for THIS query. A plan missing any of them will "
            "be repaired automatically, and the repaired question will be cruder "
            "than one you write yourself. Use each dimension's label as the "
            "sub-question's axis."
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
{intent_block}
{axis_block}
{budget_block}
{context_block}

Generate a structured research plan.
Return JSON only.
"""

    try:
        set_stage_hint("planner")
        payload: PlannerOutput = await llm.generate_json(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            response_model=PlannerOutputModel,
        )
    except Exception as e:
        logger.error(f"[Planner] LLM failed: {e}", exc_info=e)
        record_fallback("planner")
        return fallback_plan(query, target, required_axes or ("definition", "evidence", "criticism"), today, intent=intent)

    sub_questions = payload.get("sub_questions", []) if isinstance(payload, dict) else []
    if not sub_questions:
        logger.warning("[Planner] Empty LLM output, using fallback")
        record_fallback("planner")
        return fallback_plan(query, target, required_axes or ("definition", "evidence", "criticism"), today, intent=intent)

    dominant_domain = normalize_domain(
        str(payload.get("dominant_domain", "general")) if isinstance(payload, dict) else "general"
    )
    # Intent overrides the model's own domain classification when confident.
    if intent.get("domain"):
        dominant_domain = normalize_domain(str(intent["domain"]))

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
                sense=str(item.get("sense", "") or "").strip(),
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
        required_questions=required_questions,
    )

    # Select BEFORE frontier enforcement: the six mandatory tracks are not
    # subject to the target budget, so truncating after injection would drop
    # exactly the axes the frontier requirement exists to guarantee. The
    # model's own plan is trimmed first, then the mandatory tracks are added on
    # top — the plan may exceed `target` by at most the missing frontier count.
    final = select_plan(cleaned, target, required_axes)
    # Mandatory six-axis coverage + the always-on counter-evidence track. Quick
    # mode skips the mandatory tracks (latency-sensitive by design) but still
    # gets counter-evidence, because a report with no opposing view is not a
    # research report.
    final, frontier_injected = enforce_frontier_axes(
        final, query,
        domain=dominant_domain,
        minimum_sources=minimum_sources,
        today=today,
        include_counter_evidence=True,
    )
    injected = [*injected, *frontier_injected]
    final = sanitize_dependencies(final)
    final = _assign_intent_senses(final, intent)


    if not final:
        logger.warning("[Planner] All filtered out, fallback used")
        record_fallback("planner")
        return fallback_plan(query, target, required_axes or ("definition", "evidence", "criticism"), today, intent=intent)

    logger.info(
        "[Planner] %d contract(s), axes=%s, search_types=%s, injected=%s, planned_by=%s",
        len(final),
        sorted({item["axis"] for item in final}),
        sorted(diversity_coverage(final)),
        injected or "none",
        directive_meta.get("planned_by", "caller" if critique_feedback else "heuristic"),
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
