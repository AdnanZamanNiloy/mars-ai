"""Answer-first outline builder — decide the report's shape BEFORE writing.

The failure this module exists for: broad questions ("What is the current
trend of AI?") were being answered by whatever claims happened to rank
highest, so the report collapsed into a narrow thesis or a source dump.
GPT Researcher avoids this by planning a report outline (subtopics) and
researching each section — this is the MARS-native, low-risk half of that:
an outline derived from the query + the evidence already in hand, computed
deterministically and handed to the synthesizer so the writer targets the
question's dimensions instead of dumping claims.

Design
------
* Deterministic first: the outline is built from the plan's research angles
  (`sub_question`), the query's own structure, and the evidence grade
  distribution. No LLM call is required, so this cannot degrade a run or
  add latency. An optional LLM polish is available but off the critical
  path.
* Each section names a *dimension* (definition / mechanism / evidence /
  criticism / comparison / outlook / ...) and carries the facts that
  belong to it, so section-wise synthesis has a real grouping.
* Deterministic fallback (AGENTS.md 4.7): if anything is missing, the
  outline is the query's detected dimensions or a single "Answer" section.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from app.core.logging import get_logger

logger = get_logger(__name__)

# Axis → human section title. Mirrors the planner's axis enum; unknown axes
# keep their own (title-cased) name so a model-invented angle still shows.
_AXIS_TITLES: Dict[str, str] = {
    "definition": "What It Is",
    "mechanism": "How It Works",
    "application": "Applications",
    "evidence": "Evidence & Data",
    "comparison": "How It Compares",
    "criticism": "Limitations & Critique",
    "history": "Background & History",
    "outlook": "Outlook & Trends",
}

# Order sections so a report reads survey → depth → challenge, like the
# planner's phases. Unknown axes sort last (stable).
_AXIS_ORDER: List[str] = [
    "definition",
    "history",
    "mechanism",
    "application",
    "evidence",
    "comparison",
    "criticism",
    "outlook",
]

# A query with this many distinct research axes is treated as "broad" — the
# signal for section-wise synthesis rather than one monolithic pass.
BROAD_MIN_DIMENSIONS = 3


@dataclass
class OutlineSection:
    """One section of the answer, with the evidence that belongs to it."""

    axis: str
    title: str
    question: str = ""
    facts: List[Dict[str, Any]] = field(default_factory=list)
    coverage_goal: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axis": self.axis,
            "title": self.title,
            "question": self.question,
            "coverage_goal": self.coverage_goal,
            "fact_count": len(self.facts),
        }


@dataclass
class AnswerOutline:
    """The report's planned shape."""

    query: str
    sections: List[OutlineSection] = field(default_factory=list)
    broad: bool = False
    source: str = "deterministic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "broad": self.broad,
            "source": self.source,
            "sections": [s.to_dict() for s in self.sections],
        }


def _axis_for_fact(fact: Dict[str, Any]) -> str:
    """Best-effort axis for a fact: its contract axis, else its sub-question."""
    axis = str(fact.get("axis", "") or "").strip().lower()
    if axis:
        return axis
    sub_question = str(fact.get("sub_question", "") or "").strip().lower()
    if not sub_question:
        return "evidence"
    for candidate in _AXIS_ORDER:
        if candidate in sub_question:
            return candidate
    return "general"


def _sense_match(fact_sense: str, label: str) -> bool:
    """True when a fact's sense tag names the same sense as `label`.

    Case- and punctuation-insensitive containment in EITHER direction: the
    planner stamps the exact label, but a summarizer may record a lightly
    reworded tag ("Transformer neural network architecture" vs
    "transformer neural network architecture (ML)"). Containment is
    deliberately the test, not equality — an orphaned tag would silently
    starve the sense section of its own evidence.
    """
    a = re.sub(r"[^a-z0-9]+", " ", (fact_sense or "").lower()).strip()
    b = re.sub(r"[^a-z0-9]+", " ", (label or "").lower()).strip()
    if not a or not b:
        return False
    return a in b or b in a


# A heading is a label, not a question. The planner's dynamic dimensions carry
# a full multi-clause question as their `question` text, and `_section_title`
# fell back to emitting it verbatim — so live reports shipped headings like
# "## What was the causal chain of the 2023 US regional banking crisis: how did
# the March 2022-July 2023 ...". That is a duplicate of the query, not a
# section; the section writer then restated the same evidence under it, padding
# the report and diluting the answer. Cap the title, strip the interrogative
# frame, and use the axis slug when the question cannot be reduced cleanly.
_TITLE_MAX_CHARS = 64
_TITLE_MAX_WORDS = 9
_INTERROGATIVE_LEAD_RE = re.compile(
    r"^\s*(?:what|how|why|when|where|which|who|does|did|is|are|can|could|"
    r"should|will|would)\b[\s:,]*",
    re.IGNORECASE,
)


def _short_title(text: str, max_words: int = _TITLE_MAX_WORDS) -> str:
    """Reduce a clause/question to a heading-sized noun phrase.

    Deterministic and total: takes the first clause (up to a colon/dash/`?`),
    drops a leading interrogative, and truncates on a word boundary. Returns ""
    when nothing useful survives, so callers fall through to the axis slug.
    """
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return ""
    clause = re.split(r"[?:—–]|\s+-\s+", cleaned, maxsplit=1)[0].strip(" ?.!,")
    clause = _INTERROGATIVE_LEAD_RE.sub("", clause).strip(" ?.!,")
    if not clause:
        return ""
    words = clause.split(" ")
    if len(words) > max_words:
        clause = " ".join(words[:max_words]).rstrip(",;:")
    clause = clause.strip(" ?.!,")
    if len(clause) > _TITLE_MAX_CHARS:
        clause = clause[: _TITLE_MAX_CHARS + 1].rsplit(" ", 1)[0].strip(" ?.!,")
    return clause


def _normalize_title(text: str) -> str:
    """Comparison key for section titles: casefolded, alnum-only."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _section_title(axis: str, question: str) -> str:
    known = _AXIS_TITLES.get(axis)
    if known:
        return known
    # Prefer a concise, humanized axis slug — always label-shaped, never a
    # question. Only fall back to the question text when the slug is empty.
    slug = (axis or "").replace("_", " ").strip()
    if slug and slug not in ("general", "answer"):
        return slug[0].upper() + slug[1:]
    shortened = _short_title(question)
    if shortened:
        return shortened[0].upper() + shortened[1:]
    if slug:
        return slug[0].upper() + slug[1:]
    return "Key Findings"


def build_outline(
    query: str,
    facts: Sequence[Dict[str, Any]] = (),
    sub_questions: Sequence[Any] = (),
    *,
    intent: Dict[str, Any] | None = None,
) -> AnswerOutline:
    """Deterministically derive the answer's section outline.

    Sections come from (in priority order):
      1. the research plan's axes (each axis is a dimension the query needs),
      2. the facts' own axes/sub-questions when the plan is unavailable,
      3. a single fallback section so the writer never receives an empty shape.
    """
    facts = [f for f in (facts or []) if isinstance(f, dict)]
    # Group facts by axis, preserving first-seen question text for the title.
    by_axis: Dict[str, List[Dict[str, Any]]] = {}
    axis_question: Dict[str, str] = {}
    coverage_goal: Dict[str, str] = {}

    for item in sub_questions or ():
        if not isinstance(item, dict):
            continue
        axis = str(item.get("axis", "") or "").strip().lower()
        if not axis:
            continue
        by_axis.setdefault(axis, [])
        q = str(item.get("question", "") or "").strip()
        if q and axis not in axis_question:
            axis_question[axis] = q
        goal = str(item.get("coverage_goal", "") or "").strip()
        if goal and axis not in coverage_goal:
            coverage_goal[axis] = goal

    for fact in facts:
        axis = _axis_for_fact(fact)
        by_axis.setdefault(axis, []).append(fact)
        q = str(fact.get("sub_question", "") or "").strip()
        if q and axis not in axis_question:
            axis_question[axis] = q

    intent = intent or {}
    senses = [
        s for s in (intent.get("senses") or [])
        if isinstance(s, dict) and str(s.get("label", "")).strip()
    ]
    ambiguous = bool(intent.get("ambiguity")) and len(senses) >= 2

    ordered = sorted(
        by_axis.keys(),
        key=lambda a: (
            _AXIS_ORDER.index(a) if a in _AXIS_ORDER else len(_AXIS_ORDER),
            a,
        ),
    )

    sections: List[OutlineSection] = []
    if ambiguous:
        # A term with two senses needs one section per sense — never blended.
        # Each sense section receives ONLY the facts tagged with that sense.
        # The old code handed `list(facts)` (EVERY fact, both senses) to every
        # sense section, so the ML-transformer section was fed electrical-price
        # index claims and the not-researched sense was written from the other
        # sense's evidence — the exact blend the intent stage exists to stop.
        # A fact with no sense tag (unambiguous contract) is a fallback shared
        # below, so a sense-agnostic pool never yields empty sections.
        sense_by_fact: Dict[str, List[Dict[str, Any]]] = {}
        untagged: List[Dict[str, Any]] = []
        for fact in facts:
            tag = str(fact.get("sense", "") or "").strip()
            if tag:
                sense_by_fact.setdefault(tag, []).append(fact)
            else:
                untagged.append(fact)
        for sense in senses[:2]:
            label = str(sense.get("label", "")).strip()
            # Match the fact's sense tag to the sense label case-insensitively;
            # the exact label is what the planner stamps, but normalization
            # keeps a lightly-reworded tag from orphaning its evidence.
            matched = [
                f
                for tag, tagged in sense_by_fact.items()
                if _sense_match(tag, label)
                for f in tagged
            ]
            if not matched:
                # No sense-tagged evidence at all: fall back to the shared pool
                # so the section is never empty (old behaviour, only when the
                # pool carries no usable sense signal).
                matched = list(facts)
            sections.append(
                OutlineSection(
                    axis=str(sense.get("domain", "definition") or "definition"),
                    title=label,
                    question=label,
                    facts=matched,
                    coverage_goal="disambiguate this meaning from the others",
                )
            )

    # Axis sections are only added when they are NOT the senses' own dimensions:
    # a sense section already covers its evidence, so re-emitting the axis would
    # duplicate it (the report shipped an "Electrical transformer" sense section
    # AND a second evidence section reciting its price-index data).
    sense_axes = {
        str(s.get("domain", "") or "").strip().lower() for s in senses[:2]
    } if ambiguous else set()

    # Distinct axes whose short titles collide (a plan with 15 dynamic
    # dimensions can map several axes onto one label) would otherwise produce
    # two sections that repeat the same evidence. Merge on the normalized
    # title so each dimension is written once; the first section keeps its
    # facts, later colliding axes contribute only their extra facts.
    seen_titles: Dict[str, int] = {}
    for section in sections:
        seen_titles[_normalize_title(section.title)] = len(sections) - 1

    for axis in ordered:
        if ambiguous and axis in sense_axes:
            continue
        if len(sections) >= 8:
            break
        facts_for_axis = by_axis.get(axis, [])
        title = _section_title(axis, axis_question.get(axis, ""))
        key = _normalize_title(title)
        if key in seen_titles:
            # Title already used by a sense section or an earlier axis: fold
            # this axis's evidence into that section instead of restating the
            # same dimension under a second heading.
            existing = sections[seen_titles[key]]
            existing_ids = {id(f) for f in existing.facts}
            existing.facts.extend(f for f in facts_for_axis if id(f) not in existing_ids)
            continue
        section = OutlineSection(
            axis=axis,
            title=title,
            question=axis_question.get(axis, ""),
            facts=facts_for_axis,
            coverage_goal=coverage_goal.get(axis, ""),
        )
        seen_titles[key] = len(sections)
        sections.append(section)

    # An axis section with no evidence writes nothing but still consumes a
    # writer call and a heading; drop it when other sections carry evidence.
    if any(s.facts for s in sections):
        sections = [s for s in sections if s.facts]

    if not sections:
        sections = [
            OutlineSection(
                axis="answer",
                title="Answer",
                question=(query or "").strip(),
                facts=list(facts),
            )
        ]

    broad = len(sections) >= BROAD_MIN_DIMENSIONS
    outline = AnswerOutline(query=query, sections=sections, broad=broad)
    logger.info(
        "[Outline] %d section(s) (%s) broad=%s",
        len(sections), ", ".join(s.axis for s in sections), broad,
    )
    return outline


def render_outline(outline: AnswerOutline) -> str:
    """Render the outline for the writer prompt: section order + fact budget.

    The writer is told to produce one section per entry, in this order, so
    the answer covers the query's dimensions rather than the top-ranked
    claim's neighbourhood.
    """
    if not outline.sections:
        return ""
    lines = [
        "REPORT OUTLINE — write one section per entry, in this order. "
        "Cover every dimension the query needs; do not collapse the report "
        "into the first dimension that has evidence."
    ]
    for i, section in enumerate(outline.sections, 1):
        line = f"{i}. {section.title}"
        if section.coverage_goal:
            line += f" — {section.coverage_goal}"
        line += f" ({len(section.facts)} evidence item(s))"
        lines.append(line)
    return "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Adaptive answer blueprint — the presentation strategy for THIS question
# ---------------------------------------------------------------------------

# Task families that imply a presentation strategy, not a heading list. The
# strategy tells the writer HOW to organise the answer (comparison by criterion,
# mechanism chain, ordered steps, options + trade-offs, thematic synthesis …).
# It never dictates headings.
_BLUEPRINT_STRATEGIES: Dict[str, str] = {
    "comparison": (
        "Organise by CRITERION, not by item: compare the options on one "
        "dimension at a time, then close with a verdict that says which option "
        "wins for which use case."
    ),
    "howto": (
        "Present prerequisites, then numbered steps in execution order, one "
        "action each. Name the failure mode for any step that commonly fails."
    ),
    "causal": (
        "Lead with the mechanism the evidence supports (X drives Y because Z), "
        "then the competing explanations and what evidence would separate them."
    ),
    "decision": (
        "Lay out the realistic options with their trade-offs, give a "
        "recommendation conditioned on the reader's situation, and state what "
        "would change it."
    ),
    "forecast": (
        "State the current measured level and its as-of date first; label every "
        "projection as a projection with its assumptions, and give a range."
    ),
    "timeline": (
        "Order by date, attach a date to every event, and end on the current "
        "state with its as-of date."
    ),
    "status": (
        "Open with the current state and the date of the evidence; flag "
        "anything that may have changed since."
    ),
    "list": (
        "Give the enumeration as the primary content, one self-contained cited "
        "item per bullet, ordered by the ranking the reader would use."
    ),
    "definition": (
        "Open with a one-sentence plain-language definition, then how it works "
        "and where it matters; at most one marked analogy."
    ),
    "entity": (
        "Establish identity first (which entity exactly), then the facts asked "
        "for; separate similarly-named entities before anything else."
    ),
    "broad_research": (
        "Organise around the dominant themes the evidence actually supports — "
        "two to four of them — with each theme explaining what changed, the "
        "concrete evidence, and why it matters."
    ),
}


@dataclass
class AnswerBlueprint:
    """The evidence-driven plan for HOW to communicate this answer.

    This is the adaptive layer between the evidence state and the writer. It
    does not prescribe headings; it decides the presentation strategy, the
    central question, the dominant themes, and whether the answer should be
    short or deep — from the detected intent and the evidence actually in hand.
    """

    query: str
    strategy: str = "general"
    framework: str = ""
    central_question: str = ""
    themes: List[str] = field(default_factory=list)
    depth: str = "standard"
    comparison: bool = False
    contested: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "framework": self.framework,
            "central_question": self.central_question,
            "themes": list(self.themes),
            "depth": self.depth,
            "comparison": self.comparison,
            "contested": self.contested,
        }


def _blueprint_key(query: str, intent: Dict[str, Any], outline: AnswerOutline) -> str:
    """The task family for this question: intent first, then shape patterns."""
    declared = str(intent.get("query_type", "") or "").strip().lower()
    alias = {
        "factual": "status", "how_to": "howto", "procedural": "howto",
        "chronology": "timeline", "evaluative": "decision",
        "strategic": "decision", "person": "entity", "organisation": "entity",
        "organization": "entity", "exploratory": "broad_research",
        "analytical": "broad_research",
    }.get(declared, declared)
    if alias in _BLUEPRINT_STRATEGIES:
        return alias
    # Deterministic shape detection from the query text itself.
    text = (query or "").strip()
    patterns = (
        ("comparison", r"\b(vs\.?|versus|compare[ds]?|comparison|better than|difference between|which (?:is|one))\b"),
        ("howto", r"\b(how (?:do|to|can) |steps? to|guide to|set ?up|install|configure|tutorial)\b"),
        ("causal", r"\b(why|what caused|cause[ds]? of|reason[s]? (?:for|why)|leads? to)\b"),
        ("decision", r"\b(should (?:i|we|they)|worth it|is it worth|recommend|choose between|invest in)\b"),
        ("forecast", r"\b(will |forecast|projection|predicted|by 20\d\d|outlook|future of|expected to)\b"),
        ("timeline", r"\b(timeline|history of|when did|chronolog|over time|evolution of)\b"),
        ("status", r"\b(current(?:ly)?|right now|as of|latest|today|still |up to date)\b"),
        ("list", r"\b(list of|top \d+|examples? of|what are the|types of|kinds of)\b"),
        ("definition", r"^\s*(what (?:is|are|does)|define|meaning of|explain )\b"),
    )
    for name, pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            # A BROAD outline (3+ evidence-backed dimensions) inside a
            # state/trend-shaped question is a thematic survey, not a one-line
            # status update. The ordered regex list matched 'current' and
            # collapsed "current trends in AI" onto a concise status answer,
            # discarding the dimensions the research actually covered. Only the
            # `status` family is overridden: a `definition` ("What is CRISPR?")
            # or `list` question keeps its own shape even when the pool is
            # broad, and strong task families (comparison, decision, how-to,
            # timeline, forecast) always keep theirs.
            if outline.broad and name == "status":
                return "broad_research"
            return name
    # A bare future year ("most demanding jobs in 2027") is a forecast even
    # without a forecast verb; reuse the temporal module's clock-relative check
    # so a historical year is never mistaken for one.
    try:
        from app.core.temporal import query_targets_future

        if query_targets_future(text):
            return "forecast"
    except Exception as exc:  # detection failure must not break the blueprint
        logger.warning("forecast_shape_detection_failed", error=str(exc), exc_info=exc)
    return "broad_research" if outline.broad else "general"


def build_blueprint(
    query: str,
    facts: Sequence[Dict[str, Any]] = (),
    sub_questions: Sequence[Any] = (),
    *,
    intent: Dict[str, Any] | None = None,
    outline: AnswerOutline | None = None,
    mode: str = "standard",
    contradictions: Sequence[Dict[str, Any]] = (),
) -> AnswerBlueprint:
    """Derive the adaptive presentation blueprint from question + evidence.

    Deterministic and LLM-free (AGENTS 4.7): when intent is unavailable the
    query's own shape and the evidence distribution decide the strategy, so a
    degraded run still gets an adaptive answer rather than a fixed template.
    """
    intent = intent or {}
    outline = outline or build_outline(query, facts, sub_questions, intent=intent)
    key = _blueprint_key(query, intent, outline)
    strategy = _BLUEPRINT_STRATEGIES.get(key, "")

    # Dominant themes: the outline's section titles are the dimensions the
    # evidence actually supports — the emergent thematic structure.
    themes = [s.title for s in outline.sections if s.title and s.title != "Answer"][:5]

    level = str(intent.get("explanation_level", "") or "").lower()
    # Depth is question-shape driven: a broad SURVEY (broad_research) needs
    # depth even when phrased as a "current"/status question, while a genuinely
    # short definition or status question stays concise even when the pool
    # happens to be broad. A basic explanation level is always concise.
    broad_survey = key == "broad_research" and outline.broad
    if level == "basic" and not broad_survey:
        depth = "concise"
    elif broad_survey:
        depth = "deep"
    elif key in ("definition", "status", "list"):
        depth = "concise"
    elif str(mode or "standard").lower().startswith(("deep", "executive")) or outline.broad:
        depth = "deep"
    else:
        depth = "standard"

    return AnswerBlueprint(
        query=query,
        strategy=key,
        framework=strategy,
        central_question=(sub_questions[0].get("question", "") if sub_questions and isinstance(sub_questions[0], dict) else "") or query,
        themes=themes,
        depth=depth,
        comparison=key == "comparison",
        contested=bool(contradictions),
    )


def render_blueprint(blueprint: AnswerBlueprint) -> str:
    """Render the blueprint as writer guidance. Never a heading list."""
    lines: List[str] = []
    if blueprint.framework:
        lines.append(blueprint.framework)
    if blueprint.themes and blueprint.strategy == "broad_research":
        lines.append(
            "The dominant themes the evidence supports, in a sensible order — "
            "use those that carry real findings and drop the rest: "
            + "; ".join(blueprint.themes)
            + "."
        )
    if blueprint.depth == "concise":
        lines.append("Keep this answer concise; the question does not need a long report.")
    elif blueprint.depth == "deep":
        lines.append(
            "Go deeper: mechanisms, concrete numbers with context, and explicit "
            "treatment of conflicting evidence — depth of reasoning, not more headings."
        )
    if blueprint.contested:
        lines.append(
            "Sources disagree on some points: present the disagreement where it "
            "belongs in the flow rather than smoothing it into a single claim."
        )
    return "\n".join(lines)


def group_facts_by_section(
    outline: AnswerOutline,
) -> List[tuple[OutlineSection, List[Dict[str, Any]]]]:
    """Pair each section with its facts (empty list allowed)."""
    return [(section, list(section.facts)) for section in outline.sections]
