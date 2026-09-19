"""Synthesis Agent — turns a verified evidence pool into a cited report.

This revision keeps every integrity guarantee of the previous version and
fixes the thing that made its output hard to read: it wrote the SAME report
for every question.

What changed and why
--------------------
1. THE REPORT SHAPE NOW MATCHES THE QUESTION. `REQUIRED_SECTIONS` was applied
   unconditionally, so "what is a transformer?" shipped with Counterarguments,
   Open Questions, an Auditable Source Ledger and a three-sentence lecture
   about the absence of a counter-evidence search. A `ReportProfile`
   (`direct` / `brief` / `analytical` / `audit`) now selects which sections are
   mandatory, which are added ONLY when they carry real content, and how
   verbose the findings bullets are. Nothing is hidden: a section is dropped
   only when it would have said "none detected", and any profile below `audit`
   still surfaces limitations and conflicts whenever they actually exist.

2. QUESTION-TYPE AWARE WRITING. A comparison, a how-to, a causal "why", a
   decision and a definition are five different documents. `_format_guidance`
   injects the shape the reader expects (criteria + verdict; ordered steps;
   mechanism chains; options + recommendation; plain definition + analogy)
   instead of the single "angle per section" template.

3. SECTIONS ARE ORDERED, NOT APPENDED. Missing sections used to be appended at
   the END in dict order — a report could end with its Executive Summary.
   `_reorder_sections` puts every section in canonical reading order after
   assembly, keeping the writer's own deep-dive sections in their original
   sequence.

4. ONE EVIDENCE SECTION, NOT THREE. "Evidence Strength", the appendix's
   "## Source ledger" and "Auditable Source Ledger" all shipped together
   because the alias tables overlapped and the appendix ran after the presence
   check. Measured accounting is now merged INTO a single
   "## Evidence & Confidence", and the full ledger appears only in the `audit`
   profile.

5. THE AUDIT NO LONGER GRADES THE MACHINE'S OWN PROSE. Deterministic sections
   are tracked as they are added and stripped before auditing, so their counts
   ("A=3, B=5") stop being reported as ungrounded numbers and their bullets
   stop deflating citation density. Invalid markers are dropped BEFORE the
   audit, so the density printed in the report describes the text that shipped.

6. THE LENGTH BAND IS ACTUALLY ENFORCED. The appendix and legend were appended
   after `_trim_to_band`. The tail is now measured first and subtracted from
   the budget, and only writer sections are trimmed — a required section can
   never be silently emptied.

7. ONE FINALIZE PATH. The single-pass and section-wise paths duplicated ~40
   lines of assembly that had already drifted. Both now call `_finalize`.

8. CORRECTNESS FIXES: `_compress_to_themes` no longer discards the
   corroboration count it just incremented; `_value_grounded` no longer accepts
   a wrong year (relative tolerance made 2024 ≈ 2025); `_scrub_pipeline_telemetry`
   is genuinely sentence-granular (it was deleting whole paragraphs, because
   sanitization joins a paragraph into one line); every `int()` on external
   data is guarded; the legend cap no longer silently discards facts.

Backward compatibility: `synthesizer_agent(llm, query, facts, context=None)`
still returns the report as a plain string, and `synthesize()` still returns
`SynthesisResult` with all its previous fields (plus `profile` and
`word_count`). One rename: the mandatory-section key "Evidence Strength" is now
"Evidence & Confidence" (the old name remains a recognized alias).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.core.degradation import (
    EVIDENCE_WEAK,
    PROVIDER_TRANSIENT,
    record_fallback,
)
from app.core.llm import AllProvidersFailedError, LLMClient, PromptTooLargeError
from app.core.logging import get_logger
from app.core.usage import run_seconds_remaining
from app.core.schemas import SynthesizerAnswerModel

from app.agents.contradiction import numeric_ranges, summarize_contradictions
from app.agents.epistemics import EpistemicReport, assess_epistemics
from app.agents.outline import (
    AnswerOutline,
    build_outline,
    group_facts_by_section,
    render_outline,
)
from app.core.section_context import build_section_candidate_pool, select_section_facts
from app.core.synthesis_intelligence import (
    SynthesisIntelligenceReport,
    apply_synthesis_intelligence,
)
from app.agents.answer_quality import length_band
from app.agents.evidence_utils import (
    dedupe_semantic_facts,
    extract_domain,
    extract_numbers,
    filter_facts_by_domain,
    select_diverse,
    semantic_similarity,
    split_into_sentences,
)
from app.agents.sources import canonical_url, classify_source, primary_source_share
from app.agents.research_quality import (
    IndependenceReport,
    ResearchQualityReport,
    TemporalProfile,
    apply_independence,
    assess_independence,
    assess_report_quality,
    independent_corroboration,
    is_factual_sentence,
    render_quality_contract,
    temporal_profile,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Safe coercion — external fact dicts are not trusted to hold clean types
# ---------------------------------------------------------------------------

def _safe_int(value: Any, default: int = 0) -> int:
    """int() that never raises. Dirty evidence must not crash synthesis."""
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _corroboration(fact: Dict[str, Any]) -> int:
    """Independent-source count for a fact, floored at 1.

    Reads `independent_corroboration` when the independence pass has stamped
    it, so three outlets reprinting one wire story count as ONE corroborating
    voice rather than three. Falls back to the raw count when the pass did not
    run, which keeps every existing caller working unchanged.
    """
    return independent_corroboration(fact)


# ---------------------------------------------------------------------------
# Report profiles: the report's shape is chosen, not assumed
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReportProfile:
    """Which sections a report must carry, and how loudly it accounts for itself.

    `required` sections are always present (added from measured state when the
    writer omits them). `conditional` sections are added ONLY when they would
    carry real content — a "Key Figures" section that says "no figures were
    extracted" is noise, and an epistemology note about counter-evidence on a
    definitional query is worse than noise. A conditional section whose signal
    IS present is never suppressed, so nothing adverse is hidden to look tidy.
    """

    name: str
    required: Tuple[str, ...]
    conditional: Tuple[str, ...] = ()
    verbose_findings: bool = False
    include_reasoning: bool = True
    full_appendix: bool = False
    max_findings: int = 6
    writer_sections: Tuple[str, ...] = ()

    def wants(self, canonical: str) -> bool:
        return canonical in self.required or canonical in self.conditional


# Sections the WRITER is asked to produce. Everything else is machine-appended
# from measured state; telling the model to write them only creates duplicates
# it then fills with estimated numbers.
_WRITER_OWNED_ALWAYS = ("Executive Summary", "Key Findings")

PROFILE_DIRECT = ReportProfile(
    name="direct",
    required=("Executive Summary",),
    conditional=(
        "Key Findings",
        "Key Figures",
        "Limitations & Unknowns",
        "Counterarguments & Disputed Points",
    ),
    verbose_findings=False,
    include_reasoning=False,
    full_appendix=False,
    max_findings=5,
    writer_sections=_WRITER_OWNED_ALWAYS,
)

PROFILE_BRIEF = ReportProfile(
    name="brief",
    required=("Executive Summary", "Key Findings", "Evidence & Confidence"),
    conditional=(
        "Key Figures",
        "Limitations & Unknowns",
        "Counterarguments & Disputed Points",
        "Open Questions & Missing Angles",
    ),
    verbose_findings=False,
    include_reasoning=True,
    full_appendix=False,
    max_findings=6,
    writer_sections=_WRITER_OWNED_ALWAYS,
)

PROFILE_ANALYTICAL = ReportProfile(
    name="analytical",
    required=(
        "Executive Summary",
        "Key Findings",
        "Evidence & Confidence",
        "Limitations & Unknowns",
        "Counterarguments & Disputed Points",
    ),
    conditional=("Key Figures", "Open Questions & Missing Angles"),
    verbose_findings=False,
    include_reasoning=True,
    full_appendix=True,
    max_findings=8,
    writer_sections=_WRITER_OWNED_ALWAYS
    + ("Limitations & Unknowns", "Counterarguments & Disputed Points"),
)

PROFILE_AUDIT = ReportProfile(
    name="audit",
    required=(
        "Executive Summary",
        "Key Findings",
        "Evidence & Confidence",
        "Limitations & Unknowns",
        "Counterarguments & Disputed Points",
        "Open Questions & Missing Angles",
        "Auditable Source Ledger",
    ),
    conditional=("Key Figures",),
    verbose_findings=True,
    include_reasoning=True,
    full_appendix=True,
    max_findings=10,
    writer_sections=_WRITER_OWNED_ALWAYS
    + ("Limitations & Unknowns", "Counterarguments & Disputed Points"),
)

PROFILES: Dict[str, ReportProfile] = {
    p.name: p for p in (PROFILE_DIRECT, PROFILE_BRIEF, PROFILE_ANALYTICAL, PROFILE_AUDIT)
}

# Query shapes that are answered, not investigated. A definition or a lookup
# does not become more trustworthy by being wrapped in five audit sections.
_LIGHTWEIGHT_QUERY_TYPES = {
    "definition", "factual", "lookup", "status", "howto", "how_to", "procedural",
}
_CONTESTED_QUERY_TYPES = {
    "decision", "forecast", "evaluative", "strategic", "causal", "comparison",
}


def select_profile(
    ctx: Dict[str, Any] | None,
    *,
    fact_count: int = 0,
) -> ReportProfile:
    """Pick the report shape from mode, query type and what the evidence holds.

    Explicit `ctx["report_profile"]` always wins — a caller that knows what it
    needs is not overruled. Otherwise: deep/executive runs are analytical
    (depth is the product); an audit flag forces the full ledger; a small
    evidence pool answering a lightweight question is `direct`; everything else
    is `brief`. A contested question is never demoted below `brief`, and
    detected contradictions never get a profile that could hide them.
    """
    ctx = ctx or {}
    explicit = str(ctx.get("report_profile", "") or "").strip().lower()
    if explicit in PROFILES:
        return PROFILES[explicit]

    if ctx.get("audit_mode") or ctx.get("compliance_mode"):
        return PROFILE_AUDIT

    mode = str(ctx.get("mode", "standard") or "standard").lower()
    if mode.startswith(("deep", "executive")):
        return PROFILE_ANALYTICAL

    intent = ctx.get("intent") if isinstance(ctx.get("intent"), dict) else {}
    query_type = str(intent.get("query_type", "") or ctx.get("query_type", "") or "").lower()
    level = str(intent.get("explanation_level", "") or "").lower()
    contested = bool(ctx.get("contradictions")) or bool(ctx.get("redteam_findings"))

    if query_type in _CONTESTED_QUERY_TYPES or contested:
        return PROFILE_BRIEF
    if mode == "quick":
        return PROFILE_DIRECT
    if query_type in _LIGHTWEIGHT_QUERY_TYPES or level == "basic":
        # A small, uncontested pool answering a simple question: an answer,
        # its findings, its sources. Limitations still appear if measured.
        return PROFILE_DIRECT if fact_count <= 12 else PROFILE_BRIEF
    return PROFILE_BRIEF


# ---------------------------------------------------------------------------
# Writing contracts injected into the prompt
# ---------------------------------------------------------------------------

# Reasoning-depth contract injected into every section-writing prompt. A live
# deep run produced a report that recited WHAT was true per dimension and
# scored reasoning=65 with the gate note "the report states no limitations" —
# it named facts but never explained why they hold or what follows from them.
# Sections read as analysis when each one argues a mechanism and weighs a
# trade-off; this makes that an explicit requirement, while forbidding the two
# failure modes that would satisfy it dishonestly (invented causes, and
# hedging that says nothing).
_REASONING_DEPTH_INSTRUCTION = (
    "REASONING DEPTH — every section must explain, not just report:\n"
    "- Where the evidence gives a mechanism or cause, state it ('because', "
    "'driven by', 'as a result of'). Connect the facts into a causal chain "
    "instead of listing them.\n"
    "- Where sources disagree or a trade-off exists, present BOTH sides and "
    "name what the disagreement turns on — never average it away or pick a "
    "side silently.\n"
    "- Name the section's limitation or open question when the evidence does "
    "not settle it. An honest gap beats a confident assertion.\n"
    "- Do NOT invent a mechanism, cause, or number that is not in the "
    "evidence. If the evidence only establishes correlation, say so.\n"
    "\n"
    "EXPAND, DON'T RESTATE — this section is ONE part of a larger report and "
    "other sections have already stated the evidence above:\n"
    "- Do NOT open with, or repeat, a fact already given in plain form. A fact "
    "may be re-used ONLY to add something new about it: the mechanism behind "
    "it, its implication or trade-off, a comparison, or why it is uncertain.\n"
    "- If a fact must reappear for the section's argument, EXPAND it in place: "
    "keep its number and context, and attach the new meaning — never restate "
    "it bare and never open a section on a restatement.\n"
    "- State each fact once in its plain form, then spend the section on what "
    "it means. A repeat that carries no new analysis is a defect: a "
    "deterministic refinement pass will rewrite it into a transition, and the "
    "section will read as boilerplate rather than argument.\n"
    "- Never drop or renumber a [n] marker attached to a fact you keep."
)

_REASONING_DEPTH_BLOCK = "\n" + _REASONING_DEPTH_INSTRUCTION + "\n\n"


# The shape a reader expects differs by question type. Without this the writer
# produced "one section per research angle" for a how-to, a comparison and a
# decision alike — technically cited, and useless to the person who asked.
_FORMAT_GUIDANCE: Dict[str, str] = {
    "definition": (
        "SHAPE — this is a definitional question. Open with a one-sentence "
        "plain-language definition a non-expert would understand, then how it "
        "works, then where it matters and what it is commonly confused with. "
        "Include at most one analogy, and mark it as an analogy."
    ),
    "comparison": (
        "SHAPE — this is a comparison. Organise by CRITERION, not by item: each "
        "section compares both options on one dimension with their numbers side "
        "by side. Close with a short verdict naming which option wins for which "
        "use case. 'It depends' is only acceptable when you say what it depends "
        "ON and give the threshold."
    ),
    "howto": (
        "SHAPE — this is a procedural question. Give prerequisites first, then "
        "numbered steps in execution order, one action per step. For any step "
        "that can fail, name the failure mode and how to tell it happened. No "
        "step may depend on information the reader has not been given yet."
    ),
    "causal": (
        "SHAPE — this is a 'why' question. Lead with the mechanism the evidence "
        "supports, stated as a chain (X drives Y because Z). Give the competing "
        "explanation where one exists, and say what evidence would separate "
        "them. Where the evidence is only correlational, say so explicitly."
    ),
    "decision": (
        "SHAPE — this is a decision question. Lay out the realistic options with "
        "their trade-offs, then give a recommendation conditioned on the "
        "reader's situation ('if you value X over Y, then...'), then state what "
        "would change that recommendation. Never recommend without naming the "
        "cost of the recommendation."
    ),
    "forecast": (
        "SHAPE — this is a forward-looking question. State the current measured "
        "level and its as-of date FIRST, label every projection as a projection "
        "with its source's assumptions, and give a range rather than a point "
        "estimate. Never present a projection in the same voice as an "
        "observation."
    ),
    "timeline": (
        "SHAPE — this is a chronological question. Order the report by date, "
        "attach a date to every event, and mark any date the evidence gives "
        "only approximately. End with the current state and its as-of date."
    ),
    "status": (
        "SHAPE — this is a current-state question. The first sentence gives the "
        "current state and the date the evidence was published or retrieved. "
        "Flag explicitly anything that may have changed since."
    ),
    "list": (
        "SHAPE — this is an enumerative question. Give the list as the primary "
        "content, one item per bullet, each self-contained and cited, ordered "
        "by whatever the reader would rank them by (size, date, importance) and "
        "say which ordering you used."
    ),
    "entity": (
        "SHAPE — this question is about a specific person or organisation. "
        "Establish identity first (who exactly, distinguished from similarly "
        "named others), then the facts asked for. If the evidence may conflate "
        "two entities, say so before anything else."
    ),
}

_QUERY_TYPE_PATTERNS: Sequence[Tuple[str, "re.Pattern[str]"]] = (
    ("comparison", re.compile(r"\b(vs\.?|versus|compare[ds]?|comparison|better than|"
                              r"difference between|which (?:is|one))\b", re.I)),
    ("howto", re.compile(r"\b(how (?:do|to|can) |steps? to|guide to|set ?up|install|"
                         r"configure|tutorial)\b", re.I)),
    ("causal", re.compile(r"\b(why|what caused|cause[ds]? of|reason[s]? (?:for|why)|"
                          r"leads? to|because of)\b", re.I)),
    ("decision", re.compile(r"\b(should (?:i|we|they)|worth it|is it worth|"
                            r"recommend|choose between|invest in)\b", re.I)),
    ("forecast", re.compile(r"\b(will |forecast|projection|predicted|by 20\d\d|"
                            r"outlook|future of|expected to)\b", re.I)),
    ("timeline", re.compile(r"\b(timeline|history of|when did|chronolog|"
                            r"over time|evolution of)\b", re.I)),
    ("status", re.compile(r"\b(current(?:ly)?|right now|as of|latest|today|"
                          r"still |up to date)\b", re.I)),
    ("list", re.compile(r"\b(list of|top \d+|examples? of|what are the|"
                        r"types of|kinds of)\b", re.I)),
    ("definition", re.compile(r"^\s*(what (?:is|are|does)|define|meaning of|"
                              r"explain )\b", re.I)),
)


def infer_query_type(query: str) -> str:
    """Best-effort question shape when the intent classifier gave none.

    Deterministic and cheap. Order matters: "what are the differences between
    X and Y" is a comparison, not a definition, so comparison is tested first.
    Returns "" when nothing matches, and the caller falls back to the generic
    angle-per-section structure.
    """
    text = (query or "").strip()
    if not text:
        return ""
    for name, pattern in _QUERY_TYPE_PATTERNS:
        if pattern.search(text):
            return name
    return ""


def _format_guidance(query: str, intent: Dict[str, Any] | None) -> str:
    """The expected document shape for this question, or "" if unrecognised."""
    intent = intent if isinstance(intent, dict) else {}
    declared = str(intent.get("query_type", "") or "").strip().lower()
    key = declared if declared in _FORMAT_GUIDANCE else ""
    if not key:
        alias = {"how_to": "howto", "procedural": "howto", "chronology": "timeline",
                 "evaluative": "decision", "strategic": "decision",
                 "factual": "status", "person": "entity", "organisation": "entity",
                 "organization": "entity"}.get(declared, "")
        key = alias if alias in _FORMAT_GUIDANCE else ""
    if not key:
        key = infer_query_type(query)
    return _FORMAT_GUIDANCE.get(key, "")


SYNTHESIZER_SYSTEM_PROMPT = """
You are the MARS Synthesis Engine — the Final Synthesis Agent. Your only job
is to turn verified research into a clean, premium, highly readable
intelligence report. You do not dump search results; you present knowledge.

━━━ ANSWER FIRST ━━━

The first sentence of the report answers the question that was asked, in
plain language, with its citation. Not what the report will cover, not how
the research was done, not a definition of the topic — the answer. If the
evidence does not support a direct answer, the first sentence says exactly
that and names what is missing.

━━━ ABSOLUTE FORMATTING RULES ━━━

1. Clear visual hierarchy: markdown `## ` headings, a blank line before every
   new section, and headings that are LABELS ("Cost drivers") never questions.
2. Short paragraphs — 3 to 4 sentences maximum.
3. Bullets for genuinely enumerable findings; prose for reasoning. Never a
   wall of text, never a report that is nothing but bullets.
4. Calm, precise, professional tone. Write like a premium research brief, not
   raw notes. Remove repetitive and low-value sentences.
5. Never write "the research found", "the agents discovered", or "according to
   the research". Present the knowledge directly.
6. NEVER expose the pipeline's own internal metrics in the report prose. Do
   not write the confidence score, the relevance/quality score, the count of
   verified facts, the number of facts in the pool, "below the threshold",
   "relevance N/100", or any number describing the research system rather than
   the subject. Those figures are appended automatically from measured state.
   Describe the strength of the EVIDENCE in words ("well-established",
   "single-source") and let the appendix carry the numbers.
7. Plain language over jargon. Use a technical term when it is the precise
   word, and define it on first use when the reader may not know it.
8. LENGTH: follow the length instruction in the prompt exactly. It is a hard
   limit, not a target. A report trimmed by machine loses its last paragraphs.

━━━ CITATION RULES (non-negotiable) ━━━

Every evidence item you are given is prefixed with its citation number, like
`[3] claim text ...`. Use THAT number when you use THAT claim. Do not
renumber, do not guess, do not cite a number you were not given.

  - Every sentence that states a fact, name, date, or number carries at least
    one [n] marker.
  - A sentence combining two claims cites both: "... [2][5]".
  - NEVER write a number, percentage, currency amount, or date that does not
    appear verbatim in the evidence you were given. If the evidence has no
    number for something, say so in words instead of estimating.
  - Analysis sentences that draw a conclusion FROM cited facts need no marker
    of their own, but must not introduce new facts.

━━━ VERIFY BEFORE YOU WRITE A SINGLE WORD ━━━

- Every claim, name, date, or number must trace to at least one provided
  source, cited at the point of use with its given [n].
- If two sources conflict, flag the conflict — never silently pick one.
  Pre-computed ranges are provided; use them.
- Discard extraction artifacts (garbled text, fragments with no clear subject,
  unrelated names). Absence of a clean answer is a valid, reportable finding —
  state it plainly in the first sentence.
- Never mix unrelated people, organizations, or senses of a term. If the
  evidence points to multiple distinct entities with similar names, separate
  them explicitly or state that the identity is ambiguous.
- Prefer the primary source when a primary document and a news summary of it
  both appear; cite the primary and use the summary only for framing.

HARD FAILURE CONDITIONS — reject your own draft if any are true:
- The first sentence does not answer the question
- A section that is one wall of text with no breathing room
- Any factual sentence with no [n] marker
- Any number that is not in the provided evidence
- Any name, number, or fact that is not clearly corroborated
- Unrelated senses or entities blended without explicit separation
- A section written only because the template had a slot for it

Return valid JSON only in this schema:
{"answer": "<final synthesized report with [n] citations>"}
""".strip()


def _render_structure_contract(
    profile: ReportProfile,
    *,
    angles: Sequence[str],
    ambiguous: bool,
    has_figures: bool,
) -> str:
    """The exact section list this report must carry, per profile.

    Two things this fixes over a fixed prompt: the writer is told which
    sections are MACHINE-OWNED (so it stops producing a second, estimated
    "Evidence & Confidence" that then duplicates the measured one), and short
    profiles are not asked for sections whose content does not exist.
    """
    lines: List[str] = ["REQUIRED STRUCTURE (exact order, exact headings):", ""]
    lines.append(
        "1. `## Executive Summary` — 4-6 sentences. First sentence answers the "
        "question directly."
        + (
            " The query term is ambiguous: the numbered disambiguation block "
            "comes first, then the answer for the researched meaning."
            if ambiguous else ""
        )
    )
    lines.append(
        "2. `## Key Findings` — bullets only, one self-contained fact each with "
        f"its [n], highest-confidence first, maximum {profile.max_findings}."
    )
    if angles:
        lines.append(
            "3. Deep-dive sections — one `## ` section per angle below, in "
            "order, titled with a short LABEL (never the question restated). "
            "Skip an angle that adds nothing rather than padding it."
        )
    else:
        lines.append(
            "3. Deep-dive sections — 2-4 `## ` sections on the dimensions the "
            "evidence actually supports, each titled with a short label."
        )
    step = 4
    if has_figures:
        lines.append(
            f"{step}. `## Key Figures` — the quantitative claims, each with its "
            "number, unit, period and scope, each cited."
        )
        step += 1
    for heading in profile.writer_sections:
        if heading in _WRITER_OWNED_ALWAYS:
            continue
        if heading == "Limitations & Unknowns":
            lines.append(
                f"{step}. `## Limitations & Unknowns` — what the evidence does "
                "not settle, in your own words. Be specific; 'more research is "
                "needed' is not a limitation."
            )
            step += 1
        elif heading == "Counterarguments & Disputed Points":
            lines.append(
                f"{step}. `## Counterarguments & Disputed Points` — the "
                "strongest case against the report's own conclusion, and every "
                "source conflict, presented as a disagreement."
            )
            step += 1
    lines.append("")
    lines.append(
        "DO NOT WRITE these sections — they are appended automatically from "
        "measured state and a second, estimated copy is a defect: "
        "Evidence & Confidence, Open Questions & Missing Angles, Source "
        "ledger, Auditable Source Ledger, Sources/References, Evidence "
        "integrity, Reasoning."
    )
    return "\n".join(lines)


# A sentence stating a fact but carrying no [n] is a traceability hole. These
# openers mark analysis/transition sentences, which legitimately carry none.
# The second block covers the report's own scaffolding — the honesty notes the
# extractive path emits are meta-statements about the report, not factual
# claims about the world. Counting them as untraceable facts deflated citation
# density exactly when the pipeline was degraded.
# NOTE: no trailing \b — alternatives ending in ":" can never satisfy one.
_ANALYSIS_LEAD_RE = re.compile(
    r"^\s*(?:taken together|in short|overall|therefore|this means|the picture|"
    r"in practice|by contrast|as a result|on balance|the implication|"
    r"what follows|in other words|put differently|the net effect|"
    r"evidence is thin|confidence:|well-supported:|uncertain:|short answer:|"
    r"conflicting evidence:|could not verify:|pipeline stages on deterministic|"
    r"the evidence spans|based on your question|this report focuses)",
    re.IGNORECASE,
)
_FACTUAL_HINT_RE = re.compile(r"\d|\b(19|20)\d{2}\b|%|\bper cent\b|\bpercent\b")

# Numbers this small are ordinary prose ("three angles", "two sources") and are
# not worth grounding; anything with a unit, currency, percent or year is.
_TRIVIAL_NUMBERS: Set[float] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}

# The disambiguation block an ambiguous-query report MUST open with:
# "1) **Transformer neural network architecture** — attention-based ...".
# Numbered with a closing paren (not "1.") so sentence splitters keep the line
# intact. These lines are definitional common knowledge (the non-researched
# sense has no evidence by design), so the citation audit exempts them rather
# than flagging the pipeline's own disambiguation as untraceable.
_DISAMBIG_LINE_RE = re.compile(
    r"^\s*\d+\)\s*\*\*[^*]{2,120}\*\*\s*[—-]"
    r"|^\s*based on your question,\s*this report focuses on meaning",
    re.IGNORECASE,
)

_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$")


@dataclass
class CitationAudit:
    """What the finished draft actually supports, measured not assumed."""

    total_sentences: int = 0
    cited_sentences: int = 0
    uncited_factual: List[str] = field(default_factory=list)
    invalid_markers: List[int] = field(default_factory=list)
    ungrounded_numbers: List[str] = field(default_factory=list)
    weakly_supported: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def citation_density(self) -> float:
        if not self.total_sentences:
            return 0.0
        return round(self.cited_sentences / self.total_sentences, 4)

    @property
    def is_clean(self) -> bool:
        return not (self.uncited_factual or self.invalid_markers or self.ungrounded_numbers)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_sentences": self.total_sentences,
            "cited_sentences": self.cited_sentences,
            "citation_density": self.citation_density,
            "uncited_factual": self.uncited_factual[:10],
            "invalid_markers": self.invalid_markers[:10],
            "ungrounded_numbers": self.ungrounded_numbers[:10],
            "weakly_supported": self.weakly_supported[:10],
            "is_clean": self.is_clean,
        }


@dataclass
class SynthesisResult:
    """The report plus everything needed to defend it."""

    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    audit: CitationAudit = field(default_factory=CitationAudit)
    used_fallback: bool = False
    angles: List[str] = field(default_factory=list)
    synthesis_intelligence: Dict[str, Any] = field(default_factory=dict)
    profile: str = ""
    word_count: int = 0
    quality: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "audit": self.audit.to_dict(),
            "used_fallback": self.used_fallback,
            "angles": list(self.angles),
            "synthesis_intelligence": dict(self.synthesis_intelligence),
            "profile": self.profile,
            "word_count": self.word_count,
            "quality": dict(self.quality),
        }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

async def synthesizer_agent(
    llm: LLMClient,
    query: str,
    facts: List[Dict[str, Any]],
    context: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> str:
    """Synthesize the final report and return it as markdown text.

    Kept as the public entry point with its original signature and return type.
    `context` carries the decision-grade inputs the workflow already computed:
    contradictions to flag, overall confidence, degraded stages, red-team
    findings, and evidence counts. Extra keyword arguments (outline,
    section_wise, compress_context, profile) pass through to `synthesize` —
    callers that pass none keep the old behaviour.
    """
    result = await synthesize(llm, query, facts, context, **kwargs)
    return result.answer


async def synthesize(
    llm: LLMClient,
    query: str,
    facts: List[Dict[str, Any]],
    context: Dict[str, Any] | None = None,
    *,
    outline: AnswerOutline | None = None,
    section_wise: bool | None = None,
    compress_context: bool | None = None,
    compress_threshold: float | None = None,
    profile: ReportProfile | str | None = None,
) -> SynthesisResult:
    """The same synthesis, returning the audit and legend alongside the text.

    Answer-first outline: the report's section shape is derived from the query
    + evidence BEFORE writing, so a broad question is answered dimension by
    dimension instead of dumping the top-ranked claims. `section_wise` writes
    each outline section as its own call and assembles the result; it degrades
    cleanly to a single pass when unsupported. `compress_context` merges
    near-duplicate claims into one thematic entry (never dropping distinct
    claims) before the writer sees them. `profile` selects the report shape;
    when omitted it is derived from mode, query type and evidence volume.
    """
    usable_facts = dedupe_semantic_facts(filter_facts_by_domain(facts))
    ctx = context or {}
    resolved_profile = _resolve_profile(profile, ctx, len(usable_facts))

    if not usable_facts:
        concept = _normalize_query_concept(query)
        return SynthesisResult(
            answer=(
                f"## Executive Summary\n\n"
                f"No reliable evidence could be retrieved for {concept}, so this "
                "question cannot be answered from research at this time. The "
                "search returned no sources that survived verification — that is "
                "a finding about the available evidence, not about the subject.\n\n"
                "A narrower question, a specific named source or document, or a "
                "different phrasing of the key terms is the most likely way to "
                "get a usable answer."
            ),
            used_fallback=True,
            profile=resolved_profile.name,
        )

    contradictions = [c for c in (ctx.get("contradictions") or []) if isinstance(c, dict)]

    # Adjudicate BEFORE any of it reaches the page. The numeric detector fires
    # on magnitude plus topic similarity, so it cannot tell "$2bn in 2022 vs
    # $3bn in 2024" (growth) from "$2bn vs $3bn for the same year" (a real
    # disagreement). Printing the first as the range "$2-3bn" is a factual
    # error the pipeline invents on its own, so non-conflicts are stripped out
    # here and the genuine ones arrive carrying the rule that decided them.
    epistemics: EpistemicReport = ctx.get("epistemics") if isinstance(
        ctx.get("epistemics"), EpistemicReport
    ) else assess_epistemics(query, usable_facts, contradictions)
    contradictions = _apply_adjudication(contradictions, epistemics)
    ctx = {**ctx, "epistemics": epistemics}

    # Options may arrive as explicit kwargs (tests, direct callers) or inside
    # `context` (the workflow passes them there to keep this entry point's
    # signature stable for test doubles). Explicit wins; context is default.
    if compress_context is None:
        compress_context = bool(ctx.get("compress_context", True))
    if compress_threshold is None:
        compress_threshold = _safe_float(ctx.get("compress_threshold", 0.72), 0.72) or 0.72
    if outline is None:
        ctx_outline = ctx.get("outline")
        if isinstance(ctx_outline, AnswerOutline):
            outline = ctx_outline

    if compress_context:
        usable_facts = _compress_to_themes(usable_facts, similarity_threshold=compress_threshold)

    if outline is None:
        outline = build_outline(
            query,
            usable_facts,
            ctx.get("sub_questions") or [],
            intent=ctx.get("intent") or {},
        )

    mode = str(ctx.get("mode", "standard") or "standard")
    length_hint = _length_hint(mode, ctx)
    intent = ctx.get("intent") or {}

    ambiguity_block = _render_ambiguity_block(intent)
    if ambiguity_block:
        length_hint = f"{length_hint}\n\n{ambiguity_block}"

    guidance = _format_guidance(query, intent)

    # Evidence quality measured BEFORE the writer sees anything, because two of
    # these results change what the writer is allowed to say. Recency decides
    # whether "currently" is a legitimate word in this report; independence
    # decides whether a claim carried by three syndicated copies may be
    # presented as corroborated. Measuring them afterwards would leave the
    # report asserting things the evidence cannot support and only footnoting
    # the problem underneath.
    resolved_query_type = str(
        (intent or {}).get("query_type", "")
        or ctx.get("query_type", "")
        or infer_query_type(query)
    )
    temporal = temporal_profile(usable_facts, query_type=resolved_query_type)
    independence = assess_independence(usable_facts)
    usable_facts = apply_independence(usable_facts, independence)
    quality_contract = render_quality_contract(
        profile=temporal,
        confidence=ctx.get("confidence") if isinstance(ctx.get("confidence"), (int, float)) else None,
        query_type=resolved_query_type,
    )

    # Section-wise synthesis for broad questions: write each outline section as
    # its own bounded call and assemble. This is what stops a broad query from
    # collapsing into one narrow thesis, and it keeps every prompt small enough
    # to survive size-capped providers. Opt-in via the caller (the workflow
    # gates it on mode); when any section fails the whole report falls back to
    # the single-pass writer — never a mixed report and never a crash.
    if section_wise is None:
        section_wise = bool(ctx.get("section_wise", False))
    if section_wise and outline.broad and len(outline.sections) > 1:
        sectioned = await _synthesize_sectioned(
            llm,
            query,
            usable_facts,
            ctx,
            outline,
            contradictions,
            length_hint,
            resolved_profile,
            guidance,
            quality_contract,
            temporal,
            independence,
        )
        if sectioned is not None:
            return sectioned
        logger.warning("[Synthesizer] section-wise path failed; falling back to single pass")

    # Adaptive fact-cap ladder: the writer prompt carries up to 40 facts plus
    # the legend; on providers that cap request size the first attempt can be
    # rejected whole. Shrinking the evidence view keeps synthesis LLM-written
    # instead of degrading to the extractive fallback.
    angles: List[str] = []
    top_facts: List[Dict[str, Any]] = []
    numbered: List[Dict[str, Any]] = []
    cited_facts: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {}
    cap_index = 0
    timeout_second_chance = True
    while cap_index < len(_FACT_CAP_LADDER):
        fact_cap = _FACT_CAP_LADDER[cap_index]
        top_facts = _stratified_top_facts(usable_facts, per_angle=10, cap=fact_cap)
        numbered, cited_facts = _number_facts(top_facts)
        angles = _angles_of(cited_facts)

        user_prompt = (
            f"Main query: {query}\n\n"
            f"{length_hint}\n\n"
            + (f"{guidance}\n\n" if guidance else "")
            + (f"{quality_contract}\n\n" if quality_contract else "")
            + _render_structure_contract(
                resolved_profile,
                angles=angles,
                ambiguous=bool(isinstance(intent, dict) and intent.get("ambiguity")),
                has_figures=_has_numeric_facts(cited_facts),
            )
            + "\n\n"
            + render_outline(outline)
            + (
                "Angles to cover (one section each, in this order):\n"
                + "\n".join(f"- {a}" for a in angles)
                + "\n\n"
                if angles
                else ""
            )
            + _REASONING_DEPTH_BLOCK
            + "Evidence — each line begins with the citation number you MUST use for\n"
            "that claim:\n"
            + _render_evidence_block(cited_facts)
            + "\n\n"
            + _render_ranges_block(contradictions)
            + _render_context_block(ctx)
            + f"Sources (cite by number only):\n{_source_lines(numbered)}\n\n"
            "Return JSON in this schema: "
            '{"answer": "<final synthesized report with [n] citations>"}'
        )
        try:
            payload = await llm.generate_json(
                SYNTHESIZER_SYSTEM_PROMPT,
                user_prompt,
                response_model=SynthesizerAnswerModel,
            )
            break
        except PromptTooLargeError:
            logger.warning(
                "[Synthesizer] provider rejected the prompt at cap=%d facts; retrying smaller",
                fact_cap,
            )
            cap_index += 1
            continue
        except AllProvidersFailedError as exc:
            if "timeout" in str(exc).lower():
                # Slowness, not size or rate. One budget-aware second chance at
                # the SMALLEST evidence view: a flaky provider may still
                # complete a small prompt inside the run's remaining wall-clock.
                # Never more than one.
                if (
                    timeout_second_chance
                    and cap_index < len(_FACT_CAP_LADDER) - 1
                    and run_seconds_remaining() > 120.0
                ):
                    timeout_second_chance = False
                    cap_index = len(_FACT_CAP_LADDER) - 1
                    logger.warning(
                        "[Synthesizer] provider stalled; one second chance at the smallest fact cap"
                    )
                    continue
                logger.warning(
                    "[Synthesizer] provider too slow (timeout); using deterministic fallback"
                )
                record_fallback("synthesizer", reason=PROVIDER_TRANSIENT)
                payload = {}
                break
            # Rate-limited wall: shrink the evidence view — a smaller prompt
            # needs fewer tokens and can still fit a TPM-starved window.
            logger.warning(
                "[Synthesizer] providers unavailable at cap=%d facts (%s); retrying smaller",
                fact_cap, str(exc)[:140],
            )
            cap_index += 1
            continue
        except Exception as exc:
            logger.warning("[Synthesizer] LLM call failed, using deterministic fallback", exc_info=exc)
            record_fallback("synthesizer", reason=PROVIDER_TRANSIENT)
            payload = {}
            break

    answer = str(payload.get("answer", "")).strip() if isinstance(payload, dict) else ""
    if not answer:
        record_fallback("synthesizer", reason=EVIDENCE_WEAK)
        return _deterministic_report(
            query, usable_facts, top_facts, ctx, angles, resolved_profile
        )

    return _finalize(
        answer,
        query=query,
        ctx=ctx,
        usable_facts=usable_facts,
        contradictions=contradictions,
        cited_facts=cited_facts,
        numbered=numbered,
        profile=resolved_profile,
        mode=mode,
        angles=angles,
        temporal=temporal,
        independence=independence,
    )


def _resolve_profile(
    profile: ReportProfile | str | None,
    ctx: Dict[str, Any],
    fact_count: int,
) -> ReportProfile:
    if isinstance(profile, ReportProfile):
        return profile
    if isinstance(profile, str) and profile.strip().lower() in PROFILES:
        return PROFILES[profile.strip().lower()]
    return select_profile(ctx, fact_count=fact_count)


def _length_hint(mode: str, ctx: Dict[str, Any]) -> str:
    """Length instruction bounded by the SAME band the quality gate enforces.

    The writer is never told to exceed what the gate will fail, and deep runs
    are allowed the depth that is their product.
    """
    band_lo, band_hi = length_band(mode)
    intent = ctx.get("intent") or {}
    level = str(intent.get("explanation_level", "") or "")

    if mode.startswith(("deep", "executive")):
        return (
            f"LENGTH: this is a deep-research brief — aim for {band_lo}-{band_hi} "
            "words in TOTAL. Go deeper per angle: mechanisms, numbers with "
            "context, and explicit treatment of conflicting evidence."
        )
    if level == "basic":
        return (
            f"LENGTH: keep the report under {min(600, band_hi)} words. The reader "
            "asked for a basic explanation: open with a plain-language "
            "explanation of the concept and include ONE simple analogy a "
            "non-expert would immediately grasp. Prefer explanation over "
            "statistics."
        )
    return f"LENGTH: keep the report under {band_hi} words."


def _angles_of(cited_facts: Sequence[Dict[str, Any]]) -> List[str]:
    """Distinct sub-questions covered by the cited facts, in first-seen order."""
    angles: List[str] = []
    for fact in cited_facts:
        sub_question = str(fact.get("sub_question", "") or "").strip()
        if sub_question and sub_question not in angles:
            angles.append(sub_question)
    return angles


def _has_numeric_facts(facts: Sequence[Dict[str, Any]], minimum: int = 2) -> bool:
    count = 0
    for fact in facts:
        if re.search(r"\d", str(fact.get("claim", "") or "")):
            count += 1
            if count >= minimum:
                return True
    return False


# ---------------------------------------------------------------------------
# Section model: parse, order, merge, trim — one representation for all of it
# ---------------------------------------------------------------------------

@dataclass
class _Section:
    heading: str
    title: str
    blocks: List[str] = field(default_factory=list)
    key: str = ""

    def render(self) -> str:
        return "\n\n".join([self.heading] + [b for b in self.blocks if b.strip()])

    def words(self) -> int:
        return _count_words(self.render())


def _normalize_heading(text: str) -> str:
    """Comparison key for headings: casefolded, punctuation/space-insensitive."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


# Canonical section identities. Aliases are DISJOINT across keys — the previous
# version listed "source ledger" under both the evidence section and the
# ledger, so a presence check could be satisfied by one and still add the
# other, and reports shipped with three overlapping evidence sections.
_CANONICAL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "executive summary": (
        "executive summary", "summary", "overview", "answer", "direct answer",
        "bottom line", "the short answer",
    ),
    "key findings": (
        "key findings", "main findings", "findings", "takeaways", "key takeaways",
    ),
    "key figures": (
        "key figures", "key quantitative figures", "key numbers",
        "quantitative findings", "the numbers",
    ),
    "counterarguments": (
        "counterarguments and disputed points", "counterarguments & disputed points",
        "counterarguments", "disputed points", "conflicting evidence",
        "contradictions", "the case against",
    ),
    "limitations": (
        "limitations and unknowns", "limitations & unknowns", "limitations",
        "unknowns", "gaps and limitations", "evidence limitations", "caveats",
    ),
    "open questions": (
        "open questions", "open questions & missing angles",
        "open questions and missing angles", "missing angles",
        "unanswered questions", "remaining gaps",
    ),
    "reasoning": (
        "reasoning", "argument", "analysis and reasoning", "reasoning structure",
        "argument structure", "conclusions",
    ),
    "evidence": (
        "evidence strength", "evidence and confidence", "evidence & confidence",
        "evidence quality", "confidence",
    ),
    "standing objections": ("standing objections",),
    "what would change": (
        "what would change this conclusion", "what would change our mind",
        "what would change the conclusion",
    ),
    "evidence integrity": ("evidence integrity",),
    "source ledger": (
        "auditable source ledger", "source ledger", "source audit",
        "evidence ledger", "full source list",
    ),
    "sources": ("sources", "references", "source list", "citations"),
}

_ALIAS_TO_KEY: Dict[str, str] = {
    _normalize_heading(alias): key
    for key, aliases in _CANONICAL_ALIASES.items()
    for alias in aliases
}

# Canonical heading text emitted when this module adds a section.
_CANONICAL_HEADING: Dict[str, str] = {
    "executive summary": "Executive Summary",
    "key findings": "Key Findings",
    "key figures": "Key Figures",
    "counterarguments": "Counterarguments & Disputed Points",
    "limitations": "Limitations & Unknowns",
    "open questions": "Open Questions & Missing Angles",
    "reasoning": "Reasoning",
    "evidence": "Evidence & Confidence",
    "standing objections": "Standing Objections",
    "what would change": "What Would Change This Conclusion",
    "evidence integrity": "Evidence Integrity",
    "source ledger": "Auditable Source Ledger",
    "sources": "Sources",
}

# Reading order. Writer sections (no canonical key) sort at _WRITER_RANK in
# their original sequence, so the deep-dives stay between findings and figures.
_WRITER_RANK = 40
_SECTION_RANK: Dict[str, int] = {
    "executive summary": 10,
    "key findings": 20,
    "key figures": 50,
    "reasoning": 55,
    "counterarguments": 60,
    "limitations": 65,
    "open questions": 70,
    "evidence": 80,
    "standing objections": 84,
    "what would change": 88,
    "evidence integrity": 92,
    "source ledger": 96,
    "sources": 99,
}


def _canonical_key(title: str) -> str:
    return _ALIAS_TO_KEY.get(_normalize_heading(title), "")


def _split_sections(text: str) -> Tuple[List[str], List[_Section]]:
    """Parse a markdown report into a preamble plus H1/H2 sections.

    Splits on H1/H2 only, so `###` sub-structure stays inside its parent
    section. The previous trim split on any `#`, which pulled sub-headings out
    as top-level sections and let a protected parent lose its children.
    """
    preamble: List[str] = []
    sections: List[_Section] = []
    current: Optional[_Section] = None
    for para in (text or "").split("\n\n"):
        stripped = para.strip()
        if not stripped:
            continue
        match = _HEADING_RE.match(stripped)
        if match and len(match.group(1)) <= 2:
            title = match.group(2).strip()
            current = _Section(heading=stripped, title=title, key=_canonical_key(title))
            sections.append(current)
        elif current is None:
            preamble.append(stripped)
        else:
            current.blocks.append(stripped)
    return preamble, sections


def _render_report(preamble: Sequence[str], sections: Sequence[_Section]) -> str:
    parts = [p for p in preamble if p.strip()]
    parts.extend(s.render() for s in sections)
    return "\n\n".join(parts).strip()


def _reorder_sections(text: str) -> str:
    """Put sections in canonical reading order, stable within each rank.

    Deterministic sections used to be appended at the end in dict order, so a
    report whose writer omitted the Executive Summary ended with it. Sorting by
    (rank, original index) fixes the order without reordering the writer's own
    deep-dive sections relative to each other.
    """
    preamble, sections = _split_sections(text)
    if len(sections) < 2:
        return text
    decorated = [
        (_SECTION_RANK.get(section.key, _WRITER_RANK), index, section)
        for index, section in enumerate(sections)
    ]
    decorated.sort(key=lambda item: (item[0], item[1]))
    return _render_report(preamble, [item[2] for item in decorated])


def _dedupe_canonical_sections(text: str) -> str:
    """Keep one section per canonical identity — the longest occurrence.

    Safety net for the case where the writer produced a section the machine
    also appends (or produced two sections that mean the same thing). Writer
    sections with no canonical key are never touched, so genuine sub-topics
    with similar names both survive.
    """
    preamble, sections = _split_sections(text)
    best_by_key: Dict[str, int] = {}
    for index, section in enumerate(sections):
        if not section.key:
            continue
        current = best_by_key.get(section.key)
        if current is None or section.words() > sections[current].words():
            best_by_key[section.key] = index
    kept: List[_Section] = []
    for index, section in enumerate(sections):
        if section.key and best_by_key.get(section.key) != index:
            logger.debug("[Synthesizer] dropped duplicate section '%s'", section.title)
            continue
        kept.append(section)
    return _render_report(preamble, kept)


def _merge_into_section(text: str, key: str, block: str) -> str:
    """Append a measured block to an existing canonical section, or create it.

    This is what keeps the evidence accounting in ONE place. The previous
    version appended a separate `## Source ledger` next to whatever the writer
    had already called "Evidence & Confidence".
    """
    block = (block or "").strip()
    if not block:
        return text
    preamble, sections = _split_sections(text)
    for section in sections:
        if section.key == key:
            section.blocks.append(block)
            return _render_report(preamble, sections)
    heading = f"## {_CANONICAL_HEADING.get(key, key.title())}"
    sections.append(_Section(heading=heading, title=heading[3:], blocks=[block], key=key))
    return _render_report(preamble, sections)


def _strip_canonical_sections(text: str, keys: Iterable[str]) -> str:
    """Remove whole sections by canonical key (used before auditing)."""
    drop = {k for k in keys}
    if not drop:
        return text
    preamble, sections = _split_sections(text)
    return _render_report(preamble, [s for s in sections if s.key not in drop])


def _count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))


def _trim_to_budget(text: str, budget_words: int) -> str:
    """Bring a draft inside a word budget without losing required content.

    Only writer sections (those with no canonical identity) are trimmed, and
    never below their first paragraph — the one carrying the section's
    synthesis claim. Machine sections and the Executive Summary are never
    touched: a length overrun must not delete the limitations section. Drops
    the LAST paragraph of the LONGEST trimmable section, repeatedly, stopping
    as soon as the draft fits or nothing more may be trimmed.
    """
    if budget_words <= 0 or _count_words(text) <= budget_words:
        return text
    preamble, sections = _split_sections(text)
    guard = 0
    while _count_words(_render_report(preamble, sections)) > budget_words and guard < 200:
        guard += 1
        candidates = [s for s in sections if not s.key and len(s.blocks) > 1]
        if not candidates:
            break
        longest = max(candidates, key=lambda s: s.words())
        longest.blocks.pop()
    rendered = _render_report(preamble, sections)
    if _count_words(rendered) > budget_words:
        logger.info(
            "[Synthesizer] draft still %d words over budget after trimming writer "
            "sections; required sections were preserved",
            _count_words(rendered) - budget_words,
        )
    return rendered


_MAX_PARA_SENTENCES = 4
_MAX_PARA_WORDS = 95


def _split_long_paragraphs(text: str) -> str:
    """Break paragraphs that exceed the readability limits at sentence bounds.

    Rule 2 of the prompt asks for 3-4 sentence paragraphs; models comply
    unevenly and the section-wise path concatenates independently written
    prose. This enforces it deterministically. Bullet blocks, headings and
    short paragraphs pass through untouched.
    """
    out: List[str] = []
    for para in (text or "").split("\n\n"):
        stripped = para.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or stripped.lstrip().startswith(("-", "*", ">", "|"))
            or "\n" in stripped
        ):
            out.append(stripped)
            continue
        sentences = [s.strip() for s in split_into_sentences(stripped, max_sentences=40) if s.strip()]
        if len(sentences) <= _MAX_PARA_SENTENCES and _count_words(stripped) <= _MAX_PARA_WORDS:
            out.append(stripped)
            continue
        chunk: List[str] = []
        for sentence in sentences:
            chunk.append(sentence)
            long_enough = len(chunk) >= _MAX_PARA_SENTENCES or _count_words(
                " ".join(chunk)
            ) >= _MAX_PARA_WORDS
            if long_enough:
                out.append(" ".join(chunk))
                chunk = []
        if chunk:
            # A one-sentence tail reads as an orphan; fold it into the previous
            # chunk unless that would recreate an over-long paragraph.
            if len(chunk) == 1 and out and _count_words(out[-1]) + _count_words(chunk[0]) <= _MAX_PARA_WORDS + 25:
                out[-1] = out[-1] + " " + chunk[0]
            else:
                out.append(" ".join(chunk))
    return "\n\n".join(p for p in out if p).strip()


def _dedupe_repeated_bullets(text: str) -> str:
    """Drop an identical bullet repeated inside the same block."""
    out: List[str] = []
    for para in (text or "").split("\n\n"):
        if not para.lstrip().startswith(("- ", "* ")):
            out.append(para)
            continue
        seen: Set[str] = set()
        lines: List[str] = []
        for line in para.split("\n"):
            key = _normalize_heading(line)
            if key and key in seen:
                continue
            seen.add(key)
            lines.append(line)
        out.append("\n".join(lines))
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Finalization — ONE path, shared by single-pass and section-wise synthesis
# ---------------------------------------------------------------------------

def _finalize(
    answer: str,
    *,
    query: str,
    ctx: Dict[str, Any],
    usable_facts: Sequence[Dict[str, Any]],
    contradictions: Sequence[Dict[str, Any]],
    cited_facts: Sequence[Dict[str, Any]],
    numbered: Sequence[Dict[str, Any]],
    profile: ReportProfile,
    mode: str,
    angles: Sequence[str],
    used_fallback: bool = False,
    temporal: TemporalProfile | None = None,
    independence: IndependenceReport | None = None,
) -> SynthesisResult:
    """Assemble, guarantee, audit, budget and order — in that order, once.

    Both synthesis paths previously carried their own copy of this sequence and
    had already drifted. The order here is deliberate:

      sanitize → scrub telemetry → disambiguation → required sections →
      reasoning → refinement pass → dedupe → readability → measured tail →
      trim writer prose to (band - tail) → audit → integrity → order

    The tail is measured BEFORE trimming, which is what makes the length band a
    real contract: the old order trimmed the body and then appended several
    hundred words of appendix, overshooting every time.
    """
    answer = _sanitize_answer_text(answer, query)
    answer = _scrub_pipeline_telemetry(answer)
    answer = _ensure_disambiguation(answer, ctx)

    answer, machine_keys = _add_required_sections(
        answer,
        ctx=ctx,
        usable_facts=usable_facts,
        contradictions=contradictions,
        cited_facts=cited_facts,
        profile=profile,
    )

    if profile.include_reasoning:
        answer, added = _add_reasoning_structure(answer, ctx=ctx)
        machine_keys |= added

    # Cross-section refinement (deterministic; no LLM). Runs AFTER the required
    # sections exist (so Key Findings is registered as a recap) and BEFORE the
    # trim. A bare restatement becomes a transition + analysis sentence (never
    # deleted) and keeps its exact [n] markers.
    answer, si_report = apply_synthesis_intelligence_pass(answer, ctx, query=query)

    answer = _dedupe_canonical_sections(answer)
    answer = _split_long_paragraphs(answer)
    answer = _dedupe_repeated_bullets(answer)

    # Measured tail: evidence accounting merged into the single evidence
    # section, plus the profile-gated objection blocks and the source legend.
    evidence_block = _measured_evidence_block(
        ctx, usable_facts, contradictions, temporal=temporal, independence=independence
    )
    tail_blocks: List[str] = []
    if profile.full_appendix:
        tail_blocks.extend(_objection_blocks(ctx))
    legend = _legend_block(numbered)
    tail_words = _count_words(evidence_block) + sum(
        _count_words(b) for b in tail_blocks
    ) + _count_words(legend)

    # Invalid markers are dropped BEFORE the audit, so the density the report
    # prints describes the text that actually shipped.
    valid_numbers = {_safe_int(s.get("n"), 0) for s in numbered}
    valid_numbers.discard(0)
    recorded_invalid = _invalid_markers(answer, valid_numbers)
    if recorded_invalid:
        answer = _drop_invalid_markers(answer, len(numbered))

    band_lo, band_hi = length_band(mode)
    budget = max(band_lo, band_hi - tail_words - 60)
    answer = _trim_to_budget(answer, budget)

    auditable = _strip_canonical_sections(answer, machine_keys | {"sources", "evidence integrity"})
    audit = audit_citations(auditable, numbered, cited_facts)
    if recorded_invalid:
        audit.invalid_markers = sorted(set(audit.invalid_markers) | set(recorded_invalid))

    # Research-quality layer: per-citation grounding, overclaiming, internal
    # consistency and per-section coverage. Run on the WRITER's prose only
    # (machine sections are measured state, not claims to be audited) and on
    # the section map, so a conflict between two sections is visible.
    quality = assess_report_quality(
        auditable,
        cited_facts=cited_facts,
        usable_facts=usable_facts,
        sections=_section_map(auditable),
        query_type=str(
            ((ctx.get("intent") or {}) if isinstance(ctx.get("intent"), dict) else {}).get(
                "query_type", ""
            )
            or ctx.get("query_type", "")
            or infer_query_type(query)
        ),
        temporal=temporal,
        independence=independence,
    )

    answer = _merge_into_section(answer, "evidence", evidence_block)
    for block in tail_blocks:
        answer = answer.rstrip() + "\n\n" + block

    integrity = _integrity_note(audit, quality)
    if integrity:
        answer = answer.rstrip() + "\n\n" + integrity

    answer = _strip_canonical_sections(answer, {"sources"})
    answer = answer.rstrip() + "\n\n" + legend
    answer = _reorder_sections(answer)

    return SynthesisResult(
        answer=answer,
        sources=list(numbered),
        audit=audit,
        used_fallback=used_fallback,
        angles=list(dict.fromkeys(a for a in angles if a)),
        synthesis_intelligence=si_report.to_dict(),
        profile=profile.name,
        word_count=_count_words(answer),
        quality=quality.to_dict(),
    )


def _section_map(answer: str) -> Dict[str, str]:
    """Section title -> body, for checks that compare sections against each other."""
    _, sections = _split_sections(answer)
    return {s.title: "\n\n".join(s.blocks) for s in sections if s.blocks}


def apply_synthesis_intelligence_pass(
    answer: str,
    ctx: Dict[str, Any] | None = None,
    *,
    query: str = "",
) -> Tuple[str, SynthesisIntelligenceReport]:
    """Run the deterministic cross-section refinement layer on an assembled draft.

    Protected (machine-appended) sections are left whole — they describe
    measured state. Executive Summary and Key Findings are RECAP sections:
    their own text is preserved, but their claims are registered so deep-dive
    sections cannot restate them. Every other writer section is refined against
    the ledger: the first occurrence keeps its citation, an expansion with a new
    analytical dimension is kept intact, and a bare restatement is REWRITTEN
    into a contextual transition + analysis sentence (never deleted), so a
    section opening can never dangle. Evidence signals from `ctx`
    (contradictions, corroboration, primary share) select the analytical
    dimension; the transformation runs with no LLM and never empties a section.
    """
    from app.agents.sources import MACHINE_SECTIONS

    protected = tuple(MACHINE_SECTIONS) + tuple(
        f"## {_CANONICAL_HEADING[key]}"
        for key in (
            "evidence", "limitations", "open questions", "source ledger",
            "sources", "evidence integrity", "standing objections",
            "what would change", "reasoning",
        )
    )
    recap = ("## Executive Summary", "## Key Findings")
    ctx = ctx or {}
    intent = ctx.get("intent") or {}
    signals = {
        "contradicted": bool(ctx.get("contradictions")),
        "corroborated": bool(ctx.get("corroborated")),
        "primary": bool(ctx.get("primary_share")),
        # Adaptive move-selection signals (additive; the layer works without
        # them). The classified query + its type let a "should ... invest"
        # question resolve to a strategic move, a "what caused ..." question to
        # a causal one, and a "compare ..." question to a comparison — so the
        # refinement answers the question that was asked, not a generic one.
        "query": str(query or ""),
        "query_type": str(
            intent.get("query_type", "")
            or ctx.get("query_type", "")
            or infer_query_type(query)
        ),
        "domain": str(intent.get("domain", "") or ""),
    }
    return apply_synthesis_intelligence(
        answer,
        protect_headings=protected,
        recap_headings=recap,
        signals=signals,
    )


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------

def _shorten_heading(text: str) -> str:
    """Reduce a question-shaped heading to a concise label.

    The section writer is told the question to answer; it sometimes emits that
    question as its own `## ` heading. Live deep reports shipped three such
    headings on one query, each duplicating the query and padding the report.
    Deterministic, no LLM; a heading already label-shaped is returned unchanged.
    """
    cleaned = re.sub(r"\s+", " ", (text or "").strip()).strip(" ?.!,")
    if not cleaned:
        return text
    if "?" not in text and len(cleaned.split(" ")) <= 10 and len(cleaned) <= 60:
        return cleaned
    clause = re.split(r"[?:—–]|\s+-\s+", cleaned, maxsplit=1)[0].strip(" ?.!,")
    for prefix in (
        "what are the ", "what are ", "what is the ", "what is ", "what was ",
        "how did the ", "how did ", "how does the ", "how does ", "how do ",
        "how ", "why does ", "why did ", "why ", "which ", "when did ", "when ",
        "to what extent ",
    ):
        if clause.lower().startswith(prefix):
            clause = clause[len(prefix):]
            break
    words = clause.split(" ")
    if len(words) > 9:
        clause = " ".join(words[:9]).rstrip(",;:")
    clause = clause.strip(" ?.!,")
    if not clause:
        return text
    return clause[0].upper() + clause[1:]


def _dedupe_heading(answer: str) -> str:
    """Shorten any remaining question-shaped H2/H3 heading post-assembly."""
    out: List[str] = []
    for line in (answer or "").split("\n"):
        match = re.match(r"^(\s{0,3}#{1,6}\s+)(.*\S)\s*$", line)
        if not match:
            out.append(line)
            continue
        title = match.group(2)
        # Canonical machine headings are already labels and must keep their
        # exact text, or the alias table stops recognising them.
        if _canonical_key(title):
            out.append(line)
        elif title.endswith("?") or len(title.split()) > 12:
            out.append(match.group(1) + _shorten_heading(title))
        else:
            out.append(line)
    return "\n".join(out)


def _strip_duplicate_section_heading(body: str, title: str) -> str:
    """Remove a leading heading the section writer emitted for itself.

    The section-wise assembler prepends `## <title>` to every section body.
    When the writer also opens with `## <title>` (or `# / ### <title>`), the
    heading appears twice in the shipped report. Drop the writer's copy only
    when it names this section; keep any *different* heading so genuine
    sub-structure is never destroyed. Deterministic, no LLM.
    """
    if not body:
        return body
    lines = body.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return body
    match = re.match(r"^#{1,6}\s+(.*\S)\s*$", lines[i].strip())
    if not match:
        return body
    if _normalize_heading(match.group(1)) != _normalize_heading(title):
        return body
    remaining = lines[i + 1:]
    if remaining and not remaining[0].strip():
        remaining = remaining[1:]
    return "\n".join(remaining).strip() if remaining else ""


# ---------------------------------------------------------------------------
# Section-wise synthesis
# ---------------------------------------------------------------------------

def _ranked_section_groups(
    outline: AnswerOutline,
    usable_facts: Sequence[Dict[str, Any]],
) -> List[Tuple[Any, List[Dict[str, Any]]]]:
    """Pair each section with its ranked per-section context.

    Candidate pool for a section is its axis-grouped facts PLUS a BOUNDED,
    relevance-pre-ranked slice of the global pool, so the section can recover a
    relevant claim the coarse grouping placed under another axis without every
    section seeing every fact. The section's own facts are reserved a slot, and
    each section falls back to its axis-grouped facts on any failure, so a
    section is never empty when it had evidence.
    """
    pool = [f for f in (usable_facts or []) if isinstance(f, dict)]
    out: List[Tuple[Any, List[Dict[str, Any]]]] = []
    for section, own in group_facts_by_section(outline):
        candidates, own_ids = build_section_candidate_pool(section, own, pool)
        if not candidates:
            out.append((section, []))
            continue
        selected = select_section_facts(section, candidates, reserved_ids=own_ids)
        if not selected:
            selected = list(own)
        out.append((section, selected))
    return out


async def _synthesize_sectioned(
    llm: LLMClient,
    query: str,
    usable_facts: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    outline: AnswerOutline,
    contradictions: List[Dict[str, Any]],
    length_hint: str,
    profile: ReportProfile,
    guidance: str,
    quality_contract: str = "",
    temporal: TemporalProfile | None = None,
    independence: IndependenceReport | None = None,
) -> SynthesisResult | None:
    """Write each outline section as its own call, then assemble.

    Returns None (so the caller re-runs the single-pass writer) when there are
    fewer than two usable sections, when any section call fails, or when the
    assembled draft is empty. Every deterministic guarantee is applied by the
    shared `_finalize`, exactly as in the single-pass path.
    """
    groups = _ranked_section_groups(outline, usable_facts)
    if len(groups) < 2:
        return None

    # One shared legend across sections so markers stay stable and in range.
    # Number the ORIGINAL fact objects (a single legend + a per-fact marker),
    # then split them back per section. `_assign_numbers` preserves identity,
    # unlike `_number_facts` which copies. The legend budget is sized to the
    # sources actually present: capping it low used to orphan facts whose
    # source missed the legend, emptying sections and collapsing this path.
    all_section_facts: List[Dict[str, Any]] = []
    for _, facts in groups:
        all_section_facts.extend(facts)
    numbered, pairs = _assign_numbers(
        all_section_facts, max_sources=_legend_budget(all_section_facts)
    )
    cited_facts: List[Dict[str, Any]] = []
    marker_by_id: Dict[int, int] = {}
    for fact, index in pairs:
        item = dict(fact)
        item["citation"] = index
        cited_facts.append(item)
        marker_by_id[id(fact)] = index

    facts_by_section: List[List[Dict[str, Any]]] = []
    for _, facts in groups:
        section_cited: List[Dict[str, Any]] = []
        for fact in facts:
            index = marker_by_id.get(id(fact))
            if index is None:
                continue
            item = dict(fact)
            item["citation"] = index
            section_cited.append(item)
        facts_by_section.append(section_cited)

    source_lines = _source_lines(numbered)
    ranges_block = _render_ranges_block(contradictions)
    context_block = _render_context_block(ctx)

    # Per-section word budget so the assembled report lands inside the mode's
    # band. Reserve words for the Executive Summary and the machine-appended
    # sections, then divide the rest across the sections that will be written.
    mode = str(ctx.get("mode", "standard") or "standard")
    band_lo, band_hi = length_band(mode)
    writable = [g for g, cited in zip(groups, facts_by_section) if cited]
    section_count = max(1, len(writable))
    heading_reserve = 300
    per_section_hi = max(110, (band_hi - heading_reserve) // section_count)
    section_length_hint = (
        f"LENGTH: {per_section_hi} words MAXIMUM for this section — a hard "
        f"limit, not a target. The report assembles to at most {band_hi} words "
        f"total across {section_count} sections plus the Executive Summary, so "
        "exceeding this makes the report fail its own length contract. Prefer "
        "one tight synthesis paragraph over two loose ones."
    )

    # Executive Summary first. The quality gate hard-fails clarity without it,
    # and the section-wise path previously assembled only outline sections, so
    # every sectioned report shipped without one. One dedicated writer call
    # over the highest-confidence facts; when it fails the deterministic
    # required-section pass supplies one rather than shipping a blank heading.
    exec_body = ""
    opening_facts = _stratified_top_facts(all_section_facts, per_angle=3, cap=10)
    opening_cited: List[Dict[str, Any]] = []
    for fact in opening_facts:
        index = marker_by_id.get(id(fact))
        if index is None:
            continue
        item = dict(fact)
        item["citation"] = index
        opening_cited.append(item)
    if opening_cited:
        exec_prompt = (
            f"Main query: {query}\n\n"
            f"{length_hint}\n\n"
            + (f"{guidance}\n\n" if guidance else "")
            + (f"{quality_contract}\n\n" if quality_contract else "")
            + "Write ONLY the Executive Summary of a larger report. 4-6 sentences "
            "maximum, in your own words. The FIRST sentence must answer the main "
            "query directly in plain language — not describe what the report "
            "covers. If the query term has multiple distinct meanings, name them "
            "in the first sentence and keep them strictly separate. Do NOT state "
            "any pipeline metric, score or fact count. Do NOT emit a markdown "
            "heading — the assembler adds it. Cite with these exact [n] markers.\n\n"
            "Evidence:\n"
            + _render_evidence_block(opening_cited)
            + "\n\n"
            + ranges_block
            + context_block
            + f"Sources (cite by number only):\n{source_lines}\n\n"
            "Return JSON in this schema: "
            '{"answer": "<Executive Summary prose with [n] citations>"}'
        )
        try:
            exec_payload = await llm.generate_json(
                SYNTHESIZER_SYSTEM_PROMPT,
                exec_prompt,
                response_model=SynthesizerAnswerModel,
            )
            exec_body = (
                str(exec_payload.get("answer", "")).strip()
                if isinstance(exec_payload, dict) else ""
            )
        except Exception as exc:
            logger.warning(
                "[Synthesizer] executive summary call failed (%s); continuing without it",
                str(exc)[:120],
                exc_info=exc,
            )
            exec_body = ""
        exec_body = _strip_duplicate_section_heading(exec_body, "Executive Summary")

    section_bodies: List[str] = []
    written_sections = 0
    if exec_body:
        section_bodies.append(f"## Executive Summary\n\n{exec_body}")

    for (section, _), section_cited in zip(groups, facts_by_section):
        if not section_cited:
            continue
        prompt = (
            f"Main query: {query}\n\n"
            f"{section_length_hint}\n\n"
            + (f"{guidance}\n\n" if guidance else "")
            + (f"{quality_contract}\n\n" if quality_contract else "")
            + f"You are writing ONE section of a larger report, the section titled "
            f"\"{section.title}\" (dimension: {section.axis}).\n"
            + (f"Section goal: {section.coverage_goal}\n" if section.coverage_goal else "")
            + "Write 2-3 tight paragraphs of synthesis for THIS section only. "
            "Do NOT emit a markdown heading for this section — the assembler adds "
            f"the \"## {section.title}\" heading itself. Start directly with prose. "
            "Do not write an Executive Summary, a Sources list, or other sections — "
            "they are added separately. Use these exact [n] markers.\n\n"
            + _REASONING_DEPTH_INSTRUCTION
            + "\n\n"
            "Evidence:\n"
            + _render_evidence_block(section_cited)
            + "\n\n"
            + ranges_block
            + context_block
            + f"Sources (cite by number only):\n{source_lines}\n\n"
            "Return JSON in this schema: "
            '{"answer": "<section markdown with [n] citations>"}'
        )
        try:
            payload = await llm.generate_json(
                SYNTHESIZER_SYSTEM_PROMPT,
                prompt,
                response_model=SynthesizerAnswerModel,
            )
        except Exception as exc:
            logger.warning(
                "[Synthesizer] section '%s' failed (%s); abandoning section-wise path",
                section.title, str(exc)[:120],
                exc_info=exc,
            )
            return None
        body = str(payload.get("answer", "")).strip() if isinstance(payload, dict) else ""
        if not body:
            logger.warning(
                "[Synthesizer] section '%s' empty; abandoning section-wise path", section.title
            )
            return None
        body = _strip_duplicate_section_heading(body, section.title)
        written_sections += 1
        section_bodies.append(f"## {section.title}\n\n{body}")

    if written_sections < 2:
        return None

    logger.info("[Synthesizer] section-wise report: %d sections assembled", len(section_bodies))
    return _finalize(
        "\n\n".join(section_bodies),
        query=query,
        ctx=ctx,
        usable_facts=usable_facts,
        contradictions=contradictions,
        cited_facts=cited_facts,
        numbered=numbered,
        profile=profile,
        mode=mode,
        angles=_angles_of(cited_facts),
        temporal=temporal,
        independence=independence,
    )


# ---------------------------------------------------------------------------
# Mandatory report sections: enforced post-assembly, per profile
# ---------------------------------------------------------------------------

# Canonical headings this assembler may ADD, with the aliases that count as
# already present so a well-written report is never duplicated. Kept as a
# module-level mapping for callers that introspect it.
REQUIRED_SECTIONS: Dict[str, Tuple[str, ...]] = {
    heading: _CANONICAL_ALIASES[key]
    for key, heading in _CANONICAL_HEADING.items()
    if key in (
        "executive summary", "key findings", "evidence", "limitations",
        "counterarguments", "open questions", "key figures", "source ledger",
    )
}

_REQUIRED_KEY_BY_HEADING: Dict[str, str] = {
    heading: key for key, heading in _CANONICAL_HEADING.items()
}


def _present_section_keys(answer: str) -> Set[str]:
    """Normalized headings present in the report body."""
    present: Set[str] = set()
    for match in re.finditer(r"^\s{0,3}#{1,6}\s+(.*\S)\s*$", answer or "", re.M):
        present.add(_normalize_heading(match.group(1)))
    return present


def _normalized_aliases(canonical: str) -> Tuple[str, ...]:
    aliases = REQUIRED_SECTIONS.get(canonical)
    if aliases is None:
        key = _REQUIRED_KEY_BY_HEADING.get(canonical, "")
        aliases = _CANONICAL_ALIASES.get(key, (canonical,))
    return tuple(_normalize_heading(alias) for alias in aliases)


def _finding_confidence(fact: Dict[str, Any]) -> float:
    """Per-claim confidence in [0, 1], from measured signals only.

    Base is the claim's own confidence; verification and independent
    corroboration each lift it, and a single-source or unverified claim is
    capped in the provisional band. Never invented: a fact with no confidence
    field reads 0.0, not a flattering default.
    """
    base = max(0.0, min(1.0, _safe_float(fact.get("confidence", 0.0))))
    if fact.get("verified") is True:
        base += 0.05
    if _corroboration(fact) >= 2:
        base += 0.05
    if _corroboration(fact) <= 1 or fact.get("verified") is not True:
        base = min(base, 0.60)
    return round(max(0.0, min(1.0, base)), 2)


def _finding_grade(fact: Dict[str, Any]) -> str:
    """A/B/C/D evidence grade for one fact (empty when ungraded)."""
    grade = str(fact.get("evidence_grade", "") or "").strip().upper()
    if grade in ("A", "B", "C", "D"):
        return grade
    evidence = fact.get("evidence")
    if isinstance(evidence, dict):
        grade = str(evidence.get("grade", "") or "").strip().upper()
        if grade in ("A", "B", "C", "D"):
            return grade
    return ""


def _finding_justification(fact: Dict[str, Any]) -> str:
    """Short, measured justification for a finding's confidence."""
    parts: List[str] = ["verified against source" if fact.get("verified") is True else "unverified"]
    corroboration = _corroboration(fact)
    parts.append(
        f"{corroboration} independent sources" if corroboration >= 2 else "single source"
    )
    if fact.get("temporal_projection"):
        parts.append("projection, not observed")
    return "; ".join(parts)


def _render_finding_line(fact: Dict[str, Any], *, verbose: bool = False) -> str:
    """One Key-Findings bullet.

    In `audit` profiles every bullet carries its confidence, grade and
    justification. Everywhere else that annotation is noise on the most-read
    part of the report — and it contradicts the prompt's own rule against
    printing pipeline metrics in prose — so a normal bullet is the claim and
    its citation, with a short provisional flag ONLY when the claim is weak.
    Nothing adverse is hidden: the flag appears whenever it applies.
    """
    claim = re.sub(r"\s+", " ", str(fact.get("claim", "") or "")).strip()
    if not claim:
        return ""
    index = _safe_int(fact.get("citation"), 0)
    marker = f" [{index}]" if index else ""
    if verbose:
        confidence = _finding_confidence(fact)
        grade = _finding_grade(fact)
        grade_tag = f", grade {grade}" if grade else ""
        return (
            f"- {claim}{marker} — confidence {confidence:.2f}{grade_tag} "
            f"({_finding_justification(fact)})"
        )
    flags: List[str] = []
    if fact.get("verified") is not True:
        flags.append("unverified")
    if _corroboration(fact) <= 1:
        flags.append("single source")
    if fact.get("temporal_projection"):
        flags.append("projection")
    suffix = f" *({', '.join(flags)})*" if flags else ""
    return f"- {claim}{marker}{suffix}"


def _missing_skeptical_angles(ctx: Dict[str, Any]) -> List[str]:
    """Skeptical/counter-evidence angles the report itself admits are missing.

    Scans the coverage gaps and red-team findings for vocabulary indicating an
    absent counter-evidence angle. When any exists, the report cannot also claim
    it found no credible opposing views: that would be a self-contradiction.
    """
    cues = (
        "counter", "skeptic", "sceptic", "oppos", "disagree", "dissent",
        "critic", "hype", "plateau", "roi", "bubble", "risk", "limitation",
        "alternative", "contradict", "over-hype", "overhype",
    )
    sources: List[str] = []
    gaps = ctx.get("coverage_gaps")
    if isinstance(gaps, (list, tuple)):
        sources.extend(str(g).strip() for g in gaps if str(g).strip())
    findings = ctx.get("redteam_findings")
    if isinstance(findings, list):
        for item in findings:
            if isinstance(item, dict):
                text = str(item.get("statement", "") or "").strip()
                if text:
                    sources.append(text)
    return [text for text in sources if any(cue in text.lower() for cue in cues)]


def _ledger_warnings(ctx: Dict[str, Any]) -> str:
    """Deterministic composition warnings for the evidence section.

    Three composition failures are surfaced, never hidden: regulation dominance
    (>30% of evidence), non-Western under-representation, and a thin primary
    share. Each is a warning, not a silent omission.
    """
    warnings: List[str] = []
    regulation = _safe_float(ctx.get("regulation_share", 0.0))
    if regulation > 0.30:
        warnings.append(
            f"Regulation-sourced evidence is {regulation:.0%} of the pool (>30%). "
            "Regulatory material is reactive context, not the lead story; read "
            "capability, economics and adoption findings with that skew in mind."
        )
    non_western = _safe_float(ctx.get("non_western_share", 0.0))
    if non_western < 0.10:
        warnings.append(
            f"Non-Western sources are {non_western:.0%} of the pool. The picture "
            "may be US/EU-centric; treat global claims as provisional."
        )
    primary = _safe_float(ctx.get("primary_share", 0.0))
    if primary < 0.30:
        warnings.append(
            f"Only {primary:.0%} of sources are primary (papers, official reports, "
            "filings); secondary summaries dominate."
        )
    if not warnings:
        return ""
    return "\n".join(f"- {w}" for w in warnings)


def _section_has_substance(
    key: str,
    *,
    ctx: Dict[str, Any],
    usable_facts: Sequence[Dict[str, Any]],
    contradictions: Sequence[Dict[str, Any]],
    cited_facts: Sequence[Dict[str, Any]],
) -> bool:
    """Would this conditional section say anything? If not, it is not added.

    This is the difference between a report and a form. A "Key Figures" section
    reading "no quantitative figures were extracted" and a Counterarguments
    section reading "no counter-evidence search was completed" are both worse
    than their own absence on a question that never needed them. Anything
    ADVERSE that was actually measured always passes this test.
    """
    if key == "key figures":
        return _has_numeric_facts(cited_facts or usable_facts, minimum=3)
    if key == "counterarguments":
        return bool(contradictions) or bool(ctx.get("redteam_findings")) or bool(
            _missing_skeptical_angles(ctx)
        )
    if key == "limitations":
        if ctx.get("coverage_gaps") or ctx.get("degraded"):
            return True
        weak = sum(
            1 for f in usable_facts
            if f.get("verified") is not True or _corroboration(f) <= 1
        )
        return weak > 0
    if key == "open questions":
        gaps = ctx.get("coverage_gaps")
        return bool(isinstance(gaps, (list, tuple)) and any(str(g).strip() for g in gaps))
    if key == "key findings":
        return bool(cited_facts or usable_facts)
    return True


def _render_required_section(
    canonical: str,
    *,
    ctx: Dict[str, Any],
    usable_facts: Sequence[Dict[str, Any]],
    contradictions: Sequence[Dict[str, Any]],
    cited_facts: Sequence[Dict[str, Any]] = (),
    profile: ReportProfile = PROFILE_BRIEF,
) -> str:
    """Deterministic body for a missing required section, from measured state.

    Never invents facts: each section is assembled from signals the pipeline
    already measured (grade distribution, verified/corroborated counts,
    contradictions, red-team findings, source mix). A genuinely empty signal
    produces an honest "none detected" statement, not a fabricated one.
    """
    total = len(usable_facts)
    verified = sum(1 for f in usable_facts if f.get("verified") is True)
    corroborated = sum(1 for f in usable_facts if _corroboration(f) > 1)
    distribution = ctx.get("evidence_distribution")
    dist = distribution if isinstance(distribution, dict) else {}
    a = _safe_int(dist.get("A", 0))
    b = _safe_int(dist.get("B", 0))
    c = _safe_int(dist.get("C", 0))
    d = _safe_int(dist.get("D", 0))

    if canonical == "Key Findings":
        lines = ["## Key Findings", ""]
        source_facts = list(cited_facts) if cited_facts else list(usable_facts)
        top = _stratified_top_facts(source_facts, per_angle=2, cap=profile.max_findings)
        for fact in top:
            rendered = _render_finding_line(fact, verbose=profile.verbose_findings)
            if rendered:
                lines.append(rendered)
        if len(lines) == 2:
            lines.append("- No verified findings were extracted from the available evidence.")
        return "\n".join(lines)

    if canonical == "Evidence & Confidence":
        if total == 0:
            body = "No verified evidence was available to grade."
        else:
            body = (
                f"{verified} of {total} claims were verified against their cited "
                f"source, and {corroborated} are corroborated by two or more "
                "INDEPENDENT sources (syndicated copies of one story count once)."
            )
            # Only state the grade distribution when grading actually ran.
            # Printing "A=0, B=0, C=0, D=0" announces that nothing was measured
            # while looking like a measurement.
            if a or b or c or d:
                body += (
                    f" Evidence grades: A={a}, B={b}, C={c}, D={d} "
                    "(A/B = verified and strongly or independently sourced)."
                )
        return "## Evidence & Confidence\n\n" + body

    if canonical == "Limitations & Unknowns":
        lines = ["## Limitations & Unknowns", ""]
        if total:
            weak = c + d
            if weak:
                lines.append(
                    f"- {weak} claim(s) are single-source or unverified and should "
                    "be treated as provisional."
                )
            if corroborated < max(1, total // 2):
                lines.append(
                    "- Fewer than half the claims are independently corroborated; "
                    "some findings rest on a single publisher."
                )
        unknown = ctx.get("coverage_gaps")
        if isinstance(unknown, (list, tuple)):
            lines.extend(f"- {str(u).strip()}" for u in unknown[:5] if str(u).strip())
        degraded = ctx.get("degraded") or []
        if isinstance(degraded, list) and degraded:
            lines.append(
                "- Pipeline stages on deterministic fallback: "
                f"{', '.join(str(x) for x in degraded)} — those sections are "
                "extractive, not model-written."
            )
        if len(lines) == 2:
            lines.append("- No specific limitations were measured for this evidence set.")
        return "\n".join(lines)

    if canonical == "Counterarguments & Disputed Points":
        lines = ["## Counterarguments & Disputed Points", ""]
        for item in (contradictions or [])[:5]:
            if not isinstance(item, dict):
                continue
            a_text = str(item.get("claim_a", "") or "")[:140]
            b_text = str(item.get("claim_b", "") or "")[:140]
            if a_text and b_text:
                suffix = (
                    f" RESOLVED: {item.get('resolution', '')}" if item.get("resolved") else ""
                )
                lines.append(f"- \"{a_text}\" conflicts with \"{b_text}\".{suffix}")
        findings = ctx.get("redteam_findings") or []
        if isinstance(findings, list):
            for item in findings[:5]:
                if isinstance(item, dict):
                    statement = str(item.get("statement", "") or "").strip()
                    if statement:
                        lines.append(f"- {statement}")
        if len(lines) == 2:
            # Guard: never claim the absence of counterarguments unless a
            # dedicated counter-evidence search actually ran AND returned
            # nothing AND no missing skeptical angle remains anywhere in the
            # report. An empty section with no failed search is an UNKNOWN, not
            # a clean bill of health.
            counter_searched = bool(ctx.get("counter_evidence_attempted"))
            missing_skeptical = _missing_skeptical_angles(ctx)
            if missing_skeptical:
                lines.append(
                    "- Counter-evidence is INCOMPLETE: this report itself lists "
                    f"{len(missing_skeptical)} unresolved skeptical angle(s) "
                    f"({'; '.join(missing_skeptical[:3])}). No conclusion about the "
                    "absence of credible opposing claims can be drawn."
                )
            elif counter_searched:
                lines.append(
                    "- A dedicated counter-evidence search was run and returned no "
                    "credible opposing claims or source conflicts, and no missing "
                    "skeptical angle remains in this report. This is an absence of "
                    "found counter-evidence, not proof none exists."
                )
            else:
                lines.append(
                    "- No counter-evidence search was completed for this run, so "
                    "the absence of counterarguments here is UNKNOWN, not "
                    "established. Treat the dominant narrative as unopposed by "
                    "default and re-run with the counter-evidence track before "
                    "relying on it."
                )
        return "\n".join(lines)

    if canonical == "Open Questions & Missing Angles":
        lines = ["## Open Questions & Missing Angles", ""]
        gaps = ctx.get("coverage_gaps")
        listed = (
            [str(g).strip() for g in gaps if str(g).strip()]
            if isinstance(gaps, (list, tuple)) else []
        )
        for gap in listed[:8]:
            lines.append(f"- {gap}")
        if not listed:
            lines.append(
                "- No planned angle was left unsourced for this evidence set; "
                "residual uncertainty is captured in the confidence band above."
            )
        return "\n".join(lines)

    if canonical == "Key Figures":
        lines = ["## Key Figures", ""]
        figures = [
            f for f in (cited_facts or usable_facts)
            if re.search(r"\d", str(f.get("claim", "") or ""))
        ]
        for fact in figures[:10]:
            rendered = _render_finding_line(fact, verbose=profile.verbose_findings)
            if rendered:
                lines.append(rendered)
        if len(lines) == 2:
            lines.append("- No quantitative figures were extracted from the evidence.")
        return "\n".join(lines)

    if canonical == "Auditable Source Ledger":
        lines = ["## Auditable Source Ledger", ""]
        seen: Set[str] = set()
        for fact in usable_facts:
            source = str(fact.get("source", "") or "").strip()
            if not source or source in seen:
                continue
            seen.add(source)
            fetched = str(fact.get("fetched_at", "") or "")
            published = str(fact.get("published_at", "") or "")
            pulled = str(fact.get("retrieved_at", "") or fetched or "unrecorded")
            flags: List[str] = []
            if _corroboration(fact) <= 1:
                flags.append("single-source / provisional")
            if fact.get("temporal_projection"):
                flags.append("projection (not observed)")
            if fact.get("verified") is not True:
                flags.append("unverified")
            dates = ([f"published {published}"] if published else []) + [f"retrieved {pulled}"]
            suffix = f" [{'; '.join(flags)}]" if flags else ""
            lines.append(f"- {source} ({'; '.join(dates)}){suffix}")
        if len(lines) == 2:
            lines.append("- No sources were retained for this run.")
        return "\n".join(lines)

    # Executive Summary fallback (only when the writer omitted it entirely).
    # Leads with the strongest claim so the reader gets an ANSWER, not an
    # inventory of the pipeline's activity.
    headline = ""
    ranked = _stratified_top_facts(list(cited_facts) or list(usable_facts), per_angle=1, cap=1)
    if ranked:
        claim = re.sub(r"\s+", " ", str(ranked[0].get("claim", "") or "")).strip()
        index = _safe_int(ranked[0].get("citation"), 0)
        if claim:
            headline = f"Short answer: {claim.rstrip('.')}" + (f" [{index}]." if index else ".")
    confidence_note = ""
    score = ctx.get("confidence")
    if isinstance(score, (int, float)):
        band = "High" if score >= 0.75 else "Medium" if score >= 0.5 else "Low"
        confidence_note = f" Overall confidence: {band} ({float(score):.2f})."
    return (
        "## Executive Summary\n\n"
        + (headline + "\n\n" if headline else "")
        + f"This report draws on {verified} verified claim(s) out of {total} "
        f"extracted.{confidence_note}"
    )


def _add_required_sections(
    answer: str,
    *,
    ctx: Dict[str, Any],
    usable_facts: Sequence[Dict[str, Any]],
    contradictions: Sequence[Dict[str, Any]],
    cited_facts: Sequence[Dict[str, Any]] = (),
    profile: ReportProfile = PROFILE_BRIEF,
) -> Tuple[str, Set[str]]:
    """Add missing sections for this profile; report which ones were machine-made.

    The returned key set is what makes the citation audit honest: the audit used
    to grade these deterministic sections as if the writer had produced them,
    so their counts became "ungrounded numbers" and their bullets deflated
    citation density.
    """
    if not answer:
        return answer, set()
    present = _present_section_keys(answer)
    added_keys: Set[str] = set()
    blocks: List[str] = []

    for canonical in profile.required + profile.conditional:
        key = _REQUIRED_KEY_BY_HEADING.get(canonical, _canonical_key(canonical))
        if any(alias in present for alias in _normalized_aliases(canonical)):
            continue
        if canonical in profile.conditional and not _section_has_substance(
            key,
            ctx=ctx,
            usable_facts=usable_facts,
            contradictions=contradictions,
            cited_facts=cited_facts,
        ):
            continue
        blocks.append(
            _render_required_section(
                canonical,
                ctx=ctx,
                usable_facts=usable_facts,
                contradictions=contradictions,
                cited_facts=cited_facts,
                profile=profile,
            )
        )
        if key:
            added_keys.add(key)

    if not blocks:
        return answer, added_keys
    return answer.rstrip() + "\n\n" + "\n\n".join(blocks), added_keys


def ensure_required_sections(
    answer: str,
    *,
    ctx: Dict[str, Any],
    usable_facts: Sequence[Dict[str, Any]],
    contradictions: Sequence[Dict[str, Any]],
    cited_facts: Sequence[Dict[str, Any]] = (),
    profile: ReportProfile | str | None = None,
) -> str:
    """Guarantee the profile's mandatory sections are present, adding any missing.

    Deterministic and total. A missing section is appended from measured state
    (never invented), so the writer cannot omit limitations or counterarguments
    and ship a report that hides them. Existing, differently titled sections
    that mean the same thing satisfy the requirement and are left untouched.
    """
    resolved = _resolve_profile(profile, ctx or {}, len(usable_facts))
    updated, _ = _add_required_sections(
        answer,
        ctx=ctx,
        usable_facts=usable_facts,
        contradictions=contradictions,
        cited_facts=cited_facts,
        profile=resolved,
    )
    return updated


def _has_reasoning_section(answer: str) -> bool:
    present = _present_section_keys(answer)
    return any(
        _normalize_heading(alias) in present
        for alias in _CANONICAL_ALIASES["reasoning"]
    )


def _add_reasoning_structure(answer: str, *, ctx: Dict[str, Any]) -> Tuple[str, Set[str]]:
    """Append the deterministic argument structure when the writer omitted it.

    The ReasoningMap is the argument layer, so its loss must never depend on the
    writer choosing to surface it. When the report already carries a
    Reasoning/Argument/Conclusions section this is a no-op; otherwise the
    measured structure is appended verbatim — no invented content. Total and
    fail-safe: a missing, empty or unrenderable map leaves the answer untouched.
    """
    if not answer:
        return answer, set()
    reasoning = (ctx or {}).get("reasoning")
    if reasoning is None or not hasattr(reasoning, "render_for_writer"):
        return answer, set()
    if _has_reasoning_section(answer):
        return answer, set()
    render = getattr(reasoning, "render_for_report", None) or getattr(
        reasoning, "render_for_writer"
    )
    try:
        rendered = str(render()).strip()
    except Exception:  # noqa: BLE001
        return answer, set()
    if not rendered:
        return answer, set()
    return answer.rstrip() + "\n\n## Reasoning\n\n" + rendered, {"reasoning"}


def ensure_reasoning_structure(answer: str, *, ctx: Dict[str, Any]) -> str:
    """Backward-compatible wrapper around `_add_reasoning_structure`."""
    updated, _ = _add_reasoning_structure(answer, ctx=ctx)
    return updated


# ---------------------------------------------------------------------------
# Citation numbering: bind every fact to the number the model must cite
# ---------------------------------------------------------------------------

MAX_LEGEND_SOURCES = 14

# Hard ceiling on the legend. The old cap of 14 was applied to the LEGEND while
# up to 40 facts were selected, so every fact from source 15 onwards was
# silently discarded — on a broad query with 30 domains that is most of the
# evidence, and in the section-wise path it emptied sections and collapsed the
# whole path. A legend line costs ~15 tokens; discarding verified evidence
# costs the answer. The legend now sizes itself to the sources actually
# present, bounded generously.
MAX_LEGEND_SOURCES_HARD = 40

# Caps tried in order when the provider rejects the request size.
_FACT_CAP_LADDER = (40, 24, 14)


def _legend_budget(facts: Sequence[Dict[str, Any]]) -> int:
    """Legend size for this fact set: enough to cite every fact, within reason."""
    documents = {
        canonical_url(str(f.get("source", "") or "")) or str(f.get("source", "") or "")
        for f in facts
        if f.get("source")
    }
    documents.discard("")
    return max(MAX_LEGEND_SOURCES, min(MAX_LEGEND_SOURCES_HARD, len(documents)))


def _number_facts(
    top_facts: Sequence[Dict[str, Any]], max_sources: int | None = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Number the sources and stamp each fact with the number that cites it.

    Grouping is by CANONICAL url, so `page?utm_source=x` and `page` are one
    source with one number instead of two entries the model may cite
    inconsistently. Facts whose source cannot make the legend are not sent to
    the model at all — handing the writer evidence it has no legal way to cite
    makes it either drop a real finding or invent a marker — but the legend is
    now sized so that discarding is rare rather than routine, and any discard
    is logged.
    """
    budget = max_sources if max_sources is not None else _legend_budget(top_facts)
    numbered, pairs = _assign_numbers(top_facts, budget)
    cited: List[Dict[str, Any]] = []
    for fact, index in pairs:
        item = dict(fact)
        item["citation"] = index
        cited.append(item)
    dropped = len([f for f in top_facts if f.get("source")]) - len(cited)
    if dropped > 0:
        logger.info(
            "[Synthesizer] %d fact(s) dropped: their source exceeded the %d-entry legend",
            dropped, budget,
        )
    return numbered, cited


def _assign_numbers(
    facts: Sequence[Dict[str, Any]], max_sources: int | None = None
) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], int]]]:
    """Legend entries plus the (fact, number) pairs, keeping fact identity.

    Identity matters for the extractive path and the section-wise path, which
    need to attach the right marker to a claim they have already selected.
    """
    budget = max_sources if max_sources is not None else _legend_budget(facts)
    numbered: List[Dict[str, Any]] = []
    number_by_document: Dict[str, int] = {}
    pairs: List[Tuple[Dict[str, Any], int]] = []

    for fact in facts or []:
        url = str(fact.get("source", "") or "")
        if not url:
            continue
        document = canonical_url(url) or url
        index = number_by_document.get(document)
        if index is None:
            if len(numbered) >= max(1, budget):
                continue
            profile = classify_source(url)
            index = len(numbered) + 1
            number_by_document[document] = index
            numbered.append(
                {
                    "n": index,
                    "domain": profile.domain or extract_domain(url) or url or "unknown source",
                    "url": url,
                    "tier": profile.tier,
                    "authority": round(profile.authority, 2),
                    "primary": bool(profile.is_primary),
                }
            )
        pairs.append((fact, index))

    return numbered, pairs


def _render_evidence_block(cited_facts: Sequence[Dict[str, Any]], limit: int = 40) -> str:
    """One line per claim, citation number first.

    Sending raw dicts spends tokens on keys the writer cannot use and buries the
    attribution the writer needs most.
    """
    lines: List[str] = []
    for fact in list(cited_facts)[:limit]:
        claim = re.sub(r"\s+", " ", str(fact.get("claim", "") or "")).strip()
        if not claim:
            continue
        marks: List[str] = []
        if fact.get("is_primary"):
            marks.append("primary")
        if fact.get("verified") is True:
            marks.append("verified")
        corroboration = _corroboration(fact)
        if corroboration > 1:
            marks.append(f"{corroboration} sources agree")
        if fact.get("direct_quote"):
            marks.append("direct quote")
        angle = str(fact.get("sub_question", "") or "").strip()
        meta = f" ({', '.join(marks)})" if marks else ""
        angle_tag = f" [angle: {angle}]" if angle else ""
        sense = str(fact.get("sense", "") or "").strip()
        sense_tag = f" [sense: {sense}]" if sense else ""
        lines.append(f"[{fact.get('citation')}] {claim}{meta}{angle_tag}{sense_tag}")
    return "\n".join(lines)


def _source_lines(numbered: Sequence[Dict[str, Any]]) -> str:
    out: List[str] = []
    for source in numbered:
        tier = str(source.get("tier", "") or "")
        label = f"[{source['n']}] {source['domain']}"
        if tier:
            label += f" — {tier}{', primary source' if source.get('primary') else ''}"
        if source.get("url"):
            label += f" ({source['url']})"
        out.append(label)
    return "\n".join(out)


def _apply_adjudication(
    contradictions: Sequence[Dict[str, Any]],
    epistemics: "EpistemicReport",
) -> List[Dict[str, Any]]:
    """Drop non-conflicts; stamp the survivors with their verdict.

    A detected pair that turned out to be a time series, a scope mismatch or a
    unit mismatch is NOT a disagreement, and every downstream renderer treats
    the contradiction list as disagreements: the ranges block turns it into
    "report the range X to Y", the Counterarguments section lists it as a
    dispute, and the confidence engine penalises it. Removing them here fixes
    all three at once, and carrying `resolution_note` through means the ones
    that survive can be reported as adjudicated rather than as open questions.
    """
    if not contradictions:
        return []
    by_pair = {
        (r.claim_a, r.claim_b): r for r in getattr(epistemics, "resolutions", []) or []
    }
    out: List[Dict[str, Any]] = []
    for item in contradictions:
        if not isinstance(item, dict):
            continue
        verdict = by_pair.get(
            (str(item.get("claim_a", "") or ""), str(item.get("claim_b", "") or ""))
        )
        if verdict is None:
            out.append(item)
            continue
        if not verdict.is_real_conflict:
            # Not a disagreement. Reported in the evidence section as what it
            # actually is, never as a conflicting range.
            continue
        entry = dict(item)
        if verdict.resolved:
            entry["resolved"] = True
            entry["resolution"] = verdict.explanation
            entry["resolution_rule"] = verdict.rule
            entry["winner"] = verdict.winner
        out.append(entry)
    return out


def _render_ranges_block(contradictions: Sequence[Dict[str, Any]]) -> str:
    """Pre-computed ranges for numeric conflicts.

    The prompt has always forbidden averaging conflicting numbers but never
    supplied the range to use instead — so the model either picked one or hedged
    vaguely. The arithmetic is done here.
    """
    # An adjudicated conflict has an answer, so offering the writer a range
    # would invite it to hedge something the evidence actually settles.
    open_conflicts = [
        c for c in (contradictions or [])
        if isinstance(c, dict) and not c.get("resolved")
    ]
    ranges = numeric_ranges(open_conflicts)
    resolved = [
        c for c in (contradictions or [])
        if isinstance(c, dict) and c.get("resolved") and c.get("resolution")
    ]
    verdicts = ""
    if resolved:
        lines = [
            f"- Conflict resolved: prefer \"{str(c.get('claim_a' if c.get('winner') == 'a' else 'claim_b', ''))[:120]}\" "
            f"because {c.get('resolution')}. State the resolved figure and note that "
            "a weaker source disagrees; do NOT present it as an open question."
            for c in resolved[:4]
        ]
        verdicts = "Adjudicated source conflicts:\n" + "\n".join(lines) + "\n\n"
    if not ranges:
        return verdicts
    lines: List[str] = []
    for item in ranges[:4]:
        unit = "" if item["unit"] in ("", "dimensionless") else f" {item['unit']}"
        domains = ", ".join(sorted({extract_domain(s) for s in item["sources"] if s})[:4])
        lines.append(
            f"- Sources disagree: report the range {item['low']:g}{unit} to "
            f"{item['high']:g}{unit} (spread {item['spread']:g}{unit}); "
            f"disagreeing sources: {domains or 'multiple'}. Never average these."
        )
    return verdicts + "Conflicting quantities, pre-computed as ranges:\n" + "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Post-generation audit
# ---------------------------------------------------------------------------

def audit_citations(
    answer: str,
    numbered: Sequence[Dict[str, Any]],
    cited_facts: Sequence[Dict[str, Any]],
) -> CitationAudit:
    """Measure the draft's traceability instead of assuming it.

    Checks, in order of consequence:
      1. numbers in the prose that appear in no evidence fact (fabrication),
      2. citation markers pointing outside the legend (unresolvable),
      3. factual sentences with no marker at all (untraceable),
      4. sentences whose cited source has little textual overlap with them
         (probable misattribution).

    The caller strips machine-written sections first: grading the pipeline's own
    deterministic appendix reported holes the writer never made.
    """
    audit = CitationAudit()
    body = _strip_sections(answer, ("## Sources", "## Evidence Integrity", "## Evidence integrity"))
    valid = {_safe_int(s.get("n"), 0) for s in numbered}
    valid.discard(0)
    claims_by_number: Dict[int, List[str]] = {}
    for fact in cited_facts:
        index = _safe_int(fact.get("citation"), 0)
        if not index:
            continue
        claims_by_number.setdefault(index, []).append(str(fact.get("claim", "") or ""))
    evidence_text = " ".join(
        str(f.get("claim", "") or "") + " " + str(f.get("direct_quote", "") or "")
        for f in cited_facts
    )
    evidence_values = {round(q.value, 4) for q in extract_numbers(evidence_text, limit=400)}

    for sentence in _audit_units(body):
        stripped = sentence.lstrip("-* ").strip()
        if not stripped:
            continue
        markers = [int(m) for m in re.findall(r"\[(\d+)\]", stripped)]
        # Named entities, quotations, worded dates and attribution verbs all
        # make a sentence checkable. The old digits-only test let every
        # non-numeric assertion ("Acme acquired Beta") pass uncited.
        carries_fact = is_factual_sentence(stripped)
        analysis_lead = bool(_ANALYSIS_LEAD_RE.match(stripped))
        if not markers and (
            (analysis_lead and not carries_fact) or _DISAMBIG_LINE_RE.match(stripped)
        ):
            # Analysis/transition prose, the report's own scaffolding, and the
            # disambiguation lines are not evidence claims; they belong in
            # neither the numerator nor the denominator of citation density. An
            # analysis opener that nevertheless states a number or a date IS a
            # factual claim — "Overall," must not be an exemption from citing.
            continue
        audit.total_sentences += 1
        if markers:
            audit.cited_sentences += 1
        for marker in markers:
            if marker not in valid and marker not in audit.invalid_markers:
                audit.invalid_markers.append(marker)

        for quantity in extract_numbers(stripped, limit=12):
            value = round(quantity.value, 4)
            if value in _TRIVIAL_NUMBERS and quantity.unit == "":
                continue
            if not _value_grounded(value, evidence_values):
                raw = quantity.raw.strip()
                if raw and raw not in audit.ungrounded_numbers:
                    audit.ungrounded_numbers.append(raw)

        if not markers:
            if carries_fact:
                audit.uncited_factual.append(stripped[:200])
            continue

        # Misattribution check: the cited source should have something to do
        # with the sentence. Uses the max over that source's claims because one
        # sentence may compress several claims from the same page.
        supporting = [c for m in markers for c in claims_by_number.get(m, [])]
        if supporting:
            best = max(semantic_similarity(stripped, claim) for claim in supporting)
            if best < 0.12:
                audit.weakly_supported.append(
                    {"sentence": stripped[:200], "cited": markers, "support": round(best, 3)}
                )

    return audit


def _audit_units(body: str) -> List[str]:
    """Split a markdown report into auditable units.

    Line-aware on purpose. `split_into_sentences` alone splits on terminal
    punctuation, and bullets frequently have none — so a whole bullet list
    collapses into one "sentence" that counts as cited if any single bullet
    carries a marker. Uncited bullets are exactly what this audit exists to
    catch, so each line is split first, then each line into sentences.
    """
    units: List[str] = []
    for raw_line in (body or "").replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for sentence in split_into_sentences(line, max_sentences=20):
            text = sentence.strip()
            if text:
                units.append(text)
        if len(units) >= 400:
            break
    return units[:400]


# Four-digit integers in the year range are compared exactly: a 2% relative
# tolerance made 2024 and 2025 "the same number", so a wrong year passed the
# grounding check silently — the single most plausible fabrication in a
# research report.
def _is_year(value: float) -> bool:
    return float(value).is_integer() and 1000.0 <= value <= 2999.0


def _value_grounded(value: float, evidence_values: Set[float], tolerance: float = 0.02) -> bool:
    """A drafted number is grounded when some evidence number matches it.

    Tolerance is relative, so 2.5 million vs 2,500,000 and rounding differences
    between a source and a paraphrase both pass, while a fabricated figure does
    not. Years and small integers require an exact match, because a relative
    tolerance is meaningless for them.
    """
    if value in evidence_values:
        return True
    if _is_year(value) or abs(value) < 20:
        return False
    for known in evidence_values:
        if _is_year(known):
            continue
        scale = max(abs(known), abs(value), 1e-9)
        if abs(known - value) / scale <= tolerance:
            return True
    return False


def _invalid_markers(answer: str, valid: Set[int]) -> List[int]:
    """Markers in the text that resolve to nothing in the legend."""
    found = {int(m) for m in re.findall(r"\[(\d+)\]", answer or "")}
    return sorted(found - valid)


def _drop_invalid_markers(answer: str, count: int) -> str:
    """Remove [n] markers that resolve to nothing in the legend."""

    def _keep(match: "re.Match[str]") -> str:
        index = _safe_int(match.group(1), 0)
        return match.group(0) if 1 <= index <= count else ""

    return re.sub(r"\[(\d+)\]", _keep, answer or "")


def _integrity_note(
    audit: CitationAudit,
    quality: "ResearchQualityReport | None" = None,
) -> str:
    """Disclose what the audit found, in the report itself.

    A research system that silently repairs its own citations trains its users
    to trust output it has not earned. If the draft had holes, the reader is
    told which ones — including the research-quality findings (misattributed
    figures, overclaiming, internal conflicts, stale or echoed evidence),
    which are the defects an expert reader would otherwise find first.
    """
    quality_note = quality.render_note() if quality is not None else ""
    if audit.is_clean and not quality_note:
        return ""
    lines = ["## Evidence Integrity", ""]
    if audit.ungrounded_numbers:
        lines.append(
            "Figures without a matching source in the evidence pool (treat as "
            f"unverified): {', '.join(audit.ungrounded_numbers[:6])}."
        )
    if audit.invalid_markers:
        lines.append(
            f"{len(audit.invalid_markers)} citation marker(s) referenced a source "
            "outside the evidence list and were removed."
        )
    if audit.uncited_factual:
        lines.append(
            f"{len(audit.uncited_factual)} factual statement(s) carry no citation, "
            f"beginning: \"{audit.uncited_factual[0][:120]}\"."
        )
    if audit.weakly_supported:
        lines.append(
            f"{len(audit.weakly_supported)} statement(s) show weak overlap with the "
            "source cited beside them; verify those against the linked source."
        )
    if audit.total_sentences:
        lines.append(
            f"Citation density: {audit.citation_density:.0%} of factual sentences cited."
        )
    # Quality findings lead: a misattributed figure or an internal conflict is
    # what an expert reader would catch first, so the report must raise it
    # first rather than burying it under routine citation bookkeeping.
    audit_body = "\n\n".join(line for line in lines[2:] if line.strip())
    body = "\n\n".join(part for part in (quality_note, audit_body) if part.strip())
    return "## Evidence Integrity\n\n" + body


def _strip_sections(answer: str, headings: Sequence[str]) -> str:
    """Drop appended machine-written sections before auditing the prose."""
    text = answer or ""
    for heading in headings:
        index = text.find(f"\n{heading}")
        if index != -1:
            text = text[:index]
    return text


# ---------------------------------------------------------------------------
# Measured accounting: sections that must never be model-written
# ---------------------------------------------------------------------------

def _measured_evidence_block(
    ctx: Dict[str, Any],
    usable_facts: Sequence[Dict[str, Any]],
    contradictions: Sequence[Dict[str, Any]],
    *,
    temporal: TemporalProfile | None = None,
    independence: IndependenceReport | None = None,
) -> str:
    """The accounting the model cannot be trusted to produce.

    The model writes an evidence section with numbers it estimates. The real
    counts, the confidence breakdown, the source mix and the composition
    warnings are assembled here from measured state and MERGED INTO the single
    evidence section, rather than appended as a competing one.
    """
    lines: List[str] = []

    distribution = ctx.get("evidence_distribution")
    if isinstance(distribution, dict) and any(
        _safe_int(distribution.get(g, 0)) for g in ("A", "B", "C", "D")
    ):
        lines.append(
            "- Evidence grades: "
            f"A={_safe_int(distribution.get('A', 0))}, "
            f"B={_safe_int(distribution.get('B', 0))}, "
            f"C={_safe_int(distribution.get('C', 0))}, "
            f"D={_safe_int(distribution.get('D', 0))} "
            "(A/B = verified and strongly or independently sourced)"
        )

    # Dating first: a reader deciding whether to act on this report needs to
    # know how old it is before anything else in the accounting.
    if temporal is not None:
        lines.append(temporal.as_of_line())

    urls = [str(f.get("source", "") or "") for f in usable_facts if f.get("source")]
    share = primary_source_share(urls)
    documents = len({canonical_url(u) or u for u in urls} - {""})
    domains = len({extract_domain(u) for u in urls} - {""})
    if independence is not None and independence.effective_sources and (
        independence.effective_sources < domains
    ):
        lines.append(
            f"- Documents read: {documents} across {domains} domain(s), but only "
            f"about {independence.effective_sources} are independent "
            "(the rest carry near-identical wording)"
        )
    else:
        lines.append(f"- Documents read: {documents} across {domains} independent domain(s)")
    lines.append(
        f"- Primary sources (official filings, papers, datasets, standards): {share:.0%}"
    )
    verified = sum(1 for f in usable_facts if f.get("verified") is True)
    lines.append(
        f"- Claims verified against their cited source: {verified}/{len(usable_facts)}"
    )
    corroborated = sum(1 for f in usable_facts if _corroboration(f) > 1)
    if corroborated:
        lines.append(f"- Claims independently corroborated by 2+ sources: {corroborated}")
    conflict = summarize_contradictions(contradictions)
    if conflict["total"]:
        lines.append(
            f"- Source conflicts detected: {conflict['cross_source']} across sources "
            f"({conflict['severe']} severe); reported as ranges above"
        )

    report = ctx.get("confidence_report")
    rendered = ""
    if report is not None and hasattr(report, "render"):
        try:
            rendered = str(report.render()).strip()
        except Exception:  # noqa: BLE001 - never let a panel break the report
            rendered = ""

    # What the adjudication concluded belongs in the report, not just in the
    # return value: a reader who sees "sources disagree" removed deserves to
    # know it was removed because the figures were measuring different years.
    epistemics = ctx.get("epistemics")
    if epistemics is not None and hasattr(epistemics, "notes"):
        for note in (epistemics.notes() or []):
            lines.append(f"- {note}")

    block = "\n".join(lines)
    warnings = _ledger_warnings(ctx)
    if warnings:
        block += "\n" + warnings
    if rendered:
        block += "\n\n" + rendered
    return block


def _objection_blocks(ctx: Dict[str, Any]) -> List[str]:
    """Standing objections and falsifiers — analytical and audit profiles only."""
    blocks: List[str] = []
    findings = ctx.get("redteam_findings") or []
    if isinstance(findings, list) and findings:
        lines = ["## Standing Objections", ""]
        for item in findings[:5]:
            if not isinstance(item, dict):
                continue
            statement = str(item.get("statement", "") or "").strip()
            if not statement:
                continue
            test = str(item.get("test", "") or "").strip()
            lines.append(f"- {statement}" + (f" Resolve by: {test}" if test else ""))
        if len(lines) > 2:
            blocks.append("\n".join(lines))

    changers = ctx.get("what_would_change_our_mind") or []
    if isinstance(changers, list) and changers:
        lines = ["## What Would Change This Conclusion", ""]
        lines += [f"- {str(c).strip()}" for c in changers[:5] if str(c).strip()]
        if len(lines) > 2:
            blocks.append("\n".join(lines))
    return blocks


# ---------------------------------------------------------------------------
# Deterministic fallback: extractive report when the model call fails
# ---------------------------------------------------------------------------

def _deterministic_report(
    query: str,
    usable_facts: Sequence[Dict[str, Any]],
    top_facts: Sequence[Dict[str, Any]],
    ctx: Dict[str, Any],
    angles: Sequence[str],
    profile: ReportProfile = PROFILE_BRIEF,
) -> SynthesisResult:
    """An organized, honestly-labeled research digest — not imitation prose.

    Gluing source sentences into paragraphs reads exactly like chunks cut from
    different sources, so this path leans into what it is: a structured
    briefing. It now LEADS WITH THE ANSWER — the highest-confidence claim,
    stated plainly — rather than opening on a count of verified facts, which is
    metadata, not an answer, and was the first thing a user saw on exactly the
    runs where the pipeline was already degraded.
    """
    diverse = select_diverse(list(top_facts), k=40, max_similarity=0.35)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in diverse:
        if not str(item.get("claim", "")).strip():
            continue
        # Sense first: an ambiguous query's fallback report must keep the
        # meanings in separate sections, exactly like the LLM path.
        key = (
            str(item.get("sense", "") or "").strip()
            or str(item.get("sub_question", "") or "").strip()
        )
        bucket = groups.setdefault(key, [])
        if len(bucket) < 8:
            bucket.append(item)
        if len(groups) >= 6 and all(len(v) >= 3 for v in groups.values()):
            break
    if not groups:
        return SynthesisResult(
            answer=f"No reliable evidence was retrieved for {_normalize_query_concept(query)}.",
            used_fallback=True,
            profile=profile.name,
        )

    gap_stats = _gap_stats(ctx, list(usable_facts))
    n_sources = len(
        {str(i.get("source", "")) for items in groups.values() for i in items if i.get("source")}
    )

    used: List[Dict[str, Any]] = []
    seen_claims: Set[str] = set()

    def _take(item: Dict[str, Any]) -> None:
        claim = str(item.get("claim", ""))
        if claim not in seen_claims:
            seen_claims.add(claim)
            used.append(item)

    # Headline finding: the single highest-confidence claim, previewed in the
    # summary and excluded from its section (no duplicated claims anywhere).
    headline: Optional[Dict[str, Any]] = None
    for items in groups.values():
        if items:
            headline = items[0]
            break
    if headline is not None:
        _take(headline)

    sections: List[str] = []
    for key, items in list(groups.items())[:6]:
        bullets: List[str] = []
        for item in items:
            if str(item.get("claim", "")) in seen_claims:
                continue
            rendered = _with_citation(item).strip()
            if not rendered:
                continue
            _take(item)
            bullets.append(f"- {rendered}")
        if not bullets:
            continue
        title = _section_title(key) or "Findings"
        sections.append(f"## {title}\n\n" + "\n".join(bullets))

    thin_note = (
        "Evidence is thin — fewer than three verified facts support this report, "
        "so treat every finding below as provisional. "
        if gap_stats["verified"] < 3
        else ""
    )
    degraded_note = (
        f" Pipeline stages on deterministic fallback: {', '.join(gap_stats['degraded'])}."
        if gap_stats["degraded"]
        else ""
    )
    ambiguity_note = (
        f" The evidence spans {len(groups)} distinct angles of this query, each in "
        "its own section below."
        if len(groups) >= 3
        else ""
    )

    # Answer first, even here. The headline claim IS the short answer; the
    # accounting follows it instead of replacing it.
    headline_text = _with_citation(headline).strip() if headline is not None else ""
    summary_parts = ["## Executive Summary", ""]
    if headline_text:
        summary_parts.append(f"Short answer: {headline_text}")
        summary_parts.append("")
    summary_parts.append(
        f"{thin_note}This is an extractive digest of the verified evidence for "
        f"\u201c{_normalize_query_concept(query)}\u201d, assembled without a model "
        f"writing pass.{ambiguity_note}"
    )
    summary_parts.append("")
    summary_parts.append(
        f"Confidence: {_confidence_statement(ctx, gap_stats['verified'])} — "
        f"{gap_stats['verified']} verified facts across {n_sources} sources.{degraded_note}"
    )
    sections.insert(0, "\n".join(summary_parts))
    sections.append(_gaps_section(gap_stats))

    # The extractive path knows exactly which source each claim came from, so it
    # cites perfectly — claims were rendered with an identity token; now that
    # the used set is final, tokens become numbers.
    numbered, pairs = _assign_numbers(used[:40], max_sources=MAX_LEGEND_SOURCES_HARD)
    answer = "\n\n".join(sections) + "\n\n" + _legend_block(numbered)
    for fact, index in pairs:
        answer = answer.replace(_cite_token(fact), f"[{index}]")
    answer = re.sub(r"\s*\[\[c\d+\]\]", "", answer)
    answer = _reorder_sections(answer)
    cited = [dict(fact, citation=index) for fact, index in pairs]
    return SynthesisResult(
        answer=answer,
        sources=numbered,
        audit=audit_citations(answer, numbered, cited),
        used_fallback=True,
        angles=list(angles),
        profile=profile.name,
        word_count=_count_words(answer),
    )


def _cite_token(fact: Dict[str, Any]) -> str:
    """Placeholder standing in for a citation number not yet assigned."""
    return f"[[c{id(fact)}]]"


def _with_citation(fact: Dict[str, Any]) -> str:
    """A claim carrying its own citation placeholder.

    The marker goes INSIDE the sentence, before the terminal punctuation:
    "claim [1]." not "claim. [1]". Sentence splitters break after [.!?], so a
    marker placed after the period became its own orphan unit — the audit then
    scored the extractive fallback report as largely uncited even though every
    claim carried a citation.
    """
    claim = str(fact.get("claim", "") or "").strip()
    if not claim:
        return ""
    terminal = claim[-1] if claim[-1] in ".!?" else "."
    if claim[-1] in ".!?":
        claim = claim[:-1].rstrip()
    return f"{claim} {_cite_token(fact)}{terminal}"


# ---------------------------------------------------------------------------
# Retained helpers
# ---------------------------------------------------------------------------

def _gap_stats(ctx: Dict[str, Any], usable_facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts the Evidence & Confidence section needs.

    Prefers workflow context (contradictions, degraded stages, totals); derives
    the rest from the facts themselves.
    """
    total = _safe_int(ctx.get("total_facts", 0)) or len(usable_facts)
    if ctx.get("verified_count") is not None:
        verified = _safe_int(ctx.get("verified_count"))
    else:
        # Synthesis input passed verification upstream: anything explicitly
        # flagged False was already filtered, the rest counts as verified.
        verified = sum(1 for f in usable_facts if f.get("verified", True) is not False)
    contradictions = ctx.get("contradictions", []) or []
    if not isinstance(contradictions, list):
        contradictions = []
    degraded = ctx.get("degraded", []) or []
    if not isinstance(degraded, list):
        degraded = []
    return {
        "total": total,
        "verified": verified,
        "unverified_excluded": max(0, total - verified),
        "contradictions": contradictions[:5],
        "degraded": [str(d) for d in degraded if d],
    }


def _confidence_statement(ctx: Dict[str, Any], verified_count: int) -> str:
    """'High (0.76)' on the pipeline's 0-1 scale, or a volume-based level."""
    raw = ctx.get("confidence", None) if isinstance(ctx, dict) else None
    try:
        score = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "High" if verified_count >= 8 else ("Medium" if verified_count >= 3 else "Low")
    level = "High" if score >= 0.75 else ("Medium" if score >= 0.5 else "Low")
    # Thin evidence caps the stated level: fewer than three verified facts is
    # provisional by construction, so a high numeric score must not label the
    # extractive digest "High" while the same paragraph calls it provisional.
    if verified_count < 3 and level != "Low":
        level = "Low"
    return f"{level} ({score:.2f})"


def _gaps_section(stats: Dict[str, Any]) -> str:
    """Evidence & Confidence plus Limitations for the extractive path."""
    lines = [
        "## Evidence & Confidence",
        "",
        f"Well-supported: {stats['verified']} verified facts feed this report; "
        "every claim above traces to a cited source.",
    ]
    if stats["unverified_excluded"]:
        lines.append(
            f"Uncertain: {stats['unverified_excluded']} collected claims failed "
            "verification and were excluded from synthesis rather than repeated."
        )
    if stats["contradictions"]:
        lines.append(
            f"Conflicting evidence: {len(stats['contradictions'])} source conflict(s) "
            "flagged; conflicting numbers are reported as ranges, not picked."
        )
    if stats["degraded"]:
        lines.append(
            "Pipeline gaps: deterministic fallback covered "
            f"{', '.join(stats['degraded'])} — those sections are extractive, "
            "not model-written."
        )
    lines.extend([
        "",
        "## Limitations & Unknowns",
        "",
        "Could not verify: claims without a traceable source were discarded "
        "during synthesis and do not appear above.",
    ])
    return "\n".join(lines)


def _deterministic_disambiguation(intent: Dict[str, Any]) -> str:
    """The numbered 'n) **Sense** — explanation' block the gate requires."""
    if not isinstance(intent, dict) or not intent.get("ambiguity"):
        return ""
    senses = [
        s for s in (intent.get("senses") or [])
        if isinstance(s, dict) and str(s.get("label", "")).strip()
    ]
    if len(senses) < 2:
        return ""
    lines = []
    for i, sense in enumerate(senses[:3], 1):
        label = str(sense.get("label", "")).strip()
        note = str(sense.get("note", "") or "").strip()
        if not note:
            note = str(sense.get("domain", "") or "distinct meaning").strip()
        lines.append(f"{i}) **{label}** — {note}")
    return "\n".join(lines)


def _ensure_disambiguation(answer: str, ctx: Dict[str, Any]) -> str:
    """Guarantee the ambiguity contract is in the shipped body.

    If the writer already opened with numbered sense lines, leave it alone;
    otherwise prepend the deterministic block so an ambiguous query never
    silently picks one meaning. Deterministic, no LLM.
    """
    intent = ctx.get("intent") or {}
    if not isinstance(intent, dict) or not intent.get("ambiguity"):
        return answer
    if re.search(r"^\s*\d+\)\s*\*\*[^*]{2,120}\*\*", answer or "", re.M):
        return answer
    block = _deterministic_disambiguation(intent)
    if not block:
        return answer
    heading = re.search(r"^##\s+Executive Summary\s*$", answer or "", re.M)
    if heading:
        head = answer[: heading.end()]
        tail = answer[heading.end():].lstrip()
        return f"{head}\n\n{block}\n\n{tail}"
    return f"{block}\n\n{answer.lstrip()}"


def _render_ambiguity_block(intent: Dict[str, Any]) -> str:
    """Mandatory disambiguation contract for an ambiguous query."""
    if not isinstance(intent, dict) or not intent.get("ambiguity"):
        return ""
    senses = [
        s for s in (intent.get("senses") or [])
        if isinstance(s, dict) and str(s.get("label", "")).strip()
    ]
    if not senses:
        return ""
    listed = "\n".join(
        f"  {i + 1}) **{str(s.get('label')).strip()}**"
        + (f" — {str(s.get('note', '')).strip()}" if str(s.get("note", "")).strip() else "")
        for i, s in enumerate(senses[:3])
    )
    action = str(intent.get("recommended_action", "") or "")
    focus = str((senses[0] or {}).get("label", "")).strip()
    if action == "research_both" and len(senses) > 1:
        structure = (
            "Structure the report so each researched meaning gets its OWN sections "
            "(evidence lines carry [sense: ...] tags — a section about one sense "
            "cites only that sense's facts)."
        )
    else:
        structure = (
            f"The detailed sections focus on meaning 1 ('{focus}'). Do NOT spend "
            "sections or citations on the other meaning(s) — their one "
            "disambiguation line above is enough."
        )
    return (
        "AMBIGUOUS QUERY — the term has distinct meanings:\n"
        f"{listed}\n\n"
        "The Executive Summary MUST open with a numbered disambiguation in EXACTLY "
        "the shape above (one line per sense: number with a closing parenthesis, "
        "bold sense name, em dash, one-clause explanation), then one sentence: "
        f"'Based on your question, this report focuses on meaning 1.' {structure}"
    )


def _render_context_block(ctx: Dict[str, Any]) -> str:
    """Decision-grade inputs for the LLM brief: conflicts to flag and the
    honesty baseline for the evidence section. Empty when no context passed."""
    if not ctx:
        return ""
    parts: List[str] = []
    if ctx.get("revision"):
        parts.append(
            "REVISION PASS — rewrite your previous draft as the final answer. "
            "Lead with the direct answer to the query in plain language; one idea "
            "per paragraph; cut filler, repetition and tangents; remove any section "
            "that does not serve the query; keep every [n] marker valid and attached "
            "to the sentence it supports; preserve the disambiguation block when the "
            "query was ambiguous."
        )
    contradictions = ctx.get("contradictions", []) or []
    if isinstance(contradictions, list) and contradictions:
        lines = []
        for c in contradictions[:5]:
            if not isinstance(c, dict):
                continue
            lines.append(
                f"- \"{str(c.get('claim_a', ''))[:140]}\" ({c.get('source_a', '')}) "
                f"CONFLICTS WITH \"{str(c.get('claim_b', ''))[:140]}\" ({c.get('source_b', '')})"
            )
        if lines:
            parts.append(
                "Flagged source conflicts (surface these, never silently pick one):\n"
                + "\n".join(lines)
            )
    try:
        conf = float(ctx.get("confidence", "nan"))
        parts.append(f"Pipeline confidence: {conf:.2f} (0.75+ = sufficient).")
    except (TypeError, ValueError):
        pass
    degraded = ctx.get("degraded", []) or []
    if isinstance(degraded, list) and degraded:
        parts.append(f"Stages on deterministic fallback: {', '.join(str(d) for d in degraded)}.")
    total = _safe_int(ctx.get("total_facts", 0))
    verified = _safe_int(ctx.get("verified_count", 0))
    if total:
        parts.append(
            f"Evidence pool: {verified}/{total} facts verified; unverified claims were excluded."
        )
    parts.append(
        "The figures above are INTERNAL METADATA for your judgement only. "
        "Never quote them verbatim in the report body — the appendix states "
        "them, and the body describes evidence strength in words."
    )

    distribution = ctx.get("evidence_distribution")
    if isinstance(distribution, dict) and any(
        _safe_int(distribution.get(g, 0)) for g in ("A", "B", "C", "D")
    ):
        parts.append(
            "Evidence grades for this pool "
            f"(A {distribution.get('A', 0)}, B {distribution.get('B', 0)}, "
            f"C {distribution.get('C', 0)}, D {distribution.get('D', 0)}).\n"
            "Write with these epistemic tiers and label them explicitly:\n"
            "- ESTABLISHED: A-grade, independently corroborated — state plainly.\n"
            "- STRONG: B-grade — state with a source.\n"
            "- DISPUTED: contradicted claims — present as a disagreement, never pick a side silently.\n"
            "- INFERRED: reasonable synthesis from multiple claims — mark as inference, not fact.\n"
            "- UNKNOWN: no evidence — say so instead of guessing.\n"
            "Never present a D-grade claim or an unsupported number as established fact; "
            "soften it ('one source reports…') or omit it."
        )

    # Evidence-grounded reasoning structure: the deterministic argument the
    # evidence actually supports, rendered from the ReasoningMap built in
    # workflow.synthesizer_node (a no-op when the map is absent or empty).
    reasoning = ctx.get("reasoning")
    if reasoning is not None and hasattr(reasoning, "render_for_writer"):
        try:
            rendered_reasoning = str(reasoning.render_for_writer()).strip()
        except Exception:  # noqa: BLE001 - never let the structure break synthesis
            rendered_reasoning = ""
        if rendered_reasoning:
            parts.append(rendered_reasoning)

    # Surviving red-team objections belong in the brief, not only in the
    # appendix: a writer that knows the strongest counter-argument writes a
    # report that addresses it instead of one a reader can dismantle.
    findings = ctx.get("redteam_findings") or []
    if isinstance(findings, list) and findings:
        lines = []
        for item in findings[:4]:
            if not isinstance(item, dict):
                continue
            statement = str(item.get("statement", "") or "").strip()
            if statement:
                lines.append(f"- {statement}")
        if lines:
            parts.append(
                "Standing objections to this evidence (acknowledge, do not ignore):\n"
                + "\n".join(lines)
            )
    feedback = ctx.get("quality_feedback") or []
    if isinstance(feedback, list) and feedback:
        parts.append(
            "QUALITY GATE — your previous draft failed the answer-quality review. "
            "Fix every point below in the corrected report:\n"
            + "\n".join(f"- {str(f)}" for f in feedback[:8])
        )
    if not parts:
        return ""
    return "Research honesty baseline:\n" + "\n".join(parts) + "\n\n"


def _stratified_top_facts(
    facts: Sequence[Dict[str, Any]], per_angle: int = 6, cap: int = 30
) -> List[Dict[str, Any]]:
    """Top facts round-robin across sub-questions instead of pure confidence
    order. Pure ranking lets one low-scoring source domain bury a whole angle
    (observed live: the market angle cut from a top-20 while two definition
    angles filled it). Every angle keeps up to `per_angle` facts.
    """
    ranked = sorted(facts or [], key=_fact_confidence, reverse=True)
    by_angle: Dict[str, List[Dict[str, Any]]] = {}
    for fact in ranked:
        key = str(fact.get("sub_question", "") or "").strip()
        by_angle.setdefault(key, []).append(fact)
    picked: List[Dict[str, Any]] = []
    for i in range(max(1, per_angle)):
        for bucket in by_angle.values():
            if len(bucket) > i:
                picked.append(bucket[i])
                if len(picked) >= max(1, cap):
                    return picked
    return picked


def _fact_confidence(fact: Any) -> float:
    try:
        return float(fact.get("confidence", 0.0))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _compress_to_themes(
    facts: Sequence[Dict[str, Any]],
    *,
    similarity_threshold: float = 0.72,
) -> List[Dict[str, Any]]:
    """Collapse near-duplicate claims into one thematic entry per group.

    Compression here only merges claims that are already near-identical, keeps
    one representative, and records how many sources asserted it in
    `corroboration_count`. Distinct claims pass through untouched, in their
    original order — dropping them would weaken the verification guarantees.

    Deterministic and pure — no LLM, no network. Safe to run on every path.
    """
    kept: List[Dict[str, Any]] = []
    for fact in facts or []:
        if not isinstance(fact, dict):
            continue
        claim = str(fact.get("claim", "") or "").strip()
        if not claim:
            continue
        merged = False
        for existing in kept:
            if semantic_similarity(claim, str(existing.get("claim", ""))) < similarity_threshold:
                continue
            # Numeric guard: two claims that share wording but carry DIFFERENT
            # quantities are not the same assertion. Merging them silently
            # replaced specific evidence ("$11.5bn" vs "$12.7bn") with one
            # generic representative. Keep them separate.
            if _distinct_quantities(str(existing.get("claim", "")), claim):
                continue
            # Same assertion restated: keep the better-supported copy and count
            # the rest as corroboration rather than discarding them. The count
            # is computed BEFORE any swap and re-applied after — the previous
            # version incremented it and then wiped it with `clear()/update()`,
            # losing the corroboration it had just measured.
            total_corroboration = _corroboration(existing) + 1
            if _fact_confidence(fact) > _fact_confidence(existing):
                preserved_claim = existing["claim"]
                existing.clear()
                existing.update(fact)
                existing["claim"] = preserved_claim
            existing["corroboration_count"] = total_corroboration
            merged = True
            break
        if not merged:
            kept.append(dict(fact))
    return kept


def _quantity_signature(text: str) -> Set[Tuple[float, str]]:
    """(value, unit) pairs in a claim, unit-normalized for comparison."""
    return {
        (round(q.value, 4), str(q.unit or "").lower())
        for q in extract_numbers(text, limit=12)
    }


def _distinct_quantities(a: str, b: str) -> bool:
    """True when two claims carry non-overlapping quantity sets.

    Worded near-identically but quantifying differently (a different figure,
    year, or unit) means they are DIFFERENT evidence, not a restatement. Only
    claims with no numbers on either side, or with a shared quantity, are safe
    to merge.
    """
    sig_a = _quantity_signature(a)
    sig_b = _quantity_signature(b)
    if not sig_a or not sig_b:
        return False
    return not (sig_a & sig_b)


def _section_title(sub_question: str) -> str:
    """Humanize a grouping key for a `## ` header."""
    text = (sub_question or "").strip()
    if not text:
        return ""
    title = _shorten_heading(text)
    return title[0].upper() + title[1:] if title else ""


def _append_source_legend(answer: str, numbered: List[Dict[str, Any]]) -> str:
    return f"{answer.rstrip()}\n\n" + _legend_block(numbered)


def _legend_block(numbered: Sequence[Dict[str, Any]]) -> str:
    """`## Sources` section: one entry per line, blank-line separated, so the
    frontend renders a clean vertical numbered list under its own heading.

    Each entry carries its tier ("official publisher", "peer-reviewed",
    "media", ...) and a primary-source marker. A reader deciding how much to
    trust finding [4] should not have to recognise the domain to know whether it
    is a regulator's filing or a blog aggregating one.
    """
    lines: List[str] = []
    for source in numbered:
        entry = f"[{source['n']}] {source.get('domain', '')}"
        tier = str(source.get("tier", "") or "")
        if tier:
            entry += f" ({tier}{', primary' if source.get('primary') else ''})"
        if source.get("url"):
            entry += f" — {source['url']}"
        lines.append(entry)
    return "## Sources\n\n" + "\n\n".join(lines)


# Sentences that expose the pipeline's own internals rather than the subject.
# The model is told not to write these (system prompt rule 6), but live runs
# proved it copies the metadata block verbatim. The machine-appended appendix
# states these numbers correctly; the body must not narrate the research system.
#
# Patterns are anchored to SELF-REFERENTIAL phrasing. The previous version
# matched bare "evidence pool", "quality review", "pipeline stages" and
# "verified facts", which are ordinary English in a report ABOUT research,
# software or evaluation — and because sanitization joins each paragraph into a
# single line, one such match deleted the entire paragraph.
_PIPELINE_TELEMETRY_RE = re.compile(
    r"(?i)("
    r"\bpipeline (?:confidence|reports?|itself|stages?)\b|"
    r"\bthis (?:report|pipeline|system)'?s own (?:confidence|score)\b|"
    r"\brelevance \d{1,3}\s*/\s*\d{1,3}\b|"
    r"\bquality (?:score|gate|review) (?:of |is |was |returned )\b|"
    r"\bbelow the \d{1,3}\s*/\s*\d{1,3} floor\b|"
    r"\bbelow the \d\.\d+ threshold\b|"
    r"\bconfidence (?:score )?(?:is|was) \d\.\d+\b|"
    r"\b\d+ of \d+ facts? (?:were |was )?verified\b|"
    r"\b(?:the )?evidence pool (?:contains?|held?|has)\b|"
    r"\bfacts? in the (?:evidence )?pool\b|"
    r"\bself-?verif(?:ied|ication) (?:rate|pass)\b|"
    r"\bdeterministic fallback\b|\bdegraded run\b"
    r")"
)


def _scrub_pipeline_telemetry(text: str) -> str:
    """Drop sentences that narrate the pipeline's own metrics.

    Applied to the report body only, BEFORE the measured appendix is appended,
    so the correct figures in the appendix are never touched. Genuinely
    SENTENCE-granular: the previous version filtered line by line, and because
    `_sanitize_answer_text` joins a paragraph into one line, a single match
    deleted the whole paragraph. Headings and list structure are preserved; a
    paragraph that loses every sentence collapses away.
    """
    if not text:
        return text
    kept_paras: List[str] = []
    for para in text.split("\n\n"):
        stripped = para.strip()
        if not stripped or stripped.startswith("#"):
            kept_paras.append(para)
            continue
        kept_lines: List[str] = []
        for line in para.split("\n"):
            if not line.strip():
                continue
            if not _PIPELINE_TELEMETRY_RE.search(line):
                kept_lines.append(line)
                continue
            if line.lstrip().startswith(("- ", "* ")):
                # A bullet is one unit; drop it whole.
                continue
            sentences = split_into_sentences(line, max_sentences=40)
            surviving = [
                s.strip() for s in sentences
                if s.strip() and not _PIPELINE_TELEMETRY_RE.search(s)
            ]
            if surviving:
                kept_lines.append(" ".join(surviving))
        if kept_lines:
            kept_paras.append("\n".join(kept_lines))
    return "\n\n".join(kept_paras).strip()


def _normalize_query_concept(query: str) -> str:
    text = re.sub(r"\s+", " ", (query or "").strip()).strip(" ?.!")
    lower = text.lower()
    if lower.startswith("what is "):
        text = text[8:].strip()
    elif lower.startswith("what are "):
        text = text[9:].strip()
    elif lower.startswith("define "):
        text = text[7:].strip()
    if text:
        return text[0].upper() + text[1:]
    return "This topic"


# Inline bullet separators: models sometimes emit bullets on one line
# ("... induction [4]. - Electrical transformers modify ..."). A " - " right
# after a sentence end / citation bracket is a list separator, not prose;
# splitting on it restores real bullet lines. Only applied inside lines that
# already start with "- ".
_BULLET_SEP_RE = re.compile(r"(?<=[.!?\]])\s+-\s+(?=[A-Z0-9\"'(])")


def _sanitize_answer_text(answer: str, query: str) -> str:
    """Normalize the writer's markdown into the report's block structure.

    Preserves visual hierarchy: markdown headings become their own blocks,
    consecutive "- " lines stay distinct list items (joined into one bullet
    block), and paragraph breaks are kept. Only intra-line whitespace collapses;
    consecutive prose lines still merge, so the model's line-wrapping does not
    become line breaks.
    """
    raw_lines = (answer or "").replace("\r\n", "\n").split("\n")
    paras: List[str] = []
    current: List[str] = []
    bullets: List[str] = []

    def _flush_bullets() -> None:
        nonlocal bullets
        if bullets:
            paras.append("\n".join(bullets))
            bullets = []

    def _flush_prose() -> None:
        nonlocal current
        if current:
            paras.append(" ".join(current))
            current = []

    for raw_line in raw_lines:
        line = re.sub(r"\s+", " ", raw_line).strip()
        is_heading = bool(re.match(r"^#{1,6}\s+\S", line))
        if line.startswith("- "):
            _flush_prose()
            for part in _BULLET_SEP_RE.split(line):
                part = part.strip()
                if part:
                    bullets.append(part if part.startswith("- ") else f"- {part}")
        elif is_heading:
            _flush_bullets()
            _flush_prose()
            paras.append(line)
        elif line:
            _flush_bullets()
            current.append(line)
        else:
            _flush_bullets()
            _flush_prose()
    _flush_bullets()
    _flush_prose()
    text = "\n\n".join(paras).strip()

    # Fix malformed opening pattern like: "what is X refers to ..."
    q = (query or "").strip().rstrip("?")
    if q:
        text = re.sub(
            rf"(?i)^{re.escape(q)}\s+refers to",
            f"{_normalize_query_concept(query)} refers to",
            text,
        )

    # Collapse immediate repeated clause: "X is ... X is ..."
    text = re.sub(r"(?i)(\b[A-Z][A-Za-z\s\-]{2,40}\s+is\b[^.]*\.)\s+\1", r"\1", text)

    # Question-shaped headings are not labels: shorten them so a raw planner
    # question or writer-emitted question never becomes a section heading.
    text = _dedupe_heading(text)

    return text.strip()
