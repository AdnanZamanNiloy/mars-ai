"""Synthesis Agent — turns a verified evidence pool into a cited report.

What changed and why
--------------------
1. THE MODEL NO LONGER HAS TO GUESS CITATION NUMBERS. The old prompt sent
   `top_facts` (up to 40 raw dicts) and, separately, a legend of at most 12
   numbered sources. Nothing connected a fact to its number, so the model had to
   infer "which [n] is this claim's source" from a domain string — and facts
   13..40 had sources that were never numbered at all. Every claim now arrives
   with its own `[n]` attached, and only facts whose source is in the legend are
   sent. This is the single largest cause of misattributed citations.

2. CITATION VALIDATION IS AN AUDIT, NOT A REGEX. `_validate_citations` deleted
   out-of-range markers and stopped there, which silently turned a wrongly-cited
   sentence into an UNCITED sentence — a worse outcome, and invisible. The new
   `audit_citations` reports out-of-range markers, uncited factual sentences,
   numbers that appear nowhere in the evidence, and per-sentence support scored
   against the actual cited source. Findings are surfaced in the report's
   Evidence section and returned to the caller instead of being swallowed.

3. HALLUCINATED FIGURES ARE CAUGHT. A number in the draft that does not appear
   in any evidence fact is the highest-consequence failure mode of a research
   report. `_ungrounded_numbers` uses the same numeric extraction the verifier
   uses, and the audit flags them by value.

4. CONFLICTS ARE REPORTED AS RANGES. The prompt told the model not to average
   conflicting numbers but gave it no range to use. Detected numeric
   contradictions are now pre-rendered as explicit ranges via
   `contradiction.numeric_ranges`, so "between X and Y (sources [a],[b])" is
   available as text rather than something the model must construct.

5. THE REPORT CARRIES ITS OWN EVIDENCE ACCOUNTING. Confidence panel, primary
   source share, red-team findings that survived, and "what would change our
   mind" are appended deterministically from measured state, so those sections
   cannot be hallucinated or omitted.

Backward compatibility: `synthesizer_agent(llm, query, facts, context=None)`
still returns the report as a plain string. `synthesize()` returns the same
report plus the audit and legend for callers that want the structured form.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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

logger = get_logger(__name__)


# Reasoning-depth contract injected into every section-writing prompt. The
# live deep run produced a report that recited WHAT was true per dimension and
# scored reasoning=65 with the gate note "the report states no limitations" —
# it named facts but never explained why they hold or what follows from them.
# GPT Researcher's sections read as analysis because each one argues a
# mechanism and weighs a trade-off; this instruction makes that an explicit
# requirement, while forbidding the two failure modes that would satisfy it
# dishonestly (invented causes, and hedging that says nothing).
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

# The same contract, appended to the monolithic writer's prompt so the
# single-pass path (non-broad queries, section-wise fallback) is not weaker
# than the section-wise path.
_REASONING_DEPTH_BLOCK = "\n" + _REASONING_DEPTH_INSTRUCTION + "\n\n"


SYNTHESIZER_SYSTEM_PROMPT = """
You are the MARS Synthesis Engine — the Final Synthesis Agent. Your only job
is to turn verified research into a clean, premium, highly readable
intelligence report. You do not dump search results; you present knowledge.

━━━ ABSOLUTE FORMATTING RULES ━━━

1. Clear visual hierarchy: use the exact section structure below, with
   markdown `## ` headings and a blank line before every new section.
2. Short paragraphs — 3 to 4 lines maximum.
3. Prefer bullet points over long dense paragraphs.
4. Never dump everything into one continuous block of text.
5. Calm, precise, professional tone. Write like a premium research brief,
   not raw notes. Be direct and scannable. Remove repetitive or low-value
   sentences.
6. Never write "the research found", "the agents discovered", or "according
   to the research". Present the knowledge directly.
7. NEVER expose the pipeline's own internal metrics in the report prose. Do
   not write the confidence score, the relevance/quality score, the count of
   verified facts, the number of facts in the pool, "below the threshold",
   "relevance N/100", "pipeline confidence", or any number describing the
   research system rather than the subject. Those figures belong only in the
   machine-appended Evidence & Confidence appendix. Describe the strength of
   the EVIDENCE in words ("well-established", "single-source") and let the
   appendix carry the numbers.
8. LENGTH: default under 900 words. When the input carries an explicit
   length hint (deep-research modes), follow the hint instead. Slow
   providers cannot serve unbounded generation, and a timed-out synthesis
   degrades to extraction.

━━━ CITATION RULES (non-negotiable) ━━━

Every evidence item you are given is prefixed with its citation number, like
`[3] claim text ...`. Use THAT number when you use THAT claim. Do not
renumber, do not guess, do not cite a number you were not given.

  - Every sentence that states a fact, name, date, or number carries at
    least one [n] marker.
  - A sentence combining two claims cites both: "... [2][5]".
  - NEVER write a number, percentage, currency amount, or date that does not
    appear verbatim in the evidence you were given. If the evidence has no
    number for something, say so in words instead of estimating.
  - Analysis sentences that draw a conclusion FROM cited facts need no marker
    of their own, but must not introduce new facts.

━━━ REQUIRED STRUCTURE (exact order) ━━━

1. `## Executive Summary`
   4-6 sentences maximum. If the query term has multiple distinct meanings,
   clarify that in the FIRST sentence and keep the meanings strictly
   separate throughout the report. End with the overall confidence level
   and numeric score.

2. `## Key Findings`
   Bullet points only. Each bullet is ONE clear, self-contained fact with
   its [n] citation. Maximum 8 bullets, highest-confidence facts first.

3. Deep-dive sections (only where they add value):
   - If the query term has multiple distinct senses (e.g. "transformer" in
     machine learning vs electrical engineering), give each sense its own
     `## <Sense> Meaning` section, clearly separated, using short
     paragraphs and bullets. Inside a sense you may use `### ` sub-headings
     (Origin, Core Innovation, Architecture, Scale of Modern Models — or
     whatever fits the subject).
   - Otherwise, add 2-4 `## ` sections, one per research angle from the
     provided angle list (title = the angle, never the raw question
     restated). Skip straight to Evidence & Confidence when no angle adds
     anything.
    - SYNTHESIZE, DON'T PARAPHRASE: each section must weave together 2+ of
      the provided sources into a cause/effect or comparison narrative.
      Single-source recitation is what makes a report read like raw notes.
    - EXPLAIN, DON'T JUST REPORT: wherever the evidence supports it, state
      the mechanism or cause behind a finding ("because", "driven by"),
      present competing explanations or trade-offs side by side, and name
      what remains unresolved. A report that only lists WHAT is true has not
      answered an analytical query. Never invent a cause the evidence does
      not state.

   - PROSE QUALITY — this is an intelligence brief, not a note dump:
     open every section with a 1-3 sentence synthesis paragraph in your own
     words (weaving that section's cited facts), then bullets for genuinely
     enumerable findings only. Every sentence must be complete and
     self-contained — never a fragment that reads like it was cut from a
     source. Connect related findings with comparison or causation
     ("compared with", "because", "as a result") instead of listing them
     side by side.
   - DISAGREEMENT IS DATA: where sources conflict, present both positions
     side by side with their numbers and sources — never average them and
     never silently pick one. Pre-computed ranges are provided; use them.

4. `## Key Figures` — ONLY when the evidence contains quantitative claims:
   bullets with the number, unit, period, and scope attached, each cited
   [n]. Omit the section when the evidence has no solid numbers.

5. `## Evidence & Confidence`
   Overall confidence score, what is well supported, what could not be
   verified, and the major gaps. Mandatory even at high confidence.

━━━ VERIFY BEFORE YOU WRITE A SINGLE WORD ━━━

- Every claim, name, date, or number must trace to at least one provided
  source, cited at the point of use with its given [n].
- If two sources conflict, flag the conflict — never silently pick one.
- Discard extraction artifacts (garbled text, fragments with no clear
  subject, unrelated names). Absence of a clean answer is a valid,
  reportable finding — state it plainly in the Executive Summary.
- Never mix unrelated people, organizations, or senses of a term. If the
  evidence points to multiple distinct entities with similar names,
  separate them explicitly or state that the identity is ambiguous.
- Prefer the primary source when a primary document and a news summary of
  it both appear; cite the primary and use the summary only for framing.

HARD FAILURE CONDITIONS — reject your own draft if any are true:
- A section that is one wall of text with no bullets or breathing room
- Any factual sentence with no [n] marker
- Any number that is not in the provided evidence
- Any name, number, or fact that is not clearly corroborated
- Unrelated senses or entities blended without explicit separation

Return valid JSON only in this schema:
{"answer": "<final synthesized report with [n] citations>"}
""".strip()


# A sentence stating a fact but carrying no [n] is a traceability hole. These
# openers mark analysis/transition sentences, which legitimately carry none.
# The second block covers the report's own scaffolding — the honesty notes the
# extractive path emits ("Evidence is thin", "Confidence:", "Well-supported:")
# are meta-statements about the report, not factual claims about the world.
# Counting them as untraceable facts deflated citation density exactly when
# the pipeline was degraded and using the fallback writer, and made the audit
# flag the pipeline's own confidence score as an "ungrounded number".
# NOTE: no trailing \b — alternatives ending in ":" can never satisfy one.
_ANALYSIS_LEAD_RE = re.compile(
    r"^\s*(?:taken together|in short|overall|therefore|this means|the picture|"
    r"in practice|by contrast|as a result|on balance|the implication|"
    r"what follows|in other words|put differently|the net effect|"
    r"evidence is thin|confidence:|well-supported:|uncertain:|"
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
# Numbered with a closing paren (not "1.") so sentence splitters keep the
# line intact. These lines are definitional common knowledge (the
# non-researched sense has no evidence by design), so the citation audit
# exempts them rather than flagging the pipeline's own disambiguation as
# untraceable.
_DISAMBIG_LINE_RE = re.compile(r"^\s*\d+\)\s*\*\*[^*]{2,120}\*\*\s*[—-]")


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "audit": self.audit.to_dict(),
            "used_fallback": self.used_fallback,
            "angles": list(self.angles),
            "synthesis_intelligence": dict(self.synthesis_intelligence),
        }


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
    findings, and evidence counts for the Evidence & Confidence section.
    Extra keyword arguments (outline, section_wise, compress_context) pass
    through to `synthesize` — callers that pass none keep the old behaviour.
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
) -> SynthesisResult:
    """The same synthesis, returning the audit and legend alongside the text.

    Answer-first outline (new): the report's section shape is derived from
    the query + evidence BEFORE writing, so a broad question is answered
    dimension by dimension instead of dumping the top-ranked claims.
    `section_wise` writes each outline section as its own call and assembles
    the result; it degrades cleanly to a single pass when unsupported.
    `compress_context` merges near-duplicate claims into one thematic entry
    (never dropping distinct claims) before the writer sees them.
    """
    usable_facts = dedupe_semantic_facts(filter_facts_by_domain(facts))
    if not usable_facts:
        return SynthesisResult(
            answer=(
                f"{query} is an area that requires reliable evidence to explain accurately. "
                "Current retrieved evidence was too limited or low quality to produce a "
                "robust synthesis."
            ),
            used_fallback=True,
        )

    ctx = context or {}
    contradictions = [c for c in (ctx.get("contradictions") or []) if isinstance(c, dict)]

    # Options may arrive as explicit kwargs (tests, direct callers) or inside
    # `context` (the workflow passes them there to keep this entry point's
    # signature stable for test doubles). Explicit wins; context is default.
    if compress_context is None:
        compress_context = bool(ctx.get("compress_context", True))
    if compress_threshold is None:
        compress_threshold = float(ctx.get("compress_threshold", 0.72) or 0.72)
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

    # Mode-aware depth: deep/executive runs are allowed a longer, more
    # analytical report — that depth is the product. Quick/standard stay tight
    # for latency, because providers degrade to extraction on unbounded
    # generation. The upper bound comes from the SAME band the quality gate
    # enforces (answer_quality.length_band), so the writer is never told to
    # exceed what the gate will fail.
    mode = str(ctx.get("mode", "standard") or "standard")
    band_lo, band_hi = length_band(mode)
    if mode in ("deep", "executive"):
        length_hint = (
            f"LENGTH: this is a deep-research brief — aim for {band_lo}-{band_hi} "
            "words in TOTAL. Go deeper per angle: mechanisms, numbers with "
            "context, and explicit treatment of conflicting evidence."
        )
    else:
        length_hint = f"LENGTH: keep the report under {band_hi} words."

    intent = ctx.get("intent") or {}
    level = str(intent.get("explanation_level", "") or "")
    if not mode.startswith(("deep", "executive")) and level == "basic":
        length_hint = (
            "LENGTH: keep the report under 600 words. The user asked for a basic "
            "explanation: open with a plain-language explanation of the concept and "
            "include ONE simple analogy that a non-expert would immediately grasp. "
            "Prefer explanation over statistics."
        )

    ambiguity_block = _render_ambiguity_block(intent)
    if ambiguity_block:
        length_hint = f"{length_hint}\n\n{ambiguity_block}"

    # Section-wise synthesis for broad questions: write each outline section
    # as its own bounded call and assemble. This is what stops a broad query
    # from collapsing into one narrow thesis, and it also keeps every prompt
    # small enough to survive size-capped providers. It is opt-in via the
    # caller (the workflow gates it on mode); when any section fails, the
    # whole report falls back to the single-pass writer — never a mixed
    # report and never a crash.
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
        )
        if sectioned is not None:
            return sectioned
        logger.warning("[Synthesizer] section-wise path failed; falling back to single pass")

    # Adaptive fact-cap ladder: the writer prompt carries up to 40 facts plus
    # the legend; on providers that cap request size (Groq 413s ~21-41KB) the
    # first attempt can be rejected whole. Shrinking the evidence view keeps
    # synthesis LLM-written instead of degrading to the extractive fallback.
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
        angles = []
        for fact in cited_facts:
            sub_question = str(fact.get("sub_question", "") or "").strip()
            if sub_question and sub_question not in angles:
                angles.append(sub_question)

        user_prompt = (
            f"Main query: {query}\n\n"
            f"{length_hint}\n\n"
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
                # Slowness, not size or rate. One budget-aware second chance
                # at the SMALLEST evidence view: a flaky provider may still
                # complete a small prompt inside the run's remaining
                # wall-clock. Never more than one.
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
                logger.warning("[Synthesizer] provider too slow (timeout); using deterministic fallback")
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
        return _deterministic_report(query, usable_facts, top_facts, ctx, angles)

    answer = _sanitize_answer_text(answer, query)
    answer = _scrub_pipeline_telemetry(answer)
    answer = _ensure_disambiguation(answer, ctx)

    # Mandatory sections, enforced post-assembly: the writer cannot omit
    # Limitations/unknowns or Counterarguments (or any of the other three) and
    # ship a report that hides them. Missing sections are added deterministically
    # from measured state, before the appendix/legend.
    #
    # ORDER MATTERS: mandatory sections are added BEFORE the deterministic
    # length trim. The old order trimmed the writer's draft to the band and
    # THEN appended ~1000 words of required-section/appendix boilerplate, so
    # every deep report overshot the band by 50-80% (live: 2609 words against
    # a 1500 cap). Trimming last makes the band an enforced contract instead of
    # a pre-appendix suggestion.
    answer = ensure_required_sections(
        answer,
        ctx=ctx,
        usable_facts=usable_facts,
        contradictions=contradictions,
        cited_facts=cited_facts,
    )
    # The argument structure is deterministic, so its loss must not depend on
    # the writer surfacing it: append it when the report has no Reasoning/
    # Argument/Conclusions section. Runs before the trim so the band holds.
    answer = ensure_reasoning_structure(answer, ctx=ctx)
    # Cross-section refinement pass (deterministic; no LLM). Runs AFTER the
    # required sections exist (so Key Findings is registered as a recap) and
    # BEFORE the trim. A bare restatement is transformed into a transition +
    # analysis sentence (never deleted) and keeps its exact [n] markers.
    answer, _si_report = apply_synthesis_intelligence_pass(answer, ctx, query=query)
    answer = _trim_to_band(answer, mode)
    audit = audit_citations(answer, numbered, cited_facts)

    # Out-of-range markers are removed (they resolve to nothing), but unlike the
    # old silent strip we record them and say so in the report.
    if audit.invalid_markers:
        answer = _drop_invalid_markers(answer, len(numbered))

    integrity = _integrity_note(audit)
    if integrity:
        answer = f"{answer.rstrip()}\n\n{integrity}"

    answer = _append_evidence_appendix(answer, ctx, usable_facts, contradictions)
    answer = _append_source_legend(answer, numbered)
    return SynthesisResult(
        answer=answer,
        sources=numbered,
        audit=audit,
        angles=angles,
        synthesis_intelligence=_si_report.to_dict(),
    )


def _normalize_heading(text: str) -> str:
    """Comparison key for headings: casefolded, punctuation/space-insensitive."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def apply_synthesis_intelligence_pass(
    answer: str,
    ctx: Dict[str, Any] | None = None,
    *,
    query: str = "",
) -> tuple[str, SynthesisIntelligenceReport]:
    """Run the deterministic cross-section refinement layer on an assembled draft.

    Protected (machine-appended) sections are left whole — they describe
    measured state. Executive Summary and Key Findings are RECAP sections:
    their own text is preserved, but their claims are registered so deep-dive
    sections cannot restate them. Every other writer section is refined
    against the ledger: the first occurrence keeps its citation, an expansion
    with a new analytical dimension is kept intact, and a bare restatement is
    REWRITTEN into a contextual transition + analysis sentence (never deleted),
    so a section opening can never dangle. Evidence signals from `ctx`
    (contradictions, corroboration, primary share) select the analytical
    dimension; the transformation runs with no LLM and never empties a section.
    """
    from app.agents.sources import MACHINE_SECTIONS

    protected = tuple(MACHINE_SECTIONS) + ("## Limitations", "## Evidence & Confidence")
    recap = ("## Executive Summary", "## Key Findings")
    ctx = ctx or {}
    intent = ctx.get("intent") or {}
    signals = {
        "contradicted": bool(ctx.get("contradictions")),
        "corroborated": bool(ctx.get("corroborated")),
        "primary": bool(ctx.get("primary_share")),
        # Adaptive move-selection signals (additive; the layer works without
        # them). The classified query + its type let a "should … invest"
        # question resolve to a strategic move, a "what caused …" question to
        # a causal one, and a "compare …" question to a comparison — so the
        # refinement answers the question that was asked, not a generic one.
        "query": str(query or ""),
        "query_type": str(intent.get("query_type", "") or ctx.get("query_type", "") or ""),
        "domain": str(intent.get("domain", "") or ""),
    }
    return apply_synthesis_intelligence(
        answer,
        protect_headings=protected,
        recap_headings=recap,
        signals=signals,
    )


def _shorten_heading(text: str) -> str:
    """Reduce a question-shaped heading to a concise label.

    The section writer is told the question to answer; it sometimes emits that
    question as its own `## ` heading ("## How did the FDIC's systemic risk
    exception ... extend deposit protection?"). Live deep reports shipped 3
    such headings on one query, each duplicating the query and padding the
    report. Deterministic, no LLM; a heading already label-shaped is returned
    unchanged.
    """
    cleaned = re.sub(r"\s+", " ", (text or "").strip()).strip(" ?.!,")
    if not cleaned:
        return text
    words = cleaned.split(" ")
    limit = 60
    if "?" not in text and len(words) <= 10 and len(cleaned) <= limit:
        return cleaned
    clause = re.split(r"[?:—–]|\s+-\s+", cleaned, maxsplit=1)[0].strip(" ?.!,")
    if clause.lower().startswith("what are "):
        clause = clause[9:]
    elif clause.lower().startswith("what is "):
        clause = clause[8:]
    elif clause.lower().startswith("what was "):
        clause = clause[9:]
    elif clause.lower().startswith("how did "):
        clause = clause[8:]
    elif clause.lower().startswith("how does "):
        clause = clause[9:]
    elif clause.lower().startswith("how "):
        clause = clause[4:]
    elif clause.lower().startswith("why "):
        clause = clause[4:]
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
        if title.endswith("?") or len(title.split()) > 12:
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
    first = lines[i].strip()
    match = re.match(r"^#{1,6}\s+(.*\S)\s*$", first)
    if not match:
        return body
    if _normalize_heading(match.group(1)) != _normalize_heading(title):
        return body
    remaining = lines[i + 1:]
    # Drop a single blank line that separated the writer's heading from prose.
    if remaining and not remaining[0].strip():
        remaining = remaining[1:]
    return "\n".join(remaining).strip() if remaining else ""


def _count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))


def _trim_to_band(body: str, mode: str) -> str:
    """Deterministically bring an assembled draft inside the mode's word band.

    The section-wise assembler asks each writer for a per-section budget, but
    the model overshoots it (observed: 1715 words against a 1500 deep cap), and
    the revision pass re-runs the same section writers and overshoots again, so
    the length contract was never actually enforced. Evidence coverage must not
    be lost, so this never drops a section or its Executive Summary: it removes
    trailing paragraphs from the LONGEST sections first (after the first
    paragraph, which carries the section's synthesis), stopping as soon as the
    draft fits. Sentence-boundary and paragraph-granular; no LLM.
    """
    band_lo, band_hi = length_band(mode)
    if _count_words(body) <= band_hi:
        return body

    # Split into blocks on blank lines, remembering heading boundaries.
    blocks: List[str] = [b for b in (body or "").split("\n\n")]
    # Index blocks by the section they belong to. A heading starts a section;
    # prose/bullets after it belong to it.
    sections: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    preamble: List[str] = []
    for block in blocks:
        stripped = block.strip()
        if stripped.startswith("#"):
            current = {"heading": block, "paras": []}
            sections.append(current)
        elif current is None:
            preamble.append(block)
        else:
            current["paras"].append(block)

    def _render() -> str:
        out = list(preamble)
        for section in sections:
            out.append(section["heading"])
            out.extend(section["paras"])
        return "\n\n".join(b for b in out if b != "").strip()

    # Protect the Executive Summary: it is the answer's headline, never trimmed.
    protected = {"executive summary"}
    while _count_words(_render()) > band_hi:
        # Pick the longest trimmable section with >1 paragraph.
        candidates = [
            s for s in sections
            if len(s["paras"]) > 1
            and _normalize_heading(str(s["heading"]).lstrip("# ").strip()) not in protected
        ]
        if not candidates:
            break
        longest = max(candidates, key=lambda s: _count_words("\n\n".join(s["paras"])))
        # Drop the LAST paragraph (the least structural) of the longest section,
        # keeping the first, which carries its synthesis claim.
        longest["paras"].pop()
    return _render()


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
        selected = select_section_facts(
            section, candidates, reserved_ids=own_ids
        )
        # If selection somehow dropped every axis-grouped fact, fall back to
        # the raw group so the section keeps its own evidence.
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
) -> SynthesisResult | None:
    """Write each outline section as its own call, then assemble.

    Returns None (so the caller re-runs the single-pass writer) when there
    are fewer than two usable sections, when any section call fails, or when
    the assembled draft is empty. Deterministic guarantees (citation audit,
    evidence appendix, source legend) are applied to the assembled text
    exactly as in the single-pass path.
    """
    # Per-section context selection (GPT Researcher adaptation): the section's
    # own axis-grouped facts are ALWAYS candidates, and the rest of the global
    # pool is added so a section can draw a relevant claim the coarse axis
    # grouping put elsewhere. The selector then ranks the union by combined
    # relevance + evidence quality + corroboration + authority with a bounded
    # diversity bonus and a high-impact floor, instead of handing the writer
    # the axis pool in acquisition order. On any failure it returns the
    # axis-grouped pool (never an empty section — AGENTS.md 4.7).
    groups = _ranked_section_groups(outline, usable_facts)
    if len(groups) < 2:
        return None

    # One shared legend across sections so markers stay stable and in range.
    # Number the ORIGINAL fact objects (a single legend + a per-fact marker),
    # then split them back per section. `_assign_numbers` preserves identity,
    # unlike `_number_facts` which copies.
    all_section_facts: List[Dict[str, Any]] = []
    for _, facts in groups:
        all_section_facts.extend(facts)
    numbered, pairs = _assign_numbers(all_section_facts)
    cited_facts: List[Dict[str, Any]] = []
    facts_by_section: List[List[Dict[str, Any]]] = []
    marker_by_id: Dict[int, int] = {}
    for fact, index in pairs:
        item = dict(fact)
        item["citation"] = index
        cited_facts.append(item)
        marker_by_id[id(fact)] = index
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
    # band. The old prompt gave each section "2-4 tight paragraphs" with only a
    # whole-report hint, and the sections summed to 3809 words against a 1500
    # cap. Reserve words for the Executive Summary and the machine-appended
    # ledger/integrity sections, then divide the rest. The required-axes
    # contract added a 4th-5th section, so the reserve and the floor account for
    # a real section count instead of assuming three.
    mode = str(ctx.get("mode", "standard") or "standard")
    band_lo, band_hi = length_band(mode)
    heading_reserve = 260  # Executive Summary + Source ledger + Evidence integrity
    per_section_hi = max(110, (band_hi - heading_reserve) // max(1, len(groups)))
    section_length_hint = (
        f"LENGTH: {per_section_hi} words MAXIMUM for this section — a hard "
        f"limit, not a target. The report assembles to at most {band_hi} words "
        f"total across {len(groups)} sections plus the Executive Summary, so "
        "exceeding this makes the report fail its own length contract. Prefer "
        "one tight synthesis paragraph over two loose ones."
    )

    # Executive Summary first. The single-pass prompt mandates this heading and
    # the quality gate hard-fails clarity without it, but the section-wise path
    # previously assembled only outline sections — every sectioned report
    # shipped without its Executive Summary. One dedicated writer call over the
    # highest-confidence facts; when it fails we drop the heading rather than
    # ship an empty section (the gate's clarity note is honest, a blank
    # "## Executive Summary" is not).
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
            "Write ONLY the Executive Summary of a larger report. 4-6 sentences "
            "maximum, in your own words, answering the main query directly. If the "
            "query term has multiple distinct meanings, name them in the first "
            "sentence and keep them strictly separate. End with the overall "
            "confidence level and numeric score from the honesty baseline below. "
            "Do NOT emit a markdown heading — the assembler adds it. Cite with "
            "these exact [n] markers.\n\n"
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
        # Reuse the global numbering: the evidence block renders each fact's
        # own citation marker, so section prompts cannot renumber the legend.
        prompt = (
            f"Main query: {query}\n\n"
            f"{section_length_hint}\n\n"
            f"You are writing ONE section of a larger report, the section titled "
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
            logger.warning("[Synthesizer] section '%s' empty; abandoning section-wise path", section.title)
            return None
        # The assembler adds the section heading. Models routinely lead the
        # body with their own `## <title>` anyway, which duplicated every H2
        # in the shipped report ("## What It Is\n\n## What It Is"). Strip a
        # single leading ATX heading (H1-H6) when it matches the section
        # title; a differing heading is kept so real sub-structure survives.
        body = _strip_duplicate_section_heading(body, section.title)
        written_sections += 1
        section_bodies.append(f"## {section.title}\n\n{body}")

    if written_sections < 2:
        return None

    assembled = "\n\n".join(section_bodies)
    answer = _sanitize_answer_text(assembled, query)
    answer = _scrub_pipeline_telemetry(answer)
    answer = _ensure_disambiguation(answer, ctx)
    # Mandatory sections and the deterministic appendix do not enforce the
    # length band by themselves, so trim runs AFTER they are present (the old
    # order trimmed first and then appended boilerplate, overshooting the band
    # on every deep report).
    answer = ensure_required_sections(
        answer,
        ctx=ctx,
        usable_facts=usable_facts,
        contradictions=contradictions,
        cited_facts=cited_facts,
    )
    # Deterministic argument structure fallback (the writer's absence must not
    # lose it), before the trim so the band holds.
    answer = ensure_reasoning_structure(answer, ctx=ctx)
    # Section-wise reports are exactly where cross-section restatement lives
    # (each section was written blind to its siblings): run the deterministic
    # refinement pass before the trim — a repeated opening becomes a transition,
    # never a deleted sentence, and every [n] marker is preserved.
    answer, _si_report = apply_synthesis_intelligence_pass(answer, ctx, query=query)
    answer = _trim_to_band(answer, mode)
    audit = audit_citations(answer, numbered, cited_facts)
    if audit.invalid_markers:
        answer = _drop_invalid_markers(answer, len(numbered))
    integrity = _integrity_note(audit)
    if integrity:
        answer = f"{answer.rstrip()}\n\n{integrity}"
    answer = _append_evidence_appendix(answer, ctx, usable_facts, contradictions)
    answer = _append_source_legend(answer, numbered)
    angles = [
        str(f.get("sub_question", "") or "").strip()
        for f in cited_facts
        if str(f.get("sub_question", "") or "").strip()
    ]
    # Preserve order, drop duplicates.
    angles = list(dict.fromkeys(angles))
    logger.info("[Synthesizer] section-wise report: %d sections assembled", len(section_bodies))
    return SynthesisResult(
        answer=answer,
        sources=numbered,
        audit=audit,
        angles=angles,
        synthesis_intelligence=_si_report.to_dict(),
    )


# ---------------------------------------------------------------------------
# Mandatory report sections: enforce post-assembly, not just in the prompt
# ---------------------------------------------------------------------------

# Every final report MUST carry these five sections, whatever the writer chose
# to emit and whichever path wrote it. Keys are the canonical headings this
# assembler may ADD; values are the recognized aliases (normalized) that count
# as already present, so a well-written report is never duplicated.
REQUIRED_SECTIONS: Dict[str, tuple] = {
    "Executive Summary": ("executive summary", "summary", "overview"),
    "Key Findings": ("key findings", "main findings", "findings", "takeaways"),
    "Evidence Strength": (
        "evidence strength", "evidence and confidence", "evidence & confidence",
        "evidence quality", "source ledger",
    ),
    "Limitations & Unknowns": (
        "limitations and unknowns", "limitations & unknowns", "limitations",
        "unknowns", "gaps and limitations", "evidence limitations",
    ),
    "Counterarguments & Disputed Points": (
        "counterarguments and disputed points", "counterarguments & disputed points",
        "counterarguments", "disputed points", "standing objections",
        "conflicting evidence", "contradictions",
    ),
    # Uncertainty-first additions. The open-questions section is generated from
    # the critic's uncovered angles; the ledger makes every claim re-verifiable.
    "Open Questions & Missing Angles": (
        "open questions", "open questions & missing angles",
        "open questions and missing angles", "missing angles",
        "unanswered questions", "remaining gaps",
    ),
    "Key Figures": (
        "key figures", "key quantitative figures", "key numbers",
        "quantitative findings",
    ),
    "Auditable Source Ledger": (
        "auditable source ledger", "source ledger", "source audit",
        "evidence ledger", "full source list",
    ),
}


# Heading the deterministic reasoning fallback emits. Recognized so the
# writer's own "Reasoning"/"Argument" section counts as present (the structure
# is then not duplicated).
_REASONING_HEADINGS = (
    "reasoning", "argument", "analysis and reasoning", "reasoning structure",
    "argument structure", "conclusions",
)


def _has_reasoning_section(answer: str) -> bool:
    present = _present_section_keys(answer)
    return any(_normalize_heading(h) in present for h in _REASONING_HEADINGS)


def ensure_reasoning_structure(answer: str, *, ctx: Dict[str, Any]) -> str:
    """Append the deterministic argument structure when the writer omitted it.

    The ReasoningMap is the argument layer, so its loss must never depend on
    the writer choosing to surface it. When the report already carries a
    Reasoning/Argument/Conclusions section the writer has done the job and this
    is a no-op; otherwise the measured structure is appended verbatim — no
    invented content, consistent with `ensure_required_sections`. Total and
    fail-safe: a missing/empty map or an unrenderable one leaves the answer
    untouched.
    """
    if not answer:
        return answer
    reasoning = (ctx or {}).get("reasoning")
    if reasoning is None or not hasattr(reasoning, "render_for_writer"):
        return answer
    if _has_reasoning_section(answer):
        return answer
    render = getattr(reasoning, "render_for_report", None) or getattr(
        reasoning, "render_for_writer"
    )
    try:
        rendered = str(render()).strip()
    except Exception:  # noqa: BLE001
        return answer
    if not rendered:
        return answer
    return answer.rstrip() + "\n\n## Reasoning\n\n" + rendered


def _present_section_keys(answer: str) -> Set[str]:
    """Normalized headings present in the report body."""
    present: Set[str] = set()
    for match in re.finditer(r"^\s{0,3}#{1,6}\s+(.*\S)\s*$", answer or "", re.M):
        present.add(_normalize_heading(match.group(1)))
    return present


def _normalized_aliases(canonical: str) -> tuple:
    return tuple(_normalize_heading(alias) for alias in REQUIRED_SECTIONS[canonical])


def _finding_confidence(fact: Dict[str, Any]) -> float:
    """Per-claim confidence in [0, 1], from measured signals only.

    Base is the claim's own confidence; verification and independent
    corroboration each lift it (a verified, multi-source claim is more
    trustworthy than an unverified single-source one), and a single-source or
    unverified claim is capped in the provisional band. Never invented: a fact
    with no confidence field reads 0.0, not a flattering default.
    """
    try:
        base = float(fact.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        base = 0.0
    base = max(0.0, min(1.0, base))
    if fact.get("verified") is True:
        base += 0.05
    try:
        corroboration = int(fact.get("corroboration_count", 1) or 1)
    except (TypeError, ValueError):
        corroboration = 1
    if corroboration >= 2:
        base += 0.05
    # A single-source or unverified claim must never read as high-confidence:
    # cap it so a provisional finding cannot masquerade as established.
    if corroboration <= 1 or fact.get("verified") is not True:
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
    """Short, measured justification for a finding's confidence.

    States what was actually checked — verification, independent corroboration,
    source grade — so a skeptical reader can see WHY the number is what it is
    instead of trusting a bare score.
    """
    parts: List[str] = []
    if fact.get("verified") is True:
        parts.append("verified against source")
    else:
        parts.append("unverified")
    try:
        corroboration = int(fact.get("corroboration_count", 1) or 1)
    except (TypeError, ValueError):
        corroboration = 1
    if corroboration >= 2:
        parts.append(f"{corroboration} independent sources")
    else:
        parts.append("single source")
    if fact.get("temporal_projection"):
        parts.append("projection, not observed")
    return "; ".join(parts)


def _ledger_warnings(ctx: Dict[str, Any]) -> str:
    """Deterministic ledger warnings appended to Evidence Strength.

    Two composition failures are surfaced, never hidden: regulation dominance
    (>30% of evidence), which turns a trends brief into a legal summary, and
    non-Western under-representation, which turns a global picture into a US/EU
    one. Both are warnings, not silent omissions.
    """
    warnings: List[str] = []
    try:
        regulation = float(ctx.get("regulation_share", 0.0) or 0.0)
    except (TypeError, ValueError):
        regulation = 0.0
    if regulation > 0.30:
        warnings.append(
            f"Warning: regulation-sourced evidence is {regulation:.0%} of the pool "
            "(>30%). Regulatory material is reactive context, not the lead story; "
            "read capability/economics/adoption findings with that skew in mind."
        )
    try:
        non_western = float(ctx.get("non_western_share", 0.0) or 0.0)
    except (TypeError, ValueError):
        non_western = 0.0
    if non_western < 0.10:
        warnings.append(
            f"Warning: non-Western sources are {non_western:.0%} of the pool. The "
            "picture may be US/EU-centric; treat global claims as provisional."
        )
    try:
        primary = float(ctx.get("primary_share", 0.0) or 0.0)
    except (TypeError, ValueError):
        primary = 0.0
    if primary < 0.30:
        warnings.append(
            f"Warning: only {primary:.0%} of sources are primary (papers, official "
            "reports, filings); secondary summaries dominate."
        )
    return ("\n\n" + " ".join(warnings)) if warnings else ""


def _render_finding_line(fact: Dict[str, Any]) -> str:
    """One Key-Findings bullet: claim + citation + confidence + grade + why.

    Every finding carries a per-claim confidence score and an A/B/C evidence
    grade with a short measured justification, so the reader can triage which
    findings to trust and which are provisional. Returns "" for an empty claim.
    """
    claim = re.sub(r"\s+", " ", str(fact.get("claim", "") or "")).strip()
    if not claim:
        return ""
    try:
        index = int(fact.get("citation"))
    except (TypeError, ValueError):
        index = 0
    marker = f" [{index}]" if index else ""
    confidence = _finding_confidence(fact)
    grade = _finding_grade(fact)
    grade_tag = f", grade {grade}" if grade else ""
    justification = _finding_justification(fact)
    return f"- {claim}{marker} — confidence {confidence:.2f}{grade_tag} ({justification})"


def _render_required_section(
    canonical: str,
    *,
    ctx: Dict[str, Any],
    usable_facts: Sequence[Dict[str, Any]],
    contradictions: Sequence[Dict[str, Any]],
    cited_facts: Sequence[Dict[str, Any]] = (),
) -> str:
    """Deterministic body for a missing required section, from measured state.

    Never invents facts: each section is assembled from signals the pipeline
    already measured (grade distribution, verified/corroborated counts,
    contradictions, red-team findings, source mix). A genuinely empty signal
    produces an honest "none detected" statement, not a fabricated one.
    """
    total = len(usable_facts)
    verified = sum(1 for f in usable_facts if f.get("verified") is True)
    corroborated = sum(
        1 for f in usable_facts if int(f.get("corroboration_count", 1) or 1) > 1
    )
    distribution = ctx.get("evidence_distribution")
    dist = distribution if isinstance(distribution, dict) else {}
    a = int(dist.get("A", 0) or 0)
    b = int(dist.get("B", 0) or 0)
    c = int(dist.get("C", 0) or 0)
    d = int(dist.get("D", 0) or 0)

    if canonical == "Key Findings":
        lines = ["## Key Findings", ""]
        source_facts = list(cited_facts) if cited_facts else list(usable_facts)
        top = _stratified_top_facts(source_facts, per_angle=2, cap=8)
        for fact in top:
            rendered = _render_finding_line(fact)
            if rendered:
                lines.append(rendered)
        if len(lines) == 2:
            lines.append("- No verified findings were extracted from the available evidence.")
        return "\n".join(lines)

    if canonical == "Evidence Strength":
        if total == 0:
            body = "No verified evidence was available to grade."
        else:
            body = (
                f"{verified}/{total} claims were verified against their cited source; "
                f"{corroborated}/{total} are independently corroborated by 2+ sources. "
                f"Evidence grades: A={a}, B={b}, C={c}, D={d} "
                "(A/B = verified and strongly/independently sourced)."
            )
        body += _ledger_warnings(ctx)
        return "## Evidence Strength\n\n" + body

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
        for c in (contradictions or [])[:5]:
            if not isinstance(c, dict):
                continue
            a_text = str(c.get("claim_a", "") or "")[:140]
            b_text = str(c.get("claim_b", "") or "")[:140]
            if a_text and b_text:
                suffix = (
                    f" RESOLVED: {c.get('resolution', '')}" if c.get("resolved") else ""
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
            # nothing. An empty section with no failed search is an UNKNOWN,
            # not a clean bill of health.
            counter_searched = bool(ctx.get("counter_evidence_attempted"))
            if counter_searched:
                lines.append(
                    "- A dedicated counter-evidence search was run and returned no "
                    "credible opposing claims or source conflicts. This is an "
                    "absence of found counter-evidence, not proof none exists."
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
        listed = [str(g).strip() for g in gaps if str(g).strip()] if isinstance(gaps, (list, tuple)) else []
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
            rendered = _render_finding_line(fact)
            if rendered:
                lines.append(rendered)
        if len(lines) == 2:
            lines.append("- No quantitative figures were extracted from the evidence.")
        return "\n".join(lines)

    if canonical == "Auditable Source Ledger":
        lines = ["## Auditable Source Ledger", ""]
        seen: set = set()
        for fact in usable_facts:
            source = str(fact.get("source", "") or "").strip()
            if not source or source in seen:
                continue
            seen.add(source)
            fetched = str(fact.get("fetched_at", "") or "")
            published = str(fact.get("published_at", "") or "")
            pulled = str(fact.get("retrieved_at", "") or fetched or "unrecorded")
            corroboration = int(fact.get("corroboration_count", 1) or 1)
            flags = []
            if corroboration <= 1:
                flags.append("single-source / provisional")
            if fact.get("temporal_projection"):
                flags.append("projection (not observed)")
            if fact.get("verified") is not True:
                flags.append("unverified")
            dates = []
            if published:
                dates.append(f"published {published}")
            dates.append(f"retrieved {pulled}")
            suffix = f" [{'; '.join(flags)}]" if flags else ""
            lines.append(f"- {source} ({'; '.join(dates)}){suffix}")
        if len(lines) == 2:
            lines.append("- No sources were retained for this run.")
        return "\n".join(lines)

    # Executive Summary fallback (only when the writer omitted it entirely).
    confidence = ctx.get("confidence")
    level = ""
    try:
        score = float(confidence)
        level = f" Overall confidence: {'High' if score >= 0.75 else 'Medium' if score >= 0.5 else 'Low'} ({score:.2f})."
    except (TypeError, ValueError):
        level = ""
    return (
        "## Executive Summary\n\n"
        f"This report synthesizes {verified} verified claim(s) from "
        f"{total} extracted, drawing on the evidence graded above.{level}"
    )


def ensure_required_sections(
    answer: str,
    *,
    ctx: Dict[str, Any],
    usable_facts: Sequence[Dict[str, Any]],
    contradictions: Sequence[Dict[str, Any]],
    cited_facts: Sequence[Dict[str, Any]] = (),
) -> str:
    """Guarantee the five mandatory sections are present, adding any missing.

    Deterministic and total. A missing section is appended from measured state
    (never invented), so the writer cannot omit Limitations/unknowns or
    counterarguments and ship a report that hides them. Existing, differently
    titled sections that mean the same thing (e.g. "Evidence & Confidence")
    satisfy the requirement and are left untouched.
    """
    if not answer:
        return answer
    present = _present_section_keys(answer)
    missing = [
        canonical
        for canonical in REQUIRED_SECTIONS
        if not any(alias in present for alias in _normalized_aliases(canonical))
    ]
    if not missing:
        return answer
    blocks: List[str] = []
    for canonical in missing:
        blocks.append(
            _render_required_section(
                canonical,
                ctx=ctx,
                usable_facts=usable_facts,
                contradictions=contradictions,
                cited_facts=cited_facts,
            )
        )
    return answer.rstrip() + "\n\n" + "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Citation numbering: bind every fact to the number the model must cite
# ---------------------------------------------------------------------------

MAX_LEGEND_SOURCES = 14

# See _FACT_CAP_LADDER comment inside synthesize(): caps tried in order when
# the provider rejects the request size.
_FACT_CAP_LADDER = (40, 24, 14)


def _number_facts(
    top_facts: Sequence[Dict[str, Any]], max_sources: int = MAX_LEGEND_SOURCES
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Number the sources and stamp each fact with the number that cites it.

    Two fixes over the old `_numbered_sources`:

    * Grouping is by CANONICAL url, so `page?utm_source=x` and `page` are one
      source with one number instead of two entries the model may cite
      inconsistently.
    * Facts whose source did not make the legend are not sent to the model at
      all. Previously the legend capped at 12 while up to 40 facts were sent, so
      the model was handed evidence it had no legal way to cite — it either
      dropped a real finding or invented a marker.
    """
    numbered, pairs = _assign_numbers(top_facts, max_sources)
    cited: List[Dict[str, Any]] = []
    for fact, index in pairs:
        item = dict(fact)
        item["citation"] = index
        cited.append(item)
    return numbered, cited


def _assign_numbers(
    facts: Sequence[Dict[str, Any]], max_sources: int = MAX_LEGEND_SOURCES
) -> tuple[List[Dict[str, Any]], List[tuple[Dict[str, Any], int]]]:
    """Legend entries plus the (fact, number) pairs, keeping fact identity.

    Identity matters for the extractive path, which needs to attach the right
    marker to a claim it has already rendered.
    """
    numbered: List[Dict[str, Any]] = []
    number_by_document: Dict[str, int] = {}
    pairs: List[tuple[Dict[str, Any], int]] = []

    for fact in facts or []:
        url = str(fact.get("source", "") or "")
        if not url:
            continue
        document = canonical_url(url) or url
        index = number_by_document.get(document)
        if index is None:
            if len(numbered) >= max(1, max_sources):
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

    Sending raw dicts (the old behaviour) spent tokens on keys the writer cannot
    use and buried the attribution the writer needs most.
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
        corroboration = int(fact.get("corroboration_count", 1) or 1)
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


def _render_ranges_block(contradictions: Sequence[Dict[str, Any]]) -> str:
    """Pre-computed ranges for numeric conflicts.

    The prompt has always forbidden averaging conflicting numbers, but never
    supplied the range to use instead — so the model either picked one or
    hedged vaguely. Now the arithmetic is done here.
    """
    ranges = numeric_ranges(contradictions or [])
    if not ranges:
        return ""
    lines: List[str] = []
    for item in ranges[:4]:
        unit = "" if item["unit"] in ("", "dimensionless") else f" {item['unit']}"
        domains = ", ".join(sorted({extract_domain(s) for s in item["sources"] if s})[:4])
        lines.append(
            f"- Sources disagree: report the range {item['low']:g}{unit} to "
            f"{item['high']:g}{unit} (spread {item['spread']:g}{unit}); "
            f"disagreeing sources: {domains or 'multiple'}. Never average these."
        )
    return "Conflicting quantities, pre-computed as ranges:\n" + "\n".join(lines) + "\n\n"


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
    """
    audit = CitationAudit()
    body = _strip_sections(answer, ("## Sources", "## Evidence integrity"))
    valid = {int(s["n"]) for s in numbered if str(s.get("n", "")).isdigit() or isinstance(s.get("n"), int)}
    claims_by_number: Dict[int, List[str]] = {}
    for fact in cited_facts:
        try:
            index = int(fact.get("citation"))
        except (TypeError, ValueError):
            continue
        claims_by_number.setdefault(index, []).append(str(fact.get("claim", "") or ""))
    evidence_text = " ".join(
        str(f.get("claim", "") or "") + " " + str(f.get("direct_quote", "") or "")
        for f in cited_facts
    )
    evidence_values = {
        round(q.value, 4) for q in extract_numbers(evidence_text, limit=400)
    }

    for sentence in _audit_units(body):
        stripped = sentence.lstrip("-* ").strip()
        if not stripped:
            continue
        markers = [int(m) for m in re.findall(r"\[(\d+)\]", stripped)]
        if not markers and (_ANALYSIS_LEAD_RE.match(stripped) or _DISAMBIG_LINE_RE.match(stripped)):
            # Analysis/transition prose, the report's own scaffolding, and the
            # ambiguity disambiguation lines are not evidence claims; they
            # belong in neither the numerator nor the denominator of citation
            # density.
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
            if _FACTUAL_HINT_RE.search(stripped) and not _ANALYSIS_LEAD_RE.match(stripped):
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


def _value_grounded(value: float, evidence_values: Set[float], tolerance: float = 0.02) -> bool:
    """A drafted number is grounded when some evidence number matches it.

    Tolerance is relative, so 2.5 million vs 2,500,000 and rounding differences
    between a source and a paraphrase both pass, while a fabricated figure does
    not.
    """
    if value in evidence_values:
        return True
    for known in evidence_values:
        scale = max(abs(known), abs(value), 1e-9)
        if abs(known - value) / scale <= tolerance:
            return True
    return False


def _drop_invalid_markers(answer: str, count: int) -> str:
    """Remove [n] markers that resolve to nothing in the legend."""

    def _keep(match: "re.Match") -> str:
        try:
            index = int(match.group(1))
        except (TypeError, ValueError):
            return ""
        return match.group(0) if 1 <= index <= count else ""

    return re.sub(r"\[(\d+)\]", _keep, answer or "")


def _integrity_note(audit: CitationAudit) -> str:
    """Disclose what the audit found, in the report itself.

    A research system that silently repairs its own citations trains its users
    to trust output it has not earned. If the draft had holes, the reader is
    told which ones.
    """
    if audit.is_clean:
        return ""
    lines = ["## Evidence integrity", ""]
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
    lines.append(f"Citation density: {audit.citation_density:.0%} of sentences cited.")
    return "\n".join(lines)


def _strip_sections(answer: str, headings: Sequence[str]) -> str:
    """Drop appended machine-written sections before auditing the prose.

    Auditing the legend would flag every source number as an uncited factual
    sentence, and auditing the confidence panel would flag its own scores as
    ungrounded numbers.
    """
    text = answer or ""
    for heading in headings:
        index = text.find(f"\n{heading}")
        if index != -1:
            text = text[:index]
    return text


# ---------------------------------------------------------------------------
# Measured appendix: sections that must never be model-written
# ---------------------------------------------------------------------------

def _append_evidence_appendix(
    answer: str,
    ctx: Dict[str, Any],
    usable_facts: Sequence[Dict[str, Any]],
    contradictions: Sequence[Dict[str, Any]],
) -> str:
    """Append the accounting the model cannot be trusted to produce.

    The prompt asks for an Evidence & Confidence section, and the model writes
    one — with numbers it estimates. The real counts, the confidence breakdown,
    the source mix and the surviving red-team objections are appended here from
    measured state, so those figures are always the true ones.
    """
    blocks: List[str] = []

    distribution = ctx.get("evidence_distribution")
    if isinstance(distribution, dict) and any(
        int(distribution.get(g, 0) or 0) for g in ("A", "B", "C", "D")
    ):
        blocks.append(
            "Evidence grades: "
            f"A={int(distribution.get('A', 0) or 0)}, "
            f"B={int(distribution.get('B', 0) or 0)}, "
            f"C={int(distribution.get('C', 0) or 0)}, "
            f"D={int(distribution.get('D', 0) or 0)} "
            "(A/B = verified and strongly/independently sourced)."
        )

    report = ctx.get("confidence_report")
    rendered = ""
    if report is not None and hasattr(report, "render"):
        try:
            rendered = str(report.render()).strip()
        except Exception:  # noqa: BLE001 - never let a panel break the report
            rendered = ""
    if rendered:
        blocks.append(rendered)

    urls = [str(f.get("source", "") or "") for f in usable_facts if f.get("source")]
    share = primary_source_share(urls)
    documents = len({canonical_url(u) or u for u in urls} - {""})
    domains = len({extract_domain(u) for u in urls} - {""})
    mix = [
        "## Source ledger",
        "",
        f"- Documents read: {documents} across {domains} independent domain(s)",
        f"- Primary sources (official filings, papers, datasets, standards): {share:.0%}",
    ]
    verified = sum(1 for f in usable_facts if f.get("verified") is True)
    mix.append(f"- Claims verified against their cited source: {verified}/{len(usable_facts)}")
    corroborated = sum(1 for f in usable_facts if int(f.get("corroboration_count", 1) or 1) > 1)
    if corroborated:
        mix.append(f"- Claims independently corroborated by 2+ sources: {corroborated}")
    conflict = summarize_contradictions(contradictions)
    if conflict["total"]:
        mix.append(
            f"- Source conflicts detected: {conflict['cross_source']} across sources "
            f"({conflict['severe']} severe); reported as ranges above"
        )
    blocks.append("\n".join(mix))

    findings = ctx.get("redteam_findings") or []
    if isinstance(findings, list) and findings:
        lines = ["## Standing objections", ""]
        for item in findings[:5]:
            if not isinstance(item, dict):
                continue
            statement = str(item.get("statement", "") or "").strip()
            if not statement:
                continue
            test = str(item.get("test", "") or "").strip()
            suffix = f" Resolve by: {test}" if test else ""
            lines.append(f"- {statement}{suffix}")
        if len(lines) > 2:
            blocks.append("\n".join(lines))

    changers = ctx.get("what_would_change_our_mind") or []
    if isinstance(changers, list) and changers:
        lines = ["## What would change this conclusion", ""]
        lines += [f"- {str(c).strip()}" for c in changers[:5] if str(c).strip()]
        if len(lines) > 2:
            blocks.append("\n".join(lines))

    if not blocks:
        return answer
    return answer.rstrip() + "\n\n" + "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Deterministic fallback: extractive report when the model call fails
# ---------------------------------------------------------------------------

def _deterministic_report(
    query: str,
    usable_facts: Sequence[Dict[str, Any]],
    top_facts: Sequence[Dict[str, Any]],
    ctx: Dict[str, Any],
    angles: Sequence[str],
) -> SynthesisResult:
    """An organized, honestly-labeled research digest — not imitation prose.

    The old extractive path glued source sentences into paragraphs, which
    read exactly like chunks cut from different sources. This path leans
    into what it is: a structured briefing. A query-framed executive
    summary with one headline finding, then ONE BULLET PER VERIFIED CLAIM
    grouped under its sense/angle section, the measured accounting, and a
    used-only legend. Complete sentences, real structure, zero fake
    narrative. MMR overlap threshold 0.35 splits observed paraphrases
    (0.39-0.53) from cross-sense claims (0.10-0.18).
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
        # 8 per group: large enough that number-bearing claims (the key
        # figures) survive the cap — the old separate figures section that
        # rescued them is gone.
        if len(bucket) < 8:
            bucket.append(item)
        if len(groups) >= 6 and all(len(v) >= 3 for v in groups.values()):
            break
    if not groups:
        return SynthesisResult(
            answer=f"No reliable evidence was retrieved for {_normalize_query_concept(query)}.",
            used_fallback=True,
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
        sections.append(f"## {title}\n\nVerified findings:\n\n" + "\n".join(bullets))

    thin_note = (
        "Evidence is thin — fewer than 3 verified facts support this report, "
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
        f" The evidence spans {len(groups)} distinct angles of this query, each in its own section below."
        if len(groups) >= 3
        else ""
    )

    headline_bullet = f"- {_with_citation(headline).strip()}" if headline is not None else ""
    summary = (
        "## Executive Summary\n\n"
        f"{thin_note}Verified findings for \u201c{_normalize_query_concept(query)}\u201d — "
        f"{gap_stats['verified']} verified claim(s) from {n_sources} source(s)."
        f"{ambiguity_note}"
        + (f"\n\n{headline_bullet}" if headline_bullet else "")
        + "\n\n"
        f"Confidence: {_confidence_statement(ctx, gap_stats['verified'])} — "
        f"{gap_stats['verified']} verified facts across {n_sources} sources.{degraded_note}"
    )
    sections.insert(0, summary)

    sections.append(_gaps_section(gap_stats))

    # The extractive path knows exactly which source each claim came from, so
    # it cites perfectly — claims were rendered with an identity token; now
    # that the used set is final, tokens become numbers.
    numbered, pairs = _assign_numbers(used[:40], max_sources=40)
    answer = "\n\n".join(sections) + "\n\n" + _legend_block(numbered)
    for fact, index in pairs:
        answer = answer.replace(_cite_token(fact), f"[{index}]")
    answer = re.sub(r"\s*\[\[c\d+\]\]", "", answer)
    cited = [dict(fact, citation=index) for fact, index in pairs]
    return SynthesisResult(
        answer=answer,
        sources=numbered,
        audit=audit_citations(answer, numbered, cited),
        used_fallback=True,
        angles=list(angles),
    )


def _cite_token(fact: Dict[str, Any]) -> str:
    """Placeholder standing in for a citation number not yet assigned."""
    return f"[[c{id(fact)}]]"


def _with_citation(fact: Dict[str, Any]) -> str:
    """A claim carrying its own citation placeholder.

    The marker goes INSIDE the sentence, before the terminal punctuation:
    "claim [1]." not "claim. [1]". Sentence splitters (this module's citation
    audit, the evaluation lab's answer-support check) break after [.!?], so a
    marker placed after the period became its own orphan unit — the audit then
    scored the extractive fallback report as largely uncited even though every
    claim carried a citation.
    """
    claim = str(fact.get("claim", "") or "").strip()
    if not claim:
        return ""
    # Dedupe normalizes punctuation away, so a missing terminal is restored
    # rather than assumed: an unterminated claim would glue itself to the
    # next line for every sentence splitter downstream.
    terminal = claim[-1] if claim[-1] in ".!?" else "."
    if claim[-1] in ".!?":
        claim = claim[:-1].rstrip()
    return f"{claim} {_cite_token(fact)}{terminal}"



# ---------------------------------------------------------------------------
# Retained helpers (unchanged behaviour, kept for the fallback path)
# ---------------------------------------------------------------------------

def _gap_stats(ctx: Dict[str, Any], usable_facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts the Confidence & Gaps section needs. Prefers workflow context
    (contradictions, degraded stages, totals); derives the rest from facts."""
    total = int(ctx.get("total_facts", 0) or 0) or len(usable_facts)
    if ctx.get("verified_count") is not None:
        verified = int(ctx.get("verified_count") or 0)
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
    """'High (0.76)' on the pipeline's 0-1 scale (0.75 = sufficient), or a
    volume-based level when no score was passed in context."""
    raw = ctx.get("confidence", None) if isinstance(ctx, dict) else None
    try:
        score = float(raw)
        level = "High" if score >= 0.75 else ("Medium" if score >= 0.5 else "Low")
        return f"{level} ({score:.2f})"
    except (TypeError, ValueError):
        level = "High" if verified_count >= 8 else ("Medium" if verified_count >= 3 else "Low")
        return level


def _gaps_section(stats: Dict[str, Any]) -> str:
    """Evidence & Confidence + Limitations: score/counts/conflicts, then
    what could not be verified. Both mandatory even at high confidence."""
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
            "flagged — see the Contradictions section of the full report; "
            "conflicting numbers are reported as ranges, not picked."
        )
    if stats["degraded"]:
        lines.append(
            "Pipeline gaps: deterministic fallback covered "
            f"{', '.join(stats['degraded'])} — those sections are extractive, "
            "not model-written."
        )
    lines.extend([
        "",
        "## Limitations",
        "",
        "Could not verify: claims without a traceable source were discarded "
        "during synthesis and do not appear above.",
    ])
    return "\n".join(lines)


def _deterministic_disambiguation(intent: Dict[str, Any]) -> str:
    """The numbered 'n) **Sense** — explanation' block the gate requires.

    The disambiguation lines are definitional common knowledge for the
    non-researched sense, so they can be emitted deterministically from the
    intent instead of hoping the writer complies. Returns "" for a
    non-ambiguous query or one with fewer than two labelled senses.
    """
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
    silently picks one meaning (the hard relevance failure this gate exists
    for). Deterministic, no LLM.
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
        # Put the lines at the top of the Executive Summary, where the
        # contract says they belong, rather than above the heading.
        head = answer[: heading.end()]
        tail = answer[heading.end():].lstrip()
        return f"{head}\n\n{block}\n\n{tail}"
    return f"{block}\n\n{answer.lstrip()}"


def _render_ambiguity_block(intent: Dict[str, Any]) -> str:
    """Mandatory disambiguation contract for an ambiguous query.

    The report must open by naming the senses, state which one the research
    focused on, and keep each sense's evidence separate — the structural fix
    for "AI transformer limitations mixed with electrical statistics".
    """
    if not isinstance(intent, dict) or not intent.get("ambiguity"):
        return ""
    senses = [s for s in (intent.get("senses") or []) if isinstance(s, dict) and str(s.get("label", "")).strip()]
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
            "(evidence lines carry [sense: ...] tags — a section about one sense cites "
            "only that sense's facts)."
        )
    else:
        structure = (
            f"The detailed sections focus on meaning 1 ('{focus}'). Do NOT spend "
            "sections or citations on the other meaning(s) — their one disambiguation "
            "line above is enough."
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
    honesty baseline for Confidence & Gaps. Empty when no context passed."""
    if not ctx:
        return ""
    parts = []
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
            parts.append("Flagged source conflicts (surface these, never silently pick one):\n" + "\n".join(lines))
    try:
        conf = float(ctx.get("confidence", "nan"))
        parts.append(f"Pipeline confidence: {conf:.2f} (0.75+ = sufficient).")
    except (TypeError, ValueError):
        pass
    degraded = ctx.get("degraded", []) or []
    if isinstance(degraded, list) and degraded:
        parts.append(f"Stages on deterministic fallback: {', '.join(str(d) for d in degraded)}.")
    total = ctx.get("total_facts", 0) or 0
    verified = ctx.get("verified_count", 0) or 0
    if total:
        parts.append(f"Evidence pool: {verified}/{total} facts verified; unverified claims were excluded.")
    parts.append(
        "The figures above are INTERNAL METADATA for your judgement only. "
        "Never quote them verbatim in the report body — the appendix states "
        "them, and the body describes evidence strength in words."
    )

    # Evidence grades (Step 4): the measured quality distribution, so the
    # writer can separate what is established from what is merely asserted
    # instead of presenting every sentence with equal authority.
    distribution = ctx.get("evidence_distribution")
    if isinstance(distribution, dict) and any(int(distribution.get(g, 0) or 0) for g in ("A", "B", "C", "D")):
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
    # evidence actually supports. The writer must state the conclusion the
    # evidence backs, present the competing explanations where they exist,
    # separate established from inferred from unknown, and answer the question;
    # for a decision-shaped query it must state the implication or explicitly
    # say the evidence is insufficient. Rendered from the ReasoningMap built in
    # workflow.synthesizer_node (a no-op when the map is absent/empty).
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
    # report that addresses it instead of one that a reader can dismantle.
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
    facts: List[Dict[str, Any]], per_angle: int = 6, cap: int = 30
) -> List[Dict[str, Any]]:
    """Top facts round-robin across sub-questions instead of pure confidence
    order. Pure ranking lets one low-scoring source domain bury a whole
    angle (observed live: the market angle cut from a top-20 while two
    definition angles filled it). Every angle keeps up to `per_angle` facts.
    """
    ranked = sorted(
        facts or [],
        key=_fact_confidence,
        reverse=True,
    )
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

    GPT Researcher compresses scraped chunks before the writer sees them so
    many sources become coherent reasoning instead of a fact dump. The MARS
    equivalent must NOT drop distinct claims (that would weaken verification
    guarantees), so compression here only merges claims that are already
    near-identical, keeps one representative, and records how many sources
    asserted it in `corroboration_count`. Distinct claims pass through
    untouched, in their original order.

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
            if semantic_similarity(claim, str(existing.get("claim", ""))) >= similarity_threshold:
                # Numeric guard: two claims that share wording but carry
                # DIFFERENT quantities are not the same assertion. Merging them
                # silently replaced specific evidence (e.g. "cost overrun reached
                # $11.5bn" vs "cost overrun reached $12.7bn") with one generic
                # representative, which is exactly the "distinct specifics become
                # generic merged text" failure. Keep them separate.
                if _distinct_quantities(str(existing.get("claim", "")), claim):
                    continue
                # Same assertion restated: keep the better-supported copy and
                # count the rest as corroboration rather than discarding them.
                try:
                    existing_count = int(existing.get("corroboration_count", 1) or 1)
                except (TypeError, ValueError):
                    existing_count = 1
                existing["corroboration_count"] = existing_count + 1
                if _fact_confidence(fact) > _fact_confidence(existing):
                    merged_claim = existing["claim"]
                    existing.clear()
                    existing.update(fact)
                    existing["claim"] = merged_claim
                merged = True
                break
        if not merged:
            kept.append(dict(fact))
    return kept


def _quantity_signature(text: str) -> Set[Tuple[float, str]]:
    """(value, unit) pairs in a claim, unit-normalized for comparison."""
    out: Set[Tuple[float, str]] = set()
    for q in extract_numbers(text, limit=12):
        out.add((round(q.value, 4), str(q.unit or "").lower()))
    return out


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
    """Humanize a grouping key for a `## ` header: leading-capitalized,
    never raw keyword salad in lowercase."""
    text = (sub_question or "").strip()
    if not text:
        return ""
    return text[0].upper() + text[1:]


def _append_source_legend(answer: str, numbered: List[Dict[str, str]]) -> str:
    return f"{answer.rstrip()}\n\n" + _legend_block(numbered)


def _legend_block(numbered: Sequence[Dict[str, Any]]) -> str:
    """`## Sources` section: one entry per line, blank-line separated, so
    the frontend renders a clean vertical numbered list under its own
    heading (not one glued paragraph).

    Each entry now carries its tier ("official publisher", "peer-reviewed",
    "media", ...) and a primary-source marker. A reader deciding how much to
    trust finding [4] should not have to recognise the domain to know whether
    it is a regulator's filing or a blog aggregating one.
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


_FIGURE_RE = re.compile(r"\d")


# Sentences that expose the pipeline's own internals rather than the subject.
# The model is told not to write these (SYNTHESIZER_SYSTEM_PROMPT rule 7), but
# live runs proved it copies the metadata block verbatim ("pipeline confidence
# is 0.55, below the 0.75 threshold", "the pipeline itself reports only 80 of
# 96 facts verified"). The machine-appended appendix states these numbers
# correctly; the body must not narrate the research system.
_PIPELINE_TELEMETRY_RE = re.compile(
    r"(?i)\b("
    r"pipeline confidence|pipeline reports|the pipeline itself|"
    r"relevance \d{1,3}/\d{1,3}|quality (?:score|review)|"
    r"below the \d{1,3}/\d{1,3} floor|below the \d\.\d+ threshold|"
    r"confidence is \d\.\d+|fact(?:s)? (?:in the pool|verified)|"
    r"of \d+ facts|evidence pool|verified facts|"
    r"self-?verif|pipeline stages?|deterministic fallback|degraded run"
    r")\b"
)


def _scrub_pipeline_telemetry(text: str) -> str:
    """Drop sentences that narrate the pipeline's own metrics.

    Applied to the report body only, BEFORE the measured appendix is
    appended, so the correct figures in the appendix are never touched.
    Sentence-granular and deterministic: a sentence mentioning internal
    telemetry is removed whole (it is meta-commentary, not subject matter);
    paragraphs that become empty collapse away.
    """
    if not text:
        return text
    kept_paras: List[str] = []
    for para in text.split("\n\n"):
        stripped = para.strip()
        # Headings are structural — leave them alone. Everything else is
        # scrubbed line by line so one telemetry bullet does not erase a list.
        if not stripped or stripped.startswith("#"):
            kept_paras.append(para)
            continue
        kept_lines: List[str] = []
        for line in para.split("\n"):
            if not line.strip():
                continue
            if _PIPELINE_TELEMETRY_RE.search(line):
                continue
            kept_lines.append(line)
        if kept_lines:
            kept_paras.append("\n".join(kept_lines))
    return "\n\n".join(kept_paras).strip()


def _normalize_query_concept(query: str) -> str:
    text = re.sub(r"\s+", " ", (query or "").strip()).strip(" ?.!")
    lower = text.lower()
    if lower.startswith("what is "):
        text = text[8:].strip()
    elif lower.startswith("define "):
        text = text[7:].strip()
    if text:
        return text[0].upper() + text[1:]
    return "This topic"


# Inline bullet separators: models sometimes emit bullets on one line
# ("... induction [4]. - Electrical transformers modify ..."). A " - "
# right after a sentence end / citation bracket is a list separator, not
# prose; splitting on it restores real bullet lines. Only applied inside
# lines that already start with "- ".
_BULLET_SEP_RE = re.compile(r"(?<=[.!?\]])\s+-\s+(?=[A-Z0-9\"'(])")


def _sanitize_answer_text(answer: str, query: str) -> str:
    # Preserve the report's visual hierarchy: markdown headings (## / ###)
    # become their own blocks, consecutive "- " lines stay distinct list
    # items (joined with single newlines into one bullet block), and
    # paragraph breaks are kept — the frontend's section renderer maps all
    # of these to real elements. Only intra-line whitespace collapses;
    # consecutive prose lines still merge (model line-wrapping must not
    # become line breaks).
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
    q_escaped = re.escape(q)
    text = re.sub(
        rf"(?i)^{q_escaped}\s+refers to",
        f"{_normalize_query_concept(query)} refers to",
        text,
    )

    # Collapse immediate repeated clause: "X is ... X is ..."
    text = re.sub(r"(?i)(\b[A-Z][A-Za-z\s\-]{2,40}\s+is\b[^.]*\.)\s+\1", r"\1", text)

    # Question-shaped headings are not labels: shorten them so a raw planner
    # question or writer-emitted question never becomes a section heading.
    text = _dedupe_heading(text)

    return text.strip()
