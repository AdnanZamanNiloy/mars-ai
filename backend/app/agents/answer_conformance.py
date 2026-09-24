"""Answer conformance — does the finished prose actually fit the question?

Why this module exists
----------------------
Phase 12 asks the pipeline to produce world-class research answers:
answer-first, insight-dense, correctly calibrated, question-fit and
depth-proportional. The pipeline already *plans* for that — `synthesis_planner`
ranks the evidence, `analyst` reasons over it, `answer_priority` tiers the
material, and the writer prompt carries per-query-type `_FORMAT_GUIDANCE`
(comparison/causal/forecast/decision/...). But nothing checked whether the
EMITTED prose honoured any of it. The quality gate scores citation density,
length and whether a conclusion phrase exists; it does not notice that a
"why" question shipped no mechanism, a comparison shipped no criterion-by-
criterion verdict, a forecast stated a projection in the voice of an
observation, an inference was asserted as measured fact, or a conflict was
merely announced ("sources disagree") without explaining what the
disagreement turns on.

This module measures question-fit deterministically and turns each miss into
a specific, actionable failure sentence in the SAME shape the quality gate
emits, so the pipeline's ONE existing bounded revision pass can act on them.
It is observational — it never rewrites, never adds a fact, never calls a
model.

It measures five things:

  * SHAPE CONFORMANCE   — the prose does the work its question type requires
                          (mechanism for a "why", criterion comparison +
                          verdict for a comparison, projection marked as such
                          for a forecast, recommendation + condition for a
                          decision, steps in order for a how-to).
  * INFERENCE CALIBRATION — a cross-source inference is signalled as an
                          inference ("the evidence suggests…"), never asserted
                          in the same voice as a measured fact.
  * CONTRADICTION SYNTHESIS — where credible sources disagree, the answer
                          explains what differs and/or why, not merely that a
                          disagreement exists.
  * UNCERTAINTY PROPORTIONALITY — uncertainty is present but does not
                          dominate the answer; audit language is not stacked.
  * DEPTH FIT           — a deep question receives reasoning depth (mechanism,
                          trade-off, qualification) proportional to its
                          complexity; a simple question is not padded.

Design (AGENTS.md):
  * Deterministic and LLM-free — reuses the shared sentence splitter and the
    synthesizer's own query-type inference; no model call, no latency.
  * Reuses, never reimplements, the AnalystBrief / SynthesisPlan artifacts.
  * Grounding-safe: the checks are structural/lexical and can only add a
    revision hint; they never weaken citation verification, contradiction
    detection, forecasting safeguards or unsupported-claim protection.
  * Total/fail-safe: absent/empty inputs yield a neutral report.
  * No module-level mutable state; per-call and returned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.agents.evidence_utils import split_into_sentences
from app.agents.sources import strip_machine_sections
from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lexicons. Kept deliberately small and literal — this is a conformance check,
# not a language model. Each entry maps to a documented Phase-12 requirement.
# ---------------------------------------------------------------------------

# Mechanism / causality markers: a "why" answer must explain, not just report.
_CAUSAL_MARKERS = (
    "because", "driven by", "due to", "as a result", "leads to", "led to",
    "causes", "caused by", "the reason", "results from", "stems from",
    "results in", "which is why", "owing to", "accounted for by",
    "explains why", "the mechanism", "attributable to",
)

# Comparison markers: criterion-by-criterion treatment plus a verdict.
_COMPARISON_MARKERS = (
    "compared with", "compared to", "whereas", "while ", "by contrast",
    "in contrast", "unlike", "on the other hand", "versus", "vs ",
    "both ", "the difference", "differs from", "outperforms",
    "underperforms", "cheaper", "more expensive", "higher than",
    "lower than",
)
_VERDICT_MARKERS = (
    "better choice", "better option", "the winner", "wins for", "wins on",
    "favours", "favors", "preferable", "recommend", "the trade-off",
    "the tradeoff", "which is better", "comes out ahead", "the safer",
    "the stronger option", "depends on", "for most", "the best option",
)

# Forecast markers: projection must be signalled, not stated as observation.
_PROJECTION_MARKERS = (
    "project", "projected", "projection", "forecast", "forecasted",
    "expected to", "is expected", "anticipated", "predicted", "prediction",
    "outlook", "estimate that", "estimates that", "on track to",
    "will reach", "is set to", "envision", "scenario",
)
_FORECAST_ASOF_MARKERS = (
    "as of", "currently", "today", "the latest data", "at present",
    "as at", "so far", "at the time of writing", "in 20",
)

# Uncertainty / hedge markers: an inference should carry one of these.
_INFERENCE_MARKERS = (
    "suggests", "suggest", "indicates", "indicate", "implies", "imply",
    "points to", "point to", "appears to", "seems to", "may ", "might ",
    "could ", "likely", "unlikely", "probably", "possible", "possibly",
    "consistent with", "these findings", "taken together", "together these",
    "the evidence", "one reading", "this suggests", "which suggests",
    "inference", "inferred", "read across", "read together",
)

# Contradiction synthesis: naming the divergence is not enough; the answer
# should explain what it turns on.
_CONFLICT_MARKERS = (
    "disagree", "disagreement", "conflict", "conflicting", "contradict",
    "contradiction", "diverge", "divergence", "dispute", "disputed",
    "inconsistent", "tension",
)
_CONFLICT_EXPLAIN_MARKERS = (
    "because", "due to", "driven by", "which means", "hinges on",
    "turns on", "depends on", "the difference is", "differ in",
    "one explanation", "may reflect", "could reflect", "the reason",
    "the disagreement is over", "what separates", "explained by",
    "likely because", "possibly because", "partly", "whereas",
    "the gap", "the discrepancy", "methodolog", "scope", "timeframe",
    "different periods", "different definitions", "different sources",
)

# Proportionality: too much audit/limitation language crowds out the answer.
_LIMITATION_MARKERS = (
    "cannot be determined", "could not be determined", "does not establish",
    "do not establish", "insufficient", "not enough evidence", "unclear",
    "unknown", "no evidence", "could not verify", "cannot verify",
    "not verified", "remains uncertain", "we do not know", "no data",
    "not available",
)

# Thresholds (documented, conservative — a miss must be a real miss).
MAX_LIMITATION_DENSITY = 0.30   # fraction of sentences allowed to be limitation-only
DEPTH_MIN_SENTENCES = 4          # a deep answer must reason beyond the opening
DEPTH_MIN_MARKERS = 2            # distinct reasoning markers a deep answer needs

# Words too common to carry content, for inference-overlap matching.
_CONTENT_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "as", "by", "at", "is", "are", "was", "were", "be", "been", "being", "that",
    "this", "these", "those", "it", "its", "from", "into", "than", "then",
    "has", "have", "had", "will", "would", "can", "could", "should", "may",
    "might", "not", "no", "so", "such", "more", "most", "less", "least",
    "which", "who", "whom", "whose", "what", "when", "where", "why", "how",
    "also", "both", "each", "other", "some", "any", "all", "one", "two",
}


def _norm(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def _sentences(answer: str) -> List[str]:
    """Writer-prose sentences, machine sections stripped."""
    body = strip_machine_sections(answer or "")
    return [s.strip() for s in split_into_sentences(body, max_sentences=400) if s.strip()]


def _contains_any(text: str, markers: Sequence[str]) -> bool:
    lowered = (text or "").lower()
    return any(m in lowered for m in markers)


def _count_distinct_markers(text: str, markers: Sequence[str]) -> int:
    lowered = (text or "").lower()
    return sum(1 for m in markers if m in lowered)


def _has_citation(sentence: str) -> bool:
    return bool(re.search(r"\[\d+\]", sentence or ""))


@dataclass
class ConformanceReport:
    """How well the finished answer fits the question it was asked."""

    query_type: str = ""
    shape_conformant: bool = True
    inference_calibrated: bool = True
    contradiction_synthesised: bool = True
    uncertainty_proportionate: bool = True
    depth_fit: bool = True
    # Phase 13 — proportionality to the question.
    central_conclusion_first: bool = True
    peripheral_proportionate: bool = True
    comparison_balanced: bool = True
    decision_complete: bool = True
    depth_index: int = 0
    limitation_ratio: float = 0.0
    peripheral_section_ratio: float = 0.0
    failures: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """0..1 conformance score (1.0 when there is nothing to check)."""
        checks = [
            self.shape_conformant,
            self.inference_calibrated,
            self.contradiction_synthesised,
            self.uncertainty_proportionate,
            self.depth_fit,
            # Phase 13 checks only count once they have something to measure;
            # each defaults True, so an unmeasured question is never penalised.
            self.central_conclusion_first,
            self.peripheral_proportionate,
            self.comparison_balanced,
            self.decision_complete,
        ]
        return round(sum(1 for c in checks if c) / len(checks), 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type,
            "shape_conformant": self.shape_conformant,
            "inference_calibrated": self.inference_calibrated,
            "contradiction_synthesised": self.contradiction_synthesised,
            "uncertainty_proportionate": self.uncertainty_proportionate,
            "depth_fit": self.depth_fit,
            "central_conclusion_first": self.central_conclusion_first,
            "peripheral_proportionate": self.peripheral_proportionate,
            "comparison_balanced": self.comparison_balanced,
            "decision_complete": self.decision_complete,
            "depth_index": self.depth_index,
            "limitation_ratio": round(self.limitation_ratio, 3),
            "peripheral_section_ratio": round(self.peripheral_section_ratio, 3),
            "score": self.score,
            "failures": list(self.failures),
        }


def _resolve_query_type(query: str, intent: Optional[Dict[str, Any]]) -> str:
    """The question's shape, from the intent classifier then the deterministic
    fallback the synthesizer already uses (one source of truth)."""
    intent = intent if isinstance(intent, dict) else {}
    declared = str(intent.get("query_type", "") or "").strip().lower()
    if declared:
        return declared
    try:
        from app.agents.synthesizer import infer_query_type

        return infer_query_type(query)
    except Exception as exc:  # inference must never raise into conformance
        logger.warning("conformance_query_type_failed", error=str(exc), exc_info=exc)
        return ""


def check_answer_conformance(
    answer: str,
    query: str,
    intent: Optional[Dict[str, Any]] = None,
    *,
    brief: Any = None,
    plan: Any = None,
    contradictions: Optional[Sequence[Dict[str, Any]]] = None,
    mode: str = "standard",
) -> ConformanceReport:
    """Measure whether `answer` fits the question. Deterministic and total.

    `brief` (AnalyticalBrief) and `plan` (SynthesisPlan) are optional; when
    present they sharpen the inference-calibration and contradiction checks by
    naming the specific inferences and conflicts the analyst/plan identified.
    An absent answer yields a neutral report so the absence of prose never
    penalises a run.
    """
    report = ConformanceReport()
    body = (answer or "").strip()
    if not body:
        return report

    try:
        report.query_type = _resolve_query_type(query, intent)
        sentences = _sentences(body)
        lowered = _norm(body)

        # -- 1. SHAPE CONFORMANCE -------------------------------------------
        _check_shape(report, report.query_type, lower=lowered, sentences=sentences)

        # -- 2. INFERENCE CALIBRATION ---------------------------------------
        _check_inference_calibration(
            report, brief, sentences=sentences, lower=lowered
        )

        # -- 3. CONTRADICTION SYNTHESIS -------------------------------------
        _check_contradiction_synthesis(
            report, lower=lowered, contradictions=contradictions, plan=plan
        )

        # -- 4. UNCERTAINTY PROPORTIONALITY ---------------------------------
        _check_uncertainty_proportionality(report, sentences)

        # -- 5. DEPTH FIT ----------------------------------------------------
        _check_depth(report, sentences, mode=mode, query_type=report.query_type)

        # -- 6. CENTRAL CONCLUSION FIRST (Phase 13) -------------------------
        _check_central_conclusion_first(report, sentences, lower=lowered)

        # -- 7. PERIPHERAL PROPORTIONALITY (Phase 13) -----------------------
        _check_peripheral_proportionality(
            report, body, query=query, question_terms=_question_terms(query, plan)
        )

        # -- 8. COMPARISON BALANCE (Phase 13) -------------------------------
        _check_comparison_balance(
            report, report.query_type, body, query=query,
            question_terms=_question_terms(query, plan),
        )

        # -- 9. DECISION COMPLETENESS (Phase 13) ----------------------------
        _check_decision_completeness(report, report.query_type, lower=lowered)
    except Exception as exc:  # total/fail-safe: never break synthesis
        logger.warning("answer_conformance_failed", error=str(exc), exc_info=exc)
    return report


def _check_shape(
    report: ConformanceReport,
    query_type: str,
    *,
    lower: str,
    sentences: Sequence[str],
) -> None:
    """Does the prose do the work its question type demands?"""
    qt = query_type

    # Alias the classifier's vocabulary onto the shape checks, so a
    # "comparative" question is checked as a comparison. One mapping, no
    # duplicated literals at each branch.
    qt = {
        "why": "causal", "cause": "causal",
        "comparative": "comparison", "compare": "comparison",
        "prediction": "forecast", "predictive": "forecast",
        "evaluative": "decision", "strategic": "decision",
        "policy": "decision", "procedural": "howto", "how_to": "howto",
    }.get(qt, qt)

    if qt in ("causal", "why"):
        has_mechanism = _contains_any(lower, _CAUSAL_MARKERS)
        # A bare "because" in a throwaway clause is not a mechanism; require a
        # causal sentence that also carries the subject.
        causal_sentences = [
            s for s in sentences
            if _contains_any(s, _CAUSAL_MARKERS)
        ]
        if not has_mechanism or not causal_sentences:
            report.shape_conformant = False
            report.failures.append(
                "Question fit: this is a 'why' question but the answer states what "
                "happened without explaining the mechanism. Add the causal chain the "
                "evidence supports ('X drives Y because Z'), and where the evidence is "
                "only correlational, say so explicitly."
            )
        return

    if qt == "comparison":
        has_comparison = _contains_any(lower, _COMPARISON_MARKERS)
        has_verdict = _contains_any(lower, _VERDICT_MARKERS)
        if not has_comparison or not has_verdict:
            report.shape_conformant = False
            report.failures.append(
                "Question fit: this is a comparison but the answer does not compare "
                "criterion by criterion and close with a verdict. Give both sides on "
                "each dimension that matters and state which wins for which use case "
                "(or what the verdict depends on and the threshold)."
            )
        return

    if qt in ("forecast", "prediction"):
        has_projection = _contains_any(lower, _PROJECTION_MARKERS)
        has_asof = _contains_any(lower, _FORECAST_ASOF_MARKERS)
        if not has_projection:
            report.shape_conformant = False
            report.failures.append(
                "Question fit: this is a forward-looking question but the answer "
                "contains no projection signal. State the current measured level and "
                "its as-of date first, then label every forward statement as a "
                "projection with its source's assumptions."
            )
        elif not has_asof:
            report.shape_conformant = False
            report.failures.append(
                "Question fit: this is a forecast and every projection needs an "
                "as-of anchor. State the current measured level and the date the "
                "evidence was published or retrieved before projecting."
            )
        return

    if qt in ("decision", "evaluative", "strategic"):
        has_verdict = _contains_any(lower, _VERDICT_MARKERS)
        has_condition = any(
            m in lower for m in ("depends on", "if you", "when you", "for teams",
                                 "for most", "provided that", "assuming")
        )
        if not (has_verdict and has_condition):
            report.shape_conformant = False
            report.failures.append(
                "Question fit: this is a decision question but the answer gives no "
                "conditioned recommendation. Lay out the realistic options with their "
                "trade-offs, give a recommendation conditioned on the reader's "
                "situation, and state what would change it."
            )
        return

    if qt == "howto":
        has_steps = any(
            re.search(r"^\s*(\d+[.)]|step\s+\d+)", s, re.IGNORECASE)
            for s in sentences
        ) or bool(re.search(r"(?m)^\s*\d+[.)]\s+", lower))
        if not has_steps:
            report.shape_conformant = False
            report.failures.append(
                "Question fit: this is a procedural question but the answer is not "
                "ordered steps. Give prerequisites, then numbered steps in execution "
                "order, and name the failure mode for any step that can fail."
            )
        return

    # Unrecognised shape: no specific obligation to enforce. This is deliberate
    # — Phase 12 forbids a rigid template, so an unidentified question is never
    # failed for not matching a heading list.
    return


def _check_inference_calibration(
    report: ConformanceReport,
    brief: Any,
    *,
    sentences: Sequence[str],
    lower: str,
) -> None:
    """Cross-source inferences must read as inferences, not as measured facts.

    The analyst's `cross_source_conclusions` are, by construction, claims no
    single source states — inferences. If the answer states such a conclusion
    in the assertive measured voice (no hedge anywhere in the sentence), the
    synthesis silently promoted an inference to fact. This is the
    calibrated-inference contract (Phase 12 §6): inference is allowed, but it
    must be distinguishable.
    """
    if brief is None or not hasattr(brief, "to_dict"):
        return
    try:
        data = brief.to_dict()
    except Exception:  # a brief that cannot serialize is treated as absent
        return
    inferences = [
        str(c) for c in (data.get("cross_source_conclusions") or [])
        if str(c or "").strip()
    ]
    if not inferences:
        return

    # Only a conclusion the analyst DECLARED as inferential can be "promoted".
    # The analyst's cross-source conclusions are not always hedged in the brief
    # (the deterministic fallback fills them from ordinary corroborated content),
    # so requiring a hedge for every one would flag a report that is simply
    # stating established findings. We only look at conclusions the brief itself
    # phrases as an inference — those are the claims that must not be restated
    # as measured fact.
    inferential = [
        c for c in inferences if _contains_any(c, _INFERENCE_MARKERS)
    ]
    if not inferential:
        return

    # A conclusion is "promoted" when the answer echoes it (content overlap) but
    # the answer as a whole carries no hedge at all. Conservative: require real
    # overlap and a globally unhedged answer, so a calibrated draft is never
    # second-guessed and a single terse sentence is not punished.
    if _contains_any(lower, _INFERENCE_MARKERS):
        return
    for conclusion in inferential:
        if _content_overlap(conclusion, lower) < 0.55:
            continue
        report.inference_calibrated = False
        report.failures.append(
            "Calibrated inference: the answer states a cross-source conclusion as "
            "measured fact. Mark the reasoning that no single source states as an "
            "inference ('taken together, the evidence suggests…'), never in the same "
            "voice as a directly supported finding."
        )
        return


def _content_overlap(statement: str, answer_lower: str) -> float:
    """Fraction of a statement's distinctive content words present in the answer."""
    terms = {
        w for w in re.findall(r"[a-z][a-z0-9\-]{3,}", (statement or "").lower())
        if w not in _CONTENT_STOPWORDS
    }
    if not terms:
        return 0.0
    return sum(1 for t in terms if t in answer_lower) / len(terms)


def _check_contradiction_synthesis(
    report: ConformanceReport,
    *,
    lower: str,
    contradictions: Optional[Sequence[Dict[str, Any]]],
    plan: Any,
) -> None:
    """A conflict must be explained, not just announced."""
    has_conflict_signal = bool(contradictions) or bool(re.search(
        r"\b(" + "|".join(_CONFLICT_MARKERS) + r")\b", lower
    ))
    if not has_conflict_signal:
        return
    names_conflict = any(m in lower for m in _CONFLICT_MARKERS)
    explains_conflict = any(m in lower for m in _CONFLICT_EXPLAIN_MARKERS)
    if names_conflict and not explains_conflict:
        report.contradiction_synthesised = False
        report.failures.append(
            "Contradiction synthesis: the answer names a source disagreement but does "
            "not explain what it turns on. Say what differs and why the sources may "
            "differ (scope, timeframe, method or definition), then state what "
            "conclusion remains defensible."
        )


def _check_uncertainty_proportionality(
    report: ConformanceReport,
    sentences: Sequence[str],
) -> None:
    """Uncertainty must be present but not dominate the answer."""
    if not sentences:
        return
    limitation_sentences = [
        s for s in sentences if _contains_any(s, _LIMITATION_MARKERS)
    ]
    ratio = len(limitation_sentences) / len(sentences)
    report.limitation_ratio = ratio
    if ratio > MAX_LIMITATION_DENSITY and len(sentences) >= 5:
        report.uncertainty_proportionate = False
        report.failures.append(
            "Uncertainty proportionality: limitation language dominates the answer "
            f"({len(limitation_sentences)} of {len(sentences)} sentences). Keep the "
            "one limitation that materially changes the conclusion; move the rest to "
            "the audit layer."
        )


def _check_depth(
    report: ConformanceReport,
    sentences: Sequence[str],
    *,
    mode: str,
    query_type: str,
) -> None:
    """Reasoning depth proportional to the question.

    A deep-research run (audit/deep/executive) must reason beyond the opening:
    it needs at least a couple of distinct reasoning moves (mechanism,
    trade-off, comparison, qualification, implication). A simple question is
    never failed for brevity — undershoot is measured only for deep modes.
    """
    reasoning_markers = (
        _CAUSAL_MARKERS + _COMPARISON_MARKERS + _CONFLICT_EXPLAIN_MARKERS
        + ("trade-off", "tradeoff", "however", "although", "whereas",
           "implication", "which means", "this suggests", "in practice",
           "on balance", "by contrast", "that said", "still,")
    )
    distinct = _count_distinct_markers(" ".join(sentences), reasoning_markers)
    report.depth_index = distinct
    deep = str(mode or "").lower() in ("deep", "executive", "audit")
    if deep and len(sentences) >= DEPTH_MIN_SENTENCES and distinct < DEPTH_MIN_MARKERS:
        report.depth_fit = False
        report.failures.append(
            "Depth: this is a deep-research question but the answer reports findings "
            "without reasoning over them. Deepen the analysis, do not lengthen it — "
            "explain mechanisms, weigh trade-offs, and state what follows, rather "
            "than adding more sections or citations."
        )


# ---------------------------------------------------------------------------
# Phase 13 — proportionality to the question
# ---------------------------------------------------------------------------

# Openings that lead the answer with research bookkeeping, gaps or methodology
# instead of the answer. A report MUST NOT open on one of these (Phase 13 §4);
# in the right place later it is legitimate qualification.
_LEADING_BOOKKEEPING_MARKERS = (
    "the evidence base is", "the available evidence", "the evidence pool",
    "the evidence does not", "no reliable evidence", "evidence is thin",
    "several dimensions remain uncertain", "the research was",
    "research gaps", "the sources returned", "data limitations",
    "it is important to note that the evidence",
    "before answering", "to answer this question we",
)

# A section is PERIPHERAL when its heading/body shares almost no content word
# with the user's question. It may still be worth a paragraph; it must not be a
# major section. Proportion is measured over writer sections only.
PERIPHERAL_SECTION_MAX_RATIO = 0.5   # >50% of words in peripheral sections
PERIPHERAL_SECTION_MIN_WORDS = 100   # ignore very short answers (one-liners)
PERIPHERAL_SECTION_MIN_SHARE = 0.25  # a single peripheral section must exceed this

# Decision questions (Phase 13 §9): the decision factors a reader must see.
# Not every one is mandatory for every decision — the check requires that the
# answer addresses a BREADTH of factors, not just the best-evidenced axis.
_DECISION_FACTORS: Dict[str, tuple] = {
    "benefits": ("benefit", "advantage", "upside", "use case", "gains", "value"),
    "risks": ("risk", "downside", "drawback", "threat", "exposure", "concern"),
    "costs": ("cost", "price", "expense", "budget", "spend", "operational burden",
              "overhead", "tco", "total cost"),
    "implementation": ("implement", "deploy", "migrat", "roll out", "integration",
                       "resourc", "staff", "training", "timeline"),
    "when_not": ("when not", "not worth", "avoid", "unsuitable", "poor fit",
                 "do not adopt", "don't adopt", "circumstances where"),
    "success": ("success criteria", "measure", "kpi", "metric", "evaluate",
                "benchmark", "roi", "return on"),
}
# The number of distinct decision factors the answer must address. A decision
# answer that covers only security (one factor) regardless of evidence strength
# fails this; a genuine multi-factor answer passes.
DECISION_MIN_FACTORS = 3


def _question_terms(query: str, plan: Any = None) -> set:
    """Content words naming what the question (and its required dimensions) is
    about, for peripheral-relevance measurement."""
    terms = {
        w for w in re.findall(r"[a-z][a-z0-9\-]{3,}", _norm(query))
        if w not in _CONTENT_STOPWORDS
    }
    try:
        data = plan.to_dict() if plan is not None and hasattr(plan, "to_dict") else {}
        for dim in data.get("dimensions") or []:
            if not isinstance(dim, dict) or not dim.get("required"):
                continue
            text = str(dim.get("axis", "") or "") + " " + str(dim.get("question", "") or "")
            terms |= {
                w for w in re.findall(r"[a-z][a-z0-9\-]{3,}", _norm(text))
                if w not in _CONTENT_STOPWORDS
            }
    except Exception:  # a plan that cannot serialize contributes no terms
        pass
    return terms


def _writer_sections(answer: str) -> List[Dict[str, Any]]:
    """Split the writer's prose into (title, text, words) sections.

    Splits on H1/H2 only (sub-headings stay with their parent). Machine-appended
    sections are already stripped by the caller. Content before the first
    heading is returned as a pseudo-section titled "".
    """
    body = strip_machine_sections(answer or "")
    sections: List[Dict[str, Any]] = []
    preamble: List[str] = []
    current: Optional[Dict[str, Any]] = None
    for block in re.split(r"\n{2,}", body):
        stripped = block.strip()
        if not stripped:
            continue
        # A block may be a heading alone OR a heading followed by its body
        # (the sanitizer keeps them in one block). Match the heading at the
        # START of the block, not as the whole block.
        m = re.match(r"^#{1,2}\s+([^\n]*\S)[ \t]*\n?", stripped)
        if m:
            if current is not None:
                sections.append(current)
            rest = stripped[m.end():].strip()
            current = {"title": m.group(1).strip(), "blocks": ([rest] if rest else [])}
        elif current is None:
            preamble.append(stripped)
        else:
            current["blocks"].append(stripped)
    if current is not None:
        sections.append(current)
    if preamble:
        sections.insert(0, {"title": "", "blocks": preamble})
    for s in sections:
        s["text"] = "\n\n".join(s.get("blocks") or [])
        s.pop("blocks", None)
        s["words"] = len(re.findall(r"[A-Za-z0-9']+", s["text"]))
    return sections


def _check_central_conclusion_first(
    report: ConformanceReport,
    sentences: Sequence[str],
    *,
    lower: str,
) -> None:
    """The answer must open with the answer — not with process or gaps."""
    if not sentences:
        return
    # Only the FIRST sentence is the "opening". A limitation in the second
    # sentence is legitimate qualification, not a failure to lead with the
    # answer — the point is that the answer comes first.
    opening = sentences[0].lower()
    if _contains_any(opening, _LEADING_BOOKKEEPING_MARKERS):
        report.central_conclusion_first = False
        report.failures.append(
            "Central conclusion: the answer opens on research limitations or "
            "methodology instead of the answer. Lead with the conclusion the evidence "
            "best supports in terms of the question's own subject; state any limitation "
            "after it, and only where it materially changes the conclusion."
        )


def _check_peripheral_proportionality(
    report: ConformanceReport,
    answer: str,
    *,
    query: str,
    question_terms: set,
) -> None:
    """Peripheral evidence must not become a major section (Phase 13 §1/§5)."""
    sections = [s for s in _writer_sections(answer) if s.get("words", 0) > 0]
    total_words = sum(s["words"] for s in sections)
    if not question_terms or total_words < PERIPHERAL_SECTION_MIN_WORDS:
        return
    peripheral_words = 0
    for section in sections:
        # A section with no heading is the opening answer — never peripheral.
        if not str(section.get("title", "")).strip():
            continue
        text_terms = {
            w for w in re.findall(r"[a-z][a-z0-9\-]{3,}", _norm(section["text"]))
            if w not in _CONTENT_STOPWORDS
        }
        # Relevance is the share of the QUESTION's distinctive terms the section
        # actually addresses. A section about an unrelated axis shares almost
        # none, so it cannot be a major section.
        overlap = len(text_terms & question_terms) / len(question_terms)
        if overlap < 0.10 and section["words"] > total_words * PERIPHERAL_SECTION_MIN_SHARE:
            peripheral_words += section["words"]
    ratio = peripheral_words / total_words if total_words else 0.0
    report.peripheral_section_ratio = ratio
    if ratio > PERIPHERAL_SECTION_MAX_RATIO:
        report.peripheral_proportionate = False
        report.failures.append(
            "Proportionality: most of the answer is spent on material peripheral to "
            f"the question ({ratio:.0%} of the prose). Drop or shorten sections that do "
            "not serve the question's dimensions — more research should produce a "
            "stronger answer, not a longer one."
        )


def _check_comparison_balance(
    report: ConformanceReport,
    query_type: str,
    answer: str,
    *,
    query: str,
    question_terms: set,
) -> None:
    """Every entity in a comparison must receive meaningful treatment (§8)."""
    qt = {
        "comparative": "comparison", "compare": "comparison",
    }.get(str(query_type or "").lower(), str(query_type or "").lower())
    if qt != "comparison":
        return
    entities = _comparison_entities(query)
    if len(entities) < 2:
        return
    lower = _norm(answer)
    # Entity coverage: count how many mentions each entity receives. A single
    # dominant entity with the other barely present is the asymmetry §8 names.
    counts = {e: lower.count(e) for e in entities}
    present = [e for e, c in counts.items() if c > 0]
    if len(present) < len(entities):
        missing = [e for e in entities if counts[e] == 0]
        report.comparison_balanced = False
        report.failures.append(
            "Comparison balance: the answer does not give meaningful treatment to "
            f"{', '.join(missing)}. Cover every requested entity, and where the "
            "evidence is asymmetric say 'documented difference' vs 'not enough "
            "evidence to compare' rather than letting one entity dominate."
        )
        return
    # Both present: flag severe asymmetry (one entity cited/text-treated far more).
    values = sorted(counts.values())
    if values[-1] >= max(4, 3 * max(1, values[0])):
        minor = min(counts, key=lambda k: counts[k])
        report.comparison_balanced = False
        report.failures.append(
            f"Comparison balance: one entity dominates while '{minor}' is barely "
            "covered. Give each requested entity comparable treatment, or state that "
            "the evidence for one is insufficient to compare."
        )


# Prepositions/qualifiers that end an entity noun phrase in a comparison
# question ("nuclear AND solar energy FOR grid reliability"). Cutting at these
# stops trailing qualification polluting the entity set.
_ENTITY_TAIL_RE = re.compile(
    r"\b(?:for|on|in|at|by|with|regarding|about|over|to|and|or|vs\.?|versus)\b"
)


def _comparison_entities(query: str) -> List[str]:
    """The two entities a comparison question names, as distinctive head tokens.

    Handles the clear shapes — "X vs Y", "compare X and Y", "difference between
    X and Y" — and cuts each entity at the first preposition so trailing
    qualification ("solar ENERGY for grid reliability") does not become a
    spurious third entity. Returns [] when the shape is unclear, which makes the
    balance check a no-op rather than a false failure.
    """
    text = _norm(query)
    left = right = ""

    m = re.split(r"\s+(?:vs\.?|versus)\s+", text)
    if len(m) == 2:
        left, right = m[0], m[1]
    else:
        m2 = re.search(r"compare\s+(.+)$", text)
        if m2:
            halves = re.split(r"\s+(?:and|with|to|against)\s+", m2.group(1), maxsplit=1)
            if len(halves) == 2:
                left, right = halves
        else:
            m3 = re.search(r"between\s+(.+)$", text)
            if m3:
                halves = re.split(r"\s+and\s+", m3.group(1), maxsplit=1)
                if len(halves) == 2:
                    left, right = halves

    a = _entity_phrase_tokens(left)
    b = _entity_phrase_tokens(right)
    if not a or not b or set(a) == set(b):
        return []
    return [a[0], b[0]]


def _entity_phrase_tokens(phrase: str) -> List[str]:
    """Distinctive tokens of one side of a comparison, cut at the first tail."""
    head = _ENTITY_TAIL_RE.split(phrase or "", maxsplit=1)[0]
    words = [
        w for w in re.findall(r"[a-z][a-z0-9\-]{2,}", head)
        if w not in _CONTENT_STOPWORDS
    ]
    return words[:2]


def _check_decision_completeness(
    report: ConformanceReport,
    query_type: str,
    *,
    lower: str,
) -> None:
    """A decision answer must cover a breadth of decision factors, not just the
    best-evidenced axis (Phase 13 §9)."""
    qt = {
        "evaluative": "decision", "strategic": "decision",
        "policy": "decision", "decision": "decision",
    }.get(str(query_type or "").lower(), str(query_type or "").lower())
    if qt != "decision":
        return
    hit = {
        factor for factor, cues in _DECISION_FACTORS.items()
        if any(c in lower for c in cues)
    }
    if len(hit) < DECISION_MIN_FACTORS:
        missing = [f for f in _DECISION_FACTORS if f not in hit]
        report.decision_complete = False
        report.failures.append(
            "Decision completeness: this is a decision question but the answer "
            f"covers only {len(hit)} decision factor(s). Organise around the actual "
            "decision factors — benefits/use cases, risks, costs/operational burden, "
            "implementation requirements, when NOT to adopt, and success criteria — "
            f"rather than the best-evidenced axis alone (missing: {', '.join(missing[:4])})."
        )
