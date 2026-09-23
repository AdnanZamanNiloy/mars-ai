"""Evidence Synthesis Planning — reason over the evidence landscape BEFORE writing.

Why this module exists
----------------------
The pipeline grades every claim, detects conflicts, tracks investigation gaps and
builds an argument structure (`reasoning_engine`), but those artifacts were
computed *separately* and concatenated into the writer prompt as flat blocks.
Nothing ever answered the questions a human analyst answers before writing:

  1. What is the user actually asking?
  2. Which dimensions does a good answer need?
  3. Which findings are dominant — worth leading on?
  4. Which findings are merely incidental?
  5. What evidence supports each important dimension?
  6. Which important dimensions are under-researched?
  7. Which findings conflict?
  8. What can legitimately be synthesised across sources?
  9. What is directly established versus inferred?
 10. What should the answer prioritise?
 11. What can safely be omitted?
 12. What structure best communicates the synthesis?

This module produces a single, explicit `SynthesisPlan` artifact that answers all
twelve, deterministically and without an LLM. It is a *reasoning* layer over the
evidence landscape, not a summary of retrieved evidence: it ranks findings by how
central they are to the question, separates the dominant few from the incidental
many, names the dimensions that are thin relative to their importance, and picks
the presentation strategy.

Design (AGENTS.md)
------------------
* Deterministic and LLM-free: stable ordering, normalized-text tie-breaks, so two
  runs on the same evidence produce the same plan. No degradation risk, no
  latency.
* Reuses — never reimplements — `evidence_grade` (grading/impact),
  `evidence_completion` (claim impact), `reasoning_engine` (conclusions,
  competing explanations, established/inferred), `outline` (dimensions,
  strategy) and `depth_controller` (thin/uncovered dimensions).
* Total/fail-safe: garbage or empty input yields an empty, well-formed plan,
  never an exception.
* No new facts: every ranking references the claim/evidence it rests on; the
  module never invents a finding, cause or number.
* No module-level mutable state: the plan is per-call and returned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.logging import get_logger

logger = get_logger(__name__)

# Cap the dominant set. "Dominant" means the handful of findings the answer must
# lead on — a plan that names fifteen dominant findings has ranked nothing.
MAX_DOMINANT = 5
# Incidental findings are noted only so the writer may omit them safely; a
# bounded sample is enough for that judgement and keeps the prompt small.
MAX_INCIDENTAL = 6
# Dimensions the plan will name as under-researched before it stops being a
# priority signal and becomes noise.
MAX_UNDER_RESEARCHED = 4

# A dimension is "under-researched relative to its importance" when its highest
# finding centrality is at or above this bar but it carries fewer than
# `THIN_DIMENSION_MIN_FACTS` verified facts. Centrality is the dominant-score
# scale, so 2.0 means "at least a summary-used or quantitative finding".
DIMENSION_IMPORTANT_CENTRALITY = 2.0
THIN_DIMENSION_MIN_FACTS = 2

# The bar a finding must clear to be "dominant". It is the summary-use weight
# from `evidence_completion` (a finding worth a headline), or a quantitative
# claim (numbers need corroboration before they lead), or an A-grade claim with
# any corroboration. A finding below it is available but not something the
# answer must lead on.
DOMINANT_CENTRALITY_BAR = 3

# Relevance multipliers applied to a finding's absolute importance. A finding
# answer-worthy on its own merits keeps full weight when it relates to the
# question; one related only to a needed dimension is scaled down; one related
# to neither is scaled hard. Deliberately > 0 so corroborated peripheral facts
# remain available (they just cannot lead), and so the multiplier never zeroes
# out an entire pool.
PERIPHERAL_DIMENSION_PENALTY = 0.6
PERIPHERAL_UNRELATED_PENALTY = 0.25

# Words too common to signal topical relevance. Deliberately small: this is a
# relevance gate, not a language model.
_RELEVANCE_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "as", "by", "at", "is", "are", "was", "were", "be", "been", "being", "that",
    "this", "these", "those", "it", "its", "from", "into", "than", "then",
    "has", "have", "had", "will", "would", "can", "could", "should", "may",
    "might", "not", "no", "so", "such", "more", "most", "less", "least",
    "which", "who", "whom", "whose", "what", "when", "where", "why", "how",
    "current", "currently", "latest", "recent", "new", "state", "trend",
    "trends", "report", "reports", "said", "says", "according", "study",
    "studies", "data", "figure", "figures", "percent", "million", "billion",
}


def _words(text: str) -> set:
    """Content words (no stopwords) from a text, for relevance overlap."""
    import re as _re

    return {
        w for w in _re.findall(r"[a-z][a-z0-9\-]{2,}", _norm(text))
        if w not in _RELEVANCE_STOPWORDS
    }


def _query_terms(query: str) -> set:
    """Content words from the user's question."""
    return _words(query)


def _dimension_terms(outline: Any, sub_questions: Optional[Sequence[Any]]) -> set:
    """Content words naming the dimensions the answer must cover."""
    terms: set = set()
    if outline is not None:
        for section in getattr(outline, "sections", None) or []:
            terms |= _words(str(getattr(section, "title", "") or ""))
            terms |= _words(str(getattr(section, "question", "") or ""))
            terms |= _words(str(getattr(section, "axis", "") or ""))
    for item in sub_questions or []:
        if isinstance(item, dict):
            terms |= _words(str(item.get("question", "") or ""))
            terms |= _words(str(item.get("axis", "") or ""))
    return terms



# Grade weights reused for ordering (A best). Kept local and identical to the
# scales in reasoning_engine / evidence_grade so nothing disagrees.
_GRADE_WEIGHT = {"A": 3, "B": 2, "C": 1, "D": 0}


def _norm(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


@dataclass
class Finding:
    """One finding, ranked by how central it is to answering the question."""

    claim: str
    centrality: int = 0
    grade: str = ""
    sub_question: str = ""
    axes: List[str] = field(default_factory=list)
    has_numbers: bool = False
    corroboration: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "centrality": self.centrality,
            "grade": self.grade,
            "sub_question": self.sub_question,
            "axes": list(self.axes),
            "has_numbers": self.has_numbers,
            "corroboration": self.corroboration,
        }


@dataclass
class PlannedDimension:
    """A dimension the answer needs, with the evidence it has and its holes."""

    axis: str
    title: str = ""
    question: str = ""
    fact_count: int = 0
    top_centrality: int = 0
    sources: int = 0
    has_evidence: bool = False
    under_researched: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axis": self.axis,
            "title": self.title,
            "question": self.question,
            "fact_count": self.fact_count,
            "top_centrality": self.top_centrality,
            "sources": self.sources,
            "has_evidence": self.has_evidence,
            "under_researched": self.under_researched,
        }


@dataclass
class SynthesisPlan:
    """The explicit plan for synthesising the final answer.

    The twelve planning questions map onto the fields:

      * what is asked        -> `central_question`, `query_type`
      * needed dimensions    -> `dimensions`
      * dominant findings    -> `dominant`
      * incidental findings  -> `incidental`
      * evidence per dim     -> `PlannedDimension.fact_count` / `sources`
      * under-researched     -> `PlannedDimension.under_researched`, `under_researched`
      * conflicts            -> `conflicts`
      * synthesised concls   -> `synthesized_conclusions`
      * established/inferred -> `established`, `inferred`, `unknown`
      * what to prioritise   -> `dominant` ordering, `priority_note`
      * what to omit         -> `incidental`, `omit_note`
      * structure            -> `strategy`, `themes`
    """

    query: str = ""
    central_question: str = ""
    query_type: str = ""
    strategy: str = "general"
    themes: List[str] = field(default_factory=list)
    dominant: List[Finding] = field(default_factory=list)
    incidental: List[Finding] = field(default_factory=list)
    dimensions: List[PlannedDimension] = field(default_factory=list)
    under_researched: List[str] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    synthesized_conclusions: List[str] = field(default_factory=list)
    established: List[str] = field(default_factory=list)
    inferred: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)
    priority_note: str = ""
    omit_note: str = ""

    @property
    def is_empty(self) -> bool:
        return not (
            self.dominant or self.dimensions or self.conflicts
            or self.synthesized_conclusions or self.under_researched
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "central_question": self.central_question,
            "query_type": self.query_type,
            "strategy": self.strategy,
            "themes": list(self.themes),
            "dominant": [f.to_dict() for f in self.dominant],
            "incidental": [f.to_dict() for f in self.incidental],
            "dimensions": [d.to_dict() for d in self.dimensions],
            "under_researched": list(self.under_researched),
            "conflicts": [dict(c) for c in self.conflicts],
            "synthesized_conclusions": list(self.synthesized_conclusions),
            "established": list(self.established),
            "inferred": list(self.inferred),
            "unknown": list(self.unknown),
            "priority_note": self.priority_note,
            "omit_note": self.omit_note,
        }

    def render_for_writer(self) -> str:
        """Compact planning brief for the synthesis prompt (deterministic).

        This is guidance for the writer's *reasoning*, not headings: it names the
        dominant findings to lead on, the incidental ones that may be dropped,
        the conflicts to present, and the dimensions still thin. Returns "" for
        an empty plan so callers can append unconditionally.
        """
        if self.is_empty and not (self.established or self.inferred or self.unknown):
            return ""
        lines: List[str] = [
            "EVIDENCE SYNTHESIS PLAN (measured from the evidence landscape — this is "
            "how to think about the answer, not a heading list):"
        ]

        if self.central_question:
            lines.append("")
            lines.append(f"THE QUESTION BEING ANSWERED: {self.central_question}")

        if self.dominant:
            lines.append("")
            lines.append(
                "DOMINANT FINDINGS (lead with these — they are the most central to "
                "the question; state the conclusion they jointly support):"
            )
            for f in self.dominant:
                axes = f", axes: {', '.join(f.axes)}" if f.axes else ""
                lines.append(
                    f"- {f.claim} [grade {f.grade or '?'}, centrality {f.centrality}{axes}]"
                )

        if self.incidental:
            lines.append("")
            lines.append(
                "INCIDENTAL FINDINGS (lower centrality — include only if they support "
                "a dominant point; omit rather than enumerate):"
            )
            for f in self.incidental:
                lines.append(f"- {f.claim} [centrality {f.centrality}]")

        if self.synthesized_conclusions:
            lines.append("")
            lines.append(
                "LEGITIMATE CROSS-SOURCE SYNTHESES (derived from multiple cited "
                "findings above — state as synthesis, in natural prose, not as a "
                "verbatim source quote):"
            )
            lines.extend(f"- {c}" for c in self.synthesized_conclusions)

        if self.conflicts:
            lines.append("")
            lines.append(
                "CONFLICTS TO PRESENT (do not average or silently pick a side):"
            )
            for c in self.conflicts:
                a = str(c.get("position_a", "") or c.get("claim_a", ""))
                b = str(c.get("position_b", "") or c.get("claim_b", ""))
                kind = str(c.get("kind", "") or "unspecified")
                tag = "resolved" if c.get("resolved") else "open"
                lines.append(f"- [{tag} {kind}] \"{a}\" vs \"{b}\"")

        if self.under_researched:
            lines.append("")
            lines.append(
                "UNDER-RESEARCHED DIMENSIONS (important but thin — acknowledge the "
                "limitation rather than overstating):"
            )
            lines.extend(f"- {d}" for d in self.under_researched)

        if self.established or self.inferred or self.unknown:
            lines.append("")
            lines.append(
                "EPISTEMIC STATUS (keep these strictly separate in the prose):"
            )
            if self.established:
                lines.append(
                    "  ESTABLISHED (directly established — state as fact): "
                    + "; ".join(self.established[:4])
                )
            if self.inferred:
                lines.append(
                    "  INFERRED (reasonable inference, not directly measured — hedge): "
                    + "; ".join(self.inferred[:4])
                )
            if self.unknown:
                lines.append(
                    "  UNKNOWN (unresolved — say so, do not guess): "
                    + "; ".join(self.unknown[:4])
                )

        if self.priority_note:
            lines.append("")
            lines.append(self.priority_note)
        if self.omit_note:
            lines.append(self.omit_note)

        return "\n".join(lines)


def _finding_centrality(
    record: Dict[str, Any],
    summary_claims: Optional[set],
    query_terms: Optional[set] = None,
    dimension_terms: Optional[set] = None,
) -> int:
    """How central a graded record is to the ANSWER (not just to the report).

    Reuses `evidence_completion.claim_impact` (quantitative + summary-use +
    corroboration-need + primary-need minus contradictions), then adds the
    grade weight, then applies a RELEVANCE multiplier against the user's
    question and the identified answer dimensions.

    The multiplier is what stops a well-formed but peripheral fact — an old
    projection, an unrelated historical statistic, a side investment figure —
    from leading the answer purely because it was retrieved and is quantitative.
    Factuality alone must not make a finding answer-worthy. When neither the
    query nor the dimensions are available the multiplier is 1.0, so the
    pre-existing behaviour is preserved exactly.
    """
    try:
        from app.core.evidence_completion import claim_impact

        base = claim_impact(record, summary_claims=summary_claims)
    except Exception as exc:  # impact failure must never raise into planning
        logger.warning("synthesis_plan_impact_failed", error=str(exc), exc_info=exc)
        base = 0
    grade = str(record.get("grade", "D") or "D")
    absolute = base + _GRADE_WEIGHT.get(grade, 0)

    multiplier = _relevance_multiplier(record, query_terms, dimension_terms)
    return int(round(absolute * multiplier))


def _relevance_multiplier(
    record: Dict[str, Any],
    query_terms: Optional[set],
    dimension_terms: Optional[set],
) -> float:
    """Question/dimension relevance factor for a finding (deterministic).

    Overlap is measured over content words, so a finding that echoes the
    question's subject, or the subject of a dimension the answer needs, keeps
    full weight; a peripheral finding is scaled down (never to zero — a
    corroborated fact may still matter, it just must not lead).

      * related to the question           -> no penalty
      * related to a needed dimension only -> mild penalty
      * related to neither                -> strong penalty (peripheral)
    """
    if not query_terms and not dimension_terms:
        return 1.0
    claim = _norm(record.get("claim", ""))
    sub = _norm(record.get("sub_question", ""))
    axis = _norm(record.get("axis", ""))
    claim_words = _words(claim)

    def _overlaps(terms: Optional[set]) -> bool:
        if not terms:
            return False
        # A term counts when it appears in the claim text, or when the finding
        # is tagged with a sub-question/axis that shares the term.
        if any(t in claim_words for t in terms):
            return True
        tagged = f"{sub} {axis}"
        return any(t in tagged for t in terms)

    if _overlaps(query_terms):
        return 1.0
    if _overlaps(dimension_terms):
        return PERIPHERAL_DIMENSION_PENALTY
    return PERIPHERAL_UNRELATED_PENALTY


def _rank_findings(
    graded: Sequence[Dict[str, Any]],
    summary_claims: Optional[set],
    query_terms: Optional[set] = None,
    dimension_terms: Optional[set] = None,
) -> List[Finding]:
    """Rank every graded finding by centrality, highest first (stable).

    `graded` entries are the merged records: the evidence dict plus the
    top-level `source` / `axes` / `sub_question` fields the workflow keeps
    alongside it. `query_terms`/`dimension_terms` apply the relevance
    multiplier so a peripheral finding cannot lead on factuality alone.
    """
    findings: List[Finding] = []
    for record in graded or []:
        if not isinstance(record, dict):
            continue
        claim = str(record.get("claim", "") or "").strip()
        if not claim:
            continue
        axes = [str(a) for a in (record.get("axes") or []) if str(a).strip()]
        findings.append(
            Finding(
                claim=claim,
                centrality=_finding_centrality(
                    record, summary_claims, query_terms, dimension_terms
                ),
                grade=str(record.get("grade", "") or ""),
                sub_question=str(record.get("sub_question", "") or ""),
                axes=axes,
                has_numbers=bool(record.get("has_numbers")),
                corroboration=int(record.get("corroboration_count", 1) or 1),
            )
        )
    findings.sort(
        key=lambda f: (-f.centrality, -_GRADE_WEIGHT.get(f.grade, 0), _norm(f.claim))
    )
    return findings


def _split_by_centrality(
    ranked: Sequence[Finding],
) -> Tuple[List[Finding], List[Finding]]:
    """Split ranked findings into (dominant, incidental).

    A finding is dominant when its centrality clears `DOMINANT_CENTRALITY_BAR`;
    a pool with nothing above the bar treats its single best finding as dominant
    so the writer is never left without a lead. Everything below the bar is
    incidental, capped so the omission note stays useful.
    """
    dominant = [f for f in ranked if f.centrality >= DOMINANT_CENTRALITY_BAR][:MAX_DOMINANT]
    if not dominant and ranked:
        dominant = [ranked[0]]
    dominant_claims = {_norm(f.claim) for f in dominant}
    incidental = [
        f for f in ranked
        if _norm(f.claim) not in dominant_claims
    ][:MAX_INCIDENTAL]
    return dominant, incidental


def _plan_dimensions(
    outline: Any,
    findings: Sequence[Finding],
) -> Tuple[List[PlannedDimension], List[str]]:
    """Build the dimension plan and name the under-researched dimensions.

    A dimension is under-researched when its most central finding is important
    (centrality >= `DIMENSION_IMPORTANT_CENTRALITY`) but it carries fewer than
    `THIN_DIMENSION_MIN_FACTS` verified facts — i.e. the answer depends on it but
    the evidence is thin. An entirely empty planned dimension is always named.
    """
    if outline is None or not getattr(outline, "sections", None):
        return [], []
    # Map each finding to its section by the fact's own axis/sub_question so a
    # dimension's centrality is the best finding that belongs to it.
    centrality_by_sub: Dict[str, int] = {}
    centrality_by_axis: Dict[str, int] = {}
    for finding in findings:
        if finding.sub_question:
            key = _norm(finding.sub_question)
            centrality_by_sub[key] = max(
                centrality_by_sub.get(key, 0), finding.centrality
            )
        for axis in finding.axes:
            key = axis.lower()
            centrality_by_axis[key] = max(
                centrality_by_axis.get(key, 0), finding.centrality
            )

    under: List[str] = []
    planned: List[PlannedDimension] = []
    for section in outline.sections:
        axis = str(getattr(section, "axis", "") or "")
        title = str(getattr(section, "title", "") or "")
        question = str(getattr(section, "question", "") or "")
        section_facts = [
            f for f in (getattr(section, "facts", None) or []) if isinstance(f, dict)
        ]
        fact_count = len(section_facts)

        top_centrality = centrality_by_axis.get(axis.lower(), 0)
        sources: set = set()
        for fact in section_facts:
            sub = _norm(fact.get("sub_question", ""))
            if sub:
                top_centrality = max(
                    top_centrality, centrality_by_sub.get(sub, 0)
                )
            url = str(fact.get("source", "") or "").strip()
            if url:
                sources.add(url)

        empty = fact_count == 0
        thin = (
            top_centrality >= DIMENSION_IMPORTANT_CENTRALITY
            and fact_count < THIN_DIMENSION_MIN_FACTS
        )
        if empty or thin:
            # Prefer the dimension's own question over its display title so the
            # note maps back to a planned sub-question (and thus to a usable
            # follow-up query); fall back to the title, axis, or a generic label.
            label = question or title or axis or "an unnamed dimension"
            note = (
                f"'{label}' has no supporting evidence yet"
                if empty
                else f"'{label}' is thin ({fact_count} finding(s)) despite being central to the answer"
            )
            under.append(note)
        planned.append(
            PlannedDimension(
                axis=axis,
                title=title,
                question=question,
                fact_count=fact_count,
                top_centrality=top_centrality,
                sources=len(sources),
                has_evidence=not empty,
                under_researched=empty or thin,
            )
        )
    return planned, under[:MAX_UNDER_RESEARCHED]


def _cross_source_conclusions(reasoning: Any) -> List[str]:
    """Conclusions that are legitimately synthesised across sources.

    These are `reasoning_engine`'s convergent conclusions — claims jointly
    supported by independent evidence. They are the strongest licence to write
    "taken together, these findings suggest…". Deterministic and bounded.
    """
    if reasoning is None:
        return []
    conclusions = getattr(reasoning, "conclusions", None) or []
    out: List[str] = []
    for c in conclusions:
        claim = str(getattr(c, "claim", "") or "").strip()
        if not claim:
            continue
        domains = list(getattr(c, "domains", None) or [])
        if len(domains) >= 2:
            out.append(f"{claim} (converges across {', '.join(domains[:3])})")
    return out[:MAX_DOMINANT]


def build_synthesis_plan(
    facts: Sequence[Dict[str, Any]],
    contradictions: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    query: str = "",
    query_type: str = "",
    outline: Any = None,
    reasoning: Any = None,
    strategy: str = "",
    summary_claims: Optional[Sequence[Any]] = None,
    sub_questions: Optional[Sequence[Any]] = None,
) -> SynthesisPlan:
    """Build the evidence synthesis plan. Deterministic and total.

    `facts` are the run's usable facts. `outline` is an `AnswerOutline`
    (dimensions/structure), `reasoning` a `ReasoningMap` (conclusions, conflicts,
    epistemic status). Both are optional: when absent the plan degrades to what
    grading alone can rank, never raising.
    """
    plan = SynthesisPlan(
        query=str(query or ""),
        query_type=str(query_type or ""),
    )
    try:
        contradictions = [c for c in (contradictions or []) if isinstance(c, dict)]
        pool = [
            f for f in (facts or [])
            if isinstance(f, dict) and str(f.get("claim", "") or "").strip()
        ]

        summaries = {
            _norm(s) for s in (summary_claims or []) if str(s or "").strip()
        }

        graded: List[Dict[str, Any]] = []
        if pool:
            try:
                from app.core.evidence_grade import grade_facts

                raw = grade_facts(pool, contradictions=contradictions)
                for record in raw or []:
                    if not isinstance(record, dict):
                        continue
                    ev = record.get("evidence")
                    if not isinstance(ev, dict):
                        continue
                    merged = dict(ev)
                    # Top-level fields the grader keeps beside `evidence`:
                    # the source URL, the axis the fact belongs to and the
                    # sub-question it answers. Fall back to the original fact
                    # for a field the grader record omits.
                    merged["source"] = str(record.get("source", "") or "")
                    merged["axes"] = list(record.get("axes") or [])
                    merged["sub_question"] = str(record.get("sub_question", "") or "")
                    graded.append(merged)
            except Exception as exc:
                logger.warning("synthesis_plan_grading_failed", error=str(exc), exc_info=exc)

        q_terms = _query_terms(query)
        d_terms = _dimension_terms(outline, sub_questions)
        ranked = _rank_findings(graded, summaries, q_terms, d_terms)
        plan.dominant, plan.incidental = _split_by_centrality(ranked)

        plan.dimensions, plan.under_researched = _plan_dimensions(outline, ranked)

        # Conflicts: from the reasoning structure (competing explanations).
        competing = getattr(reasoning, "competing", None) or []
        for c in competing:
            plan.conflicts.append(
                c.to_dict() if hasattr(c, "to_dict") else dict(c)
            )

        # Established / inferred / unknown keep their meaning from reasoning.
        plan.established = [
            str(x) for x in (getattr(reasoning, "established", None) or []) if str(x).strip()
        ]
        plan.inferred = [
            str(x) for x in (getattr(reasoning, "inferred", None) or []) if str(x).strip()
        ]
        plan.unknown = [
            str(x) for x in (getattr(reasoning, "unknown", None) or []) if str(x).strip()
        ]

        plan.synthesized_conclusions = _cross_source_conclusions(reasoning)

        # Structure: the caller's blueprint strategy + the outline's themes.
        plan.strategy = str(strategy or "")
        if outline is not None:
            plan.themes = [
                str(getattr(s, "title", "") or "")
                for s in (getattr(outline, "sections", None) or [])
                if str(getattr(s, "title", "") or "").strip()
                and str(getattr(s, "title", "")) != "Answer"
            ][:5]

        # The question being answered is the USER'S QUERY, not the first
        # planned dimension. Using the first sub-question narrowed a broad
        # question ("current trends in AI") onto a single angle ("how fast is
        # adoption growing") — the exact collapse the blueprint exists to
        # prevent. The query is the answer's subject; the dimensions are its
        # coverage.
        plan.central_question = str(query or "").strip()
        if not plan.central_question:
            for item in sub_questions or []:
                if isinstance(item, dict) and str(item.get("question", "")).strip():
                    plan.central_question = str(item.get("question")).strip()
                    break

        # Priority + omission guidance, derived from what the ranking found.
        if plan.dominant:
            plan.priority_note = (
                f"Prioritise the {len(plan.dominant)} dominant finding(s) above; "
                "everything else supports or qualifies them."
            )
            if plan.under_researched:
                plan.priority_note += (
                    " Where a dominant finding rests on an under-researched dimension, "
                    "state the uncertainty rather than the conclusion alone."
                )
        if plan.incidental:
            plan.omit_note = (
                "The incidental findings are available if they support a dominant "
                "point; do not enumerate them for completeness."
            )
    except Exception as exc:  # total/fail-safe: return the partial plan
        logger.warning("synthesis_plan_failed", error=str(exc), exc_info=exc)
    return plan
