"""LLM Analytical Synthesis — turn a SynthesisPlan + verified evidence into a THESIS.

Why this module exists
----------------------
The deterministic `core/synthesis_planner.py` organizes the evidence landscape:
it ranks dominant vs incidental findings, names under-researched dimensions,
collects conflicts and legitimate cross-source syntheses. Measured against real
broad questions, that is an *organizer*, not an *analyst* — it restates
"convergent" claims as conclusions and produces no central thesis, no
relationships between findings, and no prioritisation story.

This stage sits between the plan and the writer:

    Verified Evidence
        ↓
    Deterministic SynthesisPlan
        ↓
    LLM Analytical Synthesis   ← this module
        ↓
    Final Writer
        ↓
    Existing Verification (unchanged)

It asks a small LLM call to reason over the SAME evidence the plan measured and
return the analysis a strong analyst would hold before writing:

  * a central thesis — the single most useful answer the evidence supports;
  * major insights — 3-6 claims about what the evidence collectively means;
  * relationships between findings — agreement, causation, tension, trend;
  * meaningful counter-evidence — the strongest evidence against the thesis;
  * implications — what follows for the reader;
  * legitimate cross-source conclusions — inferences the evidence supports
    that no single source states;
  * important uncertainties — what would change the reading.

Hard constraints (AGENTS.md):

  * It NEVER introduces a new fact, number, or citation: every output sentence
    must reference the numbered evidence it rests on, and a deterministic guard
    drops any output whose numbers are not present in the evidence pool.
  * It NEVER replaces deterministic verification: the writer still cites
    [n] markers against verified facts and `verify_answer_support` still runs
    unchanged. This stage only shapes what the writer reasons from.
  * Deterministic fallback (4.7): on any LLM failure or invalid output the
    writer simply receives the deterministic plan alone — exactly today's
    behaviour — never a broken or empty reasoning brief.
  * Chain-of-thought is never exposed: the stage returns a structured
    `AnalyticalBrief`, and only a compact, bounded rendering reaches the writer
    prompt. The model's scratch reasoning is discarded.
  * No module-level mutable state; per-call and returned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from app.core.degradation import record_fallback
from app.core.llm import LLMClient
from app.core.logging import get_logger
from app.core.schemas import AnalyticalBriefModel

logger = get_logger(__name__)

ANALYST_SYSTEM_PROMPT = """
You are the Analytical Synthesis Agent in a multi-agent research pipeline. You
run AFTER evidence collection and planning and BEFORE the final report is
written. You are the analyst: you reason over the evidence, you do not write
the report.

━━━ YOUR JOB ━━━

You receive (1) the user's question, (2) a deterministic synthesis plan that
ranks the evidence, and (3) the verified evidence statements with citation
numbers. Reason over that evidence and produce the ANALYSIS a strong analyst
would hold before writing:

  1. CENTRAL THESIS — the single most useful answer the evidence supports, in
     one or two sentences. If the evidence does not support a confident thesis,
     say what the best-supported reading is and what is unknown.
  2. MAJOR INSIGHTS — 3 to 6 statements about what the evidence COLLECTIVELY
     means. An insight SYNTHESISES across sources; it is not a restatement of a
     single evidence line. "Taken together, X and Y indicate Z" is an insight.
  3. RELATIONSHIPS — how the important findings connect: agreement,
     corroboration, causation, tension/conflict, or trend. Name the evidence
     numbers each relationship rests on.
  4. COUNTER-EVIDENCE — the strongest evidence that cuts against the thesis.
     If none exists, say so explicitly; never invent a counterpoint.
  5. IMPLICATIONS — what follows for the reader, grounded ONLY in the evidence.
  6. CROSS-SOURCE CONCLUSIONS — inferences the evidence supports that no single
     source states outright. Each must be logically entailed by cited evidence.
  7. UNCERTAINTIES — what would materially change the reading; what remains
     unmeasured or contested.

━━━ HARD RULES ━━━

- NEVER introduce a fact, figure, name or claim that is not in the evidence.
  Every statement references the evidence numbers it rests on, like [1] or
  [2][5].
- NEVER invent a number. Only numbers present in the evidence may appear.
- Distinguish DIRECTLY ESTABLISHED from INFERRED: an inference may be stated as
  an inference ("the evidence suggests..."), never as a measured fact.
- Do not write the report. No headings, no report sections, no prose answer —
  just the analysis in the JSON shape below.
- If the evidence is thin or conflicting, say so in the thesis and
  uncertainties rather than manufacturing confidence.

━━━ OUTPUT FORMAT ━━━

Return ONLY valid JSON. No markdown fences. No text outside JSON.

{
  "thesis": "<1-2 sentence central answer the evidence supports, with [n] refs>",
  "insights": [
    "<synthesis statement, with [n] refs>"
  ],
  "relationships": [
    {"kind": "<agreement|corroboration|causation|tension|trend>",
     "statement": "<how these findings connect, with [n] refs>"}
  ],
  "counter_evidence": [
    "<strongest evidence against the thesis, with [n] refs; empty if none>"
  ],
  "implications": [
    "<what follows for the reader, with [n] refs>"
  ],
  "cross_source_conclusions": [
    "<inference entailed by cited evidence that no single source states>"
  ],
  "uncertainties": [
    "<what would change the reading; what is unmeasured or contested>"
  ]
}
""".strip()

# Bounds so the brief stays a brief and the writer prompt stays affordable.
MAX_INSIGHTS = 6
MAX_RELATIONSHIPS = 5
MAX_COUNTER = 4
MAX_IMPLICATIONS = 5
MAX_CROSS_SOURCE = 5
MAX_UNCERTAINTIES = 5

_CITATION_RE = re.compile(r"\[(\d+)\]")
_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


@dataclass
class AnalyticalBrief:
    """The analyst's reasoning, structured. Never chain-of-thought."""

    thesis: str = ""
    insights: List[str] = field(default_factory=list)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    counter_evidence: List[str] = field(default_factory=list)
    implications: List[str] = field(default_factory=list)
    cross_source_conclusions: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    origin: str = "llm"          # llm | fallback

    @property
    def is_empty(self) -> bool:
        return not (
            self.thesis or self.insights or self.relationships
            or self.counter_evidence or self.implications
            or self.cross_source_conclusions or self.uncertainties
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thesis": self.thesis,
            "insights": list(self.insights),
            "relationships": [dict(r) for r in self.relationships],
            "counter_evidence": list(self.counter_evidence),
            "implications": list(self.implications),
            "cross_source_conclusions": list(self.cross_source_conclusions),
            "uncertainties": list(self.uncertainties),
            "origin": self.origin,
        }

    def render_for_writer(self) -> str:
        """Compact brief for the synthesis prompt. Returns "" when empty."""
        if self.is_empty:
            return ""
        lines: List[str] = [
            "ANALYTICAL SYNTHESIS (the analyst's reasoning over the verified "
            "evidence — write the answer FROM this, not from a source-by-source "
            "recital):"
        ]
        if self.thesis:
            lines.append("")
            lines.append(f"CENTRAL THESIS: {self.thesis}")
        if self.insights:
            lines.append("")
            lines.append("MAJOR INSIGHTS (synthesised across sources):")
            lines.extend(f"- {i}" for i in self.insights)
        if self.relationships:
            lines.append("")
            lines.append("HOW THE FINDINGS RELATE:")
            for r in self.relationships:
                kind = str(r.get("kind", "") or "").strip()
                statement = str(r.get("statement", "") or "").strip()
                if statement:
                    lines.append(f"- [{kind}] {statement}" if kind else f"- {statement}")
        if self.counter_evidence:
            lines.append("")
            lines.append("COUNTER-EVIDENCE (weigh these; do not suppress them):")
            lines.extend(f"- {c}" for c in self.counter_evidence)
        if self.cross_source_conclusions:
            lines.append("")
            lines.append(
                "CROSS-SOURCE CONCLUSIONS (no single source states these; present "
                "as synthesis: 'Taken together, these findings suggest…'):"
            )
            lines.extend(f"- {c}" for c in self.cross_source_conclusions)
        if self.implications:
            lines.append("")
            lines.append("IMPLICATIONS:")
            lines.extend(f"- {i}" for i in self.implications)
        if self.uncertainties:
            lines.append("")
            lines.append("UNCERTAINTIES (state these; they qualify the thesis):")
            lines.extend(f"- {u}" for u in self.uncertainties)
        lines.append("")
        lines.append(
            "Lead with the central thesis. Present the insights as the evidence's "
            "meaning, not as a list of what each source said. Keep every claim "
            "tied to its cited evidence and every inference marked as inference."
        )
        return "\n".join(lines)


def _safe_str_list(value: Any, limit: int) -> List[str]:
    out: List[str] = []
    for item in value or []:
        if isinstance(item, dict):
            text = str(item.get("statement", "") or item.get("text", "") or "").strip()
        else:
            text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _evidence_numbers(pool: Sequence[Dict[str, Any]]) -> set:
    """All significant numeric tokens in the evidence pool, normalized.

    Used to reject an analytical statement that invents a figure. Comma
    separators are stripped so '200,000' and '200000' compare equal.
    """
    numbers: set = set()
    for item in pool or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("claim", "") or "")
        for raw in _NUMBER_RE.findall(text):
            numbers.add(raw.replace(",", ""))
    return numbers


def _mentions_invented_number(text: str, evidence_numbers: set) -> bool:
    """True when `text` carries a number absent from the evidence pool.

    A citation marker's digits are stripped first so '[1]' is never read as an
    invented figure.
    """
    stripped = _CITATION_RE.sub(" ", text or "")
    for raw in _NUMBER_RE.findall(stripped):
        normalized = raw.replace(",", "")
        if normalized in evidence_numbers:
            continue
        return True
    return False


# Capitalized multi-word or distinctive tokens that are NOT evidence-bound
# entities: sentence-initial words and generic research vocabulary. An entity
# the analyst introduces from world knowledge lands here as a "new" token.
_ENTITY_STOPWORDS = {
    "The", "This", "These", "Those", "Taken", "Together", "Overall", "However",
    "Based", "Their", "There", "Before", "After", "When", "Where", "While",
    "Moreover", "Furthermore", "Thus", "Therefore", "In", "On", "At", "By",
    "For", "And", "But", "AI", "US", "EU", "UK", "A", "An", "It", "Its",
    "First", "Second", "Third", "Finally", "Notably", "Because", "Although",
}
_CAP_TERM_RE = __import__("re").compile(r"\b([A-Z][A-Za-z0-9&.\-]+(?:\s+[A-Z][A-Za-z0-9&.\-]+)*)\b")


def _evidence_vocabulary(pool: Sequence[Dict[str, Any]]) -> str:
    """Lowercased concatenation of every evidence claim (entity lookup source)."""
    return " ".join(
        str(f.get("claim", "") or "") for f in (pool or []) if isinstance(f, dict)
    ).lower()


def _mentions_unbound_entity(text: str, evidence_lower: str) -> bool:
    """True when `text` names a capitalized entity absent from the evidence.

    Single capitalized words are ignored (too noisy: sentence starts, common
    nouns). Only multi-word capitalized tokens and acronyms (>=2 chars) are
    checked, because those are the fabricated-entity risk (a person, company or
    product the analyst invented). Matched case-insensitively against the
    evidence text so a legitimate entity named in any evidence line passes.
    """
    for match in _CAP_TERM_RE.findall(text or ""):
        term = match.strip()
        head = term.split()[0]
        if term in _ENTITY_STOPWORDS or head in _ENTITY_STOPWORDS:
            continue
        multi_word = len(term.split()) >= 2
        acronym = term.isupper() and len(term) >= 3
        if not (multi_word or acronym):
            continue
        if term.lower() not in evidence_lower:
            return True
    return False


def _sanitize(
    brief: AnalyticalBrief,
    evidence_numbers: set,
    evidence_lower: str = "",
) -> AnalyticalBrief:
    """Drop any statement that invents a number or an unbound entity.

    The verification contract is unchanged: this only prevents an analytical
    sentence that cites no source from smuggling a fabricated figure or a
    fabricated entity into the writer's brief.
    """
    def keep(text: str) -> bool:
        if _mentions_invented_number(text, evidence_numbers):
            return False
        if evidence_lower and _mentions_unbound_entity(text, evidence_lower):
            return False
        return True

    if brief.thesis and not keep(brief.thesis):
        logger.warning("[Analyst] dropped thesis with an ungrounded number or entity")
        brief.thesis = ""
    brief.insights = [i for i in brief.insights if keep(i)]
    brief.relationships = [
        r for r in brief.relationships if keep(str(r.get("statement", "")))
    ]
    brief.counter_evidence = [c for c in brief.counter_evidence if keep(c)]
    brief.implications = [i for i in brief.implications if keep(i)]
    brief.cross_source_conclusions = [
        c for c in brief.cross_source_conclusions if keep(c)
    ]
    return brief


def _format_evidence(pool: Sequence[Dict[str, Any]]) -> str:
    """Numbered evidence statements, matching the writer's [n] numbering.

    The analyst and the writer must see the SAME numbering, or the thesis'
    [n] references would point at different evidence in the final report.
    """
    lines: List[str] = []
    for i, fact in enumerate(pool or [], 1):
        if not isinstance(fact, dict):
            continue
        claim = str(fact.get("claim", "") or "").strip()
        if not claim:
            continue
        lines.append(f"[{i}] {claim}")
    return "\n".join(lines)


def _format_plan(plan: Any) -> str:
    """The deterministic plan's high-signal fields, compactly."""
    if plan is None or not hasattr(plan, "to_dict"):
        return ""
    try:
        data = plan.to_dict()
    except Exception:  # a plan that cannot serialize must not break the stage
        return ""
    parts: List[str] = []
    if data.get("dominant"):
        parts.append(
            "Dominant findings: "
            + "; ".join(str(f.get("claim", "")) for f in data["dominant"][:5])
        )
    if data.get("conflicts"):
        parts.append(
            "Conflicts: "
            + "; ".join(
                f"'{c.get('position_a', '')}' vs '{c.get('position_b', '')}'"
                for c in data["conflicts"][:3]
            )
        )
    if data.get("under_researched"):
        parts.append("Thin dimensions: " + "; ".join(data["under_researched"][:4]))
    if data.get("established"):
        parts.append("Established: " + "; ".join(data["established"][:5]))
    if data.get("inferred"):
        parts.append("Inferred (single-source): " + "; ".join(data["inferred"][:5]))
    if data.get("unknown"):
        parts.append("Unknown: " + "; ".join(data["unknown"][:4]))
    return "\n".join(parts)


def _fallback_brief(plan: Any) -> AnalyticalBrief:
    """Deterministic brief when the LLM is unavailable.

    Degrades to the plan's own content (never empty when the plan is non-empty),
    so the writer still receives a structured brief. Marked `fallback`.
    """
    brief = AnalyticalBrief(origin="fallback")
    if plan is None:
        return brief
    try:
        data = plan.to_dict()
    except Exception:
        return brief
    conclusions = data.get("synthesized_conclusions") or []
    if conclusions:
        brief.cross_source_conclusions = [str(c) for c in conclusions[:MAX_CROSS_SOURCE]]
    established = data.get("established") or []
    if established:
        brief.insights = [str(c) for c in established[:MAX_INSIGHTS]]
    elif data.get("dominant"):
        # No independently-corroborated set: the ranked dominant findings are
        # the best available brief so the writer still gets structure.
        brief.insights = [
            str(f.get("claim", "")) for f in data["dominant"][:MAX_INSIGHTS]
            if str(f.get("claim", "")).strip()
        ]
    brief.uncertainties = [str(u) for u in (data.get("unknown") or [])[:MAX_UNCERTAINTIES]]
    return brief


async def analytical_synthesis(
    llm: LLMClient,
    query: str,
    facts: Sequence[Dict[str, Any]],
    plan: Any = None,
    *,
    query_type: str = "",
    enabled: bool = True,
) -> AnalyticalBrief:
    """Produce the analytical brief. Deterministic fallback on any failure.

    Follows the standard agent shape (AGENTS.md 4.7): one small LLM call with
    schema validation, and a deterministic fallback that preserves today's
    behaviour when the call fails, is disabled, or returns nothing usable.
    """
    pool = [f for f in (facts or []) if isinstance(f, dict) and str(f.get("claim", "")).strip()]
    if not enabled or not pool:
        return _fallback_brief(plan)

    evidence_block = _format_evidence(pool)
    plan_block = _format_plan(plan)
    user_prompt = (
        f"Question: {query}\n\n"
        + (f"Query type: {query_type}\n\n" if query_type else "")
        + (f"DETERMINISTIC PLAN (measured, ranked):\n{plan_block}\n\n" if plan_block else "")
        + "VERIFIED EVIDENCE (cite these numbers only):\n"
        + evidence_block
        + "\n\nProduce the analytical brief. Return JSON only."
    )

    try:
        payload = await llm.generate_json(
            ANALYST_SYSTEM_PROMPT,
            user_prompt,
            response_model=AnalyticalBriefModel,
        )
    except Exception as exc:
        logger.warning("[Analyst] LLM call failed, using plan fallback", exc_info=exc)
        record_fallback("analyst")
        return _fallback_brief(plan)

    if not isinstance(payload, dict) or not payload:
        logger.warning("[Analyst] empty/invalid LLM output, using plan fallback")
        record_fallback("analyst")
        return _fallback_brief(plan)

    relationships: List[Dict[str, str]] = []
    for item in payload.get("relationships") or []:
        if isinstance(item, dict):
            statement = str(item.get("statement", "") or "").strip()
            if statement:
                relationships.append({
                    "kind": str(item.get("kind", "") or "").strip(),
                    "statement": statement,
                })
        elif str(item or "").strip():
            relationships.append({"kind": "", "statement": str(item).strip()})

    brief = AnalyticalBrief(
        thesis=str(payload.get("thesis", "") or "").strip(),
        insights=_safe_str_list(payload.get("insights"), MAX_INSIGHTS),
        relationships=relationships[:MAX_RELATIONSHIPS],
        counter_evidence=_safe_str_list(payload.get("counter_evidence"), MAX_COUNTER),
        implications=_safe_str_list(payload.get("implications"), MAX_IMPLICATIONS),
        cross_source_conclusions=_safe_str_list(
            payload.get("cross_source_conclusions"), MAX_CROSS_SOURCE
        ),
        uncertainties=_safe_str_list(payload.get("uncertainties"), MAX_UNCERTAINTIES),
        origin="llm",
    )

    if brief.is_empty:
        logger.warning("[Analyst] LLM returned an empty brief, using plan fallback")
        record_fallback("analyst")
        return _fallback_brief(plan)

    brief = _sanitize(brief, _evidence_numbers(pool), _evidence_vocabulary(pool))
    logger.info(
        "[Analyst] thesis=%s insights=%d relationships=%d counter=%d cross=%d",
        bool(brief.thesis), len(brief.insights), len(brief.relationships),
        len(brief.counter_evidence), len(brief.cross_source_conclusions),
    )
    return brief
