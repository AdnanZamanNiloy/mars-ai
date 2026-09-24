"""Answer prioritization — decide what the written answer leads with.

Why this module exists
----------------------
By the time the writer runs, the pipeline holds a LOT of material: dominant
findings, incidental findings, required dimensions, uncovered dimensions,
under-researched notes, cross-source conclusions, counter-evidence,
uncertainties, conflicts and evidence grades. The writer prompt handed all of
it over with roughly equal weight, so normal answers drifted toward the
research process — "the evidence base is incomplete", "several dimensions
remain uncertain", audit bookkeeping and redundant caveats — instead of the
direct answer the user asked for.

This module makes the selection explicit and deterministic. It classifies the
material the writer will see into four tiers:

  * CORE_ANSWER   — directly answers the question; necessary to understand the
                    conclusion (the AnalystBrief thesis, the plan's dominant
                    findings, cross-source conclusions, the covered required
                    dimensions).
  * SUPPORTING    — evidence, explanation, comparison or causal reasoning that
                    strengthens the answer (counter-evidence that materially
                    changes the reading, conflicts, implications, under-
                    researched dimensions that qualify a conclusion).
  * CONTEXT       — useful background, not necessary to answer (incidental
                    findings, non-dominant themes).
  * AUDIT_ONLY    — detailed uncertainty bookkeeping, evidence gaps, research-
                    process material. Stays available to the audit layer; it is
                    NOT dropped, and it may still surface when it materially
                    changes the conclusion.

It renders a compact "how to write this" directive that goes at the TOP of the
writer prompt, next to the AnalystBrief, so the hierarchy is
CORE_ANSWER > SUPPORTING > CONTEXT > AUDIT_ONLY.

Design (AGENTS.md)
------------------
* Deterministic and LLM-free: stable ordering, no model call, no latency.
* Reuses — never reimplements — the SynthesisPlan and AnalystBrief artifacts
  the pipeline already computed.
* No new facts: it only orders material that is already in the prompt.
* Total/fail-safe: an empty/garbage plan and brief yield an empty directive,
  never an exception.
* No module-level mutable state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.core.logging import get_logger

logger = get_logger(__name__)

# Tier names. Kept as constants so callers/tests reference one spelling.
CORE_ANSWER = "CORE_ANSWER"
SUPPORTING = "SUPPORTING"
CONTEXT = "CONTEXT"
AUDIT_ONLY = "AUDIT_ONLY"

# The tier order, highest priority first. Exposed for tests and rendering.
TIER_ORDER = (CORE_ANSWER, SUPPORTING, CONTEXT, AUDIT_ONLY)

# How many items of each tier the directive names. Bounded so the directive
# stays a brief and never becomes a second evidence dump.
MAX_CORE = 6
MAX_SUPPORTING = 5
MAX_CONTEXT = 4
MAX_AUDIT = 4


@dataclass
class PrioritizedMaterial:
    """The writer-visible material, grouped by how it should be used."""

    core_answer: List[str] = field(default_factory=list)
    supporting: List[str] = field(default_factory=list)
    context: List[str] = field(default_factory=list)
    audit_only: List[str] = field(default_factory=list)
    direct_answer_hint: str = ""

    @property
    def is_empty(self) -> bool:
        return not (
            self.core_answer or self.supporting or self.context
            or self.audit_only or self.direct_answer_hint
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "core_answer": list(self.core_answer),
            "supporting": list(self.supporting),
            "context": list(self.context),
            "audit_only": list(self.audit_only),
            "direct_answer_hint": self.direct_answer_hint,
        }


def _brief_thesis(brief: Any) -> str:
    if brief is None:
        return ""
    try:
        data = brief.to_dict()
    except Exception:  # a brief that cannot serialize is treated as absent
        return ""
    return str(data.get("thesis", "") or "").strip()


def _classified(
    plan: Any,
    brief: Any,
) -> PrioritizedMaterial:
    """Classify plan + brief material into the four tiers (deterministic)."""
    material = PrioritizedMaterial()
    try:
        plan_data = plan.to_dict() if plan is not None and hasattr(plan, "to_dict") else {}
    except Exception:  # a plan that cannot serialize is treated as absent
        plan_data = {}
    try:
        brief_data = brief.to_dict() if brief is not None and hasattr(brief, "to_dict") else {}
    except Exception:
        brief_data = {}

    # CORE_ANSWER: the thesis answers the question; dominant findings and
    # cross-source conclusions are the conclusion's backbone.
    thesis = str(brief_data.get("thesis", "") or "").strip()
    if thesis:
        material.core_answer.append(f"Thesis: {thesis}")
    for c in (brief_data.get("cross_source_conclusions") or [])[:3]:
        text = str(c or "").strip()
        if text:
            material.core_answer.append(f"Cross-source conclusion: {text}")
    for f in (plan_data.get("dominant") or [])[:MAX_CORE]:
        claim = str(f.get("claim", "") or "").strip()
        if claim:
            material.core_answer.append(claim)
    for i in (brief_data.get("insights") or [])[:4]:
        text = str(i or "").strip()
        if text:
            material.core_answer.append(f"Insight: {text}")

    # SUPPORTING: relationships, material counter-evidence, conflicts and the
    # under-researched dimensions that qualify a conclusion.
    for r in (brief_data.get("relationships") or [])[:3]:
        if isinstance(r, dict):
            statement = str(r.get("statement", "") or "").strip()
            if statement:
                material.supporting.append(f"Relationship: {statement}")
    for c in (brief_data.get("counter_evidence") or [])[:MAX_SUPPORTING]:
        text = str(c or "").strip()
        if text:
            material.supporting.append(f"Counter-evidence: {text}")
    for c in (plan_data.get("conflicts") or [])[:2]:
        if isinstance(c, dict):
            a = str(c.get("position_a", "") or c.get("claim_a", "") or "").strip()
            b = str(c.get("position_b", "") or c.get("claim_b", "") or "").strip()
            if a and b:
                material.supporting.append(f"Conflict: \"{a}\" vs \"{b}\"")
    for d in (plan_data.get("under_researched") or [])[:3]:
        text = str(d or "").strip()
        if text:
            material.supporting.append(f"Thin dimension (qualify the conclusion): {text}")

    # CONTEXT: incidental findings and non-dominant themes — background that
    # helps only if it supports a core point.
    for f in (plan_data.get("incidental") or [])[:MAX_CONTEXT]:
        claim = str(f.get("claim", "") or "").strip()
        if claim:
            material.context.append(claim)
    for t in (plan_data.get("themes") or [])[4:8]:
        text = str(t or "").strip()
        if text:
            material.context.append(f"Theme: {text}")

    # AUDIT_ONLY: detailed uncertainty bookkeeping and required dimensions with
    # no evidence. NOT deleted — the writer may still state one when it
    # materially changes the conclusion; otherwise it stays out of the answer.
    for u in (brief_data.get("uncertainties") or [])[:MAX_AUDIT]:
        text = str(u or "").strip()
        if text:
            material.audit_only.append(text)
    for u in (plan_data.get("uncovered") or [])[:MAX_AUDIT]:
        text = str(u or "").strip()
        if text:
            material.audit_only.append(text)
    for u in (plan_data.get("unknown") or [])[:MAX_AUDIT]:
        text = str(u or "").strip()
        if text:
            material.audit_only.append(text)

    # Direct-answer hint: the question itself, so the writer knows what the
    # first sentence must answer.
    material.direct_answer_hint = str(plan_data.get("central_question", "") or "").strip()
    return material


def build_answer_priority(plan: Any, brief: Any) -> PrioritizedMaterial:
    """Classify the writer-visible material. Deterministic and total."""
    try:
        return _classified(plan, brief)
    except Exception as exc:  # classification failure must not break synthesis
        logger.warning("answer_priority_failed", error=str(exc), exc_info=exc)
        return PrioritizedMaterial()


def render_priority_directive(material: PrioritizedMaterial) -> str:
    """The writer directive for the priority hierarchy. "" when empty.

    The directive names the tiers and, compactly, what belongs to each, so the
    writer leads with the answer and keeps audit bookkeeping out of the normal
    answer. It never deletes uncertainty: it relocates detailed audit material
    to the audit layer, and tells the writer to break that rule only when the
    unknown materially changes the conclusion.
    """
    if material is None or material.is_empty:
        return ""
    lines: List[str] = [
        "ANSWER PRIORITY (apply this hierarchy to everything below — the "
        "material is already grouped for you):",
        "1. CORE_ANSWER — the direct answer and the reasoning needed to "
        "understand it. Lead with this; the first paragraph answers the user's "
        "question in plain language, not the state of the research.",
        "2. SUPPORTING — evidence, comparisons, causal links, material "
        "counter-evidence and tensions that strengthen or qualify the answer.",
        "3. CONTEXT — useful background only; include it if it supports a core "
        "point, otherwise leave it out.",
        "4. AUDIT_ONLY — detailed uncertainty bookkeeping, evidence gaps and "
        "research-process material. Keep it OUT of the normal answer.",
    ]
    if material.direct_answer_hint:
        lines.append("")
        lines.append(
            "The first sentence must answer: " + material.direct_answer_hint
        )
    if material.core_answer:
        lines.append("")
        lines.append("CORE_ANSWER material:")
        lines.extend(f"- {item}" for item in material.core_answer[:MAX_CORE])
    if material.supporting:
        lines.append("")
        lines.append("SUPPORTING material:")
        lines.extend(f"- {item}" for item in material.supporting[:MAX_SUPPORTING])
    if material.context:
        lines.append("")
        lines.append("CONTEXT material (secondary):")
        lines.extend(f"- {item}" for item in material.context[:MAX_CONTEXT])
    if material.audit_only:
        lines.append("")
        lines.append(
            "AUDIT_ONLY material (do NOT turn these into answer sections; state "
            "one only if it materially changes the conclusion, and then in one "
            "plain sentence, not as a list of what the research could not do):"
        )
        lines.extend(f"- {item}" for item in material.audit_only[:MAX_AUDIT])
    lines.append("")
    lines.append(
        "Do not expose internal machinery: no option A/B/C/D labels, no "
        "'recommended option' framing, no dimension-ranking, scoring, plan or "
        "agent terminology, unless the user explicitly asked for a decision "
        "framework. Do not repeat the same limitation in several places."
    )
    return "\n".join(lines) + "\n\n"


def render_for_writer(plan: Any, brief: Any) -> str:
    """Classify + render in one call. "" when there is nothing to prioritize."""
    return render_priority_directive(build_answer_priority(plan, brief))
