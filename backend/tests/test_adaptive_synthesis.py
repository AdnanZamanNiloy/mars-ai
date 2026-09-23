"""Adaptive synthesis: structure from the question + evidence, not a template.

These tests pin the behaviours the diagnosis identified as the remaining
rigidity after the profile/outline work:

  * a broad deep answer must NOT automatically reproduce the historical
    universal section skeleton (Executive Summary / Key Findings / Evidence &
    Confidence / Limitations & Unknowns / …);
  * process mechanics must not leak into the primary answer;
  * the primary answer and the audit are structurally separate, and the audit
    still carries the metadata;
  * the presentation strategy adapts to the question family;
  * citations and contradictions survive adaptive synthesis;
  * a deterministic fallback is citation-safe and process-noise-free.

Deterministic and LLM-free: stub writers return fixed prose.
"""
from __future__ import annotations

import asyncio
import re

import app.graph.workflow as wf
from app.agents.outline import build_blueprint, build_outline
from app.agents.synthesizer import synthesize, synthesizer_agent

# The historical universal headings. Any of these appearing unrequested in a
# broad adaptive answer is the regression.
UNIVERSAL_HEADINGS = (
    "## Executive Summary",
    "## Key Findings",
    "## Evidence & Confidence",
    "## Limitations & Unknowns",
    "## Counterarguments & Disputed Points",
    "## Open Questions & Missing Angles",
    "## Key Figures",
    "## Auditable Source Ledger",
    "## Evidence Integrity",
    "## Reasoning",
)


def _facts():
    return [
        {"claim": "Global AI capital expenditure reached 300 billion dollars in 2025.",
         "source": "https://a.example/x", "confidence": 0.85, "verified": True,
         "sub_question": "AI infrastructure spending", "axis": "evidence"},
        {"claim": "Foundation model training runs doubled in compute year over year.",
         "source": "https://b.example/y", "confidence": 0.8, "verified": True,
         "sub_question": "model scaling", "axis": "mechanism"},
        {"claim": "Regulators in the EU and US proposed new model-transparency rules.",
         "source": "https://c.example/z", "confidence": 0.75, "verified": True,
         "sub_question": "AI regulation", "axis": "criticism"},
        {"claim": "Robotics firms raised record funding for embodied AI in 2025.",
         "source": "https://d.example/w", "confidence": 0.7, "verified": True,
         "sub_question": "embodied AI", "axis": "outlook"},
    ]


def _sub_questions():
    return [
        {"question": "AI infrastructure spending", "axis": "evidence"},
        {"question": "model scaling", "axis": "mechanism"},
        {"question": "AI regulation", "axis": "criticism"},
        {"question": "embodied AI", "axis": "outlook"},
    ]


class _ThematicWriter:
    """Writer that produces evidence-driven thematic prose, no template slots."""

    async def generate_json(self, system_prompt, user_prompt, **kwargs):
        return {"answer": (
            "AI's centre of gravity shifted from model releases to the "
            "infrastructure and governance around them [1].\n\n"
            "## Compute and capital\nThe build-out is the defining change: "
            "capital expenditure reached 300 billion dollars in 2025 [1], "
            "while training runs doubled in compute [2].\n\n"
            "## Governance\nRule-making caught up with deployment, with the EU "
            "and US proposing transparency rules [3].\n\n"
            "## Embodied systems\nRobotics drew record funding [4], though the "
            "evidence on commercial traction is still thin."
        )}


class _ProcessNoiseWriter:
    """Worst case: writer copies pipeline mechanics into the answer."""

    async def generate_json(self, system_prompt, user_prompt, **kwargs):
        return {"answer": (
            "AI spending is rising [1].\n\n"
            "Pipeline stages on deterministic fallback: planner, verifier. "
            "Evidence grade C after 2 corroboration attempts. The search budget "
            "was 8 queries and the internal confidence calculation gave 0.55, "
            "below the threshold."
        )}


class _TemplateWriter:
    """Writer that fills the historical template slots."""

    async def generate_json(self, system_prompt, user_prompt, **kwargs):
        return {"answer": (
            "## Executive Summary\n\nAI is changing rapidly [1].\n\n"
            "## Key Findings\n\n- Spending rose [1].\n\n"
            "## Evidence & Confidence\n\nA=1, B=2.\n\n"
            "## Limitations & Unknowns\n\nMore research is needed.\n\n"
            "## Counterarguments & Disputed Points\n\nNone detected.\n\n"
            "## Open Questions & Missing Angles\n\n- none.\n\n"
            "## Sources\n\n[1] a.example — https://a.example/x"
        )}


# ---------------------------------------------------------------------------
# A. Rigid-template regression
# ---------------------------------------------------------------------------

def test_broad_adaptive_answer_does_not_reproduce_the_universal_skeleton():
    result = asyncio.run(
        synthesize(_ThematicWriter(), "What are the current trends in AI as of 2026?",
                   _facts(), {"intent": {}, "sub_questions": _sub_questions(), "mode": "deep"},
                   compress_context=False)
    )
    # The old fixed skeleton must not be reproduced wholesale.
    present = [h for h in UNIVERSAL_HEADINGS if h in result.answer]
    assert len(present) <= 1, f"universal skeleton leaked: {present}"
    # The delivered answer is the writer's own structure.
    assert "## Compute and capital" in result.answer


def test_template_slot_answer_still_leaves_structure_alone():
    """The synthesizer honours the writer's own structure; it does not add a
    required heading the writer omitted in an adaptive profile."""
    result = asyncio.run(
        synthesize(_TemplateWriter(), "What are the current trends in AI as of 2026?",
                   _facts(), {"intent": {}, "sub_questions": _sub_questions(), "mode": "deep"},
                   compress_context=False)
    )
    # Nothing is force-injected beyond what the writer produced.
    assert "## Auditable Source Ledger" not in result.answer


# ---------------------------------------------------------------------------
# B. Process-noise regression
# ---------------------------------------------------------------------------

_PROCESS_PHRASES = (
    "pipeline stage",
    "deterministic fallback",
    "corroboration attempt",
    "search budget",
    "internal confidence calculation",
    "below the threshold",
)


def test_process_noise_is_scrubbed_from_the_answer():
    result = asyncio.run(
        synthesize(_ProcessNoiseWriter(), "What are the current trends in AI as of 2026?",
                   _facts(), {"intent": {}, "sub_questions": _sub_questions(), "mode": "standard"},
                   compress_context=False)
    )
    lowered = result.answer.lower()
    for phrase in _PROCESS_PHRASES:
        assert phrase not in lowered, f"process noise leaked: {phrase!r}"
    # The real subject sentence survives.
    assert "spending is rising" in lowered


def test_fallback_answer_has_no_process_noise():
    class _Exploding:
        async def generate_json(self, *a, **k):
            raise RuntimeError("down")

    ctx: dict = {"intent": {}, "mode": "standard", "confidence": 0.6, "degraded": ["planner"]}
    answer = asyncio.run(synthesizer_agent(_Exploding(), "What are the current trends in AI?", _facts(), ctx))
    lowered = answer.lower()
    assert "pipeline stage" not in lowered
    assert "deterministic fallback" not in lowered
    assert "of 4 facts verified" not in lowered
    # The extractive fallback still cites its claims.
    assert "[1]" in answer


# ---------------------------------------------------------------------------
# C. Answer / audit separation
# ---------------------------------------------------------------------------

def test_answer_and_audit_are_separate_documents():
    state = {
        "query": "What are the current trends in AI?",
        "synthesized_answer": "AI is shifting to infrastructure and governance [1].",
        "facts": _facts(),
        "confidence": 0.72,
        "quality": {"overall": 81, "accuracy": 80, "relevance": 85, "evidence": 90,
                    "clarity": 78, "reasoning": 75, "passed": True},
        "critique": {"is_sufficient": True},
        "synthesis_machine_notes": ["## Evidence & Confidence\n\nWell-supported: 4 of 4."],
    }
    answer = wf.build_markdown_report(state)
    audit = wf.build_answer_audit(state)
    # The answer is exactly the synthesizer's text: no quality panel, no
    # confidence float, no evidence accounting.
    assert answer == state["synthesized_answer"]
    assert "# Answer Quality" not in answer
    assert "# Confidence Score" not in answer
    assert "Evidence & Confidence" not in answer
    # The audit carries all of it.
    assert "Answer quality" in audit
    assert "0.72" in audit
    assert "Evidence & Confidence" in audit


# ---------------------------------------------------------------------------
# D. Query-type adaptation
# ---------------------------------------------------------------------------

def test_blueprint_strategies_differ_by_question_family():
    outline = build_outline("Compare React and Vue", _facts(), [])
    comparison = build_blueprint("Compare React and Vue", _facts(), [], outline=outline)
    assert comparison.strategy == "comparison"
    assert "criterion" in comparison.framework.lower()

    why = build_blueprint("Why did AI capex rise in 2025?", _facts(), [])
    assert why.strategy == "causal"
    assert "mechanism" in why.framework.lower()

    how = build_blueprint("How do I configure a build pipeline?", _facts(), [])
    assert how.strategy == "howto"
    assert "steps" in how.framework.lower()

    definition = build_blueprint("What is CRISPR?", _facts(), [])
    assert definition.strategy == "definition"
    assert definition.depth == "concise"


def test_blueprint_prefers_declared_intent():
    bp = build_blueprint("Tell me about X", _facts(), [], intent={"query_type": "comparison"})
    assert bp.strategy == "comparison"
    assert bp.comparison is True


# ---------------------------------------------------------------------------
# E. Citation preservation
# ---------------------------------------------------------------------------

def test_adaptive_synthesis_preserves_valid_citations():
    result = asyncio.run(
        synthesize(_ThematicWriter(), "What are the current trends in AI as of 2026?",
                   _facts(), {"intent": {}, "sub_questions": _sub_questions(), "mode": "deep"},
                   compress_context=False)
    )
    markers = {int(m) for m in re.findall(r"\[(\d+)\]", result.answer)}
    valid = {int(s.get("n", 0)) for s in result.sources} - {0}
    assert markers, "the adaptive answer must keep its citations"
    assert markers <= valid, f"ungrounded markers: {markers - valid}"
    assert result.audit.citation_density > 0


# ---------------------------------------------------------------------------
# F. Contradiction preservation
# ---------------------------------------------------------------------------

def test_contradictions_reach_the_audit_and_survive_synthesis():
    state = {
        "query": "How large is the AI market?",
        "synthesized_answer": "Estimates of the AI market range widely [1][2].",
        "facts": _facts(),
        "confidence": 0.6,
        "contradictions": [{
            "claim_a": "Market is 300 billion", "source_a": "https://a.example/x",
            "claim_b": "Market is 900 billion", "source_b": "https://b.example/y",
            "resolved": False,
        }],
        "critique": {"is_sufficient": False, "reason": "conflicting sizing"},
    }
    audit = wf.build_answer_audit(state)
    assert "Source conflicts" in audit
    assert "300 billion" in audit and "900 billion" in audit
    # The conflict is not silently averaged away in the answer.
    assert "300 billion" in state["synthesized_answer"] or "range" in state["synthesized_answer"].lower()


# ---------------------------------------------------------------------------
# G. Mode behaviour: effort, not a report format
# ---------------------------------------------------------------------------

def test_modes_do_not_change_the_adaptive_structure_contract():
    """Quick and deep use the same adaptive contract; structural headings are
    not mode-specific. The difference is length/depth guidance, not a format."""
    for mode in ("quick", "standard", "deep"):
        result = asyncio.run(
            synthesize(_ThematicWriter(), "What are the current trends in AI?",
                       _facts(), {"intent": {}, "sub_questions": _sub_questions(), "mode": mode},
                       compress_context=False)
        )
        present = [h for h in UNIVERSAL_HEADINGS if h in result.answer]
        assert len(present) <= 1, f"{mode} forced a template: {present}"


def test_audit_profile_keeps_the_fixed_format():
    """Audit is the one profile whose contract IS a fixed structure."""
    result = asyncio.run(
        synthesize(_ThematicWriter(), "What are the current trends in AI?",
                   _facts(), {"intent": {}, "sub_questions": _sub_questions(),
                              "report_profile": "audit"},
                   compress_context=False)
    )
    assert "## Executive Summary" in result.answer
    assert "## Key Findings" in result.answer
