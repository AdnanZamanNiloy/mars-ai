"""LLM Analytical Synthesis: a thesis-producing stage between plan and writer.

Covers the four contract categories (thesis/insights, relationships,
counter-evidence + cross-source conclusions, uncertainty), the deterministic
no-fabrication guard, the fallback contract, the writer wiring, and the
central-question fix in the deterministic plan.
"""
import asyncio

from app.agents.analyst import (
    AnalyticalBrief,
    analytical_synthesis,
)
from app.core.synthesis_planner import build_synthesis_plan


# --- fixtures ---------------------------------------------------------------

_FACTS = [
    {"claim": "Enterprise AI adoption reached 78% of surveyed organizations in 2025.",
     "source": "https://mckinsey.com/a", "verified": True, "has_numbers": True,
     "corroborating_sources": ["https://gartner.com/a"]},
    {"claim": "AI inference costs fell roughly 90% over 18 months.",
     "source": "https://a16z.com/b", "verified": True, "has_numbers": True,
     "corroborating_sources": ["https://epochai.org/b"]},
]


class _LLM:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = 0

    async def generate_json(self, system_prompt, user_prompt, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.payload


def _run(llm, **kw):
    return asyncio.run(analytical_synthesis(
        llm, "What are the current trends in AI?", _FACTS, kw.pop("plan", None), **kw
    ))


# --- 1. thesis / insights ----------------------------------------------------

def test_thesis_and_insights_produced_from_llm_output():
    llm = _LLM({
        "thesis": "AI is moving from experimentation to deployment [1][2].",
        "insights": ["Adoption and falling costs reinforce each other [1][2]."],
        "relationships": [{"kind": "causation", "statement": "Cheaper inference enables adoption [2][1]."}],
        "counter_evidence": ["Some surveys overstate production use [1]."],
        "implications": ["Deployment budgets should assume continued cost declines [2]."],
        "cross_source_conclusions": ["Taken together, the two trends indicate operational maturity [1][2]."],
        "uncertainties": ["Adoption definitions vary across surveys [1]."],
    })
    brief = _run(llm)
    assert brief.origin == "llm"
    assert "experimentation" in brief.thesis
    assert brief.insights and brief.relationships
    assert brief.relationships[0]["kind"] == "causation"
    assert brief.counter_evidence and brief.cross_source_conclusions
    assert brief.uncertainties


# --- 2. no-fabrication guard -------------------------------------------------

def test_thesis_with_invented_number_is_dropped():
    llm = _LLM({
        "thesis": "AI adoption reached 99 percent in 2026 [1].",
        "insights": ["Costs fell 90% over 18 months [2]."],
    })
    brief = _run(llm)
    # 99% is not in the evidence -> thesis dropped; the grounded insight stays.
    assert brief.thesis == ""
    assert any("90%" in i for i in brief.insights)


def test_citation_marker_digits_are_not_read_as_numbers():
    llm = _LLM({
        "thesis": "The two findings jointly support operational maturity [1][2].",
        "insights": [],
    })
    brief = _run(llm)
    assert brief.thesis, "citation markers must not trip the invented-number guard"


# --- 3. fallback contract ----------------------------------------------------

def test_llm_failure_falls_back_to_plan_without_raising():
    plan = build_synthesis_plan(
        _FACTS, [], query="What are the current trends in AI?", query_type="analytical"
    )
    brief = _run(_LLM(error=RuntimeError("provider down")), plan=plan)
    assert brief.origin == "fallback"
    # The fallback carries the plan's established/conclusions, never empty.
    assert brief.insights or brief.cross_source_conclusions


def test_empty_llm_output_falls_back():
    brief = _run(_LLM({}))
    assert brief.origin == "fallback"


def test_disabled_stage_returns_fallback():
    llm = _LLM({"thesis": "should not be used [1]"})
    brief = _run(llm, enabled=False)
    assert brief.origin == "fallback"
    assert llm.calls == 0, "a disabled stage must not spend an LLM call"


def test_empty_evidence_returns_empty_brief_without_calling_llm():
    llm = _LLM({"thesis": "x [1]"})
    brief = asyncio.run(analytical_synthesis(llm, "q", [], None))
    assert brief.is_empty
    assert llm.calls == 0


# --- 4. rendering + no chain-of-thought --------------------------------------

def test_render_is_empty_for_empty_brief_and_bounded_when_populated():
    assert AnalyticalBrief().render_for_writer() == ""
    brief = _LLM({
        "thesis": "A thesis [1].",
        "insights": ["An insight [1][2]."],
    })
    rendered = _run(brief).render_for_writer()
    assert "ANALYTICAL SYNTHESIS" in rendered
    assert "CENTRAL THESIS" in rendered


# --- 5. wiring into the writer prompt ---------------------------------------

def test_analytical_brief_reaches_writer_prompt():
    from app.agents.synthesizer import synthesize
    from app.agents.outline import build_outline

    brief = AnalyticalBrief(
        thesis="AI is shifting to operational deployment [1].",
        insights=["Adoption and cost declines reinforce each other [1][2]."],
    )
    outline = build_outline("What are the current trends in AI?", _FACTS, [])
    captured: list = []

    class _RecLLM:
        async def generate_json(self, system_prompt, user_prompt, **kwargs):
            captured.append(user_prompt)
            return {"answer": "## Answer\n\nAI is shifting [1]."}

    asyncio.run(synthesize(
        _RecLLM(), "What are the current trends in AI?", _FACTS,
        {"intent": {}, "sub_questions": [], "analytical_brief": brief},
        outline=outline, section_wise=False, compress_context=False,
    ))
    assert any("ANALYTICAL SYNTHESIS" in p for p in captured)
    assert any("CENTRAL THESIS" in p for p in captured)


# --- 6. deterministic plan central-question fix ------------------------------

def test_broad_query_central_question_is_the_query_not_first_dimension():
    subs = [
        {"question": "How fast is enterprise AI adoption growing?", "axis": "evidence"},
        {"question": "Why are AI inference costs falling?", "axis": "mechanism"},
    ]
    plan = build_synthesis_plan(
        _FACTS, [], query="What are the current trends in AI as of 2026?",
        query_type="analytical", sub_questions=subs,
    )
    assert plan.central_question == "What are the current trends in AI as of 2026?"
    assert "How fast is enterprise AI adoption" not in plan.central_question
