"""Evidence Synthesis Planning: reason over the evidence landscape before writing.

Covers the plan's core responsibilities — dominant vs incidental ranking,
dimension coverage and under-researched detection, cross-source syntheses,
conflicts, epistemic separation — plus the fail-safe contract and the synthesis
wiring (the plan must reach the writer prompt).
"""
import asyncio

from app.core.synthesis_planner import (
    DOMINANT_CENTRALITY_BAR,
    SynthesisPlan,
    build_synthesis_plan,
)
from app.core.reasoning_engine import build_reasoning
from app.agents.outline import build_outline


# --- fixtures ---------------------------------------------------------------

def _corroborated(claim: str, domains=("gov.uk", "who.int"), numbers=True) -> dict:
    return {
        "claim": claim,
        "source": f"https://{domains[0]}/report",
        "verified": True,
        "has_numbers": numbers,
        "corroborating_sources": [f"https://{d}/x" for d in domains],
    }


def _single_source(claim: str, source="https://blog.example.com/a") -> dict:
    return {"claim": claim, "source": source, "verified": True}


# --- 1. dominant vs incidental ----------------------------------------------

def test_dominant_findings_clear_the_bar_and_incidental_do_not():
    facts = [
        _corroborated("Global AI spending reached 200 billion dollars in 2025."),
        _single_source("Adoption grew 42 percent in 2024."),
        {"claim": "A peripheral remark with no bearing.", "source": "https://x.example/z"},
    ]
    plan = build_synthesis_plan(
        facts, [], query="What is AI spending?", query_type="analytical"
    )
    assert plan.dominant, "at least one dominant finding expected"
    assert all(f.centrality >= DOMINANT_CENTRALITY_BAR for f in plan.dominant)
    # The peripheral claim never leads.
    assert all("peripheral" not in f.claim for f in plan.dominant)
    assert any("peripheral" in f.claim for f in plan.incidental)


def test_dominant_is_bounded():
    facts = [
        _corroborated(f"Quantitative finding number {i} reached {i * 10} billion dollars.")
        for i in range(12)
    ]
    plan = build_synthesis_plan(facts, [], query="q", query_type="factual")
    from app.core.synthesis_planner import MAX_DOMINANT

    assert len(plan.dominant) <= MAX_DOMINANT


def test_ranking_is_deterministic_across_input_order():
    facts = [
        _corroborated("Alpha finding: revenue hit 5 billion dollars.", domains=("a.com", "b.com")),
        _corroborated("Beta finding: cost fell to 2 percent.", domains=("c.com", "d.com")),
    ]
    first = build_synthesis_plan(facts, [], query="q", query_type="factual").to_dict()
    second = build_synthesis_plan(list(reversed(facts)), [], query="q", query_type="factual").to_dict()
    assert first["dominant"] == second["dominant"]
    assert first["incidental"] == second["incidental"]


# --- 2. dimension coverage / under-researched -------------------------------

def test_under_researched_dimension_named_when_thin_but_central():
    facts = [
        _corroborated("Market size reached 200 billion dollars.", domains=("gov.uk", "who.int")),
    ]
    facts[0]["sub_question"] = "market size"
    facts[0]["axis"] = "evidence"
    # A planned dimension with one central finding -> thin relative to importance.
    sub_questions = [
        {"question": "market size", "axis": "evidence"},
    ]
    outline = build_outline("What is the AI market?", facts, sub_questions)
    reasoning = build_reasoning(
        facts, [], query="What is the AI market?", query_type="factual",
        sub_questions=sub_questions,
    )
    plan = build_synthesis_plan(
        facts, [], query="What is the AI market?", query_type="factual",
        outline=outline, reasoning=reasoning, sub_questions=sub_questions,
    )
    assert plan.dimensions
    thin = [d for d in plan.dimensions if d.under_researched]
    assert thin, "a thin central dimension should be flagged under-researched"
    assert plan.under_researched


def test_well_covered_dimension_not_flagged_under_researched():
    facts = [
        _corroborated("Market size reached 200 billion dollars.", domains=("gov.uk", "who.int")),
        _corroborated("The market grew 20 percent year over year.", domains=("a.com", "b.com")),
    ]
    for f in facts:
        f["sub_question"] = "market size"
        f["axis"] = "evidence"
    sub_questions = [{"question": "market size", "axis": "evidence"}]
    outline = build_outline("What is the AI market?", facts, sub_questions)
    plan = build_synthesis_plan(
        facts, [], query="What is the AI market?", query_type="factual",
        outline=outline, sub_questions=sub_questions,
    )
    assert not any(
        d.axis == "evidence" and d.under_researched for d in plan.dimensions
    )


# --- 3. cross-source syntheses ----------------------------------------------

def test_legitimate_cross_source_synthesis_surfaces():
    facts = [
        _corroborated("Adoption is rising across workflows.", domains=("a.com", "b.com")),
    ]
    reasoning = build_reasoning(facts, [], query="q", query_type="analytical")
    plan = build_synthesis_plan(
        facts, [], query="q", query_type="analytical", reasoning=reasoning
    )
    assert plan.synthesized_conclusions
    # A single-publisher claim is not a convergence and never becomes one.
    lone = [_single_source("Only one publisher says this.")]
    lone_reasoning = build_reasoning(lone, [], query="q", query_type="analytical")
    lone_plan = build_synthesis_plan(
        lone, [], query="q", query_type="analytical", reasoning=lone_reasoning
    )
    assert all("Only one publisher" not in c for c in lone_plan.synthesized_conclusions)


# --- 4. conflicts + epistemic separation ------------------------------------

def test_conflicts_and_epistemic_status_come_from_reasoning():
    facts = [_corroborated("Spending reached 200 billion dollars.", domains=("a.com", "b.com"))]
    contradictions = [{
        "claim_a": "Spending was 200 billion",
        "claim_b": "Spending was 300 billion",
        "source_a": "https://a.com/x",
        "source_b": "https://b.com/y",
        "kind": "numeric",
        "severity": 0.7,
        "resolved": False,
    }]
    reasoning = build_reasoning(facts, contradictions, query="q", query_type="factual")
    plan = build_synthesis_plan(
        facts, contradictions, query="q", query_type="factual", reasoning=reasoning
    )
    assert plan.conflicts
    assert plan.conflicts[0]["kind"] == "numeric"
    assert plan.established


# --- 5. fail-safe / total ----------------------------------------------------

def test_empty_and_garbage_input_is_total():
    for plan in (
        build_synthesis_plan([], [], query="q"),
        build_synthesis_plan(None, None, query="q"),  # type: ignore[arg-type]
        build_synthesis_plan([{}, "garbage", 42, None], [{"bad": object()}], query="q"),  # type: ignore[list-item]
    ):
        assert isinstance(plan, SynthesisPlan)
        assert plan.to_dict()["dominant"] == []


def test_render_is_empty_for_empty_plan():
    plan = build_synthesis_plan([], [], query="q")
    assert plan.is_empty
    assert plan.render_for_writer() == ""


# --- 6. synthesis wiring -----------------------------------------------------

def test_synthesis_plan_reaches_writer_prompt():
    from app.agents.synthesizer import synthesize

    facts = [
        _corroborated("Global AI spending reached 200 billion dollars in 2025."),
        _single_source("Adoption grew 42 percent in 2024."),
    ]
    outline = build_outline("What is AI spending?", facts, [])
    reasoning = build_reasoning(facts, [], query="What is AI spending?", query_type="analytical")
    plan = build_synthesis_plan(
        facts, [], query="What is AI spending?", query_type="analytical",
        outline=outline, reasoning=reasoning,
    )
    captured: list[str] = []

    class _RecLLM:
        async def generate_json(self, system_prompt, user_prompt, **kwargs):
            captured.append(user_prompt)
            return {"answer": "## Executive Summary\n\nA concluding answer [1]."}

    asyncio.run(
        synthesize(
            _RecLLM(), "What is AI spending?", facts,
            {"intent": {}, "sub_questions": [], "reasoning": reasoning, "synthesis_plan": plan},
            outline=outline, section_wise=False, compress_context=False,
        )
    )
    assert captured, "no writer prompt captured"
    assert any("EVIDENCE SYNTHESIS PLAN" in p for p in captured)


def test_synthesis_plan_is_built_and_passed_by_the_workflow_node():
    """The real synthesizer_node builds a SynthesisPlan from graded evidence and
    hands it to the writer context (integration wiring, not just the unit)."""
    import app.graph.workflow as wf
    from app.core.config import Settings
    from app.core.llm import LLMClient

    captured: dict = {}

    async def fake_planner(llm, query, critique_feedback="", today="", **kwargs):
        return [{"id": 1, "question": "What is the AI market?", "axis": "evidence",
                 "search_type": "encyclopedia", "priority": 1, "depends_on": [],
                 "coverage_goal": "", "domain": "general", "minimum_sources": 1,
                 "stop_condition": "enough", "variants": []}]

    async def fake_summarizer(llm, query, search_results=None, specialist_role="general"):
        return [
            {"claim": "Global AI spending reached 200 billion dollars in 2025.",
             "source": "https://gov.uk/report", "verified": True, "has_numbers": True,
             "corroborating_sources": ["https://who.int/x"],
             "sub_question": "What is the AI market?", "axis": "evidence"},
        ]

    async def fake_critic(llm, query, facts=None, iteration=1, max_iterations=3,
                          contradictions=None, **kwargs):
        return {"is_sufficient": True, "reason": "ok", "improved_queries": [],
                "confidence": 0.9}

    async def capturing_synthesizer(llm, query, facts, context=None):
        captured["context"] = context or {}
        return "## Executive Summary\n\nAI spending reached 200 billion [1].\n\nSources:\n[1] gov.uk — https://gov.uk/report"

    class StubSearch:
        async def run_search(self, questions):
            first = questions[0]
            text = first[0] if isinstance(first, (tuple, list)) else first
            return [{"url": "https://gov.uk/report", "sub_question": text,
                     "snippet": "s", "content": "c"}]

    wf.planner_agent = fake_planner
    wf.summarizer_agent = fake_summarizer
    wf.critic_agent = fake_critic
    wf.synthesizer_agent = capturing_synthesizer
    try:
        settings = Settings(groq_api_key="k", _env_file=None)
        state = wf.build_initial_state("What is the AI market?", 3)
        workflow = wf.create_workflow(LLMClient(settings), StubSearch())

        async def _drive():
            async for _ in workflow.astream(state, stream_mode="values"):
                pass

        asyncio.run(_drive())
    finally:
        import importlib

        importlib.reload(wf)

    plan = captured.get("context", {}).get("synthesis_plan")
    assert plan is not None, "the workflow node must build a synthesis plan"
    assert hasattr(plan, "render_for_writer")
    assert plan.dominant, "a quantitative corroborated claim must be dominant"


# --- Phase 9: relevance / centrality selection --------------------------------

def test_factual_but_peripheral_finding_is_down_ranked():
    """A quantitative, well-formed fact unrelated to the question must not
    lead on factuality alone."""
    facts = [
        {"claim": "Global AI adoption reached 78 percent in 2025.",
         "source": "https://mckinsey.com/a", "verified": True, "has_numbers": True,
         "corroborating_sources": ["https://gartner.com/a"],
         "sub_question": "adoption", "axis": "evidence"},
        {"claim": "An unrelated historical statistic about canal tonnage reached 40 percent.",
         "source": "https://history.example/x", "verified": True, "has_numbers": True,
         "sub_question": "peripheral background", "axis": "history"},
    ]
    plan = build_synthesis_plan(
        facts, [], query="What is the current trend of AI adoption?",
        query_type="analytical",
        sub_questions=[{"question": "adoption", "axis": "evidence"}],
    )
    dominant_claims = " ".join(f.claim.lower() for f in plan.dominant)
    assert "ai adoption" in dominant_claims, plan.dominant
    assert "canal tonnage" not in dominant_claims
    # The peripheral fact is not deleted — it stays available but out of the lead.
    incidental_claims = " ".join(f.claim.lower() for f in plan.incidental)
    assert "canal tonnage" in incidental_claims


def test_relevant_finding_beats_incidental_finding():
    facts = [
        {"claim": "AI inference costs fell roughly 90 percent over 18 months.",
         "source": "https://a16z.com/b", "verified": True, "has_numbers": True,
         "corroborating_sources": ["https://epochai.org/b"],
         "sub_question": "inference costs", "axis": "mechanism"},
        {"claim": "A minor remark about office furniture trends reached 10 percent.",
         "source": "https://random.example/z", "verified": True, "has_numbers": True,
         "sub_question": "office furniture", "axis": "general"},
    ]
    plan = build_synthesis_plan(
        facts, [], query="Why are AI inference costs falling?",
        query_type="analytical",
        sub_questions=[{"question": "inference costs", "axis": "mechanism"}],
    )
    assert plan.dominant and "inference costs" in plan.dominant[0].claim.lower()
    assert all("furniture" not in f.claim.lower() for f in plan.dominant)


def test_existing_ranking_behavior_intact_without_query_context():
    """With no query/dimension terms the relevance multiplier is 1.0, so the
    pre-Phase-9 absolute ranking is unchanged."""
    facts = [
        {"claim": "Alpha quantitative finding reached 5 billion dollars.",
         "source": "https://a.com/x", "verified": True, "has_numbers": True,
         "corroborating_sources": ["https://b.com/x"]},
        {"claim": "Beta finding is a plain non-quantitative statement.",
         "source": "https://c.com/y", "verified": True},
    ]
    plan = build_synthesis_plan(facts, [], query="", query_type="")
    # The quantitative corroborated finding still leads on absolute merit.
    assert plan.dominant and "Alpha" in plan.dominant[0].claim
