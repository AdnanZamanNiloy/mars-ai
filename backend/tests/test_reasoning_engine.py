"""Evidence-grounded reasoning engine: deterministic argument structure.

Covers the four required categories (convergent conclusions, competing
explanations, epistemic separation, decision-relevant implications), the
fail-safe contract, the synthesis wiring, and the conclusion/question-answered
quality signal that replaces the saturated `reasoning` measure.
"""
import asyncio

from app.core.reasoning_engine import (
    ReasoningMap,
    build_reasoning,
)


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


def _unverified(claim: str, source="https://randomblog.example/x") -> dict:
    return {"claim": claim, "source": source}


# --- 1. convergent conclusions ----------------------------------------------

def test_convergent_conclusion_only_from_corroborated_claims():
    facts = [
        _corroborated("Global AI spending reached 200 billion dollars in 2025."),
        _single_source("Adoption grew 42 percent in 2024."),
    ]
    r = build_reasoning(facts, [], query="What is the AI market?", query_type="analytical")
    assert len(r.conclusions) == 1
    c = r.conclusions[0]
    assert "200 billion" in c.claim
    assert set(c.domains) == {"gov.uk", "who.int"}
    assert c.corroboration >= 2
    # The single-source claim never becomes a convergent conclusion.
    assert all("42 percent" not in c.claim for c in r.conclusions)


def test_convergent_ordering_is_deterministic():
    facts = [
        _corroborated("Alpha finding: revenue hit 5 billion dollars.", domains=("a.com", "b.com")),
        _corroborated("Beta finding: cost fell to 2 percent.", domains=("c.com", "d.com")),
    ]
    first = build_reasoning(facts, [], query="q", query_type="factual").to_dict()
    second = build_reasoning(list(reversed(facts)), [], query="q", query_type="factual").to_dict()
    # Same evidence, different input order -> identical structure.
    assert first["conclusions"] == second["conclusions"]
    assert first["established"] == second["established"]


def test_same_publisher_is_not_convergence():
    """Two URLs on one registrable domain are ONE source, never convergence."""
    fact = {
        "claim": "Adoption grew 42 percent in 2024.",
        "source": "https://news.example.com/a",
        "verified": True,
        "has_numbers": True,
        "corroborating_sources": ["https://blog.example.com/b"],
    }
    r = build_reasoning([fact], [], query="q", query_type="factual")
    assert r.conclusions == []
    assert fact["claim"] in r.inferred  # single publisher -> inferred


# --- 2. competing explanations ----------------------------------------------

def test_competing_only_when_real_contradictions_exist():
    contradictions = [{
        "claim_a": "Spending was 200 billion",
        "claim_b": "Spending was 300 billion",
        "source_a": "https://a.com/x",
        "source_b": "https://b.com/y",
        "kind": "numeric",
        "severity": 0.7,
        "resolved": False,
    }]
    r = build_reasoning([], contradictions, query="q", query_type="factual")
    assert len(r.competing) == 1
    c = r.competing[0]
    assert c.kind == "numeric"
    assert c.source_a.endswith("a.com/x")
    assert c.source_b.endswith("b.com/y")
    assert "figure" in c.distinguishes


def test_no_competing_invented_without_contradictions():
    r = build_reasoning([_corroborated("A well-established fact.")], [], query="q")
    assert r.competing == []


def test_competing_kind_and_severity_surface_and_sort():
    contradictions = [
        {"claim_a": "x", "claim_b": "y", "kind": "temporal", "severity": 0.3,
         "source_a": "a.com", "source_b": "b.com", "resolved": True},
        {"claim_a": "p", "claim_b": "q", "kind": "polarity", "severity": 0.55,
         "source_a": "c.com", "source_b": "d.com", "resolved": False},
    ]
    r = build_reasoning([], contradictions, query="q")
    # Unresolved before resolved, highest severity first.
    assert r.competing[0].position_a == "p"
    assert r.competing[0].kind == "polarity"
    assert r.competing[1].resolved is True


# --- 3. epistemic separation -------------------------------------------------

def test_established_inferred_unknown_separation():
    facts = [
        _corroborated("Established A/B claim with two domains."),
        _single_source("Single-source quantitative claim of 42 percent."),
        _unverified("Unverified weak claim."),
    ]
    # Stamp a dimension so the thin-dimension depth signal is measurable.
    facts[1]["sub_question"] = "uncovcred dimension"
    investigation_state = {
        "exhausted claim key": {
            "claim": "A claim that stayed single source",
            "status": "exhausted",
            "attempts": 2,
            "max_attempts": 2,
        }
    }
    dep_state = {
        "facts": facts,
        "sub_questions": [
            {"question": "uncovcred dimension", "axis": "outlook"},
        ],
        "contradictions": [],
    }
    r = build_reasoning(
        facts, [], query="q", query_type="factual",
        investigation_state=investigation_state, depth_state=dep_state,
    )
    assert any("Established A/B" in c for c in r.established)
    assert any("Single-source" in c for c in r.inferred)
    assert any("Unverified" in c for c in r.inferred)
    assert any("stayed single source" in u for u in r.unknown)
    # The uncovered planned axis and its thin dimension are both surfaced.
    assert any("outlook" in u for u in r.unknown)
    assert any("uncovcred dimension" in u or "thin" in u for u in r.unknown)


def test_contradicted_claim_is_not_established():
    fact = _corroborated("Rate fell to 3 percent.")
    contradictions = [{
        "claim_a": "Rate fell to 3 percent.", "claim_b": "Rate rose.",
        "source_a": "gov.uk", "source_b": "other.com", "kind": "numeric",
        "severity": 0.8, "resolved": False,
    }]
    r = build_reasoning([fact], contradictions, query="q")
    assert fact["claim"] not in r.established


# --- 4. decision-relevant implications --------------------------------------

def test_implications_only_for_decision_or_comparative_queries():
    facts = [_corroborated("Spending reached 200 billion dollars in 2025.")]
    factual = build_reasoning(facts, [], query="What is AI spending?", query_type="factual")
    assert factual.decision_relevant is False
    assert factual.implications == []

    decision = build_reasoning(
        facts, [], query="Should governments invest in AI?", query_type="analytical"
    )
    assert decision.decision_relevant is True
    assert decision.implications
    imp = decision.implications[0]
    assert imp.insufficient_evidence is False
    # Tied to the specific established claim that entails it.
    assert imp.claim == "Spending reached 200 billion dollars in 2025."
    assert "gov.uk" in imp.domains


def test_comparative_query_is_decision_relevant():
    facts = [_corroborated("Wind capacity reached 100 GW.")]
    r = build_reasoning(facts, [], query="Compare wind and solar on cost.", query_type="comparative")
    assert r.decision_relevant is True
    assert r.implications


def test_no_fabricated_implication_when_evidence_insufficient():
    facts = [_single_source("A lone unsupported assertion.")]
    r = build_reasoning(
        facts, [], query="Should policymakers act?", query_type="analytical"
    )
    assert r.implications
    assert r.implications[0].insufficient_evidence is True
    # Never a fabricated recommendation tied to a non-established claim.
    assert all(i.claim == "" for i in r.implications)


# --- 5. fail-safe / total ----------------------------------------------------

def test_empty_and_garbage_input_is_total():
    for r in (
        build_reasoning([], [], query="q"),
        build_reasoning(None, None, query="q"),  # type: ignore[arg-type]
        build_reasoning([{}, "garbage", 42, None], [{"bad": object()}], query="q"),  # type: ignore[list-item]
    ):
        assert isinstance(r, ReasoningMap)
        assert r.to_dict()["conclusions"] == []


def test_render_is_empty_for_empty_map():
    r = build_reasoning([], [], query="q")
    assert r.is_empty
    assert r.render_for_writer() == ""
    assert r.render_for_report() == ""


# --- 6. synthesis wiring -----------------------------------------------------

def _facts_for_synthesis():
    return [
        _corroborated("Global AI spending reached 200 billion dollars in 2025."),
        _single_source("Adoption grew 42 percent in 2024."),
    ]


def test_reasoning_structure_reaches_writer_prompt():
    from app.agents.synthesizer import synthesize
    from app.agents.outline import build_outline

    facts = _facts_for_synthesis()
    reasoning = build_reasoning(facts, [], query="What is AI spending?", query_type="analytical")
    captured: list[str] = []

    class _RecLLM:
        async def generate_json(self, system_prompt, user_prompt, **kwargs):
            captured.append(user_prompt)
            return {"answer": "## Executive Summary\n\nA concluding answer [1]."}

    outline = build_outline("What is AI spending?", facts, [])
    asyncio.run(
        synthesize(
            _RecLLM(), "What is AI spending?", facts,
            {"intent": {}, "sub_questions": [], "reasoning": reasoning},
            outline=outline, section_wise=False, compress_context=False,
        )
    )
    assert captured, "no writer prompt captured"
    assert any("EVIDENCE-GROUNDED REASONING STRUCTURE" in p for p in captured)


def test_report_still_contains_required_sections_and_fallback_reasoning():
    from app.agents.synthesizer import synthesize
    from app.agents.outline import build_outline

    facts = _facts_for_synthesis()
    reasoning = build_reasoning(facts, [], query="What is AI spending?", query_type="analytical")

    class _NoReasoningLLM:
        async def generate_json(self, system_prompt, user_prompt, **kwargs):
            # Writer omits any reasoning/conclusion structure.
            return {"answer": "## Executive Summary\n\nGlobal spending reached 200 billion [1]."}

    outline = build_outline("What is AI spending?", facts, [])
    result = asyncio.run(
        synthesize(
            _NoReasoningLLM(), "What is AI spending?", facts,
            {"intent": {}, "sub_questions": [], "reasoning": reasoning},
            outline=outline, section_wise=False, compress_context=False,
        )
    )
    # The deterministic fallback carries the argument structure even though the
    # writer omitted it...
    assert "## Reasoning" in result.answer
    # ...and the mandatory sections are still enforced.
    for heading in ("## Executive Summary", "## Key Findings", "## Limitations & Unknowns"):
        assert heading in result.answer


# --- 7. quality signal: concluding answer beats summing-only -----------------

_SUPPORT = {"rate": 1.0, "cited": 3, "supported": 3, "uncited": 0, "numeric_rate": None,
            "sentences": 5, "sentence_details": []}


def test_summing_only_answer_scores_lower_than_concluding_answer():
    from app.agents.answer_quality import evaluate_answer

    facts = [
        {"claim": "The transformer architecture uses attention mechanisms.",
         "source": "https://arxiv.org/abs/1706.03762", "verified": True,
         "sub_question": "what is the transformer architecture", "is_primary": True},
    ]
    concluding = (
        "## Executive Summary\n\n"
        "Taken together, the evidence supports a clear answer: the transformer is "
        "a neural network architecture built on attention [1].\n\n"
        "## Key Findings\n\n- Attention weighs every token against every other [1].\n\n"
        "## Limitations\n\n"
        "Could not verify: single-source claims remain provisional until corroborated.\n\n"
        "## Sources\n\n[1] arxiv.org (preprint, primary) — https://arxiv.org/abs/1706.03762"
    )
    summing = (
        "## Executive Summary\n\n"
        "The transformer architecture uses attention mechanisms [1].\n\n"
        "## Key Findings\n\n"
        "- The transformer architecture uses attention mechanisms [1].\n"
        "- The transformer architecture uses attention mechanisms [1].\n\n"
        "## Limitations\n\nCould not verify additional claims.\n\n"
        "## Sources\n\n[1] arxiv.org (preprint, primary) — https://arxiv.org/abs/1706.03762"
    )
    good = evaluate_answer(
        "What is the transformer architecture?", answer=concluding,
        facts=facts, answer_support=_SUPPORT, mode="standard", threshold=70,
    )
    weak = evaluate_answer(
        "What is the transformer architecture?", answer=summing,
        facts=facts, answer_support=_SUPPORT, mode="standard", threshold=70,
    )
    assert good.reasoning > weak.reasoning
    assert any("no conclusion" in f.lower() for f in weak.failures)


def test_off_question_answer_scores_lower_on_reasoning():
    from app.agents.answer_quality import evaluate_answer

    facts = [
        {"claim": "The transformer architecture uses attention mechanisms.",
         "source": "https://arxiv.org/abs/1706.03762", "verified": True, "is_primary": True},
    ]
    on_question = (
        "## Executive Summary\n\nOverall, the evidence supports that the transformer "
        "architecture uses attention mechanisms [1].\n\n"
        "## Limitations\n\nCould not verify single-source claims.\n\n"
        "## Sources\n\n[1] arxiv.org (preprint, primary) — https://arxiv.org/abs/1706.03762"
    )
    off_question = (
        "## Executive Summary\n\nOverall, power grids move electricity across long "
        "distances using voltage conversion equipment [1].\n\n"
        "## Limitations\n\nCould not verify single-source claims.\n\n"
        "## Sources\n\n[1] arxiv.org (preprint, primary) — https://arxiv.org/abs/1706.03762"
    )
    good = evaluate_answer("What is the transformer architecture?", answer=on_question,
                           facts=facts, answer_support=_SUPPORT, mode="standard", threshold=70)
    bad = evaluate_answer("What is the transformer architecture?", answer=off_question,
                          facts=facts, answer_support=_SUPPORT, mode="standard", threshold=70)
    assert good.reasoning > bad.reasoning
