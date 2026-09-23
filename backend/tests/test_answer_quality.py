"""Answer Quality Optimizer: the pre-delivery gate (five-axis 0-100 scoring)."""

from app.agents.answer_quality import evaluate_answer

INTENT_ML = {
    "ambiguity": False, "explanation_level": "practical",
    "senses": [], "recommended_action": "research_dominant", "domain": "machine_learning",
}

FACTS = [
    {"claim": "The transformer architecture uses attention mechanisms.",
     "source": "https://arxiv.org/abs/1706.03762", "verified": True,
     "sub_question": "what is the transformer architecture", "is_primary": True},
    {"claim": "Attention weighs every input token against every other.",
     "source": "https://arxiv.org/abs/1706.03762", "verified": True,
     "sub_question": "what is the transformer architecture", "is_primary": True},
    {"claim": "Transformers replaced recurrence in sequence models.",
     "source": "https://aclanthology.org/x", "verified": True,
     "sub_question": "transformer mechanism components", "is_primary": True},
]

SUPPORT_OK = {
    "rate": 1.0, "cited": 4, "supported": 4, "uncited": 0, "numeric_rate": None,
    "sentences": 6,
    "sentence_details": [
        {"sentence": "The transformer architecture uses attention mechanisms [1].",
         "markers": [1], "status": "supported", "support": 0.8},
        {"sentence": "Attention weighs every input token [1].",
         "markers": [1], "status": "supported", "support": 0.7},
        {"sentence": "Self-attention removes recurrence [2].",
         "markers": [2], "status": "supported", "support": 0.6},
        {"sentence": "Encoder and decoder stacks form the model.",
         "markers": [], "status": "uncited", "support": None},
    ],
}

GOOD_ANSWER = (
    "## Executive Summary\n\n"
    "The transformer is a neural network architecture built on attention: every "
    "token attends to every other, replacing recurrence entirely. [1]\n\n"
    "## Key Findings\n\n"
    "- Attention weighs every input token against every other [1].\n"
    "- Self-attention removes the recurrence bottleneck of earlier models [2].\n"
    "- Encoder and decoder stacks form the model.\n\n"
    "## Evidence & Confidence\n\n"
    "Well-supported: 3 verified facts feed this report.\n\n"
    "## Limitations\n\n"
    "Could not verify: claims without a traceable source were discarded.\n\n"
    "## Sources\n\n"
    "[1] arxiv.org (preprint, primary) — https://arxiv.org/abs/1706.03762\n\n"
    "[2] aclanthology.org (peer_reviewed, primary) — https://aclanthology.org/x"
)

DUMP_ANSWER = (
    "# Final Answer\n\n"
    "Some statistics were found.\n\n"
    "- 40% 2024 2000 GW\n"
    "- 1.2 billion usd\n"
    "- 500 mw 2023\n\n"
    "## Sources\n\n[1] arxiv.org (preprint, primary) — https://arxiv.org/abs/1706.03762"
)


def test_good_answer_passes_the_gate():
    report = evaluate_answer(
        "What is the transformer architecture?",
        intent=INTENT_ML, answer=GOOD_ANSWER, facts=FACTS,
        answer_support=SUPPORT_OK, mode="quick", threshold=70,
    )
    assert report.passed, report.failures
    assert report.overall >= 70
    assert report.accuracy >= 70
    assert report.relevance >= 60


def test_machine_appendix_does_not_penalize_evidence_density():
    """Regression: the evidence sub-score measured citation density over the
    WHOLE answer, including the machine-generated confidence panel, limitations,
    contradiction ranges and source ledger. Those sections carry no [n] markers
    by construction, so a report whose own prose was well cited was scored as
    if most of its sentences were unsupported — pinning evidence near 45/100.
    Density must measure the writer's body only."""
    answer = (
        "## Executive Summary\n\n"
        "The transformer is a neural network architecture built on attention [1].\n\n"
        "## Key Findings\n\n"
        "- Attention weighs every input token against every other [1].\n"
        "- Self-attention removes the recurrence bottleneck of earlier models [2].\n\n"
        "## Evidence & Confidence\n\n"
        "Well-supported: 3 verified facts feed this report; every claim above "
        "traces to a cited source and the panel records the measured counts.\n\n"
        "## Limitations & Unknowns\n\n"
        "- 2 claim(s) are single-source or unverified and should be treated as "
        "provisional pending a second independent publisher.\n\n"
        "## Source ledger\n\n"
        "- Documents read: 2 across 2 independent domain(s)\n"
        "- Claims verified against their cited source: 3/3\n\n"
        "## Sources\n\n"
        "[1] arxiv.org (preprint, primary) — https://arxiv.org/abs/1706.03762\n\n"
        "[2] aclanthology.org (peer_reviewed, primary) — https://aclanthology.org/x"
    )
    details = {
        "rate": 1.0, "cited": 3, "supported": 3, "uncited": 0, "numeric_rate": None,
        "sentences": 9,
        "sentence_details": [
            {"sentence": "The transformer is a neural network architecture built on attention [1].",
             "markers": [1], "status": "supported", "support": 0.8},
            {"sentence": "- Attention weighs every input token against every other [1].",
             "markers": [1], "status": "supported", "support": 0.7},
            {"sentence": "- Self-attention removes the recurrence bottleneck of earlier models [2].",
             "markers": [2], "status": "supported", "support": 0.6},
        ],
    }
    report = evaluate_answer(
        "What is the transformer architecture?",
        intent=INTENT_ML, answer=answer, facts=FACTS,
        answer_support=details, mode="quick", threshold=70,
    )
    assert report.details["citation_density"] == 1.0
    assert report.evidence >= 55, report.details


def test_appendix_without_prose_does_not_manufacture_density():
    """A report that is ONLY a well-cited body scores 1.0; an all-appendix
    answer (no writer prose) falls back to 0.0 rather than dividing by the
    machine sections it must ignore."""
    answer = (
        "## Evidence & Confidence\n\n"
        "Well-supported: 3 verified facts feed this report and every claim "
        "above traces to a cited source in the ledger below.\n\n"
        "## Source ledger\n\n"
        "- Documents read: 2 across 2 independent domain(s)\n"
        "- Claims verified against their cited source: 3/3"
    )
    details = {"rate": 1.0, "cited": 0, "supported": 0, "uncited": 0,
               "numeric_rate": None, "sentences": 4, "sentence_details": []}
    report = evaluate_answer(
        "What is the transformer architecture?",
        intent=INTENT_ML, answer=answer, facts=FACTS,
        answer_support=details, mode="quick", threshold=70,
    )
    assert report.details["citation_density"] == 0.0


def test_offtopic_answer_fails_relevance():
    """The exact reported failure: "What is transformer?" answered with
    electrical-transformer content. The query resolves ambiguous, so the
    missing disambiguation plus wrong-sense sections must fail the gate even
    with perfect evidence statistics."""
    ambiguous = {**INTENT_ML, "ambiguity": True, "recommended_action": "research_dominant",
                 "senses": [
                     {"label": "Transformer neural network architecture",
                      "domain": "machine_learning", "probability": 0.75},
                     {"label": "Electrical transformer (AC voltage device)",
                      "domain": "engineering", "probability": 0.20},
                 ]}
    report = evaluate_answer(
        "What is transformer?",
        intent=ambiguous,
        answer=(
            "## Executive Summary\n\n"
            "Electricity grids move power across long distances and transformers "
            "change voltage levels safely [1].\n\n"
            "## Limitations\n\nCould not verify.\n\n"
            "## Sources\n\n[1] arxiv.org (preprint, primary) — https://arxiv.org/abs/1706.03762"
        ),
        facts=FACTS, answer_support=SUPPORT_OK, mode="standard", threshold=70,
    )
    assert not report.passed
    assert report.relevance < 60
    assert any("disambiguation" in f.lower() for f in report.failures)


def test_offtopic_unambiguous_answer_fails_on_concept_terms():
    """Without ambiguity, a summary that never addresses the query's core
    terms still fails relevance."""
    report = evaluate_answer(
        "What is the transformer architecture?",
        intent=INTENT_ML,
        answer=(
            "## Executive Summary\n\n"
            "Electricity grids move power across long distances safely.\n\n"
            "## Limitations\n\nCould not verify.\n\n"
            "## Sources\n\n[1] arxiv.org (preprint, primary) — https://arxiv.org/abs/1706.03762"
        ),
        facts=FACTS, answer_support=SUPPORT_OK, mode="standard", threshold=70,
    )
    assert report.relevance < 60
    assert not report.passed


def test_ambiguous_answer_without_disambiguation_is_flagged():
    ambiguous = {**INTENT_ML, "ambiguity": True, "recommended_action": "research_dominant",
                 "senses": [
                     {"label": "Transformer neural network architecture", "domain": "machine_learning",
                      "probability": 0.75},
                     {"label": "Electrical transformer", "domain": "engineering", "probability": 0.20},
                 ]}
    report = evaluate_answer(
        "What is transformer?",
        intent=ambiguous, answer=GOOD_ANSWER, facts=FACTS,
        answer_support=SUPPORT_OK, mode="standard", threshold=70,
    )
    assert any("disambiguation" in f.lower() for f in report.failures)


def test_data_dump_penalizes_clarity():
    report = evaluate_answer(
        "What is the transformer architecture?",
        intent=INTENT_ML, answer=DUMP_ANSWER, facts=FACTS,
        answer_support=SUPPORT_OK, mode="standard", threshold=70,
    )
    assert report.clarity < 70
    assert not report.passed


def test_failures_are_actionable_for_the_retry_prompt():
    report = evaluate_answer(
        "What is the transformer architecture?",
        intent=INTENT_ML, answer=DUMP_ANSWER, facts=FACTS,
        answer_support=SUPPORT_OK, mode="standard", threshold=70,
    )
    assert report.failures
    assert all(isinstance(f, str) and len(f) > 20 for f in report.failures)
    # Failures must be specific and content-oriented, not a demand for a
    # particular heading. The data-dump draft fails on its statistics dump.
    assert any("dump" in f.lower() for f in report.failures)


def test_scores_are_stable_and_in_range():
    for answer in (GOOD_ANSWER, DUMP_ANSWER):
        r = evaluate_answer("What is the transformer architecture?",
                            intent=INTENT_ML, answer=answer, facts=FACTS,
                            answer_support=SUPPORT_OK, mode="standard")
        for score in (r.accuracy, r.relevance, r.evidence, r.clarity, r.reasoning, r.overall):
            assert 0 <= score <= 100


async def test_quality_gate_retries_once_and_ships_better_draft(monkeypatch):
    """Graph wiring: a failing draft gets exactly one corrective re-synthesis
    with the failures in its context; the better draft is what ships."""
    import app.graph.workflow as wf
    from app.core.config import Settings
    from app.core.llm import LLMClient

    class _FakeSearch:
        settings = Settings(groq_api_key="k", _env_file=None)

        async def run_search(self, queries):
            out = []
            for q in queries or []:
                text_q = q[0] if isinstance(q, tuple) else q
                out.append({"title": "RAG paper", "snippet": "RAG retrieves documents",
                            "url": "https://arxiv.org/a", "sub_question": text_q})
            return out

    calls = []

    async def fake_planner(**kwargs):
        return [{
            "id": 1, "question": "what is RAG", "axis": "definition",
            "search_type": "encyclopedia", "priority": 1, "depends_on": [],
            "domain": "machine_learning", "minimum_sources": 2, "coverage_goal": "",
            "stop_condition": "", "variants": [], "agent": "", "tools": ["web_search"],
            "scope": [], "output_format": "structured_findings", "specialist": "technical",
            "preferred_domains": [], "primary_source_query": "", "wave": 0, "sense": "",
        }]

    async def fake_summarizer(llm=None, query=None, search_results=None,
                              specialist_role="general", prior_findings=None, sense=""):
        return [{
            "claim": "RAG retrieves documents before generation",
            "source": "https://arxiv.org/a", "confidence": 0.9,
            "sub_question": "what is RAG", "verified": True,
        }]

    async def fake_critic(**kwargs):
        return {"is_sufficient": False, "reason": "thin", "improved_queries": [], "confidence": 0.4}

    async def fake_synthesizer(llm=None, query=None, facts=None, context=None):
        calls.append(dict(context or {}))
        if len(calls) == 1:
            return "No."
        return (
            "RAG retrieves documents before generation, so the model answers from "
            "grounded evidence rather than parametric memory alone [1]. Taken "
            "together, the evidence supports using retrieval whenever answers must "
            "be traceable to sources [1]. Limitations: the evidence here is a "
            "single source, so the claim remains provisional pending independent "
            "confirmation.\n\n"
            "## Sources\n\n[1] arxiv.org (preprint, primary) — https://arxiv.org/a"
        )

    monkeypatch.setattr(wf, "planner_agent", fake_planner)
    monkeypatch.setattr(wf, "summarizer_agent", fake_summarizer)
    monkeypatch.setattr(wf, "critic_agent", fake_critic)
    monkeypatch.setattr(wf, "synthesizer_agent", fake_synthesizer)

    settings = Settings(groq_api_key="k", _env_file=None)
    graph = wf.create_workflow(LLMClient(settings), _FakeSearch())
    state = wf.build_initial_state("What is RAG?", 3, mode="quick")
    final = None
    async for snap in graph.astream(state, stream_mode="values"):
        final = snap

    assert len(calls) == 2, "gate retry must fire exactly once"
    assert "quality_feedback" not in calls[0]
    assert calls[1].get("quality_feedback"), "failures must reach the retry draft"
    assert final["synthesized_answer"].startswith("RAG retrieves documents")
    assert final["quality"]["passed"] is True
    # The measured quality score lives in the audit layer, not the answer.
    assert final["final_report"] == final["synthesized_answer"]
    assert "Answer quality" in final["final_audit"]
