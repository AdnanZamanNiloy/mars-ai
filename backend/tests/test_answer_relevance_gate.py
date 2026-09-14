"""Answer-to-query relevance gate: evidence-rich but query-irrelevant answers.

Regression target: research can be evidence-rich, well cited, and still answer
the wrong thing. The gate must lower the score / fail for a report about a
different topic even when verification and citation signals are high, and pass
a genuinely relevant answer.
"""
from app.agents.answer_quality import evaluate_answer, score_answer_relevance

INTENT = {
    "ambiguity": False, "explanation_level": "practical",
    "senses": [], "recommended_action": "research_dominant", "domain": "machine_learning",
}

FACTS = [
    {"claim": "Solar photovoltaic capacity grew 30 percent in 2025.",
     "source": "https://iea.org/report", "verified": True,
     "sub_question": "solar capacity growth", "is_primary": True},
    {"claim": "Solar panel costs fell 12 percent year over year.",
     "source": "https://irena.org/data", "verified": True,
     "sub_question": "solar panel costs", "is_primary": True},
    {"claim": "Solar installations reached 400 gigawatts globally.",
     "source": "https://iea.org/data", "verified": True,
     "sub_question": "solar capacity growth", "is_primary": True},
]

SUPPORT_OK = {
    "rate": 1.0, "cited": 4, "supported": 4, "uncited": 0, "numeric_rate": 1.0,
    "sentences": 6,
    "sentence_details": [
        {"sentence": "Solar capacity grew 30 percent in 2025 [1].",
         "markers": [1], "status": "supported", "support": 0.8},
        {"sentence": "Solar costs fell 12 percent [2].",
         "markers": [2], "status": "supported", "support": 0.7},
        {"sentence": "Installations reached 400 gigawatts [3].",
         "markers": [3], "status": "supported", "support": 0.6},
    ],
}

RELEVANT_ANSWER = (
    "## Executive Summary\n\n"
    "Solar energy capacity expanded sharply in 2025, with photovoltaic "
    "installations growing 30 percent while panel costs fell 12 percent. [1]\n\n"
    "## Key Findings\n\n"
    "- Solar capacity grew 30 percent in 2025 [1].\n"
    "- Solar panel costs fell 12 percent year over year [2].\n\n"
    "## Evidence & Confidence\n\n"
    "Well-supported by primary statistics. Limitations: could not verify "
    "regional splits.\n\n"
    "## Sources\n\n[1] iea.org (official, primary) — https://iea.org/report\n\n"
    "[2] irena.org (official, primary) — https://irena.org/data"
)

IRRELEVANT_ANSWER = (
    "## Executive Summary\n\n"
    "Neural machine translation models encode sentences and decode them in "
    "target languages using attention networks and large parallel corpora. [1]\n\n"
    "## Key Findings\n\n"
    "- Neural machine translation models encode source sentences [1].\n"
    "- Decoders generate target-language tokens [2].\n\n"
    "## Evidence & Confidence\n\n"
    "Well-supported. Limitations: could not verify.\n\n"
    "## Sources\n\n[1] iea.org (official, primary) — https://iea.org/report\n\n"
    "[2] irena.org (official, primary) — https://irena.org/data"
)


def test_answer_relevance_high_for_matching_answer():
    score = score_answer_relevance(
        "What is the growth trend of solar energy capacity?", RELEVANT_ANSWER
    )
    assert score > 0.3, score


def test_answer_relevance_low_for_offtopic_answer():
    score = score_answer_relevance(
        "What is the growth trend of solar energy capacity?", IRRELEVANT_ANSWER
    )
    assert score < 0.25, score


def test_gate_passes_relevant_answer():
    report = evaluate_answer(
        "What is the growth trend of solar energy capacity?",
        intent=INTENT, answer=RELEVANT_ANSWER, facts=FACTS,
        answer_support=SUPPORT_OK, mode="standard", threshold=70,
    )
    assert report.passed, report.failures
    assert report.details.get("answer_relevance", 0) > 0.3


def test_gate_flags_evidence_rich_but_irrelevant_answer():
    """Same verified facts, same citation support, wrong topic: the
    answer-relevance gate must lower relevance and fail the report."""
    report = evaluate_answer(
        "What is the growth trend of solar energy capacity?",
        intent=INTENT, answer=IRRELEVANT_ANSWER, facts=FACTS,
        answer_support=SUPPORT_OK, mode="standard", threshold=70,
    )
    assert not report.passed
    assert report.relevance < 60
    assert report.details.get("answer_relevance", 1) < 0.25
    assert any("does not match the question" in f for f in report.failures)


def test_relevance_gate_does_not_fail_short_answers_on_length():
    """The gate is about topic, not length: a short but on-topic answer is
    not flagged for irrelevance."""
    short = (
        "## Executive Summary\n\nSolar capacity grew 30 percent in 2025. [1]\n\n"
        "## Limitations\n\nCould not verify regional splits.\n\n"
        "## Sources\n\n[1] iea.org (official, primary) — https://iea.org/report"
    )
    report = evaluate_answer(
        "What is the growth trend of solar energy capacity?",
        intent=INTENT, answer=short, facts=FACTS,
        answer_support=SUPPORT_OK, mode="quick", threshold=60,
    )
    assert report.details.get("answer_relevance", 0) > 0.2
