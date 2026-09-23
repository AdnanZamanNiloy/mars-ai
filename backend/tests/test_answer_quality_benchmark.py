"""Adaptive answer-quality benchmark: structure from question, not templates.

Runs the deterministic benchmark in bench/eval_answer_quality.py. It must not
reward a heading list: a structurally different answer scores well when it
answers the question. The floors assert adaptability, process cleanliness,
citation presence and relevance.
"""
from bench.eval_answer_quality import aggregate, cases, score_case


def test_benchmark_covers_the_required_categories():
    cats = {c.category for c in cases()}
    required = {
        "factual", "definition", "explanation", "how-to", "troubleshooting",
        "comparison", "causal", "current-events", "broad-research",
        "literature-review", "decision-support", "technical-investigation",
        "forecasting", "data-analysis", "ambiguous",
    }
    assert required <= cats, required - cats


def test_benchmark_floors_pass():
    agg = aggregate([score_case(c) for c in cases()])
    assert agg["adaptability_rate"] >= 0.9, agg
    assert agg["process_clean_rate"] == 1.0, agg
    assert agg["citation_rate"] >= 0.85, agg


def test_a_well_structured_but_different_answer_scores_high():
    """Structural adaptability is about matching the question, not a heading
    list: a criterion-comparison answer passes the comparison category with no
    universal headings at all."""
    from bench.eval_answer_quality import Case, _fact, score_case
    case = Case(
        "cmp", "comparison", "Compare A and B.",
        "A is faster, whereas B is more predictable [1]. On cost, B is cheaper [2].",
        facts=[_fact("A is faster.", "https://a.example"), _fact("B is cheaper.", "https://b.example")],
    )
    r = score_case(case)
    assert r["adaptive"] is True
    assert r["shape"] == "criterion_comparison"
