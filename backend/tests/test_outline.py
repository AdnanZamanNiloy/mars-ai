"""Answer-first outline + context compression: the broad-query guard.

Regression target: a broad question ("What is the current trend of AI?")
used to be answered from whatever claims ranked highest, collapsing into a
single narrow thesis or a source dump. The outline is derived from the
query + evidence BEFORE writing, deterministically.
"""
from app.agents.outline import (
    AnswerOutline,
    build_outline,
    group_facts_by_section,
    outline_dimensions,
    render_outline,
)


def _sub_questions():
    return [
        {"question": "what is AI definition", "axis": "definition", "coverage_goal": "define AI"},
        {"question": "AI market data 2026", "axis": "evidence", "coverage_goal": "quantify adoption"},
        {"question": "AI risks and criticism", "axis": "criticism", "coverage_goal": "name limits"},
        {"question": "AI outlook 2026", "axis": "outlook", "coverage_goal": "forecast"},
    ]


def _facts():
    return [
        {"claim": "AI is the simulation of human intelligence.", "axis": "definition",
         "source": "https://a.example/x", "confidence": 0.8},
        {"claim": "Global AI spending reached 200 billion in 2025.", "axis": "evidence",
         "source": "https://b.example/y", "confidence": 0.7},
        {"claim": "AI systems can encode bias at scale.", "axis": "criticism",
         "source": "https://c.example/z", "confidence": 0.6},
        {"claim": "Adoption is expected to keep rising through 2026.", "axis": "outlook",
         "source": "https://d.example/w", "confidence": 0.5},
    ]


def test_broad_query_produces_multi_dimension_outline():
    """A broad query must produce an outline covering multiple dimensions —
    not a single narrow thesis section."""
    outline = build_outline("What is the current trend of AI?", _facts(), _sub_questions())
    dims = outline_dimensions(outline)
    assert len(dims) >= 3, dims
    assert outline.broad is True
    assert set(dims) >= {"definition", "evidence", "criticism", "outlook"}
    # Each dimension carries its own evidence, and no fact is duplicated.
    grouped = group_facts_by_section(outline)
    total = sum(len(fs) for _, fs in grouped)
    assert total == len(_facts())


def test_outline_sections_are_ordered_survey_first():
    outline = build_outline("compare AI vs ML", _facts(), _sub_questions())
    dims = outline_dimensions(outline)
    # definition/history must precede evidence/criticism/outlook.
    assert dims.index("definition") < dims.index("evidence")
    assert dims.index("definition") < dims.index("criticism")


def test_narrow_query_falls_back_single_section():
    """A narrow factual query with one axis is not 'broad' and must not
    force a multi-section report."""
    outline = build_outline(
        "What is the transformer architecture?",
        [{"claim": "Transformers use attention.", "axis": "definition",
          "source": "https://a.example/x"}],
        [{"question": "transformer architecture definition", "axis": "definition"}],
    )
    assert outline.broad is False
    assert len(outline.sections) == 1


def test_empty_evidence_still_yields_outline():
    """Deterministic fallback: never hand the writer an empty shape."""
    outline = build_outline("what is X", [], [])
    assert isinstance(outline, AnswerOutline)
    assert len(outline.sections) == 1
    assert outline.sections[0].title == "Answer"


def test_ambiguous_intent_splits_senses():
    intent = {
        "ambiguity": True,
        "senses": [
            {"label": "Transformer neural network architecture", "domain": "machine_learning"},
            {"label": "Electrical transformer (AC voltage device)", "domain": "engineering"},
        ],
    }
    outline = build_outline("What is transformer?", [], [], intent=intent)
    titles = [s.title for s in outline.sections]
    assert "Transformer neural network architecture" in titles
    assert "Electrical transformer (AC voltage device)" in titles


def test_render_outline_names_sections_and_fact_counts():
    outline = build_outline("What is the current trend of AI?", _facts(), _sub_questions())
    rendered = render_outline(outline)
    assert "REPORT OUTLINE" in rendered
    for section in outline.sections:
        assert section.title in rendered
