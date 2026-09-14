"""Synthesis output-contract fixes from the GPT Researcher gap analysis.

Each test pins a defect that shipped in real deep-mode answers:

  * section-wise assembly duplicated every H2 ("## What It Is\\n\\n## What It Is")
    because the assembler prepends the section title AND the writer re-emitted
    its own heading;
  * the pipeline's internal telemetry ("pipeline confidence is 0.55, below the
    0.75 threshold", "80 of 96 facts verified", "relevance 35/100") leaked into
    the report body instead of living only in the machine-appended appendix;
  * the section-wise path assembled outline sections only and never produced
    the '## Executive Summary' the quality gate hard-requires;
  * the per-section writer had no word budget, so assembled deep reports ran to
    3809 words against the gate's 1500-word cap.

Deterministic, LLM-free: a stub writer returns a fixed body per call.
"""
import asyncio
import re

from app.agents.answer_quality import evaluate_answer, length_band
from app.agents.outline import build_outline
from app.agents.synthesizer import (
    _ensure_disambiguation,
    _scrub_pipeline_telemetry,
    _strip_duplicate_section_heading,
    synthesize,
)


def _sub_questions():
    return [
        {"question": "AI definition", "axis": "definition"},
        {"question": "AI market data", "axis": "evidence"},
        {"question": "AI risks", "axis": "criticism"},
        {"question": "AI outlook", "axis": "outlook"},
    ]


def _facts():
    return [
        {"claim": "Artificial intelligence simulates human intelligence in machines.",
         "axis": "definition", "source": "https://a.example/x", "confidence": 0.8,
         "verified": True, "sub_question": "AI definition"},
        {"claim": "Global AI spending reached 200 billion dollars in 2025.",
         "axis": "evidence", "source": "https://b.example/y", "confidence": 0.7,
         "verified": True, "sub_question": "AI market data"},
        {"claim": "AI systems can encode societal bias at scale.",
         "axis": "criticism", "source": "https://c.example/z", "confidence": 0.6,
         "verified": True, "sub_question": "AI risks"},
        {"claim": "Adoption is expected to keep rising through 2026.",
         "axis": "outlook", "source": "https://d.example/w", "confidence": 0.5,
         "verified": True, "sub_question": "AI outlook"},
    ]


class _EchoHeadingLLM:
    """Worst-case writer: every call re-emits its own '## <title>' heading."""

    def __init__(self):
        self.calls = 0

    async def generate_json(self, system_prompt, user_prompt, **kwargs):
        self.calls += 1
        match = re.search(r'titled "([^"]+)"', user_prompt)
        title = match.group(1) if match else "Section"
        return {"answer": f"## {title}\n\nSubstantive prose citing evidence [1]."}


class _PlainLLM:
    """Well-behaved writer: prose only, no heading."""

    async def generate_json(self, system_prompt, user_prompt, **kwargs):
        return {"answer": "Substantive prose citing evidence [1] and [2]."}


def _run(llm, **ctx):
    outline = build_outline("What is the current trend of AI?", _facts(), _sub_questions())
    assert outline.broad
    return asyncio.run(
        synthesize(llm, "What is the current trend of AI?", _facts(),
                   {"intent": {}, "sub_questions": _sub_questions(),
                    "mode": "deep", **ctx},
                   outline=outline, section_wise=True, compress_context=False)
    )


# ---------------------------------------------------------------------------
# 1. Section-wise assembler must not duplicate headings
# ---------------------------------------------------------------------------

def test_section_wise_does_not_duplicate_headings():
    result = _run(_EchoHeadingLLM())
    answer = result.answer
    # No two consecutive identical H2 lines ("## X\n\n## X").
    assert not re.search(r"^##\s+(.+)\n\s*\n##\s+\1\s*$", answer, re.M), answer


def test_strip_duplicate_section_heading_unit():
    assert _strip_duplicate_section_heading(
        "## What It Is\n\nAt its base, AI...", "What It Is"
    ) == "At its base, AI..."
    # A DIFFERENT heading is genuine sub-structure and must survive.
    body = "### Origin and core innovation\n\nAttention replaced recurrence."
    assert _strip_duplicate_section_heading(body, "How It Works") == body
    # No heading at all is untouched.
    assert _strip_duplicate_section_heading("Plain prose.", "X") == "Plain prose."


# ---------------------------------------------------------------------------
# 2. Pipeline telemetry must never leak into the body
# ---------------------------------------------------------------------------

def test_scrub_removes_pipeline_telemetry_sentences():
    text = (
        "AI adoption accelerated in 2025 [1].\n\n"
        "An earlier quality review assessed relevance at 35/100, below the "
        "40/100 floor, and pipeline confidence is 0.55, below the 0.75 "
        "threshold.\n\n"
        "The pipeline itself reports only 80 of 96 facts verified [2]."
    )
    scrubbed = _scrub_pipeline_telemetry(text)
    assert "35/100" not in scrubbed
    assert "pipeline confidence" not in scrubbed.lower()
    assert "96 facts verified" not in scrubbed
    # The real subject sentence is preserved.
    assert "AI adoption accelerated in 2025" in scrubbed


def test_synthesis_answer_has_no_telemetry_strings():
    """End-to-end: a writer that copies the metadata block gets cleaned."""
    class _TelemetryLLM:
        async def generate_json(self, system_prompt, user_prompt, **kwargs):
            return {"answer": (
                "AI is expanding across enterprises [1].\n\n"
                "Pipeline confidence is 0.55, below the 0.75 threshold, and "
                "the pipeline itself reports only 80 of 96 facts verified."
            )}

    result = _run(_TelemetryLLM())
    lowered = result.answer.lower()
    assert "pipeline confidence" not in lowered
    assert "below the 0.75 threshold" not in lowered
    assert "80 of 96 facts verified" not in lowered
    assert "AI is expanding across enterprises" in result.answer


# ---------------------------------------------------------------------------
# 3. Section-wise report must carry an Executive Summary
# ---------------------------------------------------------------------------

def test_section_wise_includes_executive_summary():
    result = _run(_PlainLLM())
    headings = re.findall(r"^##\s+(.+)$", result.answer, re.M)
    assert "Executive Summary" in headings, headings


# ---------------------------------------------------------------------------
# 4. Length band is shared between writer and gate
# ---------------------------------------------------------------------------
def test_length_band_is_the_single_source_of_truth():
    assert length_band("deep") == (350, 1500)
    assert length_band("standard") == (200, 950)
    # A well-formed deep answer in-band must pass the clarity length check.
    body = "## Executive Summary\n\n" + ("word " * 800)
    report = evaluate_answer("What is the current trend of AI?", answer=body,
                             facts=_facts(), mode="deep", threshold=0)
    assert "outside the deep band" not in " ".join(report.failures)


# ---------------------------------------------------------------------------
# 5. Ambiguous queries must open with the numbered disambiguation block
# ---------------------------------------------------------------------------

_AMBIGUOUS_INTENT = {
    "ambiguity": True,
    "recommended_action": "research_both",
    "senses": [
        {"label": "Transformer neural network architecture",
         "note": "attention-based deep learning model"},
        {"label": "Electrical transformer",
         "note": "device that changes AC voltage"},
    ],
}


def test_ensure_disambiguation_inserts_block_into_exec_summary():
    answer = "## Executive Summary\n\nThe term is overloaded."
    fixed = _ensure_disambiguation(answer, {"intent": _AMBIGUOUS_INTENT})
    assert re.search(r"1\) \*\*Transformer neural network architecture\*\* — ", fixed)
    assert re.search(r"2\) \*\*Electrical transformer\*\* — ", fixed)
    # Inserted under the Executive Summary heading, not before it.
    assert fixed.index("1) **Transformer") > fixed.index("## Executive Summary")


def test_ensure_disambiguation_keeps_existing_block():
    answer = (
        "## Executive Summary\n\n"
        "1) **Transformer neural network architecture** — a deep learning model\n"
        "2) **Electrical transformer** — a voltage device\n"
        "Both senses are distinct."
    )
    assert _ensure_disambiguation(answer, {"intent": _AMBIGUOUS_INTENT}) == answer


def test_ensure_disambiguation_noop_for_unambiguous_query():
    answer = "## Executive Summary\n\nA single clear meaning."
    assert _ensure_disambiguation(answer, {"intent": {}}) == answer


def test_gate_recognizes_disambiguation_after_heading():
    """The gate's numbered-sense regex must run in MULTILINE mode: the block
    lives after '## Executive Summary', not at string offset 0, so a
    non-multiline regex found zero sense lines and hard-failed an ambiguous
    report that had disambiguated correctly."""
    answer = (
        "## Executive Summary\n\n"
        "1) **Transformer neural network architecture** — a deep learning model\n"
        "2) **Electrical transformer** — a voltage device\n"
        "The report focuses on meaning 1."
    )
    intent = {
        "ambiguity": True,
        "recommended_action": "research_both",
        "senses": [
            {"label": "Transformer neural network architecture"},
            {"label": "Electrical transformer (AC voltage device)"},
        ],
    }
    report = evaluate_answer("what is a transformer?", intent=intent,
                             answer=answer, facts=[], mode="deep", threshold=0)
    assert report.details["sense_alignment"] == 1.0
    assert not any("disambiguation" in f for f in report.failures)


def test_gate_matches_sense_label_core_without_parenthetical():
    """A writer rendering 'Electrical transformer' must satisfy an intent
    label of 'Electrical transformer (AC voltage device)' — the gate compares
    the normalized core, not the verbatim string."""
    answer = (
        "## Executive Summary\n\n"
        "1) **Transformer neural network architecture** — model\n"
        "2) **Electrical transformer** — device\n"
        "Focus: meaning 1."
    )
    intent = {
        "ambiguity": True,
        "recommended_action": "research_both",
        "senses": [
            {"label": "Transformer neural network architecture"},
            {"label": "Electrical transformer (AC voltage device)"},
        ],
    }
    report = evaluate_answer("what is a transformer?", intent=intent,
                             answer=answer, facts=[], mode="deep", threshold=0)
    assert report.details["sense_alignment"] == 1.0
