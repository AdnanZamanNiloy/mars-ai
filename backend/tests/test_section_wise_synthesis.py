"""Section-wise synthesis + context compression (GPT Researcher adaptation).

Regression targets:
  * synthesis falling back to deterministic fact extraction on broad queries,
  * answers becoming source/claim dumps instead of expert synthesis,
  * many sources of the same claim inflating the evidence view.

The section-wise path must assemble per-section output and degrade cleanly
to the single-pass writer when a section call fails (deterministic fallback,
AGENTS.md 4.7). Context compression must de-duplicate without dropping
distinct claims.
"""
import asyncio

from app.agents.synthesizer import _compress_to_themes, synthesize
from app.agents.outline import build_outline


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


class _SectionLLM:
    """Returns a distinct section body per call; numbers a marker in range."""

    def __init__(self, fail_on_call: int | None = None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    async def generate_json(self, system_prompt, user_prompt, **kwargs):
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RuntimeError("provider down mid-section")
        # Each section cites marker [1] — always in the assembled legend.
        return {"answer": f"Section body number {self.calls} with a cited claim [1]."}


class _SectionFailLLM:
    """Fails the SECOND call (mid section-wise); later calls succeed."""

    def __init__(self, fail_call: int = 2):
        self.calls = 0
        self.fail_call = fail_call

    async def generate_json(self, system_prompt, user_prompt, **kwargs):
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError("provider down mid-section")
        return {"answer": "SINGLE-PASS answer citing evidence [1]."}


def test_section_wise_assembles_all_sections():
    outline = build_outline("What is the current trend of AI?", _facts(), _sub_questions())
    assert outline.broad
    llm = _SectionLLM()
    result = asyncio.run(
        synthesize(llm, "What is the current trend of AI?", _facts(),
                   {"intent": {}, "sub_questions": _sub_questions()},
                   outline=outline, section_wise=True, compress_context=False)
    )
    # One call per section with evidence (>=2), and the report carries each
    # section header — not a single monolithic pass.
    assert llm.calls >= 2
    for section in outline.sections:
        assert f"## {section.title}" in result.answer
    assert result.used_fallback is False


def test_section_wise_degrades_to_single_pass_on_failure():
    """A failed section call must abandon the section-wise path and let the
    single-pass writer (deterministic fallback included) produce the report —
    never a half-assembled, mixed draft."""
    outline = build_outline("What is the current trend of AI?", _facts(), _sub_questions())
    llm = _SectionFailLLM(fail_call=2)
    result = asyncio.run(
        synthesize(llm, "What is the current trend of AI?", _facts(),
                   {"intent": {}, "sub_questions": _sub_questions()},
                   outline=outline, section_wise=True, compress_context=False)
    )
    # The abandoned section-wise attempt left no partial headers: the report
    # is the single-pass retry's output, not a half-assembled draft.
    assert "SINGLE-PASS answer" in result.answer
    assert "Section body number" not in result.answer
    # Section 1 (ok) + section 2 (fail) + single-pass retry = 3 calls.
    assert llm.calls == 3


def test_section_wise_disabled_is_single_pass():
    """section_wise=False keeps the original single-call behaviour."""
    outline = build_outline("What is the current trend of AI?", _facts(), _sub_questions())

    class _OneShotLLM:
        def __init__(self):
            self.calls = 0

        async def generate_json(self, system_prompt, user_prompt, **kwargs):
            self.calls += 1
            return {"answer": "## Executive Summary\n\nA synthesized answer [1]."}

    llm = _OneShotLLM()
    asyncio.run(
        synthesize(llm, "What is the current trend of AI?", _facts(),
                   {"intent": {}, "sub_questions": _sub_questions()},
                   outline=outline, section_wise=False, compress_context=False)
    )
    assert llm.calls == 1


def test_compression_merges_duplicates_keeps_distinct():
    facts = [
        {"claim": "The transformer architecture uses self-attention.", "source": "https://a.example/1"},
        {"claim": "The transformer architecture uses self attention.", "source": "https://b.example/2"},
        {"claim": "Transformers replaced recurrent sequence models.", "source": "https://c.example/3"},
        {"claim": "Global spending on AI reached 200 billion dollars.", "source": "https://d.example/4"},
    ]
    compressed = _compress_to_themes(facts, similarity_threshold=0.7)
    claims = [f["claim"] for f in compressed]
    # The two near-identical claims collapse to one entry...
    assert len(compressed) == 3
    # ...and the collapsed entry records corroboration from both sources.
    merged = next(f for f in compressed if "self" in f["claim"].lower())
    assert int(merged.get("corroboration_count", 1)) >= 2
    # Distinct claims are never dropped.
    assert any("recurrent" in c for c in claims)
    assert any("200 billion" in c for c in claims)


def test_compression_preserves_order_and_identity():
    facts = [{"claim": f"Distinct finding number {i} about topic {i}.", "source": f"https://s{i}.example/x"}
             for i in range(5)]
    compressed = _compress_to_themes(facts, similarity_threshold=0.9)
    assert [f["claim"] for f in compressed] == [f["claim"] for f in facts]
