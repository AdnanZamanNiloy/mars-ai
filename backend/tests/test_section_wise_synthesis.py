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
    # Sections now write concurrently (latency fix): exec summary + ALL
    # issued section calls (gather runs them even when one fails) + the
    # single-pass retry. Serial-abandon semantics (no partial report) hold.
    assert llm.calls >= 4  # 1 exec + >=2 sections + 1 single-pass retry


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


def test_reasoning_depth_contract_reaches_every_writer_prompt():
    """Every writing prompt must demand WHY/mechanism and trade-offs.

    Live evidence: the deep AI report scored reasoning=65 with the quality
    gate noting it stated no limitations — it recounted WHAT per dimension but
    never explained causes or weighed trade-offs. The instruction must reach
    the section writer AND the single-pass writer, never one or the other.
    """
    from app.agents.synthesizer import _REASONING_DEPTH_BLOCK

    outline = build_outline("What is the current trend of AI?", _facts(), _sub_questions())

    prompts: list[str] = []

    class _RecordingLLM:
        async def generate_json(self, system_prompt, user_prompt, **kwargs):
            prompts.append(user_prompt)
            return {"answer": "A synthesized section body with a cited claim [1]."}

    # Section-wise path: exec summary + per-section calls.
    asyncio.run(
        synthesize(_RecordingLLM(), "What is the current trend of AI?", _facts(),
                   {"intent": {}, "sub_questions": _sub_questions()},
                   outline=outline, section_wise=True, compress_context=False)
    )
    section_prompts = [p for p in prompts if "ONE section of a larger report" in p]
    assert section_prompts, "no section-writing prompt captured"
    assert all("REASONING DEPTH" in p for p in section_prompts)

    # Single-pass path.
    prompts.clear()
    asyncio.run(
        synthesize(_RecordingLLM(), "What is the current trend of AI?", _facts(),
                   {"intent": {}, "sub_questions": _sub_questions()},
                   outline=outline, section_wise=False, compress_context=False)
    )
    assert prompts and all("REASONING DEPTH" in p for p in prompts)

    # And the contract is delivered to the writer prompt, not the top-level
    # system prompt (which stays the shared formatting/citation contract).
    assert "REASONING DEPTH" in _REASONING_DEPTH_BLOCK


def test_trim_to_band_brings_overlong_draft_inside_the_band():
    """A draft over the mode's word cap must be trimmed deterministically to
    fit, without dropping any section or the Executive Summary.

    Regression: the section-wise writers overshot the per-section budget (1715
    words against a 1500 deep cap) and the revision pass re-ran the same writer
    and overshot again, so the length contract was never enforced."""
    from app.agents.synthesizer import _count_words, _trim_to_budget
    from app.agents.answer_quality import length_band

    para = " ".join(["word"] * 200)
    body = "\n\n".join([
        "## Executive Summary",
        para,
        "## What It Is",
        para, para, para,
        "## Evidence & Data",
        para, para, para,
        "## Limitations & Critique",
        para, para, para,
    ])
    assert _count_words(body) > 1500
    _, hi = length_band("deep")
    trimmed = _trim_to_budget(body, hi)
    assert _count_words(trimmed) <= hi
    # Every section heading survives; the Executive Summary is never trimmed.
    for heading in ("## Executive Summary", "## What It Is",
                    "## Evidence & Data", "## Limitations & Critique"):
        assert heading in trimmed
    assert "Executive Summary" in trimmed


def test_trim_to_band_leaves_in_band_draft_untouched():
    from app.agents.synthesizer import _trim_to_budget
    from app.agents.answer_quality import length_band

    body = "## Answer\n\nA short, in-band answer [1]."
    _, hi = length_band("deep")
    assert _trim_to_budget(body, hi) == body


def test_compression_never_merges_claims_with_distinct_numbers():
    """Regression: near-identical wording with DIFFERENT quantities is not a
    restatement. Merging '$11.5bn overrun' with '$12.7bn overrun' replaced
    specific evidence with one generic representative and silently dropped a
    figure the report should have reported as a range."""
    facts = [
        {"claim": "The Rooppur project cost overrun reached 11.5 billion dollars.",
         "source": "https://a.example/1"},
        {"claim": "The Rooppur project cost overrun reached 12.7 billion dollars.",
         "source": "https://b.example/2"},
        {"claim": "The Rooppur project cost overrun reached 11.5 billion dollars in 2025.",
         "source": "https://c.example/3"},
    ]
    compressed = _compress_to_themes(facts, similarity_threshold=0.6)
    claims = [f["claim"] for f in compressed]
    # Two distinct figures survive...
    assert any("11.5" in c for c in claims)
    assert any("12.7" in c for c in claims)
    # ...and the genuine restatement (same figure) is merged into one entry.
    assert len(compressed) == 2


def test_required_sections_added_before_length_trim_keeps_band():
    """Regression: mandatory sections and the appendix were appended AFTER the
    deterministic trim, so every deep report overshot the 1500-word band
    (live: 2274-2750 words). The ordering must be _add_required_sections →
    trim, and the required headings must survive the trim."""
    from app.agents.answer_quality import length_band
    from app.agents.synthesizer import (
        _add_required_sections, _count_words, _trim_to_budget, select_profile,
    )

    para = " ".join(["word"] * 200)
    writer_draft = "\n\n".join([
        "## Executive Summary", para,
        "## What It Is", para, para, para,
        "## How It Works", para, para, para,
    ])
    _, hi = length_band("deep")

    # Reproduce the pipeline order: add required sections, then trim.
    ctx = {"evidence_distribution": {"A": 1, "B": 1, "C": 1, "D": 0}}
    body, _ = _add_required_sections(
        writer_draft,
        ctx=ctx,
        usable_facts=_facts(),
        contradictions=[],
        profile=select_profile(ctx, fact_count=len(_facts())),
    )
    assert _count_words(body) > hi
    trimmed = _trim_to_budget(body, hi)

    # Required sections for the resolved profile (plus any substantive
    # conditional ones) must all survive the trim.
    for heading in ("## Executive Summary", "## Key Findings", "## Evidence & Confidence",
                    "## Limitations & Unknowns"):
        assert heading in trimmed
    assert _count_words(trimmed) <= hi


def test_question_shaped_headings_are_shortened():
    """Regression: live deep reports shipped raw questions as headings
    ("## How did the FDIC's systemic risk exception ... ?"), duplicating the
    query and padding the answer. A heading must be a concise label."""
    from app.agents.synthesizer import _dedupe_heading, _shorten_heading

    assert _shorten_heading(
        "How did the FDIC's systemic risk exception for SVB and Signature "
        "(March 12, 2023) extend deposit protection?"
    ) != "How did the FDIC's systemic risk exception for SVB and Signature (March 12, 2023) extend deposit protection?"
    assert len(_shorten_heading("What Are Bangladesh's Projected Electricity Demand and Generation Requirements?").split()) <= 10
    assert "?" not in _shorten_heading("What caused the 2023 crisis?")

    body = (
        "## How did the FDIC extend deposit protection beyond the insured limit?\n\n"
        "FDIC used a systemic risk exception [1].\n\n"
        "## Key Findings\n\n- A finding [1].\n"
    )
    out = _dedupe_heading(body)
    heads = [ln for ln in out.splitlines() if ln.startswith("#")]
    assert all(not h.endswith("?") for h in heads)
    # A normal short heading is untouched.
    assert "## Key Findings" in out
