"""Answer-finishing + summarizer-resilience + confidence-cap (workstreams C/D).

C. Every synthesized report MUST contain Executive Summary, Key Findings,
   Evidence Strength, Limitations & Unknowns, and Counterarguments & Disputed
   Points — enforced POST-ASSEMBLY so the writer cannot omit any of them,
   including in the section-wise path.
D. The summarizer must parse fenced / prose-wrapped / alternate-key / single
   object JSON and retry once with a tightened instruction before falling back;
   a summarizer fallback that still produced verified, corroborated evidence
   must not force the 0.55 confidence cap, while a weak pool still does.
"""
import asyncio

from app.agents.synthesizer import (
    REQUIRED_SECTIONS,
    ensure_required_sections,
    synthesize,
)
from app.agents.outline import build_outline
from app.core.schemas import SummarizerFactsModel
from app.core.llm import LLMClient
from app.core.config import Settings


REQUIRED_TITLES = [
    "Executive Summary",
    "Key Findings",
    "Evidence Strength",
    "Limitations & Unknowns",
    "Counterarguments & Disputed Points",
    "Open Questions & Missing Angles",
    "Key Figures",
    "Auditable Source Ledger",
]


def _facts():
    return [
        {"claim": "Artificial intelligence simulates human intelligence in machines.",
         "source": "https://a.example/x", "confidence": 0.8, "verified": True,
         "sub_question": "AI definition"},
        {"claim": "Global AI spending reached 200 billion dollars in 2025.",
         "source": "https://b.example/y", "confidence": 0.7, "verified": True,
         "sub_question": "AI market data"},
        {"claim": "AI systems can encode societal bias at scale.",
         "source": "https://c.example/z", "confidence": 0.6, "verified": True,
         "sub_question": "AI risks"},
        {"claim": "Adoption is expected to keep rising through 2026.",
         "source": "https://d.example/w", "confidence": 0.5, "verified": True,
         "sub_question": "AI outlook"},
    ]


def _sub_questions():
    return [
        {"question": "AI definition", "axis": "definition"},
        {"question": "AI market data", "axis": "evidence"},
        {"question": "AI risks", "axis": "criticism"},
        {"question": "AI outlook", "axis": "outlook"},
    ]


# ---------------------------------------------------------------------------
# C. mandatory sections
# ---------------------------------------------------------------------------

class _MinimalWriter:
    """Writes only an Executive Summary and one outline section."""

    def __init__(self):
        self.calls = 0

    async def generate_json(self, system_prompt, user_prompt, **kwargs):
        self.calls += 1
        return {"answer": "A short synthesized section body citing a claim [1]."}


def test_every_synthesized_report_contains_five_required_sections():
    result = asyncio.run(
        synthesize(_MinimalWriter(), "What is the current trend of AI?", _facts(),
                   {"intent": {}, "sub_questions": _sub_questions()}, compress_context=False)
    )
    for title in REQUIRED_TITLES:
        assert f"## {title}" in result.answer, f"missing required section: {title}"


def test_section_wise_report_contains_five_required_sections():
    outline = build_outline("What is the current trend of AI?", _facts(), _sub_questions())
    assert outline.broad
    result = asyncio.run(
        synthesize(_MinimalWriter(), "What is the current trend of AI?", _facts(),
                   {"intent": {}, "sub_questions": _sub_questions()},
                   outline=outline, section_wise=True, compress_context=False)
    )
    for title in REQUIRED_TITLES:
        assert f"## {title}" in result.answer, f"missing required section: {title}"


def test_writer_omitting_limitations_gets_it_added():
    answer = "## Executive Summary\n\nA direct answer [1]."
    filled = ensure_required_sections(
        answer, ctx={"evidence_distribution": {"A": 2, "B": 1, "C": 1, "D": 0}},
        usable_facts=_facts(), contradictions=[],
    )
    assert "## Limitations & Unknowns" in filled
    assert "## Counterarguments & Disputed Points" in filled
    # Existing sections are never duplicated.
    assert filled.count("## Executive Summary") == 1


def test_existing_aliases_satisfy_the_requirement():
    answer = (
        "## Executive Summary\n\nx [1].\n\n"
        "## Key Findings\n\n- y [1].\n\n"
        "## Evidence & Confidence\n\nz [1].\n\n"
        "## Limitations\n\n- none measured.\n\n"
        "## Standing objections\n\n- none.\n\n"
        "## Open Questions & Missing Angles\n\n- none.\n\n"
        "## Key Figures\n\n- 40% [1].\n\n"
        "## Auditable Source Ledger\n\n- https://x (retrieved 2026-01-01).\n"
    )
    filled = ensure_required_sections(
        answer, ctx={}, usable_facts=[], contradictions=[],
    )
    # Every required heading is recognized via an alias: nothing appended.
    assert filled == answer


def test_required_section_aliases_cover_all_eight():
    # The uncertainty-first upgrade adds three mandatory sections (open
    # questions, key figures, auditable ledger) to the original five.
    assert set(REQUIRED_SECTIONS.keys()) == set(REQUIRED_TITLES)
    assert len(REQUIRED_SECTIONS) == 8


# ---------------------------------------------------------------------------
# D. summarizer JSON parsing resilience
# ---------------------------------------------------------------------------

def _clamp():
    return Settings(groq_api_key="k", _env_file=None)


def _payload_of(text):
    client = LLMClient(_clamp())
    return client._extract_json(text)


def test_extract_json_unwraps_markdown_fences():
    payload = _payload_of('```json\n{"facts": [{"claim": "c", "source": "u"}]}\n```')
    assert payload["facts"][0]["claim"] == "c"


def test_extract_json_tolerates_leading_and_trailing_prose():
    payload = _payload_of(
        'Here is the JSON you asked for:\n{"facts": [{"claim": "c", "source": "u"}]}\n'
        "Let me know if you need more."
    )
    assert payload["facts"][0]["source"] == "u"


def test_extract_json_wraps_bare_list_and_ignores_braces_in_strings():
    payload = _payload_of('[{"claim": "uses { and } braces", "source": "u"}]')
    assert payload["facts"][0]["claim"] == "uses { and } braces"


def test_schema_accepts_fenced_alternate_key_and_single_object():
    fenced = _payload_of('```\n{"results": [{"text": "A claim", "url": "https://x.com/a"}]}\n```')
    model = SummarizerFactsModel.model_validate(fenced)
    assert model.facts[0].claim == "A claim"
    assert model.facts[0].source == "https://x.com/a"

    single = SummarizerFactsModel.model_validate({"claim": "One fact", "source": "https://y.com/b"})
    assert len(single.facts) == 1
    assert single.facts[0].claim == "One fact"


def test_schema_rejects_genuinely_malformed_payload():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SummarizerFactsModel.model_validate({"nope": []})


class _FencedLLM:
    """Returns fenced/alternate-key JSON only through a raw generate_json.

    The summarizer calls `generate_json(response_model=...)`, so this double
    mimics the client by validating the same way; the point under test is that
    the schema + extract path accept the wrapped shape.
    """

    def __init__(self, tmp_path=None):
        db = str((tmp_path or "/tmp") / "fenced.db")
        self.settings = Settings(groq_api_key="k", database_url=db, _env_file=None)
        self.client = LLMClient(self.settings)
        self.calls = 0

    async def generate_json(self, system_prompt, user_prompt, response_model=None, **kwargs):
        self.calls += 1
        text = '```json\n{"results": [{"text": "Solar capacity grew 40 percent in 2024", ' \
               '"url": "https://solar.gov/report"}]}\n```'
        payload = self.client._extract_json(text)
        if response_model is not None:
            return response_model.model_validate(payload).model_dump()
        return payload


def test_summarizer_accepts_fenced_json_without_fallback(tmp_path):
    from app.agents.summarizer import summarizer_agent

    llm = _FencedLLM(tmp_path)
    results = [{
        "title": "Solar report",
        "url": "https://solar.gov/report",
        "snippet": "",
        "content": "Solar capacity grew 40 percent in 2024 according to the annual report.",
        "sub_question": "solar growth",
    }]
    facts = asyncio.run(summarizer_agent(llm, "solar capacity growth", results))
    assert facts
    assert facts[0]["extraction"] == "llm"
    assert facts[0]["source"] == "https://solar.gov/report"


class _RetryRecoverLLM:
    """First call returns unusable JSON; the tightened retry returns facts."""

    def __init__(self, tmp_path=None):
        db = str((tmp_path or "/tmp") / "retry.db")
        self.settings = Settings(groq_api_key="k", database_url=db, _env_file=None)
        self.calls = 0

    async def generate_json(self, system_prompt, user_prompt, response_model=None, **kwargs):
        self.calls += 1
        if "Return ONLY a single JSON object" in user_prompt:
            return {"facts": [{"claim": "Nuclear capacity reached 2,400 MW by 2026",
                               "source": "https://iaea.org/report", "confidence": 0.9}]}
        # First attempt: the model "answers" but the payload has no fact list.
        return {"facts": []}


def test_summarizer_tightened_retry_recovers_unusable_output(tmp_path):
    from app.agents.summarizer import summarizer_agent

    llm = _RetryRecoverLLM(tmp_path)
    results = [{
        "title": "IAEA report",
        "url": "https://iaea.org/report",
        "snippet": "",
        "content": "Nuclear capacity reached 2,400 MW by 2026 according to the agency report.",
        "sub_question": "nuclear capacity",
    }]
    facts = asyncio.run(summarizer_agent(llm, "nuclear capacity 2026", results))
    assert facts
    assert facts[0]["extraction"] == "llm"
    assert llm.calls == 2, "the tightened retry must run exactly once"


# ---------------------------------------------------------------------------
# D. confidence cap depends on evidence quality
# ---------------------------------------------------------------------------

def test_strong_corroborated_pool_is_not_capped_for_summarizer_fallback():
    from app.core.confidence import DEGRADED_CAP, compute_confidence

    facts = [
        {
            "claim": "Global solar capacity grew 40% in 2024",
            "source": "https://iea.org/a",
            "verified": True,
            "verification_score": 0.95,
            "corroboration_count": 2,
            "corroborating_sources": ["https://iea.org/a", "https://irena.org/b"],
        }
        for _ in range(6)
    ]
    result = compute_confidence(
        facts, {"is_sufficient": True}, 1, 3, degraded=["summarizer"],
    )
    assert result["overall"] > DEGRADED_CAP, "strong corroborated evidence must not be false-capped"
    assert not any("capped" in n for n in result["notes"])


def test_weak_pool_is_still_capped_for_summarizer_fallback():
    from app.core.confidence import DEGRADED_CAP, compute_confidence

    facts = [
        {"claim": f"Claim {i} about something", "source": f"https://blog{i}.com/a",
         "verified": True, "verification_score": 0.6}
        for i in range(6)
    ]
    result = compute_confidence(
        facts, {"is_sufficient": False}, 1, 3, degraded=["summarizer"],
    )
    assert result["overall"] <= DEGRADED_CAP
    assert any("capped" in n for n in result["notes"])


def test_synthesizer_fallback_always_caps():
    from app.core.confidence import DEGRADED_CAP, compute_confidence

    facts = [
        {
            "claim": "Global solar capacity grew 40% in 2024",
            "source": "https://iea.org/a",
            "verified": True,
            "corroboration_count": 3,
        }
    ]
    result = compute_confidence(
        facts, {"is_sufficient": True}, 1, 3, degraded=["synthesizer"],
    )
    assert result["overall"] <= DEGRADED_CAP
    assert any("capped" in n for n in result["notes"])


# ---------------------------------------------------------------------------
# E. Per-finding confidence + A/B/C grade, ledger warnings, counter-arg guard
# ---------------------------------------------------------------------------

def test_key_findings_carry_confidence_grade_and_justification():
    from app.agents.synthesizer import _render_finding_line

    corroborated = {
        "claim": "AI capex reached $200B in 2025", "citation": 3,
        "confidence": 0.9, "verified": True, "corroboration_count": 3,
        "evidence_grade": "A",
    }
    line = _render_finding_line(corroborated)
    assert "confidence" in line
    assert "grade A" in line
    assert "3 independent sources" in line
    assert "[3]" in line


def test_single_source_finding_is_capped_provisional():
    from app.agents.synthesizer import _render_finding_line, _finding_confidence

    provisional = {
        "claim": "A lone unverified claim", "citation": 5,
        "confidence": 0.95, "verified": False, "corroboration_count": 1,
    }
    assert _finding_confidence(provisional) <= 0.60
    assert "single source" in _render_finding_line(provisional)


def test_grade_absent_is_omitted_not_invented():
    from app.agents.synthesizer import _render_finding_line

    line = _render_finding_line(
        {"claim": "A plain factual claim here", "citation": 1, "confidence": 0.7,
         "verified": True, "corroboration_count": 2}
    )
    assert "grade " not in line


def test_ledger_warns_on_regulation_dominance():
    from app.agents.synthesizer import _ledger_warnings

    text = _ledger_warnings(
        {"regulation_share": 0.45, "non_western_share": 0.3, "primary_share": 0.5}
    )
    assert "45%" in text
    assert ">30%" in text


def test_ledger_silent_when_composition_healthy():
    from app.agents.synthesizer import _ledger_warnings

    assert _ledger_warnings(
        {"regulation_share": 0.1, "non_western_share": 0.3, "primary_share": 0.6}
    ) == ""


def test_counterargument_guard_never_claims_absence_without_search():
    from app.agents.synthesizer import _render_required_section

    unsearched = _render_required_section(
        "Counterarguments & Disputed Points",
        ctx={"counter_evidence_attempted": False},
        usable_facts=[], contradictions=[],
    )
    assert "UNKNOWN" in unsearched
    assert "No credible counterarguments" not in unsearched

    searched = _render_required_section(
        "Counterarguments & Disputed Points",
        ctx={"counter_evidence_attempted": True},
        usable_facts=[], contradictions=[],
    )
    assert "counter-evidence search was run" in searched
    assert "not proof none exists" in searched
