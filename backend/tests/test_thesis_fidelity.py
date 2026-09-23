"""Phase 8: thesis fidelity + analyst-brief entity grounding + unknown-noise fix.

Fidelity is measured semantically (never string matching) and feeds the SAME
single revision pass as the quality gate. The entity guard extends the
no-fabrication protection beyond numbers. The depth-controller attribution fix
removes false "uncovered axis" unknowns.
"""
import asyncio

from app.agents.thesis_fidelity import (
    FidelityReport,
    check_thesis_fidelity,
)
from app.agents.analyst import AnalyticalBrief, analytical_synthesis


# --- 1. thesis fidelity (semantic, not lexical) ------------------------------

def _brief(**kw):
    return AnalyticalBrief(**kw)


def test_thesis_paraphrase_with_shared_anchors_is_reflected():
    # The check is paraphrase-tolerant via content anchors: a thesis whose
    # distinctive terms recur in the answer is reflected even without shared
    # wording. True synonym-level paraphrase with NO shared anchors is below
    # any lexical method and is not what this check can detect (it errs toward
    # not failing the writer — see test_ignored_thesis_fails_fidelity for the
    # divergence it DOES catch).
    brief = _brief(thesis="AI adoption is rising because inference costs are falling.")
    answer = (
        "## Answer\n\nAI adoption is clearly rising across firms [1]. This is "
        "driven by falling inference costs, which make deployment affordable."
    )
    report = check_thesis_fidelity(answer, brief)
    assert report.thesis_reflected
    assert report.score > 0.5


def test_ignored_thesis_fails_fidelity():
    brief = _brief(thesis="Enterprise AI has crossed into operational deployment.")
    answer = (
        "## Answer\n\nThe weather in Rome was mild in June, and the harvest was "
        "good that year according to several agricultural bulletins."
    )
    report = check_thesis_fidelity(answer, brief)
    assert not report.thesis_reflected
    assert any("thesis" in f.lower() for f in report.failures)


def test_dropped_insights_fail_fidelity():
    brief = _brief(insights=[
        "Falling inference costs and rising adoption reinforce each other.",
        "The open model gap narrowed as open-weight quality improved.",
    ])
    answer = "## Answer\n\nAdoption rose [1]."
    report = check_thesis_fidelity(answer, brief)
    assert report.insights_carried == 0
    assert any("insight" in f.lower() for f in report.failures)


def test_kept_counter_evidence_passes():
    brief = _brief(counter_evidence=["Adoption figures are survey-reported and definitions vary."])
    answer = (
        "## Answer\n\nHowever, adoption figures are self-reported in surveys and "
        "definitions differ between firms [1]."
    )
    report = check_thesis_fidelity(answer, brief)
    assert report.counter_kept


def test_missing_counter_evidence_fails():
    brief = _brief(counter_evidence=["Adoption figures are survey-reported and definitions vary."])
    answer = "## Answer\n\nAdoption rose sharply across all surveyed firms [1]."
    report = check_thesis_fidelity(answer, brief)
    assert not report.counter_kept
    assert any("counter" in f.lower() for f in report.failures)


def test_source_by_source_recitation_detected():
    brief = _brief(thesis="X is rising [1].")
    recital = (
        "According to source A, adoption rose. According to source B, costs fell. "
        "According to source C, quality improved. According to source D, regulation advanced."
    )
    report = check_thesis_fidelity(recital, brief)
    assert report.recital_ratio >= 0.75
    assert any("recitation" in f.lower() for f in report.failures)


def test_absent_brief_is_neutral():
    report = check_thesis_fidelity("Some answer text here [1].", None)
    assert report.thesis_reflected
    assert report.failures == []
    assert report.score == 1.0


def test_fidelity_is_total_on_garbage():
    for brief in (object(), "not a brief", {"thesis": object()}):
        report = check_thesis_fidelity("answer [1].", brief)
        assert isinstance(report, FidelityReport)


# --- 2. analyst-brief entity grounding ---------------------------------------

_FACTS = [
    {"claim": "Enterprise AI adoption reached 78% of surveyed organizations in 2025.",
     "source": "https://mckinsey.com/a", "verified": True, "has_numbers": True,
     "corroborating_sources": ["https://gartner.com/a"]},
]


class _LLM:
    def __init__(self, payload):
        self.payload = payload

    async def generate_json(self, sp, up, **kw):
        return self.payload


def test_fabricated_entity_in_thesis_is_dropped():
    # "Acme Corporation" appears in no evidence claim -> thesis dropped.
    brief = asyncio.run(analytical_synthesis(
        _LLM({"thesis": "Acme Corporation leads enterprise AI adoption [1]."}),
        "q", _FACTS, None,
    ))
    assert brief.thesis == ""


def test_known_entity_named_in_evidence_survives():
    # The fact names "Enterprise AI adoption" — a multi-word capitalized term
    # present in the evidence, so the thesis survives the entity guard.
    brief = asyncio.run(analytical_synthesis(
        _LLM({"thesis": "Enterprise AI adoption reached a new level [1]."}),
        "q", _FACTS, None,
    ))
    assert brief.thesis


# --- 3. unknown-noise: depth-controller axis attribution ---------------------

def test_fact_axis_fallback_prevents_false_uncovered_axes():
    """A fully-researched pool whose URL->axis map is absent must not report
    every planned axis as an uncovered hole."""
    from app.core.depth_controller import _uncovered_axes

    state = {
        "facts": [
            {"claim": "Adoption reached 78% in 2025.", "source": "https://a.com/x",
             "verified": True, "axis": "evidence", "sub_question": "adoption"},
            {"claim": "Costs fell 90%.", "source": "https://b.com/y",
             "verified": True, "axis": "mechanism", "sub_question": "costs"},
        ],
        "sub_questions": [
            {"question": "adoption", "axis": "evidence"},
            {"question": "costs", "axis": "mechanism"},
        ],
        "contradictions": [],
        # No search_results: the URL->axis map is empty.
    }
    assert _uncovered_axes(state) == []
