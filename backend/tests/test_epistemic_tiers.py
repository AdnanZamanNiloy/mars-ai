"""Epistemic tiers (Step 4): the measured evidence-grade distribution drives
both the writer's tier guidance and the report's deterministic grade appendix.
Pure and deterministic — no LLM, no network."""
from app.agents.synthesizer import _measured_evidence_block, _render_context_block


DIST = {"A": 2, "B": 1, "C": 1, "D": 0}


# --- prompt-side tier block --------------------------------------------------

def test_context_block_renders_tiers_and_counts():
    block = _render_context_block({"evidence_distribution": DIST})
    assert "ESTABLISHED" in block
    assert "UNKNOWN" in block
    assert "A 2" in block
    assert "B 1" in block
    assert "C 1" in block
    assert "D 0" in block


def test_context_block_absent_without_distribution():
    assert "ESTABLISHED" not in _render_context_block({})
    assert "UNKNOWN" not in _render_context_block({})


def test_context_block_absent_when_all_zero():
    block = _render_context_block({"evidence_distribution": {"A": 0, "B": 0, "C": 0, "D": 0}})
    assert "ESTABLISHED" not in block
    assert "UNKNOWN" not in block


# --- appendix-side grade summary --------------------------------------------

def test_measured_block_includes_grade_summary_when_nonzero():
    out = _measured_evidence_block({"evidence_distribution": DIST}, [], [])
    assert "Evidence grades:" in out
    assert "A=2" in out
    assert "B=1" in out
    assert "C=1" in out
    assert "D=0" in out


def test_measured_block_omits_grade_summary_when_absent():
    out = _measured_evidence_block({}, [], [])
    assert "Evidence grades:" not in out


def test_measured_block_omits_grade_summary_when_all_zero():
    out = _measured_evidence_block(
        {"evidence_distribution": {"A": 0, "B": 0, "C": 0, "D": 0}}, [], []
    )
    assert "Evidence grades:" not in out
