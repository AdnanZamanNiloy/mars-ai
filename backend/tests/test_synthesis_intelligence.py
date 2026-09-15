"""Synthesis intelligence layer — cross-section repetition tracking.

Regression targets (measured on real deep reports):
  * the same evidence fact re-asserted in section after section instead of
    being expanded (e.g. "Rooppur … US$13 billion [1]" opened ~7 sections of
    the Bangladesh report);
  * a fact repeated in a later section with NO added analysis;
  * citation makers dropped or invented while transforming a fact into
    reasoning.

The layer is deterministic (claim-key tracking + a fuzzy anchor match), so
these tests exercise it with no LLM at all — the fallback IS the mechanism.
"""
from app.core.synthesis_intelligence import (
    ClaimLedger,
    analytical_dimensions,
    analyze_report,
    apply_synthesis_intelligence,
    claim_key,
    claim_polarity,
)

MACHINE = ("## Sources", "## Evidence integrity", "## Source ledger")
RECAP = ("## Executive Summary", "## Key Findings")


def _report(*sections):
    return "\n\n".join(f"## {title}\n\n{body}" for title, body in sections)


def test_repeated_claim_across_two_sections_second_is_removed():
    """A verbatim restatement of a fact already stated must not survive."""
    answer = _report(
        ("What It Is", "Rooppur is Bangladesh's first nuclear plant [1]."),
        ("Cost", "Rooppur is Bangladesh's first nuclear plant [1]. "
                 "The plant anchors the national supply debate."),
    )
    compressed, report = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert report.removed_restatements == 1
    # The first occurrence (and its citation) is kept, the repeat is gone.
    assert compressed.count("Rooppur is Bangladesh's first nuclear plant") == 1
    assert "[1]" in compressed
    # The unique (non-repeating) sentence in the second section survives.
    assert "anchors the national supply debate" in compressed


def test_repeat_with_new_mechanism_is_allowed():
    """Re-using a claim IS allowed when it adds a mechanism the first use did
    not — the whole point is expansion, not suppression."""
    answer = _report(
        ("What It Is", "Bangladesh's grid relies on imported LNG [2]."),
        ("How It Works", "Bangladesh's grid relies on imported LNG [2] because "
                         "domestic gas fields are depleting faster than "
                         "replacements can be commissioned [3]."),
    )
    compressed, report = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert report.removed_restatements == 0
    assert report.allowed_expansions == 1
    # Both the fact and the added mechanism are present.
    assert compressed.count("imported LNG") == 2
    assert "domestic gas fields are depleting" in compressed


def test_repeat_with_new_tradeoff_or_comparison_is_allowed():
    """An implication/trade-off or comparison dimension also counts as expansion."""
    answer = _report(
        ("Cost", "Solar is the cheapest new-build generation [4]."),
        ("Outlook", "Solar is the cheapest new-build generation [4], but its "
                    "comparatively low capacity factor means firm backup is "
                    "required, which raises the delivered cost [5]."),
    )
    _, report = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert report.removed_restatements == 0
    assert report.allowed_expansions == 1


def test_paraphrased_restatement_is_caught():
    """Different wording, same fact: the anchor match must still catch it."""
    answer = _report(
        ("What It Is",
         "Rooppur, the country's first nuclear project, was built by Rosatom at "
         "an estimated cost close to US$13 billion [1]."),
        ("Applications",
         "Bangladesh's only nuclear build, the Rosatom-constructed Rooppur "
         "plant, carries an estimated cost of close to US$13 billion [1]."),
    )
    compressed, report = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert report.removed_restatements == 1
    assert compressed.count("13 billion") == 1
    assert "[1]" in compressed


def test_negated_claim_is_not_treated_as_repeat():
    """Polarity guard: "X" and "not X" are different claims (AGENTS.md)."""
    answer = _report(
        ("Cost", "Nuclear is cost-competitive with solar [1]."),
        ("Critique", "Nuclear is not cost-competitive with solar [2]; the "
                     "levelized comparison is the crux of the debate."),
    )
    _, report = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert report.removed_restatements == 0
    assert claim_polarity("Nuclear is not cost-competitive") == -1
    assert claim_polarity("Nuclear is cost-competitive") == 1


def test_citations_preserved_on_retained_facts():
    """A retained fact keeps its exact marker; no marker is dropped."""
    answer = _report(
        ("What It Is", "The transformer uses self-attention [3][7]."),
        ("How It Works",
         "The transformer uses self-attention to connect distant positions "
         "directly, which removed the sequential bottleneck [3][7]."),
    )
    compressed, _ = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert "[3][7]" in compressed
    # Both uses (the fact and its mechanism expansion) are retained.
    assert compressed.count("[3][7]") == 2


def test_no_citation_markers_invented():
    """The layer must never add a [n] that was not in the source text."""
    answer = _report(
        ("What It Is", "Coal remains a major source [1]."),
        ("Cost", "Coal remains a major source."),
    )
    compressed, _ = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    # The repeat is removed; the only marker present is the original [1].
    assert "[1]" in compressed
    assert "[2]" not in compressed
    assert "[0]" not in compressed


def test_deterministic_fallback_removes_verbatim_repeat():
    """The compressor must work with no LLM at all — it is the fallback."""
    ledger = ClaimLedger()
    assert ledger.register("S1", "Global renewables capacity grew 50% in 2023 [1].") is True
    # Verbatim repeat in a later section with no new dimension: rejected.
    assert ledger.register("S2", "Global renewables capacity grew 50% in 2023 [1].") is False
    assert ledger.removed_restatements == 1
    assert ledger.unique_claims == 1


def test_recap_sections_are_kept_but_registered():
    """Executive Summary / Key Findings previews stay whole, but deep-dive
    sections may not restate them."""
    answer = _report(
        ("Executive Summary", "The project cost US$13 billion [1]."),
        ("Cost", "The project cost US$13 billion [1]."),
    )
    compressed, report = apply_synthesis_intelligence(
        answer, protect_headings=MACHINE, recap_headings=RECAP
    )
    # Executive Summary keeps its sentence; the Cost section's copy is removed.
    assert compressed.count("The project cost US$13 billion") == 1
    assert "## Executive Summary" in compressed
    assert "## Cost" in compressed
    assert report.removed_restatements == 1


def test_compression_never_empties_the_report():
    """A pathological draft (everything a restatement) falls back to the
    original rather than shipping an empty body."""
    body = "Same fact stated here [1]."
    answer = _report(("A", body), ("B", body), ("C", body))
    compressed, _ = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert compressed.strip()
    assert "Same fact stated here" in compressed


def test_analyze_report_measures_redundancy_without_mutating():
    """The benchmark score: repeated claims over unique claims, non-destructive."""
    answer = _report(
        ("What It Is", "Rooppur is Bangladesh's first nuclear plant [1]."),
        ("Cost", "Rooppur is Bangladesh's first nuclear plant [1]."),
        ("Outlook", "Rooppur is Bangladesh's first nuclear plant [1]."),
    )
    before = answer
    report = analyze_report(answer)
    assert answer == before  # measurement never edits
    assert report.repeated_claims >= 1
    assert report.redundancy_ratio > 0.0
    assert report.unique_claims >= 1


def test_analytical_dimension_detection():
    """The four expansion dimensions are detected deterministically."""
    assert "mechanism" in analytical_dimensions("It rose because fuel costs fell.")
    assert "implication" in analytical_dimensions("Costs therefore rise for consumers.")
    assert "comparison" in analytical_dimensions("Solar is cheaper than nuclear.")
    assert "uncertainty" in analytical_dimensions("The outcome remains unresolved.")
    assert analytical_dimensions("Solar capacity grew 20% in 2023.") == set()


def test_claim_key_ignores_citations_and_order():
    a = claim_key("The plant cost 13 billion dollars [1].")
    b = claim_key("The plant cost 13 billion dollars [4].")
    assert a == b
    assert a  # not empty for a real claim


def test_short_transition_units_have_no_claim_key():
    assert claim_key("In short.") == ""
    assert claim_key("Taken together.") == ""
