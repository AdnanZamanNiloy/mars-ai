"""Synthesis intelligence layer — cross-section repetition refinement.

Regression targets (measured on real deep reports):
  * the same evidence fact re-asserted in section after section instead of
    being expanded (e.g. "Rooppur … US$13 billion [1]" opened ~7 sections of
    the Bangladesh report);
  * a repeated claim being silently DELETED so a section opened mid-argument
    with a dangling or missing first sentence;
  * a fact repeated in a later section with NO added analysis;
  * citation markers dropped or invented while transforming a fact into
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
    refine_restatement,
)

MACHINE = ("## Sources", "## Evidence integrity", "## Source ledger")
RECAP = ("## Executive Summary", "## Key Findings")


def _report(*sections):
    return "\n\n".join(f"## {title}\n\n{body}" for title, body in sections)


def test_repeated_claim_across_two_sections_second_is_refined_not_deleted():
    """A verbatim restatement of a fact already stated must be TRANSFORMED into
    a contextual transition, never silently removed."""
    answer = _report(
        ("What It Is", "Rooppur is Bangladesh's first nuclear plant [1]."),
        ("Cost", "Rooppur is Bangladesh's first nuclear plant [1]. "
                 "The plant anchors the national supply debate."),
    )
    compressed, report = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert report.refined_transitions == 1
    # The repeated claim is retained (its citation intact) inside a new
    # transition sentence, so the section still begins with a complete sentence.
    assert compressed.count("Rooppur is Bangladesh's first nuclear plant") == 2
    assert "[1]" in compressed
    cost_body = compressed.split("## Cost", 1)[1].strip()
    assert cost_body.split("\n")[0].strip().endswith("."), cost_body
    assert "Building on" in compressed or "Extending that" in compressed or "Taking that" in compressed
    # The unique (non-repeating) sentence in the second section survives.
    assert "anchors the national supply debate" in compressed


def test_refinement_adds_analytical_marker():
    """The transformed sentence must carry one of implication / mechanism /
    trade-off / comparison / uncertainty — a bare restatement has none."""
    answer = _report(
        ("What It Is", "Rooppur is Bangladesh's first nuclear plant [1]."),
        ("Cost", "Rooppur is Bangladesh's first nuclear plant [1]."),
    )
    compressed, _ = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    cost_body = compressed.split("## Cost", 1)[1].strip()
    assert analytical_dimensions(cost_body), cost_body


def test_refinement_retains_number_and_citation():
    """The specific number and its [n] survive the transformation; no marker
    is invented or renumbered."""
    answer = _report(
        ("What It Is", "Rooppur costs about US$13 billion [3]."),
        ("Cost", "Rooppur costs about US$13 billion [3]."),
    )
    compressed, report = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert report.refined_transitions == 1
    assert "13 billion" in compressed
    assert compressed.count("[3]") == 2
    assert "[4]" not in compressed and "[2]" not in compressed


def test_refinement_produces_transition_without_llm():
    """The deterministic path must yield a usable transition with no LLM."""
    refined = refine_restatement(
        "The project costs about $13 billion [1].",
        prior_section="What It Is",
        dimension="implication",
    )
    assert refined.endswith(".")
    assert "$13 billion" in refined and "[1]" in refined
    assert analytical_dimensions(refined)
    # A transition cue ties the sentence back to the earlier use.
    lower = refined.lower()
    assert any(cue in lower for cue in ("building on", "extending", "taking", "reading"))


def test_repeat_with_new_mechanism_is_left_intact():
    """Re-using a claim IS allowed when it adds a mechanism the first use did
    not — the whole point is expansion, not suppression, so the text is not
    rewritten."""
    answer = _report(
        ("What It Is", "Bangladesh's grid relies on imported LNG [2]."),
        ("How It Works", "Bangladesh's grid relies on imported LNG [2] because "
                         "domestic gas fields are depleting faster than "
                         "replacements can be commissioned [3]."),
    )
    compressed, report = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert report.refined_transitions == 0
    assert report.allowed_expansions == 1
    # Both the fact and the added mechanism are present, neither rewritten.
    assert compressed.count("imported LNG") == 2
    assert "domestic gas fields are depleting" in compressed
    assert "Building on" not in compressed


def test_repeat_with_new_tradeoff_or_comparison_is_allowed():
    """An implication/trade-off or comparison dimension also counts as expansion."""
    answer = _report(
        ("Cost", "Solar is the cheapest new-build generation [4]."),
        ("Outlook", "Solar is the cheapest new-build generation [4], but its "
                    "comparatively low capacity factor means firm backup is "
                    "required, which raises the delivered cost [5]."),
    )
    _, report = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert report.refined_transitions == 0
    assert report.allowed_expansions == 1


def test_paraphrased_restatement_is_caught():
    """Different wording, same fact: the anchor match must still catch it and
    refine it, retaining the shared figure and citation."""
    answer = _report(
        ("What It Is",
         "Rooppur, the country's first nuclear project, was built by Rosatom at "
         "an estimated cost close to US$13 billion [1]."),
        ("Applications",
         "Bangladesh's only nuclear build, the Rosatom-constructed Rooppur "
         "plant, carries an estimated cost of close to US$13 billion [1]."),
    )
    compressed, report = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert report.refined_transitions == 1
    # Both the original fact and its refined restatement retain the figure and
    # the citation; nothing was dropped.
    assert compressed.count("13 billion") == 2
    assert compressed.count("[1]") == 2


def test_negated_claim_is_not_treated_as_repeat():
    """Polarity guard: "X" and "not X" are different claims (AGENTS.md)."""
    answer = _report(
        ("Cost", "Nuclear is cost-competitive with solar [1]."),
        ("Critique", "Nuclear is not cost-competitive with solar [2]; the "
                     "levelized comparison is the crux of the debate."),
    )
    _, report = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert report.refined_transitions == 0
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
    # The repeat is refined; no marker is invented — still only [1].
    assert "[1]" in compressed
    assert "[2]" not in compressed
    assert "[0]" not in compressed


def test_deterministic_fallback_refines_verbatim_repeat():
    """The refinement must work with no LLM at all — it is the fallback."""
    ledger = ClaimLedger()
    assert ledger.register("S1", "Global renewables capacity grew 50% in 2023 [1].") is True
    # Verbatim repeat in a later section with no new dimension: refined into a
    # transition, kept (never deleted).
    outcome = ledger.refine("S2", "Global renewables capacity grew 50% in 2023 [1].")
    assert outcome.kept is True
    assert outcome.transformed is True
    assert outcome.text != "Global renewables capacity grew 50% in 2023 [1]."
    assert "50%" in outcome.text and "[1]" in outcome.text
    assert analytical_dimensions(outcome.text)
    assert ledger.refined_restatements == 1
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
    # Executive Summary keeps its sentence verbatim; the Cost section's copy is
    # refined into a transition (the fact and its citation remain).
    assert compressed.count("The project cost US$13 billion") == 2
    assert "## Executive Summary" in compressed
    assert "## Cost" in compressed
    assert report.refined_transitions == 1


def test_compression_never_empties_the_report():
    """A pathological draft (everything a restatement) is refined in place
    rather than emptied or collapsed to its first section."""
    body = "The same fact is stated here in the opening [1]."
    answer = _report(("A", body), ("B", body), ("C", body))
    compressed, report = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert compressed.strip()
    assert "The same fact is stated here" in compressed
    for heading in ("## A", "## B", "## C"):
        assert heading in compressed
    assert report.refined_transitions == 2


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


def test_section_opening_repeat_becomes_complete_transition_sentence():
    """The acceptance case: a section opening that duplicates an earlier claim
    is replaced by a transition, and the section still begins with a complete
    sentence (no dangling opening, no fragment)."""
    answer = _report(
        ("Executive Summary", "Rooppur Nuclear Power Plant costs about $13 billion [1]."),
        ("Cost", "Rooppur Nuclear Power Plant costs about $13 billion [1]. "
                 "Financing terms shape the tariff."),
    )
    compressed, report = apply_synthesis_intelligence(
        answer, protect_headings=MACHINE, recap_headings=RECAP
    )
    assert report.refined_transitions == 1
    cost_body = compressed.split("## Cost", 1)[1].strip()
    opening = cost_body.split("\n")[0].strip()
    # Complete sentence, not a fragment, and it keeps the number + citation.
    assert opening and opening.endswith(".")
    assert len(opening.split()) >= 5
    assert "$13 billion" in opening and "[1]" in opening
    assert analytical_dimensions(opening)


def test_contradiction_signal_selects_uncertainty_dimension():
    """With a contradiction in scope the refinement must hedge, not assert."""
    answer = _report(
        ("Cost", "The plant cost $13 billion [1]."),
        ("Critique", "The plant cost $13 billion [1]."),
    )
    compressed, report = apply_synthesis_intelligence(
        answer, protect_headings=MACHINE, signals={"contradicted": True}
    )
    assert report.refined_transitions == 1
    critique = compressed.split("## Critique", 1)[1].strip()
    assert "uncertainty" in analytical_dimensions(critique)
    assert "disputed" in critique or "provisional" in critique


def test_analysis_clause_is_scoped_to_the_claim_topic():
    """A generic financing tail must not be stapled onto a non-financial claim;
    the clause is scoped to the sentence's subject vocabulary."""
    transformer = refine_restatement(
        "The transformer removes recurrence from the sequence model [1].",
        dimension="implication",
    )
    cost = refine_restatement(
        "The plant costs about $13 billion [1].",
        dimension="implication",
    )
    assert "financing" not in transformer.lower()
    assert "financing" in cost.lower()
    # Both still carry an analytical marker and the original citation.
    assert analytical_dimensions(transformer) and "[1]" in transformer
    assert analytical_dimensions(cost) and "[1]" in cost
