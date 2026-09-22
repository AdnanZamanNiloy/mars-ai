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
    MOVES,
    ClaimLedger,
    analytical_dimensions,
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
    """A pathological draft (everything a restatement) is neither emptied nor
    collapsed to its first section. When no topic-specific move applies the
    repeated sentence is left byte-for-byte unchanged — the layer must not
    invent a generic topic-agnostic tail to justify a transformation."""
    body = "The same fact is stated here in the opening [1]."
    answer = _report(("A", body), ("B", body), ("C", body))
    compressed, report = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert compressed.strip()
    assert "The same fact is stated here" in compressed
    for heading in ("## A", "## B", "## C"):
        assert heading in compressed
    # No topic-specific clause exists for this claim, so nothing was appended.
    assert report.refined_transitions == 0
    assert compressed.count("The same fact is stated here in the opening [1].") == 3

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
    """A generic financing tail must never be stapled onto a non-financial
    claim. A topic-specific claim is refined with its own scoped clause; a claim
    with NO applicable topic clause is left byte-for-byte unchanged."""
    transformer = refine_restatement(
        "The transformer removes recurrence from the sequence model [1].",
        dimension="implication",
    )
    cost = refine_restatement(
        "The plant costs about $13 billion [1].",
        dimension="implication",
    )
    # No topic-specific clause matches the transformer claim: unchanged, and
    # certainly no generic financing tail.
    assert transformer == "The transformer removes recurrence from the sequence model [1]."
    assert "financing" not in transformer.lower()
    # The cost claim has its own topic-scoped clause.
    assert "financing" in cost.lower()
    assert analytical_dimensions(cost) and "[1]" in cost


# ---------------------------------------------------------------------------
# Adaptive reasoning moves
# ---------------------------------------------------------------------------
# The generic "the figure is significant because …" tail this layer used to
# append is replaced by a MOVE chosen from the evidence signals. These tests
# pin the signal -> move mapping, verify the phrasing actually expresses the
# chosen move, and check the transformation stays deterministic, citation-safe,
# fact-safe and length-bounded.


def test_contradicted_claim_picks_uncertainty_or_tradeoff():
    """A disputed claim must hedge (uncertainty) rather than assert; when it is
    already a comparison it becomes an explicit trade-off."""
    plain = refine_restatement(
        "The plant cost $13 billion [1].",
        move="uncertainty",
    )
    assert "disputed" in plain or "provisional" in plain
    comparative = refine_restatement(
        "Solar is cheaper than nuclear [1].",
        move="tradeoff",
    )
    assert "cost" in comparative.lower() or "weighed" in comparative.lower()


def test_comparative_section_selects_comparison_move():
    """A 'How It Compares' section restatement must be framed as a comparison."""
    answer = _report(
        ("What It Is", "Solar capacity grew 20% in 2023 [1]."),
        ("How It Compares", "Solar capacity grew 20% in 2023 [1]."),
    )
    compressed, report = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert report.refined_transitions == 1


def test_decision_query_policy_claim_selects_strategic_move():
    """A decision/policy query turns a repeated policy claim into a strategic
    consequence move."""
    answer = _report(
        ("What It Is", "Bangladesh should expand nuclear capacity [1]."),
        ("Contribution", "Bangladesh should expand nuclear capacity [1]."),
    )
    compressed, report = apply_synthesis_intelligence(
        answer,
        protect_headings=MACHINE,
        signals={
            "query": "Should Bangladesh increase nuclear energy investment over 20 years?",
            "query_type": "analytical",
        },
    )
    assert report.refined_transitions == 1
    section = compressed.split("## Contribution", 1)[1].strip()
    assert "decision" in section.lower() or "changes the role" in section.lower()


def test_quantitative_authoritative_claim_selects_implication():
    """A quantitative claim from strong sourcing gets an implication move."""
    answer = _report(
        ("Evidence & Data", "The plant adds 1,200 MW of capacity [1]."),
        ("Applications", "The plant adds 1,200 MW of capacity [1]."),
    )
    compressed, report = apply_synthesis_intelligence(
        answer, protect_headings=MACHINE, signals={"authoritative": True}
    )
    assert report.refined_transitions == 1


def test_causal_query_selects_causal_move():
    """A 'what caused …' query turns a repeated fact into a causal move."""
    answer = _report(
        ("Evidence & Data", "Nuclear investment fell after 2011 [1]."),
        ("Outlook & Trends", "Nuclear investment fell after 2011 [1]."),
    )
    compressed, report = apply_synthesis_intelligence(
        answer,
        protect_headings=MACHINE,
        signals={"query": "What caused the decline in nuclear investment?"},
    )
    assert report.refined_transitions == 1


def test_each_move_is_reachable_and_signal_driven():
    """Every declared move is reachable from a real signal combination, and the
    chosen move is exactly the one the top-precedence rule implies."""
    from app.core.synthesis_intelligence import _choose_move

    cases = [
        ("uncertainty", "Rooppur costs $13 billion [1].", {"contradicted": True}),
        ("comparison", "Solar capacity grew 20% [1].", {"axis": "How It Compares"}),
        (
            "strategic",
            "Bangladesh should expand nuclear capacity [1].",
            {"query": "Should Bangladesh invest over 20 years?"},
        ),
        ("implication", "The plant adds 1,200 MW [1].", {"authoritative": True}),
        ("mechanism", "The reactor uses a water design [1].", {"axis": "How It Works"}),
        ("causal", "Investment fell in 2011 [1].", {"query": "What caused the fall?"}),
        ("tradeoff", "The plant adds 1,200 MW [1].", {}),
        ("comparison", "Solar is cheaper than nuclear [1].", {}),
    ]
    seen = set()
    for expected, sentence, signals in cases:
        move = _choose_move(sentence, signals)
        seen.add(move)
        assert move == expected, (sentence, signals, move)
    assert seen <= set(MOVES)
    assert "strategic" in seen and "mechanism" in seen and "causal" in seen


def test_adaptive_move_deterministic_without_llm_and_citations_preserved():
    """With no LLM and no signals the layer still produces a valid move; the
    citation survives and no new number or [n] marker is invented."""
    ledger = ClaimLedger()
    source = "Rooppur costs about $13 billion [1]."
    ledger.register("S1", source)
    outcome = ledger.refine("S2", source)
    assert outcome.transformed and outcome.move in MOVES
    assert "[1]" in outcome.text
    assert outcome.text.count("[1]") == 1
    assert "[2]" not in outcome.text and "[0]" not in outcome.text
    # No new figure appears: the only number in the refined text is the one
    # the source sentence carried.
    import re as _re

    src_numbers = set(_re.findall(r"\d[\d,\.]*", source))
    out_numbers = set(_re.findall(r"\d[\d,\.]*", outcome.text))
    assert out_numbers <= src_numbers, (src_numbers, out_numbers)
    # And determinism: the same inputs refine identically.
    again = ClaimLedger()
    again.register("S1", source)
    assert again.refine("S2", source).text == outcome.text


def test_no_person_or_entity_is_invented():
    """The move only reorganizes the claim; it never introduces a named entity
    that was not in the source sentence."""
    source = "The reactor design was licensed in 2017 [4]."
    refined = refine_restatement(source, move="mechanism")
    for invented in ("Rosatom", "IAEA", "Westinghouse", "Fukushima"):
        assert invented not in refined
    assert "[4]" in refined


def test_refinement_is_length_bounded():
    """A transformation must not inflate the sentence beyond a fixed bound over
    the source claim + a short clause."""
    from app.core.synthesis_intelligence import MAX_APPENDED_WORDS

    source = "Nuclear plants have high capacity factors."
    for move in MOVES:
        refined = refine_restatement(source, move=move)
        appended = len(refined.split()) - len(source.split())
        assert appended <= MAX_APPENDED_WORDS, (move, refined)
        assert refined.endswith(".")


def test_acceptance_class_strategic_transform_is_achievable():
    """The acceptance example class: a reliability claim from a nuclear plan is
    reframed as a strategic consequence (role change), not a generic clause."""
    source = "Nuclear plants have high capacity factors."
    refined = refine_restatement(source, move="strategic")
    assert "Nuclear plants have high capacity factors" in refined
    lowered = refined.lower()
    assert "changes" in lowered and "role" in lowered
    assert "dispatchable" in lowered and "stabilizer" in lowered
    # It is NOT the old templated tail.
    assert "the figure is significant because" not in lowered


def test_move_phrasing_matches_move_not_generic_template():
    """Different moves must produce different phrasings for the same claim."""
    source = "Rooppur costs about $13 billion [1]."
    outputs = {move: refine_restatement(source, move=move) for move in MOVES}
    assert len(set(outputs.values())) == len(MOVES)


# ---------------------------------------------------------------------------
# Generic-fallback removal — leave the sentence unchanged when no topic-specific
# move clause applies
# ---------------------------------------------------------------------------
# Live pilot evidence: on the mRNA query a genuine prose restatement about LNP
# delivery received the topic-agnostic tail "…, and the underlying mechanism is
# that the earlier conditions compound, which forces the trade-offs described
# here." Fewer generic clauses were appended to label bullets by the earlier
# guard, but the generic-fallback path itself was never removed. The refinement
# layer must only ever transform a sentence when it has a concrete,
# topic-scoped move for that claim; otherwise the sentence stays byte-for-byte
# unchanged.

_NO_TOPIC_SENTENCE = "The transformer removes recurrence from the sequence model [1]."


def test_prose_restatement_without_topic_move_is_byte_for_byte_unchanged():
    """No topic-specific clause matches this claim, so nothing is appended."""
    assert refine_restatement(
        _NO_TOPIC_SENTENCE, dimension="implication"
    ) == _NO_TOPIC_SENTENCE


def test_generic_fallback_clause_is_never_appended_end_to_end():
    """A repeated prose claim with no topic-specific move keeps its exact text;
    the old generic mechanism/implication tails never appear."""
    answer = _report(
        ("What It Is", _NO_TOPIC_SENTENCE),
        ("How It Works", _NO_TOPIC_SENTENCE),
    )
    compressed, report = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert compressed.count(_NO_TOPIC_SENTENCE) == 2
    assert report.refined_transitions == 0
    for clause in _CORRUPT_CLAUSES:
        assert clause not in compressed.lower(), clause
    # Specifically the generic implication/mechanism tails of _ANALYSIS_BY_MOVE.
    assert "constrains the choices the rest of the analysis depends on" not in compressed.lower()


def test_prose_restatement_with_topic_move_is_still_refined():
    """A claim WITH an applicable topic-scoped move is still transformed."""
    answer = _report(
        ("Cost", "The project costs about US$13 billion [1]."),
        ("Outlook", "The project costs about US$13 billion [1]."),
    )
    compressed, report = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert report.refined_transitions == 1
    assert compressed.count("[1]") == 2
    outlook = compressed.split("## Outlook", 1)[1].strip()
    assert "financing" in outlook.lower()
    assert outlook.split("\n")[0].strip().endswith(".")


def test_bullets_and_fragments_still_untouched_without_topic_move():
    """The no-generic-fallback change must not start refining bullets or
    fragments: they remain verbatim even when no topic-specific move exists."""
    bullet_prose = (
        "The first mRNA vaccines authorized in humans were the COVID-19 vaccines "
        "in 2020 [6]."
    )
    answer = _report(
        ("History", f"- {bullet_prose}"),
        ("Key Figures", f"- {bullet_prose}"),
        ("Notes", "A label fragment with no finite verb: reported by the lab [7]."),
    )
    compressed, report = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert f"- {bullet_prose}" in compressed
    assert "A label fragment with no finite verb: reported by the lab [7]." in compressed
    assert report.refined_transitions == 0
    for clause in _CORRUPT_CLAUSES:
        assert clause not in compressed.lower(), clause


# ---------------------------------------------------------------------------
# Bug 1 regression — adaptive-moves boilerplate injected into label bullets
# ---------------------------------------------------------------------------
# Live pilot evidence: on 4/6 queries the refinement pass appended generic move
# clauses into nearly every Key-Figures bullet, e.g.
#   "- SWE-bench Verified frontier performance: 60% rising to near 100% within
#    a single year [2], and the underlying mechanism is that the earlier
#    conditions compound, which forces the trade-offs described here."
# These are label/value bullets (a label, a colon, a value list), not prose
# restatements, so the refinement layer must leave them intact.

# The exact corrupted strings observed in the live report.
_CORRUPT_CLAUSES = (
    "the underlying mechanism is that the earlier conditions compound",
    "the cause behind that established point",
    "reading that figure for its consequences",
)


_LABEL_BULLETS = (
    "SWE-bench Verified frontier performance: 60% rising to near 100% within a single year [2].",
    "U.S.–China top-model performance gap: 2.7%, as of March 2026 [2].",
    "Organizational AI adoption: 88%; university students using generative AI: four in five [2].",
    "Sector AI automation risk scores: financial services 62, information technology 58, "
    "administrative & support services 55; healthcare support 24, construction 28, surgeons 8 [11].",
    "UpstreamBench v0.1 size: 7,440 questions across 372 technical reference books [4].",
    "German firm survey base: over 7,000 manufacturing and services firms, Q2 2025 [3].",
    "SpaceX–Cursor acquisition value: $60 billion in stock, expected to close Q3 2026 [14].",
)


def test_label_bullets_are_not_refined():
    """Bug 1: a label/value bullet (no finite verb before the delimiter) is a
    fragment, not a prose restatement, and must never receive a move clause."""
    from app.core.synthesis_intelligence import _is_refinable_sentence

    for bullet in _LABEL_BULLETS:
        assert _is_refinable_sentence(bullet) is False, bullet


def test_bold_label_bullet_is_not_refined():
    """Bug 1: a bold label + value/title bullet is left intact."""
    from app.core.synthesis_intelligence import _is_refinable_sentence

    assert _is_refinable_sentence(
        "**$12.65 billion** — reported cost of the project [1]."
    ) is False


def test_label_bullets_are_not_corrupted_end_to_end():
    """Bug 1 acceptance: a Key Figures section repeated across sections keeps
    every label bullet verbatim; no corrupt clause is injected anywhere."""
    body = "\n".join(f"- {b}" for b in _LABEL_BULLETS)
    answer = _report(("Key Figures", body), ("More Figures", body))
    compressed, report = apply_synthesis_intelligence(answer, protect_headings=MACHINE)

    # No label bullet was transformed and none was dropped.
    assert report.refined_transitions == 0
    for bullet in _LABEL_BULLETS:
        assert bullet in compressed, bullet
    for clause in _CORRUPT_CLAUSES:
        assert clause not in compressed.lower(), clause


def test_enumerable_bullet_is_never_refined_even_when_full_sentence():
    """Bug 1 acceptance (live mRNA/transformer case): a Key Figures bullet whose
    value is a full sentence is a data item, not a prose restatement, and must
    be left verbatim. This was the exact residual corruption after the label
    guard landed: 'The first mRNA vaccines … in 2020 [6]' inside a figure list
    received a move clause because it parsed as a sentence."""
    fact = "The first mRNA vaccines authorized in humans were the COVID-19 vaccines in 2020 [6]."
    answer = _report(
        ("Executive Summary", fact),
        ("History", f"- {fact}"),
        ("Key Figures", f"- {fact}"),
    )
    compressed, report = apply_synthesis_intelligence(
        answer, protect_headings=MACHINE, recap_headings=RECAP
    )
    # The bullet itself is never rewritten.
    assert f"- {fact}" in compressed
    assert report.refined_transitions == 0
    for clause in _CORRUPT_CLAUSES:
        assert clause not in compressed.lower(), clause


def test_prose_paragraph_restatement_is_still_refined_not_bullet():
    """The refinement layer still transforms genuine prose paragraphs (the
    feature it exists for) — the bullet rule must not disable it entirely."""
    answer = _report(
        ("Executive Summary", "Rooppur is Bangladesh's first nuclear plant [1]."),
        ("Cost", "Rooppur is Bangladesh's first nuclear plant [1]. Financing terms shape the tariff."),
    )
    compressed, report = apply_synthesis_intelligence(
        answer, protect_headings=MACHINE, recap_headings=RECAP
    )
    assert report.refined_transitions == 1


def test_genuine_prose_restatement_is_still_refined():
    """Bug 1 guard must not over-reject: a real prose restatement WITH a
    topic-specific move available is still refined. A prose restatement with no
    applicable topic clause is instead left byte-for-byte unchanged (see the
    generic-fallback removal test below)."""
    from app.core.synthesis_intelligence import _is_refinable_sentence

    prose = (
        "The project costs about US$13 billion and the financing shapes the "
        "tariff for decades [1]."
    )
    assert _is_refinable_sentence(prose) is True
    ledger = ClaimLedger()
    ledger.register("S1", prose)
    outcome = ledger.refine("S2", prose)
    assert outcome.transformed is True


def test_already_refined_sentence_is_not_refined_twice():
    """Bug 1: a sentence that already carries a move transition must not be
    refined again (no double-appending)."""
    from app.core.synthesis_intelligence import _is_refinable_sentence

    refined = refine_restatement(
        "Rooppur costs about US$13 billion [3].",
        move="implication",
    )
    assert _is_refinable_sentence(refined) is False
    # End-to-end: re-running the layer on its own output is idempotent.
    answer = _report(("What It Is", refined), ("Cost", refined))
    compressed, _ = apply_synthesis_intelligence(answer, protect_headings=MACHINE)
    assert compressed.count("Building on") <= 2  # the two originals, no new ones
    for clause in _CORRUPT_CLAUSES:
        assert clause not in compressed.lower()


# ---------------------------------------------------------------------------
# Bug 2 regression — off-topic source bleed into Limitations/evidence
# ---------------------------------------------------------------------------
# Live pilot evidence: MARS's Limitations list carried unrelated items (Iran
# nuclear, NBER clientelism, Shell PLC, SpaceX) on a "What is a transformer?"
# query. Limitations and evidence scoring must draw only from the query's own
# evidence pool.

_OFF_TOPIC_FACTS = (
    "It houses a heavily fortified tunnel facility associated with Iran's nuclear program [3].",
    "NBER WORKING PAPER SERIES CLIENTELISM: HOW IT WORKS, WHY IT PERSISTS AND HOW TO BREAK IT [4].",
    "However, a core challenge is that the actions and competence of politicians are not "
    "fully observable to citizens and the paper models that agency problem [5].",
)

_ON_TOPIC_FACTS = (
    "The transformer uses multi-head self-attention to process sequences [1].",
    "A transformer changes AC voltage levels through electromagnetic induction [2].",
)


def test_coverage_gaps_exclude_off_topic_facts():
    """Bug 2: gaps for Limitations name only claims from the query's own pool."""
    from app.graph.workflow import _query_inscope_facts

    facts = [{"claim": c} for c in (*_ON_TOPIC_FACTS, *_OFF_TOPIC_FACTS)]
    inscope = _query_inscope_facts(
        facts,
        "What is a transformer?",
        ["Transformer neural network architecture", "Electrical transformer"],
    )
    kept = {f["claim"] for f in inscope}
    assert set(_ON_TOPIC_FACTS) <= kept
    for off in _OFF_TOPIC_FACTS:
        assert off not in kept, off


def test_measured_coverage_gaps_have_no_off_topic_sources():
    """Bug 2 end-to-end: the Limitations gap strings never name the off-topic
    Iran/NBER/politics claims on a transformer query."""
    from app.graph.workflow import _measured_coverage_gaps

    state = {
        "query": "What is a transformer?",
        "intent": {
            "senses": [
                {"label": "Transformer neural network architecture"},
                {"label": "Electrical transformer"},
            ]
        },
        "facts": [
            {
                "claim": c,
                "source": "https://example.org/x",
                "corroboration_count": 1,
                "verified": True,
            }
            for c in (*_ON_TOPIC_FACTS, *_OFF_TOPIC_FACTS)
        ],
    }
    gaps = " ".join(_measured_coverage_gaps(state)).lower()
    assert gaps  # a single-source pool still reports honest gaps
    for token in ("iran", "nber", "clientelism", "politicians"):
        assert token not in gaps, (token, gaps)


def test_inscope_filter_never_starves_limitations():
    """Bug 2 safety: if nothing clears the relevance bar, the original pool is
    returned rather than emptied (AGENTS.md 4.7 — no starved section)."""
    from app.graph.workflow import _query_inscope_facts

    facts = [{"claim": c} for c in _OFF_TOPIC_FACTS]
    assert _query_inscope_facts(facts, "What is a transformer?", []) == facts
