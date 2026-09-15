"""Per-section ranked context selection (GPT Researcher ContextCompressor
adaptation, implemented on the existing TF-IDF hybrid engine).

Regression targets:
  * a section's evidence was its AXIS-GROUPED pool, so a section could receive
    claims relevant to a different dimension (off-topic evidence),
  * pure similarity ranking would drop a mid-relevance quantitative or
    corroborated claim in favour of near-identical tangential restatements,
  * a selector must never mutate the input pool, never starve a section when
    it fails, and always preserve full claim dicts for citation traceability.

Deterministic — no LLM, no network.
"""
from app.agents.outline import OutlineSection, build_outline
from app.core.section_context import (
    DEFAULT_MAX_FACTS,
    select_context_for_outline,
    select_section_facts,
)
from app.core.semantic import rank_by_similarity


def _section(axis: str, title: str, question: str = "", goal: str = "") -> OutlineSection:
    return OutlineSection(axis=axis, title=title, question=question, coverage_goal=goal)


def test_section_gets_its_own_facts_not_the_global_pool():
    """Two topics in one pool: each section's selection is relevant to IT."""
    solar = [
        {
            "claim": "Solar photovoltaic module costs fell 89% between 2010 and 2023.",
            "source": "https://irena.org/report",
            "confidence": 0.8,
            "verified": True,
            "axis": "evidence",
        },
        {
            "claim": "Solar farms require large land areas per megawatt of capacity.",
            "source": "https://nrel.gov/a",
            "confidence": 0.7,
            "verified": True,
            "axis": "criticism",
        },
    ]
    nuclear = [
        {
            "claim": "Nuclear plants provide firm baseload power with a 90% capacity factor.",
            "source": "https://iaea.org/x",
            "confidence": 0.8,
            "verified": True,
            "axis": "evidence",
        },
        {
            "claim": "Nuclear construction costs and timelines have repeatedly overrun.",
            "source": "https://worldbank.org/y",
            "confidence": 0.7,
            "verified": True,
            "axis": "criticism",
        },
    ]
    pool = solar + nuclear

    solar_section = _section(
        "comparison", "Solar Power", "Solar photovoltaic costs and land use for solar farms"
    )
    # Force a small cap so selection must actually choose.
    solar_picked = select_section_facts(solar_section, pool, max_facts=2, impact_reserve=1)
    nuclear_section = _section(
        "comparison", "Nuclear Power", "Nuclear baseload capacity and reactor construction"
    )
    nuclear_picked = select_section_facts(nuclear_section, pool, max_facts=2, impact_reserve=1)

    solar_claims = " ".join(f["claim"] for f in solar_picked).lower()
    nuclear_claims = " ".join(f["claim"] for f in nuclear_picked).lower()

    assert "solar" in solar_claims
    assert "nuclear" not in solar_claims
    assert "nuclear" in nuclear_claims
    assert "solar" not in nuclear_claims


def test_high_impact_quantitative_claim_retained_at_mid_similarity():
    """A unique, corroborated quantitative claim must survive even when its
    similarity is mid-range and the pool is capped."""
    section = _section(
        "evidence", "Evidence & Data", "What data supports the trend?"
    )
    # Three wordy, highly similar on-topic claims with no numbers...
    filler = [
        {
            "claim": (
                "The trend is widely discussed and the discussion continues to "
                f"expand across many outlets and commentary pieces number {i}."
            ),
            "source": f"https://blog{i}.example.com/x",
            "confidence": 0.5,
            "verified": True,
            "axis": "evidence",
        }
        for i in range(3)
    ]
    # ...and one mid-relevance but high-impact, corroborated, primary figure.
    impact = {
        "claim": "Global capacity additions reached 510 gigawatts in 2023.",
        "source": "https://iea.org/report",
        "confidence": 0.75,
        "verified": True,
        "axis": "evidence",
        "corroboration_count": 3,
        "evidence": {
            "grade": "A",
            "corroboration_count": 3,
            "authority": 0.95,
            "is_primary": True,
            "domain": "iea.org",
            "needs_corroboration": False,
        },
    }
    pool = filler + [impact]
    picked = select_section_facts(section, pool, max_facts=2, impact_reserve=1)
    claims = [f["claim"] for f in picked]
    assert any("510" in c for c in claims), "high-impact quantitative claim was discarded"


def test_selection_diversifies_domains_when_alternatives_exist():
    """Selection must not return N facts from one domain when alternatives exist."""
    section = _section("evidence", "Evidence & Data", "What does the data show?")
    # Five near-identical restatements from one publisher, plus one distinct
    # claim from another publisher of comparable relevance.
    same_domain = [
        {
            "claim": f"Data on the trend is broadly consistent across measurements {i}.",
            "source": f"https://same.example.com/{i}",
            "confidence": 0.7,
            "verified": True,
            "axis": "evidence",
            "evidence": {"grade": "B", "domain": "same.example.com"},
        }
        for i in range(5)
    ]
    other = {
        "claim": "Independent measurement of the trend reached a record level in 2024.",
        "source": "https://other.example.com/a",
        "confidence": 0.7,
        "verified": True,
        "axis": "evidence",
        "evidence": {"grade": "B", "domain": "other.example.com"},
    }
    picked = select_section_facts(section, same_domain + [other], max_facts=3, impact_reserve=0)
    domains = {
        (f.get("evidence") or {}).get("domain") or f.get("source") for f in picked
    }
    assert len(domains) >= 2, f"selection returned only one domain: {domains}"


def test_selection_preserves_evidence_traceability_fields():
    """Selection returns full claim dicts, never stripped summaries."""
    section = _section("evidence", "Evidence & Data", "Numbers?")
    fact = {
        "claim": "Capacity reached 510 gigawatts in 2023.",
        "source": "https://iea.org/report",
        "verified": True,
        "axis": "evidence",
        "corroboration_count": 3,
        "corroborating_sources": ["https://iea.org/report", "https://irena.org/x"],
        "numbers": [{"value": 510, "unit": "gw"}],
        "evidence": {
            "grade": "A",
            "corroboration_count": 3,
            "authority": 0.95,
            "is_primary": True,
            "domain": "iea.org",
        },
        "verification": {"verified": True, "score": 0.9},
    }
    other = {
        "claim": "A tangential claim about something else entirely here.",
        "source": "https://blog.example.com/x",
        "verified": True,
        "axis": "evidence",
    }
    picked = select_section_facts(section, [fact, other], max_facts=1, impact_reserve=1)
    assert len(picked) == 1
    kept = picked[0]
    assert kept["evidence"]["grade"] == "A"
    assert kept["corroborating_sources"] == [
        "https://iea.org/report",
        "https://irena.org/x",
    ]
    assert kept["numbers"] == [{"value": 510, "unit": "gw"}]
    assert kept["source"] == "https://iea.org/report"
    assert kept["verification"]["verified"] is True


def test_selector_never_mutates_input_facts():
    section = _section("evidence", "Evidence & Data", "Numbers?")
    facts = [
        {
            "claim": f"Claim number {i} with a quantity {i * 10}% attached.",
            "source": f"https://s{i}.example.com/x",
            "verified": True,
            "axis": "evidence",
        }
        for i in range(5)
    ]
    import copy

    snapshot = copy.deepcopy(facts)
    select_section_facts(section, facts, max_facts=2, impact_reserve=1)
    assert facts == snapshot


def test_deterministic_fallback_returns_facts_on_empty_and_small_pools():
    """Empty candidates -> empty; a fitting pool passes through unchanged
    (the axis-grouped order), so a section is never starved."""
    section = _section("evidence", "Evidence & Data", "Numbers?")
    assert select_section_facts(section, []) == []

    small = [
        {"claim": "One claim about the topic.", "source": "https://a.example/x"},
        {"claim": "Another distinct claim about the topic.", "source": "https://b.example/y"},
    ]
    picked = select_section_facts(section, small, max_facts=DEFAULT_MAX_FACTS)
    assert [f["claim"] for f in picked] == [f["claim"] for f in small]


def test_select_context_for_outline_pairs_every_section():
    facts = [
        {
            "claim": "Artificial intelligence simulates human intelligence in machines.",
            "axis": "definition",
            "source": "https://a.example/x",
            "verified": True,
            "sub_question": "AI definition",
        },
        {
            "claim": "Global AI spending reached 200 billion dollars in 2025.",
            "axis": "evidence",
            "source": "https://b.example/y",
            "verified": True,
            "sub_question": "AI market data",
        },
    ]
    outline = build_outline("What is the current trend of AI?", facts)
    pairs = select_context_for_outline(outline)
    assert len(pairs) == len(outline.sections)
    for section, selected in pairs:
        assert isinstance(section, OutlineSection)
        assert isinstance(selected, list)


def test_global_pool_candidates_recover_off_axis_relevant_fact():
    """A fact whose AXIS is wrong for a section but whose CONTENT is relevant
    must be selectable (the off-topic-evidence fix)."""
    from app.agents.synthesizer import _ranked_section_groups

    # The solar fact is mislabeled under "outlook"; the section asking about
    # solar comparison should still surface it over unrelated history facts.
    # Extra off-topic facts force the cap to bite so ranking actually runs.
    facts = [
        {
            "claim": "Solar photovoltaic costs fell 89% while nuclear costs rose.",
            "axis": "outlook",
            "source": "https://irena.org/x",
            "verified": True,
        },
        {
            "claim": "The discovery of fission dates to 1938 laboratory work.",
            "axis": "comparison",
            "source": "https://history.example/x",
            "verified": True,
        },
        {
            "claim": "Wind turbine blade recycling remains an unresolved engineering problem.",
            "axis": "comparison",
            "source": "https://wind.example/x",
            "verified": True,
        },
        {
            "claim": "Coal plants emit sulfur dioxide and particulate matter.",
            "axis": "comparison",
            "source": "https://coal.example/x",
            "verified": True,
        },
    ]
    outline = build_outline(
        "Compare solar vs nuclear for grid baseload",
        facts,
        [
            {"question": "Solar versus nuclear cost comparison", "axis": "comparison"},
        ],
    )
    groups = _ranked_section_groups(outline, facts)
    comparison = next(g for s, g in groups if s.axis == "comparison")
    selected_claims = [f["claim"].lower() for f in comparison]
    # The mislabeled solar claim is recovered into the comparison section even
    # though its axis points elsewhere, and outranks the off-topic wind/coal
    # claims it shares the pool with.
    solar_pos = next((i for i, c in enumerate(selected_claims) if "solar" in c), None)
    assert solar_pos is not None, selected_claims
    wind_pos = next(i for i, c in enumerate(selected_claims) if "wind turbine" in c)
    coal_pos = next(i for i, c in enumerate(selected_claims) if "coal plants" in c)
    assert solar_pos < wind_pos
    assert solar_pos < coal_pos

    # Sanity: the underlying engine agrees the section label is closer to the
    # solar claim than to the fission claim.
    scores = rank_by_similarity(
        "How It Compares Solar versus nuclear cost comparison",
        [f["claim"] for f in facts],
    )
    assert scores[0] > scores[1]
