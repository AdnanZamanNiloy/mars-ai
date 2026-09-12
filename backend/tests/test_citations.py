"""Citation-dense synthesis: markers resolve to real numbered sources."""

import asyncio

from app.agents.synthesizer import (
    _append_source_legend,
    _assign_numbers,
    _drop_invalid_markers,
    synthesizer_agent,
)


TOPICS = [
    "transformer attention architectures benchmarks",
    "vector database indexing strategies comparison",
    "quantization techniques memory footprint analysis",
]


def _fact(i: int, host: str) -> dict:
    return {
        "claim": f"Study finding: {TOPICS[i % len(TOPICS)]}",
        "source": f"https://{host}/paper-{i}",
        "confidence": 0.85,
    }


class FakeLLM:
    def __init__(self, answer):
        self.answer = answer
        self.seen = {}

    async def generate_json(self, system_prompt, user_prompt, response_model=None):
        self.seen["system"] = system_prompt
        self.seen["user"] = user_prompt
        return {"answer": self.answer}


def test_valid_markers_kept_invalid_stripped_legend_appended():
    facts = [_fact(0, "en.wikipedia.org"), _fact(1, "arxiv.org")]
    llm = FakeLLM("RAG is effective [1] and widely used [2] everywhere [99].")
    answer = asyncio.run(synthesizer_agent(llm, "What is RAG?", facts))
    assert "[1]" in answer and "[2]" in answer
    assert "[99]" not in answer
    assert "## Sources" in answer
    assert "[1] en.wikipedia.org" in answer
    assert "[2] arxiv.org" in answer


def test_prompt_numbers_sources_and_bans_invention():
    facts = [_fact(0, "en.wikipedia.org")]
    llm = FakeLLM("Short answer [1].")
    asyncio.run(synthesizer_agent(llm, "What is RAG?", facts))
    assert "[1] en.wikipedia.org" in llm.seen["user"]
    # v3 CITATION RULES block: no renumbering, no guessing, no uncited
    # numbers — stronger than the old single sentence, different words.
    assert "do not cite a number you were not given" in llm.seen["system"]
    assert "does not\n    appear verbatim in the evidence" in llm.seen["system"]


def test_fallback_appends_legend():
    class ExplodingLLM:
        async def generate_json(self, *a, **k):
            raise RuntimeError("down")

    facts = [_fact(0, "en.wikipedia.org"), _fact(1, "arxiv.org")]
    answer = asyncio.run(synthesizer_agent(ExplodingLLM(), "What is RAG?", facts))
    assert "## Sources" in answer
    assert "en.wikipedia.org" in answer


def test_legend_capped_at_top_ten():
    facts = [_fact(i, f"host-{i}.org") for i in range(12)]
    numbered, _ = _assign_numbers(facts[:10])
    legend = _append_source_legend("Answer [1].", numbered)
    assert legend.count("\n[") == 10


def test_validate_citations_edge_cases():
    assert _drop_invalid_markers("a [1] b [0] c [3]", 2) == "a [1] b  c "
    assert _drop_invalid_markers("", 2) == ""


def test_sanitizer_preserves_paragraphs():
    from app.agents.synthesizer import _sanitize_answer_text

    raw = "First paragraph here.\n\n\nSecond paragraph  with   spaces.\n### Costs heading\nCost line [1]."
    out = _sanitize_answer_text(raw, "What is X?")
    assert out == ("First paragraph here.\n\nSecond paragraph with spaces.\n\n"
                   "### Costs heading\n\nCost line [1].")


def test_sanitizer_repairs_query_opener():
    from app.agents.synthesizer import _sanitize_answer_text

    out = _sanitize_answer_text("what is RAG refers to retrieval.", "What is RAG?")
    assert out.startswith("RAG refers to retrieval.")


def test_clean_snippet_strips_date_stamps():
    from app.agents.evidence_utils import clean_snippet_text

    out = clean_snippet_text(
        "Mar 17, 2026 · Transfer learning is a machine learning technique where a model "
        "trained on one task is reused as the starting point for a different task."
    )
    assert out.startswith("Transfer learning is a machine learning technique")
    assert "Mar 17" not in out


def test_clean_snippet_trims_mid_sentence_cuts():
    from app.agents.evidence_utils import clean_snippet_text

    out = clean_snippet_text(
        "Jul 2, 2025 · This article delves into the mechanics of transfer learning, "
        "exploring its theoretical foundations, practical applications, and effica"
    )
    assert "Jul 2" not in out
    assert "effica" not in out
    assert out.endswith(".")


def test_clean_snippet_drops_relative_dates_and_stumps():
    from app.agents.evidence_utils import clean_snippet_text

    assert clean_snippet_text("2 days ago · Short frag") == ""
    assert clean_snippet_text("May 9, 2026 · Transfer learning is a technique where a model is reused.") != ""


def test_fallback_answer_has_no_boilerplate():
    from app.agents.synthesizer import synthesizer_agent

    class ExplodingLLM:
        async def generate_json(self, *a, **k):
            raise RuntimeError("down")

    facts = [
        {"claim": "Transfer learning reuses a model trained on one task for a related task with limited data.",
         "source": "https://en.wikipedia.org/wiki/Transfer_learning", "confidence": 0.9},
        {"claim": "Fine-tuning pretrained networks reduces training time substantially.",
         "source": "https://arxiv.org/abs/1234", "confidence": 0.85},
    ]
    answer = asyncio.run(synthesizer_agent(ExplodingLLM(), "What is transfer learning?", facts))
    assert "supported by reliable evidence" not in answer
    assert "Transfer learning reuses" in answer
    assert "## Sources" in answer


def test_overlap_drops_off_topic_junk():
    from app.agents.evidence_utils import MIN_QUERY_OVERLAP, claim_query_overlap

    query = "What is transfer learning?"
    assert claim_query_overlap(query, "Crypto has real uses beyond investing today") < MIN_QUERY_OVERLAP
    assert claim_query_overlap(query, "Transfer learning reuses models trained before") >= MIN_QUERY_OVERLAP
    assert claim_query_overlap("What is transformer?",
                               "Electrical transformers step voltage up or down") >= MIN_QUERY_OVERLAP
    # Short words still need exact matches: "in" must not match "instrument".
    assert claim_query_overlap("Routes in Oslo", "Instrument transformers hum") == 0.0


def test_select_diverse_skips_near_dupes():
    from app.agents.evidence_utils import select_diverse

    dupes = [
        {"claim": "Transfer learning reuses a model trained on one task for another task", "confidence": 0.95},
        {"claim": "Transfer learning reuses a model trained on one task for a new task", "confidence": 0.9},
        {"claim": "Quantum error correction uses surface codes on superconducting qubits", "confidence": 0.85},
    ]
    picked = select_diverse(dupes, k=3)
    assert len(picked) == 2
    assert any("Quantum" in p["claim"] for p in picked)


def test_model_path_fragment_rejected(tmp_path):
    from app.agents.summarizer import summarizer_agent
    from app.core.config import Settings

    class FakeLLM:
        settings = Settings(groq_api_key="k", database_url=str(tmp_path / "t.db"), _env_file=None)

        async def generate_json(self, *a, **k):
            return {"facts": [
                {"claim": "Transfer learning reuses models learned from a large dat",
                 "source": "https://en.wikipedia.org/wiki/X", "confidence": 0.9},
            ]}

    facts = asyncio.run(summarizer_agent(
        FakeLLM(), "What is transfer learning?",
        search_results=[{"url": "https://en.wikipedia.org/wiki/X", "snippet": "s", "content": "c"}]))
    assert all(" dat" not in f["claim"] for f in facts)


def test_heuristic_drops_off_topic_snippet(tmp_path):
    from app.agents.summarizer import summarizer_agent
    from app.core.config import Settings

    class ExplodingLLM:
        settings = Settings(groq_api_key="k", database_url=str(tmp_path / "t.db"), _env_file=None)

        async def generate_json(self, *a, **k):
            raise RuntimeError("down")

    facts = asyncio.run(summarizer_agent(
        ExplodingLLM(), "What is transfer learning?",
        search_results=[{"url": "https://crypto.example.com/x",
                         "snippet": "Crypto has real world uses beyond investing today for payments",
                         "content": "crypto payments everywhere today"}]))
    assert facts == []


def test_fallback_assembly_prefers_diverse_claims():
    from app.agents.synthesizer import synthesizer_agent

    class ExplodingLLM:
        async def generate_json(self, *a, **k):
            raise RuntimeError("down")

    dupes = [
        {"claim": "Transfer learning reuses a model trained on one task for another task",
         "source": "https://a.com/1", "confidence": 0.95},
        {"claim": "Transfer learning reuses a model trained on one task for a new task",
         "source": "https://b.com/2", "confidence": 0.9},
        {"claim": "Fine-tuning pretrained networks cuts training time substantially here",
         "source": "https://c.com/3", "confidence": 0.85},
    ]
    answer = asyncio.run(synthesizer_agent(ExplodingLLM(), "What is transfer learning?", dupes))
    assert "Fine-tuning pretrained networks" in answer
    assert answer.count("reuses a model trained on one task") == 1


def test_link_text_artifacts_stripped():
    from app.agents.evidence_utils import clean_snippet_text

    out = clean_snippet_text(
        "Transformers are employed for widely varying purposes. Learn mor The question "
        "arises about function composition in electrical networks today."
    )
    assert "Learn mor" not in out
    out2 = clean_snippet_text(
        "A full explanation of attention mechanisms is available here. Read more about "
        "transformer variants and their growing adoption in production systems now."
    )
    assert "Read more" not in out2


def test_fallback_mmr_collapses_paraphrase_dupes():
    from app.agents.synthesizer import synthesizer_agent

    class ExplodingLLM:
        async def generate_json(self, *a, **k):
            raise RuntimeError("down")

    dupes = [
        {"claim": "Transformer, device that transfers electric energy from one alternating-current circuit to other circuits",
         "source": "https://a.com/1", "confidence": 0.95},
        {"claim": "In electrical engineering, a transformer is a passive component that transfers electrical energy between circuits",
         "source": "https://b.com/2", "confidence": 0.9},
        {"claim": "The Transformer architecture uses self-attention for sequence modeling tasks",
         "source": "https://c.com/3", "confidence": 0.85},
    ]
    answer = asyncio.run(synthesizer_agent(ExplodingLLM(), "What is transformer?", dupes))
    assert "self-attention" in answer
    assert "passive component" not in answer


def test_fallback_mines_content_across_many_sources(tmp_path):
    """Degraded answers must not be thin: the heuristic fallback works from
    full fetched content (not snippets) over up to 12 sources and tags each
    fact with its sub-question for grouped synthesis."""
    from app.agents.summarizer import summarizer_agent
    from app.core.config import Settings

    class ExplodingLLM:
        settings = Settings(groq_api_key="k", database_url=str(tmp_path / "t.db"), _env_file=None)

        async def generate_json(self, *a, **k):
            raise RuntimeError("down")

    bodies = [
        "The transformer architecture relies on self-attention for sequence tasks.",
        "Electrical transformers step voltage up or down between alternating-current circuits.",
        "Transformer models train in parallel unlike recurrent networks of the past.",
        "Distribution transformers hum because magnetostriction vibrates the iron core.",
        "Positional encodings give transformer layers a sense of token order.",
        "Instrument transformers scale high voltages down for safe measurement.",
        "Large transformer checkpoints need gigabytes of accelerator memory.",
        "Autotransformers share one winding between primary and secondary sides.",
        "Cross-attention lets a transformer decoder read the encoder output.",
        "Three-phase transformer banks power entire industrial districts.",
    ]
    results = [
        {"url": f"https://en.wikipedia.org/wiki/Transformer_{i}", "snippet": "",
         "content": body, "sub_question": f"angle {i % 3}"}
        for i, body in enumerate(bodies)
    ]
    facts = asyncio.run(summarizer_agent(ExplodingLLM(), "What is transformer?", results))
    assert len(facts) >= 8, f"fallback must cover many sources, got {len(facts)}"
    assert all(f.get("sub_question", "").startswith("angle") for f in facts)
    assert any("magnetostriction" in f["claim"] for f in facts)


def test_fallback_answer_groups_by_sub_question():
    """Grouped fallback synthesis keeps one paragraph per angle instead of
    a single flat blob."""
    from app.agents.synthesizer import synthesizer_agent

    class ExplodingLLM:
        async def generate_json(self, *a, **k):
            raise RuntimeError("down")

    facts = [
        {"claim": "Electrical transformers step voltage between alternating-current circuits",
         "source": "https://a.com/1", "confidence": 0.9, "sub_question": "electrical sense"},
        {"claim": "Autotransformers share a single winding for step regulation",
         "source": "https://b.com/2", "confidence": 0.85, "sub_question": "electrical sense"},
        {"claim": "Transformer neural models use self-attention over token sequences",
         "source": "https://c.com/3", "confidence": 0.9, "sub_question": "machine learning sense"},
    ]
    answer = asyncio.run(synthesizer_agent(ExplodingLLM(), "What is transformer?", facts))
    assert "step voltage" in answer and "self-attention" in answer
    assert "\n\n" in answer.split("Sources:")[0], "angles must be separate paragraphs"


def test_clean_snippet_drops_excerpt_seams():
    """Page-furniture fragments mined from full content must die in the
    cleaner, not surface in degraded answers (live ragged claims)."""
    from app.agents.evidence_utils import clean_snippet_text

    assert clean_snippet_text(
        "Main article: Attention (machine learning) History section here today") == ""
    assert clean_snippet_text(
        "The previous model.( [...] A 380M-parameter model followed suit here") == ""
    assert clean_snippet_text(
        'The mechanism uses dot-product attention "Attention (ML)") units today') == ""
    out = clean_snippet_text(
        "Basic Outline of a Transformer ### Key Components of the model #### Self-Attention wins")
    assert "###" not in out and "Self-Attention wins" in out
    keep = "Transformer models train in parallel unlike recurrent networks of the past."
    assert clean_snippet_text(keep) == keep


def test_clean_snippet_drops_bare_title_prefix():
    """A page heading mined as a claim ('Title: subtitle', no sentence) is
    not a finding — but headed real sentences survive."""
    from app.agents.evidence_utils import clean_snippet_text

    assert clean_snippet_text(
        "Exploring Transformer Models: Key Uses, Examples, and Innovations") == ""
    assert clean_snippet_text(
        "Exploring Transformer Models: Key Uses, Examples, and Innovations.") == ""
    headed = "Transformers are used for many purposes: power delivery and signal isolation today."
    assert clean_snippet_text(headed) == headed
    out = clean_snippet_text(
        "Basic Outline of a Transformer ### Key Components of the model #### Self-Attention wins")
    assert "###" not in out and "Self-Attention wins" in out


def test_clean_snippet_drops_question_shaped_claims():
    """A claim ending in '?' is a glued heading+question, never a finding."""
    from app.agents.evidence_utils import clean_snippet_text

    assert clean_snippet_text(
        "Transmission of power How does a transformer work?") == ""
    assert clean_snippet_text(
        "Transformer models train in parallel unlike recurrent networks.") != ""


def test_clean_snippet_strips_blockquote_markers():
    """Flattened quote markers are page furniture; the quoted claim stays."""
    from app.agents.evidence_utils import clean_snippet_text

    out = clean_snippet_text(
        "> > > However RNNs may struggle to capture long-range dependencies effectively.")
    assert not out.startswith(">")
    assert "However RNNs may struggle" in out
    keep = "Attention scores show that a exceeds b in most transformer heads today."
    assert clean_snippet_text(keep) == keep


def test_clean_snippet_drops_dangling_endings_and_entities():
    """Mid-sentence cuts ending on function words, and HTML entities, must
    not reach answers (both seen live in fallback findings)."""
    from app.agents.evidence_utils import clean_snippet_text

    assert clean_snippet_text(
        "Demonstrate the ability of transformers to perform a wide variety of "
        "NLP-related subtasks including") == ""
    assert clean_snippet_text(
        "Demonstrate the ability of transformers to perform a wide variety of "
        "NLP-related subtasks and their related applications, including.") == ""
    out = clean_snippet_text(
        "Power Transformer Core Losses and Core Design Concepts&quot "
        "are explained with clear diagrams in this transformer guide.")
    assert "&quot" not in out and "Core Design Concepts" in out
    keep = "Transformer models train in parallel unlike recurrent networks."
    assert clean_snippet_text(keep) == keep


def test_clean_snippet_drops_unbalanced_paren_cuts():
    """A mid-excerpt cut with a tail longer than 3 letters ('...number of
    nega') slips past the stub check — but its unclosed paren gives it away."""
    from app.agents.evidence_utils import clean_snippet_text

    assert clean_snippet_text(
        "Transformers cannot recognize parity (e.g., whether a phrase contains an even "
        "number of nega") == ""
    keep = "Transformers use attention (see Vaswani et al.) for sequence modeling tasks."
    assert clean_snippet_text(keep) == keep


def test_clean_snippet_drops_latex_soup_and_boilerplate():
    """Unreadable math markup and instructional leads are not claims."""
    from app.agents.evidence_utils import clean_snippet_text

    assert clean_snippet_text(
        r"One set of {\displaystyle \left(W^{Q},W^{K}\right)} matrices is called a head "
        r"and each transformer layer has several heads today") == ""
    assert clean_snippet_text(
        "The attention output scales with \sqrt{N} heads in large transformer models today") == ""
    assert clean_snippet_text(
        "Use this page to revise transformer concepts within transmission of electricity") == ""
    assert clean_snippet_text(
        "This article will explore transformer models and their many practical applications today") == ""
    keep = "Attention layers mix information across chunks in transformer pipelines."
    assert clean_snippet_text(keep) == keep


def test_clean_snippet_strips_read_time_and_heading_mark():
    """Date + read-time furniture and leading heading markers must not open
    a claim (live ragged lead: 'Dec 20, 2024 13 minutes read # What ...')."""
    from app.agents.evidence_utils import clean_snippet_text

    out = clean_snippet_text(
        "Dec 20, 2024 13 minutes read # What are transformer models? "
        "Transformer models are deep learning networks for sequences.")
    assert not out.startswith("Dec 20")
    assert "#" not in out.split("?")[0]
    assert "Transformer models are deep learning networks" in out


def test_split_into_sentences_basic():
    from app.agents.evidence_utils import split_into_sentences

    parts = split_into_sentences(
        "Transformers use attention. They train in parallel! Do they scale? Yes, broadly.")
    assert len(parts) == 4
    assert parts[0] == "Transformers use attention."
    assert split_into_sentences("") == []
def test_fallback_extracts_multiple_claims_per_source(tmp_path):
    """One content-rich page yields several facts, not one (sentence-scale
    extraction is what makes degraded answers substantive)."""
    from app.agents.summarizer import summarizer_agent
    from app.core.config import Settings

    class ExplodingLLM:
        settings = Settings(groq_api_key="k", database_url=str(tmp_path / "t.db"), _env_file=None)

        async def generate_json(self, *a, **k):
            raise RuntimeError("down")

    content = (
        "Transformer neural models use self-attention over token sequences. "
        "Electrical transformers step voltage between alternating-current circuits. "
        "Transformer checkpoints need gigabytes of accelerator memory.")
    facts = asyncio.run(summarizer_agent(
        ExplodingLLM(), "What is transformer?",
        search_results=[{"url": "https://en.wikipedia.org/wiki/Transformer_X",
                         "snippet": "", "content": content,
                         "sub_question": "mixed senses"}]))
    assert len(facts) >= 2, f"expected multi-claim mining, got {facts}"
    assert all(f.get("sub_question") == "mixed senses" for f in facts)


def test_fallback_report_has_sections_and_figures():
    """Degraded output must read like a report: angle sections, a Key
    figures section for number-claims beyond the section budget, and a
    legend covering the used sources."""
    from app.agents.synthesizer import synthesizer_agent

    class ExplodingLLM:
        async def generate_json(self, *a, **k):
            raise RuntimeError("down")

    facts = [
        {"claim": "Electrical transformers step voltage between alternating-current circuits",
         "source": "https://a.com/1", "confidence": 0.95, "sub_question": "markets"},
        {"claim": "Autotransformers share a single winding for compact step regulation",
         "source": "https://b.com/2", "confidence": 0.93, "sub_question": "markets"},
        {"claim": "Transformer neural models use self-attention over token sequences",
         "source": "https://c.com/3", "confidence": 0.92, "sub_question": "markets"},
        {"claim": "Instrument transformers scale line voltage for metering equipment",
         "source": "https://d.com/4", "confidence": 0.91, "sub_question": "markets"},
        {"claim": "Three-phase banks power industrial districts reliably",
         "source": "https://e.com/5", "confidence": 0.90, "sub_question": "markets"},
        {"claim": "Cross-attention links decoders to encoder output representations",
         "source": "https://f.com/6", "confidence": 0.89, "sub_question": "markets"},
        {"claim": "The power transformer market was valued at USD 23 billion in 2025",
         "source": "https://g.com/7", "confidence": 0.70, "sub_question": "markets"},
        {"claim": "Shipments grew 12 percent year over year across Asia",
         "source": "https://h.com/8", "confidence": 0.65, "sub_question": "markets"},
    ]
    answer = asyncio.run(synthesizer_agent(ExplodingLLM(), "What is transformer?", facts))
    assert not answer.startswith("# Final Answer"), "workflow adds the title; no double lead"
    assert "## Markets" in answer
    assert "## Key figures" in answer
    figures = answer.split("## Key figures")[1].split("## Sources")[0]
    assert "USD 23 billion" in figures and "12 percent" in figures
    legend = answer.split("## Sources")[1]
    for host in ("a.com", "g.com", "h.com"):
        assert host in legend


def test_llm_prompt_carries_angles_and_evidence():
    """The LLM brief must name the angles and pass the widened fact pool."""
    from app.agents.synthesizer import synthesizer_agent

    seen = {}

    class FakeLLM:
        async def generate_json(self, system_prompt, user_prompt, response_model=None):
            seen["system"] = system_prompt
            seen["user"] = user_prompt
            return {"answer": "Report text here with [1] marker."}

    facts = [
        {"claim": "Electrical transformers step voltage between alternating-current circuits",
         "source": "https://a.com/1", "confidence": 0.9, "sub_question": "angle one"},
        {"claim": "Autotransformers share a single winding for compact regulation",
         "source": "https://b.com/2", "confidence": 0.88, "sub_question": "angle one"},
        {"claim": "Instrument transformers scale line voltage for metering",
         "source": "https://c.com/3", "confidence": 0.87, "sub_question": "angle one"},
        {"claim": "Transformer neural models use self-attention over token sequences",
         "source": "https://d.com/4", "confidence": 0.9, "sub_question": "angle two"},
        {"claim": "Cross-attention links decoders to encoder output representations",
         "source": "https://e.com/5", "confidence": 0.88, "sub_question": "angle two"},
        {"claim": "Parallel training replaced slow recurrent loops entirely",
         "source": "https://f.com/6", "confidence": 0.87, "sub_question": "angle two"},
    ]
    answer = asyncio.run(synthesizer_agent(FakeLLM(), "What is transformer?", facts))
    assert "Angles to cover" in seen["user"]
    assert "angle one" in seen["user"] and "angle two" in seen["user"]
    assert "separate them explicitly" in seen["system"], "entity/sense separation rule must be present"
    system_flat = " ".join(seen["system"].lower().split())
    assert "according to the research" in system_flat, "banned-phrase rule must be present"
    assert "Key Findings" in seen["system"]
    assert "Report text here" in answer


def test_stratified_top_facts_keeps_weak_angles():
    """Pure confidence ranking buries whole low-scoring angles; round-robin
    keeps every sub-question represented (live: market angle cut entirely)."""
    from app.agents.synthesizer import _stratified_top_facts

    facts = [
        {"claim": f"Strong angle claim {i} with distinct wording here",
         "source": f"https://s{i}.com/x", "confidence": 0.9, "sub_question": "strong"}
        for i in range(8)
    ] + [
        {"claim": "Weak angle market size was USD 5 billion in 2024",
         "source": "https://w.com/x", "confidence": 0.3, "sub_question": "weak"},
    ]
    top = _stratified_top_facts(facts, per_angle=6, cap=30)
    assert any(f["sub_question"] == "weak" for f in top)
    assert len([f for f in top if f["sub_question"] == "strong"]) == 6


def test_fallback_full_decision_shape():
    """Degraded output honors the decision-grade contract: summary with a
    confidence line, findings, angle analysis, gaps naming degraded stages
    and conflicts — and a legend covering only used sources."""
    from app.agents.synthesizer import synthesizer_agent

    class ExplodingLLM:
        async def generate_json(self, *a, **k):
            raise RuntimeError("down")

    facts = [
        {"claim": "Electrical transformers step voltage between alternating-current circuits",
         "source": "https://a.com/1", "confidence": 0.9, "sub_question": "angle one",
         "verified": True},
        {"claim": "Autotransformers share a single winding for compact regulation",
         "source": "https://b.com/2", "confidence": 0.85, "sub_question": "angle one",
         "verified": True},
        {"claim": "Transformer neural models use self-attention over token sequences",
         "source": "https://c.com/3", "confidence": 0.9, "sub_question": "angle two",
         "verified": True},
        {"claim": "Cross-attention links decoders to encoder output representations",
         "source": "https://d.com/4", "confidence": 0.85, "sub_question": "angle two",
         "verified": True},
    ]
    context = {
        "contradictions": [{"claim_a": "Market is USD 23 billion", "source_a": "https://a.com/1",
                            "claim_b": "Market is USD 78 billion", "source_b": "https://c.com/3"}],
        "confidence": 0.6,
        "degraded": ["planner"],
        "total_facts": 6,
        "verified_count": 4,
    }
    answer = asyncio.run(synthesizer_agent(ExplodingLLM(), "What is transformer?", facts, context))
    assert "## Executive Summary" in answer
    assert "## Key Findings" in answer
    assert "## Angle one" in answer and "## Angle two" in answer
    assert "## Evidence & Confidence" in answer
    assert "## Limitations" in answer
    assert "Confidence: Medium (0.60)" in answer
    assert "planner" in answer.split("## Evidence & Confidence")[1]
    assert "1 source conflict" in answer
    assert "Uncertain: 2 collected claims" in answer
    assert "Evidence is thin" not in answer
    legend = answer.split("## Sources")[1]
    assert "a.com" in legend and "d.com" in legend


def test_fallback_key_findings_are_bullets_with_score_and_ambiguity():
    """Brief shape: findings as bullets, numeric score in the confidence
    line, and a senses-separated note when angles multiply."""
    from app.agents.synthesizer import synthesizer_agent

    class ExplodingLLM:
        async def generate_json(self, *a, **k):
            raise RuntimeError("down")

    facts = [
        {"claim": "Electrical transformers step voltage between alternating-current circuits",
         "source": "https://a.com/1", "confidence": 0.9, "sub_question": "angle one"},
        {"claim": "Transformer neural models use self-attention over token sequences",
         "source": "https://b.com/2", "confidence": 0.9, "sub_question": "angle two"},
        {"claim": "Market size for transformers grows steadily year over year",
         "source": "https://c.com/3", "confidence": 0.9, "sub_question": "angle three"},
    ]
    answer = asyncio.run(synthesizer_agent(
        ExplodingLLM(), "What is transformer?", facts, {"confidence": 0.8}))
    findings = answer.split("## Key Findings")[1].split("## ")[0]
    assert "- Transformer neural models" in findings
    assert "distinct angles" in answer
    assert "Confidence: High (0.80)" in answer


def test_thin_evidence_disclaimer():
    """Fewer than 3 verified facts: say so up front, rate Low."""
    from app.agents.synthesizer import synthesizer_agent

    class ExplodingLLM:
        async def generate_json(self, *a, **k):
            raise RuntimeError("down")

    facts = [{"claim": "Electrical transformers step voltage between alternating circuits",
              "source": "https://a.com/1", "confidence": 0.9, "verified": True}]
    answer = asyncio.run(synthesizer_agent(
        ExplodingLLM(), "What is transformer?", facts,
        {"total_facts": 1, "verified_count": 1}))
    assert "Evidence is thin" in answer
    assert "Confidence: Low" in answer


def test_render_context_block_carries_conflicts():
    """The LLM brief must surface conflicts and the honesty baseline."""
    from app.agents.synthesizer import _render_context_block

    block = _render_context_block({
        "contradictions": [{"claim_a": "A", "source_a": "https://a.com",
                            "claim_b": "B", "source_b": "https://b.com"}],
        "confidence": 0.8,
        "degraded": ["critic"],
        "total_facts": 10,
        "verified_count": 7,
    })
    assert "CONFLICTS WITH" in block
    assert "0.80" in block
    assert "critic" in block
    assert "7/10" in block
    assert _render_context_block({}) == ""
    assert _render_context_block(None) == ""


def test_sanitizer_preserves_markdown_headings():
    """The premium-report contract: ## / ### headings are the visual
    hierarchy and must survive sanitization as their own blocks — the old
    sanitizer stripped the markers and glued headings into paragraphs."""
    from app.agents.synthesizer import _sanitize_answer_text

    raw = "## Executive Summary\nRAG combines retrieval with generation.\n\n## Key Findings\n- Finding one [1].\n- Finding two [2]."
    out = _sanitize_answer_text(raw, "What is RAG?")
    assert "## Executive Summary" in out
    assert "## Key Findings" in out
    assert out.index("## Executive Summary") < out.index("RAG combines")
    blocks = out.split("\n\n")
    assert "## Key Findings" in blocks, "heading must be its own block, not glued to prose"


def test_legend_dedupes_repeated_sources():
    """One page cited by five claims is ONE source, not five legend rows."""
    from app.agents.synthesizer import _assign_numbers

    facts = [
        {"claim": "Claim one", "source": "https://a.org/page", "confidence": 0.9},
        {"claim": "Claim two", "source": "https://a.org/page", "confidence": 0.85},
        {"claim": "Claim three", "source": "https://b.org/other", "confidence": 0.8},
    ]
    numbered, _ = _assign_numbers(facts)
    assert [s["n"] for s in numbered] == [1, 2]
    assert numbered[0]["domain"] == "a.org"


def test_legend_caps_at_fourteen_strongest_sources():
    """v3 legend cap is 14 (was 12), grouped by canonical URL with
    tier/primary annotations."""
    from app.agents.synthesizer import _assign_numbers

    facts = [{"claim": f"Claim {i}", "source": f"https://host-{i}.org/x", "confidence": 0.9}
             for i in range(20)]
    numbered, _ = _assign_numbers(facts)
    assert len(numbered) == 14


def test_answer_support_parses_hash_sources_heading():
    """verify_answer_support must resolve the current `## Sources` legend
    form, not only the legacy bare `Sources:` line."""
    from app.agents.evidence_utils import verify_answer_support

    facts = [{"claim": "RAG combines retrieval with generation",
              "source": "https://en.wikipedia.org/wiki/RAG", "verified": True}]
    answer = (
        "RAG combines retrieval with generation [1].\n\n"
        "## Sources\n\n[1] en.wikipedia.org — https://en.wikipedia.org/wiki/RAG"
    )
    support = verify_answer_support(answer, facts)
    assert support["cited"] == 1
    assert support["supported"] == 1
    assert support["rate"] == 1.0


def test_sanitizer_keeps_bullets_as_list_items():
    """Consecutive '- ' lines must stay distinct list items (one bullet
    block), and inline ' - ' separators from model output must be split
    into real bullets — never flattened into one prose line."""
    from app.agents.synthesizer import _sanitize_answer_text

    raw = (
        "## Key Findings\n"
        "- Finding one [1].\n"
        "- Finding two [2]. - Finding three [3].\n\n"
        "Closing paragraph text."
    )
    out = _sanitize_answer_text(raw, "What is RAG?")
    bullet_block = [b for b in out.split("\n\n") if b.startswith("- ")]
    assert len(bullet_block) == 1, "consecutive bullets form one block"
    items = bullet_block[0].split("\n")
    assert items == ["- Finding one [1].", "- Finding two [2].", "- Finding three [3]."], items
    # Prose after the list stays its own paragraph.
    assert out.endswith("Closing paragraph text.")


def test_sanitizer_inline_bullet_split_requires_sentence_boundary():
    """' - ' mid-sentence (after commas, inside hyphenated phrases) is
    prose and must NOT be split."""
    from app.agents.synthesizer import _sanitize_answer_text

    raw = "- Voltage is raised for transmission - typically - to minimize loss [1]."
    out = _sanitize_answer_text(raw, "How do transformers work?")
    assert "\n" not in out and out.startswith("- Voltage"), out
    assert " - typically - " in out, "mid-sentence hyphens must remain prose"


def test_extractive_fallback_cites_every_claim():
    """The degraded-path writer must produce traceable output: every claim
    carries its marker inside the sentence, with a used-only legend."""
    from app.agents.synthesizer import synthesize

    class ExplodingLLM:
        async def generate_json(self, *a, **k):
            raise RuntimeError("down")

    facts = [
        {"claim": "RAG retrieves external documents before generating answers",
         "source": "https://arxiv.org/abs/2005.11401", "confidence": 0.9, "verified": True},
        {"claim": "RAG grounds model outputs in cited sources",
         "source": "https://en.wikipedia.org/wiki/RAG", "confidence": 0.8, "verified": True},
    ]
    result = asyncio.run(synthesize(ExplodingLLM(), "What is RAG?", facts))
    assert result.used_fallback is True
    assert "[1]" in result.answer and "[2]" in result.answer
    assert "## Sources" in result.answer
    assert result.audit.cited_sentences > 0


def test_audit_flags_untraceable_sentences():
    from app.agents.synthesizer import audit_citations

    facts = [{"claim": "RAG retrieves documents", "source": "https://a.org/x",
              "confidence": 0.9, "verified": True}]
    numbered = [{"n": 1, "domain": "a.org", "url": "https://a.org/x"}]
    audit = audit_citations(
        "RAG retrieves documents [1]. Models have exactly 4 layers and cost $9.",
        numbered, facts)
    assert len(audit.uncited_factual) >= 1
    assert len(audit.ungrounded_numbers) >= 1
    assert audit.is_clean is False
