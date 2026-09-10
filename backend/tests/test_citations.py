"""Citation-dense synthesis: markers resolve to real numbered sources."""

import asyncio

from app.agents.synthesizer import (
    _append_source_legend,
    _numbered_sources,
    _validate_citations,
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
    assert "Sources:" in answer
    assert "[1] en.wikipedia.org" in answer
    assert "[2] arxiv.org" in answer


def test_prompt_numbers_sources_and_bans_invention():
    facts = [_fact(0, "en.wikipedia.org")]
    llm = FakeLLM("Short answer [1].")
    asyncio.run(synthesizer_agent(llm, "What is RAG?", facts))
    assert "[1] en.wikipedia.org" in llm.seen["user"]
    assert "ONLY the source numbers" in llm.seen["system"]
    assert "Do NOT include source links" not in llm.seen["system"]


def test_fallback_appends_legend():
    class ExplodingLLM:
        async def generate_json(self, *a, **k):
            raise RuntimeError("down")

    facts = [_fact(0, "en.wikipedia.org"), _fact(1, "arxiv.org")]
    answer = asyncio.run(synthesizer_agent(ExplodingLLM(), "What is RAG?", facts))
    assert "Sources:" in answer
    assert "en.wikipedia.org" in answer


def test_legend_capped_at_top_ten():
    facts = [_fact(i, f"host-{i}.org") for i in range(12)]
    numbered = _numbered_sources(facts[:10])
    legend = _append_source_legend("Answer [1].", numbered)
    assert legend.count("\n[") == 10


def test_validate_citations_edge_cases():
    assert _validate_citations("a [1] b [0] c [3]", 2) == "a [1] b  c "
    assert _validate_citations("", 2) == ""


def test_sanitizer_preserves_paragraphs():
    from app.agents.synthesizer import _sanitize_answer_text

    raw = "First paragraph here.\n\n\nSecond paragraph  with   spaces.\n### Costs heading\nCost line [1]."
    out = _sanitize_answer_text(raw, "What is X?")
    assert out == ("First paragraph here.\n\nSecond paragraph with spaces. "
                   "Costs heading Cost line [1].")


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
    assert "Sources:" in answer


def test_overlap_drops_off_topic_junk():
    from app.agents.evidence_utils import MIN_QUERY_OVERLAP, claim_query_overlap

    query = "What is transfer learning?"
    assert claim_query_overlap(query, "Crypto has real uses beyond investing today") < MIN_QUERY_OVERLAP
    assert claim_query_overlap(query, "Transfer learning reuses models trained before") >= MIN_QUERY_OVERLAP


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
