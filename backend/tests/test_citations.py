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
