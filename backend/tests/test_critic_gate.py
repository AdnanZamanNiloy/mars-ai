"""Synthesis gate: the critic cannot call thin evidence sufficient."""

import asyncio

from app.agents.critic import critic_agent
from app.agents.planner import diversity_coverage


class FakeLLM:
    def __init__(self, verdict=True):
        self.verdict = verdict

    async def generate_json(self, system_prompt, user_prompt, response_model=None):
        return {
            "is_sufficient": self.verdict,
            "reason": "looks complete",
            "improved_queries": [],
            "confidence": 0.85,
        }


def _fact(i: int, host: str, verified: bool, confidence: float) -> dict:
    topics = [
        "Retrieval augmented generation is a technique combining search with language models",
        "The retriever component is a dense index queried with vector similarity",
        "Evaluation is a benchmark suite measuring answer faithfulness scores",
        "Chunking strategy is a preprocessing choice affecting recall rates",
        "Reranking is a second-stage filter improving precision metrics",
        "Caching is a latency optimization storing frequent query results",
    ]
    return {
        "claim": topics[i % len(topics)],
        "source": f"https://{host}/article-{i}",
        "confidence": confidence,
        "verified": verified,
    }


def _run(facts, verdict=True):
    llm = FakeLLM(verdict)
    return asyncio.run(critic_agent(llm, "What is RAG?", facts, iteration=1, max_iterations=3))


def test_gate_blocks_unverified_evidence():
    facts = [_fact(0, "en.wikipedia.org", False, 0.9), _fact(1, "en.wikipedia.org", False, 0.9),
             _fact(2, "arxiv.org", False, 0.9), _fact(3, "arxiv.org", False, 0.9)]
    result = _run(facts)
    assert result["is_sufficient"] is False
    assert "verified" in result["reason"]


def test_gate_blocks_single_source():
    facts = [_fact(i, "en.wikipedia.org", True, 0.9) for i in range(4)]
    result = _run(facts)
    assert result["is_sufficient"] is False


def test_gate_passes_verified_multi_source():
    facts = [_fact(0, "en.wikipedia.org", True, 0.9), _fact(1, "en.wikipedia.org", True, 0.9),
             _fact(2, "arxiv.org", True, 0.9), _fact(3, "arxiv.org", True, 0.9)]
    result = _run(facts)
    assert result["is_sufficient"] is True
    assert result["confidence"] >= 0.78


def test_strong_stats_override_noisy_insufficient():
    """Pre-existing philosophy, kept: stellar verified multi-domain stats
    override a noisy insufficient verdict — but the gate still applies."""
    facts = [_fact(0, "en.wikipedia.org", True, 0.9), _fact(1, "en.wikipedia.org", True, 0.9),
             _fact(2, "arxiv.org", True, 0.9), _fact(3, "arxiv.org", True, 0.9)]
    result = _run(facts, verdict=False)
    assert result["is_sufficient"] is True


def test_diversity_coverage_counts_distinct_types():
    plan = [
        {"search_type": "encyclopedia"},
        {"search_type": "statistical"},
        {"search_type": "statistical"},
        {"search_type": "bogus"},
        {},
    ]
    assert diversity_coverage(plan) == {"encyclopedia", "statistical"}
