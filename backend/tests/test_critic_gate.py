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


def test_comparative_query_without_definition_can_pass():
    """Regression: the gate required an ' is ' claim unconditionally, which
    forced every comparative/analytical query ('A vs B economics?') to loop
    to the iteration ceiling and be stamped incomplete no matter how good
    the evidence was. Definitional shape is only required for definitional
    queries (query_type=factual or a 'what is/define/explain' query)."""
    facts = [
        {"claim": "Solar LCOE in Bangladesh fell below grid parity in 2024", "source": "https://a.org/x",
         "confidence": 0.9, "verified": True},
        {"claim": "Nuclear capital costs exceed solar by a wide margin per MW", "source": "https://b.edu/y",
         "confidence": 0.9, "verified": True},
        {"claim": "Levelized cost comparisons favor solar for new capacity", "source": "https://c.gov/z",
         "confidence": 0.9, "verified": True},
        {"claim": "Financing terms drive the lifetime economics of both options", "source": "https://d.org/w",
         "confidence": 0.9, "verified": True},
    ]
    result = asyncio.run(critic_agent(FakeLLM(True), "Compare the economics of nuclear vs solar energy in Bangladesh",
                                      facts, iteration=1, max_iterations=3, query_type="comparative"))
    assert result["is_sufficient"] is True


def test_definitional_query_still_requires_definition():
    facts = [
        {"claim": "Solar LCOE fell below grid parity in 2024", "source": "https://a.org/x",
         "confidence": 0.9, "verified": True},
        {"claim": "Nuclear capital costs exceed solar per MW", "source": "https://b.edu/y",
         "confidence": 0.9, "verified": True},
        {"claim": "Levelized cost comparisons favor solar", "source": "https://c.gov/z",
         "confidence": 0.9, "verified": True},
        {"claim": "Financing terms drive lifetime economics", "source": "https://d.org/w",
         "confidence": 0.9, "verified": True},
    ]
    result = asyncio.run(critic_agent(FakeLLM(True), "What is RAG?", facts, iteration=1, max_iterations=3))
    assert result["is_sufficient"] is False


def test_gate_failure_reason_names_the_failing_signal():
    facts = [_fact(0, "en.wikipedia.org", True, 0.9), _fact(1, "en.wikipedia.org", True, 0.9)]
    result = _run(facts)
    assert result["is_sufficient"] is False
    assert "facts=" in result["reason"] and "sources=" in result["reason"]
