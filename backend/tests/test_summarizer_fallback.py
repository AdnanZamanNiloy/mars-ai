"""Summarizer heuristic-fallback relevance (v14): the fallback must rank
candidates by query/sub-question overlap instead of taking the first
sentences that clear a bare threshold.

Regression: a fully-degraded run on a Bangladesh nuclear-vs-solar query
surfaced a UN transcript's Malawi electrification paragraph as a top claim —
the sentence cleared a 0.15 whole-query overlap while being about a
different country entirely.
"""

import asyncio


from app.agents.summarizer import summarizer_agent
from app.core.config import Settings
from app.core.degradation import clear_fallbacks, reset_fallbacks


class ExplodingLLM:
    def __init__(self, tmp_path):
        self.settings = Settings(
            groq_api_key="k", database_url=str(tmp_path / "t.db"), _env_file=None
        )

    async def generate_json(self, *a, **k):
        raise RuntimeError("providers down")


def _results():
    return [
        {
            "title": "Bangladesh power sector outlook",
            "url": "https://example-power.gov.bd/outlook",
            "snippet": "",
            "content": (
                "Solar power in Bangladesh reached 3.2 percent of generation in 2025. "
                "The Rooppur nuclear plant dominates the capital budget with two reactors. "
                "Levelized costs for new solar fell below grid parity last year. "
                "Officials expect tariff pressure to ease after commissioning."
            ),
            "sub_question": "What are the levelized cost values for nuclear and solar power in Bangladesh?",
        },
        {
            # Same source quality, but its text is about a different region.
            "title": "Regional electrification meeting transcript",
            "url": "https://example-transcript.un.org/asset/k1u/x",
            "snippet": "",
            "content": (
                "The government is implementing key programs such as the Malawi Rural "
                "Electrification Program while scaling up mini grids to reach underserved "
                "communities across the lake region. Delegates discussed cookstove adoption. "
                "The chair closed the session thanking interpreters and delegates."
            ),
            "sub_question": "What are the levelized cost values for nuclear and solar power in Bangladesh?",
        },
    ]


def test_fallback_prefers_topically_matching_sentences(tmp_path):
    reset_fallbacks()
    try:
        facts = asyncio.run(
            summarizer_agent(ExplodingLLM(tmp_path), "Compare the economics of nuclear vs solar energy in Bangladesh", _results())
        )
    finally:
        clear_fallbacks()

    assert facts, "fallback must still produce claims"
    claims = " ".join(str(f.get("claim", "")) for f in facts).lower()
    # On-topic sentences from the power-sector source survive.
    assert "solar" in claims or "nuclear" in claims
    # The off-topic regional transcript contributes nothing about Malawi.
    assert "malawi" not in claims
    assert "cookstove" not in claims
    assert "interpreter" not in claims


def test_fallback_ranks_better_sentences_first(tmp_path):
    reset_fallbacks()
    try:
        facts = asyncio.run(
            summarizer_agent(ExplodingLLM(tmp_path), "Compare the economics of nuclear vs solar energy in Bangladesh", _results())
        )
    finally:
        clear_fallbacks()

    power_source_facts = [f for f in facts if "power.gov.bd" in str(f.get("source", ""))]
    assert power_source_facts, "on-topic source must contribute claims"
    # The highest-ranked claims should be the tariff/LCOE/nuclear sentences,
    # not the generic closing sentence.
    top = str(power_source_facts[0].get("claim", "")).lower()
    assert any(term in top for term in ("solar", "nuclear", "cost", "tariff")), top


def test_failed_llm_result_not_cached(tmp_path):
    """An empty result from a failed LLM call must not poison the cache:
    after a provider blip, the next call retries the LLM instead of serving
    the cached [] for a full TTL hour (extended degradation past recovery)."""
    from app.core.cache import get_cache

    class FlakyLLM:
        def __init__(self):
            self.settings = Settings(
                groq_api_key="k", database_url=str(tmp_path / "t.db"), _env_file=None
            )
            self.calls = 0

        async def generate_json(self, *a, **k):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("provider blip")
            return {"facts": [
                {"claim": "Solar power in Bangladesh reached grid parity for new utility projects",
                 "source": "https://power.gov.bd/outlook", "confidence": 0.9},
            ]}

    llm = FlakyLLM()
    results = [{
        "title": "Bangladesh power sector outlook",
        "url": "https://power.gov.bd/outlook",
        "snippet": "",
        "content": "Solar power in Bangladesh reached grid parity for new utility projects.",
        "sub_question": "What are the levelized cost values for solar power in Bangladesh?",
    }]
    reset_fallbacks()
    try:
        first = asyncio.run(summarizer_agent(llm, "Compare the economics of nuclear vs solar energy in Bangladesh", results))
        assert first == [] or all("solar" in f["claim"].lower() for f in first)
        cache = get_cache(llm.settings)
        # The failed call wrote nothing usable; a recovered provider must
        # be attempted again on the next run.
        second = asyncio.run(summarizer_agent(llm, "Compare the economics of nuclear vs solar energy in Bangladesh", results))
        assert llm.calls == 2, "empty LLM result was cached — provider recovery is blocked"
        assert any("grid parity" in f["claim"] for f in second)
    finally:
        clear_fallbacks()
