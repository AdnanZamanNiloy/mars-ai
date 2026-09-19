"""Query Router Agent: direct-answer vs. research decision (R1).

R1 is unwired — these tests pin the decision logic, not the graph wiring.
The two properties that matter most:
  * hard signals (freshness/quantitative/decision/contested/type/ambiguity)
    always force research, even against a model that says "direct";
  * the deterministic fallback NEVER grants a direct answer, so a router
    failure degrades to research, never to an ungrounded answer.
"""

from app.agents.router import (
    DEFAULT_MIN_DIRECT_CONFIDENCE,
    DIRECT,
    RESEARCH,
    RouteDecision,
    deterministic_route,
    route_query,
)


class FakeLLM:
    def __init__(self, payload=None, error=None, settings=None):
        self.payload = payload
        self.error = error
        self.settings = settings
        self.calls = []

    async def generate_json(self, system_prompt, user_prompt, retries=3, response_model=None):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if self.error is not None:
            raise self.error
        return self.payload


# ---------------------------------------------------------------------------
# Deterministic gate
# ---------------------------------------------------------------------------

def test_freshness_forces_research():
    d = deterministic_route("What is the latest price of gold?")
    assert d.path == RESEARCH
    assert "freshness" in d.signals["hard_blockers"]


def test_bare_year_forces_research():
    d = deterministic_route("How many electric vehicles were sold in 2025?")
    assert d.path == RESEARCH
    assert "freshness" in d.signals["hard_blockers"]


def test_quantitative_forces_research():
    d = deterministic_route("What is the market size of the battery industry?")
    assert d.path == RESEARCH
    assert "quantitative" in d.signals["hard_blockers"]


def test_decision_framing_forces_research():
    d = deterministic_route("Should we invest in nuclear energy for our grid?")
    assert d.path == RESEARCH
    assert "decision" in d.signals["hard_blockers"]


def test_contested_topic_forces_research():
    d = deterministic_route("Is this supplement safe to take daily?")
    assert d.path == RESEARCH
    assert "contested" in d.signals["hard_blockers"]


def test_comparative_query_forces_research():
    d = deterministic_route("Solar vs nuclear energy for Bangladesh")
    assert d.path == RESEARCH
    assert "query_type" in d.signals["hard_blockers"]


def test_ambiguity_forces_research():
    intent = {"ambiguity": True, "query_type": "factual"}
    d = deterministic_route("What is transformer?", intent=intent)
    assert d.path == RESEARCH
    assert "ambiguity" in d.signals["hard_blockers"]


def test_deterministic_never_grants_direct():
    """The fallback can block, never grant: an unremarkable factual query
    still routes to research until the model explicitly clears it."""
    d = deterministic_route("What is a for-loop?")
    assert d.path == RESEARCH
    assert d.origin == "heuristic"


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

async def test_model_clearance_grants_direct():
    llm = FakeLLM({
        "path": "direct",
        "confidence": 0.9,
        "reason": "Stable programming concept.",
        "needs_research": False,
        "answer_sketch": "A for-loop iterates over a sequence.",
    })
    d = await route_query(llm, "What is a for-loop?")
    assert d.path == DIRECT
    assert d.origin == "llm"
    assert d.answer_sketch.startswith("A for-loop")


async def test_model_direct_vetoed_by_freshness_gate():
    """A model that wrongly says 'direct' on a freshness query is overridden
    by the deterministic gate — and no LLM call is even made."""
    llm = FakeLLM({
        "path": "direct",
        "confidence": 0.99,
        "reason": "I know this.",
        "needs_research": False,
        "answer_sketch": "Gold is about $2000.",
    })
    d = await route_query(llm, "What is the latest price of gold?")
    assert d.path == RESEARCH
    assert llm.calls == [], "hard-blocked queries skip the LLM call entirely"


async def test_low_confidence_stays_research():
    llm = FakeLLM({
        "path": "direct",
        "confidence": DEFAULT_MIN_DIRECT_CONFIDENCE - 0.1,
        "reason": "Probably know it.",
        "needs_research": False,
        "answer_sketch": "Answer.",
    })
    d = await route_query(llm, "What is a for-loop?")
    assert d.path == RESEARCH


async def test_explicit_needs_research_stays_research():
    llm = FakeLLM({
        "path": "direct",
        "confidence": 0.99,
        "reason": "Uncertain.",
        "needs_research": True,
        "answer_sketch": "",
    })
    d = await route_query(llm, "What is a for-loop?")
    assert d.path == RESEARCH


async def test_llm_failure_falls_back_to_research():
    llm = FakeLLM(error=RuntimeError("provider down"))
    d = await route_query(llm, "What is a for-loop?")
    assert d.path == RESEARCH
    assert d.origin == "heuristic"


async def test_empty_payload_falls_back_to_research():
    llm = FakeLLM(payload={})
    d = await route_query(llm, "What is a for-loop?")
    assert d.path == RESEARCH
    assert d.origin == "heuristic"


async def test_router_disabled_uses_deterministic():
    from app.core.config import Settings

    settings = Settings(router_enabled=False, groq_api_key="k", _env_file=None)
    llm = FakeLLM(payload={"path": "direct", "confidence": 0.99,
                           "reason": "", "needs_research": False}, settings=settings)
    d = await route_query(llm, "What is a for-loop?")
    assert d.path == RESEARCH
    assert llm.calls == []


async def test_min_direct_confidence_setting_honored():
    from app.core.config import Settings

    settings = Settings(router_min_direct_confidence=0.95, groq_api_key="k", _env_file=None)
    llm = FakeLLM(payload={"path": "direct", "confidence": 0.9,
                           "reason": "", "needs_research": False}, settings=settings)
    d = await route_query(llm, "What is a for-loop?")
    assert d.path == RESEARCH


def test_to_dict_shape():
    d = RouteDecision(path=DIRECT, reason="r", confidence=0.9)
    assert set(d.to_dict()) == {
        "path", "reason", "confidence", "signals", "origin", "answer_sketch"
    }
    assert d.is_direct is True
