"""Reliability #4: distinguish provider-transient degradation from weak evidence.

A provider failure must never be silently interpreted as weak evidence, and a
transient 429/timeout must not be treated as an outage that opens the 60s
breaker. These six cases are the contract:

  1. healthy provider                        -> no degradation recorded
  2. transient 429 then 200                   -> retry succeeds, no degradation
  3. repeated 429 then fallback provider      -> completes, reason=provider-transient
  4. provider exhaustion                      -> explicit degraded state, no weak-evidence
  5. genuine weak evidence (provider OK)      -> weak-evidence, NOT provider degradation
  6. provider degradation + weak evidence     -> BOTH represented separately

All network is mocked with respx; no real provider is ever contacted.
"""
import json

import httpx
import pytest
import respx

from app.core.config import Settings
from app.core.degradation import (
    EVIDENCE_WEAK,
    PROVIDER_HARD,
    PROVIDER_TRANSIENT,
    clear_fallbacks,
    degradation_summary,
    has_provider_degradation,
    record_fallback,
    reset_fallbacks,
)
from app.core.llm import (
    AllProvidersFailedError,
    LLMClient,
    classify_provider_failure,
)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
HF_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"


def _groq_response(content: dict) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}]})


def _hf_response(content: dict) -> httpx.Response:
    return httpx.Response(200, json=[{"generated_text": json.dumps(content)}])


def _client(**over) -> LLMClient:
    base = {
        "groq_api_key": "test-key",
        "huggingface_api_key": "test-hf-key",
        "_env_file": None,
    }
    base.update(over)
    return LLMClient(Settings(**base))


# ---------------------------------------------------------------------------
# Case 1 — healthy provider: no degradation recorded
# ---------------------------------------------------------------------------

async def test_case1_healthy_provider_records_no_degradation():
    client = _client()
    reset_fallbacks()
    try:
        with respx.mock(assert_all_called=False) as mock:
            mock.post(GROQ_URL).mock(return_value=_groq_response({"ok": True}))
            result = await client.generate_json("sp", "up")
        assert result == {"ok": True}
        assert has_provider_degradation() is False
        summary = degradation_summary()
        assert summary["agents"] == []
        assert summary["provider_degraded"] is False
        assert summary["provider_kinds"] == []
    finally:
        clear_fallbacks()


# ---------------------------------------------------------------------------
# Case 2 — transient 429 then 200: retry succeeds, NO degradation
# ---------------------------------------------------------------------------

async def test_case2_transient_429_then_success_no_degradation():
    client = _client()
    reset_fallbacks()
    try:
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(GROQ_URL).mock(side_effect=[
                httpx.Response(429, headers={"retry-after": "0"}, json={"error": "TPM"}),
                _groq_response({"ok": True}),
            ])
            result = await client.generate_json("sp", "up")
        assert result == {"ok": True}
        assert route.call_count == 2
        # Recovered inside the call: the failure never escaped the retry loop,
        # so nothing is degraded and the breaker was NOT opened.
        assert has_provider_degradation() is False
        assert client.groq_breaker.is_open() is False
        assert degradation_summary()["provider_kinds"] == []
    finally:
        clear_fallbacks()


# ---------------------------------------------------------------------------
# Case 3 — repeated 429: fallback provider used, run completes, provider-transient
# ---------------------------------------------------------------------------

async def test_case3_repeated_429_falls_back_and_records_provider_transient():
    client = _client()
    reset_fallbacks()
    try:
        with respx.mock(assert_all_called=False) as mock:
            groq_route = mock.post(GROQ_URL).mock(
                return_value=httpx.Response(429, headers={"retry-after": "0"}, json={"error": "TPM"}))
            hf_route = mock.post(HF_URL).mock(return_value=_hf_response({"via": "hf"}))
            result = await client.generate_json("sp", "up")
        assert result == {"via": "hf"}
        assert groq_route.call_count >= 1 and hf_route.call_count == 1
        # The run completed, but on a fallback provider after a transient
        # failure — that is recorded and must NOT read as weak evidence.
        summary = degradation_summary()
        assert summary["provider_degraded"] is True
        assert PROVIDER_TRANSIENT in summary["provider_kinds"]
        assert summary["evidence_agents"] == []
        # Crucially: a burst of 429s must not open the 60s breaker.
        assert client.groq_breaker.is_open() is False
    finally:
        clear_fallbacks()


# ---------------------------------------------------------------------------
# Case 4 — provider exhaustion: explicit degraded state, no weak-evidence
# ---------------------------------------------------------------------------

async def test_case4_provider_exhaustion_is_explicit_not_weak_evidence():
    client = _client(huggingface_api_key="")
    reset_fallbacks()
    try:
        with respx.mock(assert_all_called=False) as mock:
            mock.post(GROQ_URL).mock(
                return_value=httpx.Response(429, headers={"retry-after": "0"}, json={"error": "TPM"}))
            with pytest.raises(AllProvidersFailedError):
                await client.generate_json("sp", "up")
        summary = degradation_summary()
        assert summary["provider_degraded"] is True
        assert PROVIDER_TRANSIENT in summary["provider_kinds"]
        # No agent fallback was recorded, so nothing is mislabeled weak evidence.
        assert summary["agents"] == []
        assert summary["evidence_agents"] == []
    finally:
        clear_fallbacks()


# ---------------------------------------------------------------------------
# Case 5 — genuine weak evidence: provider OK, evidence thin
# ---------------------------------------------------------------------------

def test_case5_weak_evidence_is_not_provider_degradation():
    reset_fallbacks()
    try:
        # Provider answered fine (no transport failure recorded); the stage
        # fell back because the evidence/model output was unusable.
        record_fallback("summarizer", reason=EVIDENCE_WEAK)
        summary = degradation_summary()
        assert summary["provider_degraded"] is False
        assert summary["provider_kinds"] == []
        assert summary["agents"] == ["summarizer"]
        assert summary["reasons"]["summarizer"] == EVIDENCE_WEAK
        assert summary["evidence_agents"] == ["summarizer"]
        assert summary["provider_agents"] == []
    finally:
        clear_fallbacks()


# ---------------------------------------------------------------------------
# Case 6 — provider degradation + weak evidence: BOTH represented separately
# ---------------------------------------------------------------------------

async def test_case6_provider_degradation_and_weak_evidence_both_reported():
    client = _client(huggingface_api_key="")
    reset_fallbacks()
    try:
        with respx.mock(assert_all_called=False) as mock:
            mock.post(GROQ_URL).mock(
                return_value=httpx.Response(429, headers={"retry-after": "0"}, json={"error": "TPM"}))
            with pytest.raises(AllProvidersFailedError):
                await client.generate_json("sp", "up")
        # The evidence stage also degraded for a non-provider reason.
        record_fallback("summarizer", reason=EVIDENCE_WEAK)
        record_fallback("synthesizer", reason=PROVIDER_TRANSIENT)

        summary = degradation_summary()
        assert summary["provider_degraded"] is True
        assert PROVIDER_TRANSIENT in summary["provider_kinds"]
        # Weak evidence and provider degradation are listed SEPARATELY:
        assert summary["reasons"]["summarizer"] == EVIDENCE_WEAK
        assert summary["reasons"]["synthesizer"] == PROVIDER_TRANSIENT
        assert summary["evidence_agents"] == ["summarizer"]
        assert summary["provider_agents"] == ["synthesizer"]
    finally:
        clear_fallbacks()


# ---------------------------------------------------------------------------
# Classification unit contract (reused by breaker + degradation)
# ---------------------------------------------------------------------------

def test_classify_transient_vs_hard():
    req = httpx.Request("POST", GROQ_URL)

    def status(code: int) -> httpx.HTTPStatusError:
        return httpx.HTTPStatusError("e", request=req, response=httpx.Response(code, json={}))

    assert classify_provider_failure(httpx.ReadTimeout("slow")) == PROVIDER_TRANSIENT
    assert classify_provider_failure(httpx.ConnectError("down")) == PROVIDER_TRANSIENT
    assert classify_provider_failure(status(429)) == PROVIDER_TRANSIENT
    assert classify_provider_failure(status(503)) == PROVIDER_TRANSIENT
    assert classify_provider_failure(status(401)) == PROVIDER_HARD
    assert classify_provider_failure(status(402)) == PROVIDER_HARD
    assert classify_provider_failure(status(404)) == PROVIDER_HARD


async def test_hard_failure_fails_fast_and_records_hard_kind():
    client = _client(huggingface_api_key="")
    reset_fallbacks()
    try:
        with respx.mock(assert_all_called=False) as mock:
            route = mock.post(GROQ_URL).mock(
                return_value=httpx.Response(402, json={"error": "payment required"}))
            with pytest.raises(AllProvidersFailedError):
                await client.generate_json("sp", "up")
        assert route.call_count == 1
        summary = degradation_summary()
        assert summary["provider_degraded"] is True
        assert PROVIDER_HARD in summary["provider_kinds"]
    finally:
        clear_fallbacks()


async def test_429_burst_does_not_open_breaker_but_5xx_does():
    """Regression: a throttled (429) provider is reachable and must keep being
    tried; a genuine outage (5xx) still opens the breaker after the threshold."""
    reset_fallbacks()
    try:
        throttled = _client(huggingface_api_key="")
        with respx.mock(assert_all_called=False) as mock:
            mock.post(GROQ_URL).mock(
                return_value=httpx.Response(429, headers={"retry-after": "0"}, json={"error": "TPM"}))
            for _ in range(3):
                with pytest.raises(AllProvidersFailedError):
                    await throttled.generate_json("sp", "up")
        assert throttled.groq_breaker.is_open() is False

        down = _client(huggingface_api_key="")
        down.groq_breaker = type(down.groq_breaker)(threshold=1, cooldown_sec=60.0)
        with respx.mock(assert_all_called=False) as mock:
            mock.post(GROQ_URL).mock(return_value=httpx.Response(500, json={"error": "down"}))
            with pytest.raises(AllProvidersFailedError):
                await down.generate_json("sp", "up")
        assert down.groq_breaker.is_open() is True
    finally:
        clear_fallbacks()


def test_confidence_capped_when_provider_degraded():
    """A transport failure must not inflate confidence even when the pool
    happens to look well-corroborated (extractive claims self-verify)."""
    from app.core.confidence import compute_confidence

    facts = [
        {
            "claim": f"Transformer {i} study measured efficiency at {i * 10} percent",
            "source": f"https://site{i}.org/paper", "confidence": 0.9,
            "verified": True, "verification_score": 0.9, "corroboration_count": 2,
        }
        for i in range(1, 5)
    ]
    critique = {"is_sufficient": True}
    healthy = compute_confidence(facts, critique, 1, 3, degraded=[])
    degraded = compute_confidence(facts, critique, 1, 3, degraded=[], provider_degraded=True)
    assert healthy["overall"] > degraded["overall"]
    assert degraded["overall"] <= 0.55
    assert any("provider" in n for n in degraded["notes"])


def test_degradation_summary_shape_is_stable():
    """The event payload keys must always exist so the frontend never has to
    guard against absence (additive contract)."""
    reset_fallbacks()
    try:
        summary = degradation_summary()
        for key in ("agents", "reasons", "provider_failures", "provider_degraded",
                    "provider_kinds", "provider_agents", "evidence_agents"):
            assert key in summary
    finally:
        clear_fallbacks()
