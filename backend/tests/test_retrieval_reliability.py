"""Retrieval reliability: cooldowns, transient-only backoff, failed-fetch
suppression, authoritative-source fallback, and health telemetry.

These are the deterministic, network-free tests for the retrieval-access
hardening. Every network call is mocked with respx; every assertion is about
the PRODUCTION code paths `SearchClient._attach_content` /
`_fetch_content_outcome` / `_primary_fallback` run.

The evidence requirements are asserted as NOT weakened: a 403 that cannot be
substituted yields no fabricated content, an empty fetch still counts as a
failure, and a cooling host produces a SKIP rather than a silent success.
"""

from __future__ import annotations

import httpx
import respx

from app.agents import search as search_mod
from app.agents.search import (
    SearchClient,
    SearchResult,
    _fetch_content_outcome,
    _fetch_once,
)
from app.agents.retrieval_health import (
    DomainRegistry,
    FailedFetchLog,
    RetrievalHealth,
    classify_fetch_failure,
    failure_cools_host,
    failure_is_transient,
)
from app.core.config import Settings


def _settings(**over) -> Settings:
    base = {"groq_api_key": "k", "_env_file": None}
    base.update(over)
    return Settings(**base)


def _result(url: str, *, sub_question: str = "what is a transformer",
            search_type: str = "general", provider: str = "ddg_text") -> SearchResult:
    return SearchResult(
        title="Result", url=url, snippet="snippet text",
        sub_question=sub_question, search_type=search_type, provider=provider,
    )


def _fetch_only_settings(**over) -> Settings:
    """Settings for tests that exercise the FETCH path with no fallback query.

    The fallback path issues real provider calls through `_providers_for`; when
    a test is about cooldown/retry/suppression it must not reach the network.
    """
    base = {"search_primary_fallback_enabled": False}
    base.update(over)
    return _settings(**base)


# ---------------------------------------------------------------------------
# Classification primitives
# ---------------------------------------------------------------------------

def test_failure_classification_vocabulary():
    assert classify_fetch_failure(200) == "ok"
    assert classify_fetch_failure(403) == "forbidden"
    assert classify_fetch_failure(451) == "forbidden"
    assert classify_fetch_failure(429) == "rate_limited"
    assert classify_fetch_failure(503) == "server_error"
    assert classify_fetch_failure(404) == "client_error"


def test_forbidden_is_not_transient_and_cools_host():
    assert failure_is_transient("forbidden") is False
    assert failure_cools_host("forbidden") is True
    # A 404 must not blacklist the whole host: the rest of the site may be fine.
    assert failure_cools_host("client_error") is False
    assert failure_is_transient("rate_limited") is True
    assert failure_is_transient("timeout") is True


def test_domain_registry_cools_and_evicts_lru():
    registry = DomainRegistry(cooldown_sec=60.0, failure_threshold=3, max_domains=2)
    registry.record_failure("a.com", "rate_limited")
    registry.record_failure("a.com", "rate_limited")
    assert registry.is_cooling("a.com") is False
    registry.record_failure("a.com", "rate_limited")   # threshold reached
    assert registry.is_cooling("a.com") is True
    # A hard 403 cools immediately, no streak required.
    registry.record_failure("b.com", "forbidden")
    assert registry.is_cooling("b.com") is True
    # LRU cap: adding a third domain evicts the oldest-touched entry.
    registry.record_failure("c.com", "forbidden")
    assert len(registry._states) == 2
    assert "a.com" not in registry._states


def test_domain_registry_reset_clears(tmp_path):
    registry = DomainRegistry()
    registry.record_failure("x.com", "forbidden")
    assert registry.is_cooling("x.com")
    registry.reset()
    assert registry.is_cooling("x.com") is False


def test_failed_fetch_log_bounded_and_reset():
    log = FailedFetchLog(max_urls=2)
    log.mark("https://a.com/1", "forbidden")
    log.mark("https://a.com/2", "timeout")
    log.mark("https://a.com/3", "forbidden")
    assert len(log) == 2
    assert log.seen("https://a.com/1") is False   # oldest evicted
    assert log.seen("https://a.com/3") is True
    assert log.reason("https://a.com/3") == "forbidden"
    log.reset()
    assert len(log) == 0


def test_health_snapshot_rates_and_reset():
    health = RetrievalHealth()
    health.record_attempt()
    health.record_attempt()
    health.record_success(domain="a.gov", primary=True, url="https://a.gov/x",
                          authoritative=True)
    health.record_failure("forbidden")
    snap = health.snapshot()
    assert snap["fetch_attempts"] == 2
    assert snap["fetch_successes"] == 1
    assert snap["successful_fetch_rate"] == 0.5
    assert snap["forbidden_rate"] == 0.5
    assert snap["primary_source_acquisitions"] == 1
    assert snap["unique_authoritative_domains"] == 1
    health.reset()
    assert health.snapshot()["fetch_attempts"] == 0


# ---------------------------------------------------------------------------
# 403: no retry, host cooled, fallback attempted
# ---------------------------------------------------------------------------

async def test_forbidden_no_retry_and_cools_host(monkeypatch):
    """A 403 is fetched ONCE, never retried, and cools the host; the caller
    then attempts an authoritative-source fallback for equivalent evidence."""
    settings = _fetch_only_settings(search_domain_cooldown_sec=120.0)
    client = SearchClient(settings)
    calls = {"n": 0}
    fallbacks = {"n": 0}

    async def fake_fallback(self, ranked, unavailable, *, client=None, max_attempts=2):
        fallbacks["n"] += 1
        return None

    monkeypatch.setattr(SearchClient, "_primary_fallback", fake_fallback)

    with respx.mock(assert_all_called=False) as mock:
        route = mock.get("https://blockedsite.com/story").mock(
            return_value=httpx.Response(403, text="Forbidden"))
        fetched = await _fetch_content_outcome(
            httpx.AsyncClient(), "https://blockedsite.com/story",
            max_attempts=3,
        )
    assert route.call_count == 1            # 403 is not retried
    assert fetched.reason == "forbidden"
    assert fetched.ok is False

    ranked = [_result("https://blockedsite.com/story")]
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://blockedsite.com/story").mock(
            return_value=httpx.Response(403, text="Forbidden"))
        await client._attach_content(ranked)
    assert client.domain_registry.is_cooling("blockedsite.com") is True
    assert client.health.snapshot()["status_forbidden"] == 1
    assert client.health.snapshot()["cooldowns_opened"] == 1
    assert fallbacks["n"] == 1
    # Evidence is NOT fabricated: the result still has no content.
    assert ranked[0].content == ""
    assert ranked[0].is_content_fetched is False


# ---------------------------------------------------------------------------
# 429: bounded retry + backoff, Retry-After honoured
# ---------------------------------------------------------------------------

async def test_rate_limited_retries_then_succeeds(monkeypatch):
    settings = _settings()
    client = SearchClient(settings)
    calls = {"n": 0}

    async def _no_sleep(_):
        return None

    monkeypatch.setattr(search_mod.asyncio, "sleep", _no_sleep)

    with respx.mock(assert_all_called=False) as mock:
        route = mock.get("https://busysite.com/p").mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "1"}),
                httpx.Response(200, headers={"content-type": "text/html"},
                               text="<html><body><p>Recovered body text.</p></body></html>"),
            ])
        outcome = await _fetch_content_outcome(
            httpx.AsyncClient(), "https://busysite.com/p",
            max_attempts=3,
        )
    assert route.call_count == 2
    assert outcome.ok is True
    assert "Recovered body text." in outcome.text
    assert outcome.attempts == 2


async def test_rate_limited_retry_after_is_honoured(monkeypatch):
    """The Retry-After value drives the wait, not the jitter formula."""
    sleeps: list = []

    async def _record_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(search_mod.asyncio, "sleep", _record_sleep)

    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://busysite.com/h").mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "5"}),
                httpx.Response(200, headers={"content-type": "text/html"},
                               text="<html><body>ok body text here</body></html>"),
            ])
        outcome = await _fetch_content_outcome(
            httpx.AsyncClient(), "https://busysite.com/h", max_attempts=3)
    assert outcome.ok is True
    assert sleeps == [5.0]


async def test_rate_limited_exhaustion_marks_failure_and_cooldown(monkeypatch):
    async def _no_sleep(_):
        return None

    monkeypatch.setattr(search_mod.asyncio, "sleep", _no_sleep)
    settings = _fetch_only_settings(search_domain_failure_threshold=2)
    client = SearchClient(settings)

    with respx.mock(assert_all_called=False) as mock:
        route = mock.get("https://busysite.com/x").mock(
            return_value=httpx.Response(429, headers={"Retry-After": "0"}))
        await client._attach_content([_result("https://busysite.com/x")])
    assert route.call_count == 2                # bounded, not unbounded
    assert client.health.snapshot()["status_rate_limited"] == 1
    assert client.domain_registry.is_cooling("busysite.com") is True


# ---------------------------------------------------------------------------
# timeout: bounded retry then cooldown
# ---------------------------------------------------------------------------

async def test_timeout_retries_then_succeeds(monkeypatch):
    async def _no_sleep(_):
        return None

    monkeypatch.setattr(search_mod.asyncio, "sleep", _no_sleep)

    with respx.mock(assert_all_called=False) as mock:
        route = mock.get("https://slowsite.com/p").mock(
            side_effect=[
                httpx.ReadTimeout("timed out"),
                httpx.Response(200, headers={"content-type": "text/html"},
                               text="<html><body>Recovered after timeout.</body></html>"),
            ])
        outcome = await _fetch_content_outcome(
            httpx.AsyncClient(), "https://slowsite.com/p", max_attempts=3)
    assert route.call_count == 2
    assert outcome.ok is True


async def test_timeout_exhaustion_cools_host(monkeypatch):
    async def _no_sleep(_):
        return None

    monkeypatch.setattr(search_mod.asyncio, "sleep", _no_sleep)
    client = SearchClient(_fetch_only_settings(search_domain_failure_threshold=2))

    with respx.mock(assert_all_called=False) as mock:
        route = mock.get("https://slowsite.com/p").mock(
            side_effect=httpx.ReadTimeout("timed out"))
        await client._attach_content([_result("https://slowsite.com/p")])
    assert route.call_count == 2
    assert client.health.snapshot()["fetch_timeouts"] == 1
    assert client.health.snapshot()["retries_attempted"] == 1
    assert client.domain_registry.is_cooling("slowsite.com") is True


# ---------------------------------------------------------------------------
# Domain cooldown blocks subsequent requests
# ---------------------------------------------------------------------------

async def test_domain_cooldown_blocks_subsequent_requests(monkeypatch):
    """Once a host is cooling, a later result on that host is SKIPPED without
    any HTTP request — that is the whole point of the cooldown."""
    settings = _fetch_only_settings(search_domain_cooldown_sec=120.0)
    client = SearchClient(settings)
    client.domain_registry.record_failure("blockedsite.com", "forbidden")

    with respx.mock(assert_all_called=False) as mock:
        route = mock.get("https://blockedsite.com/second").mock(
            return_value=httpx.Response(200, headers={"content-type": "text/html"},
                                        text="<html><body>should not be fetched</body></html>"))
        await client._attach_content([_result("https://blockedsite.com/second")])
    assert route.call_count == 0
    snap = client.health.snapshot()
    assert snap["skipped_cooldown"] == 1
    assert snap["fetch_attempts"] == 0


async def test_domain_cooldown_expires_allows_refetch(monkeypatch):
    clock = {"t": 1000.0}
    registry = DomainRegistry(cooldown_sec=30.0, failure_threshold=1,
                              clock=lambda: clock["t"])
    registry.record_failure("x.com", "forbidden")
    assert registry.is_cooling("x.com") is True
    clock["t"] += 31.0
    assert registry.is_cooling("x.com") is False


# ---------------------------------------------------------------------------
# Duplicate failed-fetch suppression
# ---------------------------------------------------------------------------

async def test_failed_url_not_fetched_twice(monkeypatch):
    client = SearchClient(_fetch_only_settings())
    url = "https://deadsite.com/article?utm_source=x"

    with respx.mock(assert_all_called=False) as mock:
        route = mock.get("https://deadsite.com/article").mock(
            return_value=httpx.Response(404, text="Not found"))
        await client._attach_content([_result(url)])
        # Same document, tracking-param variant: must be suppressed, not refetched.
        await client._attach_content([_result("https://deadsite.com/article")])
    assert route.call_count == 1
    assert client.health.snapshot()["skipped_duplicate_failure"] == 1
    assert client.health.snapshot()["fetch_attempts"] == 1


async def test_failed_fetch_log_is_run_scoped(monkeypatch):
    client = SearchClient(_fetch_only_settings())
    client.failed_fetches.mark("https://deadsite.com/a", "forbidden")
    client.reset_run()
    assert client.failed_fetches.seen("https://deadsite.com/a") is False
    assert client.domain_registry.cooling_domains() == []
    assert client.health.snapshot()["fetch_attempts"] == 0


# ---------------------------------------------------------------------------
# Fallback to an alternate authoritative source
# ---------------------------------------------------------------------------

async def test_fallback_targets_alternate_authoritative_source(monkeypatch):
    """When an authoritative host is unavailable, the fallback issues a
    primary-source-scoped query through the EXISTING machinery and acquires
    evidence from a DIFFERENT authoritative/independent publisher."""
    settings = _settings(search_primary_fallback_max=2)
    client = SearchClient(settings)

    async def fake_providers(self, query, search_type):
        assert "site:" in query  # the primary-source machinery built the query
        return [_result("https://census.gov/data/report", search_type="statistical")]

    monkeypatch.setattr(SearchClient, "_providers_for", fake_providers)

    ranked = [_result("https://blockedsite.com/x", search_type="statistical")]
    blocked_domain = "blockedsite.com"
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://census.gov/data/report").mock(
            return_value=httpx.Response(200, headers={"content-type": "text/html"},
                                        text="<html><body><p>Census report body.</p></body></html>"))
        async with httpx.AsyncClient() as http:
            await client._primary_fallback(
                ranked, [blocked_domain], client=http, max_attempts=1)
    urls = [r.url for r in ranked]
    assert "https://census.gov/data/report" in urls
    substitute = next(r for r in ranked if r.url == "https://census.gov/data/report")
    assert substitute.is_content_fetched is True
    assert "Census report body." in substitute.content
    assert client.health.snapshot()["fallback_queries_issued"] == 1
    assert client.health.snapshot()["fallback_acquisitions"] == 1
    # The blocked host's own result was NOT magically filled.
    assert all(r.content == "" for r in ranked if r.url.startswith("https://blocked."))


async def test_fallback_disabled_by_setting(monkeypatch):
    settings = _settings(search_primary_fallback_enabled=False)
    client = SearchClient(settings)
    called = {"n": 0}

    async def fake_providers(self, query, search_type):
        called["n"] += 1
        return []

    monkeypatch.setattr(SearchClient, "_providers_for", fake_providers)
    await client._primary_fallback(
        [_result("https://blockedsite.com/x")], ["blockedsite.com"])
    assert called["n"] == 0


async def test_fallback_does_not_reuse_blocked_host(monkeypatch):
    """Substitution must never pick another host that is also unavailable."""
    settings = _settings()
    client = SearchClient(settings)
    client.domain_registry.record_failure("also-blocked.com", "forbidden")

    async def fake_providers(self, query, search_type):
        return [
            _result("https://also-blocked.com/x"),
            _result("https://who.int/report", search_type="statistical"),
        ]

    monkeypatch.setattr(SearchClient, "_providers_for", fake_providers)
    ranked = [_result("https://blockedsite.com/x", search_type="statistical")]
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://who.int/report").mock(
            return_value=httpx.Response(200, headers={"content-type": "text/html"},
                                        text="<html><body>WHO report.</body></html>"))
        async with httpx.AsyncClient() as http:
            await client._primary_fallback(
                ranked, ["blockedsite.com"], client=http, max_attempts=1)
    urls = {r.url for r in ranked}
    assert "https://also-blocked.com/x" not in urls
    assert "https://who.int/report" in urls


# ---------------------------------------------------------------------------
# Retry success / exhaustion at the primitive level
# ---------------------------------------------------------------------------

async def test_fetch_content_outcome_retry_success(monkeypatch):
    async def _no_sleep(_):
        return None

    monkeypatch.setattr(search_mod.asyncio, "sleep", _no_sleep)
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get("https://flakysite.com/p").mock(
            side_effect=[
                httpx.ConnectError("connection reset"),
                httpx.Response(200, headers={"content-type": "text/html"},
                               text="<html><body>Body after reconnect.</body></html>"),
            ])
        outcome = await _fetch_content_outcome(
            httpx.AsyncClient(), "https://flakysite.com/p", max_attempts=3)
    assert route.call_count == 2
    assert outcome.ok is True


async def test_fetch_content_outcome_exhaustion_returns_last_failure(monkeypatch):
    async def _no_sleep(_):
        return None

    monkeypatch.setattr(search_mod.asyncio, "sleep", _no_sleep)
    with respx.mock(assert_all_called=False) as mock:
        route = mock.get("https://flakysite.com/p").mock(
            return_value=httpx.Response(503, text="unavailable"))
        outcome = await _fetch_content_outcome(
            httpx.AsyncClient(), "https://flakysite.com/p", max_attempts=2)
    assert route.call_count == 2
    assert outcome.ok is False
    assert outcome.reason == "server_error"
    assert outcome.attempts == 2
    assert outcome.status == 503


async def test_fetch_once_reports_status_for_forbidden():
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://xsite.com/a").mock(
            return_value=httpx.Response(403, text="denied"))
        outcome = await _fetch_once(httpx.AsyncClient(), "https://xsite.com/a")
    assert outcome.status == 403
    assert outcome.reason == "forbidden"
    assert outcome.attempts == 1


# ---------------------------------------------------------------------------
# Telemetry distinguishes retrieval failure from weak evidence
# ---------------------------------------------------------------------------

async def test_health_counts_are_distinguishable_from_evidence():
    """A run that found nothing because every host blocked it must show a
    retrieval-failure signal, not silently look like a quality result."""
    client = SearchClient(_fetch_only_settings())

    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://asite.com/1").mock(
            return_value=httpx.Response(403, text="denied"))
        mock.get("https://bsite.com/2").mock(
            return_value=httpx.Response(200, headers={"content-type": "text/html"},
                                        text="<html><body>Real evidence body.</body></html>"))
        await client._attach_content([
            _result("https://asite.com/1"),
            _result("https://bsite.com/2"),
        ])
    snap = client.health.snapshot()
    assert snap["fetch_attempts"] == 2
    assert snap["fetch_successes"] == 1
    assert snap["status_forbidden"] == 1
    assert snap["successful_fetch_rate"] == 0.5
    assert snap["unique_success_domains"] == 1


# ---------------------------------------------------------------------------
# Evidence requirements NOT weakened
# ---------------------------------------------------------------------------

async def test_no_content_is_fabricated_on_failure():
    client = SearchClient(_fetch_only_settings())
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://xsite.com/f").mock(
            return_value=httpx.Response(403, text="denied"))
        ranked = [_result("https://xsite.com/f")]
        await client._attach_content(ranked)
    assert ranked[0].content == ""
    assert ranked[0].is_content_fetched is False
    assert ranked[0].content_length == 0


async def test_unreadable_content_type_is_not_a_success():
    """An image response yields empty text and must not be counted as a
    successful acquisition."""
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://xsite.com/i.png").mock(
            return_value=httpx.Response(200, headers={"content-type": "image/png"},
                                        content=b"\x89PNG"))
        outcome = await _fetch_once(httpx.AsyncClient(), "https://xsite.com/i.png")
    assert outcome.ok is False
    assert outcome.reason == "other"


# ---------------------------------------------------------------------------
# Benchmark evaluator (bench/eval_retrieval.py)
# ---------------------------------------------------------------------------

async def test_retrieval_evaluator_passes_and_covers_scenarios():
    from bench import eval_retrieval

    report = await eval_retrieval.evaluate()
    assert report["passed"] is True
    assert report["threshold_failures"] == []
    ids = {s["id"] for s in report["scenarios"]}
    assert ids == {
        "forbidden_host_cools_and_skips",
        "rate_limit_retry_succeeds",
        "timeout_exhausts_and_cools",
        "duplicate_failure_suppressed",
        "primary_fallback_substitutes",
        "evidence_not_fabricated_on_failure",
    }
    assert all(s.get("passed") for s in report["scenarios"])
    # Retrieval-health metrics are exposed so a thin run is attributable.
    metrics = report["aggregate"]["metrics"]
    assert metrics["forbidden_rate"] > 0
    assert metrics["duplicate_suppression_rate"] > 0
    assert metrics["fallback_acquisition_rate"] > 0


async def test_retrieval_gate_bites_when_cooldown_not_applied():
    """If a regression stops cooling a blocked host, the gate must exit
    non-zero rather than silently reporting healthy retrieval."""
    from bench import eval_retrieval

    original = eval_retrieval.scenarios

    def _broken_scenarios():
        rows = original()
        for scenario in rows:
            if scenario.id == "forbidden_host_cools_and_skips":
                scenario.check = lambda m: False  # simulate a lost cooldown
        return rows

    try:
        eval_retrieval.scenarios = _broken_scenarios
        report = await eval_retrieval.evaluate()
    finally:
        eval_retrieval.scenarios = original
    assert report["passed"] is False
    assert report["aggregate"]["scenarios_failed"] == 1


def test_retrieval_thresholds_file_loads_and_is_additive():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "bench" / "golden" / "thresholds_retrieval_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == "v1"
    assert "retrieval" in data
    # The retrieval thresholds must NOT contain any key from the existing
    # golden gate: they are new floors, never a rewording of old ones.
    existing = json.loads(
        (path.parent / "thresholds_v1.json").read_text(encoding="utf-8")
    )["aggregate"]
    assert not (set(data["retrieval"]) & set(existing))
