"""Citation validation v2: live URL re-check + per-source health verdicts."""

import httpx
import respx

from app.agents.citation_check import (
    check_citations,
    citation_health_note,
)

ANSWER = (
    "Solar grew fast in 2024 [1]. "
    "Wind faced headwinds last year [2]. "
    "Storage costs fell sharply [3]."
    "\n\nSources:\n"
    "[1] iea.org — https://iea.org/solar-2024\n"
    "[2] example.com — https://example.com/wind-dead\n"
    "[3] ember.org — https://ember.org/storage"
)

SUPPORT = {
    "rate": 2 / 3,
    "sentence_details": [
        {"sentence": "Solar grew fast in 2024", "markers": [1], "status": "supported", "support": 0.7},
        {"sentence": "Wind faced headwinds last year", "markers": [2], "status": "unsupported", "support": 0.1},
        {"sentence": "Storage costs fell sharply", "markers": [3], "status": "supported", "support": 0.8},
    ],
}


def _enable_real(monkeypatch):
    from app.agents import citation_check as mod

    monkeypatch.setattr(mod, "check_citations", check_citations)


async def test_reachable_and_broken_urls_classified(monkeypatch):
    _enable_real(monkeypatch)
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://iea.org/solar-2024").mock(return_value=httpx.Response(200))
        mock.head("https://iea.org/solar-2024").mock(return_value=httpx.Response(200))
        mock.head("https://example.com/wind-dead").mock(return_value=httpx.Response(404))
        mock.get("https://example.com/wind-dead").mock(return_value=httpx.Response(404))
        mock.head("https://ember.org/storage").mock(return_value=httpx.Response(200))

        health = await check_citations(ANSWER, SUPPORT, timeout=2.0)

    by_marker = {s["marker"]: s for s in health["sources"]}
    # [1]: reachable + supported -> ok
    assert by_marker[1]["url_status"] == "reachable"
    assert by_marker[1]["status"] == "ok"
    assert by_marker[1]["support_rate"] == 1.0
    # [2]: 404 + unsupported sentence -> bad
    assert by_marker[2]["url_status"] == "unreachable"
    assert by_marker[2]["status"] == "bad"
    # [3]: HEAD-reachable -> ok
    assert by_marker[3]["status"] == "ok"

    assert health["summary"]["ok"] == 2
    assert health["summary"]["bad"] == 1


async def test_head_rejection_falls_back_to_ranged_get(monkeypatch):
    _enable_real(monkeypatch)
    # Alphabetically first URL (ember.org) is the one checked with max_sources=1.
    with respx.mock(assert_all_called=False) as mock:
        head = mock.head("https://ember.org/storage").mock(
            return_value=httpx.Response(405))
        get = mock.get("https://ember.org/storage").mock(
            return_value=httpx.Response(206))

        health = await check_citations(ANSWER, SUPPORT, timeout=2.0, max_sources=1)

    assert head.call_count == 1
    assert get.call_count == 1
    by_marker = {s["marker"]: s for s in health["sources"]}
    assert by_marker[3]["url_status"] == "reachable"


async def test_disabled_check_marks_unchecked(monkeypatch):
    _enable_real(monkeypatch)
    health = await check_citations(ANSWER, SUPPORT, enabled=False)
    assert health["checked"] == 0
    assert all(s["url_status"] == "unchecked" for s in health["sources"])
    # Unreachable-unknown + unsupported sentences still counts as warn-able?
    # No — without live data only sentence support decides:
    assert health["summary"]["unchecked"] == 3


async def test_warn_for_partial_support(monkeypatch):
    _enable_real(monkeypatch)
    with respx.mock(assert_all_called=False) as mock:
        mock.head("https://iea.org/solar-2024").mock(return_value=httpx.Response(200))
        mock.head("https://example.com/wind-dead").mock(return_value=httpx.Response(200))
        mock.head("https://ember.org/storage").mock(return_value=httpx.Response(200))
        health = await check_citations(ANSWER, SUPPORT, timeout=2.0)

    by_marker = {s["marker"]: s for s in health["sources"]}
    assert by_marker[2]["status"] == "warn"   # reachable but unsupported sentence
    assert by_marker[1]["status"] == "ok"
    assert health["summary"]["warn"] == 1


async def test_no_legend_returns_empty(monkeypatch):
    _enable_real(monkeypatch)
    health = await check_citations("No citations here.", None)
    assert health == {"checked": 0, "sources": [], "summary": {}, "enabled": True}


def test_health_note_wording():
    note = citation_health_note({
        "enabled": True,
        "summary": {"ok": 2, "warn": 1, "broken": 1, "bad": 0},
    })
    assert note and "unreachable" in note and "unsupported" in note


def test_health_note_none_when_healthy():
    assert citation_health_note({"enabled": True, "summary": {"ok": 3}}) is None
    assert citation_health_note(None) is None
    assert citation_health_note({"enabled": False, "summary": {"bad": 2}}) is None


async def test_network_failure_is_unreachable_not_exception(monkeypatch):
    _enable_real(monkeypatch)
    with respx.mock(assert_all_called=False) as mock:
        mock.head("https://iea.org/solar-2024").mock(side_effect=httpx.ConnectError("no net"))
        mock.get("https://iea.org/solar-2024").mock(side_effect=httpx.ConnectError("no net"))
        mock.head("https://example.com/wind-dead").mock(side_effect=httpx.ConnectError("no net"))
        mock.get("https://example.com/wind-dead").mock(side_effect=httpx.ConnectError("no net"))
        mock.head("https://ember.org/storage").mock(side_effect=httpx.ConnectError("no net"))
        mock.get("https://ember.org/storage").mock(side_effect=httpx.ConnectError("no net"))
        health = await check_citations(ANSWER, SUPPORT, timeout=1.0)

    assert all(s["url_status"] == "unreachable" for s in health["sources"])
    # [2] unreachable + unsupported -> bad; [1],[3] unreachable + supported -> broken
    by_marker = {s["marker"]: s for s in health["sources"]}
    assert by_marker[1]["status"] == "broken"
    assert by_marker[2]["status"] == "bad"


async def test_max_sources_bounds_work(monkeypatch):
    _enable_real(monkeypatch)
    calls = []

    async def fake_probe(url, timeout):
        calls.append(url)
        return {"url": url, "url_status": "reachable", "http_status": 200, "final_url": url}

    import app.agents.citation_check as mod

    monkeypatch.setattr(mod, "_probe_url", fake_probe)
    health = await check_citations(ANSWER, SUPPORT, max_sources=2)
    # Alphabetical order: ember.org, example.com checked; iea.org unchecked.
    assert sorted(calls) == ["https://ember.org/storage", "https://example.com/wind-dead"]
    by_marker = {s["marker"]: s for s in health["sources"]}
    assert by_marker[3]["url_status"] == "reachable"
    assert by_marker[1]["url_status"] == "unchecked"
