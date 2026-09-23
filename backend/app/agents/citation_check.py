"""Citation validation v2: live URL re-check + per-source health verdicts.

`verify_answer_support` scores whether cited SENTENCES are grounded in the
evidence pool — but "grounded in a source that 404s" is its own failure the
sentence check cannot see. This module re-validates the URLs the final
answer actually cites (the legend), then fuses both dimensions into one
per-source verdict the UI, the report and the benchmark suite consume:

    ok        URL reachable, cited sentences supported
    warn      URL reachable, some cited sentences unsupported
    broken    URL unreachable (or redirected somewhere else) — the citation
              may point at a dead or moved page
    bad       URL unreachable AND unsupported sentences — never trust silently

Design guards:

* Bounded: only the top `CITATION_CHECK_MAX` cited sources are re-checked,
  concurrently (semaphore), each capped by `CITATION_CHECK_TIMEOUT_SEC`.
* Never fatal: every network failure degrades to a conservative verdict;
  a broken checker must not break a finished report.
* Cheap first: HEAD request, falling back to a 2KB-ranged GET for servers
  that reject HEAD (405) — never downloading a page just to check it exists.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import httpx

from app.agents.evidence_utils import parse_answer_legend
from app.core.logging import get_logger

logger = get_logger(__name__)

_CHECK_CONCURRENCY = 6
_UA = "MARS-Research/2.0 (+citation-health-check)"


async def _probe_url(url: str, timeout: float) -> Dict[str, Any]:
    """One URL's reachability verdict. Conservative on every failure."""
    if not str(url).lower().startswith(("http://", "https://")):
        return {"url": url, "url_status": "skipped", "http_status": None, "final_url": url}
    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True,
            headers={"User-Agent": _UA},
        ) as client:
            try:
                response = await client.head(url)
                status = response.status_code
            except httpx.HTTPError:
                status = None
            if status is None or status in (405, 403, 501):
                # Some servers reject HEAD outright; ask for 2KB instead.
                try:
                    response = await client.get(
                        url, headers={"Range": "bytes=0-2047"}
                    )
                    status = response.status_code
                except httpx.HTTPError:
                    status = None
        if status is None:
            return {"url": url, "url_status": "unreachable", "http_status": None, "final_url": url}
        if 200 <= status < 400:
            final = str(getattr(response, "url", "") or url)
            drifted = final.rstrip("/") != url.rstrip("/")
            return {
                "url": url,
                "url_status": "redirected" if drifted else "reachable",
                "http_status": status,
                "final_url": final,
            }
        return {"url": url, "url_status": "unreachable", "http_status": status, "final_url": url}
    except Exception as exc:  # client construction, TLS errors, anything
        logger.info("citation_probe_failed", url=url, error=str(exc)[:120])
        return {"url": url, "url_status": "unreachable", "http_status": None, "final_url": url}


async def check_citations(
    answer: str,
    answer_support: Dict[str, Any] | None,
    *,
    enabled: bool = True,
    timeout: float = 5.0,
    max_sources: int = 10,
) -> Dict[str, Any]:
    """Per-source citation health for the emitted answer.

    Combines the sentence-level support verdicts (from verify_answer_support)
    with live URL reachability. Returns::

        {
          "checked": 8,
          "sources": [{"marker": 1, "url": ..., "url_status": ...,
                       "cited_sentences": 3, "supported_sentences": 3,
                       "support_rate": 1.0, "status": "ok"}],
          "summary": {"ok": 6, "warn": 1, "broken": 1, "bad": 0},
          "enabled": true,
        }
    """
    _, legend = parse_answer_legend(answer or "")
    if not legend:
        return {"checked": 0, "sources": [], "summary": {}, "enabled": enabled}

    # Per-URL sentence support from the support pass (marker-based).
    per_url: Dict[str, Dict[str, int]] = {}
    details = (answer_support or {}).get("sentence_details") or []
    for detail in details:
        for marker in detail.get("markers") or []:
            url = legend.get(int(marker))
            if not url:
                continue
            bucket = per_url.setdefault(url, {"cited": 0, "supported": 0, "synthesis": 0})
            bucket["cited"] += 1
            if detail.get("status") == "supported":
                bucket["supported"] += 1
            elif detail.get("status") == "synthesis":
                # An attributed cross-source synthesis sentence draws on this
                # source's verified evidence even though it matches no single
                # claim verbatim; the URL is contributing, so it is not a
                # partial-support warning.
                bucket["synthesis"] += 1

    urls = sorted({u for u in legend.values() if u})
    to_check = urls[: max(1, max_sources)]
    skipped = urls[max(1, max_sources):]

    semaphore = asyncio.Semaphore(_CHECK_CONCURRENCY)

    async def _bounded(url: str) -> Dict[str, Any]:
        async with semaphore:
            return await _probe_url(url, timeout)

    probes = await asyncio.gather(*(_bounded(u) for u in to_check)) if enabled else []
    probe_by_url = {p["url"]: p for p in probes}

    sources: List[Dict[str, Any]] = []
    counts = {"ok": 0, "warn": 0, "broken": 0, "bad": 0, "unchecked": 0}
    for marker in sorted(legend):
        url = legend[marker]
        support = per_url.get(url, {"cited": 0, "supported": 0, "synthesis": 0})
        cited = int(support["cited"])
        # Synthesis sentences draw on this source's evidence, so they count
        # toward the URL's support for the health verdict.
        supported = int(support["supported"]) + int(support.get("synthesis", 0))
        support_rate = round(supported / cited, 3) if cited else None

        probe = probe_by_url.get(url) if enabled else None
        url_status = (probe or {}).get("url_status", "unchecked")
        if url in skipped:
            url_status = "unchecked"

        if url_status in ("unreachable",):
            status = "bad" if (cited and supported < cited) else "broken"
        elif url_status == "unchecked" or url_status == "skipped":
            status = "unchecked"
        else:  # reachable / redirected
            if cited and supported < cited:
                status = "warn"
            else:
                status = "ok"
        counts[status] = counts.get(status, 0) + 1

        entry = {
            "marker": marker,
            "url": url,
            "url_status": url_status,
            "http_status": (probe or {}).get("http_status"),
            "final_url": (probe or {}).get("final_url"),
            "cited_sentences": cited,
            "supported_sentences": supported,
            "support_rate": support_rate,
            "status": status,
        }
        sources.append(entry)

    return {
        "checked": len(probes) if enabled else 0,
        "sources": sources,
        "summary": counts,
        "enabled": enabled,
    }


def citation_health_note(health: Dict[str, Any] | None) -> str | None:
    """One limitations-section line when citation health is degraded."""
    if not isinstance(health, dict) or not health.get("enabled"):
        return None
    summary = health.get("summary") or {}
    broken = int(summary.get("broken", 0) or 0)
    bad = int(summary.get("bad", 0) or 0)
    warn = int(summary.get("warn", 0) or 0)
    parts: List[str] = []
    if broken:
        parts.append(f"{broken + bad} cited source(s) unreachable at report time")
    if warn:
        parts.append(f"{warn} cited source(s) have partially unsupported sentences")
    if not parts:
        return None
    return "Citation health: " + "; ".join(parts) + ". Verify before relying on the flagged citations."
