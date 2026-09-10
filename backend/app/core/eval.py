"""Evaluation Lab scoring (4.1): pure functions over finished-run metrics.

A "metric set" is what scripts/run_eval.py collects per query — either from
the live NDJSON stream or from a persisted trace. Scoring never touches
I/O so it stays unit-testable without API keys.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

# Strongest model on our provider key — used ONLY as a rater, never as
# ground truth. Scores measure prose quality (which counts saturate on),
# with the known caveat that model judges favor fluent-but-thin text.
JUDGE_MODEL_DEFAULT = "openai/gpt-oss-120b"
JUDGE_REPORT_CHARS = 6000


def score_query(expectation: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Check one eval query's expectations against its observed metrics.

    Returns per-check booleans plus an overall pass. Missing metrics count
    as failures — an eval that can't observe something shouldn't pass it.
    """
    claims = metrics.get("claims")
    verified = metrics.get("verified")
    decisions = metrics.get("decisions")

    claim_check = (
        isinstance(claims, int)
        and claims >= int(expectation.get("min_claims", 0))
    )
    verified_check = (
        isinstance(verified, int)
        and verified >= int(expectation.get("min_verified", 0))
    )
    if expectation.get("expect_decision_options"):
        decision_check = isinstance(decisions, int) and decisions >= 2
    else:
        decision_check = True
    completed_check = metrics.get("status") == "completed"

    passed = claim_check and verified_check and decision_check and completed_check
    return {
        "claims_ok": claim_check,
        "verified_ok": verified_check,
        "decisions_ok": decision_check,
        "completed_ok": completed_check,
        "passed": passed,
    }


def summarize_batch(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate one eval batch into trend-trackable headline numbers."""
    if not rows:
        return {
            "queries": 0,
            "pass_rate": 0.0,
            "avg_confidence": 0.0,
            "avg_claims": 0.0,
            "avg_verified": 0.0,
            "contradiction_rate": 0.0,
            "avg_cost": 0.0,
            "degraded": 0,
            "degraded_rate": 0.0,
        }
    n = len(rows)

    def _mean(key: str) -> float:
        vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    degraded = sum(1 for r in rows if r.get("degraded"))
    judged = [r.get("judge_score") for r in rows if isinstance(r.get("judge_score"), (int, float))]
    return {
        "queries": n,
        "pass_rate": sum(1 for r in rows if r.get("passed")) / n,
        "avg_confidence": _mean("confidence"),
        "avg_claims": _mean("claims"),
        "avg_verified": _mean("verified"),
        "contradiction_rate": sum(1 for r in rows if (r.get("contradictions") or 0) > 0) / n,
        "avg_cost": _mean("cost"),
        "degraded": degraded,
        "degraded_rate": degraded / n,
        "avg_judge": (sum(judged) / len(judged)) if judged else None,
    }


def format_trend(current: Dict[str, Any], previous: Dict[str, Any] | None) -> str:
    """One-line-per-metric comparison of this batch vs the previous one."""
    lines = [
        f"queries: {current['queries']}  pass rate: {current['pass_rate']:.0%}"
        + (f" (was {previous['pass_rate']:.0%})" if previous else " (no prior batch)"),
        _line("avg confidence", current["avg_confidence"], previous["avg_confidence"] if previous else None, ".2f"),
        _line("avg claims", current["avg_claims"], previous["avg_claims"] if previous else None, ".1f"),
        _line("avg verified", current["avg_verified"], previous["avg_verified"] if previous else None, ".1f"),
        _line("contradiction rate", current["contradiction_rate"], previous["contradiction_rate"] if previous else None, "%"),
        _line("avg cost $", current["avg_cost"], previous["avg_cost"] if previous else None, ".4f"),
        _line("degraded rows", float(current["degraded"]), float(previous["degraded"]) if previous else None, ".0f"),
    ]
    cur_judge = current.get("avg_judge")
    prev_judge = previous.get("avg_judge") if previous else None
    if cur_judge is not None or prev_judge is not None:
        cur_s = f"{cur_judge:.2f}" if cur_judge is not None else "—"
        tail = ""
        if prev_judge is not None:
            tail = f" (was {prev_judge:.2f})"
            if cur_judge is not None and abs(cur_judge - prev_judge) > 1e-9:
                tail += " ▲" if cur_judge > prev_judge else " ▼"
        lines.append(f"avg judge /5: {cur_s}{tail}")
    return "\n".join(lines)


def _line(label: str, cur: float, prev: float | None, fmt: str) -> str:
    if fmt == "%":
        cur_s = f"{cur:.0%}"
        prev_s = f"{prev:.0%}" if prev is not None else None
    else:
        cur_s = format(cur, fmt)
        prev_s = format(prev, fmt) if prev is not None else None
    arrow = ""
    if prev is not None and abs(cur - prev) > 1e-9:
        arrow = " ▲" if cur > prev else " ▼"
    return f"{label}: {cur_s}" + (f" (was {prev_s}){arrow}" if prev_s else "") + ""


JUDGE_RUBRIC = """\
grounding (1-5): are factual claims backed by cited sources [n]? 1 = bare assertions, 5 = every claim cited.
coverage (1-5): does the answer address the query's angles? 1 = off-topic/thin, 5 = complete.
clarity (1-5): is it structured and readable? 1 = wall of text, 5 = clear paragraphs.
"""


def judge_prompt(query: str, report_markdown: str) -> str:
    """Rater prompt for an LLM judge. Pure string building — the HTTP call
    lives in scripts/run_eval.py so this stays offline-testable."""
    excerpt = (report_markdown or "")[:JUDGE_REPORT_CHARS]
    return (
        "You are rating a research answer. Score ONLY what is in the text below.\n\n"
        f"Query: {query}\n\nAnswer:\n{excerpt}\n\n"
        f"Rubric:\n{JUDGE_RUBRIC}\n"
        'Return ONLY valid JSON: {"grounding": 1-5, "coverage": 1-5, "clarity": 1-5}'
    )


def parse_judge_scores(payload: Any) -> Dict[str, float] | None:
    """Validate a judge response; None when malformed (never crash scoring)."""
    if not isinstance(payload, dict):
        return None
    scores = {}
    for key in ("grounding", "coverage", "clarity"):
        try:
            value = int(payload.get(key))
        except (TypeError, ValueError):
            return None
        if not 1 <= value <= 5:
            return None
        scores[key] = float(value)
    scores["overall"] = round(sum(scores.values()) / 3, 2)
    return scores


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        parsed = json.loads(text[start:end])
        return parsed if isinstance(parsed, dict) else None
    except (ValueError, AttributeError):
        return None
