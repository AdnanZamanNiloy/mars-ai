"""Temporal integrity gate: hard validation against the system clock.

The live failure this exists for: reports citing "2026 events" as settled fact
when the system clock was in 2026 but the source was a model hallucination or a
speculative projection page. A research assistant that cannot tell the present
from the future is not decision-grade.

Two independent checks, both deterministic and LLM-free:

* FUTURE-DATED EVIDENCE. A claim whose own text asserts a dated event, or whose
  source's publish date, is AFTER the system clock is either rejected or flagged
  `temporal_violation`. A future *metadata* date is a provider quirk (see the
  freshness engine); a future *asserted event* is a fabricated or speculative
  claim and must never be reported as fact.

* PUBLICATION ORDER. A source cannot be published after the event it describes.
  When a claim names a year/date and the source's publish date is later than a
  tolerance window, the claim is downstream reporting of a projection, not
  primary evidence, and is flagged `temporal_projection` (a softer signal — the
  claim may still be usable, but not as an observed fact).

Everything is additive and total: unparseable dates mean "unknown", never a
violation. The gate returns annotations; callers decide whether to drop or
merely label (the confirence engine lowers the band, the report surfaces the
label).
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from app.agents.evidence_utils import parse_published_date
from app.core.logging import get_logger

logger = get_logger(__name__)

# A future event asserted by a claim is only tolerated within this window
# (publication lag, timezone skew, "as of next quarter" legitimate phrasing).
FUTURE_TOLERANCE_DAYS = 30

# A source published this many days before the event it describes is treated as
# a projection rather than observed reporting. Real primary reporting can trail
# an event by months; this is a forward-leak detector, not a staleness rule.
PROJECTION_LEAD_DAYS = 1

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# "in 2027", "by 2030", "as of Q3 2026", "in March 2027", "of 2026"
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_QUARTER_RE = re.compile(r"\bQ([1-4])\s*(20\d{2})\b", re.IGNORECASE)
_MONTH_YEAR_RE = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\.?\s+(20\d{2})\b", re.IGNORECASE
)

# Vocabulary that means "this is a projection, not an observed event". A future
# year introduced by these is EXPECTED (a forecast) and must be labelled as
# such rather than rejected.
_PROJECTION_CUES = (
    "will", "expected", "forecast", "projected", "projection", "planned",
    "scheduled", "target", "outlook", "predict", "anticipate", "by 20",
    "estimate", "guidance", "roadmap", "aims to", "intends to",
)


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _parse_year_month(year: int, month: int = 1) -> date:
    try:
        return date(year, month, 1)
    except ValueError:
        return date(year, 1, 1)


def claim_event_date(claim: str) -> Optional[date]:
    """Best-effort date an assertive claim refers to.

    Returns None when the claim names no date. Prefers the most specific form
    present (quarter/month-year over bare year). A bare year in a projection
    phrase is still returned — the caller checks the cue separately.
    """
    text = str(claim or "")
    if not text.strip():
        return None
    q = _QUARTER_RE.search(text)
    if q:
        return _parse_year_month(int(q.group(2)), (int(q.group(1)) - 1) * 3 + 1)
    my = _MONTH_YEAR_RE.search(text)
    if my:
        return _parse_year_month(int(my.group(2)), _MONTHS[my.group(1).lower()])
    y = _YEAR_RE.search(text)
    if y:
        return _parse_year_month(int(y.group(1)))
    return None


def is_projection_phrasing(claim: str) -> bool:
    """True when the claim's date is introduced as a forecast/plan/outlook.

    Such a claim about a future date is legitimate (it is a prediction), but it
    must be labelled a projection, never reported as an observed fact.
    """
    text = str(claim or "").lower()
    return any(cue in text for cue in _PROJECTION_CUES)


def validate_temporal(
    claim: str,
    published_at: str = "",
    *,
    today: Optional[date] = None,
) -> Dict[str, Any]:
    """Validate one claim+source against the system clock.

    Returns a dict with:
      temporal_violation  bool  — future ASSERTED event, not a projection
      temporal_projection bool  — future date legitimately framed as a forecast
      temporal_reason     str
      event_date          str   — ISO date if one was found, else ""
      published_date      str   — ISO date if parseable, else ""
    """
    ref = today or _today()
    pub_iso = parse_published_date(published_at)
    event = claim_event_date(claim)
    result: Dict[str, Any] = {
        "temporal_violation": False,
        "temporal_projection": False,
        "temporal_reason": "",
        "event_date": event.isoformat() if event else "",
        "published_date": pub_iso,
    }
    if event is None:
        return result

    future = (event - ref).days > FUTURE_TOLERANCE_DAYS
    if future:
        if is_projection_phrasing(claim):
            result["temporal_projection"] = True
            result["temporal_reason"] = (
                f"future date {event.isoformat()} framed as a projection, "
                "not an observed event"
            )
        else:
            result["temporal_violation"] = True
            result["temporal_reason"] = (
                f"asserts a future-dated event ({event.isoformat()}) relative to "
                f"the system clock ({ref.isoformat()}) without projection framing"
            )
        return result

    # Publication-order check: a source cannot report as observed fact an event
    # dated before the source existed. Only meaningful when both dates exist.
    if pub_iso:
        try:
            pub = date.fromisoformat(pub_iso)
        except (TypeError, ValueError):
            return result
        if (event - pub).days > PROJECTION_LEAD_DAYS:
            result["temporal_projection"] = True
            result["temporal_reason"] = (
                f"source published {pub.isoformat()} predates the "
                f"{event.isoformat()} event it describes — projection, not report"
            )
    return result


def apply_temporal_gate(
    facts: List[Dict[str, Any]],
    *,
    today: Optional[date] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Annotate every fact with the temporal verdict; drop hard violations.

    Violations (future asserted events) are removed from the pool: they are
    fabricated or speculative and must never be synthesized as fact. Projections
    are KEPT and annotated so the report can label them. Returns the surviving
    facts and a count summary.
    """
    ref = today or _today()
    kept: List[Dict[str, Any]] = []
    summary = {"checked": 0, "violations": 0, "projections": 0, "dropped": 0}
    for fact in facts or []:
        if not isinstance(fact, dict):
            continue
        verdict = validate_temporal(
            str(fact.get("claim", "")),
            str(fact.get("published_at", "") or ""),
            today=ref,
        )
        summary["checked"] += 1
        if verdict["temporal_violation"]:
            summary["violations"] += 1
            summary["dropped"] += 1
            continue
        if verdict["temporal_projection"]:
            summary["projections"] += 1
        out = dict(fact)
        out.update(verdict)
        kept.append(out)
    if summary["violations"]:
        logger.info(
            "[Temporal] dropped %d future-dated claim(s); %d projection(s) retained",
            summary["violations"], summary["projections"],
        )
    return kept, summary
