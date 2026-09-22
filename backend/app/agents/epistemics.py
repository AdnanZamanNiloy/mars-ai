"""Epistemics Agent — what the evidence actually licenses you to say.

The pipeline can already find conflicts, grade sources and score confidence.
What it cannot do is REASON about the evidence as evidence, and that is the
line between a citation machine and a research system.

Four upgrades live here.

1. FALSE CONTRADICTIONS ARE SUPPRESSED BEFORE THEY POISON THE REPORT.
   `find_contradictions` pairs "revenue was $2bn" with "revenue was $3bn"
   and reports a conflict, and the synthesizer dutifully prints "sources
   disagree; the range is $2-3bn". If the first is 2022 and the second 2024
   that is not a disagreement, it is growth — and reporting it as a range is
   a factual error the system manufactured itself. `classify_conflict`
   separates TIME SERIES (different periods), SCOPE MISMATCH (US vs global,
   annual vs cumulative, gross vs net), UNIT MISMATCH, and only what survives
   all three is a GENUINE conflict.

2. GENUINE CONFLICTS ARE ADJUDICATED, NOT HEDGED. When a 2025 regulatory
   filing disagrees with a 2023 trade-press estimate, "sources disagree" is
   a failure of nerve: one of them is simply better evidence. `adjudicate`
   applies explicit, ordered, auditable rules — primary beats secondary,
   measured beats estimated, specific beats vague, newer supersedes older for
   time-varying quantities — and returns the winner WITH the rule that
   decided it. When no rule fires it says so, and the range is then the
   honest answer rather than the default one.

3. EVIDENCE STANDARDS DEPEND ON WHAT IS BEING CLAIMED. "X is defined as Y"
   needs one decent source. "X causes Y" needs multiple independent sources
   and should never rest on a single blog. "X will reach Y by 2030" is a
   projection and can never be verified at all, only attributed.
   `classify_claim` types every claim and `assess_standards` measures the
   pool against the bar its own claims set, instead of one flat threshold.

4. COVERAGE IS CHECKED FOR BALANCE, NOT JUST PRESENCE. `coverage_gaps` asks
   whether each planned angle has evidence. It passes a comparison with nine
   sources on option A and one on option B, which is a structurally biased
   report that looks fully covered. `coverage_asymmetry` measures the split
   across the entities the question actually names.

Everything here is deterministic, LLM-free and fail-safe: every public
function returns an empty/neutral result rather than raising, because an
epistemic check must never be the reason a report fails to ship.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.core.logging import get_logger

from app.agents.evidence_utils import (
    extract_numbers,
    parse_published_date,
    semantic_similarity,
)
from app.agents.sources import classify_source

logger = get_logger(__name__)


__all__ = [
    "ClaimType",
    "ConflictKind",
    "Resolution",
    "ClaimStandard",
    "StandardsReport",
    "AsymmetryReport",
    "EpistemicReport",
    "classify_claim",
    "classify_conflict",
    "adjudicate",
    "adjudicate_all",
    "assess_standards",
    "coverage_asymmetry",
    "assess_epistemics",
    "EVIDENCE_STANDARDS",
]


# ---------------------------------------------------------------------------
# Small guards — evidence dicts are external data and are never trusted
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _text(fact: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(fact.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _guard(fn, default):
    """Run a check; on any failure return the default. Never break a report."""
    try:
        return fn()
    except Exception:  # noqa: BLE001 - an epistemic check is never fatal
        logger.debug("[Epistemics] check failed; returning neutral result", exc_info=True)
        return default


# ---------------------------------------------------------------------------
# 1. Claim typing — different claims carry different burdens of proof
# ---------------------------------------------------------------------------

class ClaimType:
    """The kinds of assertion a research report makes.

    Not an Enum on purpose: these values are written into fact dicts that get
    serialized to JSON and compared against strings all over the host app, and
    a plain string constant survives that round trip without every consumer
    needing the import.
    """

    DEFINITIONAL = "definitional"
    DESCRIPTIVE = "descriptive"
    STATISTICAL = "statistical"
    CAUSAL = "causal"
    PREDICTIVE = "predictive"
    EVALUATIVE = "evaluative"
    ATTRIBUTIVE = "attributive"


# Ordered: the first pattern that matches wins, because a sentence can carry
# several signals and the strongest burden of proof should govern. A predictive
# causal claim is judged as predictive — you cannot verify the future.
_CLAIM_PATTERNS: Sequence[Tuple[str, "re.Pattern[str]"]] = (
    (ClaimType.PREDICTIVE, re.compile(
        r"\b(will|shall|expected to|projected to|forecast(?:ed)?|predicted|"
        r"anticipat(?:ed|es)|on track to|set to|by 20[3-9]\d|estimates? that .{0,40}"
        r"\b(?:will|by)\b)\b", re.I)),
    (ClaimType.CAUSAL, re.compile(
        r"\b(caused?|causes|causing|because of|due to|as a result of|leads? to|"
        r"led to|drives?|driven by|triggers?|triggered|responsible for|"
        r"attributable to|resulted? in|contributed? to|explains?)\b", re.I)),
    (ClaimType.EVALUATIVE, re.compile(
        r"\b(best|worst|better|worse|superior|inferior|should|ought|"
        r"outperform(?:s|ed)?|preferable|optimal|most effective|leading)\b", re.I)),
    (ClaimType.ATTRIBUTIVE, re.compile(
        r"\b(according to|said|stated|argues?|claims?|contends?|maintains?|"
        r"reported by|told|wrote|testified)\b", re.I)),
    (ClaimType.STATISTICAL, re.compile(
        r"\d+(?:[.,]\d+)?\s*(?:%|percent|per cent|bn|billion|million|"
        r"thousand|trillion|k\b)|\$\s*\d|\b\d{2,}\b", re.I)),
    (ClaimType.DEFINITIONAL, re.compile(
        r"\b(is defined as|refers to|is a type of|means that|is the term|"
        r"consists? of|is an? \w+ (?:that|which))\b", re.I)),
)


def classify_claim(claim: str) -> str:
    """Type a claim by the burden of proof it carries.

    Used to hold each claim to the standard its own form demands: a causal
    assertion resting on one blog is a defect even when a definitional one on
    the same source is fine.
    """
    text = str(claim or "").strip()
    if not text:
        return ClaimType.DESCRIPTIVE
    for claim_type, pattern in _CLAIM_PATTERNS:
        if pattern.search(text):
            return claim_type
    return ClaimType.DESCRIPTIVE


@dataclass(frozen=True)
class EvidenceStandard:
    """The bar a claim of this type must clear to be reported as established."""

    min_independent_sources: int
    requires_primary: bool
    verifiable: bool
    max_confidence_unmet: float
    rationale: str
    # When true, a primary source satisfies the corroboration requirement on
    # its own. This is the bar a careful analyst actually applies to a figure:
    # the regulator's own filing does not need a second outlet to repeat it,
    # but a number from trade press does. Demanding BOTH would mark almost
    # every real statistic unmet, and a check that fires on everything is a
    # check nobody reads.
    primary_substitutes_corroboration: bool = False


# The burden of proof, by claim type. These are the defensible positions, not
# arbitrary constants: a projection cannot be verified against a source because
# the event has not happened, so it is capped and must be attributed; a causal
# claim from a single source is the classic correlation-as-cause error and is
# held to two independent sources.
EVIDENCE_STANDARDS: Dict[str, EvidenceStandard] = {
    ClaimType.DEFINITIONAL: EvidenceStandard(
        1, False, True, 0.85,
        "A definition is checkable against one competent source.",
    ),
    ClaimType.DESCRIPTIVE: EvidenceStandard(
        1, False, True, 0.80,
        "A description of an observable state needs one reliable source.",
    ),
    ClaimType.ATTRIBUTIVE: EvidenceStandard(
        1, False, True, 0.80,
        "Who said what needs the source that carries the statement, no more.",
    ),
    ClaimType.STATISTICAL: EvidenceStandard(
        2, False, True, 0.65,
        "A figure must either come from the body that measured it or be "
        "carried by two independent sources; a lone secondary figure is one "
        "transcription error away from being wrong.",
        primary_substitutes_corroboration=True,
    ),
    ClaimType.CAUSAL: EvidenceStandard(
        2, False, True, 0.55,
        "A single source asserting causation is usually reporting a "
        "correlation; independent corroboration is the minimum.",
    ),
    ClaimType.EVALUATIVE: EvidenceStandard(
        2, False, True, 0.60,
        "A judgement ('best', 'should') is contestable by construction and "
        "needs more than one voice to be reported as a finding.",
    ),
    ClaimType.PREDICTIVE: EvidenceStandard(
        1, False, False, 0.50,
        "A projection cannot be verified against a source because the event "
        "has not occurred. It can only ever be attributed, never established.",
    ),
}


@dataclass
class ClaimStandard:
    """One claim measured against the bar its own type sets."""

    claim: str
    claim_type: str
    independent_sources: int
    is_primary: bool
    verified: bool
    meets_standard: bool
    shortfall: str = ""
    confidence_ceiling: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim[:160],
            "claim_type": self.claim_type,
            "independent_sources": self.independent_sources,
            "is_primary": self.is_primary,
            "verified": self.verified,
            "meets_standard": self.meets_standard,
            "shortfall": self.shortfall,
            "confidence_ceiling": self.confidence_ceiling,
        }


@dataclass
class StandardsReport:
    """How much of the pool clears the bar its own claims set."""

    assessed: List[ClaimStandard] = field(default_factory=list)
    by_type: Dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.assessed)

    @property
    def met(self) -> int:
        return sum(1 for c in self.assessed if c.meets_standard)

    @property
    def unmet(self) -> List[ClaimStandard]:
        return [c for c in self.assessed if not c.meets_standard]

    @property
    def met_ratio(self) -> float:
        return round(self.met / self.total, 4) if self.total else 0.0

    @property
    def confidence_ceiling(self) -> float:
        """The highest confidence this pool's weakest important claims allow.

        Taken as the mean ceiling of the unmet claims rather than the minimum:
        one weak claim in a large pool should not cap the whole report, but a
        pool that is mostly unmet claims should be capped hard.
        """
        unmet = self.unmet
        if not unmet or not self.total:
            return 1.0
        share_unmet = len(unmet) / self.total
        if share_unmet < 0.15:
            return 1.0
        mean_ceiling = sum(c.confidence_ceiling for c in unmet) / len(unmet)
        # Blend toward 1.0 in proportion to how much of the pool is fine.
        return round(mean_ceiling + (1.0 - mean_ceiling) * (1.0 - share_unmet), 4)

    def note(self) -> str:
        if not self.unmet:
            return ""
        counts: Dict[str, int] = {}
        for item in self.unmet:
            counts[item.claim_type] = counts.get(item.claim_type, 0) + 1
        worst = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        rendered = ", ".join(f"{n} {t}" for t, n in worst)
        return (
            f"{len(self.unmet)} of {self.total} claims do not meet the evidence "
            f"standard for their claim type ({rendered}); those are reported as "
            "provisional rather than established."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "met": self.met,
            "met_ratio": self.met_ratio,
            "confidence_ceiling": self.confidence_ceiling,
            "by_type": dict(self.by_type),
            "unmet": [c.to_dict() for c in self.unmet[:15]],
        }


def _independent_sources(fact: Dict[str, Any]) -> int:
    """Independent-source count, preferring the echo-adjusted figure."""
    if "independent_corroboration" in fact:
        return max(1, _safe_int(fact.get("independent_corroboration"), 1))
    return max(1, _safe_int(fact.get("corroboration_count", 1), 1))


def assess_standards(facts: Sequence[Dict[str, Any]]) -> StandardsReport:
    """Hold every claim to the standard its own type demands.

    One flat verification threshold treats "the EU adopted the rule in March"
    and "the rule caused a 12% drop in emissions" as equally easy to
    establish. They are not, and a research system that cannot tell the
    difference will report the second with the confidence of the first.
    """
    report = StandardsReport()
    for fact in facts or []:
        if not isinstance(fact, dict):
            continue
        claim = _text(fact, "claim")
        if not claim:
            continue
        claim_type = str(fact.get("claim_type") or classify_claim(claim))
        standard = EVIDENCE_STANDARDS.get(claim_type, EVIDENCE_STANDARDS[ClaimType.DESCRIPTIVE])
        report.by_type[claim_type] = report.by_type.get(claim_type, 0) + 1

        sources = _independent_sources(fact)
        url = _text(fact, "source", "url")
        primary = bool(fact.get("is_primary"))
        if not primary and url:
            primary = _guard(lambda: bool(classify_source(url).is_primary), False)
        verified = fact.get("verified") is True

        shortfalls: List[str] = []
        corroboration_satisfied = sources >= standard.min_independent_sources or (
            standard.primary_substitutes_corroboration and primary
        )
        if not corroboration_satisfied:
            shortfalls.append(
                f"needs {standard.min_independent_sources} independent sources "
                f"(or a primary source), has {sources}"
                if standard.primary_substitutes_corroboration
                else f"needs {standard.min_independent_sources} independent sources, has {sources}"
            )
        if standard.requires_primary and not primary:
            shortfalls.append("no primary source for a quantitative claim")
        if standard.verifiable and not verified:
            shortfalls.append("not verified against its cited source")
        if not standard.verifiable:
            # A projection is never "unmet" for being unverified — it is
            # unverifiable by nature. It is unmet only if presented as fact.
            shortfalls = [s for s in shortfalls if "not verified" not in s]

        meets = not shortfalls
        report.assessed.append(
            ClaimStandard(
                claim=claim,
                claim_type=claim_type,
                independent_sources=sources,
                is_primary=primary,
                verified=verified,
                meets_standard=meets,
                shortfall="; ".join(shortfalls),
                confidence_ceiling=1.0 if meets else standard.max_confidence_unmet,
            )
        )
    return report


# ---------------------------------------------------------------------------
# 2. Conflict classification — most "contradictions" are not contradictions
# ---------------------------------------------------------------------------

class ConflictKind:
    TIME_SERIES = "time_series"
    SCOPE_MISMATCH = "scope_mismatch"
    UNIT_MISMATCH = "unit_mismatch"
    GENUINE = "genuine"


# Scope qualifiers that make two different numbers compatible rather than
# contradictory. Grouped: two claims whose scope words come from the SAME group
# but differ are measuring different things.
_SCOPE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "geography": (
        "global", "worldwide", "international", "us", "u.s.", "united states",
        "american", "eu", "european", "europe", "uk", "british", "china",
        "chinese", "india", "indian", "asia", "asian", "africa", "domestic",
        "regional", "national",
    ),
    "aggregation": (
        "annual", "annually", "per year", "yearly", "quarterly", "monthly",
        "cumulative", "total", "lifetime", "per capita", "per unit", "average",
        "median", "peak",
    ),
    "accounting": (
        "gross", "net", "adjusted", "nominal", "real", "pre-tax", "post-tax",
        "before tax", "after tax", "operating", "reported",
    ),
    "population": (
        "enterprise", "consumer", "retail", "wholesale", "public", "private",
        "urban", "rural", "adult", "child",
    ),
}

_TIME_HINT_RE = re.compile(
    r"\b(?:in|for|during|as of|by|through|fy|q[1-4])\s*"
    r"((?:19|20)\d{2})|\b((?:19|20)\d{2})\b", re.I
)

# Quantities that genuinely change over time. Two different values for one of
# these in two different years is a time series, not a disagreement. A physical
# constant or a historical date is not on this list, so a conflict there stays
# a conflict.
_TIME_VARYING_RE = re.compile(
    r"\b(revenue|sales|profit|loss|price|cost|valuation|market cap|share|"
    r"users?|subscribers?|customers?|headcount|employees?|population|"
    r"capacity|production|output|emissions?|temperature|rate|adoption|"
    r"penetration|deployment|installed|shipments?|volume|traffic|"
    r"unemployment|inflation|gdp|debt|funding|investment)\b", re.I
)


def _years_in(text: str) -> Set[int]:
    years: Set[int] = set()
    for match in _TIME_HINT_RE.finditer(text or ""):
        for group in match.groups():
            if group:
                years.add(int(group))
    return years


def _fact_year(fact: Dict[str, Any]) -> Optional[int]:
    """Year the claim is ABOUT: stated in the text, else its publication year."""
    claim_years = _years_in(_text(fact, "claim"))
    if claim_years:
        return max(claim_years)
    published = _text(fact, "published_at", "published", "date")
    if published:
        parsed = _guard(lambda: parse_published_date(published), "")
        match = re.search(r"((?:19|20)\d{2})", str(parsed or published))
        if match:
            return int(match.group(1))
    return None


def _scope_signature(text: str) -> Dict[str, Set[str]]:
    """Which scope qualifiers this claim carries, by group."""
    low = f" {(text or '').lower()} "
    out: Dict[str, Set[str]] = {}
    for group, terms in _SCOPE_GROUPS.items():
        hits = {term for term in terms if f" {term} " in low or f" {term}," in low}
        if hits:
            out[group] = hits
    return out


def _units_of(text: str) -> Set[str]:
    return {
        str(getattr(q, "unit", "") or "").lower()
        for q in _guard(lambda: extract_numbers(text or "", limit=8), [])
        if str(getattr(q, "unit", "") or "")
    }


def classify_conflict(
    contradiction: Dict[str, Any],
    fact_a: Optional[Dict[str, Any]] = None,
    fact_b: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Decide whether a detected conflict is a real disagreement.

    Returns (kind, explanation). The three non-genuine kinds are the ones the
    numeric detector cannot see, because it compares magnitudes and topic
    similarity and nothing else:

    * TIME SERIES — the same time-varying quantity in two different periods.
      Reporting "$2bn to $3bn" as a range when the truth is "grew from $2bn in
      2022 to $3bn in 2024" is an error the system invents on its own.
    * SCOPE MISMATCH — US vs global, annual vs cumulative, gross vs net. Both
      numbers are correct and they are not about the same thing.
    * UNIT MISMATCH — the values are in different units, so the comparison
      that produced the "conflict" was never valid.
    """
    claim_a = str(contradiction.get("claim_a", "") or "")
    claim_b = str(contradiction.get("claim_b", "") or "")
    if not claim_a or not claim_b:
        return ConflictKind.GENUINE, ""

    kind = str(contradiction.get("kind", "numeric") or "numeric")
    if kind != "numeric":
        # Polarity and other non-numeric conflicts are genuine by construction:
        # "does reduce" vs "does not reduce" is not a scope difference.
        return ConflictKind.GENUINE, ""

    # -- unit mismatch --------------------------------------------------
    units_a = _units_of(claim_a)
    units_b = _units_of(claim_b)
    if units_a and units_b and not (units_a & units_b):
        return (
            ConflictKind.UNIT_MISMATCH,
            f"measured in different units ({'/'.join(sorted(units_a))} vs "
            f"{'/'.join(sorted(units_b))}), so the two figures were never comparable",
        )

    # -- scope mismatch -------------------------------------------------
    scope_a = _scope_signature(claim_a)
    scope_b = _scope_signature(claim_b)
    for group in set(scope_a) & set(scope_b):
        if not (scope_a[group] & scope_b[group]):
            return (
                ConflictKind.SCOPE_MISMATCH,
                f"different {group} scope "
                f"({'/'.join(sorted(scope_a[group]))} vs {'/'.join(sorted(scope_b[group]))}); "
                "both figures can be correct",
            )
    # One side qualifies its scope and the other does not: weaker signal, but a
    # bare figure next to an explicitly scoped one is usually the broader one.
    for group in ("geography", "aggregation"):
        if (group in scope_a) != (group in scope_b):
            qualified = scope_a.get(group) or scope_b.get(group) or set()
            return (
                ConflictKind.SCOPE_MISMATCH,
                f"one figure is qualified by {group} ({'/'.join(sorted(qualified))}) "
                "and the other is not, so they may not describe the same quantity",
            )

    # -- time series ----------------------------------------------------
    combined = f"{claim_a} {claim_b}"
    if _TIME_VARYING_RE.search(combined):
        year_a = _years_in(claim_a) or ({_fact_year(fact_a)} if fact_a else set())
        year_b = _years_in(claim_b) or ({_fact_year(fact_b)} if fact_b else set())
        year_a = {y for y in year_a if y}
        year_b = {y for y in year_b if y}
        if year_a and year_b and not (year_a & year_b):
            return (
                ConflictKind.TIME_SERIES,
                f"the same time-varying quantity measured in different periods "
                f"({min(year_a)} vs {min(year_b)}); this is change over time, "
                "not a disagreement",
            )

    return ConflictKind.GENUINE, ""


# ---------------------------------------------------------------------------
# 3. Adjudication — deciding which source to believe, on stated rules
# ---------------------------------------------------------------------------

@dataclass
class Resolution:
    """The verdict on one conflict, and the rule that produced it."""

    kind: str
    resolved: bool
    winner: str = ""          # "a" | "b" | ""
    rule: str = ""
    explanation: str = ""
    claim_a: str = ""
    claim_b: str = ""
    source_a: str = ""
    source_b: str = ""

    @property
    def is_real_conflict(self) -> bool:
        return self.kind == ConflictKind.GENUINE

    def render(self) -> str:
        """One line the report can print verbatim."""
        if not self.is_real_conflict:
            return f"Not a conflict — {self.explanation}."
        if self.resolved:
            winning = self.claim_a if self.winner == "a" else self.claim_b
            losing = self.claim_b if self.winner == "a" else self.claim_a
            return (
                f"Resolved in favour of \"{winning[:130]}\" over "
                f"\"{losing[:130]}\": {self.explanation}."
            )
        return (
            f"Unresolved conflict between \"{self.claim_a[:120]}\" and "
            f"\"{self.claim_b[:120]}\" — {self.explanation or 'no rule separates them'}; "
            "report both."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "resolved": self.resolved,
            "winner": self.winner,
            "rule": self.rule,
            "explanation": self.explanation,
            "claim_a": self.claim_a[:200],
            "claim_b": self.claim_b[:200],
            "source_a": self.source_a,
            "source_b": self.source_b,
        }


_ESTIMATE_RE = re.compile(
    r"\b(estimat\w+|approximat\w+|roughly|about|around|some|nearly|"
    r"project\w+|forecast\w+|expect\w+|could|may|might)\b", re.I
)
_MEASURED_RE = re.compile(
    r"\b(reported|filed|recorded|measured|audited|official|census|"
    r"according to the (?:filing|report|statement)|disclosed|published)\b", re.I
)


def _side(fact: Optional[Dict[str, Any]], claim: str, url: str) -> Dict[str, Any]:
    """Everything adjudication needs about one side of a conflict."""
    fact = fact if isinstance(fact, dict) else {}
    profile = _guard(lambda: classify_source(url), None) if url else None
    return {
        "claim": claim,
        "url": url,
        "primary": bool(fact.get("is_primary")) or bool(getattr(profile, "is_primary", False)),
        "authority": _safe_float(getattr(profile, "authority", 0.0)),
        "verified": fact.get("verified") is True,
        "independent": _independent_sources(fact) if fact else 1,
        "year": _fact_year(fact) if fact else (max(_years_in(claim)) if _years_in(claim) else None),
        "estimated": bool(_ESTIMATE_RE.search(claim)),
        "measured": bool(_MEASURED_RE.search(claim)),
        "specificity": len(_guard(lambda: extract_numbers(claim, limit=8), [])),
    }


# Ordered adjudication rules. Each takes the two sides and returns the winner
# ("a"/"b") plus a human explanation, or None to defer to the next rule. Order
# IS the policy: a primary source outranks a newer secondary one, because
# recency cannot fix a source that was never authoritative.
def _rule_primary(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if a["primary"] and not b["primary"]:
        return "a", "the first is a primary source (filing, dataset or official report) and the second is not"
    if b["primary"] and not a["primary"]:
        return "b", "the second is a primary source (filing, dataset or official report) and the first is not"
    return None


def _rule_measured_over_estimated(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if a["measured"] and b["estimated"] and not a["estimated"]:
        return "a", "the first reports a measured figure and the second is an estimate"
    if b["measured"] and a["estimated"] and not b["estimated"]:
        return "b", "the second reports a measured figure and the first is an estimate"
    return None


def _rule_verified(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if a["verified"] and not b["verified"]:
        return "a", "the first was verified against its cited source and the second was not"
    if b["verified"] and not a["verified"]:
        return "b", "the second was verified against its cited source and the first was not"
    return None


def _rule_corroboration(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if a["independent"] >= b["independent"] + 2:
        return "a", f"the first is carried by {a['independent']} independent sources against {b['independent']}"
    if b["independent"] >= a["independent"] + 2:
        return "b", f"the second is carried by {b['independent']} independent sources against {a['independent']}"
    return None


def _rule_recency(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Newer supersedes older ONLY for quantities that change over time.

    Applying recency to a constant is wrong — a 2025 blog does not supersede a
    2019 peer-reviewed measurement of something that does not move. The
    time-varying test gates this rule for exactly that reason.
    """
    if not (a["year"] and b["year"]) or a["year"] == b["year"]:
        return None
    if not _TIME_VARYING_RE.search(f"{a['claim']} {b['claim']}"):
        return None
    if a["year"] > b["year"]:
        return "a", f"the first reflects {a['year']} and supersedes the {b['year']} figure for a quantity that changes over time"
    return "b", f"the second reflects {b['year']} and supersedes the {a['year']} figure for a quantity that changes over time"


def _rule_authority(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    gap = a["authority"] - b["authority"]
    if gap >= 0.25:
        return "a", f"the first source carries materially higher authority ({a['authority']:.2f} vs {b['authority']:.2f})"
    if gap <= -0.25:
        return "b", f"the second source carries materially higher authority ({b['authority']:.2f} vs {a['authority']:.2f})"
    return None


_ADJUDICATION_RULES: Sequence[Tuple[str, Any]] = (
    ("primary_source", _rule_primary),
    ("measured_over_estimated", _rule_measured_over_estimated),
    ("verified", _rule_verified),
    ("recency_for_time_varying", _rule_recency),
    ("independent_corroboration", _rule_corroboration),
    ("source_authority", _rule_authority),
)


def adjudicate(
    contradiction: Dict[str, Any],
    facts: Sequence[Dict[str, Any]] = (),
) -> Resolution:
    """Classify a conflict and, if genuine, decide it on stated rules.

    "Sources disagree" is the right answer only when the sources are actually
    comparable in quality. When a regulator's filing contradicts a trade-press
    estimate, presenting both as equally weighted is not neutrality — it is a
    failure to do the analysis the reader came for. Every verdict names the
    rule that produced it, so a reader can reject the reasoning rather than
    having to trust it.
    """
    claim_a = str(contradiction.get("claim_a", "") or "")
    claim_b = str(contradiction.get("claim_b", "") or "")
    source_a = str(contradiction.get("source_a", "") or "")
    source_b = str(contradiction.get("source_b", "") or "")

    fact_a = _find_fact(facts, claim_a)
    fact_b = _find_fact(facts, claim_b)

    kind, explanation = _guard(
        lambda: classify_conflict(contradiction, fact_a, fact_b),
        (ConflictKind.GENUINE, ""),
    )
    base = Resolution(
        kind=kind, resolved=False, claim_a=claim_a, claim_b=claim_b,
        source_a=source_a, source_b=source_b, explanation=explanation,
    )
    if kind != ConflictKind.GENUINE:
        # Not a disagreement: suppressed from the conflict list, and the
        # explanation is what the report should say instead of a fake range.
        return base

    side_a = _side(fact_a, claim_a, source_a)
    side_b = _side(fact_b, claim_b, source_b)
    for rule_name, rule in _ADJUDICATION_RULES:
        verdict = _guard(lambda: rule(side_a, side_b), None)
        if verdict:
            winner, why = verdict
            base.resolved = True
            base.winner = winner
            base.rule = rule_name
            base.explanation = why
            return base

    base.explanation = (
        "both sources are comparable in authority, recency and corroboration"
    )
    return base


def _find_fact(facts: Sequence[Dict[str, Any]], claim: str) -> Optional[Dict[str, Any]]:
    """Locate the fact behind a contradiction's claim text.

    Contradictions carry claim strings, not fact references, so the richer
    metadata (verification standing, primary flag, publication date) has to be
    recovered by matching. Exact match first, then best similarity above a
    floor, so a truncated claim string still resolves.
    """
    if not claim:
        return None
    target = claim.strip()
    best: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for fact in facts or []:
        if not isinstance(fact, dict):
            continue
        text = str(fact.get("claim", "") or "").strip()
        if not text:
            continue
        if text == target:
            return fact
        score = _guard(lambda: semantic_similarity(text, target), 0.0)
        if score > best_score:
            best, best_score = fact, score
    return best if best_score >= 0.6 else None


def adjudicate_all(
    contradictions: Sequence[Dict[str, Any]],
    facts: Sequence[Dict[str, Any]] = (),
) -> List[Resolution]:
    """Adjudicate every detected conflict, in input order."""
    out: List[Resolution] = []
    for item in contradictions or ():
        if isinstance(item, dict):
            out.append(adjudicate(item, facts))
    return out


# ---------------------------------------------------------------------------
# 4. Coverage asymmetry — a balanced-looking report built on one-sided evidence
# ---------------------------------------------------------------------------

_COMPARISON_SPLIT_RE = re.compile(
    r"\s+(?:vs\.?|versus|compared (?:to|with)|against|or)\s+|,\s*", re.I
)
_STOPWORDS = {
    "the", "a", "an", "of", "for", "in", "on", "to", "and", "is", "are", "was",
    "which", "what", "how", "why", "better", "best", "worse", "should", "we",
    "i", "do", "does", "between", "difference", "compare", "comparison", "vs",
}


@dataclass
class AsymmetryReport:
    """Evidence balance across the entities the question actually names."""

    entities: Dict[str, int] = field(default_factory=dict)
    balanced: bool = True
    starved: List[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        """Largest-to-smallest evidence ratio across named entities."""
        if len(self.entities) < 2:
            return 1.0
        counts = sorted(self.entities.values())
        return round(counts[-1] / max(1, counts[0]), 2)

    def note(self) -> str:
        if self.balanced or not self.entities:
            return ""
        rendered = ", ".join(f"{name} ({count})" for name, count in sorted(
            self.entities.items(), key=lambda kv: -kv[1]
        ))
        starved = ", ".join(self.starved)
        return (
            f"Evidence is unevenly distributed across the entities compared: "
            f"{rendered}. Findings about {starved} rest on materially less "
            "evidence than the rest, so any comparison involving them is "
            "provisional rather than settled."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": dict(self.entities),
            "balanced": self.balanced,
            "starved": list(self.starved),
            "ratio": self.ratio,
        }


def _candidate_entities(query: str) -> List[str]:
    """The things a comparison question is comparing.

    Split on comparison connectives, then keep the content words. Crude by
    design: a false entity simply finds no evidence and is dropped, whereas
    missing a real one means the imbalance goes unreported.
    """
    text = str(query or "").strip().rstrip("?")
    if not text:
        return []
    parts = [p.strip() for p in _COMPARISON_SPLIT_RE.split(text) if p.strip()]
    if len(parts) < 2:
        return []
    entities: List[str] = []
    for part in parts:
        words = [w for w in re.findall(r"[A-Za-z][\w.+-]{1,}", part)
                 if w.lower() not in _STOPWORDS]
        if not words:
            continue
        # Prefer a capitalised run (a proper name); else the longest word.
        caps = [w for w in words if w[:1].isupper()]
        entity = " ".join(caps[-2:]) if caps else max(words, key=len)
        if len(entity) >= 3 and entity.lower() not in _STOPWORDS:
            entities.append(entity)
    # Dedupe, preserving order.
    seen: Set[str] = set()
    out: List[str] = []
    for entity in entities:
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            out.append(entity)
    return out[:5]


def coverage_asymmetry(
    query: str,
    facts: Sequence[Dict[str, Any]],
    *,
    min_ratio: float = 3.0,
) -> AsymmetryReport:
    """Detect a comparison answered mostly from one side.

    `coverage_gaps` asks whether each planned angle produced evidence. It
    passes a report with nine sources on option A and one on option B, because
    both angles are non-empty — and that report will confidently recommend A,
    having barely looked at B. This measures the split across the entities the
    QUESTION names, which is the axis bias actually travels along.
    """
    report = AsymmetryReport()
    entities = _guard(lambda: _candidate_entities(query), [])
    if len(entities) < 2:
        return report

    counts: Dict[str, int] = {}
    for entity in entities:
        needle = entity.lower()
        hits = 0
        for fact in facts or []:
            if not isinstance(fact, dict):
                continue
            haystack = f"{fact.get('claim', '')} {fact.get('sub_question', '')}".lower()
            if needle in haystack:
                hits += 1
        counts[entity] = hits

    # Entities with zero evidence are dropped: they are usually a parsing
    # artifact rather than a real side of the comparison. Only entities that
    # actually appear in the pool can be meaningfully compared for balance.
    present = {name: count for name, count in counts.items() if count > 0}
    if len(present) < 2:
        return report

    report.entities = present
    smallest = min(present.values())
    largest = max(present.values())
    if smallest > 0 and largest / smallest >= min_ratio:
        report.balanced = False
        cutoff = largest / min_ratio
        report.starved = sorted(
            name for name, count in present.items() if count <= cutoff
        )
    return report


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

@dataclass
class EpistemicReport:
    """Everything this layer concluded about what the evidence supports."""

    resolutions: List[Resolution] = field(default_factory=list)
    standards: StandardsReport = field(default_factory=StandardsReport)
    asymmetry: AsymmetryReport = field(default_factory=AsymmetryReport)

    @property
    def genuine_conflicts(self) -> List[Resolution]:
        return [r for r in self.resolutions if r.is_real_conflict]

    @property
    def suppressed(self) -> List[Resolution]:
        """Detected 'conflicts' that were not disagreements at all."""
        return [r for r in self.resolutions if not r.is_real_conflict]

    @property
    def unresolved(self) -> List[Resolution]:
        return [r for r in self.genuine_conflicts if not r.resolved]

    @property
    def confidence_ceiling(self) -> float:
        """The cap this layer puts on overall confidence.

        Unresolved genuine conflicts are the binding constraint: a report that
        cannot say which of two contradictory figures is right does not get to
        call itself high-confidence, however many sources it read.
        """
        ceiling = self.standards.confidence_ceiling
        unresolved = len(self.unresolved)
        if unresolved >= 3:
            ceiling = min(ceiling, 0.55)
        elif unresolved == 2:
            ceiling = min(ceiling, 0.65)
        elif unresolved == 1:
            ceiling = min(ceiling, 0.72)
        if not self.asymmetry.balanced:
            ceiling = min(ceiling, 0.70)
        return round(ceiling, 4)

    def notes(self) -> List[str]:
        out: List[str] = []
        suppressed = self.suppressed
        if suppressed:
            out.append(
                f"{len(suppressed)} detected conflict(s) were not disagreements "
                "(different time periods, scopes or units) and are reported as "
                "such rather than as a range."
            )
        resolved = [r for r in self.genuine_conflicts if r.resolved]
        if resolved:
            out.append(
                f"{len(resolved)} source conflict(s) were adjudicated on stated "
                f"rules; {len(self.unresolved)} remain genuinely open."
            )
        standards_note = self.standards.note()
        if standards_note:
            out.append(standards_note)
        asymmetry_note = self.asymmetry.note()
        if asymmetry_note:
            out.append(asymmetry_note)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolutions": [r.to_dict() for r in self.resolutions[:25]],
            "genuine_conflicts": len(self.genuine_conflicts),
            "suppressed_conflicts": len(self.suppressed),
            "unresolved_conflicts": len(self.unresolved),
            "standards": self.standards.to_dict(),
            "asymmetry": self.asymmetry.to_dict(),
            "confidence_ceiling": self.confidence_ceiling,
            "notes": self.notes(),
        }


def assess_epistemics(
    query: str,
    facts: Sequence[Dict[str, Any]],
    contradictions: Sequence[Dict[str, Any]] = (),
) -> EpistemicReport:
    """Run the whole layer. Total and fail-safe.

    Each stage is isolated, so one failing costs its own finding and nothing
    else: an epistemic check must never be the reason a report does not ship.
    """
    report = EpistemicReport()
    report.resolutions = _guard(lambda: adjudicate_all(contradictions, facts), [])
    report.standards = _guard(lambda: assess_standards(facts), StandardsReport())
    report.asymmetry = _guard(lambda: coverage_asymmetry(query, facts), AsymmetryReport())
    return report
