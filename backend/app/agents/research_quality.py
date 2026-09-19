"""Research-quality layer — the checks that separate a cited report from a
defensible one.

The synthesizer already guarantees that every claim carries a marker and that
every number appears somewhere in the evidence. That is table stakes. It does
not catch the six failures that actually discredit a research product in front
of an expert reader:

1.  MISATTRIBUTION. A sentence cites [3] and states 47%, but [3] contains no
    47% — the figure came from [7]. The pool-wide grounding check passes it,
    because the number does exist somewhere. `check_citation_grounding` binds
    each figure to the source cited BESIDE it.

2.  UNTRACEABLE NON-NUMERIC CLAIMS. "Acme acquired Beta in a cash deal" has no
    digits, so a digits-only factual test never flags it when it carries no
    marker. `is_factual_sentence` adds proper-noun, quotation and date-word
    detection, which is where most uncited assertions actually live.

3.  OVERCLAIMING. A D-grade, single-source, unverified claim stated flatly as
    fact reads identically to an A-grade corroborated one. `detect_overclaims`
    finds assertive sentences whose cited evidence cannot carry that weight.

4.  STALENESS. "What is the current state of X" answered from 2021 sources,
    with no date anywhere in the report, is the most common real-world failure
    of research assistants. `temporal_profile` measures the evidence's date
    range and flags it against the question's time sensitivity.

5.  ECHO CHAMBERS. Three outlets reprinting one wire story is not three
    sources. `assess_independence` finds near-identical claims across domains
    and reports EFFECTIVE source count, so corroboration stops being a
    headcount.

6.  SELF-CONTRADICTION. The summary says 12% and a later section says 18% for
    the same quantity. `detect_internal_conflicts` catches the report
    disagreeing with itself.

Every check here is deterministic, needs no LLM and no network, and is
fail-safe: any check that errors returns an empty finding rather than breaking
synthesis. Findings are REPORTED, never silently repaired — a system that
quietly fixes its own citations teaches its users to trust output it has not
earned.

The module also renders the writing contracts that prevent these failures up
front (calibrated estimative language, as-of dating, premise challenge), so
the prompt and the audit enforce the same standard from both ends.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.agents.evidence_utils import (
    extract_domain,
    extract_numbers,
    semantic_similarity,
    split_into_sentences,
)

__all__ = [
    "TemporalProfile",
    "IndependenceReport",
    "QualityFinding",
    "ResearchQualityReport",
    "temporal_profile",
    "assess_independence",
    "apply_independence",
    "independent_corroboration",
    "is_factual_sentence",
    "check_citation_grounding",
    "detect_overclaims",
    "detect_internal_conflicts",
    "section_coverage",
    "assess_report_quality",
    "render_quality_contract",
    "estimative_band",
]


# ---------------------------------------------------------------------------
# Shared primitives
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


# Ordinary prose numbers ("three angles", "two sources"); not worth binding to
# a source unless they carry a unit.
_TRIVIAL_NUMBERS: Set[float] = {float(n) for n in range(0, 11)}


def _significant_values(text: str, limit: int = 16) -> Set[Tuple[float, str]]:
    """(value, unit) pairs worth checking, with bare small integers dropped."""
    out: Set[Tuple[float, str]] = set()
    try:
        quantities = extract_numbers(text or "", limit=limit)
    except Exception:  # noqa: BLE001 - never let extraction break the audit
        return out
    for q in quantities:
        unit = str(getattr(q, "unit", "") or "").lower()
        value = round(_safe_float(getattr(q, "value", 0.0)), 4)
        if value in _TRIVIAL_NUMBERS and not unit:
            continue
        out.add((value, unit))
    return out


def _values_match(a: Tuple[float, str], b: Tuple[float, str], tolerance: float = 0.02) -> bool:
    """Same quantity, allowing paraphrase rounding but not a different year.

    Years and small magnitudes compare exactly: a relative tolerance would make
    2024 and 2025 the same number, which is the single most plausible
    fabrication in a research report.
    """
    value_a, unit_a = a
    value_b, unit_b = b
    if unit_a and unit_b and unit_a != unit_b:
        return False
    if value_a == value_b:
        return True
    if _is_year(value_a) or _is_year(value_b) or abs(value_a) < 20 or abs(value_b) < 20:
        return False
    scale = max(abs(value_a), abs(value_b), 1e-9)
    return abs(value_a - value_b) / scale <= tolerance


def _is_year(value: float) -> bool:
    return float(value).is_integer() and 1000.0 <= value <= 2999.0


def _now() -> datetime:
    """Naive UTC now. `_now()` is deprecated from Python 3.12."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _markers(sentence: str) -> List[int]:
    return [int(m) for m in re.findall(r"\[(\d+)\]", sentence or "")]


def _strip_markers(sentence: str) -> str:
    return re.sub(r"\[\d+\]", "", sentence or "").strip()


# ---------------------------------------------------------------------------
# Finding model
# ---------------------------------------------------------------------------

@dataclass
class QualityFinding:
    """One defect, with enough context for a reader to check it themselves."""

    kind: str
    detail: str
    sentence: str = ""
    severity: str = "warn"  # "warn" | "serious"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "sentence": self.sentence[:200],
            "severity": self.severity,
        }


# ---------------------------------------------------------------------------
# 1. Temporal profile — how old is this answer, really
# ---------------------------------------------------------------------------

_DATE_PATTERNS: Sequence["re.Pattern[str]"] = (
    re.compile(r"(\d{4})-(\d{2})-(\d{2})"),
    re.compile(r"(\d{4})/(\d{2})/(\d{2})"),
    re.compile(r"^(\d{4})-(\d{2})$"),
    re.compile(r"^(\d{4})$"),
)

# Question shapes whose answer decays. A definition does not go stale in a
# year; a "current state", a price, a forecast base rate and a decision
# recommendation all do.
_TIME_SENSITIVE_TYPES = {"status", "forecast", "timeline", "decision", "comparison"}

# Days after which evidence is called out. Time-sensitive questions get a much
# tighter bar because the reader will act on the answer as if it were current.
_STALE_DAYS_SENSITIVE = 270
_STALE_DAYS_GENERAL = 1095


def _parse_date(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    # ISO with time, the common case from most crawlers.
    try:
        cleaned = text.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(cleaned)
        return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError:
        pass
    for pattern in _DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        groups = [int(g) for g in match.groups()]
        try:
            if len(groups) == 3:
                return datetime(groups[0], groups[1], groups[2])
            if len(groups) == 2:
                return datetime(groups[0], groups[1], 1)
            return datetime(groups[0], 1, 1)
        except ValueError:
            continue
    return None


def _fact_date(fact: Dict[str, Any]) -> Optional[datetime]:
    """Publication date preferred; retrieval date is a weak upper bound only.

    Retrieval tells you when the crawler ran, not when the claim was true, so
    it is never used to make evidence look fresh — only to date a source that
    published no date at all, and such facts are counted as undated.
    """
    for key in ("published_at", "published", "date", "article_date"):
        parsed = _parse_date(fact.get(key))
        if parsed:
            return parsed
    return None


@dataclass
class TemporalProfile:
    """When this report's evidence was actually published."""

    oldest: Optional[datetime] = None
    newest: Optional[datetime] = None
    median_age_days: Optional[int] = None
    dated: int = 0
    undated: int = 0
    time_sensitive: bool = False
    stale: bool = False

    @property
    def coverage(self) -> float:
        total = self.dated + self.undated
        return round(self.dated / total, 3) if total else 0.0

    def as_of_line(self) -> str:
        """The dating line every research report should carry and most don't."""
        if not self.newest:
            return (
                "- Evidence dating: no source carried a publication date, so the "
                "currency of these findings cannot be established"
            )
        span = (
            f"{self.oldest:%b %Y} to {self.newest:%b %Y}"
            if self.oldest and self.oldest != self.newest
            else f"{self.newest:%b %Y}"
        )
        line = f"- Evidence published: {span} (most recent source {self.newest:%d %b %Y})"
        if self.undated:
            line += f"; {self.undated} source(s) carried no date"
        return line

    def warning(self) -> str:
        if not self.stale or not self.newest:
            return ""
        age_days = (_now() - self.newest).days
        months = max(1, age_days // 30)
        if self.time_sensitive:
            return (
                f"The most recent source is about {months} month(s) old, and this "
                "question asks about a current or forward-looking state. Treat "
                f"every 'current' claim as true AS OF {self.newest:%B %Y}, not "
                "today, and re-run before acting on it."
            )
        return (
            f"The evidence base is dated — the most recent source is about "
            f"{months} month(s) old. Anything that has moved since then is not "
            "reflected here."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "oldest": self.oldest.isoformat() if self.oldest else None,
            "newest": self.newest.isoformat() if self.newest else None,
            "median_age_days": self.median_age_days,
            "dated": self.dated,
            "undated": self.undated,
            "date_coverage": self.coverage,
            "time_sensitive": self.time_sensitive,
            "stale": self.stale,
        }


def temporal_profile(
    facts: Sequence[Dict[str, Any]],
    *,
    query_type: str = "",
    now: Optional[datetime] = None,
) -> TemporalProfile:
    """Measure the evidence's age and decide whether it is too old to be silent.

    Staleness is relative to the QUESTION, not an absolute cutoff: an 18-month
    old paper is perfectly good evidence for "how does attention work" and
    disqualifying for "what is the current market leader".
    """
    now = now or _now()
    profile = TemporalProfile(time_sensitive=str(query_type or "").lower() in _TIME_SENSITIVE_TYPES)
    dates: List[datetime] = []
    for fact in facts or []:
        if not isinstance(fact, dict):
            continue
        parsed = _fact_date(fact)
        if parsed and parsed <= now:
            dates.append(parsed)
        else:
            profile.undated += 1
    if not dates:
        return profile
    dates.sort()
    profile.dated = len(dates)
    profile.oldest = dates[0]
    profile.newest = dates[-1]
    ages = sorted((now - d).days for d in dates)
    profile.median_age_days = ages[len(ages) // 2]

    limit = _STALE_DAYS_SENSITIVE if profile.time_sensitive else _STALE_DAYS_GENERAL
    newest_age = (now - profile.newest).days
    profile.stale = newest_age > limit or (profile.median_age_days or 0) > limit * 2
    return profile


# ---------------------------------------------------------------------------
# 2. Source independence — corroboration is not a headcount
# ---------------------------------------------------------------------------

# Above this, two claims from different domains are the same text, which means
# syndication or a shared press release, not independent confirmation.
_ECHO_SIMILARITY = 0.82


@dataclass
class IndependenceReport:
    """How many genuinely independent voices are behind this evidence."""

    nominal_sources: int = 0
    effective_sources: int = 0
    echo_groups: List[List[str]] = field(default_factory=list)
    dominant_domain: str = ""
    dominant_share: float = 0.0

    @property
    def echoed_claims(self) -> int:
        return sum(len(g) for g in self.echo_groups)

    def warning(self) -> str:
        parts: List[str] = []
        if self.echo_groups:
            parts.append(
                f"{len(self.echo_groups)} claim(s) appear in near-identical wording "
                f"across {self.echoed_claims} sources, which indicates syndication "
                "or a shared origin rather than independent confirmation; "
                f"{self.nominal_sources} nominal sources reduce to about "
                f"{self.effective_sources} independent ones."
            )
        if self.dominant_share > 0.55 and self.dominant_domain:
            parts.append(
                f"{self.dominant_share:.0%} of the evidence comes from "
                f"{self.dominant_domain} alone; this report largely reflects one "
                "publisher's account."
            )
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nominal_sources": self.nominal_sources,
            "effective_sources": self.effective_sources,
            "echo_groups": len(self.echo_groups),
            "echoed_claims": self.echoed_claims,
            "dominant_domain": self.dominant_domain,
            "dominant_share": self.dominant_share,
        }


def assess_independence(facts: Sequence[Dict[str, Any]]) -> IndependenceReport:
    """Find claims that are the same text wearing different domain names.

    Two claims from DIFFERENT domains whose wording is near-identical are a
    wire story, a press release, or one outlet quoting another. Counting them
    as two corroborating sources is how a research system manufactures false
    confidence. Same-domain duplicates are ignored here — that is ordinary
    repetition within one publisher, and the deduper already handles it.
    """
    report = IndependenceReport()
    items = [f for f in (facts or []) if isinstance(f, dict) and str(f.get("claim", "")).strip()]
    if not items:
        return report

    domains = [extract_domain(str(f.get("source", "") or "")) or "" for f in items]
    distinct = {d for d in domains if d}
    report.nominal_sources = len(distinct)

    if distinct:
        counts: Dict[str, int] = {}
        for domain in domains:
            if domain:
                counts[domain] = counts.get(domain, 0) + 1
        report.dominant_domain, top = max(counts.items(), key=lambda kv: kv[1])
        report.dominant_share = round(top / max(1, len([d for d in domains if d])), 3)

    # Union-find over cross-domain near-duplicates.
    parent = list(range(len(items)))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if not domains[i] or not domains[j] or domains[i] == domains[j]:
                continue
            try:
                similarity = semantic_similarity(
                    str(items[i].get("claim", "")), str(items[j].get("claim", ""))
                )
            except Exception:  # noqa: BLE001
                continue
            if similarity >= _ECHO_SIMILARITY:
                _union(i, j)

    clusters: Dict[int, List[int]] = {}
    for index in range(len(items)):
        clusters.setdefault(_find(index), []).append(index)

    echoed_domains: Set[str] = set()
    for members in clusters.values():
        if len(members) < 2:
            continue
        member_domains = sorted({domains[m] for m in members if domains[m]})
        if len(member_domains) < 2:
            continue
        report.echo_groups.append(member_domains)
        echoed_domains.update(member_domains[1:])

    # Each echo group collapses to one independent voice.
    report.effective_sources = max(1, report.nominal_sources - len(echoed_domains)) if distinct else 0
    return report


def apply_independence(
    facts: Sequence[Dict[str, Any]],
    report: IndependenceReport,
) -> List[Dict[str, Any]]:
    """Stamp each fact with `independent_corroboration`, never inflating it.

    The raw `corroboration_count` is preserved so nothing downstream breaks;
    the new field is what confidence and the findings bullets should read. When
    a claim belongs to an echo group its independent count is 1, because every
    copy traces to one origin.
    """
    out: List[Dict[str, Any]] = []
    echo_domains: Set[str] = {d for group in report.echo_groups for d in group}
    for fact in facts or []:
        if not isinstance(fact, dict):
            continue
        item = dict(fact)
        nominal = max(1, _safe_int(item.get("corroboration_count", 1), 1))
        domain = extract_domain(str(item.get("source", "") or "")) or ""
        item["independent_corroboration"] = 1 if domain in echo_domains else nominal
        out.append(item)
    return out


def independent_corroboration(fact: Dict[str, Any]) -> int:
    """Independent-source count, falling back to the nominal one."""
    if "independent_corroboration" in fact:
        return max(1, _safe_int(fact.get("independent_corroboration"), 1))
    return max(1, _safe_int(fact.get("corroboration_count", 1), 1))


# ---------------------------------------------------------------------------
# 3. Factual-sentence detection beyond digits
# ---------------------------------------------------------------------------

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december"
)
_NUMERIC_HINT_RE = re.compile(r"\d|%|\bper cent\b|\bpercent\b", re.I)
_DATE_WORD_RE = re.compile(rf"\b(?:{_MONTHS})\b|\b(?:19|20)\d{{2}}\b", re.I)
_QUOTE_RE = re.compile(r"[\"“][^\"”]{8,}[\"”]")
# A capitalised token that is NOT at the start of the sentence and is not a
# common sentence-initial word: the signature of a named entity.
_PROPER_NOUN_RE = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-zA-Z]{2,}\b")
_COMMON_CAPS = {
    "The", "This", "That", "These", "Those", "There", "Their", "They", "It",
    "However", "Although", "While", "Because", "Where", "When", "What", "Which",
    "Both", "Each", "Most", "Some", "Many", "Several", "One", "Two", "Three",
    "First", "Second", "Third", "Overall", "Taken", "In", "By", "As", "For",
    "But", "And", "Its", "Not", "No", "Yes", "If", "So", "At", "On", "To",
}
# Attribution phrasing is itself a factual assertion about who said what.
_ATTRIBUTION_RE = re.compile(
    r"\b(according to|reported by|announced|stated|published|filed|said)\b", re.I
)
# Past-tense event verbs: "the regulator opened an investigation" asserts that
# a thing happened in the world and is checkable, yet carries no digit, no
# proper noun and no attribution. Reported events are where a large share of
# uncited claims live, so the verb itself is the signal.
_EVENT_VERB_RE = re.compile(
    r"\b(acquired|merged|launched|opened|closed|filed|approved|rejected|banned|"
    r"ruled|resigned|appointed|raised|cut|fell|rose|grew|declined|shrank|"
    r"agreed|settled|sued|fined|recalled|withdrew|halted|resumed|found|"
    r"reported|awarded|signed|issued)\b",
    re.I,
)


def is_factual_sentence(sentence: str) -> bool:
    """Does this sentence assert something checkable?

    The previous digits-only test let every non-numeric assertion through
    uncited — "Acme acquired Beta", "the regulator opened an investigation",
    "the study found no effect" all have no digits. Named entities, quoted
    text, dates in words and attribution verbs are added, which is where most
    untraceable claims actually live. Analysis and transition prose carries
    none of these.
    """
    text = (sentence or "").strip()
    if not text:
        return False
    if _NUMERIC_HINT_RE.search(text) or _DATE_WORD_RE.search(text):
        return True
    if _QUOTE_RE.search(text) or _ATTRIBUTION_RE.search(text):
        return True
    if _EVENT_VERB_RE.search(text):
        return True
    for match in _PROPER_NOUN_RE.finditer(text):
        if match.group(0) not in _COMMON_CAPS:
            return True
    return False


# ---------------------------------------------------------------------------
# 4. Per-citation grounding — the figure must be in the source cited beside it
# ---------------------------------------------------------------------------

def check_citation_grounding(
    answer: str,
    cited_facts: Sequence[Dict[str, Any]],
    *,
    sentences: Optional[Sequence[str]] = None,
) -> List[QualityFinding]:
    """Bind every figure to the source cited next to it, not to the pool.

    The pool-wide check answers "does this number exist in the evidence at
    all", which passes a sentence that cites [3] while quoting a figure that
    only [7] supports. To a reader following the citation that is a
    fabrication: they open [3] and the number is not there. This check is the
    difference between citations that decorate and citations that verify.
    """
    findings: List[QualityFinding] = []
    by_number: Dict[int, List[str]] = {}
    for fact in cited_facts or []:
        index = _safe_int(fact.get("citation"), 0)
        if not index:
            continue
        text = f"{fact.get('claim', '')} {fact.get('direct_quote', '')}"
        by_number.setdefault(index, []).append(text)

    pool_values: Set[Tuple[float, str]] = set()
    for texts in by_number.values():
        for text in texts:
            pool_values |= _significant_values(text, limit=60)

    units = sentences if sentences is not None else _sentence_units(answer)
    for sentence in units:
        markers = _markers(sentence)
        if not markers:
            continue
        stated = _significant_values(_strip_markers(sentence))
        if not stated:
            continue
        cited_values: Set[Tuple[float, str]] = set()
        for marker in markers:
            for text in by_number.get(marker, []):
                cited_values |= _significant_values(text, limit=60)
        if not cited_values:
            continue
        for value in stated:
            if any(_values_match(value, known) for known in cited_values):
                continue
            elsewhere = any(_values_match(value, known) for known in pool_values)
            rendered = f"{value[0]:g}{(' ' + value[1]) if value[1] else ''}"
            findings.append(
                QualityFinding(
                    kind="misattributed_figure" if elsewhere else "unsupported_figure",
                    detail=(
                        f"{rendered} is cited to "
                        + ", ".join(f"[{m}]" for m in markers)
                        + (
                            ", but that figure appears in a different source"
                            if elsewhere
                            else ", which does not contain it"
                        )
                    ),
                    sentence=sentence,
                    severity="serious",
                )
            )
    return findings


def _sentence_units(answer: str, limit: int = 400) -> List[str]:
    """Line-aware sentence split, so an uncited bullet cannot hide in a list."""
    units: List[str] = []
    for raw_line in (answer or "").replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.lstrip("-* ").strip()
        if not line:
            continue
        try:
            pieces = split_into_sentences(line, max_sentences=20)
        except Exception:  # noqa: BLE001
            pieces = [line]
        for piece in pieces:
            text = piece.strip()
            if text:
                units.append(text)
        if len(units) >= limit:
            break
    return units[:limit]


# ---------------------------------------------------------------------------
# 5. Overclaiming — weak evidence stated as settled fact
# ---------------------------------------------------------------------------

_HEDGE_RE = re.compile(
    r"\b(may|might|could|appears?|seems?|suggests?|indicates?|reportedly|"
    r"allegedly|estimated|approximately|roughly|around|about|likely|unlikely|"
    r"probabl[ey]|possibl[ey]|one source|a single source|according to|claims?|"
    r"unverified|provisional|preliminary|so far|as of|not confirmed|"
    r"if accurate|on one account)\b",
    re.I,
)
_ASSERTIVE_RE = re.compile(
    r"\b(is|are|was|were|will|has|have|does|do|proves?|demonstrates?|shows?|"
    r"confirms?|establishes?|means?)\b",
    re.I,
)
# Strong verbs that need strong evidence no matter how the sentence is built.
_STRONG_CLAIM_RE = re.compile(
    r"\b(proves?|proven|confirms?|confirmed|establishes?|demonstrates?|"
    r"guarantees?|always|never|all |every |no one|definitive(?:ly)?)\b", re.I
)


def detect_overclaims(
    answer: str,
    cited_facts: Sequence[Dict[str, Any]],
    *,
    sentences: Optional[Sequence[str]] = None,
) -> List[QualityFinding]:
    """Find sentences asserting more than their cited evidence can carry.

    An unverified, single-source, C/D-grade claim written as "X is Y" is
    indistinguishable to the reader from an A-grade corroborated one. The whole
    value of grading evidence is lost if the prose flattens it. A hedge, an
    attribution, or an estimative qualifier discharges the requirement — this
    is not a demand for vagueness, it is a demand that certainty be earned.
    """
    findings: List[QualityFinding] = []
    by_number: Dict[int, List[Dict[str, Any]]] = {}
    for fact in cited_facts or []:
        index = _safe_int(fact.get("citation"), 0)
        if index:
            by_number.setdefault(index, []).append(fact)

    units = sentences if sentences is not None else _sentence_units(answer)
    for sentence in units:
        markers = _markers(sentence)
        if not markers:
            continue
        supporting = [f for m in markers for f in by_number.get(m, [])]
        if not supporting:
            continue

        best_confidence = max(_safe_float(f.get("confidence", 0.0)) for f in supporting)
        any_verified = any(f.get("verified") is True for f in supporting)
        best_independence = max(independent_corroboration(f) for f in supporting)
        grades = {str(f.get("evidence_grade", "") or "").upper() for f in supporting}
        weak_grade = bool(grades & {"C", "D"}) and not (grades & {"A", "B"})

        weak = (not any_verified) or best_independence <= 1 or weak_grade or best_confidence < 0.45
        if not weak:
            continue

        body = _strip_markers(sentence)
        if _HEDGE_RE.search(body):
            continue
        if not _ASSERTIVE_RE.search(body):
            continue

        reasons: List[str] = []
        if not any_verified:
            reasons.append("unverified")
        if best_independence <= 1:
            reasons.append("single independent source")
        if weak_grade:
            reasons.append("C/D grade")
        findings.append(
            QualityFinding(
                kind="overclaim",
                detail=(
                    "stated as established fact, but its cited evidence is "
                    + " and ".join(reasons or ["weak"])
                ),
                sentence=sentence,
                severity="warn",
            )
        )

    # An absolute claim ("proves", "always", "never") is flagged regardless of
    # grade: research evidence essentially never supports that register.
    for sentence in units:
        if _STRONG_CLAIM_RE.search(_strip_markers(sentence)):
            findings.append(
                QualityFinding(
                    kind="absolute_claim",
                    detail=(
                        "uses absolute or proof language, which research evidence "
                        "rarely supports; state the strength of support instead"
                    ),
                    sentence=sentence,
                    severity="warn",
                )
            )
    return findings


# ---------------------------------------------------------------------------
# 6. Internal consistency — the report must not contradict itself
# ---------------------------------------------------------------------------

_TOPIC_OVERLAP = 0.42


def detect_internal_conflicts(sections: Dict[str, str]) -> List[QualityFinding]:
    """Catch the same quantity stated two different ways in one report.

    Section-wise synthesis writes each section blind to its siblings, so the
    Executive Summary can say 12% while a deep-dive says 18% for the same
    thing. Deliberately conservative: two sentences must be about the same
    subject (token overlap) AND carry the same unit with different values
    before anything is flagged, so ordinary different-but-related figures are
    left alone.
    """
    findings: List[QualityFinding] = []
    indexed: List[Tuple[str, str, Set[Tuple[float, str]]]] = []
    for title, body in (sections or {}).items():
        for sentence in _sentence_units(body):
            values = _significant_values(_strip_markers(sentence))
            if values:
                indexed.append((title, sentence, values))

    seen: Set[Tuple[str, str]] = set()
    for i in range(len(indexed)):
        title_a, sentence_a, values_a = indexed[i]
        for j in range(i + 1, len(indexed)):
            title_b, sentence_b, values_b = indexed[j]
            if title_a == title_b:
                continue
            try:
                overlap = semantic_similarity(
                    _strip_markers(sentence_a), _strip_markers(sentence_b)
                )
            except Exception:  # noqa: BLE001
                continue
            if overlap < _TOPIC_OVERLAP:
                continue
            for value_a in values_a:
                for value_b in values_b:
                    if not value_a[1] or value_a[1] != value_b[1]:
                        continue
                    if _values_match(value_a, value_b):
                        continue
                    key = (f"{value_a[0]:g}{value_a[1]}", f"{value_b[0]:g}{value_b[1]}")
                    if key in seen:
                        continue
                    seen.add(key)
                    findings.append(
                        QualityFinding(
                            kind="internal_conflict",
                            detail=(
                                f"'{title_a}' gives {value_a[0]:g} {value_a[1]} where "
                                f"'{title_b}' gives {value_b[0]:g} {value_b[1]} for what "
                                "appears to be the same quantity"
                            ),
                            sentence=sentence_a,
                            severity="serious",
                        )
                    )
    return findings


# ---------------------------------------------------------------------------
# 7. Per-section citation coverage
# ---------------------------------------------------------------------------

def section_coverage(sections: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Citation density per section, so one clean section cannot mask a bare one.

    A global density of 70% looks acceptable and can hide a section at 0%.
    Readers trust or distrust a report section by section, so it is measured
    that way.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for title, body in (sections or {}).items():
        factual = 0
        cited = 0
        for sentence in _sentence_units(body):
            if not is_factual_sentence(_strip_markers(sentence)):
                continue
            factual += 1
            if _markers(sentence):
                cited += 1
        if factual:
            out[title] = {
                "factual_sentences": factual,
                "cited": cited,
                "density": round(cited / factual, 3),
            }
    return out


# ---------------------------------------------------------------------------
# Writing contracts — prevent at the prompt what the audit measures after
# ---------------------------------------------------------------------------

# Words of estimative probability. Binding confidence bands to a fixed
# vocabulary is what lets a reader translate the report into a decision; free
# hedging ("some evidence suggests it may possibly") communicates nothing.
_ESTIMATIVE_BANDS: Sequence[Tuple[float, str]] = (
    (0.90, "almost certainly"),
    (0.75, "very likely"),
    (0.55, "likely"),
    (0.45, "roughly even odds"),
    (0.25, "unlikely"),
    (0.00, "very unlikely"),
)


def estimative_band(confidence: float) -> str:
    """The estimative phrase a given confidence licenses."""
    score = max(0.0, min(1.0, _safe_float(confidence)))
    for threshold, phrase in _ESTIMATIVE_BANDS:
        if score >= threshold:
            return phrase
    return "very unlikely"


_CALIBRATION_CONTRACT = (
    "CALIBRATED LANGUAGE — the reader must be able to tell how much to trust "
    "each statement from the wording alone:\n"
    "- State an A-grade, independently corroborated finding plainly, with no "
    "hedge. Hedging established facts is its own failure.\n"
    "- A single-source, unverified or C/D-grade finding MUST carry an "
    "attribution or qualifier ('one source reports', 'estimated at', "
    "'reportedly'). Never write it in the same flat voice as a verified one.\n"
    "- Where you express a likelihood, use these exact phrases and nothing "
    "vaguer: almost certainly / very likely / likely / roughly even odds / "
    "unlikely / very unlikely.\n"
    "- Never write 'proves', 'confirms', 'always', 'never' or 'definitively'. "
    "Research evidence supports degrees of confidence, not proof.\n"
    "- Do not stack hedges. One qualifier per claim; 'may possibly somewhat "
    "suggest' communicates nothing."
)

_PREMISE_CONTRACT = (
    "CHECK THE QUESTION'S PREMISE FIRST. If the evidence contradicts something "
    "the question assumes (it asks why X happened and the evidence says X did "
    "not happen, or names an entity the evidence shows does not exist or is "
    "being confused with another), say so in the FIRST sentence and answer the "
    "question the reader should have asked. Answering a false premise fluently "
    "is the most damaging thing this report can do."
)


def render_quality_contract(
    *,
    profile: Optional[TemporalProfile] = None,
    confidence: Optional[float] = None,
    query_type: str = "",
) -> str:
    """The writing contract that prevents the defects this module measures.

    Injected into the writer prompt so the standard is enforced from both ends:
    the model is told what calibrated, dated, premise-checked writing looks
    like, and the audit afterwards measures whether it delivered.
    """
    parts: List[str] = [_PREMISE_CONTRACT, _CALIBRATION_CONTRACT]

    temporal_lines: List[str] = []
    if profile and profile.newest:
        temporal_lines.append(
            "DATING — the evidence for this report was published between "
            f"{profile.oldest:%B %Y} and {profile.newest:%B %Y}."
            if profile.oldest and profile.oldest != profile.newest
            else f"DATING — the evidence for this report dates to {profile.newest:%B %Y}."
        )
        temporal_lines.append(
            "Attach an explicit as-of date to every claim about a current "
            "state, price, ranking, headcount or status. Write 'as of "
            f"{profile.newest:%B %Y}', never a bare 'currently'."
        )
        if profile.stale:
            temporal_lines.append(
                "This evidence is OLD relative to the question. Say so plainly "
                "in the Executive Summary rather than presenting dated findings "
                "as the present state."
            )
    elif profile:
        temporal_lines.append(
            "DATING — no source carried a publication date. Do not write "
            "'currently', 'as of today', 'the latest' or any other claim of "
            "currency you cannot support; say the evidence is undated."
        )
    if temporal_lines:
        parts.append("\n".join(temporal_lines))

    if confidence is not None:
        phrase = estimative_band(_safe_float(confidence))
        parts.append(
            "OVERALL CALIBRATION — the evidence base supports a conclusion that "
            f"is {phrase} correct. Do not write the report in a register more "
            "confident than that."
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

@dataclass
class ResearchQualityReport:
    """Everything this layer measured, for the report and for the caller."""

    findings: List[QualityFinding] = field(default_factory=list)
    temporal: TemporalProfile = field(default_factory=TemporalProfile)
    independence: IndependenceReport = field(default_factory=IndependenceReport)
    coverage: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def serious(self) -> List[QualityFinding]:
        return [f for f in self.findings if f.severity == "serious"]

    @property
    def is_clean(self) -> bool:
        return not self.findings

    def by_kind(self, kind: str) -> List[QualityFinding]:
        return [f for f in self.findings if f.kind == kind]

    def weakest_sections(self, threshold: float = 0.5) -> List[Tuple[str, float]]:
        return sorted(
            (
                (title, float(stats["density"]))
                for title, stats in self.coverage.items()
                if float(stats["density"]) < threshold and int(stats["factual_sentences"]) >= 3
            ),
            key=lambda item: item[1],
        )

    def render_note(self) -> str:
        """Report the defects in the report itself, or return "" when clean."""
        lines: List[str] = []

        misattributed = self.by_kind("misattributed_figure")
        if misattributed:
            lines.append(
                f"{len(misattributed)} figure(s) are attributed to a source that "
                "does not contain them — the number exists elsewhere in the "
                f"evidence. First: {misattributed[0].detail}."
            )
        unsupported = self.by_kind("unsupported_figure")
        if unsupported:
            lines.append(
                f"{len(unsupported)} figure(s) appear in no cited source. "
                f"First: {unsupported[0].detail}."
            )
        conflicts = self.by_kind("internal_conflict")
        if conflicts:
            lines.append(
                f"{len(conflicts)} internal inconsistency(ies): {conflicts[0].detail}."
            )
        overclaims = self.by_kind("overclaim")
        if overclaims:
            lines.append(
                f"{len(overclaims)} statement(s) are written as settled fact on "
                "evidence that is unverified or single-source; read them as "
                "provisional."
            )
        absolutes = self.by_kind("absolute_claim")
        if absolutes:
            lines.append(
                f"{len(absolutes)} statement(s) use proof or absolute language "
                "that the evidence does not support."
            )
        weak = self.weakest_sections()
        if weak:
            named = ", ".join(f"{title} ({density:.0%})" for title, density in weak[:3])
            lines.append(f"Thinly cited section(s): {named}.")

        temporal_warning = self.temporal.warning()
        if temporal_warning:
            lines.append(temporal_warning)
        independence_warning = self.independence.warning()
        if independence_warning:
            lines.append(independence_warning)

        if not lines:
            return ""
        return "\n\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings[:20]],
            "serious_count": len(self.serious),
            "temporal": self.temporal.to_dict(),
            "independence": self.independence.to_dict(),
            "section_coverage": self.coverage,
            "is_clean": self.is_clean,
        }


def assess_report_quality(
    answer: str,
    *,
    cited_facts: Sequence[Dict[str, Any]] = (),
    usable_facts: Sequence[Dict[str, Any]] = (),
    sections: Optional[Dict[str, str]] = None,
    query_type: str = "",
    temporal: Optional[TemporalProfile] = None,
    independence: Optional[IndependenceReport] = None,
) -> ResearchQualityReport:
    """Run every check on a finished draft. Total and fail-safe.

    Each check is isolated: one raising does not lose the others, and the worst
    case is fewer findings, never a broken report.
    """
    report = ResearchQualityReport()
    report.temporal = temporal or _guarded(
        lambda: temporal_profile(usable_facts, query_type=query_type), TemporalProfile()
    )
    report.independence = independence or _guarded(
        lambda: assess_independence(usable_facts), IndependenceReport()
    )

    units = _guarded(lambda: _sentence_units(answer), [])
    report.findings.extend(
        _guarded(lambda: check_citation_grounding(answer, cited_facts, sentences=units), [])
    )
    report.findings.extend(
        _guarded(lambda: detect_overclaims(answer, cited_facts, sentences=units), [])
    )
    if sections:
        report.findings.extend(_guarded(lambda: detect_internal_conflicts(sections), []))
        report.coverage = _guarded(lambda: section_coverage(sections), {})
    return report


def _guarded(fn, default):
    """Run a check; on any failure return the default rather than propagating."""
    try:
        return fn()
    except Exception:  # noqa: BLE001 - a quality check must never break synthesis
        return default
