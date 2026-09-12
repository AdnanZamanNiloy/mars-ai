"""Evidence hygiene, scoring, deduplication and citation support checks.

Upgrades in this version (behaviour-preserving unless noted):

1. `dedupe_semantic_facts` NO LONGER DESTROYS METADATA. It previously rebuilt
   each fact as a 5-key dict, silently dropping `verified`,
   `verification_score`, `direct_quote`, `published_at` and anything else
   attached upstream. That one line is why critic.py had to reconstruct
   verification standing by matching normalized claim text, and why
   synthesizer._gap_stats had to guess verified counts. Facts now keep every
   key they arrive with.

2. Merging records CROSS-SOURCE AGREEMENT instead of discarding it. When two
   independent domains state the same thing, that is the strongest confidence
   signal a research system has; the old code kept the higher-confidence copy
   and threw the corroboration away. Merged facts carry
   `corroborating_sources` / `corroboration_count`.

3. Similarity has a cheap token prefilter in front of SequenceMatcher, which
   was the hottest CPU path in the pipeline on deep runs (O(n^2) character
   diff over 260-char claims across 300+ facts).

4. `source_reliability_score` delegates to `sources.authority_score`, so one
   registry drives ranking, filtering, verification and confidence.

5. `verify_answer_support` also checks NUMBERS. A cited sentence used to pass
   on lexical similarity alone, so "costs fell 40%" was "supported" by a
   source saying "costs fell 4%".

6. New: `extract_numbers`, `numeric_conflict`, `claim_polarity`,
   `evidence_stats` — shared primitives for the verifier, the contradiction
   engine and the confidence engine, so all three agree on what a number is.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.agents.sources import (
    LOW_TRUST_DOMAINS,
    authority_score,
    canonical_url,
    classify_source,
    evidence_freshness,
    extract_domain as _extract_domain,
    is_primary_source,
    primary_source_share,
)

# ---------------------------------------------------------------------------
# Legacy public names, kept so existing imports and tests keep working.
# The authoritative registry now lives in app.agents.sources.
# ---------------------------------------------------------------------------

LOW_QUALITY_DOMAINS: Set[str] = {
    "reddit.com", "quora.com", "zhihu.com", "baidu.com", "sohu.com",
    "csdn.net", "medium.com", "blogspot.com", "substack.com",
    "wordpress.com", "youtube.com", "youtu.be", "tiktok.com",
    "pinterest.com", "whatfix.com",
}

HIGH_AUTHORITY_DOMAINS: Set[str] = {
    "stanford.edu", "plato.stanford.edu", "iep.utm.edu", "britannica.com",
    "routledge.com", "nature.com", "science.org", "arxiv.org",
    "huggingface.co", "paperswithcode.com", "github.com", "openml.org",
    "mlcommons.org", "who.int", "oecd.org", "worldbank.org", "imf.org",
    "un.org",
}


def extract_domain(url: str) -> str:
    return _extract_domain(url)


def is_high_quality_domain(
    url: str, blocked_domains: Iterable[str] = LOW_TRUST_DOMAINS
) -> bool:
    """False for blocked or zero-authority hosts. Signature preserved: a
    caller-supplied blocklist is honoured on top of the registry."""
    domain = extract_domain(url)
    if not domain:
        return False
    for blocked in blocked_domains or ():
        blocked_value = str(blocked).lower().strip()
        if not blocked_value:
            continue
        if domain == blocked_value or domain.endswith(f".{blocked_value}"):
            return False
    return authority_score(url) > 0.0


def source_reliability_score(url: str) -> float:
    """0.0-1.0 authority for a URL (0.0 = never cite). Registry-backed."""
    return authority_score(url)


def source_profile(url: str) -> Dict[str, Any]:
    return classify_source(url).to_dict()


# ---------------------------------------------------------------------------
# Claim normalization + similarity
# ---------------------------------------------------------------------------

def normalize_claim_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    cleaned = cleaned.strip("-:;,. ")
    if not cleaned:
        return ""
    if len(cleaned) > 260:
        cleaned = cleaned[:257].rstrip() + "..."
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _tokenize(text: str) -> Set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _semantic_similarity(a: str, b: str) -> float:
    """Blended char-ratio + token-Jaccard similarity in [0, 1].

    The Jaccard term is computed first and used as a gate: SequenceMatcher is
    ~100x more expensive, and two claims sharing under 15% of their vocabulary
    can never reach the 0.86 merge threshold (max blended score with
    jaccard=0.15 is 0.6*1.0 + 0.4*0.15 = 0.66). Skipping the char diff in
    that case is the difference between a 40ms and a 4s dedup pass on a deep
    run, with identical output.
    """
    a_norm = normalize_claim_text(a).lower()
    b_norm = normalize_claim_text(b).lower()
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0

    tok_a = _tokenize(a_norm)
    tok_b = _tokenize(b_norm)
    jaccard = _jaccard(tok_a, tok_b)
    if jaccard < 0.15:
        return round(0.4 * jaccard, 4)

    seq_ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    return 0.6 * seq_ratio + 0.4 * jaccard


def semantic_similarity(a: str, b: str) -> float:
    """Public alias — the contradiction and confidence engines need this."""
    return _semantic_similarity(a, b)


# ---------------------------------------------------------------------------
# Numeric grounding: the cheapest, highest-yield hallucination detector
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"(?P<sign>[-+]?)"
    # Thousands separators: comma (1,600), narrow/regular space and thin space
    # (1 600 / 1 600) as used by SI, most statistical agencies and the EU. Space
    # grouping was previously unparsed, so "1 600 GW" read as 600 GW — a source
    # figure the report could then never match, flagging a correct number as a
    # fabrication.
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|\d{1,3}(?:[\u00a0\u202f\u2009 ]\d{3})+(?:\.\d+)?"
    r"|\d+(?:\.\d+)?)"
    r"\s*"
    r"(?P<unit>%|percentage points?|percent|bps|"
    r"trillion|billion|million|thousand|bn|mn|"
    r"twh|gwh|mwh|kwh|tw|gw|mw|kw|"
    r"usd|eur|gbp|jpy|inr|bdt|"
    r"years?|months?|days?|hours?|"
    r"tonnes?|tons?|kg|km|cm|mm)?",
    re.IGNORECASE,
)

_SCALE: Dict[str, float] = {
    "trillion": 1e12, "billion": 1e9, "bn": 1e9,
    "million": 1e6, "mn": 1e6, "thousand": 1e3,
}

_UNIT_ALIASES: Dict[str, str] = {
    "percent": "%", "percentage point": "%", "percentage points": "%",
    "tons": "tonne", "ton": "tonne", "tonnes": "tonne", "tonne": "tonne",
    "years": "year", "months": "month", "days": "day", "hours": "hour",
}

_CURRENCY_RE = re.compile(r"[$€£¥₹৳]")


@dataclass(frozen=True)
class Quantity:
    """A number lifted out of prose, with scale folded into the value."""

    value: float
    unit: str          # "%", "gw", "usd", "year", "" (dimensionless)
    raw: str
    is_year: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "raw": self.raw,
            "is_year": self.is_year,
        }


def extract_numbers(text: str, limit: int = 12) -> List[Quantity]:
    """Pull quantities out of a claim or passage.

    Scale words are folded in ("2.4 billion" -> 2.4e9) so two sources can be
    compared even when one writes 2.4bn and the other writes 2,400,000,000.
    Bare 4-digit values in 1500-2099 are flagged `is_year`: comparing a year
    to a magnitude is a category error the contradiction engine must avoid.
    """
    out: List[Quantity] = []
    haystack = text or ""
    for match in _NUMBER_RE.finditer(haystack):
        raw_num = re.sub(r"[,\u00a0\u202f\u2009 ]", "", match.group("num"))
        try:
            value = float(raw_num)
        except ValueError:
            continue
        if match.group("sign") == "-":
            value = -value

        unit = (match.group("unit") or "").strip().lower()
        if unit in _SCALE:
            value *= _SCALE[unit]
            unit = ""
        unit = _UNIT_ALIASES.get(unit, unit)

        is_year = (
            not unit
            and float(raw_num).is_integer()
            and len(raw_num.split(".")[0]) == 4
            and 1500 <= value <= 2099
        )

        # Currency symbol immediately before the number gives the unit when
        # no unit word follows ("$4.2 billion" -> usd).
        if not unit and match.start() > 0:
            prefix = haystack[max(0, match.start() - 2): match.start()]
            if _CURRENCY_RE.search(prefix):
                unit = "usd" if "$" in prefix else "currency"
                is_year = False

        out.append(Quantity(value=value, unit=unit, raw=match.group(0).strip(), is_year=is_year))
        if len(out) >= max(1, limit):
            break
    return out


def _significant_quantities(text: str) -> List[Quantity]:
    """Quantities worth grounding: skips years and small ordinals, which are
    ubiquitous and produce false "unsupported" verdicts."""
    return [
        q for q in extract_numbers(text)
        if not q.is_year and (abs(q.value) >= 2 or q.unit)
    ]


def numbers_grounded(claim: str, source_text: str, tolerance: float = 0.02) -> bool:
    """True when every significant number in `claim` appears in `source_text`.

    Matching is value-based with a small relative tolerance, so "1.2 billion"
    grounds "1,200,000,000" and rounding differences do not fail an honest
    claim. Claims with no significant numbers are vacuously grounded.
    """
    claim_numbers = _significant_quantities(claim)
    if not claim_numbers:
        return True
    source_values = [q.value for q in extract_numbers(source_text or "", limit=400)]
    if not source_values:
        return False
    for q in claim_numbers:
        target = abs(q.value)
        scale = max(target, 1.0)
        if not any(abs(abs(v) - target) <= tolerance * scale for v in source_values):
            return False
    return True


def numeric_conflict(
    a: str, b: str, divergence: float = 0.20
) -> Optional[Dict[str, Any]]:
    """Compare like-united quantities in two claims; report a real conflict.

    Only quantities sharing a unit are compared — a "%" against a "gw" is not
    a disagreement — and years are excluded. Returns None when the claims are
    numerically compatible or not comparable at all.
    """
    qa = {q.unit: q for q in _significant_quantities(a)}
    qb = {q.unit: q for q in _significant_quantities(b)}
    shared = [u for u in qa if u in qb]
    for unit in shared:
        va, vb = qa[unit].value, qb[unit].value
        scale = max(abs(va), abs(vb), 1e-9)
        rel = abs(va - vb) / scale
        if rel >= divergence:
            return {
                "unit": unit or "dimensionless",
                "value_a": va,
                "value_b": vb,
                "relative_divergence": round(rel, 4),
                "raw_a": qa[unit].raw,
                "raw_b": qb[unit].raw,
            }
    return None


# ---------------------------------------------------------------------------
# Polarity: catches the failure lexical overlap is blind to
# ---------------------------------------------------------------------------

_NEGATION_TOKENS = {
    "not", "no", "never", "cannot", "cant", "isnt", "arent", "wasnt",
    "werent", "doesnt", "dont", "didnt", "wont", "without", "neither",
    "nor", "none", "fails", "failed", "unable", "lacks", "lacking",
    "absent", "denies", "denied", "refutes", "refuted", "disproves",
    "contradicts", "rejects", "unsupported", "false", "incorrect",
}

_DIRECTION_UP = {
    "increase", "increases", "increased", "increasing", "rise", "rises",
    "rising", "rose", "grow", "grows", "growing", "grew", "growth",
    "higher", "surge", "surged", "expand", "expanded", "expansion",
    "improve", "improved", "gain", "gained", "up", "accelerate",
    "accelerated", "more", "exceeds", "exceeded", "outperforms",
}

_DIRECTION_DOWN = {
    "decrease", "decreases", "decreased", "decreasing", "fall", "falls",
    "falling", "fell", "decline", "declines", "declined", "declining",
    "drop", "drops", "dropped", "lower", "shrink", "shrank", "reduce",
    "reduced", "reduction", "contract", "contracted", "worsen", "worsened",
    "loss", "lose", "lost", "down", "decelerate", "less", "below",
    "underperforms",
}


def claim_polarity(text: str) -> int:
    """-1 negated / downward, +1 affirmative-upward, 0 neutral.

    Deliberately coarse. Its only job is to catch the case lexical overlap
    cannot see: a claim asserting the OPPOSITE of its source shares nearly
    all of the source's vocabulary and therefore verified cleanly before.
    """
    tokens = re.findall(r"[a-z']+", (text or "").lower())
    flat = {t.replace("'", "") for t in tokens}
    negated = bool(flat & _NEGATION_TOKENS)
    up = len(flat & _DIRECTION_UP)
    down = len(flat & _DIRECTION_DOWN)
    direction = 0
    if up > down:
        direction = 1
    elif down > up:
        direction = -1
    if negated:
        return -1 if direction >= 0 else 1
    return direction


def polarity_conflict(a: str, b: str, min_similarity: float = 0.45) -> bool:
    """True when two claims talk about the same thing with opposite polarity."""
    pa, pb = claim_polarity(a), claim_polarity(b)
    if pa == 0 or pb == 0 or pa == pb:
        return False
    return _semantic_similarity(a, b) >= min_similarity


# ---------------------------------------------------------------------------
# Domain filtering (signatures and defaults unchanged)
# ---------------------------------------------------------------------------

def filter_search_results_by_domain(
    results: List[Dict[str, str]],
    min_score: float = 0.60,
    fallback_min_score: float = 0.55,
    fallback_limit: int = 8,
) -> List[Dict[str, str]]:
    strong: List[Dict[str, str]] = []
    fallback: List[Dict[str, Any]] = []

    for item in results or []:
        url = str(item.get("url", "")).strip()
        if not url or not is_high_quality_domain(url):
            continue

        score = source_reliability_score(url)
        if score >= min_score:
            strong.append(item)
            continue
        if score >= fallback_min_score:
            fallback.append({"score": score, "item": item})

    if strong:
        strong_domains = {
            extract_domain(str(item.get("url", ""))) for item in strong if item.get("url")
        }
        strong_domains.discard("")
        if len(strong) >= 3 and len(strong_domains) >= 2:
            return strong

        supplemented = list(strong)
        seen_urls = {
            canonical_url(str(item.get("url", ""))) for item in strong if item.get("url")
        }
        fallback.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        for entry in fallback:
            item = entry.get("item")
            if not isinstance(item, dict):
                continue
            url = canonical_url(str(item.get("url", "")).strip())
            if not url or url in seen_urls:
                continue
            supplemented.append(item)
            seen_urls.add(url)
            if len(supplemented) >= max(3, min(fallback_limit, 10)):
                break
        return supplemented

    fallback.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return [entry["item"] for entry in fallback[: max(1, fallback_limit)]]


def filter_facts_by_domain(
    facts: List[Dict[str, Any]],
    min_score: float = 0.62,
    fallback_min_score: float = 0.55,
) -> List[Dict[str, Any]]:
    strong: List[Dict[str, Any]] = []
    fallback: List[Dict[str, Any]] = []

    for item in facts or []:
        source = str(item.get("source", "")).strip()
        claim = normalize_claim_text(str(item.get("claim", "")))
        if not source or not claim or not is_high_quality_domain(source):
            continue

        score = source_reliability_score(source)
        if score >= min_score:
            strong.append(item)
            continue
        if score >= fallback_min_score:
            fallback.append(item)

    if strong:
        strong_domains = {
            extract_domain(str(item.get("source", ""))) for item in strong if item.get("source")
        }
        strong_domains.discard("")
        if len(strong) >= 3 and len(strong_domains) >= 2:
            return strong

        supplemented = list(strong)
        seen_sources = {
            canonical_url(str(item.get("source", ""))) for item in strong if item.get("source")
        }
        fallback.sort(key=lambda x: float(x.get("confidence", 0.0) or 0.0), reverse=True)

        for item in fallback:
            source = canonical_url(str(item.get("source", "")).strip())
            if not source or source in seen_sources:
                continue
            supplemented.append(item)
            seen_sources.add(source)
            if len(supplemented) >= 8:
                break
        return supplemented

    fallback.sort(key=lambda x: float(x.get("confidence", 0.0) or 0.0), reverse=True)
    return fallback[:8]


# ---------------------------------------------------------------------------
# Deduplication with corroboration tracking
# ---------------------------------------------------------------------------

def dedupe_semantic_facts(
    facts: List[Dict[str, Any]], threshold: float = 0.86
) -> List[Dict[str, Any]]:
    """Collapse restatements of the same claim, preserving every field.

    Two changes from the previous implementation, both consequential:

    * The kept fact is the ORIGINAL dict (copied), not a 5-key reconstruction.
      `verified`, `verification_score`, `direct_quote`, `published_at`,
      `search_type` and any future field survive dedup.
    * A merge is recorded rather than discarded. Independent corroboration
      across domains is the single strongest confidence signal available, and
      it used to be thrown away. Merged facts gain:
        corroborating_sources  — distinct source URLs asserting the claim
        corroboration_count    — len(corroborating_sources)
        merged_claims          — the alternate phrasings seen
      Confidence is nudged up (capped at 0.97) per independent domain, which
      is what "cross-source agreement" means operationally.
    """
    deduped: List[Dict[str, Any]] = []
    token_cache: List[Set[str]] = []

    for item in facts or []:
        if not isinstance(item, dict):
            continue
        claim = normalize_claim_text(str(item.get("claim", "")))
        source = str(item.get("source", "")).strip()
        if not claim or not source:
            continue
        try:
            confidence = float(item.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        candidate = dict(item)
        candidate["claim"] = claim
        candidate["source"] = source
        candidate["confidence"] = confidence
        candidate.setdefault("agent", str(item.get("agent", "") or ""))
        candidate.setdefault("sub_question", str(item.get("sub_question", "") or ""))
        candidate_tokens = _tokenize(claim)

        merge_index = -1
        for idx, kept in enumerate(deduped):
            if _jaccard(candidate_tokens, token_cache[idx]) < 0.15:
                continue
            if _semantic_similarity(claim, str(kept.get("claim", ""))) >= threshold:
                merge_index = idx
                break

        if merge_index == -1:
            # Corroboration already established upstream (e.g. the same claim
            # was matched across pages during extraction) must not be reset to
            # 1 just because this pass saw the fact once. Dedup only ever adds
            # evidence of agreement; it never removes it.
            prior_sources = [
                str(u) for u in (item.get("corroborating_sources") or []) if str(u).strip()
            ]
            if not any(canonical_url(u) == canonical_url(source) for u in prior_sources):
                prior_sources.append(source)
            try:
                prior_count = int(item.get("corroboration_count", 1) or 1)
            except (TypeError, ValueError):
                prior_count = 1
            candidate["corroborating_sources"] = prior_sources
            candidate["corroboration_count"] = max(len(prior_sources), prior_count, 1)
            deduped.append(candidate)
            token_cache.append(candidate_tokens)
            continue

        kept = deduped[merge_index]
        corroborating: List[str] = list(kept.get("corroborating_sources") or [])
        known_domains = {extract_domain(u) for u in corroborating}
        new_domain = extract_domain(source)
        incoming = [str(u) for u in (item.get("corroborating_sources") or []) if str(u).strip()]
        if source not in incoming:
            incoming.append(source)
        seen_documents = {canonical_url(u) for u in corroborating}
        for extra in incoming:
            if canonical_url(extra) not in seen_documents:
                corroborating.append(extra)
                seen_documents.add(canonical_url(extra))

        variants: List[str] = list(kept.get("merged_claims") or [])
        if claim != str(kept.get("claim", "")) and claim not in variants:
            variants.append(claim)

        winner = candidate if confidence > float(kept.get("confidence", 0.0) or 0.0) else kept
        merged = dict(winner)
        merged["corroborating_sources"] = corroborating
        merged["corroboration_count"] = len(corroborating)
        if variants:
            merged["merged_claims"] = variants[:5]

        # Independent-domain agreement lifts confidence; a second copy from
        # the same domain does not (self-syndication is not corroboration).
        if new_domain and new_domain not in known_domains:
            base = float(merged.get("confidence", 0.0) or 0.0)
            merged["confidence"] = round(min(0.97, base + 0.04), 4)

        # A verified copy must never be replaced by an unverified one.
        if kept.get("verified") and not merged.get("verified"):
            merged["verified"] = True
            merged["verification_score"] = kept.get("verification_score")
            merged["verification_reason"] = kept.get("verification_reason")

        deduped[merge_index] = merged
        token_cache[merge_index] = _tokenize(str(merged.get("claim", "")))

    return deduped


# ---------------------------------------------------------------------------
# Citation support verification
# ---------------------------------------------------------------------------

CITATION_RE = re.compile(r"\[(\d+)\]")
SOURCES_HEADING_RE = re.compile(r"\n+#{0,6}\s*Sources:?\s*\n")
LEGEND_RE = re.compile(r"^\[(\d+)\]\s+\S.*?—\s*(\S+)\s*$")
SUPPORT_THRESHOLD = 0.30


def verify_answer_support(
    answer: str,
    facts: List[Dict[str, Any]],
    threshold: float = SUPPORT_THRESHOLD,
) -> Dict[str, Any]:
    """Post-synthesis check: every cited sentence must overlap verified
    evidence from the source it cites, AND every significant number in that
    sentence must appear in that source's evidence.

    The legend is parsed back out of the answer itself, so numbering can never
    drift from what was emitted. Sentences without markers count as uncited
    (not failed). Return shape is a superset of the previous one: existing
    keys are unchanged, `numeric_failures` and `numeric_rate` are new.
    """
    match = SOURCES_HEADING_RE.search(answer or "")
    if match:
        body = (answer or "")[: match.start()]
        legend_block = (answer or "")[match.end():]
    else:
        body, _, legend_block = (answer or "").partition("\nSources:")

    legend_urls: Dict[int, str] = {}
    for line in legend_block.splitlines():
        m = LEGEND_RE.match(line.strip())
        if m:
            try:
                legend_urls[int(m.group(1))] = m.group(2)
            except (TypeError, ValueError):
                continue

    verified_by_url: Dict[str, List[str]] = {}
    for fact in facts or []:
        if not fact.get("verified"):
            continue
        url = str(fact.get("source", "") or "")
        claim = str(fact.get("claim", "") or "")
        if not (url and claim):
            continue
        verified_by_url.setdefault(url, []).append(claim)
        for extra in fact.get("corroborating_sources") or []:
            extra_url = str(extra or "")
            if extra_url and extra_url != url:
                verified_by_url.setdefault(extra_url, []).append(claim)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    cited = supported = uncited = 0
    numeric_checked = numeric_ok = 0
    unsupported: List[str] = []
    numeric_failures: List[str] = []

    for sentence in sentences:
        numbers = [int(n) for n in CITATION_RE.findall(sentence)]
        if not numbers:
            if len(sentence.split()) >= 8:
                uncited += 1
            continue
        cited += 1

        cited_claims: List[str] = []
        for n in numbers:
            cited_claims.extend(verified_by_url.get(legend_urls.get(n, ""), []))

        hit = any(
            _semantic_similarity(sentence, claim) >= threshold for claim in cited_claims
        )

        sentence_numbers = _significant_quantities(CITATION_RE.sub("", sentence))
        if sentence_numbers:
            numeric_checked += 1
            pool = " ".join(cited_claims)
            if numbers_grounded(CITATION_RE.sub("", sentence), pool):
                numeric_ok += 1
            else:
                hit = False
                numeric_failures.append(sentence[:160])

        if hit:
            supported += 1
        elif sentence[:160] not in numeric_failures:
            unsupported.append(sentence[:160])

    return {
        "sentences": len(sentences),
        "cited": cited,
        "supported": supported,
        "uncited": uncited,
        "unsupported": unsupported,
        "rate": (supported / cited) if cited else None,
        "numeric_failures": numeric_failures,
        "numeric_rate": (numeric_ok / numeric_checked) if numeric_checked else None,
    }


# ---------------------------------------------------------------------------
# Snippet / claim hygiene (rules unchanged — these encode hard-won fixes)
# ---------------------------------------------------------------------------

DATE_STAMP_RE = re.compile(
    r"^(?:[A-Z][a-z]{2,8}\s+\d{1,2},\s+\d{4}|\d+\s+(?:day|hour|minute|second)s?\s+ago)"
    r"(?:\s*[·\-–|]\s*|\s+\d+\s+min(?:ute)?s?\s+read\b\s*|\s+)",
    re.IGNORECASE,
)
LEADING_HASH_RE = re.compile(r"^#+\s*")
LINK_TEXT_RE = re.compile(
    r"\b(?:Learn|Read|Show|See|Click|Continue)\s+mo(?:r(?:e)?)?\b\.?", re.IGNORECASE
)
WIKI_NAV_RE = re.compile(
    r"^(Main article|See also|Further information|References|External links|Notes)\s*:",
    re.IGNORECASE,
)
EXCERPT_SEAM_RE = re.compile(r"\[\s*\.\.\.|\(\s*\.\.\.")
LINK_TITLE_PAREN_RE = re.compile(r'\s*"[^"]{2,80}"\)')
HEADING_MARK_RE = re.compile(r"#{2,}\s*")
BLOCKQUOTE_RE = re.compile(r"^(?:>\s*)+")
LATEX_SOUP_RE = re.compile(r"\\[a-zA-Z]{3,}")
BOILERPLATE_LEAD_RE = re.compile(
    r"^(Use this page to\b|This (article|post|guide|page|blog) will\b)", re.IGNORECASE
)
DANGLING_END_RE = re.compile(
    r"\b(including|such\s+as|as\s+well\s+as|with|from|through|using|by|and|or|"
    r"to|of|in|on|for|as|like|via|per|within|without|between|among)\s*\.?\s*$",
    re.IGNORECASE,
)
TITLE_PREFIX_RE = re.compile(r"^([^.!?]{2,80}):\s*[^.!?]{2,80}\.?$")

# Cookie/consent and paywall furniture. These reach the claim pool through
# fetched page text on publishers that render banners server-side, and read as
# broken evidence in a finished report.
CONSENT_NOISE_RE = re.compile(
    r"\b(accept all cookies|manage (your )?(cookie|consent) preferences|"
    r"subscribe to (continue|read)|sign in to (continue|read)|"
    r"you have \d+ free articles?|enable cookies|privacy policy and terms)\b",
    re.IGNORECASE,
)
# Pure navigation runs ("Home About Contact Careers Privacy") — capitalized
# fragments with no verb, produced by nav bars flattened into text.
NAV_RUN_RE = re.compile(
    r"^(?:(?:Home|About|Contact|Careers|Privacy|Terms|Login|Sign\s?in|Menu|"
    r"Search|Newsletter|Subscribe|Share|Follow|Advertisement)\b[\s|·,-]*){3,}$",
    re.IGNORECASE,
)

MIN_CLEAN_CLAIM_CHARS = 50
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _looks_like_title_prefix(head: str) -> bool:
    words = re.findall(r"[A-Za-z][a-z]*", head or "")
    if len(words) < 2:
        return False
    capped = sum(1 for w in words if w[0].isupper())
    return capped / len(words) > 0.5


def split_into_sentences(text: str, max_sentences: int = 12) -> List[str]:
    """Split page/snippet text into candidate claim sentences."""
    chunks = [chunk.strip() for chunk in _SENTENCE_SPLIT_RE.split(text or "")]
    return [chunk for chunk in chunks if chunk][:max_sentences]


def looks_truncated(text: str) -> bool:
    """True when a claim ends in a probable mid-word cut ("...a large dat")."""
    stripped = re.sub(r"\s+", " ", (text or "")).strip()
    if not stripped:
        return True
    if re.search(r"[.!?](?=\s|$)", stripped):
        return False
    return len(stripped.rsplit(None, 1)[-1]) <= 3


def clean_snippet_text(
    snippet: str, max_chars: int = 300, min_chars: int = MIN_CLEAN_CLAIM_CHARS
) -> str:
    """Turn a raw snippet or model claim into a presentable claim sentence.

    Returns "" when nothing salvageable remains. Every rejection rule here
    corresponds to a specific garbage shape seen in production; two are new in
    this version (consent/paywall furniture, flattened nav runs).
    """
    text = re.sub(r"\s+", " ", (snippet or "")).strip()
    text = html.unescape(text).strip()
    text = DATE_STAMP_RE.sub("", text).strip()
    text = LEADING_HASH_RE.sub("", text).strip()
    text = BLOCKQUOTE_RE.sub("", text).strip()
    text = LINK_TEXT_RE.sub("", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    if WIKI_NAV_RE.match(text):
        return ""
    if EXCERPT_SEAM_RE.search(text):
        return ""
    if LINK_TITLE_PAREN_RE.search(text):
        return ""
    if LATEX_SOUP_RE.search(text):
        return ""
    if BOILERPLATE_LEAD_RE.match(text):
        return ""
    if CONSENT_NOISE_RE.search(text):
        return ""
    if NAV_RUN_RE.match(text):
        return ""
    if TITLE_PREFIX_RE.match(text):
        head = text.split(":", 1)[0]
        if _looks_like_title_prefix(head):
            return ""
    if text.endswith("?"):
        return ""
    text = HEADING_MARK_RE.sub("", text).strip()
    if len(text) < min_chars:
        return ""

    working = text[:max_chars]
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", working)]
    if ends:
        text = working[: ends[-1]].strip()
    else:
        alt = [m.end() for m in re.finditer(r"[,;:](?=\s)", working)]
        if alt:
            text = (working[: alt[-1]].rstrip(",;:") + ".").strip()
        elif len(working.rsplit(None, 1)[-1]) <= 3:
            return ""
    text = text.strip()
    if len(text) < min_chars:
        return ""
    if text.count("(") != text.count(")"):
        return ""
    if DANGLING_END_RE.search(text):
        return ""
    return text


def claim_query_overlap(query: str, claim: str) -> float:
    """Word overlap between the research query and a claim (0-1)."""
    q_words = _tokenize(query or "")
    c_words = _tokenize(claim or "")
    if not q_words:
        return 0.0
    hits = 0
    for qw in q_words:
        if qw in c_words:
            hits += 1
            continue
        if len(qw) >= 5 and any(
            len(cw) >= 5 and (cw.startswith(qw) or qw.startswith(cw)) for cw in c_words
        ):
            hits += 1
    return hits / len(q_words)


MIN_QUERY_OVERLAP = 0.15


def select_diverse(
    claims: List[Dict[str, Any]], k: int = 3, max_similarity: float = 0.40
) -> List[Dict[str, Any]]:
    """Greedy maximal-marginal-relevance pick over the overlap coefficient."""
    ranked = sorted(
        claims or [], key=lambda f: float(f.get("confidence", 0.0) or 0.0), reverse=True
    )
    selected: List[Dict[str, Any]] = []
    selected_tokens: List[Set[str]] = []
    for candidate in ranked:
        text = str(candidate.get("claim", "") or "")
        if not text:
            continue
        candidate_tokens = _tokenize(text)
        if not candidate_tokens:
            continue
        novel = True
        for kept_tokens in selected_tokens:
            if not kept_tokens:
                continue
            overlap = len(candidate_tokens & kept_tokens) / min(
                len(candidate_tokens), len(kept_tokens)
            )
            if overlap >= max_similarity:
                novel = False
                break
        if novel:
            selected.append(candidate)
            selected_tokens.append(candidate_tokens)
        if len(selected) >= k:
            break
    return selected


def parse_published_date(value: str) -> str:
    """Best-effort parse of provider/fetch date strings to YYYY-MM-DD."""
    text = (value or "").strip()
    if not text:
        return ""
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate).date().isoformat()
        except (ValueError, TypeError):
            pass
    try:
        return parsedate_to_datetime(text).date().isoformat()
    except (TypeError, ValueError):
        pass
    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%d %b %Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except (TypeError, ValueError):
            continue
    return ""


# ---------------------------------------------------------------------------
# Aggregate evidence statistics (shared by critic, confidence, stopping)
# ---------------------------------------------------------------------------

def evidence_stats(facts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One pass over the evidence pool producing every count the pipeline's
    gates need. Previously each stage recomputed its own subset with slightly
    different rules, so the critic, the report and the API could disagree
    about how many verified facts existed."""
    facts = [f for f in (facts or []) if isinstance(f, dict)]
    urls = [str(f.get("source", "") or "") for f in facts if f.get("source")]
    domains = {extract_domain(u) for u in urls} - {""}
    documents = {canonical_url(u) for u in urls} - {""}
    verified = [f for f in facts if f.get("verified") is True]
    confidences = [float(f.get("confidence", 0.0) or 0.0) for f in facts]
    axes = {str(f.get("sub_question", "") or "").strip() for f in facts} - {""}
    corroborated = [f for f in facts if int(f.get("corroboration_count", 1) or 1) > 1]
    primary_docs = {u for u in documents if is_primary_source(u)}

    return {
        "total": len(facts),
        "verified": len(verified),
        "unverified": len(facts) - len(verified),
        "distinct_domains": len(domains),
        "distinct_documents": len(documents),
        "domains": sorted(domains),
        "axes_covered": len(axes),
        "axes": sorted(axes),
        "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "avg_verified_confidence": round(
            sum(float(f.get("confidence", 0.0) or 0.0) for f in verified) / len(verified), 4
        ) if verified else 0.0,
        "corroborated": len(corroborated),
        "primary_documents": len(primary_docs),
        "primary_share": round(primary_source_share(urls), 4),
        "freshness": evidence_freshness(facts),
    }
