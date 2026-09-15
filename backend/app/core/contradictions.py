"""Contradiction Engine v2 (Phase 3.2, Feature 09).

After dedupe_semantic_facts collapses near-identical claims, this pass flags
claims that are TOPICALLY similar but materially different. v2 upgrades the
original numeric-only scan with three detectors, one output shape:

* numeric   — like-united quantities that diverge beyond tolerance
              (upgraded from raw number-bag matching to the unit-aware
              `numeric_conflict`: a "%" vs a "gw" is no longer a conflict,
              and years no longer masquerade as values)
* polarity  — same subject, opposite assertion ("outperforms" vs "does not
              outperforms") — the failure mode lexical overlap is blind to
* temporal  — same measure reported for different periods ("as of 2023: 5
              billion" vs "as of 2024: 8 billion") — not a lie, but a
              staleness hazard the report must surface instead of silently
              averaging

Similarity comes from the shared semantic engine as ONE batch matrix; the
similarity band (same topic, not same claim) is unchanged. Severity ranks
what the critic gates on and what the report shows first.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from app.agents.evidence_utils import (
    claim_polarity,
    numeric_conflict,
)
from app.core.semantic import similarity_matrix

# Just below the dedup merge threshold (0.86): same topic, not same claim.
SIMILARITY_LOW = 0.50
SIMILARITY_HIGH = 0.86

# Polarity conflicts need enough topical overlap to be about the same thing.
# 0.45 admits antonym-swap pairs ("contracted" vs "expanded") whose surface
# difference alone drags hybrid similarity just under the numeric band floor.
POLARITY_MIN_SIMILARITY = 0.45

# Relative numeric difference that counts as a genuine conflict.
SIGNIFICANT_DIFF = 0.05

# Temporal: values for different periods differing at least this much.
TEMPORAL_DIVERGENCE = 0.20

MAX_CONTRADICTIONS = 5

# Severity bands (0-1): numeric > polarity > temporal, divergence-scaled.
SEVERE_SEVERITY = 0.60

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Scope markers: a numeric difference between a "global" figure and a "US"
# figure (or any two disjoint geographies/segments) is a scope mismatch, not a
# contradiction. Deliberately a curated token set with word boundaries — the
# ambiguous bare pronoun "us" is excluded so ordinary prose never triggers a
# false scope separation.
_SCOPE_MARKERS: Dict[str, str] = {
    "global": "global", "globally": "global", "worldwide": "global",
    "world": "global", "international": "global",
    "u.s.": "us", "u.s": "us", "usa": "us", "united states": "us",
    "uk": "uk", "britain": "uk", "british": "uk", "united kingdom": "uk",
    "eu": "eu", "europe": "eu", "european": "eu",
    "china": "china", "chinese": "china",
    "india": "india", "indian": "india",
    "japan": "japan", "japanese": "japan",
    "germany": "germany", "german": "germany",
    "africa": "africa", "african": "africa",
    "asia": "asia", "asian": "asia",
    "canada": "canada", "canadian": "canada",
    "australia": "australia", "australian": "australia",
}

_SCOPE_PHRASE_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_SCOPE_MARKERS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


_NUMBER_RE = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<suffix>%|percent|thousand|million|billion|bn|k\b|m\b|b\b)?",
    re.IGNORECASE,
)

_SUFFIX_SCALE = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "million": 1e6,
    "bn": 1e9, "b": 1e9, "billion": 1e9,
    "%": None, "percent": None,  # percent keeps its own scale
}


def extract_numbers(text: str) -> List[float]:
    """Extract scale-normalized numbers (legacy contract: years included —
    this feeds the display lists; the CONFLICT detectors use the unit-aware
    `numeric_conflict`, which excludes years properly)."""
    values: List[float] = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group("value").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        suffix = (match.group("suffix") or "").lower()
        scale = _SUFFIX_SCALE.get(suffix, 1.0)
        if scale is not None:
            value *= scale
        values.append(value)
    return values


def _years_in(text: str) -> List[int]:
    return [int(m.group(0)) for m in _YEAR_RE.finditer(text or "")]


def _scopes_in(text: str) -> set:
    """Scope labels a claim asserts (global / us / eu / ...)."""
    scopes = set()
    for match in _SCOPE_PHRASE_RE.finditer(text or ""):
        key = match.group(0).lower().rstrip(".")
        if key in _SCOPE_MARKERS:
            scopes.add(_SCOPE_MARKERS[key])
    return scopes


def _scopes_conflict(a: set, b: set) -> bool:
    """True when the two claims describe different scopes.

    Disjoint explicit scopes ("global" vs "us") conflict; a scoped claim
    against an unscoped one ("US market" vs "the market") also conflicts, since
    the unscoped figure may be a different population entirely. Conservative by
    design: a false separation only downgrades a numeric conflict to a
    scope note, never invents one.
    """
    if not a and not b:
        return False
    if not a or not b:
        return True
    return not (a & b)


def _severity(kind: str, divergence: float) -> float:
    if kind == "numeric":
        return min(1.0, 0.35 + divergence)
    if kind == "polarity":
        return 0.55
    if kind == "scope":
        return min(0.45, 0.20 + divergence / 4.0)
    return min(0.5, 0.25 + divergence / 2.0)


def find_contradictions(
    facts: List[Dict[str, Any]], max_results: int = MAX_CONTRADICTIONS
) -> List[Dict[str, Any]]:
    """Scan for numeric, polarity and temporal conflicts between
    topically-similar claims from different sources.

    Returns at most `max_results` entries, sorted by severity (numeric
    conflicts with large divergence first). Existing consumers read the
    legacy keys (claim_a/source_a/value_a/...); `kind`, `severity` and
    unit-aware `values` are additive.
    """
    cleaned = [
        f for f in facts or []
        if isinstance(f, dict)
        and str(f.get("claim", "")).strip()
        and str(f.get("source", "")).strip()
    ]
    if len(cleaned) < 2:
        return []

    claims = [str(f.get("claim", "")) for f in cleaned]
    sim = similarity_matrix(claims)
    polarities = [claim_polarity(c) for c in claims]

    contradictions: List[Dict[str, Any]] = []

    def _claim_key(text: str) -> str:
        return re.sub(r"\W+", " ", text.lower()).strip()

    def _already_flagged(a: str, b: str) -> bool:
        return any(
            _claim_key(a) == _claim_key(existing.get("claim_a", ""))
            and _claim_key(b) == _claim_key(existing.get("claim_b", ""))
            for existing in contradictions
        )

    for i in range(len(cleaned)):
        for j in range(i + 1, len(cleaned)):
            a, b = cleaned[i], cleaned[j]
            if str(a.get("source", "")).strip() == str(b.get("source", "")).strip():
                continue  # same source restating itself isn't a contradiction

            similarity = float(sim[i, j])
            claim_a, claim_b = claims[i], claims[j]
            pa, pb = polarities[i], polarities[j]
            found: Dict[str, Any] | None = None

            # ---- polarity (checked FIRST, outside the similarity band) ----
            # An antonym swap ("X" vs "not X") scores ~0.90 — near-duplicate
            # territory the numeric band gate would reject, and below the
            # band floor when only the verb differs ("contracted" vs
            # "expanded"). Opposite polarity is the strongest conflict signal
            # there is, so it needs only a topical floor.
            if pa != 0 and pb != 0 and pa != pb and similarity >= POLARITY_MIN_SIMILARITY:
                found = {
                    "kind": "polarity",
                    "severity": _severity("polarity", 0.0),
                    "values": {"polarity_a": pa, "polarity_b": pb},
                    "note": (
                        "claims make opposite assertions about the same subject "
                        "(one affirms what the other negates)"
                    ),
                }

            # ---- temporal (period mismatch explains the value spread) -----
            # Requires the same topical floor as the numeric band. Without it,
            # ANY two claims carrying different 4-digit years and a shared-unit
            # number were reported as a period conflict: a live run paired a
            # Rooppur cost claim with "38 countries endorsed the tripling
            # declaration" and "100 reactors in the US" (topical similarity
            # 0.02-0.25) purely because both contained a year. Real temporal
            # pairs — same measure, different reporting years — score 0.51+.
            if found is None and similarity >= SIMILARITY_LOW:
                years_a, years_b = _years_in(claim_a), _years_in(claim_b)
                if years_a and years_b and set(years_a) != set(years_b):
                    temporal = numeric_conflict(
                        claim_a, claim_b, divergence=TEMPORAL_DIVERGENCE
                    )
                    if temporal is not None:
                        found = {
                            "kind": "temporal",
                            "severity": _severity(
                                "temporal", temporal["relative_divergence"]
                            ),
                            "values": {
                                "unit": temporal["unit"],
                                "value_a": temporal["value_a"],
                                "value_b": temporal["value_b"],
                                "relative_divergence": temporal["relative_divergence"],
                                "years_a": years_a[:2],
                                "years_b": years_b[:2],
                            },
                            "note": (
                                "same measure reported for different periods — "
                                "check which period each figure describes before "
                                "citing either as current"
                            ),
                        }

            # ---- numeric (unit-aware, inside the topical band) ------------
            # Normalization guard: a numeric conflict is only real when the two
            # figures are unit-, scope- AND period-compatible. Units are already
            # enforced by `numeric_conflict`; periods by the temporal detector
            # above; here a scope mismatch downgrades the finding to a scope
            # note instead of a numeric contradiction (a "global" figure and a
            # "US" figure diverging is not the sources disagreeing).
            if found is None:
                if SIMILARITY_LOW <= similarity < SIMILARITY_HIGH:
                    conflict = numeric_conflict(claim_a, claim_b, divergence=SIGNIFICANT_DIFF)
                    if conflict is not None:
                        scopes_a, scopes_b = _scopes_in(claim_a), _scopes_in(claim_b)
                        if _scopes_conflict(scopes_a, scopes_b):
                            found = {
                                "kind": "scope",
                                "severity": _severity("scope", conflict["relative_divergence"]),
                                "values": {
                                    "unit": conflict["unit"],
                                    "value_a": conflict["value_a"],
                                    "value_b": conflict["value_b"],
                                    "relative_divergence": conflict["relative_divergence"],
                                    "scopes_a": sorted(scopes_a),
                                    "scopes_b": sorted(scopes_b),
                                },
                                "note": (
                                    "figures describe different scopes "
                                    f"({sorted(scopes_a) or ['unspecified']} vs "
                                    f"{sorted(scopes_b) or ['unspecified']}) — not a "
                                    "contradiction; compare like-for-like before citing"
                                ),
                            }
                        else:
                            found = {
                                "kind": "numeric",
                                "severity": _severity("numeric", conflict["relative_divergence"]),
                                "values": {
                                    "unit": conflict["unit"],
                                    "value_a": conflict["value_a"],
                                    "value_b": conflict["value_b"],
                                    "relative_divergence": conflict["relative_divergence"],
                                },
                                "note": (
                                    f"topically similar claims cite significantly different "
                                    f"{conflict['unit'] or 'dimensionless'} figures "
                                    f"({conflict['raw_a']} vs {conflict['raw_b']})"
                                ),
                            }

            if found is None or _already_flagged(claim_a, claim_b):
                continue

            contradictions.append({
                "topic_similarity": round(similarity, 3),
                "claim_a": claim_a,
                "source_a": str(a.get("source", "")),
                "value_a": extract_numbers(claim_a)[:3],
                "claim_b": claim_b,
                "source_b": str(b.get("source", "")),
                "value_b": extract_numbers(claim_b)[:3],
                "note": found["note"],
                "kind": found["kind"],
                "severity": round(found["severity"], 3),
                "values": found["values"],
            })
            if len(contradictions) >= max(1, max_results):
                break
        if len(contradictions) >= max(1, max_results):
            break

    contradictions.sort(key=lambda c: -float(c.get("severity", 0.0)))
    return contradictions
