"""Loader + validation for a versioned golden benchmark pair.

Pure and offline. Validates the STRUCTURE and internal consistency of a
golden query file and its thresholds file so a malformed/edited golden set
fails loudly at load time, not silently midway through a run.

Public surface:
    load_queries(path=...)      -> List[query dict]        (validated)
    load_thresholds(path=...)   -> dict                    (validated)
    validate_queries(data)      -> List[str]               (errors, [] = ok)
    validate_thresholds(data, queries) -> List[str]
    GoldenValidationError
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

GOLDEN_DIR = Path(__file__).resolve().parent
DEFAULT_QUERIES = GOLDEN_DIR / "queries_v1.json"
DEFAULT_THRESHOLDS = GOLDEN_DIR / "thresholds_v1.json"

KNOWN_CATEGORIES = frozenset({
    "factual_explanation", "current_trend", "comparison", "decision_policy",
    "ambiguous_term", "causal", "quantitative",
})
KNOWN_QUERY_TYPES = frozenset({"factual", "comparative", "analytical", "exploratory"})
KNOWN_SECTIONS = frozenset({
    "Executive Summary", "Key Findings", "Evidence Strength",
    "Limitations & Unknowns", "Counterarguments & Disputed Points",
})
_MIN_QUERIES, _MAX_QUERIES = 20, 40


class GoldenValidationError(ValueError):
    """Raised when a golden set is malformed or internally inconsistent."""


def validate_queries(data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(data, dict):
        return ["golden file is not a JSON object"]
    queries = data.get("queries")
    if not isinstance(queries, list):
        return ["'queries' must be a list"]
    if not (_MIN_QUERIES <= len(queries) <= _MAX_QUERIES):
        errors.append(
            f"query count {len(queries)} outside the required {_MIN_QUERIES}-{_MAX_QUERIES} range"
        )
    seen_ids: set = set()
    for i, q in enumerate(queries):
        label = f"queries[{i}]"
        if not isinstance(q, dict):
            errors.append(f"{label}: not an object")
            continue
        qid = q.get("id")
        if not qid or not isinstance(qid, str):
            errors.append(f"{label}: missing/invalid id")
        elif qid in seen_ids:
            errors.append(f"{label}: duplicate id {qid!r}")
        else:
            seen_ids.add(qid)
            label = qid
        if not str(q.get("query", "")).strip():
            errors.append(f"{label}: empty query")
        if q.get("category") not in KNOWN_CATEGORIES:
            errors.append(f"{label}: unknown category {q.get('category')!r}")
        if q.get("query_type_required") not in KNOWN_QUERY_TYPES:
            errors.append(f"{label}: unknown query_type_required {q.get('query_type_required')!r}")
        dims = q.get("required_dimensions")
        if not isinstance(dims, list) or not dims:
            errors.append(f"{label}: required_dimensions must be a non-empty list")
            dims = []
        hit = q.get("min_dimensions_hit")
        if not isinstance(hit, int) or hit < 0 or hit > len(dims):
            errors.append(f"{label}: min_dimensions_hit must be an int in [0, {len(dims)}]")
        if not isinstance(q.get("forbidden_domains"), list):
            errors.append(f"{label}: forbidden_domains must be a list")
        sections = q.get("required_sections")
        if not isinstance(sections, list) or not sections:
            errors.append(f"{label}: required_sections must be a non-empty list")
        else:
            for s in sections:
                if s not in KNOWN_SECTIONS:
                    errors.append(f"{label}: unknown required section {s!r}")
        inv = q.get("citation_invariants")
        if not isinstance(inv, dict):
            errors.append(f"{label}: citation_invariants must be an object")
        minimums = q.get("minimums")
        if not isinstance(minimums, dict):
            errors.append(f"{label}: minimums must be an object")
        else:
            for key in ("verified_claims", "corroborated_claims", "distinct_domains",
                        "primary_share", "sections"):
                if key not in minimums:
                    errors.append(f"{label}: minimums missing {key!r}")
    return errors


def validate_thresholds(data: Dict[str, Any], queries: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    if not isinstance(data, dict):
        return ["thresholds file is not a JSON object"]
    agg = data.get("aggregate")
    if not isinstance(agg, dict) or not agg:
        errors.append("thresholds: 'aggregate' must be a non-empty object")
    per_cat = data.get("per_category")
    if not isinstance(per_cat, dict) or not per_cat:
        errors.append("thresholds: 'per_category' must be a non-empty object")
    else:
        cats_in_queries = {q.get("category") for q in queries if isinstance(q, dict)}
        for cat in per_cat:
            if cat not in KNOWN_CATEGORIES:
                errors.append(f"thresholds: unknown category {cat!r}")
            if cat not in cats_in_queries:
                errors.append(f"thresholds: category {cat!r} has no queries")
    version = data.get("version")
    if not isinstance(version, str) or not version:
        errors.append("thresholds: missing version")
    return errors


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GoldenValidationError(f"golden file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GoldenValidationError(f"golden file is not valid JSON: {path}: {exc}") from exc


def load_queries(path: str | Path | None = None) -> List[Dict[str, Any]]:
    data = _read_json(Path(path) if path else DEFAULT_QUERIES)
    errors = validate_queries(data)
    if errors:
        raise GoldenValidationError(
            "invalid golden query set:\n  " + "\n  ".join(errors)
        )
    return data["queries"]


def load_thresholds(path: str | Path | None = None,
                    queries: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    if queries is None:
        queries = load_queries()
    data = _read_json(Path(path) if path else DEFAULT_THRESHOLDS)
    errors = validate_thresholds(data, queries)
    if errors:
        raise GoldenValidationError(
            "invalid golden thresholds:\n  " + "\n  ".join(errors)
        )
    return data
