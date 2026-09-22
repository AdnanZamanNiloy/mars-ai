"""Golden contradiction-resolution fixtures (v1).

The contradiction engine (`app.core.contradictions.find_contradictions`) and its
resolution pass (`app.core.contradiction_resolution.resolve_contradictions`) are
deterministic, LLM-free scorers with no offline regression gate of their own.
The full-graph golden evaluator exercises them only incidentally (the scripted
mocks rarely produce a genuine cross-source conflict), so a resolution
regression can ship green.

This module loads the LABELED fixture set
(`bench/golden/contradictions_v1.json`) and exposes it to the offline evaluator
(`bench/eval_contradictions.py`) and to pytest. Each case is a small, honest
claim pool plus the EXPECTED classification label, so the evaluator can score
detection precision/recall and resolution accuracy separately.

Labels (closed set, validated at load time):
    contradiction       genuine UNRESOLVED conflict (same unit+scope+period+
                        metric, materially different values)
    not_contradiction   the engine must not flag it
    resolved_explained  flagged as a spread, but resolution sets resolved=true
                        with an explanation (period/scope/metric difference)

Pure and offline. A malformed fixture file fails loudly at load time, mirroring
`bench.golden.loader`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

FIXTURE_PATH = Path(__file__).resolve().parent / "contradictions_v1.json"

LABELS = frozenset({"contradiction", "not_contradiction", "resolved_explained"})


class ContradictionFixtureError(ValueError):
    """Raised when the labeled fixture set is malformed."""


def _read() -> Dict[str, Any]:
    try:
        return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - shipped file
        raise ContradictionFixtureError(f"fixture file not found: {FIXTURE_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise ContradictionFixtureError(f"fixture file is not valid JSON: {exc}") from exc


def validate(data: Dict[str, Any]) -> List[str]:
    """Structural validation; returns errors ([] = ok)."""
    errors: List[str] = []
    if not isinstance(data, dict):
        return ["fixture file is not a JSON object"]
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        return ["'cases' must be a non-empty list"]
    seen: set = set()
    for i, case in enumerate(cases):
        label = f"cases[{i}]"
        if not isinstance(case, dict):
            errors.append(f"{label}: not an object")
            continue
        cid = case.get("id")
        if not cid or not isinstance(cid, str):
            errors.append(f"{label}: missing/invalid id")
        elif cid in seen:
            errors.append(f"{label}: duplicate id {cid!r}")
        else:
            seen.add(cid)
            label = cid
        if case.get("expected_label") not in LABELS:
            errors.append(f"{label}: unknown expected_label {case.get('expected_label')!r}")
        claims = case.get("claims")
        if not isinstance(claims, list) or len(claims) < 2:
            errors.append(f"{label}: needs at least two claims")
            continue
        for j, claim in enumerate(claims):
            if not isinstance(claim, dict):
                errors.append(f"{label}: claims[{j}] not an object")
                continue
            if not str(claim.get("claim", "")).strip():
                errors.append(f"{label}: claims[{j}] empty claim text")
            if not str(claim.get("source", "")).strip():
                errors.append(f"{label}: claims[{j}] empty source")
    return errors


def load_cases() -> List[Dict[str, Any]]:
    """Return the validated list of fixture cases."""
    data = _read()
    errors = validate(data)
    if errors:
        raise ContradictionFixtureError(
            "invalid contradiction fixture set:\n  " + "\n  ".join(errors)
        )
    return data["cases"]
