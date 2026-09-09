"""Scenario Engine comparison logic (4.3): pure functions, no I/O.

A scenario differs from a plain rerun in two structural ways, both
represented here as data rather than prose:
  1. the assumption constrains the inquiry — scope_questions() derives
     assumption-scoped sub-questions from the base plan;
  2. the base run's verified claims are reused as shared context —
     scenario_new_claims() isolates each scenario's marginal contribution
     instead of re-crediting shared facts per scenario.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


def scope_questions(assumption: str, base_questions: List[str], limit: int = 4) -> List[str]:
    """Derive assumption-scoped sub-questions from the base plan.

    Deterministic and transparent: each scoped question names the
    assumption, so the scenario's delegation contract is auditable
    against the base plan it came from.
    """
    scoped = []
    for question in base_questions:
        if len(scoped) >= limit:
            break
        question = (question or "").strip()
        if not question:
            continue
        scoped.append(f"Given that {assumption}, {question}")
    return scoped


def build_scenario_query(
    base_query: str, assumption: str, scoped_questions: List[str], max_len: int = 500
) -> str:
    """Compose the scenario's pipeline query: assumption + base query +
    explicit scoped sub-questions constraining the planner.

    max_len mirrors the API's query limit — trailing scoped questions are
    dropped (never the assumption or base query) until the query fits, so
    the pipeline always receives a valid, assumption-led inquiry.
    """
    head = f"Under the assumption that {assumption}, {base_query.strip()}"
    numbered = [f"({i + 1}) {q}" for i, q in enumerate(scoped_questions)]
    while numbered and len(head) + len(" Address these scoped questions: " + "; ".join(numbered)) > max_len:
        numbered.pop()
    if numbered:
        return head + " Address these scoped questions: " + "; ".join(numbered)
    return head[:max_len]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def scenario_new_claims(scenario_claims: List[Dict[str, Any]], base_claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Scenario claims not already covered by base claims (word-overlap).

    A scenario claim counts as new when fewer than half of its significant
    words appear in any single base claim — restatements of shared facts
    stay credited to the base run.
    """
    base_sets = []
    for claim in base_claims:
        words = {w for w in _normalize(claim.get("claim", "")).split() if len(w) > 3}
        if words:
            base_sets.append(words)
    new_claims = []
    for claim in scenario_claims:
        words = {w for w in _normalize(claim.get("claim", "")).split() if len(w) > 3}
        if not words:
            continue
        if not any(len(words & base) >= len(words) / 2 for base in base_sets):
            new_claims.append(claim)
    return new_claims


def compare_scenarios(base: Dict[str, Any], scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Side-by-side comparison: confidence, findings, recommended option.

    Each scenario dict carries: id, assumption, confidence, claims (list),
    new_claims (list), recommended_option (or None). Returns table rows plus
    an explicit robust-vs-flipped verdict on the Decision Layer.
    """
    rows = []
    for scenario in scenarios:
        rows.append({
            "id": scenario.get("id", ""),
            "assumption": scenario.get("assumption", ""),
            "confidence": scenario.get("confidence"),
            "total_claims": len(scenario.get("claims") or []),
            "new_claims": len(scenario.get("new_claims") or []),
            "recommended_option": scenario.get("recommended_option"),
        })
    options = {r["recommended_option"] for r in rows if r["recommended_option"]}
    base_option = base.get("recommended_option")
    if not options:
        verdict = "no scenario produced a recommendation — nothing to compare."
    elif len(options) == 1:
        only = next(iter(options))
        verdict = f"robust across scenarios: Option {only} wins under every assumption."
        if base_option:
            if base_option == only:
                verdict += f" The base run agreed (Option {base_option})."
            else:
                verdict += (
                    f" Note: the base run recommended Option {base_option} — "
                    "the assumptions shift the conclusion."
                )
    else:
        by_option: Dict[str, List[str]] = {}
        for r in rows:
            by_option.setdefault(str(r["recommended_option"]), []).append(r["id"])
        flips = "; ".join(f"Option {opt} under {', '.join(ids)}" for opt, ids in sorted(by_option.items()))
        verdict = f"assumption flips the recommendation: {flips}."
    return {"base_option": base_option, "rows": rows, "verdict": verdict}
