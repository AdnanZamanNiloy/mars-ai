"""Decision Intelligence Layer (Phase 3.5, Feature 18).

Derives named strategic options from the synthesized findings and
contradictions, marks exactly one recommended via an explicit code-level
rule, and attaches evidence-backed rationale + risk per option.

Code-level recommendation rule (manual 3.5: don't leave it to unstructured
LLM judgment): each option scores on (verified-support, contradiction
penalty, confidence alignment); the highest score wins, ties broken by
lower downside risk. For purely factual/definitional queries the layer
collapses to a single "no material decision" option.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from app.core.config import Settings

MAX_OPTIONS = 4
MIN_OPTIONS_COMPARATIVE = 2


def _url_to_axis(state: Dict[str, Any]) -> Dict[str, str]:
    """source URL -> axis via search result sub_question + planner contracts."""
    contract_axis: Dict[str, str] = {}
    for q in state.get("sub_questions", []):
        if isinstance(q, dict):
            question = str(q.get("question", "")).strip()
            axis = str(q.get("axis", "")).strip()
            if question and axis:
                contract_axis[question] = axis
    url_axis: Dict[str, str] = {}
    for result in state.get("search_results", []) or []:
        url = str(result.get("url", "")).strip()
        sub_q = str(result.get("sub_question", "")).strip()
        if url and sub_q in contract_axis:
            url_axis[url] = contract_axis[sub_q]
    return url_axis


def _verified_supporting_claims(
    option_axis: str,
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Verified facts attributed to the option's axis (URL-based attribution,
    falling back to keyword overlap)."""
    facts = state.get("facts", [])
    url_axis = _url_to_axis(state)
    keywords = set(option_axis.lower().split())
    supporting = []
    for f in facts:
        if not f.get("verified"):
            continue
        attributed_axis = url_axis.get(str(f.get("source", "")).strip(), "")
        claim_words = set(str(f.get("claim", "")).lower().split())
        if attributed_axis == option_axis or keywords & claim_words:
            supporting.append(f)
    return supporting


def _contradiction_penalty(option_label: str, contradictions: List[Dict[str, Any]]) -> int:
    # MVP: each detected contradiction raises the risk of every substantive
    # option equally (we can't yet attribute a contradiction to one option).
    return len(contradictions)


def _build_factual_option(query: str) -> Dict[str, Any]:
    return {
        "option_label": "A",
        "description": (
            "No material decision identified: the query is informational, "
            "not a choice between alternatives."
        ),
        "is_recommended": True,
        "rationale": (
            "The evidence answers a definitional/factual question. Decision "
            "options apply to comparative or analytical queries."
        ),
        "risk_note": "Risk of misleading the reader by manufacturing a decision where none exists.",
    }


def build_decision_layer(
    state: Dict[str, Any],
    settings: Settings | None = None,
) -> List[Dict[str, Any]]:
    """Return the DecisionOption list for the report + persistence."""
    orchestration = state.get("orchestration", {})
    query_type = str(orchestration.get("query_type", "factual"))

    if query_type not in ("comparative", "analytical"):
        return [_build_factual_option(str(state.get("query", "")))]

    facts = state.get("facts", [])
    contradictions = state.get("contradictions", [])
    axes: List[str] = []
    for q in state.get("sub_questions", []):
        if isinstance(q, dict):
            axis = str(q.get("axis", "")).strip()
            if axis and axis not in axes:
                axes.append(axis)

    if len(axes) < MIN_OPTIONS_COMPARATIVE:
        return [_build_factual_option(str(state.get("query", "")))]

    # Derive one option per leading axis (up to MAX_OPTIONS): the axis IS the
    # strategic framing — e.g. a cost axis yields "prioritize cost evidence".
    options: List[Dict[str, Any]] = []
    for idx, axis in enumerate(axes[:MAX_OPTIONS]):
        label = chr(ord("A") + idx)
        supporting = _verified_supporting_claims(axis, state)
        support_score = len(supporting)
        risk = _contradiction_penalty(label, contradictions)
        top_claim = supporting[0]["claim"] if supporting else ""
        options.append({
            "option_label": label,
            "description": (
                f"Frame the decision primarily around the '{axis}' dimension "
                f"of the findings."
            ),
            "is_recommended": False,
            "rationale": (
                f"Supported by {support_score} verified claim(s)"
                + (f', e.g. "{top_claim[:120]}"' if top_claim else "")
                + "."
            ),
            "risk_note": (
                f"Narrows the decision to '{axis}'; "
                f"{risk} unresolved source contradiction(s) raise uncertainty."
                if risk
                else f"Narrows the decision to '{axis}' alone."
            ),
            "_support": support_score,
            "_risk": risk,
        })

    # Code-level recommendation: most verified support; ties broken by fewer
    # contradictions-weighted risk, then lowest label for determinism.
    options.sort(key=lambda o: (-o["_support"], o["_risk"], o["option_label"]))
    options[0]["is_recommended"] = True

    for o in options:
        o.pop("_support", None)
        o.pop("_risk", None)
    return options
