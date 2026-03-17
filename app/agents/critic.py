from typing import Any, Dict, List

from app.agents.evidence_utils import dedupe_semantic_facts, filter_facts_by_domain
from app.core.llm import LLMClient, clamp_confidence


CRITIC_SYSTEM_PROMPT = """
You are the Critic Agent in a multi-agent research pipeline.
You are the quality gate. Your decision to pass or loop determines
whether the Writer produces a complete, trustworthy report.
 
You must be demanding but fair:
  - Too strict → unnecessary loops, wasted API calls, slow output
  - Too lenient → shallow reports, missing perspectives, low confidence
 
━━━ EVALUATION CHECKLIST ━━━
 
Run all 5 checks. Count how many fail.
 
CHECK 1 — COVERAGE (weight: high)
  Is overall_coverage from the Synthesizer ≥ 0.70?
  Are there themes with coverage < 0.55?
  FAIL if: overall_coverage < 0.70 OR any theme coverage < 0.55
 
CHECK 2 — CONFIDENCE (weight: high)
  Compute average confidence across all facts in all themes.
  FAIL if: average confidence < 0.65
 
CHECK 3 — AXIS BALANCE (weight: medium)
  Does the synthesized knowledge cover at least 2 distinct axes?
  Is any single axis dominating (> 60% of all facts)?
  FAIL if: fewer than 2 axes covered OR one axis has > 60% of facts
 
CHECK 4 — CONTRADICTION RESOLUTION (weight: medium)
  Are there themes with has_conflict = true?
  If yes — are the conflicts acknowledged (not silently ignored)?
  FAIL if: has_conflict = true AND no conflict_note provided
 
CHECK 5 — QUERY COMPLETENESS (weight: low)
  Would a knowledgeable reader consider the query answered?
  Are there obvious sub-questions the plan missed?
  FAIL if: an obvious major angle of the original query is entirely absent
 
━━━ DECISION RULES ━━━
 
  0 checks fail    → passed = true  (excellent research)
  1 check fails    → passed = true  (acceptable, note the gap)
  2 checks fail    → passed = false (loop with targeted queries)
  3+ checks fail   → passed = false (loop with full re-plan signal)
 
  HARD RULE: Never loop more than MAX_ITERATIONS times total.
  If iteration >= MAX_ITERATIONS: passed = true regardless of checks.
  Write a limitations_note explaining what remains uncovered.
 
━━━ WHEN LOOPING: WRITE TARGETED QUERIES ━━━
 
Do NOT return vague suggestions like "search for more information about X".
Each suggested_query must be a specific, search-ready string targeting
the exact gap identified — same format as the Planner's sub-questions.
 
  BAD  → "Find more about knowledge types"
  GOOD → "metacognitive knowledge definition examples learning research"
 
━━━ OUTPUT FORMAT ━━━
 
Return ONLY valid JSON. No markdown fences. No text outside JSON.
 
{
  "passed": <true|false>,
  "iteration": <current iteration number>,
  "checks": {
    "coverage_pass":      <true|false>,
    "confidence_pass":    <true|false>,
    "axis_balance_pass":  <true|false>,
    "contradiction_pass": <true|false>,
    "completeness_pass":  <true|false>
  },
  "failed_count": <0–5>,
  "issues": [
    "<specific issue 1>",
    "<specific issue 2>"
  ],
  "suggested_queries": [
    "<targeted search query to fix gap 1>",
    "<targeted search query to fix gap 2>"
  ],
  "confidence_score": <0.0–1.0>,
  "limitations_note": "<empty string if passed cleanly, else note remaining gaps>"
}
""".strip()
 
 
CRITIC_USER_TEMPLATE = """
Original query: {query}
Current iteration: {iteration} of {max_iterations}
Query type: {query_type}
 
Synthesized knowledge:
{synthesized}
""".strip()
 


async def critic_agent(
    llm: LLMClient,
    query: str,
    facts: List[Dict[str, Any]],
    iteration: int,
    max_iterations: int,
) -> Dict[str, Any]:
    quality_facts = dedupe_semantic_facts(filter_facts_by_domain(facts))
    if not quality_facts:
        return {
            "is_sufficient": False,
            "reason": "No reliable evidence extracted.",
            "improved_queries": [f"Latest evidence for: {query}", f"Key statistics: {query}"],
            "confidence": 0.2,
        }

    user_prompt = (
        f"Main query: {query}\n"
        f"Current iteration: {iteration}/{max_iterations}\n"
        f"Extracted reliable facts: {quality_facts[:10]}\n\n"
        "Evaluate using these criteria:\n"
        "1) Is the answer complete?\n"
        "2) Is there enough material to write a clear definition?\n"
        "3) Are sources reliable?\n"
        "4) Is information redundant or fragmented?\n\n"
        "Return JSON: "
        '{"is_sufficient": true/false, "reason": "...", "improved_queries": ["..."], "confidence": 0.0}'
    )

    try:
        payload = await llm.generate_json(CRITIC_SYSTEM_PROMPT, user_prompt)
    except Exception:
        payload = {}

    is_sufficient = bool(payload.get("is_sufficient", False)) if isinstance(payload, dict) else False
    reason = str(payload.get("reason", "Insufficient assessment.")) if isinstance(payload, dict) else "Insufficient assessment."
    improved_queries = payload.get("improved_queries", []) if isinstance(payload, dict) else []
    confidence = clamp_confidence(payload.get("confidence", 0.4) if isinstance(payload, dict) else 0.4)

    cleaned_queries: List[str] = []
    for item in improved_queries:
        if isinstance(item, str) and item.strip():
            cleaned_queries.append(item.strip())

    reason_lower = reason.lower()
    if any(token in reason_lower for token in ["incomplete", "lacks a clear definition", "insufficient"]):
        is_sufficient = False

    # Deterministic guardrails improve consistency when model judgments are noisy.
    has_definition = any(" is " in str(f.get("claim", "")).lower() for f in quality_facts[:5])
    avg_fact_conf = sum(float(f.get("confidence", 0.0) or 0.0) for f in quality_facts) / max(1, len(quality_facts))
    min_facts_required = 4

    if len(quality_facts) < min_facts_required or not has_definition or avg_fact_conf < 0.74:
        is_sufficient = False
        confidence = min(confidence, 0.58)
        if not cleaned_queries:
            cleaned_queries = [
                f"Authoritative definition of: {query}",
                f"Peer-reviewed or encyclopedia explanation of: {query}",
            ]
        reason = (
            "Evidence is still incomplete for a high-quality synthesis; "
            "the answer lacks enough reliable, non-redundant coverage or a clear definition."
        )
    elif avg_fact_conf >= 0.80:
        is_sufficient = True
        confidence = max(confidence, 0.78)

    return {
        "is_sufficient": is_sufficient,
        "reason": reason,
        "improved_queries": cleaned_queries[:5],
        "confidence": confidence,
    }
