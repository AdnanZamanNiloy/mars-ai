from typing import Any, Dict, List

import logging

from app.agents.evidence_utils import dedupe_semantic_facts, filter_facts_by_domain
from app.core.llm import LLMClient, clamp_confidence

logger = logging.getLogger(__name__)


CRITIC_SYSTEM_PROMPT = """
You are the Critic Agent in a multi-agent research pipeline.
You are the quality gate. Your decision to pass or loop determines
whether the Writer produces a complete, trustworthy report.

You must be demanding but fair:
  - Too strict → unnecessary loops, wasted API calls, slow output
  - Too lenient → shallow reports, missing perspectives, low confidence

━━━ EVALUATION CRITERIA ━━━

1) COMPLETENESS
   Would a knowledgeable reader consider the original query answered?
   Is any obvious major angle of the query entirely absent?

2) SUBSTANTIVE MATERIAL
   Is there enough reliable, non-redundant material to write a clear,
   well-supported answer (e.g. a definition with mechanisms and
   examples, not just fragments)?

3) SOURCE RELIABILITY
   Are the extracted facts backed by credible, authoritative sources?

4) REDUNDANCY / FRAGMENTATION
   Is the evidence mostly duplicated or too fragmented to synthesize?

━━━ DECISION RULES ━━━

  - is_sufficient = true  ONLY if the evidence clearly supports a
    complete answer for the original query.
  - If insufficient, improved_queries must contain 1-3 specific,
    search-ready strings targeting the exact gaps — never vague
    suggestions like "search for more information about X".
      BAD  → "Find more about knowledge types"
      GOOD → "metacognitive knowledge definition examples learning research"
  - Be conservative: when in doubt, mark insufficient and say what
    is missing in the reason field.

━━━ OUTPUT FORMAT ━━━

Return ONLY valid JSON. No markdown fences. No text outside JSON.

{
  "is_sufficient": <true|false>,
  "reason": "<one or two sentences: what is complete or what is missing>",
  "improved_queries": ["<targeted search query to fix gap 1>", "..."],
  "confidence": <0.0 to 1.0, your confidence in the current evidence>
}
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
    except Exception as exc:
        logger.warning("[Critic] LLM call failed, treating as insufficient", exc_info=exc)
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
