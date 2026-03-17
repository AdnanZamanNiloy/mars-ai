from typing import Any, Dict, List

from app.core.llm import LLMClient, clamp_confidence


CRITIC_SYSTEM_PROMPT = """
You are the Critic Agent in a research workflow.
Assess if findings are complete and relevant.
If weak, propose improved search directions.
Return JSON only.
""".strip()


async def critic_agent(
    llm: LLMClient,
    query: str,
    facts: List[Dict[str, Any]],
    iteration: int,
    max_iterations: int,
) -> Dict[str, Any]:
    if not facts:
        return {
            "is_sufficient": False,
            "reason": "No evidence extracted.",
            "improved_queries": [f"Latest evidence for: {query}", f"Key statistics: {query}"],
            "confidence": 0.2,
        }

    user_prompt = (
        f"Main query: {query}\n"
        f"Current iteration: {iteration}/{max_iterations}\n"
        f"Extracted facts: {facts[:10]}\n\n"
        "Evaluate completeness and relevance. Return JSON: "
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

    if iteration >= max_iterations:
        is_sufficient = True
        reason = f"Reached max iterations ({max_iterations})."

    return {
        "is_sufficient": is_sufficient,
        "reason": reason,
        "improved_queries": cleaned_queries[:5],
        "confidence": confidence,
    }
