from typing import Any, Dict, List

from app.core.llm import LLMClient


PLANNER_SYSTEM_PROMPT = """
You are the Planner Agent in a multi-agent research pipeline.
Return compact JSON only.
Generate 3 to 5 distinct, non-overlapping sub-questions optimized for web search.
""".strip()


async def planner_agent(llm: LLMClient, query: str, critique_feedback: str = "") -> List[str]:
    feedback_block = f"\nCritique feedback: {critique_feedback}" if critique_feedback else ""
    user_prompt = (
        "Create 3-5 concise sub-questions for this research query."
        " Avoid redundancy and make each question searchable."
        f"\n\nMain query: {query}{feedback_block}\n\n"
        "Return JSON: {\"sub_questions\": [\"...\"]}"
    )

    payload: Dict[str, Any] = await llm.generate_json(PLANNER_SYSTEM_PROMPT, user_prompt)
    questions = payload.get("sub_questions", []) if isinstance(payload, dict) else []

    cleaned: List[str] = []
    for item in questions:
        if isinstance(item, str):
            text = item.strip()
            if text and text.lower() not in {q.lower() for q in cleaned}:
                cleaned.append(text)

    return cleaned[:5] if cleaned else [query]
