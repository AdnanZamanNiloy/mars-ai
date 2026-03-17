from typing import Any, Dict, List

from app.core.llm import LLMClient, clamp_confidence


SUMMARIZER_SYSTEM_PROMPT = """
You are the Summarizer Agent.
Convert raw search snippets into grounded factual claims with source citations.
Return valid JSON only.
""".strip()


async def summarizer_agent(
    llm: LLMClient,
    query: str,
    search_results: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    if not search_results:
        return []

    compact_results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("snippet", "")[:500],
            "sub_question": item.get("sub_question", ""),
        }
        for item in search_results[:16]
    ]

    user_prompt = (
        f"Research query: {query}\n\n"
        f"Search evidence: {compact_results}\n\n"
        "Extract key findings as JSON in this schema: "
        '{"facts": [{"claim": "...", "source": "https://...", "confidence": 0.0}]}'
        ". Keep claims concise and source-grounded."
    )

    try:
        payload = await llm.generate_json(SUMMARIZER_SYSTEM_PROMPT, user_prompt)
        facts = payload.get("facts", []) if isinstance(payload, dict) else []
    except Exception:
        facts = []

    cleaned: List[Dict[str, Any]] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        claim = str(fact.get("claim", "")).strip()
        source = str(fact.get("source", "")).strip()
        confidence = clamp_confidence(fact.get("confidence", 0.0))
        if claim and source:
            cleaned.append({"claim": claim, "source": source, "confidence": confidence})

    if cleaned:
        return cleaned

    # Heuristic fallback when the model output is malformed or empty.
    fallback: List[Dict[str, Any]] = []
    for item in search_results[:6]:
        snippet = item.get("snippet", "").strip()
        if not snippet:
            continue
        fallback.append(
            {
                "claim": snippet[:180],
                "source": item.get("url", ""),
                "confidence": 0.45,
            }
        )
    return fallback
