from __future__ import annotations

import re
from typing import Any, Dict, List

from app.agents.evidence_utils import dedupe_semantic_facts, filter_facts_by_domain
from app.core.llm import LLMClient


SYNTHESIZER_SYSTEM_PROMPT = """
You are the Synthesizer Agent in a research workflow.
Write a clean, coherent explanation for the user query using only the provided facts.

Rules:
- Start with a direct definition in 1-2 sentences.
- Follow with short structured explanation (compact paragraphs).
- Merge overlapping ideas and remove redundancy.
- Be concise but informative.
- Do NOT include source links or citations inline.

Return valid JSON only in this schema:
{"answer": "<final synthesized explanation>"}
""".strip()
 
 
SYNTHESIZER_USER_TEMPLATE = """
Original query: {query}
Sub-questions from planner:
{questions}
 
Extracted claims to synthesize:
{claims}
""".strip()



async def synthesizer_agent(llm: LLMClient, query: str, facts: List[Dict[str, Any]]) -> str:
    usable_facts = dedupe_semantic_facts(filter_facts_by_domain(facts))
    if not usable_facts:
        return (
            f"{query} is an area that requires reliable evidence to explain accurately. "
            "Current retrieved evidence was too limited or low quality to produce a robust synthesis."
        )

    def _safe_conf(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    top_facts = sorted(usable_facts, key=lambda x: _safe_conf(x.get("confidence", 0.0)), reverse=True)[:10]

    user_prompt = (
        f"Main query: {query}\n\n"
        f"Evidence facts: {top_facts}\n\n"
        "Return JSON in this schema: "
        '{"answer": "<final synthesized explanation>"}'
    )

    try:
        payload = await llm.generate_json(SYNTHESIZER_SYSTEM_PROMPT, user_prompt)
    except Exception:
        payload = {}

    answer = str(payload.get("answer", "")).strip() if isinstance(payload, dict) else ""
    if answer:
        return _sanitize_answer_text(answer, query)

    # Deterministic fallback keeps output coherent if LLM JSON parsing fails.
    concept = _normalize_query_concept(query)
    definition = f"{concept} is a concept supported by reliable evidence and clear explanatory claims."
    body = " ".join([str(item.get("claim", "")).strip() for item in top_facts[:4] if item.get("claim")])
    if not body:
        return definition
    return _sanitize_answer_text(f"{definition}\n\n{body}".strip(), query)


def _normalize_query_concept(query: str) -> str:
    text = re.sub(r"\s+", " ", (query or "").strip()).strip(" ?.!")
    lower = text.lower()
    if lower.startswith("what is "):
        text = text[8:].strip()
    elif lower.startswith("define "):
        text = text[7:].strip()
    if text:
        return text[0].upper() + text[1:]
    return "This topic"


def _sanitize_answer_text(answer: str, query: str) -> str:
    text = re.sub(r"\s+", " ", (answer or "").strip())

    # Remove accidental markdown headings from model output.
    text = re.sub(r"^#+\s*", "", text)

    # Fix malformed opening pattern like: "what is X refers to ..."
    q = (query or "").strip().rstrip("?")
    q_escaped = re.escape(q)
    text = re.sub(
        rf"^(?i){q_escaped}\s+refers to",
        f"{_normalize_query_concept(query)} refers to",
        text,
    )

    # Collapse immediate repeated clause: "X is ... X is ..."
    text = re.sub(r"(?i)(\b[A-Z][A-Za-z\s\-]{2,40}\s+is\b[^.]*\.)\s+\1", r"\1", text)

    return text.strip()
