from __future__ import annotations

from app.core.logging import get_logger
import re
from typing import Any, Dict, List

from app.agents.evidence_utils import dedupe_semantic_facts, extract_domain, filter_facts_by_domain
from app.core.degradation import record_fallback
from app.core.llm import LLMClient
from app.core.schemas import SynthesizerAnswerModel

logger = get_logger(__name__)


SYNTHESIZER_SYSTEM_PROMPT = """
You are the Synthesizer Agent in a research workflow.
Write a clean, coherent explanation for the user query using only the provided facts.

Rules:
- Start with a direct definition in 1-2 sentences (first paragraph).
- Follow with 3-5 short paragraphs, one idea each: how it works, key
  evidence and numbers, real-world examples, limitations and open questions.
  Skip angles the evidence does not support — never pad.
- Merge overlapping ideas and remove redundancy.
- Write for an informed reader: concrete, specific, no filler openers.
- CITE EVERY FACTUAL CLAIM: end each paragraph that states facts with the
  relevant source number(s) from the provided Sources list, like [1] or [1] [3].
  A paragraph with no citation marker reads as opinion — avoid that.
- Use ONLY the source numbers given. Never invent numbers, links, or sources.

Return valid JSON only in this schema:
{"answer": "<final synthesized explanation with [n] citations>"}
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
    numbered = _numbered_sources(top_facts)
    source_lines = "\n".join(f"[{s['n']}] {s['domain']}" + (f" ({s['url']})" if s["url"] else "")
                             for s in numbered)

    user_prompt = (
        f"Main query: {query}\n\n"
        f"Evidence facts: {top_facts}\n\n"
        f"Sources (cite these by number):\n{source_lines}\n\n"
        "Return JSON in this schema: "
        '{"answer": "<final synthesized explanation with [n] citations>"}'
    )

    try:
        payload = await llm.generate_json(
            SYNTHESIZER_SYSTEM_PROMPT,
            user_prompt,
            response_model=SynthesizerAnswerModel,
        )
    except Exception as exc:
        logger.warning("[Synthesizer] LLM call failed, using deterministic fallback", exc_info=exc)
        record_fallback("synthesizer")
        payload = {}

    answer = str(payload.get("answer", "")).strip() if isinstance(payload, dict) else ""
    if not answer:
        record_fallback("synthesizer")
    if answer:
        answer = _sanitize_answer_text(_validate_citations(answer, len(numbered)), query)
        return _append_source_legend(answer, numbered)

    # Deterministic fallback keeps output coherent if LLM JSON parsing fails.
    concept = _normalize_query_concept(query)
    definition = f"{concept} is a concept supported by reliable evidence and clear explanatory claims."
    body = " ".join([str(item.get("claim", "")).strip() for item in top_facts[:4] if item.get("claim")])
    if not body:
        return definition
    return _append_source_legend(_sanitize_answer_text(f"{definition}\n\n{body}".strip(), query), numbered)


def _numbered_sources(top_facts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Deterministic [n] legend built from evidence — the model cites
    numbers, never URLs, so markers always resolve to real sources."""
    numbered = []
    for i, fact in enumerate(top_facts, 1):
        url = str(fact.get("source", "") or "")
        numbered.append({
            "n": i,
            "domain": extract_domain(url) or url or "unknown source",
            "url": url,
        })
    return numbered


def _validate_citations(answer: str, count: int) -> str:
    """Strip [n] markers pointing outside the provided source list —
    the model must only cite what it was given."""

    def _keep(match: "re.Match") -> str:
        try:
            n = int(match.group(1))
        except (TypeError, ValueError):
            return ""
        return match.group(0) if 1 <= n <= count else ""

    return re.sub(r"\[(\d+)\]", _keep, answer or "")


def _append_source_legend(answer: str, numbered: List[Dict[str, str]]) -> str:
    lines = [f"[{s['n']}] {s['domain']}" + (f" — {s['url']}" if s["url"] else "")
             for s in numbered]
    return f"{answer.rstrip()}\n\nSources:\n" + "\n".join(lines)


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
    # Preserve paragraph structure: the model is instructed to write short
    # paragraphs, and flattening them into one block (as before) visibly
    # degrades readability. Only intra-line whitespace collapses.
    raw_lines = (answer or "").replace("\r\n", "\n").split("\n")
    paras: List[str] = []
    current: List[str] = []
    for line in raw_lines:
        cleaned = re.sub(r"\s+", " ", line).strip()
        cleaned = re.sub(r"^#+\s*", "", cleaned)  # keep heading text, drop markers
        if cleaned:
            current.append(cleaned)
        elif current:
            paras.append(" ".join(current))
            current = []
    if current:
        paras.append(" ".join(current))
    text = "\n\n".join(paras).strip()

    # Fix malformed opening pattern like: "what is X refers to ..."
    q = (query or "").strip().rstrip("?")
    q_escaped = re.escape(q)
    text = re.sub(
        rf"(?i)^{q_escaped}\s+refers to",
        f"{_normalize_query_concept(query)} refers to",
        text,
    )

    # Collapse immediate repeated clause: "X is ... X is ..."
    text = re.sub(r"(?i)(\b[A-Z][A-Za-z\s\-]{2,40}\s+is\b[^.]*\.)\s+\1", r"\1", text)

    return text.strip()
