"""Direct Answer Agent — answer a stable-knowledge question without research.

Why this module exists
----------------------
The query router (app/agents/router.py) may decide a question needs no
external evidence. When it does, this agent produces the answer in a single
LLM call instead of the full research pipeline.

The critical safety property is the ESCAPE HATCH: the answering model is
asked to answer AND to self-assess. If it reports `needs_research=true`, or
if it is not confident, the caller falls through to the research pipeline
rather than shipping an ungrounded answer. A wrong routing decision
therefore costs one LLM call, never a wrong answer.

Design constraints
------------------
* Standard agent shape (AGENTS.md 4.7): LLM path with schema validation and
  a deterministic fallback. The fallback ALWAYS sets needs_research=True —
  a degraded run must research, never answer from a template.
* A direct answer is not evidence-graded. Its confidence is the model's own
  self-assessment, and the caller must cap it below the research
  sufficiency threshold (a direct answer must never be mistakable for a
  researched one).
* No markdown report scaffolding is invented here: the answer is returned
  verbatim and the caller wraps it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.core.degradation import record_fallback
from app.core.llm import LLMClient
from app.core.logging import get_logger
from app.core.schemas import DirectAnswerModel

logger = get_logger(__name__)

# Below this self-assessed confidence the direct answer is refused and the
# query is researched instead, even when the model set needs_research=false.
MIN_ANSWER_CONFIDENCE = 0.70

DIRECT_ANSWER_SYSTEM_PROMPT = """
You are the Direct Answer Agent in a multi-agent research system. The Query
Router has decided this question does NOT require web research. Your job is
to answer it correctly and completely from stable general knowledge — OR to
refuse and request research when you cannot.

━━━ WHEN TO ANSWER ━━━

Answer when the question is stable general knowledge: definitions,
concepts, established facts, how something works, math, code, standard
procedures. This knowledge does not change over time.

━━━ WHEN TO REFUSE (needs_research = true) ━━━

Refuse — even though the router thought otherwise — whenever ANY of these
is true:

- The answer depends on the present: prices, current events, recent
  releases, live data, who currently holds a position. Your training has a
  cutoff and you cannot know today's facts.
- The question needs a specific number, statistic, date or quote that must
  be sourced to be trusted.
- You are not genuinely confident the answer you can give is correct AND
  complete.
- The topic is health, safety, legal or financial and getting it wrong
  could cause harm.
- The question is ambiguous and you are unsure which meaning is intended.

Refusing is CORRECT and never penalized. A confident-sounding wrong answer
is the only failure mode that matters. When in doubt, set
needs_research=true.

━━━ ANSWER STYLE ━━━

When you answer: be direct, correct and complete. Lead with the answer.
Use plain language suited to the requested explanation level. Do not invent
citations, URLs or statistics. No markdown headings are required.

━━━ OUTPUT FORMAT ━━━

Return ONLY valid JSON. No markdown fences. No text outside JSON.

{
  "answer": "<the complete answer, or empty when refusing>",
  "needs_research": <true|false>,
  "confidence": <0.0-1.0, your confidence the answer is correct and complete>,
  "reason": "<one sentence: why you answered or refused>"
}
""".strip()


@dataclass
class DirectAnswer:
    """The agent's verdict: either a confident answer, or a refusal."""

    answer: str
    needs_research: bool
    confidence: float
    reason: str
    origin: str = "llm"          # llm | fallback
    signals: Dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """An answer may be delivered only when the model cleared it AND was
        confident enough. Any other combination means research."""
        return (
            not self.needs_research
            and bool(self.answer.strip())
            and self.confidence >= MIN_ANSWER_CONFIDENCE
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "needs_research": self.needs_research,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "origin": self.origin,
        }


def _refusal(reason: str) -> DirectAnswer:
    return DirectAnswer(
        answer="",
        needs_research=True,
        confidence=0.0,
        reason=reason,
        origin="fallback",
    )


def _coerce_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _finalize(payload: Dict[str, Any]) -> DirectAnswer:
    """Validate the parsed payload into a DirectAnswer.

    A payload that requests research, or carries no answer, is a refusal —
    this is the escape hatch and must always be honoured.
    """
    if not isinstance(payload, dict):
        return _refusal("No usable model output.")
    answer = str(payload.get("answer", "") or "").strip()
    needs_research = bool(payload.get("needs_research", True))
    confidence = _coerce_confidence(payload.get("confidence"))
    reason = str(payload.get("reason", "") or "").strip()
    if needs_research or not answer:
        return DirectAnswer(
            answer="", needs_research=True,
            confidence=confidence, reason=reason or "Model requested research.",
        )
    return DirectAnswer(
        answer=answer, needs_research=False,
        confidence=confidence, reason=reason,
    )


async def direct_answer_agent(
    llm: LLMClient,
    query: str,
    intent: Optional[Dict[str, Any]] = None,
) -> DirectAnswer:
    """Answer `query` directly, or refuse and request research.

    Standard agent shape (AGENTS.md 4.7): the fallback is a refusal, so any
    failure (provider error, invalid output) routes the query into the full
    research pipeline — never to an ungrounded answer.
    """
    intent = intent or {}
    level = str(intent.get("explanation_level", "practical") or "practical")
    domain = str(intent.get("domain", "general") or "general")
    user_prompt = (
        f"Question: {query}\n"
        f"Requested explanation level: {level}\n"
        f"Domain: {domain}\n\n"
        "Answer from stable general knowledge, or refuse and request "
        "research. Return JSON only."
    )

    try:
        payload = await llm.generate_json(
            DIRECT_ANSWER_SYSTEM_PROMPT,
            user_prompt,
            response_model=DirectAnswerModel,
        )
    except Exception as exc:
        logger.warning("[DirectAnswer] LLM call failed, requesting research", exc_info=exc)
        record_fallback("direct_answer")
        return _refusal("Direct answer LLM call failed.")

    if not isinstance(payload, dict) or not payload:
        logger.warning("[DirectAnswer] empty/invalid LLM output, requesting research")
        record_fallback("direct_answer")
        return _refusal("Empty direct answer output.")

    result = _finalize(payload)
    if result.needs_research:
        logger.info("[DirectAnswer] refused -> research: %s", result.reason)
    else:
        logger.info(
            "[DirectAnswer] answered (confidence=%.2f): %s",
            result.confidence, result.reason,
        )
    return result
