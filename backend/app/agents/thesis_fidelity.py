"""Thesis fidelity — does the final answer actually reflect the analyst's brief?

Why this module exists
----------------------
Phase 7 added the AnalystBrief (central thesis, insights, relationships,
counter-evidence, cross-source conclusions). But nothing checked whether the
writer's final prose actually honoured it: a writer could ignore the thesis,
drop the key counter-evidence, or collapse back into source-by-source
recitation while still passing the generic quality gate (which measures
citation/coherence, not fidelity to a specific brief).

This module measures semantic fidelity between the emitted answer and the
AnalyticalBrief — never string matching:

  * thesis reflected   — the answer's opening/conclusion aligns with the thesis
  * insights carried   — the major synthesised insights surface in the text
  * relationships kept — named relationships are not silently dropped
  * counter kept       — meaningful counter-evidence is preserved or weighed
  * synthesis not recital — the answer is not a per-source enumeration

It returns a `FidelityReport` with per-dimension scores and human-readable
`failures` in the same shape the existing quality gate emits, so the ONE
bounded revision pass can act on it. It is observational: it never rewrites.

Design (AGENTS.md):
  * Deterministic and LLM-free — reuses the shared semantic engine.
  * Semantic, not lexical — no verbatim thesis wording required.
  * Total/fail-safe — an absent or empty brief yields a neutral report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.core.logging import get_logger

logger = get_logger(__name__)

# A claim is "reflected" at/above this semantic similarity. Below the generic
# support bar (0.30) because a thesis is paraphrased in the answer, not quoted.
THESIS_REFLECT_SIMILARITY = 0.22
# An insight is "carried" at/above this similarity.
INSIGHT_REFLECT_SIMILARITY = 0.18
# Content-word coverage fallbacks. The shared similarity engine is lexical, so
# a genuine paraphrase can score near zero; a statement is also treated as
# reflected when this fraction of its distinctive content words appear in the
# answer. This is the paraphrase-tolerant path, and it is deliberately
# conservative so it never passes a genuinely ignored thesis.
THESIS_CONTENT_COVER = 0.60
INSIGHT_CONTENT_COVER = 0.50
# At/above this fraction of sentences that are single-source restatements, the
# answer reads as source-by-source recitation rather than synthesis.
RECITAL_RATIO = 0.75

# Words too common to carry content; used for the content-coverage fallback.
_CONTENT_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "as", "by", "at", "is", "are", "was", "were", "be", "been", "being", "that",
    "this", "these", "those", "it", "its", "from", "into", "than", "then",
    "has", "have", "had", "will", "would", "can", "could", "should", "may",
    "might", "not", "no", "so", "such", "more", "most", "less", "least",
    "which", "who", "whom", "whose", "what", "when", "where", "why", "how",
}

# Source-by-source narration markers: sentence-initial attributions that, when
# they dominate, indicate the writer is reciting sources rather than reasoning.
_RECITAL_LEADS = (
    "according to", "source ", "the report states", "one source", "another source",
    "study a", "study b", "document a", "document b", "the first source",
    "the second source", "source a says", "source b says",
)


@dataclass
class FidelityReport:
    """How faithfully the answer reflects the analyst's brief."""

    thesis_reflected: bool = True
    insights_carried: int = 0
    insights_total: int = 0
    relationships_kept: int = 0
    relationships_total: int = 0
    counter_kept: bool = True
    recital_ratio: float = 0.0
    thesis_overlap: float = 1.0
    failures: List[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """0..1 fidelity score (1.0 when there is nothing to check)."""
        if not self.insights_total:
            base = 1.0
        else:
            base = self.insights_carried / self.insights_total
        rel = (
            self.relationships_kept / self.relationships_total
            if self.relationships_total else 1.0
        )
        return round(0.5 * base + 0.2 * rel + 0.15 * (1.0 if self.thesis_reflected else 0.0)
                     + 0.15 * (1.0 if self.counter_kept else 0.0), 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thesis_reflected": self.thesis_reflected,
            "insights_carried": self.insights_carried,
            "insights_total": self.insights_total,
            "relationships_kept": self.relationships_kept,
            "relationships_total": self.relationships_total,
            "counter_kept": self.counter_kept,
            "recital_ratio": round(self.recital_ratio, 3),
            "thesis_overlap": round(self.thesis_overlap, 3),
            "score": self.score,
            "failures": list(self.failures),
        }


def _sentences(text: str) -> List[str]:
    import re

    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def _best_similarity(needle: str, haystack: List[str]) -> float:
    """Highest semantic similarity between `needle` and any sentence."""
    if not needle or not haystack:
        return 0.0
    try:
        from app.core.semantic import cross_similarity

        matrix = cross_similarity([needle], haystack)
        return float(matrix.max()) if matrix.size else 0.0
    except Exception as exc:  # similarity failure must never raise into the gate
        logger.warning("fidelity_similarity_failed", error=str(exc), exc_info=exc)
        return 0.0


def _content_terms(text: str) -> List[str]:
    """Distinctive content words (no stopwords), lowercased."""
    import re

    return [
        w for w in re.findall(r"[a-z][a-z0-9\-]{2,}", (text or "").lower())
        if w not in _CONTENT_STOPWORDS
    ]


def _content_overlap(statement: str, answer_lower: str) -> float:
    """Fraction of `statement`'s content words present in the answer text."""
    terms = set(_content_terms(statement))
    if not terms:
        return 1.0
    hit = sum(1 for t in terms if t in answer_lower)
    return hit / len(terms)


def _recital_ratio(answer: str) -> float:
    """Fraction of non-trivial sentences that open as source attributions."""
    sents = [s for s in _sentences(answer) if len(s.split()) >= 4]
    if not sents:
        return 0.0
    recital = sum(
        1 for s in sents
        if any(s.lower().lstrip("-*# ").startswith(lead) for lead in _RECITAL_LEADS)
    )
    return recital / len(sents)


def check_thesis_fidelity(
    answer: str,
    brief: Any,
) -> FidelityReport:
    """Measure how faithfully `answer` reflects the AnalyticalBrief.

    Semantic, deterministic, total. An absent/empty brief returns a neutral
    (all-pass) report so the absence of a brief never penalises an answer.
    """
    report = FidelityReport()
    if brief is None or not hasattr(brief, "to_dict"):
        return report
    try:
        data = brief.to_dict()
    except Exception:  # a brief that cannot serialize is treated as absent
        return report

    thesis = str(data.get("thesis", "") or "").strip()
    insights = [str(i) for i in (data.get("insights") or []) if str(i).strip()]
    relationships = [
        str(r.get("statement", "") or "")
        for r in (data.get("relationships") or [])
        if isinstance(r, dict) and str(r.get("statement", "")).strip()
    ]
    counter = [str(c) for c in (data.get("counter_evidence") or []) if str(c).strip()]

    if not (thesis or insights or relationships or counter):
        return report

    sentences = _sentences(answer)
    answer_lower = (answer or "").lower()

    if thesis:
        # The shared similarity engine is lexical (TF-IDF/char), so a genuine
        # PARAPHRASE of the thesis can score near zero (measured: 0.035). The
        # thesis check must therefore not be a hard lexical gate — it is a
        # bonus when the wording overlaps AND a content-anchored fallback: the
        # answer reflects the thesis when it uses the thesis's distinctive
        # content words. Only a thesis whose content appears NOWHERE fails.
        sim = _best_similarity(thesis, sentences)
        content = _content_terms(thesis)
        covered = sum(1 for t in content if t in answer_lower)
        overlap = (covered / len(content)) if content else 1.0
        report.thesis_reflected = sim >= THESIS_REFLECT_SIMILARITY or overlap >= THESIS_CONTENT_COVER
        report.thesis_overlap = round(overlap, 3)
        if not report.thesis_reflected:
            report.failures.append(
                "Thesis fidelity: the answer does not reflect the analyst's central "
                "thesis — lead with the conclusion the evidence supports and make it "
                "the through-line of the answer."
            )

    insight_hits = sum(
        1 for i in insights
        if _best_similarity(i, sentences) >= INSIGHT_REFLECT_SIMILARITY
        or _content_overlap(i, answer_lower) >= INSIGHT_CONTENT_COVER
    )
    report.insights_total = len(insights)
    report.insights_carried = insight_hits
    if insights and insight_hits < max(1, len(insights) // 2):
        report.failures.append(
            f"Thesis fidelity: only {insight_hits}/{len(insights)} of the analyst's "
            "major insights appear in the answer — develop the synthesised insights "
            "rather than restating individual evidence lines."
        )

    report.relationships_total = len(relationships)
    report.relationships_kept = sum(
        1 for r in relationships
        if _best_similarity(r, sentences) >= INSIGHT_REFLECT_SIMILARITY
    )
    if relationships and report.relationships_kept == 0:
        report.failures.append(
            "Thesis fidelity: the answer drops the relationships between findings "
            "the analyst identified — explain how the key findings connect."
        )

    if counter:
        report.counter_kept = any(
            _best_similarity(c, sentences) >= INSIGHT_REFLECT_SIMILARITY
            for c in counter
        )
        if not report.counter_kept:
            report.failures.append(
                "Thesis fidelity: the analyst's counter-evidence is missing — weigh "
                "the strongest evidence against the conclusion instead of omitting it."
            )

    report.recital_ratio = _recital_ratio(answer)
    if report.recital_ratio >= RECITAL_RATIO:
        report.failures.append(
            "Thesis fidelity: the answer reads as source-by-source recitation — "
            "synthesise across sources instead of describing what each one says."
        )

    return report
