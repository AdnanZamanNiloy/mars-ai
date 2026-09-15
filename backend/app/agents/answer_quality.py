"""Answer Quality Optimizer — score the finished answer before it ships.

The last unguarded handoff in the pipeline was synthesizer -> user: a report
could be perfectly verified and confidently scored while answering the wrong
question, dumping statistics, or burying the explanation. This module is the
pre-delivery gate the vision demands ("is this actually answering the user
question?").

Design
------
* Deterministic and LLM-free: every input is already measured by the pipeline
  (answer-support verdicts, verification flags, source registry, citation
  health). A quality judge that costs a model call is a judge you consult
  less often — and one that can be overruled by its own model.
* Five public subscores on the 0-100 scale the product promises:
  Accuracy, Relevance, Evidence, Clarity, Reasoning.
* `failures` are written FOR the re-synthesis prompt: specific, actionable
  sentences ("the Executive Summary does not mention ..."), not scores.
* The gate allows exactly ONE bounded re-synthesis with the failures fed
  back (never a loop — budget rule), and the better of the two drafts wins.

Known limits, stated honestly: the scorer is lexical/structural, so a
fluent-but-wrong answer that cites verified evidence can still pass on
accuracy — the accuracy signal is only as good as verification + answer
support, which is the same trust the rest of the pipeline rests on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.agents.sources import primary_source_share, strip_machine_sections
from app.core.logging import get_logger

logger = get_logger(__name__)

# Relevance below this fails regardless of overall — a confident answer to the
# wrong question is the failure mode this gate exists for.
RELEVANCE_FLOOR = 40
ACCURACY_FLOOR = 40

_QUERY_STOP = {
    "what", "who", "when", "where", "which", "does", "did", "is", "are", "was",
    "the", "a", "an", "of", "for", "to", "and", "or", "in", "on", "with",
    "about", "should", "would", "could", "can", "tell", "me", "explain",
    "define", "between", "from", "that", "this", "how", "why", "their", "there",
}

_DISAMBIG_LINE_RE = re.compile(r"^\s*\d+\)\s*\*\*[^*]{2,120}\*\*", re.M)
_FOCUS_LINE_RE = re.compile(
    r"^\s*(based on your question|this report focuses)", re.IGNORECASE
)


def _label_core(label: str) -> str:
    """Normalized sense-label key for matching against report prose.

    Intent labels often carry a parenthetical gloss ("Electrical transformer
    (AC voltage device)") while a writer naturally renders only the core
    ("Electrical transformer"). Comparing the full string verbatim made the
    gate reject a report that DID disambiguate correctly. Strip parentheticals
    and trailing qualifiers, casefold, and match on the core.
    """
    text = str(label or "").strip()
    text = re.sub(r"\s*\([^)]*\)", "", text)
    text = re.split(r"\s+[—–-]\s+", text)[0]
    return text.strip().lower()


def _concept_terms(text: str) -> List[str]:
    return [
        t for t in re.findall(r"[a-z0-9]{4,}", (text or "").lower())
        if t not in _QUERY_STOP
    ]


def _exec_summary_text(answer: str) -> str:
    match = re.search(r"^##\s+Executive Summary\s*$", answer or "", re.M)
    if not match:
        return (answer or "")[:700]
    start = match.end()
    nxt = re.search(r"^##\s+", answer[start:], re.M)
    return answer[start: start + nxt.start()] if nxt else answer[start:]


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text or ""))


_LENGTH_BANDS = {
    "quick": (120, 650),
    "standard": (200, 950),
    "audit": (200, 950),
    "redteam": (200, 950),
    "deep": (350, 1500),
    "executive": (350, 1500),
}


def length_band(mode: str) -> tuple[int, int]:
    """The (min, max) word band the clarity score enforces for a mode.

    Public so the synthesizer's length hint and the quality gate can never
    drift apart — a deep report told "up to 1400 words" while the gate caps
    at 1500 is fine, but a section-wise writer with no per-section budget
    overshot to 3809 words. One band, used by both.
    """
    return _LENGTH_BANDS.get(str(mode or "standard").lower(), _LENGTH_BANDS["standard"])


def _length_band_score(words: int, mode: str) -> float:
    lo, hi = length_band(mode)
    if words < lo:
        return words / lo
    if words > hi:
        return hi / words
    return 1.0


def score_answer_relevance(query: str, answer: str) -> float:
    """How much of the answer's content serves the QUESTION asked (0-1).

    This is the "evidence-rich but answers the wrong thing" guard, and unlike
    the in-gate relevance blend it is a pure answer-vs-query measure — it
    cannot be inflated by verified evidence or citation density. It uses the
    MARS semantic engine (TF-IDF cosine) over the query against the answer's
    substantive sentences, plus the query-concept hit rate, so a report about
    a neighbouring topic scores low even when every claim is well sourced.

    Deterministic; no LLM, no network.
    """
    query = (query or "").strip()
    answer = (answer or "").strip()
    if not query or not answer:
        return 0.0

    concept_terms = _concept_terms(query)
    concept_hit = (
        sum(1 for t in concept_terms if t in answer.lower()) / len(concept_terms)
        if concept_terms else 1.0
    )

    # Semantic alignment: best match of the query against the answer's
    # substantive sentences (skip bullets that are pure data dumps and very
    # short lines, which produce noisy cosine scores).
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", answer)
        if s.strip() and _word_count(s) >= 5
    ]
    semantic = 0.0
    if sentences:
        try:
            from app.core.semantic import rank_by_similarity

            scores = rank_by_similarity(query, sentences[:200])
            if scores:
                top = sorted(scores, reverse=True)[:5]
                semantic = sum(top) / len(top)
        except Exception:
            semantic = 0.0

    return round(0.5 * concept_hit + 0.5 * semantic, 4)


def _is_data_dump_line(line: str) -> bool:
    """A bullet that is mostly digits/punctuation with no prose — the
    'citation spam / statistics dump' shape the gate must reject."""
    cleaned = line.lstrip("-* ").strip()
    if not cleaned:
        return False
    chars = [c for c in cleaned if not c.isspace()]
    digits = sum(1 for c in chars if c.isdigit() or not c.isalpha())
    return len(chars) > 0 and digits / len(chars) > 0.45


@dataclass
class QualityReport:
    accuracy: int = 0
    relevance: int = 0
    evidence: int = 0
    clarity: int = 0
    reasoning: int = 0
    overall: int = 0
    threshold: float = 70.0
    passed: bool = False
    failures: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "relevance": self.relevance,
            "evidence": self.evidence,
            "clarity": self.clarity,
            "reasoning": self.reasoning,
            "overall": self.overall,
            "threshold": self.threshold,
            "passed": self.passed,
            "failures": list(self.failures),
            "details": dict(self.details),
        }

    def render(self) -> str:
        return (
            f"Accuracy {self.accuracy}/100 | Relevance {self.relevance}/100 | "
            f"Evidence {self.evidence}/100 | Clarity {self.clarity}/100 | "
            f"Reasoning {self.reasoning}/100 — overall {self.overall}/100 "
            f"({'PASSED' if self.passed else 'BELOW THRESHOLD'})"
        )


def evaluate_answer(
    query: str,
    *,
    intent: Optional[Dict[str, Any]] = None,
    answer: str,
    facts: Sequence[Dict[str, Any]] = (),
    answer_support: Optional[Dict[str, Any]] = None,
    citation_health: Optional[Dict[str, Any]] = None,
    contradictions: Sequence[Dict[str, Any]] = (),
    redteam_findings: Sequence[Dict[str, Any]] = (),
    mode: str = "standard",
    threshold: float = 70.0,
) -> QualityReport:
    intent = intent or {}
    support = answer_support or {}
    lowered = (answer or "").lower()
    failures: List[str] = []

    # ---- Accuracy: cited-sentence support + verified share of the pool ----
    rate = support.get("rate")
    verified = [f for f in facts if isinstance(f, dict) and f.get("verified") is True]
    checked = [f for f in facts if isinstance(f, dict) and "verified" in f]
    verified_share = (len(verified) / len(checked)) if checked else 0.0
    if rate is None:
        accuracy = round(100 * (0.35 + 0.65 * verified_share))
        if accuracy > 75:
            accuracy = 75  # no support measurement -> cannot claim full accuracy
    else:
        accuracy = round(100 * (0.65 * float(rate) + 0.35 * verified_share))
    numeric_rate = support.get("numeric_rate")
    if numeric_rate is not None:
        accuracy = round(accuracy * (0.85 + 0.15 * float(numeric_rate)))

    # ---- Relevance: does the answer address the question actually asked? ----
    exec_text = _exec_summary_text(answer)
    concept_terms = _concept_terms(query)
    concept_hit = (
        sum(1 for t in concept_terms if t in exec_text.lower()) / len(concept_terms)
        if concept_terms else 1.0
    )

    senses = [
        s for s in (intent.get("senses") or [])
        if isinstance(s, dict) and str(s.get("label", "")).strip()
    ]
    disambig_ok = True
    if intent.get("ambiguity") and senses:
        opening = (answer or "")[:900]
        disambig_lines = len(_DISAMBIG_LINE_RE.findall(opening))
        sense_alignment = 0.0
        if disambig_lines >= min(2, len(senses)):
            sense_alignment = 0.6
            cores = [c for c in (_label_core(s["label"]) for s in senses[:2]) if c]
            if intent.get("recommended_action") == "research_both":
                if cores and all(core in lowered for core in cores):
                    sense_alignment = 1.0
            else:
                sense_alignment = 1.0
        if sense_alignment < 1.0:
            failures.append(
                "Relevance: the report does not open with the required disambiguation "
                "(numbered 'n) **Sense name** — explanation' lines) naming each meaning "
                "of the ambiguous term."
            )
        if disambig_lines == 0:
            # A report that never names the senses silently picked one — the
            # exact "wrong domain" failure. Hard-fail below the relevance floor.
            disambig_ok = False
    else:
        # Angle coverage: the plan's researched angles should surface in the text.
        angles = {
            str(f.get("sub_question", "") or "").strip()
            for f in facts if isinstance(f, dict) and str(f.get("sub_question", "") or "").strip()
        }
        if angles:
            covered = sum(
                1
                for a in angles
                if any(t in lowered for t in _concept_terms(a)[:6])
            )
            sense_alignment = covered / len(angles)
        else:
            sense_alignment = 1.0

    details = support.get("sentence_details") or []
    usable_units = [
        d for d in details
        if isinstance(d, dict) and not _DISAMBIG_LINE_RE.match(str(d.get("sentence", "")))
        and not _FOCUS_LINE_RE.match(str(d.get("sentence", "")))
    ]
    cited_units = [d for d in usable_units if d.get("markers")]
    unsupported_units = [d for d in usable_units if d.get("status") == "unsupported"]
    contamination = (
        1 - len(unsupported_units) / len(cited_units) if cited_units else 0.6
    )
    # Answer-to-query relevance gate: a separate, evidence-independent measure
    # of whether the report addresses the QUESTION. Evidence-rich text about
    # the wrong topic scores high on accuracy/evidence but low here.
    answer_relevance = score_answer_relevance(query, answer)
    relevance = round(
        100 * (0.25 * concept_hit + 0.25 * sense_alignment + 0.20 * contamination + 0.30 * answer_relevance)
    )
    if not disambig_ok:
        relevance = min(relevance, RELEVANCE_FLOOR - 5)
    if answer_relevance < 0.18:
        # Hard-fail territory: the body does not semantically match the query
        # at all. Cap below the floor regardless of other signals.
        relevance = min(relevance, RELEVANCE_FLOOR - 5)
        failures.append(
            "Relevance: the report's content does not match the question asked — "
            "it reads as evidence about a different topic. Restructure around the "
            f"query's actual subject: {query[:80]!r}."
        )
    if concept_hit < 0.5:
        failures.append(
            "Relevance: the Executive Summary does not address the query's core "
            f"terms ({', '.join(concept_terms[:4]) or query[:60]}). Answer the asked "
            "question in the first paragraph."
        )
    if sense_alignment < 0.6:
        failures.append(
            "Relevance: the report drifts from the researched angles/meaning — "
            "restructure so every section serves the query's resolved intent."
        )

    # ---- Evidence: citation density, verified share, primary share, health ----
    # Density is measured over the WRITER's prose only. The report's trailing
    # sections (confidence panel, limitations, contradiction ranges, source
    # ledger) are appended from measured pipeline state and carry no [n]
    # markers by construction; counting them as uncited factual sentences
    # penalized a report whose own body was ~68% cited down to a 38% "density"
    # and pinned the evidence sub-score near 45 regardless of writer quality.
    # audit_citations already strips these sections; the two now share one
    # canonical list (sources.MACHINE_SECTIONS) so they cannot drift apart.
    writer_body = strip_machine_sections(answer)
    disambig_free_units = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", writer_body)
        if s.strip()
        and len(s.strip().split()) >= 8
        and not _DISAMBIG_LINE_RE.match(s.strip())
        and not _FOCUS_LINE_RE.match(s.strip())
    ]
    cited_n = sum(1 for s in disambig_free_units if re.search(r"\[\d+\]", s))
    density = cited_n / len(disambig_free_units) if disambig_free_units else 0.0
    urls = [str(f.get("source", "")) for f in facts if isinstance(f, dict) and f.get("source")]
    primary = primary_source_share(urls)
    health = (citation_health or {}).get("summary") or {}
    health_total = sum(int(health.get(k, 0) or 0) for k in ("ok", "warn", "broken", "bad", "unchecked"))
    health_ok = (int(health.get("ok", 0) or 0) / health_total) if health_total else 0.6
    evidence = round(100 * (0.40 * density + 0.30 * verified_share + 0.15 * primary + 0.15 * health_ok))

    # ---- Clarity: structure, length band, no data dumps ----
    has_exec = bool(re.search(r"^##\s+Executive Summary", answer or "", re.M)) or "# Final Answer" in (answer or "")
    has_bullets = "\n- " in (answer or "") or "\n* " in (answer or "")
    words = _word_count(answer)
    band = _length_band_score(words, mode)
    bullets = [
        line for line in (answer or "").splitlines()
        if line.lstrip().startswith(("- ", "* "))
    ]
    dumps = sum(1 for line in bullets if _is_data_dump_line(line))
    dump_term = 1 - (dumps / len(bullets)) if bullets else 1.0
    clarity = round(100 * (0.25 * has_exec + 0.25 * has_bullets + 0.30 * band + 0.20 * dump_term))
    if not has_exec:
        failures.append("Clarity: the report has no '## Executive Summary' section.")
    if band < 0.7:
        failures.append(
            f"Clarity: report length ({words} words) is outside the {mode} band — "
            "tighten or expand to the appropriate depth."
        )
    if dump_term < 0.8:
        failures.append(
            "Clarity: some bullets are bare statistics dumps — give every number "
            "context in prose or drop it."
        )

    # ---- Reasoning: conflicts surfaced, limitations present, objections kept ----
    if contradictions:
        conflict_term = 1.0 if any(
            t in lowered for t in ("range", "disagree", "conflict", "spread")
        ) else 0.3
        if conflict_term < 1.0:
            failures.append(
                "Reasoning: source conflicts were detected but not surfaced — report "
                "them as ranges, never silently pick one."
            )
    else:
        conflict_term = 1.0
    limitations = 1.0 if ("limitations" in lowered or "could not verify" in lowered) else 0.0
    if not limitations:
        failures.append("Reasoning: the report states no limitations — add what could not be verified.")
    objections = 1.0
    if redteam_findings:
        objections = 1.0 if any(
            t in lowered for t in ("objection", "red team", "weakness", "would invalidate")
        ) else 0.4
    reasoning = round(100 * (0.40 * conflict_term + 0.35 * limitations + 0.25 * objections))

    overall = round(
        0.30 * accuracy + 0.25 * relevance + 0.20 * evidence + 0.15 * clarity + 0.10 * reasoning
    )
    passed = overall >= threshold and relevance >= RELEVANCE_FLOOR and accuracy >= ACCURACY_FLOOR
    for name, value, floor in (
        ("Accuracy", accuracy, ACCURACY_FLOOR),
        ("Relevance", relevance, RELEVANCE_FLOOR),
        ("Evidence", evidence, 35),
        ("Clarity", clarity, 40),
        ("Reasoning", reasoning, 40),
    ):
        if value < floor:
            failures.append(f"{name} {value}/100 is below the {floor}/100 floor.")

    report = QualityReport(
        accuracy=accuracy,
        relevance=relevance,
        evidence=evidence,
        clarity=clarity,
        reasoning=reasoning,
        overall=overall,
        threshold=threshold,
        passed=passed,
        failures=failures[:10],
        details={
            "words": words,
            "concept_hit": round(concept_hit, 3),
            "sense_alignment": round(sense_alignment, 3),
            "answer_relevance": round(answer_relevance, 3),
            "citation_density": round(density, 3),
            "primary_share": round(primary, 3),
        },
    )
    logger.info(
        "[Quality] overall=%d passed=%s (acc=%d rel=%d ev=%d clar=%d reas=%d)",
        overall, passed, accuracy, relevance, evidence, clarity, reasoning,
    )
    return report
