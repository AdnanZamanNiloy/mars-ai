from __future__ import annotations

from app.core.logging import get_logger
import re
from typing import Any, Dict, List

from app.agents.evidence_utils import dedupe_semantic_facts, extract_domain, filter_facts_by_domain, select_diverse
from app.core.degradation import record_fallback
from app.core.llm import LLMClient
from app.core.schemas import SynthesizerAnswerModel

logger = get_logger(__name__)


SYNTHESIZER_SYSTEM_PROMPT = """
You are the MARS Synthesis Engine — the Final Synthesis Agent. Your only job
is to turn verified research into a clean, premium, highly readable
intelligence report. You do not dump search results; you present knowledge.

━━━ ABSOLUTE FORMATTING RULES ━━━

1. Clear visual hierarchy: use the exact section structure below, with
   markdown `## ` headings and a blank line before every new section.
2. Short paragraphs — 3 to 4 lines maximum.
3. Prefer bullet points over long dense paragraphs.
4. Never dump everything into one continuous block of text.
5. Calm, precise, professional tone. Write like a premium research brief,
   not raw notes. Be direct and scannable. Remove repetitive or low-value
   sentences.
6. Never write "the research found", "the agents discovered", or "according
   to the research". Present the knowledge directly.
7. LENGTH: default under 900 words. When the input carries an explicit
   length hint (deep-research modes), follow the hint instead. Slow
   providers cannot serve unbounded generation, and a timed-out synthesis
   degrades to extraction.

━━━ REQUIRED STRUCTURE (exact order) ━━━

1. `## Executive Summary`
   4-6 sentences maximum. If the query term has multiple distinct meanings,
   clarify that in the FIRST sentence and keep the meanings strictly
   separate throughout the report. End with the overall confidence level
   and numeric score.

2. `## Key Findings`
   Bullet points only. Each bullet is ONE clear, self-contained fact with
   its [n] citation. Maximum 8 bullets, highest-confidence facts first.

3. Deep-dive sections (only where they add value):
   - If the query term has multiple distinct senses (e.g. "transformer" in
     machine learning vs electrical engineering), give each sense its own
     `## <Sense> Meaning` section, clearly separated, using short
     paragraphs and bullets. Inside a sense you may use `### ` sub-headings
     (Origin, Core Innovation, Architecture, Scale of Modern Models — or
     whatever fits the subject).
   - Otherwise, add 2-4 `## ` sections, one per research angle from the
     provided angle list (title = the angle, never the raw question
     restated). Skip straight to Evidence & Confidence when no angle adds
     anything.
   - SYNTHESIZE, DON'T PARAPHRASE: each section must weave together 2+ of
     the provided sources into a cause/effect or comparison narrative.
     Single-source recitation is what makes a report read like raw notes.
   - DISAGREEMENT IS DATA: where sources conflict, present both positions
     side by side with their numbers and sources — never average them and
     never silently pick one.

5. `## Key Figures` — ONLY when the evidence contains quantitative claims:
   bullets with the number, unit, period, and scope attached, each cited
   [n]. Omit the section when the evidence has no solid numbers.

6. `## Evidence & Confidence`
   Overall confidence score, what is well supported, what could not be
   verified, and the major gaps. Mandatory even at high confidence.

━━━ VERIFY BEFORE YOU WRITE A SINGLE WORD ━━━

- Every claim, name, date, or number must trace to at least one provided
  source. Cite at the point of use with [n] from the provided Sources list
  only. Never invent numbers, links, or sources.
- If two sources conflict, flag the conflict — never silently pick one.
- Discard extraction artifacts (garbled text, fragments with no clear
  subject, unrelated names). Absence of a clean answer is a valid,
  reportable finding — state it plainly in the Executive Summary.
- Never mix unrelated people, organizations, or senses of a term. If the
  evidence points to multiple distinct entities with similar names,
  separate them explicitly or state that the identity is ambiguous.

HARD FAILURE CONDITIONS — reject your own draft if any are true:
- A section that is one wall of text with no bullets or breathing room
- Any claim in the body with no traceable source
- Any name, number, or fact that is not clearly corroborated
- Unrelated senses or entities blended without explicit separation

Return valid JSON only in this schema:
{"answer": "<final synthesized report with [n] citations>"}
""".strip()


async def synthesizer_agent(
    llm: LLMClient,
    query: str,
    facts: List[Dict[str, Any]],
    context: Dict[str, Any] | None = None,
) -> str:
    """Synthesize the final report. `context` carries the decision-grade
    inputs the workflow already computed: contradictions to flag,
    overall confidence, degraded stages, and evidence counts for the
    Confidence & Gaps section. Optional so unit tests and scripts calling
    with (llm, query, facts) keep working — gaps then derive from counts.
    """
    usable_facts = dedupe_semantic_facts(filter_facts_by_domain(facts))
    if not usable_facts:
        return (
            f"{query} is an area that requires reliable evidence to explain accurately. "
            "Current retrieved evidence was too limited or low quality to produce a robust synthesis."
        )
    ctx = context or {}

    # Mode-aware depth (3.7): deep/executive runs are allowed a longer,
    # more analytical report — that depth is the product. Quick/standard
    # stay tight for latency. Bounded: providers degrade to extraction on
    # unbounded generation.
    mode = str(ctx.get("mode", "standard") or "standard")
    if mode in ("deep", "executive"):
        length_hint = (
            "LENGTH: this is a deep-research brief — up to 1400 words. "
            "Go deeper per angle: mechanisms, numbers with context, and "
            "explicit treatment of conflicting evidence."
        )
    else:
        length_hint = "LENGTH: keep the report under 900 words."

    top_facts = _stratified_top_facts(usable_facts, per_angle=10, cap=40)
    numbered = _numbered_sources(top_facts)
    source_lines = "\n".join(f"[{s['n']}] {s['domain']}" + (f" ({s['url']})" if s["url"] else "")
                             for s in numbered)
    angles = []
    for fact in top_facts:
        sq = str(fact.get("sub_question", "") or "").strip()
        if sq and sq not in angles:
            angles.append(sq)

    user_prompt = (
        f"Main query: {query}\n\n"
        f"{length_hint}\n\n"
        f"Angles to cover (one section each, in this order):\n"
        + ("\n".join(f"- {sq}" for sq in angles) + "\n\n" if angles else "")
        + f"Evidence facts: {top_facts}\n\n"
        + _render_context_block(ctx)
        + f"Sources (cite these by number):\n{source_lines}\n\n"
        "Return JSON in this schema: "
        '{"answer": "<final synthesized report with [n] citations>"}'
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

    # Deterministic fallback mirrors the decision-grade shape extractively:
    # Executive Summary (lead claim + confidence line), Key Findings (one
    # representative claim per angle, each its own paragraph), angle
    # sections from the remaining claims (no claim used twice), Key figures,
    # Confidence & Gaps, and a used-only legend. MMR overlap threshold 0.35
    # splits observed paraphrases (0.39-0.53) from cross-sense claims
    # (0.10-0.18). No boilerplate openers.
    diverse = select_diverse(top_facts, k=40, max_similarity=0.35)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in diverse:
        if not str(item.get("claim", "")).strip():
            continue
        key = str(item.get("sub_question", "") or "").strip()
        bucket = groups.setdefault(key, [])
        if len(bucket) < 6:
            bucket.append(item)
        if len(groups) >= 6 and all(len(v) >= 3 for v in groups.values()):
            break
    if not groups:
        return f"No reliable evidence was retrieved for {_normalize_query_concept(query)}."

    gap_stats = _gap_stats(ctx, usable_facts)
    n_sources = len({str(i.get("source", "")) for _, items in groups.items() for i in items if i.get("source")})

    used: List[Dict[str, Any]] = []
    seen_claims = set()

    def _take(item: Dict[str, Any]) -> None:
        if item.get("claim") not in seen_claims:
            seen_claims.add(item.get("claim"))
            used.append(item)

    # Key Findings: one representative claim per angle (max 5), each a
    # standalone paragraph — never a headline with nothing under it.
    finding_items = [items[0] for _, items in list(groups.items())[:6] if items][:5]
    for item in finding_items:
        _take(item)

    # Angle sections from the remaining claims (findings not repeated).
    sections: List[str] = []
    for sq, items in list(groups.items())[:6]:
        rest = [i for i in items if i.get("claim") not in seen_claims]
        if not rest:
            continue
        body = _sanitize_answer_text(" ".join(str(i.get("claim", "")) for i in rest), query)
        if not body:
            continue
        title = _section_title(sq)
        sections.append(f"## {title}\n\n{body}" if title else body)
        for i in rest:
            _take(i)

    figures = [
        item for item in diverse
        if _FIGURE_RE.search(str(item.get("claim", "") or ""))
        and item.get("claim") not in seen_claims
    ][:5]
    if figures:
        sections.append(
            "## Key figures\n\n"
            + _sanitize_answer_text(" ".join(str(i.get("claim", "")) for i in figures), query)
        )
        for i in figures:
            _take(i)

    summary_lead = str(finding_items[0].get("claim", "")) if finding_items else ""
    thin_note = (
        "Evidence is thin — fewer than 3 verified facts support this report, "
        "so treat every finding below as provisional. "
        if gap_stats["verified"] < 3 else ""
    )
    degraded_note = (
        f" Pipeline stages on deterministic fallback: {', '.join(gap_stats['degraded'])}."
        if gap_stats["degraded"] else ""
    )
    ambiguity_note = (
        f" The evidence spans {len(groups)} distinct angles of this query, "
        "separated below."
        if len(groups) >= 3 else ""
    )
    summary = (
        f"## Executive Summary\n\n{thin_note}{summary_lead}{ambiguity_note}\n\n"
        f"Confidence: {_confidence_statement(ctx, gap_stats['verified'])} — "
        f"{gap_stats['verified']} verified facts "
        f"across {n_sources} sources.{degraded_note}"
    )
    sections.insert(0, summary)
    rest_findings = finding_items[1:]
    if rest_findings:
        bullets = "\n\n".join(
            "- " + str(i.get("claim", "")).strip() for i in rest_findings
        )
        sections.insert(1, "## Key Findings\n\n" + _sanitize_answer_text(bullets, query))

    sections.append(_gaps_section(gap_stats))

    numbered = _numbered_sources(used[:40])
    # No "# Final Answer" lead here: the workflow's finalize step wraps the
    # synthesizer output with the report title itself (double title if both).
    return "\n\n".join(sections) + "\n\n" + _legend_block(numbered)


def _gap_stats(ctx: Dict[str, Any], usable_facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts the Confidence & Gaps section needs. Prefers workflow context
    (contradictions, degraded stages, totals); derives the rest from facts."""
    total = int(ctx.get("total_facts", 0) or 0) or len(usable_facts)
    if ctx.get("verified_count") is not None:
        verified = int(ctx.get("verified_count") or 0)
    else:
        # Synthesis input passed verification upstream: anything explicitly
        # flagged False was already filtered, the rest counts as verified.
        verified = sum(1 for f in usable_facts if f.get("verified", True) is not False)
    contradictions = ctx.get("contradictions", []) or []
    if not isinstance(contradictions, list):
        contradictions = []
    degraded = ctx.get("degraded", []) or []
    if not isinstance(degraded, list):
        degraded = []
    return {
        "total": total,
        "verified": verified,
        "unverified_excluded": max(0, total - verified),
        "contradictions": contradictions[:5],
        "degraded": [str(d) for d in degraded if d],
    }


def _confidence_statement(ctx: Dict[str, Any], verified_count: int) -> str:
    """'High (0.76)' on the pipeline's 0-1 scale (0.75 = sufficient), or a
    volume-based level when no score was passed in context."""
    raw = ctx.get("confidence", None) if isinstance(ctx, dict) else None
    try:
        score = float(raw)
        level = "High" if score >= 0.75 else ("Medium" if score >= 0.5 else "Low")
        return f"{level} ({score:.2f})"
    except (TypeError, ValueError):
        level = "High" if verified_count >= 8 else ("Medium" if verified_count >= 3 else "Low")
        return level


def _gaps_section(stats: Dict[str, Any]) -> str:
    """Evidence & Confidence + Limitations: score/counts/conflicts, then
    what could not be verified. Both mandatory even at high confidence."""
    lines = [
        "## Evidence & Confidence",
        "",
        f"Well-supported: {stats['verified']} verified facts feed this report; "
        "every claim above traces to a cited source.",
    ]
    if stats["unverified_excluded"]:
        lines.append(
            f"Uncertain: {stats['unverified_excluded']} collected claims failed "
            "verification and were excluded from synthesis rather than repeated."
        )
    if stats["contradictions"]:
        lines.append(
            f"Conflicting evidence: {len(stats['contradictions'])} source conflict(s) "
            "flagged — see the Contradictions section of the full report; "
            "conflicting numbers are reported as ranges, not picked."
        )
    if stats["degraded"]:
        lines.append(
            "Pipeline gaps: deterministic fallback covered "
            f"{', '.join(stats['degraded'])} — those sections are extractive, "
            "not model-written."
        )
    lines.extend([
        "",
        "## Limitations",
        "",
        "Could not verify: claims without a traceable source were discarded "
        "during synthesis and do not appear above.",
    ])
    return "\n".join(lines)


def _render_context_block(ctx: Dict[str, Any]) -> str:
    """Decision-grade inputs for the LLM brief: conflicts to flag and the
    honesty baseline for Confidence & Gaps. Empty when no context passed."""
    if not ctx:
        return ""
    parts = []
    contradictions = ctx.get("contradictions", []) or []
    if isinstance(contradictions, list) and contradictions:
        lines = []
        for c in contradictions[:5]:
            if not isinstance(c, dict):
                continue
            lines.append(
                f"- \"{str(c.get('claim_a', ''))[:140]}\" ({c.get('source_a', '')}) "
                f"CONFLICTS WITH \"{str(c.get('claim_b', ''))[:140]}\" ({c.get('source_b', '')})"
            )
        if lines:
            parts.append("Flagged source conflicts (surface these, never silently pick one):\n" + "\n".join(lines))
    try:
        conf = float(ctx.get("confidence", "nan"))
        parts.append(f"Pipeline confidence: {conf:.2f} (0.75+ = sufficient).")
    except (TypeError, ValueError):
        pass
    degraded = ctx.get("degraded", []) or []
    if isinstance(degraded, list) and degraded:
        parts.append(f"Stages on deterministic fallback: {', '.join(str(d) for d in degraded)}.")
    total = ctx.get("total_facts", 0) or 0
    verified = ctx.get("verified_count", 0) or 0
    if total:
        parts.append(f"Evidence pool: {verified}/{total} facts verified; unverified claims were excluded.")
    if not parts:
        return ""
    return "Research honesty baseline:\n" + "\n".join(parts) + "\n\n"


def _stratified_top_facts(
    facts: List[Dict[str, Any]], per_angle: int = 6, cap: int = 30
) -> List[Dict[str, Any]]:
    """Top facts round-robin across sub-questions instead of pure confidence
    order. Pure ranking lets one low-scoring source domain bury a whole
    angle (observed live: the market angle cut from a top-20 while two
    definition angles filled it). Every angle keeps up to `per_angle` facts.
    """
    ranked = sorted(
        facts or [],
        key=_fact_confidence,
        reverse=True,
    )
    by_angle: Dict[str, List[Dict[str, Any]]] = {}
    for fact in ranked:
        key = str(fact.get("sub_question", "") or "").strip()
        by_angle.setdefault(key, []).append(fact)
    picked: List[Dict[str, Any]] = []
    for i in range(max(1, per_angle)):
        for bucket in by_angle.values():
            if len(bucket) > i:
                picked.append(bucket[i])
                if len(picked) >= max(1, cap):
                    return picked
    return picked


def _fact_confidence(fact: Any) -> float:
    try:
        return float(fact.get("confidence", 0.0))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _section_title(sub_question: str) -> str:
    """Humanize a grouping key for a `## ` header: leading-capitalized,
    never raw keyword salad in lowercase."""
    text = (sub_question or "").strip()
    if not text:
        return ""
    return text[0].upper() + text[1:]


def _numbered_sources(top_facts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Deterministic [n] legend built from evidence — the model cites
    numbers, never URLs, so markers always resolve to real sources.

    One entry per DISTINCT source URL (a page cited by five claims is one
    source, not five) and capped at the strongest 12: first occurrence in
    the confidence-stratified fact order wins, so the legend lists the
    strongest sources first and a clean numbered list is what renders.
    """
    numbered: List[Dict[str, str]] = []
    seen_urls: set = set()
    for fact in top_facts:
        url = str(fact.get("source", "") or "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        numbered.append({
            "n": len(numbered) + 1,
            "domain": extract_domain(url) or url or "unknown source",
            "url": url,
        })
        if len(numbered) >= 12:
            break
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
    return f"{answer.rstrip()}\n\n" + _legend_block(numbered)


def _legend_block(numbered: List[Dict[str, str]]) -> str:
    """`## Sources` section: one entry per line, blank-line separated, so
    the frontend renders a clean vertical numbered list under its own
    heading (not one glued paragraph)."""
    lines = [f"[{s['n']}] {s['domain']}" + (f" — {s['url']}" if s["url"] else "")
             for s in numbered]
    return "## Sources\n\n" + "\n\n".join(lines)


_FIGURE_RE = re.compile(r"\d")


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


# Inline bullet separators: models sometimes emit bullets on one line
# ("... induction [4]. - Electrical transformers modify ..."). A " - "
# right after a sentence end / citation bracket is a list separator, not
# prose; splitting on it restores real bullet lines. Only applied inside
# lines that already start with "- ".
_BULLET_SEP_RE = re.compile(r"(?<=[.!?\]])\s+-\s+(?=[A-Z0-9\"'(])")


def _sanitize_answer_text(answer: str, query: str) -> str:
    # Preserve the report's visual hierarchy: markdown headings (## / ###)
    # become their own blocks, consecutive "- " lines stay distinct list
    # items (joined with single newlines into one bullet block), and
    # paragraph breaks are kept — the frontend's section renderer maps all
    # of these to real elements. Only intra-line whitespace collapses;
    # consecutive prose lines still merge (model line-wrapping must not
    # become line breaks).
    raw_lines = (answer or "").replace("\r\n", "\n").split("\n")
    paras: List[str] = []
    current: List[str] = []
    bullets: List[str] = []

    def _flush_bullets() -> None:
        nonlocal bullets
        if bullets:
            paras.append("\n".join(bullets))
            bullets = []

    def _flush_prose() -> None:
        nonlocal current
        if current:
            paras.append(" ".join(current))
            current = []

    for raw_line in raw_lines:
        line = re.sub(r"\s+", " ", raw_line).strip()
        is_heading = bool(re.match(r"^#{1,6}\s+\S", line))
        if line.startswith("- "):
            _flush_prose()
            for part in _BULLET_SEP_RE.split(line):
                part = part.strip()
                if part:
                    bullets.append(part if part.startswith("- ") else f"- {part}")
        elif is_heading:
            _flush_bullets()
            _flush_prose()
            paras.append(line)
        elif line:
            _flush_bullets()
            current.append(line)
        else:
            _flush_bullets()
            _flush_prose()
    _flush_bullets()
    _flush_prose()
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
