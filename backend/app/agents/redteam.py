"""Red Team (vision Feature 08) — try to break the conclusion, not rate it.

The existing critic asks "is this evidence sufficient?" and returns a verdict.
Its prompt contains three red-team questions, but the answers are folded into a
single `reason` string and then thrown away: nothing downstream can see which
assumption was weak, what the alternative explanation was, or what evidence was
missing. The report cannot show it, the next pass cannot target it, and
confidence cannot discount it.

This module separates the two jobs. The critic remains the *gate*. The red team
is an *attacker* that produces structured findings which are:

  * reported to the user (a research product that cannot say how it might be
    wrong is a marketing product)
  * turned into targeted follow-up searches for the next pass
  * turned into a survival score that feeds the confidence engine

A deterministic attack layer runs first and always. It costs nothing and it
finds the failures models are worst at noticing about themselves — single-source
claims, zero primary sources, stale evidence on a recency-sensitive question,
an evidence pool with no criticism axis, unresolved numeric spread. The LLM
attack runs on top and can be skipped entirely under budget pressure without
losing the floor.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.core.degradation import record_fallback
from app.core.llm import LLMClient, clamp_confidence
from app.core.logging import get_logger

from app.agents.contradiction import numeric_ranges, summarize_contradictions
from app.agents.evidence_utils import evidence_stats
from app.agents.schemas_ext import RedTeamReportModel

logger = get_logger(__name__)

KIND_WEAK_ASSUMPTION = "weak_assumption"
KIND_ALTERNATIVE = "alternative_explanation"
KIND_MISSING_EVIDENCE = "missing_evidence"
KIND_INVALIDATING = "invalidating_condition"

VALID_KINDS = (
    KIND_WEAK_ASSUMPTION,
    KIND_ALTERNATIVE,
    KIND_MISSING_EVIDENCE,
    KIND_INVALIDATING,
)


@dataclass
class RedTeamFinding:
    kind: str
    statement: str
    severity: float
    target_claim: str = ""
    test: str = ""
    origin: str = "heuristic"      # heuristic | model

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "statement": self.statement,
            "severity": round(self.severity, 3),
            "target_claim": self.target_claim,
            "test": self.test,
            "origin": self.origin,
        }


@dataclass
class RedTeamReport:
    findings: List[RedTeamFinding] = field(default_factory=list)
    survival_score: float = 0.6
    targeted_queries: List[str] = field(default_factory=list)
    summary: str = ""

    @property
    def survives(self) -> bool:
        return self.survival_score >= 0.6 and not self.blocking

    @property
    def blocking(self) -> List[RedTeamFinding]:
        """Findings severe enough that publishing without addressing them would
        be misleading."""
        return [f for f in self.findings if f.severity >= 0.75]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "survival_score": round(self.survival_score, 3),
            "survives": self.survives,
            "blocking": [f.to_dict() for f in self.blocking],
            "targeted_queries": list(self.targeted_queries),
            "summary": self.summary,
            "by_kind": self.by_kind(),
        }

    def by_kind(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for finding in self.findings:
            out[finding.kind] = out.get(finding.kind, 0) + 1
        return out

    def render(self) -> str:
        """Markdown block the report embeds under `## Red Team Review`."""
        if not self.findings:
            return (
                "## Red Team Review\n\n"
                "No material weakness was found in the evidence base."
            )
        lines = ["## Red Team Review", ""]
        if self.summary:
            lines += [self.summary, ""]
        label = {
            KIND_WEAK_ASSUMPTION: "Weak assumption",
            KIND_ALTERNATIVE: "Alternative explanation",
            KIND_MISSING_EVIDENCE: "Missing evidence",
            KIND_INVALIDATING: "Would invalidate the conclusion",
        }
        for finding in sorted(self.findings, key=lambda f: -f.severity)[:8]:
            lines.append(
                f"- **{label.get(finding.kind, finding.kind)}** "
                f"(severity {finding.severity:.2f}): {finding.statement}"
                + (f" _Test:_ {finding.test}" if finding.test else "")
            )
        lines.append("")
        lines.append(
            f"Evidence survival score: {self.survival_score:.2f}"
            + (" — unresolved blocking issues remain." if self.blocking else ".")
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic attacks
# ---------------------------------------------------------------------------

def heuristic_attacks(
    query: str,
    facts: Sequence[Dict[str, Any]],
    *,
    contradictions: Sequence[Dict[str, Any]] = (),
    planned_axes: Sequence[str] = (),
    needs_recency: bool = False,
) -> List[RedTeamFinding]:
    """Attacks derivable from the evidence pool's own shape.

    Every one of these is a real weakness a reader would raise, and none of them
    requires a model to notice — which is precisely why they were being missed:
    the model was asked to introspect instead of the system being asked to
    measure.
    """
    stats = evidence_stats(facts)
    findings: List[RedTeamFinding] = []
    total = int(stats["total"])
    if total == 0:
        return [
            RedTeamFinding(
                KIND_MISSING_EVIDENCE,
                "No usable evidence was extracted, so any conclusion rests on the model's prior knowledge.",
                0.95,
                test="Retrieve at least three independent sources on the question.",
            )
        ]

    if stats["distinct_domains"] < 2:
        findings.append(
            RedTeamFinding(
                KIND_INVALIDATING,
                f"Every claim traces to a single domain ({', '.join(stats['domains'][:1])}); "
                "one publisher's framing is indistinguishable from consensus here.",
                0.85,
                test="Find a source from an unrelated organisation that states the same thing.",
            )
        )
    elif stats["distinct_domains"] < 4 and total >= 8:
        findings.append(
            RedTeamFinding(
                KIND_WEAK_ASSUMPTION,
                f"{total} claims rest on only {stats['distinct_domains']} domains, so the "
                "apparent volume of evidence overstates its independence.",
                0.55,
                test="Add sources from at least two more independent publishers.",
            )
        )

    if stats["primary_documents"] == 0:
        findings.append(
            RedTeamFinding(
                KIND_WEAK_ASSUMPTION,
                "No primary source (official publisher, peer-reviewed venue, or dataset) "
                "was reached; every figure is second-hand reporting.",
                0.70,
                test="Locate the originating agency release, filing, or paper behind the key figures.",
            )
        )
    elif stats["primary_share"] < 0.25:
        findings.append(
            RedTeamFinding(
                KIND_WEAK_ASSUMPTION,
                f"Only {stats['primary_share']:.0%} of sources are primary; the report largely "
                "inherits other people's summaries and their errors.",
                0.45,
                test="Trace the two most load-bearing numbers back to their publisher.",
            )
        )

    if stats["corroborated"] == 0 and total >= 5:
        findings.append(
            RedTeamFinding(
                KIND_INVALIDATING,
                "Not one claim is independently corroborated by a second domain, so a single "
                "error anywhere propagates unchecked into the conclusion.",
                0.75,
                test="Confirm the three highest-confidence claims against a second source each.",
            )
        )

    if stats["verified"] == 0:
        findings.append(
            RedTeamFinding(
                KIND_INVALIDATING,
                "No claim passed verification against its cited source, so citations cannot be "
                "treated as support.",
                0.90,
                test="Re-fetch the cited pages and confirm the claims appear in them.",
            )
        )
    elif stats["verified"] / total < 0.5:
        findings.append(
            RedTeamFinding(
                KIND_WEAK_ASSUMPTION,
                f"Only {stats['verified']} of {total} claims verified against their source; "
                "the unverified half is being excluded, which may bias what survives.",
                0.50,
                test="Inspect the verification failures for a systematic cause.",
            )
        )

    if needs_recency and stats["freshness"] < 0.4:
        findings.append(
            RedTeamFinding(
                KIND_INVALIDATING,
                "The question is time-sensitive but the evidence skews old; a more recent "
                "development could reverse the conclusion outright.",
                0.72,
                test="Search for developments within the last 6 months specifically.",
            )
        )

    axes = {str(a) for a in (planned_axes or ())}
    covered = {str(a) for a in stats["axes"]}
    if axes and not any("critic" in a or "limitation" in a or "risk" in a for a in covered):
        findings.append(
            RedTeamFinding(
                KIND_MISSING_EVIDENCE,
                "No angle in the evidence base argues against the conclusion; the research "
                "only looked for confirmation.",
                0.65,
                test="Search explicitly for criticism, failed cases, and counter-evidence.",
            )
        )

    summary = summarize_contradictions(contradictions)
    for spread in numeric_ranges(contradictions)[:2]:
        findings.append(
            RedTeamFinding(
                KIND_ALTERNATIVE,
                f"Sources give a {spread['low']:g}-{spread['high']:g} range for the same "
                f"{spread['unit']} measure; any single figure in the conclusion is a choice, "
                "not a finding.",
                min(0.85, 0.5 + spread["spread"] / max(abs(spread["high"]), 1) * 0.5),
                test="Compare the methodologies behind the extreme estimates.",
            )
        )
    if summary["severe"]:
        findings.append(
            RedTeamFinding(
                KIND_INVALIDATING,
                f"{summary['severe']} severe cross-source contradiction(s) remain unresolved.",
                0.78,
                test="Determine which source's methodology is applicable to this question.",
            )
        )

    return findings


def heuristic_survival(findings: Sequence[RedTeamFinding]) -> float:
    """Survival score from attack severities.

    Multiplicative rather than additive: three moderate weaknesses should not
    average away into "mostly fine", because in practice they compound.
    """
    score = 1.0
    for finding in findings or ():
        score *= max(0.0, 1.0 - 0.55 * float(finding.severity))
    return round(max(0.05, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Model-driven attack
# ---------------------------------------------------------------------------

RED_TEAM_SYSTEM_PROMPT = """
You are the Red Team Agent in a multi-agent research pipeline. You are NOT a
reviewer and NOT a summarizer. Your job is to make the strongest available case
that the current conclusion is WRONG.

Assume the research is confident and plausible. Your value comes entirely from
finding what it missed.

━━━ ATTACK ALONG FOUR LINES ━━━

weak_assumption
  A premise the evidence takes for granted that could fail. Name the premise and
  why it might not hold, not "the evidence is limited".

alternative_explanation
  A different reading of the SAME evidence that leads somewhere else. Selection
  effects, reverse causation, confounders, definitional shifts, survivorship.

missing_evidence
  Evidence that would exist if the conclusion were true, and is conspicuously
  absent from what was retrieved. Absence of the expected is the strongest
  attack you have.

invalidating_condition
  A concrete, checkable state of the world that would make the conclusion wrong.
  It must be falsifiable: "if capacity additions fell in 2025" not "if things
  change".

━━━ RULES ━━━

1. Be specific to THIS evidence. Attacks that would apply to any research
   ("more sources would help") are worthless and will be discarded.
2. Quote or name the claim you are attacking in target_claim when it is
   specific to one.
3. severity is how much the attack matters if correct: 0.8+ means the
   conclusion would change, 0.4 means it would need qualifying.
4. Every finding needs a "test": the search or check that would settle it.
5. targeted_queries must be search-ready strings that would resolve your
   strongest attacks, not restatements of them.
6. If the evidence genuinely withstands attack, say so and set survives=true
   with a high survival_score — a red team that always finds fatal flaws is as
   useless as one that never does.

━━━ OUTPUT ━━━

Return ONLY valid JSON:

{
  "findings": [
    {"kind": "weak_assumption|alternative_explanation|missing_evidence|invalidating_condition",
     "statement": "...", "target_claim": "...", "severity": 0.0, "test": "..."}
  ],
  "survives": true,
  "survival_score": 0.0,
  "targeted_queries": ["..."],
  "summary": "<one or two sentences: the single biggest reason this could be wrong>"
}
""".strip()


async def redteam_agent(
    llm: Optional[LLMClient],
    query: str,
    facts: Sequence[Dict[str, Any]],
    *,
    contradictions: Sequence[Dict[str, Any]] = (),
    planned_axes: Sequence[str] = (),
    needs_recency: bool = False,
    draft_conclusion: str = "",
    use_llm: bool = True,
    max_facts: int = 14,
) -> RedTeamReport:
    """Attack the evidence base; return structured findings and a survival score.

    Heuristic attacks always run and always count. The model attack is additive
    and skippable (`use_llm=False`, or no client), which is what makes the red
    team affordable to run on every mission instead of only on deep ones.
    """
    heuristic = heuristic_attacks(
        query, facts,
        contradictions=contradictions,
        planned_axes=planned_axes,
        needs_recency=needs_recency,
    )
    findings: List[RedTeamFinding] = list(heuristic)
    targeted: List[str] = []
    summary = ""

    if use_llm and llm is not None and facts:
        compact = [
            {
                "claim": str(f.get("claim", ""))[:240],
                "source": str(f.get("source", "")),
                "confidence": round(float(f.get("confidence", 0.0) or 0.0), 2),
                "verified": bool(f.get("verified")),
            }
            for f in list(facts)[:max_facts]
        ]
        conflict_lines = "\n".join(
            f"- \"{str(c.get('claim_a', ''))[:120]}\" ({c.get('source_a', '')}) vs "
            f"\"{str(c.get('claim_b', ''))[:120]}\" ({c.get('source_b', '')})"
            for c in list(contradictions)[:4]
        )
        user_prompt = (
            f"Research question: {query}\n\n"
            + (f"Draft conclusion:\n{draft_conclusion[:1500]}\n\n" if draft_conclusion else "")
            + f"Evidence ({len(compact)} of {len(facts)} claims):\n{compact}\n\n"
            + (f"Already-detected conflicts:\n{conflict_lines}\n\n" if conflict_lines else "")
            + "Attack this. Return JSON only."
        )
        try:
            payload = await llm.generate_json(
                RED_TEAM_SYSTEM_PROMPT,
                user_prompt,
                response_model=RedTeamReportModel,
            )
        except Exception as exc:
            logger.warning("[RedTeam] LLM attack failed; heuristic findings only", exc_info=exc)
            record_fallback("redteam")
            payload = {}

        if isinstance(payload, dict):
            summary = str(payload.get("summary", "") or "").strip()
            for raw in payload.get("findings", []) or []:
                if not isinstance(raw, dict):
                    continue
                statement = str(raw.get("statement", "") or "").strip()
                if len(statement) < 20 or _is_generic(statement):
                    continue
                kind = str(raw.get("kind", "") or "").strip().lower()
                if kind not in VALID_KINDS:
                    kind = KIND_WEAK_ASSUMPTION
                findings.append(
                    RedTeamFinding(
                        kind=kind,
                        statement=statement[:600],
                        severity=clamp_confidence(raw.get("severity", 0.5)),
                        target_claim=str(raw.get("target_claim", "") or "")[:240],
                        test=str(raw.get("test", "") or "")[:300],
                        origin="model",
                    )
                )
            for q in payload.get("targeted_queries", []) or []:
                text = re.sub(r"\s+", " ", str(q or "")).strip()
                if len(text) >= 8 and text not in targeted:
                    targeted.append(text)

    # Deduplicate findings that say the same thing from both layers.
    findings = _dedupe_findings(findings)

    # Survival is measured from the findings, never taken from the model's
    # self-report: an attacker grading its own attack is the same circularity
    # that made the critic's confidence meaningless.
    survival = heuristic_survival(findings)

    if not targeted:
        targeted = [f.test for f in sorted(findings, key=lambda f: -f.severity)[:3] if f.test]

    if not summary and findings:
        strongest = max(findings, key=lambda f: f.severity)
        summary = strongest.statement

    report = RedTeamReport(
        findings=findings,
        survival_score=survival,
        targeted_queries=targeted[:5],
        summary=summary,
    )
    logger.info(
        "[RedTeam] %d finding(s) (%d blocking), survival %.2f",
        len(findings), len(report.blocking), survival,
    )
    return report


_GENERIC_PATTERNS = (
    "more sources", "more research", "further research", "additional data would",
    "limited evidence", "insufficient information", "could be improved",
    "not enough information", "more evidence is needed",
)


def _is_generic(statement: str) -> bool:
    """Reject attacks that would apply to literally any research.

    Without this filter the model reliably fills the findings list with
    "more sources would strengthen this", which consumes the red team's entire
    output budget while telling the user nothing.
    """
    lowered = statement.lower()
    if any(p in lowered for p in _GENERIC_PATTERNS) and len(lowered) < 160:
        return True
    return False


def _dedupe_findings(findings: Sequence[RedTeamFinding]) -> List[RedTeamFinding]:
    kept: List[RedTeamFinding] = []
    for finding in sorted(findings, key=lambda f: -f.severity):
        tokens = set(re.findall(r"[a-z]{4,}", finding.statement.lower()))
        duplicate = False
        for existing in kept:
            other = set(re.findall(r"[a-z]{4,}", existing.statement.lower()))
            if tokens and other and len(tokens & other) / min(len(tokens), len(other)) >= 0.6:
                duplicate = True
                break
        if not duplicate:
            kept.append(finding)
    return kept[:10]
