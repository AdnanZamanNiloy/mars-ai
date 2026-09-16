"""Evidence-grounded reasoning — turn graded evidence into an ARGUMENT STRUCTURE.

Why this module exists
----------------------
The pipeline already grades every claim (A-D), detects and (sometimes) resolves
contradictions, tracks investigation gaps, and measures depth. But synthesis
received all of that as a flat pool plus a grade distribution: the writer was
asked to "reason" without ever being told WHAT the evidence collectively
supports, what genuinely competes, what is established versus merely inferred,
or what is simply unknown. The quality gate's `reasoning` sub-score measured
only whether conflicts/limitations were mentioned, so a report that packed in
well-formed sentences saturated it without doing any actual synthesis.

This module is the missing argument layer. From the same state inputs every
other stage reads — graded facts, contradiction records, the investigation
state and the depth signals — it deterministically builds a ReasoningMap:

  * convergent conclusions — claims jointly supported by independent evidence
    (agreeing across distinct registrable domains), ranked by impact/grade;
  * competing explanations — the real divergences between sources, each naming
    what the disagreement turns on (kind) and who holds each position;
  * epistemic separation — established vs inferred vs unknown;
  * decision-relevant implications — ONLY for decision/comparative queries and
    ONLY from established claims, each tied to the claims that entail it.

Hard constraints, by design (AGENTS.md):

  * Deterministic and LLM-free: stable ordering with a normalized-text
    tie-break, so two runs on the same evidence produce the same structure.
  * No new facts: every conclusion references the claim(s) it rests on; the
    module never invents a claim, cause or number.
  * Total/fail-safe: garbage or empty input yields an empty (well-formed)
    structure, never an exception.
  * No module-level mutable state: the map is per-call and returned.

It reuses — never reimplements — `evidence_grade` (grading, registrable
domains), `contradictions` records as the workflow emits them, and
`investigation_state` / `depth_controller` signals for the unknown set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.core.logging import get_logger

logger = get_logger(__name__)

# Similarity at/above which two established claims are about the same
# assertion and therefore one convergent conclusion rather than two. Matches
# the corroboration band (app.core.evidence_grade.CORROBORATION_SIMILARITY) so
# "same assertion" means the same thing here as it does in grading.
CONVERGENT_SIMILARITY = 0.55

# A convergent conclusion must rest on evidence from at least this many
# distinct registrable domains, OR on a claim the evidence spine already
# counted as independently corroborated. One publisher is a report, not a
# convergence.
MIN_CONVERGENT_DOMAINS = 2

# Query terms that make a question decision/policy-shaped even when the coarse
# classifier returned `analytical`. Kept small and explicit; `comparative` is
# handled via query_type directly.
_DECISION_MARKERS = (
    "should ",
    " ought ",
    "worth investing",
    "invest in",
    "recommend",
    "policy",
    "policymakers",
    "prioritize",
    "trade-off",
    "tradeoff",
    "decision",
    "better to",
    "more effective to",
)

# Cap on each list so the structure stays a brief, not a dump. Deterministic:
# the lists are already ranked before the cap is applied.
MAX_CONCLUSIONS = 6
MAX_COMPETING = 5
MAX_ESTABLISHED = 8
MAX_INFERRED = 8
MAX_UNKNOWN = 8
MAX_IMPLICATIONS = 5

_SEVERITY_SEVERE = 0.60

# Kinds the contradiction engine can emit; each maps to the question that
# actually separates the two positions. A kind we do not recognise is named
# verbatim rather than forced into a category (never mislabel a divergence).
_KIND_DISTINGUISHERS: Dict[str, str] = {
    "numeric": "which figure is correct (the sources cite materially different values)",
    "polarity": "which side is right (the sources make opposite assertions)",
    "temporal": "which reporting period applies (the same measure is cited for different periods)",
    "scope": "which population/geography each figure describes (the scopes differ)",
}


def _norm(text: str) -> str:
    """Deterministic comparison/ordering key for a claim or label."""
    return " ".join(str(text or "").lower().split())


def _domain_of(value: str) -> str:
    try:
        from app.core.evidence_grade import registrable_domain

        return registrable_domain(str(value or ""))
    except Exception:  # a keying failure must never raise into the builder
        return ""


@dataclass
class Conclusion:
    """A claim (or agreeing claims) the evidence jointly supports."""

    claim: str
    grade: str = ""
    corroboration: int = 1
    domains: List[str] = field(default_factory=list)
    supporting_claims: List[str] = field(default_factory=list)
    impact: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "grade": self.grade,
            "corroboration": self.corroboration,
            "domains": list(self.domains),
            "supporting_claims": list(self.supporting_claims),
            "impact": self.impact,
        }


@dataclass
class CompetingExplanation:
    """A real source divergence and what it turns on."""

    topic: str
    position_a: str
    position_b: str
    source_a: str = ""
    source_b: str = ""
    kind: str = ""
    severity: float = 0.0
    resolved: bool = False
    distinguishes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "position_a": self.position_a,
            "position_b": self.position_b,
            "source_a": self.source_a,
            "source_b": self.source_b,
            "kind": self.kind,
            "severity": round(self.severity, 3),
            "resolved": self.resolved,
            "distinguishes": self.distinguishes,
        }


@dataclass
class Implication:
    """A decision-relevant implication, tied to the claims that entail it."""

    statement: str
    claim: str = ""
    domains: List[str] = field(default_factory=list)
    insufficient_evidence: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "claim": self.claim,
            "domains": list(self.domains),
            "insufficient_evidence": self.insufficient_evidence,
        }


@dataclass
class ReasoningMap:
    """The argument structure built from the run's graded evidence."""

    query: str = ""
    query_type: str = ""
    decision_relevant: bool = False
    conclusions: List[Conclusion] = field(default_factory=list)
    competing: List[CompetingExplanation] = field(default_factory=list)
    established: List[str] = field(default_factory=list)
    inferred: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)
    implications: List[Implication] = field(default_factory=list)

    # --- accessors (small, stable surface) ---------------------------------
    def established_claims(self) -> List[str]:
        return list(self.established)

    def inferred_claims(self) -> List[str]:
        return list(self.inferred)

    def unknown_gaps(self) -> List[str]:
        return list(self.unknown)

    def competing_explanations(self) -> List[CompetingExplanation]:
        return list(self.competing)

    def decision_implications(self) -> List[Implication]:
        return list(self.implications)

    @property
    def is_empty(self) -> bool:
        return not (
            self.conclusions or self.competing or self.established
            or self.inferred or self.unknown or self.implications
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "decision_relevant": self.decision_relevant,
            "conclusions": [c.to_dict() for c in self.conclusions],
            "competing": [c.to_dict() for c in self.competing],
            "established": list(self.established),
            "inferred": list(self.inferred),
            "unknown": list(self.unknown),
            "implications": [i.to_dict() for i in self.implications],
        }

    def render_for_report(self) -> str:
        """Deterministic, reader-facing rendering of the structure (markdown).

        Used as the no-writer fallback so the argument the evidence supports is
        never lost when the model omits it. Bullets and short sentences only;
        every line traces to the structure, no new facts.
        """
        if self.is_empty:
            return ""
        lines: List[str] = []
        if self.conclusions:
            lines.append("**What the evidence supports**")
            for c in self.conclusions:
                domains = ", ".join(c.domains[:4]) or "multiple sources"
                lines.append(f"- {c.claim} _(grade {c.grade}; corroborated by {domains})_")
        if self.competing:
            lines.append("")
            lines.append("**Competing explanations**")
            for c in self.competing:
                status = "resolved" if c.resolved else "open"
                lines.append(
                    f"- \"{c.position_a}\" vs \"{c.position_b}\" — {c.kind} "
                    f"disagreement ({status}); {c.distinguishes}."
                )
        if self.established:
            lines.append("")
            lines.append("**Established (independently corroborated)**")
            lines.extend(f"- {c}" for c in self.established)
        if self.inferred:
            lines.append("")
            lines.append("**Inferred (supported but single-source)**")
            lines.extend(f"- {c}" for c in self.inferred)
        if self.unknown:
            lines.append("")
            lines.append("**Unknown (unsettled or uncorroborated)**")
            lines.extend(f"- {c}" for c in self.unknown)
        if self.implications:
            lines.append("")
            lines.append("**Decision-relevant implications**")
            lines.extend(f"- {imp.statement}" for imp in self.implications)
        return "\n".join(lines)

    def render_for_writer(self) -> str:
        """Compact text for the synthesis prompt (deterministic, bounded).

        Returns "" for an empty map so callers can append unconditionally
        without leaving a dangling heading in the prompt.
        """
        if self.is_empty:
            return ""
        lines: List[str] = ["EVIDENCE-GROUNDED REASONING STRUCTURE (measured, not written by you):"]

        if self.established:
            lines.append("")
            lines.append("ESTABLISHED (independently corroborated — state as fact):")
            lines.extend(f"- {c}" for c in self.established)

        if self.inferred:
            lines.append("")
            lines.append("INFERRED (supported but single-source — label as provisional):")
            lines.extend(f"- {c}" for c in self.inferred)

        if self.unknown:
            lines.append("")
            lines.append("UNKNOWN (exhausted/uncovered — state as an open gap, do not guess):")
            lines.extend(f"- {c}" for c in self.unknown)

        if self.conclusions:
            lines.append("")
            lines.append("CONCLUSIONS THE EVIDENCE SUPPORTS (state the one the evidence backs):")
            for c in self.conclusions:
                domains = ", ".join(c.domains[:4]) or "multiple"
                lines.append(
                    f"- {c.claim} [grade {c.grade}, {c.corroboration} independent "
                    f"source(s): {domains}]"
                )

        if self.competing:
            lines.append("")
            lines.append("COMPETING EXPLANATIONS (present both sides; do not average or pick one silently):")
            for c in self.competing:
                tag = "RESOLVED" if c.resolved else "OPEN"
                lines.append(
                    f"- [{tag} {c.kind}] \"{c.position_a}\" vs \"{c.position_b}\"; "
                    f"what separates them: {c.distinguishes}."
                )

        if self.implications:
            lines.append("")
            lines.append("DECISION-RELEVANT IMPLICATIONS (derive only from the established claims above):")
            for imp in self.implications:
                lines.append(f"- {imp.statement}")

        lines.append("")
        lines.append(
            "Every statement above is grounded in the cited evidence pool. Answer the "
            "question by stating the conclusion the evidence actually supports, present "
            "the competing explanations where they exist, and keep established, inferred "
            "and unknown strictly separate. If the evidence does not settle a decision, "
            "say so rather than inventing one."
        )
        return "\n".join(lines)


def _is_decision_relevant(query: str, query_type: str) -> bool:
    qtype = str(query_type or "").strip().lower()
    if qtype in ("comparative", "decision", "policy"):
        return True
    q = f" {(query or '').lower()} "
    return any(marker in q for marker in _DECISION_MARKERS)


def _cluster_conclusions(
    established: List[Dict[str, Any]],
    summary_claims: Sequence[str],
) -> List[Conclusion]:
    """Group established claims about the same assertion into conclusions.

    Clustering uses the shared semantic engine (one similarity matrix over the
    established claims only); a pair at/above `CONVERGENT_SIMILARITY` joins the
    same group. Deterministic: claims are processed in a stable order and the
    union-find is order-independent.
    """
    if not established:
        return []
    claims = [str(f.get("claim", "") or "") for f in established]
    parent = list(range(len(claims)))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(i: int, j: int) -> None:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    try:
        from app.core.semantic import similarity_matrix

        sim = similarity_matrix(claims)
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                try:
                    if float(sim[i, j]) >= CONVERGENT_SIMILARITY:
                        _union(i, j)
                except (TypeError, ValueError, IndexError):
                    continue
    except Exception as exc:  # clustering failure -> each claim its own group
        logger.warning("reasoning_clustering_failed", error=str(exc), exc_info=exc)

    groups: Dict[int, List[int]] = {}
    for i in range(len(claims)):
        groups.setdefault(_find(i), []).append(i)

    from app.core.evidence_completion import claim_impact

    summaries = {_norm(s) for s in (summary_claims or []) if str(s or "").strip()}
    conclusions: List[Conclusion] = []
    for members in groups.values():
        records = [established[i] for i in members]
        # Distinct registrable domains across the whole group — the
        # independence measure, not a URL count.
        domains: List[str] = []
        for record in records:
            for domain in record.get("_domains", []) or []:
                if domain and domain not in domains:
                    domains.append(domain)
        corroboration = max(
            int(record.get("evidence", {}).get("corroboration_count", 1) or 1)
            for record in records
        )
        if len(domains) < MIN_CONVERGENT_DOMAINS and corroboration < MIN_CONVERGENT_DOMAINS:
            continue
        # Representative claim: highest grade, then corroboration, then the
        # deterministic normalized-text order.
        best = max(
            records,
            key=lambda r: (
                {"A": 3, "B": 2, "C": 1, "D": 0}.get(
                    str(r.get("evidence", {}).get("grade", "D")), 0
                ),
                int(r.get("evidence", {}).get("corroboration_count", 1) or 1),
            ),
        )
        rep = str(best.get("claim", "") or "")
        grade = str(best.get("evidence", {}).get("grade", ""))
        impact = max(
            claim_impact(record.get("evidence", {}), summary_claims=summaries)
            for record in records
        )
        conclusions.append(
            Conclusion(
                claim=rep,
                grade=grade,
                corroboration=corroboration,
                domains=domains,
                supporting_claims=sorted({str(r.get("claim", "")) for r in records}),
                impact=impact,
            )
        )

    conclusions.sort(
        key=lambda c: (
            -c.impact,
            -{"A": 3, "B": 2, "C": 1, "D": 0}.get(c.grade, 0),
            -c.corroboration,
            _norm(c.claim),
        )
    )
    return conclusions[:MAX_CONCLUSIONS]


def _competing_explanations(
    contradictions: Sequence[Dict[str, Any]],
) -> List[CompetingExplanation]:
    """One entry per real source divergence, most severe first."""
    out: List[CompetingExplanation] = []
    for c in contradictions or []:
        if not isinstance(c, dict):
            continue
        a = str(c.get("claim_a", "") or "").strip()
        b = str(c.get("claim_b", "") or "").strip()
        if not a or not b:
            continue
        kind = str(c.get("kind", "") or "").strip().lower()
        try:
            severity = float(c.get("severity", 0.0) or 0.0)
        except (TypeError, ValueError):
            severity = 0.0
        out.append(
            CompetingExplanation(
                topic=a,
                position_a=a,
                position_b=b,
                source_a=str(c.get("source_a", "") or ""),
                source_b=str(c.get("source_b", "") or ""),
                kind=kind or "unspecified",
                severity=severity,
                resolved=bool(c.get("resolved", False)),
                distinguishes=_KIND_DISTINGUISHERS.get(
                    kind, "which account the evidence supports"
                ),
            )
        )
    # Unresolved first, then by severity, then deterministic text order.
    out.sort(key=lambda e: (e.resolved, -e.severity, _norm(e.position_a), _norm(e.position_b)))
    return out[:MAX_COMPETING]


def _investigation_unknowns(investigation_state: Any) -> List[str]:
    """Exhausted, still-single-source claims, as human-readable gaps."""
    if not investigation_state:
        return []
    try:
        from app.core.investigation_state import exhausted_limitations

        return [
            str(line)
            for line in exhausted_limitations(investigation_state)
            if str(line).strip()
        ]
    except Exception as exc:  # a signal failure must never raise
        logger.warning("reasoning_investigation_lookup_failed", error=str(exc), exc_info=True)
        return []


def _depth_unknowns(state: Dict[str, Any], facts: Sequence[Any]) -> List[str]:
    """Uncovered axes / thin dimensions from the existing depth signals.

    Consumes `depth_controller`'s public helpers rather than recomputing the
    signals, so the reasoning layer and the stopping policy cannot disagree.
    Any failure degrades to no extra unknowns (never raises).
    """
    if not state:
        return []
    try:
        from app.core.depth_controller import _thin_dimensions, _uncovered_axes

        out: List[str] = []
        for axis in _uncovered_axes(state):
            out.append(
                f"the planned dimension '{axis}' has no supporting evidence "
                "(uncovered research axis)"
            )
        for dim in _thin_dimensions(state):
            out.append(f"evidence for the dimension '{dim}' is thin (weak or single-source)")
        return out
    except Exception as exc:
        logger.warning("reasoning_depth_lookup_failed", error=str(exc), exc_info=True)
        return []


def _build_implications(
    conclusions: List[Conclusion],
    established: List[Dict[str, Any]],
    *,
    decision_relevant: bool,
) -> List[Implication]:
    """Decision-relevant implications, each tied to established claims.

    Derived ONLY from conclusions the evidence independently supports. When the
    query is decision-relevant but no conclusion is established, the honest
    output is a single insufficiency note — never a fabricated recommendation.
    """
    if not decision_relevant:
        return []
    if not conclusions:
        return [
            Implication(
                statement=(
                    "The evidence does not establish a basis for a decision: no claim "
                    "is independently corroborated strongly enough to support a "
                    "recommendation. Treat any conclusion as provisional."
                ),
                insufficient_evidence=True,
            )
        ]
    out: List[Implication] = []
    for conclusion in conclusions[:MAX_IMPLICATIONS]:
        domains = ", ".join(conclusion.domains[:3]) or "multiple sources"
        out.append(
            Implication(
                statement=(
                    f"Given that {conclusion.claim} (corroborated by {domains}), the "
                    "decision should be framed to account for this established finding."
                ),
                claim=conclusion.claim,
                domains=list(conclusion.domains),
            )
        )
    return out


def build_reasoning(
    facts: Sequence[Dict[str, Any]],
    contradictions: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    query: str = "",
    query_type: str = "",
    investigation_state: Any = None,
    sub_questions: Optional[Sequence[Any]] = None,
    depth_state: Optional[Dict[str, Any]] = None,
) -> ReasoningMap:
    """Build the argument structure from graded evidence. Deterministic, total.

    `facts` are the run's facts (verified ones are used for the argument; a
    fact carrying no `verified` flag is treated as usable evidence, matching
    the depth controller's convention). `depth_state` is a `depth_controller`
    state dict (facts + sub_questions + contradictions); when omitted it is
    reconstructed from the arguments so a caller can pass just the essentials.
    An empty or malformed input yields an empty ReasoningMap, never an error.
    """
    contradictions = [c for c in (contradictions or []) if isinstance(c, dict)]
    result = ReasoningMap(
        query=str(query or ""),
        query_type=str(query_type or ""),
    )
    try:
        pool = [
            f for f in (facts or [])
            if isinstance(f, dict) and str(f.get("claim", "") or "").strip()
        ]
        if pool:
            from app.core.evidence_grade import grade_facts, independent_corroboration

            graded = grade_facts(pool, contradictions=contradictions)

            # Attach each record's independent domains once, for clustering.
            established_records: List[Dict[str, Any]] = []
            for record in graded:
                if not isinstance(record, dict):
                    continue
                ev = record.get("evidence")
                if not isinstance(ev, dict):
                    continue
                verified = bool(ev.get("verified"))
                grade = str(ev.get("grade", "D"))
                corroboration = int(ev.get("corroboration_count", 1) or 1)
                contradicted = int(ev.get("contradiction_count", 0) or 0) > 0

                # Source domains: the primary source plus any recorded
                # corroborating URLs, measured by registrable domain.
                urls = [
                    str(record.get("source", "") or ""),
                    *[str(u) for u in (record.get("corroborating_sources") or [])],
                ]
                domains, _ = independent_corroboration(urls)
                record["_domains"] = domains

                if verified and grade in ("A", "B") and not contradicted:
                    established_records.append(record)
                    continue
                if contradicted:
                    continue
                if verified and (ev.get("needs_corroboration") or corroboration < 2):
                    result.inferred.append(str(record.get("claim", "") or ""))
                elif grade == "D":
                    result.inferred.append(str(record.get("claim", "") or ""))

            # Established, deterministically ordered (grade desc, corroboration
            # desc, normalized text) so the writer sees a stable list.
            established_records.sort(
                key=lambda r: (
                    -{"A": 3, "B": 2, "C": 1, "D": 0}.get(
                        str(r.get("evidence", {}).get("grade", "D")), 0
                    ),
                    -int(r.get("evidence", {}).get("corroboration_count", 1) or 1),
                    _norm(str(r.get("claim", ""))),
                )
            )
            result.established = [
                str(r.get("claim", "") or "") for r in established_records
            ][:MAX_ESTABLISHED]

            summary_claims = _summary_claims_from(established_records)
            result.conclusions = _cluster_conclusions(
                established_records, summary_claims
            )

        # Competing explanations come only from real contradiction records.
        result.competing = _competing_explanations(contradictions)

        # Unknown: exhausted investigation gaps, then depth-signal holes. Dedupe
        # while preserving the ranked order, then cap.
        unknowns: List[str] = []
        for item in [
            *_investigation_unknowns(investigation_state),
            *_depth_unknowns(
                depth_state if depth_state is not None else {
                    "facts": pool if pool else [],
                    "sub_questions": list(sub_questions or []),
                    "contradictions": contradictions,
                },
                pool,
            ),
        ]:
            text = " ".join(str(item).split())
            if text and text not in unknowns:
                unknowns.append(text)

        # Unresolved severe contradictions are an explicit unknown: the run
        # could not settle them, so the reader must not read a verdict.
        for c in contradictions:
            try:
                severity = float(c.get("severity", 0.0) or 0.0)
            except (TypeError, ValueError):
                severity = 0.0
            if severity >= _SEVERITY_SEVERE and not c.get("resolved"):
                text = (
                    "an unresolved severe source conflict over "
                    f"\"{str(c.get('claim_a', ''))[:120]}\" vs "
                    f"\"{str(c.get('claim_b', ''))[:120]}\" could not be settled"
                )
                if text not in unknowns:
                    unknowns.append(text)
        result.unknown = unknowns[:MAX_UNKNOWN]

        result.inferred = list(dict.fromkeys(result.inferred))[:MAX_INFERRED]
        result.decision_relevant = _is_decision_relevant(query, query_type)
        result.implications = _build_implications(
            result.conclusions,
            established_records if pool else [],
            decision_relevant=result.decision_relevant,
        )
    except Exception as exc:  # total/fail-safe: return the partial structure
        logger.warning("reasoning_build_failed", error=str(exc), exc_info=True)
    return result


def _summary_claims_from(records: Sequence[Dict[str, Any]]) -> List[str]:
    """Best-effort summary-claim texts for impact scoring.

    The workflow does not pass its summary claim list here; the highest-grade
    established claims are the closest deterministic proxy and keep impact
    ranking meaningful (a corroborated quantitative claim leads).
    """
    return [str(r.get("claim", "") or "") for r in records]
