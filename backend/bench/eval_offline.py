#!/usr/bin/env python3
"""Deterministic OFFLINE golden evaluator for MARS.

Runs the versioned golden query set (bench/golden/queries_v1.json) through the
PRODUCTION research graph with deterministic mocks (bench/mock_pipeline.py
GoldenFakeLLM/GoldenFakeSearch, extending the existing e2e mocks), then scores
the resulting reports with the SAME deterministic scorers the product ships:

  * claim/evidence quality ...... app.core.evidence_grade (grades, corroboration)
  * citation correctness ........ app.agents.evidence_utils.verify_answer_support
                                  + parse_answer_legend + numeric grounding
  * contradiction handling ...... app.core.contradictions
  * answer quality .............. app.agents.answer_quality.evaluate_answer
                                  (+ score_answer_relevance)
  * required sections ........... app.agents.synthesizer.REQUIRED_SECTIONS
  * machine-section guard ....... app.agents.sources.strip_machine_sections

This evaluator is FULLY DETERMINISTIC and NETWORK-FREE. There is no LLM judge
here — live/model-judged quality is measured by bench/run_live.py. Keeping the
two separate is deliberate: an offline gate that depends on a model call is not
a gate.

Usage (from backend/):

    python bench/eval_offline.py
    python bench/eval_offline.py --golden bench/golden/queries_v1.json \
        --thresholds bench/golden/thresholds_v1.json

Exit code 0 when every aggregate and per-category threshold passes, 1 on any
regression, 2 on a setup/validation error. Results:
    bench/results/offline_eval_v1.json   machine-readable
    bench/results/OFFLINE_EVAL_v1.md     readable
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.answer_quality import evaluate_answer, score_answer_relevance
from app.agents.evidence_utils import (
    _significant_quantities,
    numbers_grounded,
    parse_answer_legend,
    verify_answer_support,
)
from app.agents.sources import strip_machine_sections
from app.agents.synthesizer import REQUIRED_SECTIONS
from app.core.config import Settings
from app.core.contradictions import find_contradictions
from app.core.evidence_grade import (
    GRADE_A,
    GRADE_B,
    GRADE_C,
    GRADE_D,
    grade_facts,
    registrable_domain,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Metrics that are compared against thresholds. Each maps to the per-query
# metric name it is averaged from (higher is better).
AGGREGATE_METRICS = (
    "query_type_accuracy",
    "dimension_hit_rate",
    "forbidden_clean_rate",
    "section_presence_rate",
    "citation_resolution_rate",
    "verified_claims_mean",
    "corroborated_claims_mean",
    "distinct_domains_mean",
    "primary_share_mean",
    "grade_ab_share",
    "answer_quality_mean",
    "answer_relevance_mean",
    "support_rate_mean",
    "minimums_pass_rate",
    # Populated only when --with-depth is set; absent otherwise (the markdown
    # writer skips metrics it has no value for).
    "depth_routing_pass_rate",
    # Populated only when --with-contradictions is set.
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "resolution_precision",
    "resolution_recall",
    "resolution_f1",
    "classification_accuracy",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_ratio(num: float, den: float, default: float = 0.0) -> float:
    return round(num / den, 4) if den else default


def _normalize_heading(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _present_headings(report: str) -> set:
    found = set()
    for line in (report or "").splitlines():
        match = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
        if match:
            found.add(_normalize_heading(match.group(1)))
    return found


def _required_sections_present(report: str, required: Sequence[str]) -> List[str]:
    """Which of the named sections are present, alias-tolerant.

    Reuses REQUIRED_SECTIONS' alias table so the evaluator can never disagree
    with the synthesizer about what counts as, say, "Evidence Strength"."""
    present = _present_headings(report)
    missing: List[str] = []
    for canonical in required:
        aliases = REQUIRED_SECTIONS.get(canonical, (canonical,))
        normalized = {_normalize_heading(a) for a in aliases}
        normalized.add(_normalize_heading(canonical))
        if not (normalized & present):
            missing.append(canonical)
    return missing


def _dimensions_hit(required: Sequence[str], sub_questions: Sequence[Dict[str, Any]],
                    facts: Sequence[Dict[str, Any]]) -> int:
    """Loose-match: how many expected dimensions are visible in the plan or
    evidence. A dimension counts when its token appears in a planned axis, a
    planned search_type, a planned question, or a fact's sub_question/claim.
    Deliberately loose (substring, casefolded) — the golden set tests COVERAGE,
    not exact planner wording."""
    haystack_parts: List[str] = []
    for q in sub_questions or []:
        if isinstance(q, dict):
            haystack_parts.extend(str(q.get(k, "") or "") for k in
                                  ("axis", "search_type", "question"))
    for f in facts or []:
        if isinstance(f, dict):
            haystack_parts.extend(str(f.get(k, "") or "") for k in
                                  ("sub_question", "search_type", "claim"))
    haystack = " ".join(haystack_parts).lower()
    hits = 0
    for dim in required:
        if str(dim).lower() in haystack:
            hits += 1
    return hits


def _forbidden_hits(forbidden: Sequence[str], report: str,
                    facts: Sequence[Dict[str, Any]]) -> List[str]:
    """Off-topic senses/domains that leaked into the cited evidence or report.

    A forbidden token matched against source URL+text is a hard signal. The
    generic root of a hyphenated sense ("electrical-transformer" ->
    "electrical_transformer" and "electrical") is checked so wording variants
    are caught. Matches in the report body are recorded too."""
    if not forbidden:
        return []
    source_blob_parts: List[str] = []
    for f in facts or []:
        if isinstance(f, dict):
            source_blob_parts.append(str(f.get("source", "") or ""))
            source_blob_parts.append(str(f.get("claim", "") or ""))
    source_blob = " ".join(source_blob_parts).lower()
    report_low = (report or "").lower()
    hits: List[str] = []
    for term in forbidden:
        variants = {str(term).lower()}
        if "-" in str(term):
            variants.add(str(term).lower().replace("-", " "))
            variants.add(str(term).lower().split("-")[0])
        for variant in variants:
            if variant and variant in source_blob:
                hits.append(f"source:{term}")
                break
        else:
            for variant in variants:
                if variant and variant in report_low:
                    hits.append(f"report:{term}")
                    break
    return hits


def _citation_metrics(answer: str) -> Dict[str, Any]:
    """Citation-legend invariants over the WRITER BODY (machine sections
    excluded by the guard)."""
    body_full, legend = parse_answer_legend(answer or "")
    body = strip_machine_sections(body_full)
    markers = [int(m) for m in re.findall(r"\[(\d+)\]", body)]
    unresolved = sorted({m for m in markers if m not in legend})
    legend_numbers = set(legend.keys())
    # Dangling legend entries: numbered sources the body never cites.
    unused = sorted(n for n in legend_numbers if n not in set(markers))
    total = len(markers)
    resolved = total - sum(1 for m in markers if m in unresolved)
    return {
        "markers_total": total,
        "markers_resolved": resolved,
        "markers_unresolved": unresolved,
        "legend_entries": len(legend_numbers),
        "legend_unused": unused,
        "resolution_rate": _safe_ratio(resolved, total, default=1.0 if not markers else 0.0),
        "no_dangling": len(unresolved) == 0,
        "no_unsupported_numbers": None,  # filled from support check
    }


def _unsupported_number_count(answer: str, facts: Sequence[Dict[str, Any]]) -> int:
    """Count cited body sentences whose significant numbers are not grounded
    in the claims of the source they cite. Body-scoped (machine guard).

    Sentences are split on terminal punctuation AND line breaks, matching
    `answer_quality`'s own framing: a `[n]` marker at the end of one paragraph
    must not drag the next section's numbers into its grounding check."""
    body_full, legend = parse_answer_legend(answer or "")
    body = strip_machine_sections(body_full)
    verified_by_url: Dict[str, List[str]] = {}
    for f in facts or []:
        if not isinstance(f, dict) or not f.get("verified"):
            continue
        url = str(f.get("source", "") or "")
        if url:
            verified_by_url.setdefault(url, []).append(str(f.get("claim", "") or ""))
    failures = 0
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", body):
        sentence = sentence.strip()
        numbers = [int(m) for m in re.findall(r"\[(\d+)\]", sentence)]
        if not numbers:
            continue
        cleaned = re.sub(r"\[\d+\]", "", sentence)
        if not _significant_quantities(cleaned):
            continue
        pool = " ".join(
            claim
            for n in numbers
            for claim in verified_by_url.get(legend.get(n, ""), [])
        )
        if not numbers_grounded(cleaned, pool):
            failures += 1
    return failures


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------

def evaluate_query_run(query: Dict[str, Any], final: Dict[str, Any],
                       degraded: Sequence[str]) -> Dict[str, Any]:
    """Score one completed golden run against the query's expectations."""
    answer = str(final.get("synthesized_answer", "") or "")
    report = str(final.get("final_report", "") or "")
    facts = [f for f in (final.get("facts") or []) if isinstance(f, dict)]
    support = final.get("answer_support") or {}
    quality = final.get("quality") or {}
    contradictions = [c for c in (final.get("contradictions") or []) if isinstance(c, dict)]
    sub_questions = [q for q in (final.get("sub_questions") or []) if isinstance(q, dict)]

    # -- evidence grading (reused production scorer) --
    graded = grade_facts(facts, contradictions=contradictions)
    records = [(g.get("evidence") or {}) for g in graded]
    grade_dist = {g: 0 for g in (GRADE_A, GRADE_B, GRADE_C, GRADE_D)}
    for rec in records:
        grade = str(rec.get("grade", GRADE_D))
        grade_dist[grade] = grade_dist.get(grade, 0) + 1
    verified_claims = sum(1 for f in facts if f.get("verified"))
    corroborated = sum(1 for rec in records if int(rec.get("corroboration_count", 1) or 1) > 1)
    needs_corroboration = sum(1 for rec in records if rec.get("needs_corroboration"))
    domains = {registrable_domain(str(f.get("source", "") or "")) for f in facts}
    domains.discard("")
    from app.agents.sources import primary_source_share

    primary_share = primary_source_share([str(f.get("source", "")) for f in facts])

    # -- citation correctness (writer body, machine-section guarded) --
    citations = _citation_metrics(answer)
    citations["no_unsupported_numbers"] = _unsupported_number_count(answer, facts) == 0
    citations["unsupported_numbers"] = _unsupported_number_count(answer, facts)
    # Independent reuse of the production checker on the writer body: the
    # workflow already runs verify_answer_support on the full answer (captured
    # in `support` below), but re-scoring the guarded body here gives the
    # evaluator its own citation-correctness signal that machine sections can
    # never inflate. `body_answer` reconstructs a citable unit (body + legend)
    # so legend resolution still works after the guard.
    body_full, legend = parse_answer_legend(answer or "")
    guarded_body = strip_machine_sections(body_full)
    legend_block = "\n\n" + "\n\n".join(
        f"[{n}] {url}" for n, url in sorted(legend.items())
    ) if legend else ""
    guarded_support = verify_answer_support(
        guarded_body + legend_block, facts
    )
    citations["production_support_rate"] = guarded_support.get("rate")
    citations["production_uncited"] = guarded_support.get("uncited")

    # -- contradiction handling (reused engine output) --
    resolved = sum(1 for c in contradictions if c.get("resolved"))
    unresolved = len(contradictions) - resolved
    # Independent scan on the fact pool to confirm the engine ran (audit).
    scanned = find_contradictions(facts) if len(facts) >= 2 else []

    # -- answer quality (reused gate) --
    required_sections = list(query.get("required_sections") or [])
    missing_sections = _required_sections_present(report or answer, required_sections)
    sections_present = len(required_sections) - len(missing_sections)
    relevance = score_answer_relevance(query["query"], answer)
    q_eval = evaluate_answer(
        query["query"],
        intent=final.get("intent") or {},
        answer=answer,
        facts=facts,
        answer_support=support,
        citation_health=final.get("citation_health") or {},
        contradictions=contradictions,
        mode=str(final.get("mode", "standard") or "standard"),
    )

    # -- dimensions + off-topic guard --
    dim_hits = _dimensions_hit(query.get("required_dimensions") or [], sub_questions, facts)
    dim_required = int(query.get("min_dimensions_hit", 0))
    forbidden_hits = _forbidden_hits(query.get("forbidden_domains") or [], report, facts)

    # -- query type --
    query_type = str((final.get("orchestration") or {}).get("query_type", "") or "")
    query_type_ok = query_type == query.get("query_type_required")

    # -- minimums --
    minimums = query.get("minimums") or {}
    min_checks = {
        "verified_claims": verified_claims >= int(minimums.get("verified_claims", 0)),
        "corroborated_claims": corroborated >= int(minimums.get("corroborated_claims", 0)),
        "distinct_domains": len(domains) >= int(minimums.get("distinct_domains", 0)),
        "primary_share": primary_share >= float(minimums.get("primary_share", 0.0)),
        "sections": sections_present >= int(minimums.get("sections", 0)),
    }
    citation_invariants = query.get("citation_invariants") or {}
    invariant_checks = {
        "every_marker_resolves": (
            citations["no_dangling"] if citation_invariants.get("every_marker_resolves", True) else True
        ),
        "no_dangling_markers": (
            citations["no_dangling"] if citation_invariants.get("no_dangling_markers", True) else True
        ),
        "no_unsupported_numbers": (
            citations["no_unsupported_numbers"]
            if citation_invariants.get("no_unsupported_numbers", True) else True
        ),
    }
    minimums_ok = (
        all(min_checks.values())
        and all(invariant_checks.values())
        and dim_hits >= dim_required
        and not forbidden_hits
    )

    return {
        "id": query["id"],
        "category": query["category"],
        "query": query["query"],
        "query_type": query_type,
        "query_type_required": query.get("query_type_required"),
        "query_type_ok": query_type_ok,
        "dimensions_required": query.get("required_dimensions") or [],
        "dimensions_hit": dim_hits,
        "dimensions_min": dim_required,
        "dimensions_ok": dim_hits >= dim_required,
        "forbidden_hits": forbidden_hits,
        "forbidden_ok": not forbidden_hits,
        "required_sections": required_sections,
        "missing_sections": missing_sections,
        "sections_present": sections_present,
        "section_presence_rate": _safe_ratio(sections_present, len(required_sections), 1.0),
        "citation": citations,
        "evidence": {
            "grade_distribution": grade_dist,
            "grade_ab_share": _safe_ratio(grade_dist[GRADE_A] + grade_dist[GRADE_B], len(facts), 0.0),
            "verified_claims": verified_claims,
            "corroborated_claims": corroborated,
            "needs_corroboration": needs_corroboration,
            "distinct_domains": len(domains),
            "primary_share": primary_share,
            "total_claims": len(facts),
        },
        "contradictions": {
            "total": len(contradictions),
            "resolved": resolved,
            "unresolved": unresolved,
            "independent_scan_count": len(scanned),
        },
        "answer_quality": {
            "overall": q_eval.overall,
            "passed": q_eval.passed,
            "accuracy": q_eval.accuracy,
            "relevance": q_eval.relevance,
            "evidence": q_eval.evidence,
            "clarity": q_eval.clarity,
            "reasoning": q_eval.reasoning,
            "answer_relevance": round(relevance, 4),
            "failures": q_eval.failures[:5],
        },
        "support_rate": support.get("rate") if support.get("rate") is not None else 1.0,
        "minimums_checks": min_checks,
        "invariant_checks": invariant_checks,
        "minimums_ok": minimums_ok,
        "degraded": list(degraded),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_query(query: Dict[str, Any], settings: Settings) -> Dict[str, Any]:
    from bench.golden import fixtures_v1
    from bench.mock_pipeline import GoldenFakeLLM, GoldenFakeSearch

    # Offline guarantees: no provider calls, no cache masking, no live citation
    # HEAD requests. Imports are local so the module stays import-light.
    from app.core import llm_cache
    from app.agents import citation_check

    llm_cache._force_disabled = True

    async def _offline_citations(answer, answer_support, **kwargs):
        return {"checked": 0, "sources": [], "summary": {}, "enabled": False}

    original_check = citation_check.check_citations
    citation_check.check_citations = _offline_citations

    from app.core.degradation import reset_fallbacks, take_fallbacks
    from app.core.usage import clear_run_usage, start_run_usage
    from app.graph.workflow import build_initial_state, create_workflow, graph_recursion_limit

    reset_fallbacks()
    topic = fixtures_v1.TOPIC_BY_QUERY_ID.get(query["id"])
    llm = GoldenFakeLLM(settings, query_id=query["id"], query=query["query"],
                        critic_pass_on_iteration=1)
    search = GoldenFakeSearch(settings, query=query["query"])
    workflow = create_workflow(llm, search_client=search)
    state = build_initial_state(query["query"], settings.max_iterations, mode="standard")

    usage = start_run_usage(f"golden-{query['id']}", settings, mode="standard")
    t0 = time.perf_counter()
    final: Dict[str, Any] = dict(state)
    error: Optional[str] = None
    try:
        async for snapshot in workflow.astream(
            state, stream_mode="values",
            config={"recursion_limit": graph_recursion_limit(state)},
        ):
            final = {**final, **{k: v for k, v in snapshot.items() if v}}
    except Exception as exc:  # one query must not kill the whole evaluation
        error = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        clear_run_usage()
        citation_check.check_citations = original_check
    wall = time.perf_counter() - t0
    degraded = take_fallbacks()

    if error:
        return {
            "id": query["id"], "category": query["category"], "query": query["query"],
            "error": error, "wall_sec": round(wall, 3), "topic": topic,
            "minimums_ok": False, "query_type_ok": False, "forbidden_ok": False,
            "section_presence_rate": 0.0, "citation": {"resolution_rate": 0.0},
            "evidence": {"grade_ab_share": 0.0, "verified_claims": 0,
                         "corroborated_claims": 0, "distinct_domains": 0,
                         "primary_share": 0.0},
            "answer_quality": {"overall": 0.0, "answer_relevance": 0.0},
            "support_rate": 0.0, "dimensions_hit": 0, "dimensions_ok": False,
        }

    result = evaluate_query_run(query, final, degraded)
    result["error"] = None
    result["wall_sec"] = round(wall, 3)
    result["topic"] = topic
    result["llm_calls"] = len(llm.calls)
    return result


def aggregate(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean of each aggregate metric across all successfully-scored queries."""
    scored = [r for r in per_query if r.get("error") is None]
    n = len(scored)

    def mean(fn: Callable[[Dict[str, Any]], float]) -> float:
        if not n:
            return 0.0
        return round(sum(fn(r) for r in scored) / n, 4)

    metrics = {
        "query_type_accuracy": mean(lambda r: 1.0 if r["query_type_ok"] else 0.0),
        "dimension_hit_rate": mean(
            lambda r: _safe_ratio(r["dimensions_hit"], max(1, len(r["dimensions_required"])), 1.0)
        ),
        "forbidden_clean_rate": mean(lambda r: 1.0 if r["forbidden_ok"] else 0.0),
        "section_presence_rate": mean(lambda r: float(r["section_presence_rate"])),
        "citation_resolution_rate": mean(lambda r: float(r["citation"]["resolution_rate"])),
        "verified_claims_mean": mean(lambda r: float(r["evidence"]["verified_claims"])),
        "corroborated_claims_mean": mean(lambda r: float(r["evidence"]["corroborated_claims"])),
        "distinct_domains_mean": mean(lambda r: float(r["evidence"]["distinct_domains"])),
        "primary_share_mean": mean(lambda r: float(r["evidence"]["primary_share"])),
        "grade_ab_share": mean(lambda r: float(r["evidence"]["grade_ab_share"])),
        "answer_quality_mean": mean(lambda r: float(r["answer_quality"]["overall"])),
        "answer_relevance_mean": mean(lambda r: float(r["answer_quality"]["answer_relevance"])),
        "support_rate_mean": mean(lambda r: float(r["support_rate"])),
        "minimums_pass_rate": mean(lambda r: 1.0 if r["minimums_ok"] else 0.0),
    }
    return {
        "queries_total": len(per_query),
        "queries_scored": n,
        "queries_failed": len(per_query) - n,
        "metrics": metrics,
    }


def per_category(per_query: List[Dict[str, Any]]) -> Dict[str, Any]:
    cats: Dict[str, List[Dict[str, Any]]] = {}
    for r in per_query:
        cats.setdefault(r["category"], []).append(r)
    out: Dict[str, Any] = {}
    for cat, rows in sorted(cats.items()):
        agg = aggregate(rows)
        out[cat] = {
            "queries": len(rows),
            "metrics": agg["metrics"],
        }
    return out


def check_thresholds(aggregate_result: Dict[str, Any],
                     category_result: Dict[str, Any],
                     thresholds: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list of threshold violations ([] = pass)."""
    failures: List[Dict[str, Any]] = []
    agg_metrics = aggregate_result["metrics"]
    for name, floor in (thresholds.get("aggregate") or {}).items():
        actual = agg_metrics.get(name)
        if actual is None:
            failures.append({"scope": "aggregate", "metric": name,
                             "actual": None, "floor": floor,
                             "reason": "metric not produced"})
            continue
        if float(actual) < float(floor):
            failures.append({"scope": "aggregate", "metric": name,
                             "actual": float(actual), "floor": float(floor)})
    for cat, spec in (thresholds.get("per_category") or {}).items():
        cat_metrics = (category_result.get(cat) or {}).get("metrics") or {}
        for name, floor in (spec or {}).items():
            actual = cat_metrics.get(name)
            if actual is None:
                failures.append({"scope": cat, "metric": name, "actual": None,
                                 "floor": floor, "reason": "metric not produced"})
                continue
            if float(actual) < float(floor):
                failures.append({"scope": cat, "metric": name,
                                 "actual": float(actual), "floor": float(floor)})
    return failures


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_json(report: Dict[str, Any], out_dir: Path = RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "offline_eval_v1.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def write_markdown(report: Dict[str, Any], out_dir: Path = RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = report["aggregate"]["metrics"]
    lines: List[str] = []
    a = lines.append
    a("# MARS Offline Golden Evaluation (v1)")
    a("")
    a(f"*Generated {report['ran_at'][:19]}Z — deterministic, network-free.*")
    a("")
    a(f"Golden set: `{report['golden_path']}`  ")
    a(f"Thresholds: `{report['thresholds_path']}`  ")
    a(f"Queries: {report['aggregate']['queries_total']} "
      f"(scored {report['aggregate']['queries_scored']}, "
      f"failed {report['aggregate']['queries_failed']})")
    a("")
    a(f"**Result: {'PASS' if report['passed'] else 'FAIL'}** "
      f"({len(report['threshold_failures'])} threshold violation(s))")
    a("")
    a("## Aggregate metrics")
    a("")
    a("| Metric | Actual | Floor | Status |")
    a("|---|---|---|---|")
    floors = report["thresholds"].get("aggregate", {})
    for name in AGGREGATE_METRICS:
        actual = agg.get(name)
        floor = floors.get(name)
        if actual is None:
            continue
        status = "ok"
        if floor is not None:
            status = "ok" if float(actual) >= float(floor) else "FAIL"
        a(f"| {name} | {actual:.4f} | {'' if floor is None else f'{float(floor):.4f}'} | {status} |")
    a("")
    a("## Per category")
    a("")
    a("| Category | Queries | Quality | Support | Citation res. | Min-pass |")
    a("|---|---|---|---|---|---|")
    for cat, spec in report["per_category"].items():
        m = spec["metrics"]
        a(f"| {cat} | {spec['queries']} | {m['answer_quality_mean']:.3f} | "
          f"{m['support_rate_mean']:.3f} | {m['citation_resolution_rate']:.3f} | "
          f"{m['minimums_pass_rate']:.3f} |")
    a("")
    if report["threshold_failures"]:
        a("## Threshold violations")
        a("")
        for f in report["threshold_failures"]:
            a(f"- **{f['scope']} / {f['metric']}**: actual {f['actual']} < floor {f['floor']}")
        a("")
    if report.get("contradictions"):
        cagg = report["contradictions"]["aggregate"]
        a("## Contradiction detection / resolution")
        a("")
        a(f"Labeled cases: {cagg['cases_total']} "
          f"(correct {cagg['cases_correct']}, incorrect {cagg['cases_incorrect']})")
        a("")
        a("| id | expected | observed | ok |")
        a("|---|---|---|---|")
        for row in report["contradictions"]["cases"]:
            a(f"| {row['id']} | {row['expected_label']} | {row['observed_label']} | "
              f"{'Y' if row['correct'] else 'N'} |")
        a("")
    a("## Per query")
    a("")
    a("| id | cat | type ok | dims | sections | cit. res | verified | corrob | domains | quality | min ok |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in report["queries"]:
        if r.get("error"):
            a(f"| {r['id']} | {r['category']} | ERROR | | | | | | | | {r['error'][:60]} |")
            continue
        a(f"| {r['id']} | {r['category']} | "
          f"{'Y' if r['query_type_ok'] else 'N'} | "
          f"{r['dimensions_hit']}/{len(r['dimensions_required'])} | "
          f"{r['sections_present']}/{len(r['required_sections'])} | "
          f"{r['citation']['resolution_rate']:.2f} | "
          f"{r['evidence']['verified_claims']} | "
          f"{r['evidence']['corroborated_claims']} | "
          f"{r['evidence']['distinct_domains']} | "
          f"{r['answer_quality']['overall']} | "
          f"{'Y' if r['minimums_ok'] else 'N'} |")
    a("")
    a("## Determinism / separation from live judging")
    a("")
    a("- Every number here comes from deterministic, LLM-free scorers and a")
    a("  scripted offline pipeline. No model is called and the network is not")
    a("  touched. Live/model-judged quality stays in `bench/run_live.py`.")
    a("- Citation and numeric metrics are computed over the writer body only:")
    a("  `sources.strip_machine_sections` removes the machine-appended sections")
    a("  (Evidence integrity, Source ledger, Sources, Limitations, ...) so the")
    a("  accounting is never scored as if it were unsupported prose.")
    path = out_dir / "OFFLINE_EVAL_v1.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def evaluate(golden_path: Optional[str] = None,
                   thresholds_path: Optional[str] = None,
                   out_dir: Optional[Path] = None,
                   with_depth: bool = False,
                   depth_thresholds_path: Optional[str] = None,
                   with_contradictions: bool = False,
                   contradiction_thresholds_path: Optional[str] = None) -> Dict[str, Any]:
    from bench.golden.loader import load_queries, load_thresholds

    queries = load_queries(golden_path)
    thresholds = load_thresholds(thresholds_path, queries)

    tmp = tempfile.mkdtemp(prefix="mars-golden-")
    settings = Settings(
        groq_api_key="golden-offline",
        database_url=f"{tmp}/golden.db",
        _env_file=None,
    )

    per_query: List[Dict[str, Any]] = []
    for query in queries:
        result = await run_query(query, settings)
        per_query.append(result)
        status = "ERROR" if result.get("error") else ("ok" if result["minimums_ok"] else "gap")
        print(f"[{status:>5}] {result['id']:<28} "
              f"q={result.get('answer_quality', {}).get('overall', '-')} "
              f"min={'Y' if result.get('minimums_ok') else 'N'}")

    aggregate_result = aggregate(per_query)
    category_result = per_category(per_query)
    failures = check_thresholds(aggregate_result, category_result, thresholds)

    depth_report: Optional[Dict[str, Any]] = None
    if with_depth:
        # Adaptive-depth routing coverage lives in its own evaluator
        # (bench/eval_depth.py); fold its per-scenario results and threshold
        # failures into THIS gate so one command is the whole offline gate.
        from bench import eval_depth

        depth_report = eval_depth.evaluate(depth_thresholds_path)
        aggregate_result["metrics"].update(depth_report["aggregate"]["metrics"])
        for f in depth_report["threshold_failures"]:
            failures.append({"scope": "depth", "metric": f["metric"],
                             "actual": f["actual"], "floor": f["floor"]})
        for row in depth_report["scenarios"]:
            status = "ok" if row["passed"] else "FAIL"
            print(f"[depth {status:>4}] {row['id']:<42} "
                  f"decision={row['actual_decision']:<8}")

    contradiction_report: Optional[Dict[str, Any]] = None
    if with_contradictions:
        # Contradiction detection/resolution precision+recall lives in its own
        # evaluator (bench/eval_contradictions.py) over a labeled fixture set;
        # fold its aggregate metrics and threshold failures into THIS gate so
        # one command remains the whole offline gate.
        from bench import eval_contradictions

        contradiction_report = eval_contradictions.evaluate(contradiction_thresholds_path)
        aggregate_result["metrics"].update(contradiction_report["aggregate"]["metrics"])
        for f in contradiction_report["threshold_failures"]:
            failures.append({"scope": "contradictions", "metric": f["metric"],
                             "actual": f["actual"], "floor": f["floor"]})
        for row in contradiction_report["cases"]:
            status = "ok" if row["correct"] else "FAIL"
            print(f"[contra {status:>4}] {row['id']:<44} "
                  f"expected={row['expected_label']:<18} observed={row['observed_label']}")

    report = {
        "suite": "mars-golden-offline-eval",
        "version": thresholds.get("version", "v1"),
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "golden_path": str(golden_path or "bench/golden/queries_v1.json"),
        "thresholds_path": str(thresholds_path or "bench/golden/thresholds_v1.json"),
        "aggregate": aggregate_result,
        "per_category": category_result,
        "queries": per_query,
        "depth": depth_report,
        "contradictions": contradiction_report,
        "thresholds": thresholds,
        "threshold_failures": failures,
        "passed": not failures,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default=None, help="golden query JSON path")
    parser.add_argument("--thresholds", default=None, help="thresholds JSON path")
    parser.add_argument("--out", default=None, help="results directory")
    parser.add_argument("--with-depth", action="store_true",
                        help="also run the adaptive-depth routing evaluator")
    parser.add_argument("--depth-thresholds", default=None,
                        help="adaptive-depth thresholds JSON path")
    parser.add_argument("--with-contradictions", action="store_true",
                        help="also run the contradiction detection/resolution evaluator")
    parser.add_argument("--contradiction-thresholds", default=None,
                        help="contradiction thresholds JSON path")
    args = parser.parse_args()

    try:
        report = asyncio.run(evaluate(args.golden, args.thresholds,
                                      Path(args.out) if args.out else None,
                                      with_depth=args.with_depth,
                                      depth_thresholds_path=args.depth_thresholds,
                                      with_contradictions=args.with_contradictions,
                                      contradiction_thresholds_path=args.contradiction_thresholds))
    except Exception as exc:
        print(f"golden evaluation failed to run: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else RESULTS_DIR
    json_path = write_json(report, out_dir)
    md_path = write_markdown(report, out_dir)

    print("=" * 62)
    print("MARS offline golden evaluation (v1)")
    print("=" * 62)
    agg = report["aggregate"]["metrics"]
    for name in AGGREGATE_METRICS:
        print(f"  {name:<28} {agg.get(name)}")
    print(f"\nthreshold failures: {len(report['threshold_failures'])}")
    for f in report["threshold_failures"]:
        print(f"  - {f['scope']}/{f['metric']}: {f['actual']} < {f['floor']}")
    print(f"results: {json_path}")
    print(f"report:  {md_path}")
    print(f"RESULT: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
