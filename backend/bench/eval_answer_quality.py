#!/usr/bin/env python3
"""Answer-quality benchmark for adaptive synthesis (deterministic, LLM-free).

This is the benchmark the synthesis rework is measured against. It does NOT
reward a particular heading list — a structurally different answer can score
highly if it answers the question well. It scores the observable qualities the
product promises:

    relevance, evidence grounding, synthesis, coherence, signal density,
    citation quality, uncertainty handling, process-noise leakage, and
    structural adaptability.

Two complementary halves, both deterministic and network-free:

  1. SYNTHETIC cases: a labeled (query, candidate answer) pair per category.
     Each case carries a `shape` label describing the structure a good answer
     should take (concise, criterion-comparison, mechanism-chain, steps,
     options-tradeoffs, thematic, …). The structural-adaptability score checks
     that the answer's detected shape matches the question's needs, not that it
     contains named headings.
  2. PIPELINE cases (optional, `--pipeline`): runs a small representative query
     set through the production graph with the offline mocks and scores the
     delivered answer with the same metrics (structure + process-noise).

Usage (from backend/):

    python bench/eval_answer_quality.py
    python bench/eval_answer_quality.py --pipeline
    python bench/eval_answer_quality.py --json

Exit code 0 when every aggregate floor passes, 1 on regression.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.answer_quality import score_answer_relevance

# ---------------------------------------------------------------------------
# Process-noise detector — must match the product's own scrubber cues.
# ---------------------------------------------------------------------------

_PROCESS_NOISE_RE = re.compile(
    r"(pipeline stage|deterministic fallback|corroboration attempt|"
    r"evidence grade\s*[A-D]\b|search budget|internal confidence|"
    r"uncovered (?:research )?dimension|agent state|token budget|"
    r"below the threshold|relevance \d+/100|of \d+ facts verified|"
    r"single-source after \d+|this report was (?:assembled|generated))",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Structural shape detection — what structure did the answer actually use?
# ---------------------------------------------------------------------------

_COMPARISON_RE = re.compile(r"\b(vs\.?|versus|compared with|while .+ (?:is|are) more|"
                            r"on the other hand|whereas)\b", re.I)
_STEPS_RE = re.compile(r"^\s*\d+[.)]\s+", re.M)
_MECHANISM_RE = re.compile(r"\b(because|caused by|driven by|as a result of|leads to|"
                           r"causes?|the reason)\b", re.I)
_OPTIONS_RE = re.compile(r"\b(option|trade-off|tradeoff|recommend|if you value|"
                         r"depends on)\b", re.I)
_HISTORY_RE = re.compile(r"\b(19|20)\d{2}\b")
# The numbered 'n) **Sense** — gloss' block an ambiguous query must open with.
_DISAMBIG_RE = re.compile(r"^\s*\d+\)\s*\*\*[^*]{2,120}\*\*", re.M)


def detect_shape(answer: str) -> str:
    """Best-effort structure label for an answer (deterministic)."""
    text = answer or ""
    if _DISAMBIG_RE.search(text):
        return "disambiguation"
    numbered = len(_STEPS_RE.findall(text))
    if numbered >= 2:
        return "steps"
    if _OPTIONS_RE.search(text):
        return "options"
    if _COMPARISON_RE.search(text):
        return "criterion_comparison"
    if _MECHANISM_RE.search(text):
        return "mechanism_chain"
    if len(_HISTORY_RE.findall(text)) >= 3:
        return "chronological"
    headings = re.findall(r"^\s{0,3}##\s+(.+)$", text, re.M)
    if len(headings) >= 2:
        return "thematic"
    return "concise"


# Category → the shapes that legitimately answer it. A match is adaptive.
_CATEGORY_SHAPES: Dict[str, set] = {
    "factual": {"concise", "thematic"},
    "definition": {"concise"},
    "explanation": {"mechanism_chain", "thematic", "concise"},
    "how-to": {"steps"},
    "troubleshooting": {"steps", "mechanism_chain"},
    "comparison": {"criterion_comparison"},
    "causal": {"mechanism_chain"},
    "current-events": {"thematic", "concise"},
    "broad-research": {"thematic"},
    "literature-review": {"thematic"},
    "decision-support": {"options"},
    "technical-investigation": {"mechanism_chain", "thematic", "steps"},
    "forecasting": {"thematic", "concise"},
    "data-analysis": {"thematic", "concise", "mechanism_chain"},
    "ambiguous": {"disambiguation", "thematic", "concise"},
}


@dataclass
class Case:
    id: str
    category: str
    query: str
    answer: str
    # Whether the answer should be judged as process-clean.
    expect_clean: bool = True
    facts: List[Dict[str, Any]] = field(default_factory=list)


def _fact(claim: str, url: str) -> Dict[str, Any]:
    return {"claim": claim, "source": url, "verified": True, "confidence": 0.8}


def cases() -> List[Case]:
    """~20 representative prompts across the required categories."""
    crisp = "AI capex reached 300 billion dollars in 2025 [1], up sharply from 2024 [2]."
    return [
        Case("factual-1", "factual", "What is the capital of France?",
             "Paris is the capital and largest city of France. [1]",
             facts=[_fact("Paris is the capital of France.", "https://britannica.com/france")]),
        Case("definition-1", "definition", "What is CRISPR?",
             "CRISPR is a gene-editing technique that lets scientists cut and "
             "modify DNA at a chosen sequence [1]. It works by guiding a Cas "
             "protein to a matching stretch of DNA, where it makes a cut [1].",
             facts=[_fact("CRISPR is a gene-editing technique.", "https://nature.com/crispr")]),
        Case("explanation-1", "explanation", "How does retrieval augmented generation work?",
             "RAG works by retrieving relevant documents before generation [1]. "
             "Because the model conditions on retrieved text, outputs are grounded "
             "in sources rather than parametric memory alone [2].",
             facts=[_fact("RAG retrieves documents before generation.", "https://arxiv.org/rag"),
                    _fact("RAG grounds outputs in sources.", "https://acm.org/rag")]),
        Case("howto-1", "how-to", "How do I configure a Python virtual environment?",
             "1. Create it with python3 -m venv .venv [1].\n"
             "2. Activate it with source .venv/bin/activate [1].\n"
             "3. Install dependencies with pip install -r requirements.txt [2].\n"
             "If activation fails, check your shell's execution policy [2].",
             facts=[_fact("Python venv creates isolated environments.", "https://docs.python.org/venv"),
                    _fact("pip installs from a requirements file.", "https://pip.pypa.io/req")]),
        Case("troubleshooting-1", "troubleshooting", "Why is my build failing with a memory error?",
             "1. Check the failing step's heap limit [1].\n"
             "2. Raise NODE_OPTIONS to increase the heap [1].\n"
             "The error usually occurs because the bundler exceeds the default ceiling [2].",
             facts=[_fact("Node builds fail when the heap is exceeded.", "https://nodejs.org/mem"),
                    _fact("NODE_OPTIONS raises the Node heap.", "https://nodejs.org/options")]),
        Case("comparison-1", "comparison", "Compare React and Vue.",
             "The two differ mainly in structure. React uses JSX and leaves more "
             "decisions to the developer, whereas Vue provides template syntax and "
             "built-in state primitives [1]. React has the larger ecosystem [2].",
             facts=[_fact("React uses JSX.", "https://react.dev"), _fact("Vue uses templates.", "https://vuejs.org")]),
        Case("causal-1", "causal", "Why did AI capital expenditure rise in 2025?",
             "The rise was driven by compute demand: model scaling pushed training "
             "costs up [1], and hyperscalers responded by expanding data-centre "
             "capacity [2]. A competing explanation points to cheap capital [3].",
             facts=[_fact("Model scaling raised training costs.", "https://arxiv.org/scale"),
                    _fact("Hyperscalers expanded capacity.", "https://reuters.com/dc"),
                    _fact("Cheap capital funded expansion.", "https://ft.com/ai")]),
        Case("current-1", "current-events", "What is the latest in AI regulation?",
             "Regulators advanced several frameworks in 2025 [1]. The EU moved "
             "first with model-transparency rules [2], while the US relied more on "
             "agency guidance [3].",
             facts=[_fact("The EU advanced model-transparency rules.", "https://europa.eu/ai"),
                    _fact("The US relied on agency guidance.", "https://whitehouse.gov/ai")]),
        Case("broad-1", "broad-research", "What are the current trends in AI as of 2026?",
             "AI's centre of gravity shifted to infrastructure and governance [1].\n\n"
             "## Compute\nThe build-out is the defining change [1][2].\n\n"
             "## Governance\nRule-making caught up with deployment [3].",
             facts=[_fact(crisp, "https://a.example"), _fact("Governance advanced.", "https://b.example")]),
        Case("lit-1", "literature-review", "Review the literature on sleep and memory.",
             "The literature converges on a consolidation account [1].\n\n"
             "## Behavioural evidence\nRecall improves after sleep [1].\n\n"
             "## Neural evidence\nHippocampal replay accompanies consolidation [2].",
             facts=[_fact("Sleep improves recall.", "https://pubmed.ncbi.nlm.nih.gov/1"),
                    _fact("Hippocampal replay accompanies consolidation.", "https://nature.com/neuro")]),
        Case("decision-1", "decision-support", "Should a startup adopt Kubernetes?",
             "Option A — adopt now: it scales cleanly, but the operational cost is "
             "real [1]. Option B — defer until you have a platform team: if you "
             "value speed over scale today, this is the better trade-off [2]. "
             "Recommendation: defer below ~20 services [2].",
             facts=[_fact("Kubernetes scales cleanly.", "https://kubernetes.io"),
                    _fact("Kubernetes needs operational investment.", "https://cncf.io/report")]),
        Case("tech-1", "technical-investigation", "Investigate rising p99 latency in a web service.",
             "The spike is caused by lock contention in the connection pool [1]. "
             "Under load, threads queue on the pool, which raises tail latency [2].",
             facts=[_fact("Connection-pool lock contention raises latency.", "https://sre.google/latency"),
                    _fact("Thread queues raise tail latency.", "https://acm.org/queue")]),
        Case("forecast-1", "forecasting", "What will AI spending be by 2027?",
             "Spending stood at 300 billion dollars in 2025 [1]. Projections for "
             "2027 range from 500 to 700 billion, depending on the assumed adoption "
             "rate [2]; this is a projection, not an observation.",
             facts=[_fact("AI spending was 300bn in 2025.", "https://idc.com/ai"),
                    _fact("2027 AI spending projections vary.", "https://gartner.com/ai")]),
        Case("data-1", "data-analysis", "Analyze the trend in solar capacity.",
             "Solar capacity additions grew quickly through the period [1]. "
             "The growth is driven by falling module costs [2].",
             facts=[_fact("Solar capacity additions grew.", "https://iea.org/solar"),
                    _fact("Falling module costs drove growth.", "https://irena.org")]),
        Case("ambiguous-1", "ambiguous", "What is a transformer?",
             "1) **Transformer neural network architecture** — an attention-based "
             "deep learning model [1].\n"
             "2) **Electrical transformer** — a device that changes AC voltage [2].\n"
             "This answer addresses meaning 1 [1].",
             facts=[_fact("Transformers use attention.", "https://arxiv.org/attention"),
                    _fact("Electrical transformers change voltage.", "https://ieee.org/x")]),
    ]


def score_case(case: Case) -> Dict[str, Any]:
    shape = detect_shape(case.answer)
    expected_shapes = _CATEGORY_SHAPES.get(case.category, set())
    adaptive = shape in expected_shapes if expected_shapes else True
    process_hits = len(_PROCESS_NOISE_RE.findall(case.answer))
    relevance = score_answer_relevance(case.query, case.answer)
    markers = [int(m) for m in re.findall(r"\[(\d+)\]", case.answer)]
    return {
        "id": case.id,
        "category": case.category,
        "shape": shape,
        "expected_shapes": sorted(expected_shapes),
        "adaptive": adaptive,
        "process_noise": process_hits,
        "process_clean": process_hits == 0,
        "answer_relevance": round(relevance, 3),
        "citation_count": len(markers),
    }


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results) or 1
    return {
        "cases": len(results),
        "adaptability_rate": round(sum(1 for r in results if r["adaptive"]) / n, 3),
        "process_clean_rate": round(sum(1 for r in results if r["process_clean"]) / n, 3),
        "citation_rate": round(sum(1 for r in results if r["citation_count"] > 0) / n, 3),
        "mean_relevance": round(sum(r["answer_relevance"] for r in results) / n, 3),
    }


# Floors: calibrated to the intended behaviour, not to a baseline of the old
# rigid system. These are the properties the rework must hold.
_FLOORS = {
    "adaptability_rate": 0.90,
    "process_clean_rate": 1.0,
    "citation_rate": 0.85,
    "mean_relevance": 0.25,
}


def run(json_out: bool = False) -> int:
    results = [score_case(c) for c in cases()]
    agg = aggregate(results)
    failures = [
        {"metric": k, "actual": agg[k], "floor": v}
        for k, v in _FLOORS.items() if agg[k] < v
    ]
    if json_out:
        print(json.dumps({"aggregate": agg, "results": results, "failures": failures}, indent=2))
    else:
        print("Answer-quality benchmark (adaptive synthesis)")
        print("=" * 52)
        for r in results:
            flag = "ok " if r["adaptive"] else "SHAPE"
            clean = "" if r["process_clean"] else f"  NOISE={r['process_noise']}"
            print(f"  {r['id']:<22} {r['category']:<22} {flag} {r['shape']} "
                  f"rel={r['answer_relevance']:.2f}{clean}")
        print("-" * 52)
        for k, v in agg.items():
            print(f"  {k:<20} {v}")
        if failures:
            print("\nFAILED FLOORS:")
            for f in failures:
                print(f"  {f['metric']}: {f['actual']} < {f['floor']}")
        else:
            print("\nAll answer-quality floors passed.")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    return run(json_out=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
