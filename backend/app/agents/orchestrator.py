"""Adaptive Orchestrator (vision Feature 01).

What it does now that it did not before
---------------------------------------
The previous orchestrator scored complexity with keyword markers and mapped the
score to a sub-question count. Useful, but it produced a single number and
stopped there, so three things the rest of the pipeline needed were left
undecided (despite the docstring promising them): research dimensions, a
budget, and stop conditions.

Worse, the number it produced was ignored downstream — `planner_agent` hard-cut
its plan to `cleaned[:5]` regardless of `target_agents`, so on a low-complexity
query the orchestrator asked for 3 angles and the planner shipped 5. The
adaptive layer was decorative. `plan_targets()` here plus the `target_count`
argument added to the planner closes that loop.

Everything is still heuristic and LLM-free by design: spending a model call to
decide how many model calls to spend is a bad trade, and the signals that
matter (question shape, dimension count, recency need) are visible in the text.

Public surface is backwards compatible: `score_complexity`, `classify_query_type`,
`orchestrate`, `ComplexityScore`, `OrchestrationPlan`, `MODE_PRESETS`,
`VALID_MODES`, `LEVEL_TARGETS` all keep their previous names, fields and
meanings. New fields are additive.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Lexical signals
# ---------------------------------------------------------------------------

COMPARATIVE_MARKERS = (
    " vs ", " versus ", "compare", "compared", "comparison",
    "better", "worse", "difference between", "which is better",
    "alternative to", "instead of", "trade-offs between",
)
ANALYTICAL_MARKERS = (
    "should", "why ", "how do", "how does", "impact", "effect of",
    "consequence", "implication", "trade-off", "tradeoff", "evaluate",
    "assess", "risk", "strategy", "policy", "influence", "feasible",
    "viable", "worth it", "justify", "cause", "drives", "predict",
)
EXPLORATORY_MARKERS = (
    "overview", "landscape", "state of the art", "survey", "future of",
    "trends", "evolution", "history of", "current state", "ecosystem",
)

# Decision framing: the question is not "what is true" but "what should we do".
DECISION_MARKERS = (
    "should we", "should i", "should our", "should the", "worth investing",
    "invest", "recommend", "decide", "decision", "choose between",
    "go ahead", "buy or", "build or buy", "prioritize", "budget for",
)

# Recency: an answer that is 3 years stale is a wrong answer.
RECENCY_MARKERS = (
    "latest", "current", "currently", "recent", "recently", "today",
    "now", "this year", "as of", "up to date", "newest", "2024", "2025",
    "2026", "upcoming", "next year", "state of",
)

# Quantitative: the question demands numbers, so a plan without a statistical
# angle cannot answer it.
QUANTITATIVE_MARKERS = (
    "how much", "how many", "cost", "price", "market size", "growth",
    "rate", "percentage", "percent", "share", "revenue", "forecast",
    "projection", "statistics", "data", "benchmark", "roi", "capacity",
    "emissions", "budget", "salary", "population",
)

# Contested topics need explicit counter-evidence and red teaming.
CONTROVERSY_MARKERS = (
    "controversial", "debate", "criticism", "criticized", "myth",
    "safe", "danger", "harmful", "banned", "ethical", "bias",
    "misinformation", "conspiracy", "lawsuit", "scandal", "risk of",
)

# ---------------------------------------------------------------------------
# Research dimensions — the vision's "required expertise" list, made concrete.
# Each dimension maps to a specialist role the planner can request and the
# summarizer already has a prompt overlay for.
# ---------------------------------------------------------------------------

DIMENSION_SIGNALS: Dict[str, Tuple[str, ...]] = {
    "financial": (
        "cost", "price", "capex", "opex", "investment", "invest", "funding",
        "financing", "roi", "revenue", "profit", "budget", "subsidy",
        "valuation", "loan", "interest rate", "tariff", "tax",
    ),
    "market": (
        "market", "demand", "supply", "customer", "competitor", "competition",
        "adoption", "share", "growth", "forecast", "industry", "vendor",
        "pricing", "consumer",
    ),
    "technical": (
        "technology", "technical", "architecture", "engineering", "algorithm",
        "model", "performance", "benchmark", "implementation", "hardware",
        "software", "efficiency", "throughput", "latency", "reactor",
        "battery", "grid", "infrastructure",
    ),
    "legal": (
        "legal", "law", "regulation", "regulatory", "compliance", "licence",
        "license", "liability", "contract", "patent", "gdpr", "antitrust",
        "statute", "court", "ruling", "jurisdiction",
    ),
    "policy": (
        "policy", "government", "governance", "subsidy", "mandate",
        "legislation", "public", "national", "state", "ministry",
        "regulator", "treaty", "target", "net zero",
    ),
    "scientific": (
        "study", "research", "evidence", "clinical", "trial", "experiment",
        "physics", "chemistry", "biology", "climate", "emissions", "health",
        "safety", "mechanism", "efficacy",
    ),
    "academic": (
        "theory", "philosophy", "literature", "scholar", "epistemology",
        "framework", "concept", "definition", "school of thought",
    ),
    "geopolitical": (
        "geopolitical", "geopolitics", "china", "russia", "usa", "eu",
        "sanction", "trade war", "alliance", "security", "military",
        "sovereignty", "import", "export", "supply chain",
    ),
    "environmental": (
        "environment", "environmental", "climate", "carbon", "emission",
        "pollution", "waste", "sustainability", "biodiversity", "water",
        "land use", "renewable",
    ),
    "social": (
        "workforce", "jobs", "employment", "labour", "labor", "education",
        "inequality", "public opinion", "community", "demographic",
        "culture", "adoption barrier",
    ),
}

# Which specialist prompt overlay each dimension maps to. The summarizer's
# SPECIALIST_PROMPT_ADDITIONS keys are the target vocabulary.
DIMENSION_TO_SPECIALIST: Dict[str, str] = {
    "financial": "financial",
    "market": "market",
    "technical": "technical",
    "legal": "legal",
    "policy": "policy",
    "scientific": "scientific",
    "academic": "academic",
    "geopolitical": "policy",
    "environmental": "scientific",
    "social": "policy",
}

_ENTITY_SPLIT_RE = re.compile(r",|\band\b|\bor\b|\bvs\.?\b|\bversus\b|/")

QUERY_TYPES = ("factual", "comparative", "analytical", "exploratory")


# ---------------------------------------------------------------------------
# Complexity
# ---------------------------------------------------------------------------

@dataclass
class ComplexityScore:
    score: int
    level: str                       # low | medium | high | very_high
    query_type: str
    markers_found: List[str] = field(default_factory=list)
    # --- additive fields ---
    dimensions: List[str] = field(default_factory=list)
    entity_count: int = 1
    needs_recency: bool = False
    needs_quantitative: bool = False
    is_decision: bool = False
    is_contested: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "level": self.level,
            "query_type": self.query_type,
            "markers_found": list(self.markers_found),
            "dimensions": list(self.dimensions),
            "entity_count": self.entity_count,
            "needs_recency": self.needs_recency,
            "needs_quantitative": self.needs_quantitative,
            "is_decision": self.is_decision,
            "is_contested": self.is_contested,
        }


def classify_query_type(query: str) -> str:
    q = f" {(query or '').lower().strip()} "
    if any(m in q for m in COMPARATIVE_MARKERS):
        return "comparative"
    if any(m in q for m in ANALYTICAL_MARKERS):
        return "analytical"
    if any(m in q for m in EXPLORATORY_MARKERS):
        return "exploratory"
    return "factual"


def detect_dimensions(query: str, limit: int = 6) -> List[str]:
    """Which expertise areas the question actually spans.

    Ranked by signal count so a query that mentions cost once and technology
    six times leads with `technical`. Capped because the plan can only carry so
    many angles, and an over-broad plan researches everything shallowly — the
    exact failure the vision's specialist workforce is meant to avoid.
    """
    q = (query or "").lower()
    hits: List[Tuple[int, str]] = []
    for dimension, signals in DIMENSION_SIGNALS.items():
        count = sum(1 for s in signals if s in q)
        if count:
            hits.append((count, dimension))
    hits.sort(key=lambda pair: (-pair[0], pair[1]))
    return [dimension for _, dimension in hits[: max(1, limit)]]


def _count_entities(query: str) -> int:
    """Distinct research targets named in the question.

    The previous implementation counted every non-empty fragment of a split on
    and/or/comma, including single stopwords, so "the cost and the price of X"
    scored 3 entities. Fragments must now carry a capitalized token or at least
    two content words to count.
    """
    raw = (query or "").strip()
    if not raw:
        return 0
    parts = [p.strip() for p in _ENTITY_SPLIT_RE.split(raw) if p.strip()]
    entities = 0
    for part in parts:
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", part)]
        if not words:
            continue
        has_proper = any(w[0].isupper() for w in words[1:]) or len(words) >= 2
        if has_proper:
            entities += 1
    return max(1, entities)


def score_complexity(query: str) -> ComplexityScore:
    q = (query or "").lower().strip()
    padded = f" {q} "
    markers: List[str] = []
    score = 0

    words = len(q.split())
    if words > 25:
        score += 3
        markers.append("very_long_query")
    elif words > 12:
        score += 2
        markers.append("long_query")
    elif words > 6:
        score += 1
        markers.append("medium_query")

    if any(m in padded for m in COMPARATIVE_MARKERS):
        score += 3
        markers.append("comparative")
    if any(m in padded for m in ANALYTICAL_MARKERS):
        score += 2
        markers.append("analytical")
    if any(m in padded for m in EXPLORATORY_MARKERS):
        score += 2
        markers.append("exploratory")

    entity_count = _count_entities(query)
    if entity_count >= 3:
        score += 1
        markers.append("multi_entity")

    if re.search(r"\b(20\d{2}|over|last|past|since|within|decade|year)\b", q):
        score += 1
        markers.append("time_bound")

    # Additive signals. Each is worth points because each demands real extra
    # research work, not because it sounds impressive.
    dimensions = detect_dimensions(query)
    if len(dimensions) >= 4:
        score += 3
        markers.append("multi_dimensional")
    elif len(dimensions) >= 2:
        score += 1
        markers.append("cross_dimensional")

    is_decision = any(m in padded for m in DECISION_MARKERS)
    if is_decision:
        score += 2
        markers.append("decision_framing")

    is_contested = any(m in padded for m in CONTROVERSY_MARKERS)
    if is_contested:
        score += 1
        markers.append("contested")

    needs_recency = any(m in padded for m in RECENCY_MARKERS)
    if needs_recency:
        markers.append("recency_sensitive")

    needs_quantitative = any(m in padded for m in QUANTITATIVE_MARKERS)
    if needs_quantitative:
        markers.append("quantitative")

    if score >= 7:
        level = "very_high"
    elif score >= 5:
        level = "high"
    elif score >= 3:
        level = "medium"
    else:
        level = "low"

    return ComplexityScore(
        score=score,
        level=level,
        query_type=classify_query_type(query),
        markers_found=markers,
        dimensions=dimensions,
        entity_count=entity_count,
        needs_recency=needs_recency,
        needs_quantitative=needs_quantitative,
        is_decision=is_decision,
        is_contested=is_contested,
    )


# ---------------------------------------------------------------------------
# Modes and targets
# ---------------------------------------------------------------------------

LEVEL_TARGETS: Dict[str, int] = {
    "low": 3,
    "medium": 4,
    "high": 5,
    "very_high": 6,
}

# Confidence a mode is trying to reach before it stops. Quick answers are
# allowed to be less certain — that is the trade the user chose by asking for
# a quick answer — while audit exists precisely to be strict.
MODE_CONFIDENCE_TARGET: Dict[str, float] = {
    "quick": 0.60,
    "standard": 0.75,
    "deep": 0.80,
    "executive": 0.78,
    "audit": 0.85,
    "redteam": 0.72,
}

MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "quick":     {"max_agents": 2, "max_iterations": 1, "deep_research": False},
    "standard":  {"max_agents": 4, "max_iterations": 3, "deep_research": False},
    "deep":      {"max_agents": 5, "max_iterations": 5, "deep_research": True},
    "executive": {"max_agents": 5, "max_iterations": 4, "deep_research": True},
    "audit":     {"max_agents": 3, "max_iterations": 4, "deep_research": False},
    "redteam":   {"max_agents": 3, "max_iterations": 2, "deep_research": False},
}
VALID_MODES = tuple(MODE_PRESETS.keys())


def recommend_mode(complexity: ComplexityScore) -> str:
    """Mode MARS would pick if the user did not.

    A recommendation, never an override: modes cost money and latency, and the
    user's explicit choice always wins in `orchestrate`. Exposed so the UI can
    show "Deep Research recommended" with a reason instead of making the user
    guess.
    """
    if complexity.is_decision and complexity.level in ("high", "very_high"):
        return "executive"
    if complexity.level == "very_high":
        return "deep"
    if complexity.level == "low" and complexity.query_type == "factual":
        return "quick"
    if complexity.is_contested:
        return "redteam" if complexity.level == "medium" else "deep"
    return "standard"


@dataclass
class PlanTargets:
    """Concrete numbers the planner and the loop must honour."""

    sub_questions: int
    min_sources_per_axis: int
    max_iterations: int
    min_iterations: int
    confidence_target: float
    required_axes: List[str] = field(default_factory=list)
    specialists: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub_questions": self.sub_questions,
            "min_sources_per_axis": self.min_sources_per_axis,
            "max_iterations": self.max_iterations,
            "min_iterations": self.min_iterations,
            "confidence_target": self.confidence_target,
            "required_axes": list(self.required_axes),
            "specialists": list(self.specialists),
        }


def plan_targets(
    complexity: ComplexityScore, mode: str, target_agents: int
) -> PlanTargets:
    """Translate complexity + mode into the plan's hard requirements.

    `required_axes` is the important output. The planner prompt asks for a
    statistical angle and a criticism angle, but nothing ever verified that the
    model complied, so plans routinely shipped four background questions and no
    evidence-hunting one. These axes are enforced deterministically after
    planning (see planner.enforce_axis_coverage).
    """
    mode = mode if mode in MODE_PRESETS else "standard"
    preset = MODE_PRESETS[mode]

    required: List[str] = []
    if mode != "quick":
        # Every non-trivial plan needs something to measure and something to
        # argue against; these two axes are what separate research from recall.
        required.append("evidence")
        required.append("criticism")
    if complexity.query_type == "comparative":
        required.append("comparison")
    if complexity.needs_recency:
        required.append("outlook")
    if complexity.query_type == "factual" or complexity.level == "low":
        required.insert(0, "definition")

    specialists: List[str] = []
    for dimension in complexity.dimensions:
        role = DIMENSION_TO_SPECIALIST.get(dimension, "general")
        if role not in specialists:
            specialists.append(role)

    min_sources = 2
    if mode in ("deep", "executive", "audit"):
        min_sources = 3
    if complexity.needs_quantitative:
        min_sources = max(min_sources, 3)

    max_iterations = int(preset["max_iterations"])
    min_iterations = 1
    if mode in ("deep", "audit"):
        min_iterations = 2

    return PlanTargets(
        sub_questions=max(1, int(target_agents)),
        min_sources_per_axis=min_sources,
        max_iterations=max_iterations,
        min_iterations=min(min_iterations, max_iterations),
        confidence_target=MODE_CONFIDENCE_TARGET.get(mode, 0.75),
        required_axes=required[:5],
        specialists=specialists[:5],
    )


# ---------------------------------------------------------------------------
# Orchestration plan
# ---------------------------------------------------------------------------

@dataclass
class OrchestrationPlan:
    complexity: ComplexityScore
    target_agents: int
    clamped: bool
    deep_research: bool
    max_parallel_agents: int
    notes: List[str] = field(default_factory=list)
    # --- additive fields ---
    mode: str = "standard"
    recommended_mode: str = "standard"
    targets: Optional[PlanTargets] = None
    budget_multiplier: float = 1.0

    @property
    def dimensions(self) -> List[str]:
        return list(self.complexity.dimensions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "complexity": self.complexity.to_dict(),
            "target_agents": self.target_agents,
            "clamped": self.clamped,
            "deep_research": self.deep_research,
            "max_parallel_agents": self.max_parallel_agents,
            "notes": list(self.notes),
            "mode": self.mode,
            "recommended_mode": self.recommended_mode,
            "targets": self.targets.to_dict() if self.targets else None,
            "budget_multiplier": self.budget_multiplier,
        }

    def rationale(self) -> str:
        """One paragraph explaining the strategy, for the trace and the UI."""
        c = self.complexity
        bits = [
            f"{c.level} complexity (score {c.score}, {c.query_type})",
            f"{len(c.dimensions)} dimension(s): {', '.join(c.dimensions) or 'general'}",
            f"{self.target_agents} research angle(s)",
            f"mode {self.mode}",
        ]
        if self.targets:
            bits.append(f"up to {self.targets.max_iterations} pass(es)")
            bits.append(f"confidence target {self.targets.confidence_target:.2f}")
        if self.clamped:
            bits.append("clamped by hardware cap")
        return "; ".join(bits) + "."


def orchestrate(
    query: str,
    max_parallel_agents: int,
    deep_research: bool = False,
    mode: str = "",
) -> OrchestrationPlan:
    """Map a query to a research strategy under the hardware cap.

    Signature is backwards compatible; `mode` is new and optional. When it is
    empty the orchestrator recommends one but still uses the default target
    mapping, so existing callers see identical `target_agents` behaviour.

    very_high complexity with an explicit deep_research=true flag (or a deep /
    executive mode, which IS the opt-in) may exceed the default cap; otherwise
    the clamp holds.
    """
    complexity = score_complexity(query)
    recommended = recommend_mode(complexity)

    requested_mode = str(mode or "").strip().lower()
    effective_mode = requested_mode if requested_mode in MODE_PRESETS else "standard"
    notes: List[str] = []
    if requested_mode and requested_mode not in MODE_PRESETS:
        notes.append(f"unknown mode {requested_mode!r}; falling back to standard")

    # A deep/executive mode selection is itself the deep-research opt-in.
    if effective_mode in ("deep", "executive"):
        deep_research = True

    raw_target = LEVEL_TARGETS[complexity.level]
    if requested_mode in MODE_PRESETS:
        # An explicit mode bounds the plan: quick must stay quick even on a
        # very_high query, which is the user trading depth for latency.
        mode_cap = int(MODE_PRESETS[effective_mode]["max_agents"])
        if mode_cap < raw_target:
            notes.append(
                f"mode {effective_mode} caps the plan at {mode_cap} angle(s) "
                f"(complexity suggested {raw_target})"
            )
        raw_target = min(raw_target, mode_cap)

    # A question spanning many dimensions needs at least one angle per
    # dimension, up to the caps below — otherwise a 6-dimension decision
    # question gets researched along 4 axes and two dimensions never appear.
    dimension_floor = min(len(complexity.dimensions), 6)
    if dimension_floor > raw_target and effective_mode in ("deep", "executive"):
        notes.append(
            f"raised target to {dimension_floor} to cover every detected dimension"
        )
        raw_target = dimension_floor

    effective_cap = max_parallel_agents
    if complexity.level == "very_high" and deep_research:
        effective_cap = max(max_parallel_agents, raw_target)
    elif effective_mode in ("deep", "executive"):
        # The mode IS the depth opt-in: covering every detected dimension may
        # exceed the default hardware cap, or the dimension-coverage raise
        # above would be clamped straight back down — recreating exactly the
        # "six dimensions researched along four axes" failure it exists to
        # prevent. The mode's own max_agents still bounds the plan.
        effective_cap = max(max_parallel_agents, raw_target)

    target_agents = max(1, min(raw_target, effective_cap))
    clamped = target_agents < raw_target
    if clamped:
        if complexity.level == "very_high" and not deep_research:
            notes.append(
                "very_high complexity clamped to MAX_PARALLEL_AGENTS; "
                "pass deep_research=true to lift the cap"
            )
        else:
            notes.append(
                f"{complexity.level} complexity target {raw_target} clamped to "
                f"MAX_PARALLEL_AGENTS={max_parallel_agents}"
            )

    from app.agents.budget import MODE_BUDGET_MULTIPLIER

    return OrchestrationPlan(
        complexity=complexity,
        target_agents=target_agents,
        clamped=clamped,
        deep_research=deep_research,
        max_parallel_agents=max_parallel_agents,
        notes=notes,
        mode=effective_mode,
        recommended_mode=recommended,
        targets=plan_targets(complexity, effective_mode, target_agents),
        budget_multiplier=MODE_BUDGET_MULTIPLIER.get(effective_mode, 1.0),
    )
