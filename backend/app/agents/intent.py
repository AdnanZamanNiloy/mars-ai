"""Intent Classification Agent — understand the question before searching.

Why this agent exists
---------------------
The "What is transformer?" failure: the pipeline treated the raw query as a
search string, the grounding search and the planner happily mixed electrical-
transformer statistics with ML-architecture papers, and the report answered a
question the user did not ask. Every downstream stage (verification, critic,
confidence) worked perfectly on the WRONG target. Intent must be resolved
before any research is shaped — this module is the pipeline's
"understand question" step of the vision loop.

What it produces
----------------
* query_type / domain / explanation_level — the plan's target register
* ambiguity + ranked senses with probabilities — "transformer" as an AI
  architecture (0.75) vs an electrical device (0.20)
* recommended_action — research the dominant sense only, or research both
  when the probabilities genuinely split

Design constraints
------------------
* MARS never silently picks the wrong domain, and never BLOCKS on a clarifying
  question (the stream contract is one-shot). When ambiguous, the answer is
  structured to disambiguate first ("X can mean two things...") and the
  research targets the likely sense(s) — the "provide both meanings briefly"
  branch of the vision's clarification policy.
* LLM path is one small, cache-friendly call. The deterministic fallback keeps
  degraded runs correct-but-cruder (planner-classified type/domain, curated
  homonym hints) instead of wrong — same contract as every other agent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.core.degradation import record_fallback
from app.core.llm import LLMClient
from app.core.logging import get_logger
from app.core.schemas import IntentOutputModel

from app.agents.orchestrator import classify_query_type

logger = get_logger(__name__)

INTENT_SYSTEM_PROMPT = """
You are the Intent Classification Agent in a multi-agent research pipeline.
You run BEFORE any research. Your job is to understand what the user means —
not what the words literally say.

━━━ YOUR RULES ━━━

RULE 1 — RESOLVE AMBIGUITY FIRST
  Many research terms name several distinct things ("transformer": a neural
  network architecture OR an electrical AC-voltage device; "python": a
  programming language OR a snake). If the query could plausibly target more
  than one meaning:
    - set ambiguity = true,
    - list every plausible sense with a probability (all probabilities
      together should be roughly comparable to 1.0),
    - order senses most-likely first.
  If the query names one clear thing, ambiguity = false and senses may be
  empty (a single sense entry is also acceptable).

RULE 2 — CHOOSE THE DOMAIN THE RESEARCH SHOULD TARGET
  domain must be one of: machine_learning | software | engineering |
  philosophy | economics | science | legal | policy | academic | general.
  When ambiguous, use the domain of the MOST LIKELY sense.

RULE 3 — GAUGE THE EXPLANATION LEVEL
  basic     — the user wants the concept explained simply ("what is X?",
              "explain X simply")
  practical — the user wants understanding with real examples and details
  expert    — the user signals technical depth ("architecture",
              "implementation", "internals", research-level phrasing)

RULE 4 — CLASSIFY THE QUESTION TYPE
  query_type: factual | comparative | analytical | exploratory
  (same enum as the planner uses).

RULE 5 — GROUND IN CONTEXT WHEN PROVIDED
  Web context snippets for the raw query show how the world uses these words.
  Use them to estimate sense probabilities — but do not let a snippet's
  favorite meaning override what the phrasing implies ("What is transformer?"
  bare, with mixed electrical + AI results, is still most likely the AI sense
  in a research-assistant context).

━━━ OUTPUT FORMAT ━━━

Return ONLY valid JSON. No markdown fences. No text outside JSON.

{
  "query_type": "<factual|comparative|analytical|exploratory>",
  "domain": "<machine_learning|software|engineering|philosophy|economics|science|legal|policy|academic|general>",
  "explanation_level": "<basic|practical|expert>",
  "ambiguity": <true|false>,
  "senses": [
    {
      "label": "<short human name of the sense, e.g. 'Transformer neural network architecture'>",
      "domain": "<same enum as domain>",
      "probability": <0.0-1.0>,
      "note": "<one clause: how you'd briefly explain this sense>"
    }
  ],
  "reasoning": "<one sentence: what the user most likely wants>"
}
""".strip()


# Second-sense probability at which a query counts as ambiguous even if the
# model hedged. Below this, one dominant reading is safe to research.
AMBIGUITY_FLOOR = 0.15
# Top-two gap under which both senses are worth researching (the answer will
# structure both); above it, research the dominant sense and let the report's
# disambiguation paragraph cover the rest.
RESEARCH_BOTH_GAP = 0.30

# Non-planner domain names the model may emit, mapped onto the planner's
# domain vocabulary (which routes specialists). Anything unmapped falls
# through to normalize_domain's "general".
_DOMAIN_ALIASES: Dict[str, str] = {
    "electrical_engineering": "engineering",
    "electrical": "engineering",
    "power_engineering": "engineering",
    "medicine": "science",
    "medical": "science",
    "healthcare": "science",
    "finance": "economics",
    "financial": "economics",
    "business": "economics",
    "ml": "machine_learning",
    "ai": "machine_learning",
    "cs": "software",
    "computer_science": "software",
}


# ---------------------------------------------------------------------------
# Deterministic sense hints (the no-LLM path's ambiguity detector)
# ---------------------------------------------------------------------------

# Curated homonyms a research assistant actually gets asked about. Deliberately
# tiny and honest: this is the degraded-mode detector, not the primary one.
HOMONYM_SENSES: Dict[str, List[Dict[str, Any]]] = {
    "transformer": [
        {"label": "Transformer neural network architecture", "domain": "machine_learning",
         "probability": 0.75, "note": "attention-based deep learning architecture behind GPT/BERT"},
        {"label": "Electrical transformer (AC voltage device)", "domain": "engineering",
         "probability": 0.20, "note": "device that steps AC voltage up or down"},
    ],
    "python": [
        {"label": "Python programming language", "domain": "software",
         "probability": 0.80, "note": "general-purpose programming language"},
        {"label": "Python (snake)", "domain": "science",
         "probability": 0.15, "note": "large non-venomous constrictor snake"},
    ],
    "apple": [
        {"label": "Apple Inc. (company)", "domain": "general",
         "probability": 0.70, "note": "consumer technology company"},
        {"label": "Apple (fruit)", "domain": "science",
         "probability": 0.25, "note": "widely cultivated tree fruit"},
    ],
    "amazon": [
        {"label": "Amazon (company)", "domain": "economics",
         "probability": 0.75, "note": "e-commerce and cloud computing company"},
        {"label": "Amazon rainforest / river", "domain": "science",
         "probability": 0.20, "note": "South American rainforest and river basin"},
    ],
    "jaguar": [
        {"label": "Jaguar (car manufacturer)", "domain": "engineering",
         "probability": 0.55, "note": "British luxury vehicle brand"},
        {"label": "Jaguar (animal)", "domain": "science",
         "probability": 0.40, "note": "large spotted cat species"},
    ],
    "eclipse": [
        {"label": "Eclipse (software IDE)", "domain": "software",
         "probability": 0.45, "note": "open-source development environment"},
        {"label": "Eclipse (astronomy)", "domain": "science",
         "probability": 0.50, "note": "one body obscuring another from view"},
    ],
    "windows": [
        {"label": "Microsoft Windows (operating system)", "domain": "software",
         "probability": 0.75, "note": "desktop operating system family"},
        {"label": "Windows (architecture)", "domain": "engineering",
         "probability": 0.20, "note": "glazed openings in building design"},
    ],
    "shell": [
        {"label": "Unix shell (command line)", "domain": "software",
         "probability": 0.60, "note": "command-line interpreter (bash, zsh)"},
        {"label": "Shell (energy company)", "domain": "economics",
         "probability": 0.30, "note": "multinational oil and gas company"},
    ],
}

_BASIC_MARKERS = ("like i'm five", "like im 5", "simply", "simple terms", "in short",
                  "basically", "eli5", "briefly")
_EXPERT_MARKERS = ("architecture", "implementation", "internals", "technical",
                   "in depth", "in-depth", "research", "paper", "benchmark",
                   "compare", "trade-off", "tradeoff")


def _normalize_llm_domain(domain: str) -> str:
    """LLM domain vocabulary -> planner domain vocabulary."""
    key = re.sub(r"\s+", "_", str(domain or "").strip().lower())
    key = _DOMAIN_ALIASES.get(key, key)
    from app.agents.planner import normalize_domain

    return normalize_domain(key)


def _heuristic_level(query: str) -> str:
    q = f" {(query or '').lower()} "
    if any(m in q for m in _BASIC_MARKERS):
        return "basic"
    if any(m in q for m in _EXPERT_MARKERS):
        return "expert"
    return "practical"


@dataclass
class SenseCandidate:
    label: str
    domain: str                 # planner-vocabulary domain
    probability: float
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "domain": self.domain,
            "probability": round(self.probability, 3),
            "note": self.note,
        }


@dataclass
class IntentReport:
    """What the pipeline knows about the question BEFORE researching it."""

    query: str
    query_type: str                    # factual | comparative | analytical | exploratory
    domain: str                        # planner-vocabulary research domain
    explanation_level: str             # basic | practical | expert
    ambiguity: bool
    senses: List[SenseCandidate] = field(default_factory=list)
    reasoning: str = ""
    origin: str = "heuristic"          # llm | heuristic
    # Set when the user's phrasing itself demands both meanings ("What is
    # transformer?") regardless of the probability gap.
    forced_both: bool = False

    @property
    def dominant_sense(self) -> Optional[SenseCandidate]:
        return self.senses[0] if self.senses else None

    @property
    def secondary_sense(self) -> Optional[SenseCandidate]:
        return self.senses[1] if len(self.senses) > 1 else None

    @property
    def recommended_action(self) -> str:
        """research_both (answer structures both senses) vs research_dominant
        (answer disambiguates in one paragraph, then goes deep on the likely
        meaning). MARS never blocks on a clarifying question; the disambiguation
        lives in the answer itself."""
        if self.forced_both and len(self.senses) >= 2:
            return "research_both"
        if not self.ambiguity or len(self.senses) < 2:
            return "research_dominant"
        gap = self.senses[0].probability - self.senses[1].probability
        return "research_both" if gap < RESEARCH_BOTH_GAP else "research_dominant"

    @property
    def research_senses(self) -> List[SenseCandidate]:
        if self.recommended_action == "research_both":
            return self.senses[:2]
        return self.senses[:1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "domain": self.domain,
            "explanation_level": self.explanation_level,
            "ambiguity": self.ambiguity,
            "senses": [s.to_dict() for s in self.senses],
            "recommended_action": self.recommended_action,
            "reasoning": self.reasoning,
            "origin": self.origin,
        }


# Words that appear in a sense's label/note and RESOLVE the homonym when the
# user actually used them ("python snake feeding habits" is not ambiguous).
_SENSE_DISTINCTIVE_STOP = {
    "python", "transformer", "apple", "amazon", "jaguar", "eclipse", "windows",
    "shell", "inc", "company", "device",
}


def _sense_distinctive_words(sense: Dict[str, Any]) -> List[str]:
    text = " ".join([str(sense.get("label", "")), str(sense.get("note", ""))]).lower()
    words = [
        w for w in re.findall(r"[a-z]{4,}", text)
        if w not in _SENSE_DISTINCTIVE_STOP
    ]
    # de-duplicate, keep order
    return list(dict.fromkeys(words))


def heuristic_intent(query: str) -> IntentReport:
    """Deterministic intent for when the LLM call fails or is disabled.

    Correct-but-cruder: query type and domain come from the same lexical
    classifier the orchestrator uses, ambiguity from a curated homonym table.
    It cannot read the user's mind — but it never invents a wrong confident
    reading, and unambiguous queries pass through untouched.
    """
    lowered = (query or "").lower()
    senses: List[SenseCandidate] = []
    for token, raw_senses in HOMONYM_SENSES.items():
        if re.search(rf"\b{re.escape(token)}\b", lowered):
            senses = [
                SenseCandidate(
                    label=s["label"],
                    domain=_normalize_llm_domain(s["domain"]),
                    probability=float(s["probability"]),
                    note=s["note"],
                )
                for s in sorted(raw_senses, key=lambda x: -x["probability"])
            ]
            break

    ambiguity = bool(senses)
    domain = "general"
    if senses:
        # The user may have resolved the homonym themselves ("python snake
        # feeding habits"): when exactly one sense's distinctive words appear
        # in the query, that sense IS the meaning — no ambiguity.
        matched = [
            s for s in senses
            if any(w in lowered for w in _sense_distinctive_words(s.to_dict()))
        ]
        if len(matched) == 1:
            senses = matched
            ambiguity = False
        if senses[0].domain != "general":
            domain = senses[0].domain
    if not ambiguity and domain == "general":
        from app.agents.orchestrator import detect_dimensions

        dimensions = detect_dimensions(query, limit=1)
        domain = _normalize_llm_domain(dimensions[0]) if dimensions else "general"

    return IntentReport(
        query=str(query or ""),
        query_type=classify_query_type(query),
        domain=domain,
        explanation_level=_heuristic_level(query),
        ambiguity=ambiguity,
        senses=senses,
        reasoning="Classified lexically (deterministic fallback; LLM intent unavailable).",
        origin="heuristic",
        forced_both=bool(ambiguity and len(senses) >= 2
                         and _DEFINITIONAL_QUERY_RE.match((query or "").strip())),
    )


_DEFINITIONAL_QUERY_RE = re.compile(
    r"^\s*(what\s+is|what\s+are|what's|define|explain)\b", re.IGNORECASE
)


def _finalize(
    query: str,
    payload: Dict[str, Any],
    origin: str,
    fallback: IntentReport,
) -> IntentReport:
    """Shared post-processing: normalize domains, clamp + rank probabilities,
    apply the deterministic ambiguity floor so a hedging model can never
    silently drop a real second sense."""
    lowered = (query or "").lower()
    senses = [
        SenseCandidate(
            label=str(s.get("label", "")).strip() or "unnamed sense",
            domain=_normalize_llm_domain(str(s.get("domain", ""))),
            probability=max(0.0, min(1.0, float(s.get("probability", 0.0) or 0.0))),
            note=str(s.get("note", "") or "").strip(),
        )
        for s in (payload.get("senses") or [])
        if isinstance(s, dict) and str(s.get("label", "")).strip()
    ]
    senses.sort(key=lambda s: -s.probability)

    llm_ambiguity = bool(payload.get("ambiguity", False))
    ambiguity = llm_ambiguity or (
        len(senses) >= 2 and senses[1].probability >= AMBIGUITY_FLOOR
    )
    # A lone sense is not an ambiguity no matter what the flag said.
    if len(senses) < 2:
        ambiguity = False
        senses = senses[:1]

    # The user may have resolved the homonym themselves ("what is python
    # snake?"): when exactly one sense's distinctive words appear in the
    # query, that sense IS the meaning — never research the other one.
    forced_both = False
    if ambiguity and len(senses) >= 2:
        matched = [
            s for s in senses
            if any(w in lowered for w in _sense_distinctive_words(s.to_dict()))
        ]
        if len(matched) == 1:
            senses = matched
            ambiguity = False
        elif _DEFINITIONAL_QUERY_RE.match((query or "").strip()):
            # Bare "What is X?" on an ambiguous term: the user's phrasing
            # carries no sense signal, so the answer must present BOTH
            # meanings ("X can mean two things...") — never pick one.
            forced_both = True

    return IntentReport(
        query=str(query or ""),
        query_type=str(payload.get("query_type", fallback.query_type) or fallback.query_type),
        domain=_normalize_llm_domain(str(payload.get("domain", "")) or fallback.domain),
        explanation_level=str(payload.get("explanation_level", fallback.explanation_level) or fallback.explanation_level),
        ambiguity=ambiguity,
        senses=senses,
        reasoning=str(payload.get("reasoning", "") or "").strip() or fallback.reasoning,
        origin=origin,
        forced_both=forced_both,
    )


async def classify_intent(
    llm: LLMClient,
    query: str,
    context_snippets: Optional[Sequence[str]] = None,
) -> IntentReport:
    """Classify what the user means before any research is shaped.

    Follows the standard agent shape: LLM path with schema validation, and a
    deterministic fallback on failure or empty output (AGENTS.md 4.7).
    """
    fallback = heuristic_intent(query)

    snippets = [str(s).strip() for s in (context_snippets or []) if str(s).strip()]
    context_block = ""
    if snippets:
        context_block = (
            "\nWeb context — top results for the raw query, for sense "
            "probability only:\n" + "\n".join(f"- {s}" for s in snippets[:6]) + "\n"
        )

    user_prompt = f"Query: {query}{context_block}\nClassify the intent. Return JSON only."

    try:
        payload = await llm.generate_json(
            INTENT_SYSTEM_PROMPT,
            user_prompt,
            response_model=IntentOutputModel,
        )
    except Exception as exc:
        logger.warning("[Intent] LLM call failed, using heuristic fallback", exc_info=exc)
        record_fallback("intent")
        return fallback

    if not isinstance(payload, dict) or not payload:
        logger.warning("[Intent] empty/invalid LLM output, using fallback")
        record_fallback("intent")
        return fallback

    report = _finalize(query, payload, "llm", fallback)
    logger.info(
        "[Intent] %s | type=%s domain=%s level=%s ambiguous=%s action=%s",
        "ambiguous" if report.ambiguity else "clear",
        report.query_type, report.domain, report.explanation_level,
        report.ambiguity, report.recommended_action,
    )
    return report
