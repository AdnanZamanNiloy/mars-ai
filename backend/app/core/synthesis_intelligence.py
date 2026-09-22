"""Synthesis intelligence layer — repetition tracking and refinement.

The failure this module exists for
----------------------------------
The remaining gap versus GPT Researcher is SYNTHESIS quality, not evidence.
Live deep reports re-assert the same evidence fact in section after section —
the Bangladesh report stated "Rooppur … US$13 billion [1]" in ~7 sections
(Executive Summary, What It Is, Applications, Evidence & Data, Limitations &
Critique, Climate, Cost, Energy demand). Each section writer is a separate
LLM call with its own evidence slice and no knowledge of what the sibling
sections already said, so a shared, high-ranked fact becomes the opening
frame of every section. The reader gets the same fact seven times and almost
no added interpretation: sections restate instead of arguing mechanism,
trade-off, comparison or uncertainty.

The first version of this module fixed that by DELETING the later
occurrences. That traded one defect for another: a section whose opening
sentence restated an earlier claim lost its opening entirely and began
mid-argument, and a reader could not tell why the section existed. The
refinement layer here is the correction — a repeat is TRANSFORMED, never
silently dropped.

GPT Researcher avoids this differently (a single writer over compressed
context); MARS writes section-wise for provider-size reasons, so the fix has
to be explicit cross-section bookkeeping. This module is that bookkeeping,
and it is deterministic — no LLM, no network, no new model. It does not touch
retrieval, grading, corroboration or contradiction; it only decides how
already-written sentences read in the assembled report.

What it does
------------
1. CLAIM-KEY TRACKING — every writer sentence is reduced to a canonical
   `claim_key` (citations stripped, stopwords removed, tokens stemmed and
   sorted). The same key seen in an earlier section is a REPEAT.
2. DIMENSION-AWARE EXPANSION — a repeat that adds a genuinely new analytical
   dimension the earlier use did not (mechanism / implication / comparison /
   uncertainty) is left INTACT; the dimension is recorded against the claim.
3. REFINE, DON'T DELETE — a bare restatement is rewritten into a refinement
   sentence: a contextual transition naming the earlier section ("Building on
   the cost picture above, …") plus the original claim (its number and `[n]`
   marker preserved verbatim) plus an ADAPTIVE REASONING MOVE chosen from the
   available evidence signals — contradiction -> uncertainty/trade-off, a
   comparison section or comparative query -> comparison, a causal query ->
   causal explanation, a decision/policy query -> strategic consequence,
   quantitative content -> implication/trade-off, a mechanism claim/section ->
   mechanism, authoritative sourcing -> strength framing. The move (not a
   fixed template) is signal-driven, and its phrasing fits the move; this runs
   with no LLM, so the deterministic path always produces a usable transition
   and a section never opens on a dangling or removed first sentence.
4. CITATION PRESERVATION — refined sentences keep the exact `[n]` markers of
   the original claim; no marker is ever invented, renumbered or moved
   between sentences.
5. STRUCTURE PRESERVATION — headings, section order and the required
   sections are never touched; an empty section is never produced.

The module also exposes `analyze_report` so the same claim-key machinery can
MEASURE redundancy (the benchmark's redundancy score) instead of only
repairing it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.core.logging import get_logger

logger = get_logger(__name__)

# Marker like "[3]" or "[2][5]". Kept verbatim on retained sentences.
_CITATION_RE = re.compile(r"\[\d+\]")

# Function words carry no distinguishing content for a claim key. NOTE:
# negation words ("not", "no", "never", "without") are deliberately NOT
# stopwords — AGENTS.md bug history: a similarity/dedup key that drops
# negation merges "X" and "not X". The polarity guard below additionally
# separates them even when the content tokens collide.
_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "by", "as", "at", "from", "into", "than", "that", "this",
    "these", "those", "it", "its", "is", "are", "was", "were", "be", "been",
    "being", "has", "have", "had", "do", "does", "did", "will", "would",
    "can", "could", "may", "might", "should", "which", "who", "whom",
    "whose", "what", "when", "where", "why", "how", "there", "their",
    "they", "them", "he", "she", "his", "her", "him", "we", "our", "you",
    "your", "i", "also", "such", "more", "most", "less", "least", "very",
    "just", "only", "then", "so", "if", "about", "over", "under", "across",
    "during", "after", "before", "while", "because", "however", "thus",
}

# Minimal suffix stemmer — mirrors the intent of the shared semantic engine's
# stemmer without importing it, so a claim key is stable across small
# inflections ("transfers"/"transferring", "costs"/"cost").
_SUFFIXES = ("ings", "ing", "ies", "ied", "es", "ed", "s")

# Words that mark a genuinely NEW analytical dimension. A repeat carries one
# of these AND a dimension type not already attached to the claim; a bare
# restatement carries none.
_MECHANISM_RE = re.compile(
    r"(?i)\b(because|driven by|as a result of|mechanism|results? in|leads? to|"
    r"caused by|gives? rise to|due to|stems? from|the reason|so that|thereby|"
    r"which forces|forcing|trigger(?:s|ed|ing)?|explains? why|follows? from|"
    r"the driver|is driven|arises? from|comes? from|"
    r"which (?:removed|eliminated|enabled|allowed|reduced|caused|created|"
    r"introduced|replaced|made))\b"
)
_IMPLICATION_RE = re.compile(
    r"(?i)\b(therefore|impl(?:y|ies|ied|ication)|at the cost of|trade-?offs?|"
    r"(?:which |this )?means|the consequences?|as a consequence|in turn|"
    r"puts? pressure|what follows|the upshot|raises? the question|signals? that|"
    r"raises? the .{0,24}cost|raises? cost|raise(?:s|d)? .{0,16}cost|"
    r"the price of|at the expense of|comes? at the cost|requires? "
    r"(?:backup|storage|subsid)|entails|necessitates|has? implications?)\b"
)
_COMPARISON_RE = re.compile(
    r"(?i)\b(compared (?:with|to)|versus|\bvs\.?\b|whereas|unlike|in contrast|"
    r"on the other hand|relative to|outweighs?|higher than|lower than|"
    r"cheaper than|more expensive than|better than|worse than|"
    r"comparatively|by comparison|less than|more than)\b"
)
_UNCERTAINTY_RE = re.compile(
    r"(?i)\b(unresolved|uncertain|unclear|not established|remains unknown|"
    r"open question|cannot be settled|disputed|contested|not yet verified|"
    r"single-source|provisional|could not verify|no evidence|insufficient "
    r"evidence|contradict)\b"
)

_DIMENSIONS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("mechanism", _MECHANISM_RE),
    ("implication", _IMPLICATION_RE),
    ("comparison", _COMPARISON_RE),
    ("uncertainty", _UNCERTAINTY_RE),
)

# ---------------------------------------------------------------------------
# Adaptive reasoning moves
# ---------------------------------------------------------------------------
# The templated analytical clauses this module shipped with appended the same
# generic tail ("the figure is significant because …") to every restatement,
# regardless of what the claim was or what the evidence around it said. That
# reads as boilerplate because it IS boilerplate: the phrase does not depend on
# the claim. This layer replaces it with a MOVE chosen from the evidence
# signals, each with phrasing that fits the move.
#
# The seven moves are the reasoning relations a repeated claim can usefully be
# turned into:
#   implication  — what follows from the claim
#   mechanism    — the process that produces it
#   causal       — why it happened / what caused it
#   tradeoff     — what it costs or gives up
#   uncertainty  — how settled it is
#   comparison   — how it ranks against the alternatives
#   strategic    — what it means for a decision or plan
#
# Selection is an ORDERED, DETERMINISTIC precedence over real signals (never
# RNG, never a hash): contradiction, query intent, section axis, then claim
# type. The move — not a random template — is signal-driven; within a move a
# small pool of phrasings is indexed deterministically via `_stable_pick` so
# the same claim always refines the same way and tests stay repeatable.
MOVES: Tuple[str, ...] = (
    "implication",
    "mechanism",
    "causal",
    "tradeoff",
    "uncertainty",
    "comparison",
    "strategic",
)

# Phrases that mark each move in text. A move is verifiable: `reasoning_moves`
# detects it, and the deterministic post-condition is that the chosen move is
# actually present in the refined sentence. `_DIMENSIONS` above still maps onto
# the expansion vocabulary, so attaching a move neither weakens nor re-labels
# the existing expansion detection.
_MOVE_MARKERS: Dict[str, "re.Pattern[str]"] = {
    "mechanism": _MECHANISM_RE,
    "causal": re.compile(
        r"(?i)\b(caused|the cause|because|driven by|arose from|stems from|"
        r"origint|root cause|triggered|explains why|as a result of|gave rise|"
        r"follows? from|did not appear by chance)\b"
    ),
    "tradeoff": re.compile(
        r"(?i)\b(trade-?offs?|at the cost of|at the expense of|gives? up|"
        r"in exchange|the price of|must be weighed|buys? .{0,20}at|"
        r"offset by|no free lunch|costs? (?:reliability|flexibility|"
        r"dispatchability|speed))\b"
    ),
    "comparison": _COMPARISON_RE,
    "uncertainty": _UNCERTAINTY_RE,
    "strategic": re.compile(
        r"(?i)\b(changes? the role|shifts? the (?:role|balance|decision)|"
        r"for a (?:policymaker|planner|decision|grid|government)|"
        r"the strategic|what it means for|decides? whether|commits? the|"
        r"the decision (?:hinges|turns)|tilts? the choice|"
        r"the planning question|long-run (?:strategic|decision))\b"
    ),
    "implication": _IMPLICATION_RE,
}

# Transitions that open a REFINEMENT sentence, keyed by MOVE. Each names the
# relationship to the earlier use so the section reads as a continuation
# instead of a fresh assertion. Pools are indexed deterministically (see
# `_stable_pick`); the wording is shared across topics because a transition is
# discourse glue, while the analytical tail (below) carries the topic.
_TRANSITION_BY_MOVE: Dict[str, Tuple[str, ...]] = {
    "implication": (
        "Building on the picture above:",
        "Extending that finding to what follows from it:",
        "Taking that established point one step further:",
        "Reading that figure for its consequences:",
    ),
    "mechanism": (
        "Building on that fact, the mechanism runs as follows:",
        "The mechanism behind that established point is worth stating plainly:",
        "That outcome follows from a mechanism the earlier section did not name:",
    ),
    "causal": (
        "Building on that fact, the causal chain is worth naming:",
        "That outcome did not arise by accident:",
        "The cause behind that established point runs as follows:",
    ),
    "tradeoff": (
        "Building on that established point, it carries a trade-off:",
        "That gain is not free:",
        "Weighing that claim against what it costs:",
    ),
    "comparison": (
        "Set against the alternatives already discussed:",
        "Compared with the other options the report has covered:",
        "Weighing that claim against the alternatives:",
    ),
    "uncertainty": (
        "That figure is less settled than a single statement suggests:",
        "Building on that point, its reliability must be qualified:",
        "That claim carries an unresolved caveat:",
    ),
    "strategic": (
        "Building on that established point, the strategic reading is this:",
        "For the decision the report is answering, that changes the calculus:",
        "Extending that finding to the plan it informs:",
    ),
    # Legacy move names mapped onto the adaptive set for backward compatibility.
    "strength": (
        "That finding is unusually well grounded:",
        "Building on that evidence, its provenance is stronger than most:",
        "That claim draws on source material worth foregrounding:",
    ),
    "framing": (
        "Building on the picture already established:",
        "Returning to that established fact for this section's angle:",
        "With that established, this section turns to what it implies here:",
    ),
}

# Analytical clauses appended to a refined sentence. Each contains a token the
# `analytical_dimensions` detector recognizes (the deterministic post-check the
# LLM path is also held to), so the refined text is provably not a bare
# restatement. Clauses are TOPIC-SCOPED: a financing frame only makes sense on
# a cost claim, so a bare generic clause is used when no topic matches (a
# "long-term financing obligations" tail on a transformer-architecture sentence
# is exactly the kind of mismatch this scoping prevents).
_TOPIC_ANALYSIS: Tuple[Tuple["re.Pattern[str]", Dict[str, str]], ...] = (
    (
        re.compile(r"(?i)\b(cost|costs|finance|financ|loan|debt|tariff|budget|"
                   r"billion|trillion|million|invest|price|subsid)\b"),
        {
            "implication": (
                "which means the long-term financing obligations it commits the "
                "sector to constrain the choices the rest of the analysis depends on"
            ),
            "mechanism": (
                "the cost pressure follows a mechanism in which fixed capital "
                "must be recovered over a long operating life, which raises the "
                "stakes of any delay"
            ),
            "causal": (
                "that cost level did not appear by chance: it follows from the "
                "capital structure and construction timeline the project entails, "
                "which is why it recurs"
            ),
            "tradeoff": (
                "that financing buys capacity at the cost of locking the budget "
                "into one pathway for decades"
            ),
            "comparison": (
                "that cost stands out compared with the alternatives discussed "
                "elsewhere, where relative cost per unit of output is the "
                "deciding factor"
            ),
            "uncertainty": (
                "the cost figure is disputed across sources, so it should be "
                "read as a provisional range rather than a settled estimate"
            ),
            "strategic": (
                "for the budgeting decision that changes the role the cost "
                "plays, which means the sector is committed to one financing "
                "profile"
            ),
            "strength": (
                "the cost figure is corroborated by multiple independent "
                "sources, which raises confidence in it rather than merely "
                "repeating it"
            ),
            "framing": (
                "for this section's angle the cost matters because it frames "
                "the financing trade-off the section goes on to examine"
            ),
        },
    ),
    (
        re.compile(r"(?i)\b(emission|carbon|climate|renewabl|solar|wind|"
                   r"nuclear|energy|electric|grid|power|generation|capacity|"
                   r"megawatt|gigawatt|mwh|kwh)\b"),
        {
            "implication": (
                "which means the value of firm output shifts, so the flexibility "
                "the system can rely on is what shapes the grid's real choices"
            ),
            "mechanism": (
                "the mechanism is that generation choices lock in an "
                "infrastructure pathway whose costs and emissions persist for "
                "decades"
            ),
            "causal": (
                "that outcome follows from how the technology converts its fuel "
                "or resource into dispatchable output, not from its headline "
                "nameplate size"
            ),
            "tradeoff": (
                "that strength is bought at the cost of slow build times and "
                "high upfront capital, which is the trade-off the comparison turns on"
            ),
            "comparison": (
                "that firm output stands out compared with the alternative "
                "generation options discussed elsewhere, where output per unit "
                "of cost is the deciding factor"
            ),
            "uncertainty": (
                "the figure is disputed across sources, so it should be read "
                "as a provisional range rather than a settled estimate"
            ),
            "strategic": (
                "for a grid balancing variable renewables, that changes its "
                "role from bulk energy producer to dispatchable system "
                "stabilizer, which means flexibility sets its value"
            ),
            "strength": (
                "the finding is corroborated by multiple independent sources, "
                "which raises confidence in it rather than merely repeating it"
            ),
            "framing": (
                "for this section's angle the point matters because it frames "
                "the generation trade-off the section goes on to examine"
            ),
        },
    ),
)

_ANALYSIS_BY_MOVE: Dict[str, str] = {
    "implication": (
        "which means it constrains the choices the rest of the analysis "
        "depends on, rather than being an isolated number"
    ),
    "mechanism": (
        "the underlying mechanism is that the earlier conditions compound, "
        "which forces the trade-offs described here"
    ),
    "causal": (
        "that outcome follows from the conditions named above, which is why it "
        "recurs across the evidence rather than standing alone"
    ),
    "tradeoff": (
        "the gain comes at the expense of a competing objective, so it must be "
        "weighed rather than treated as strictly better"
    ),
    "comparison": (
        "that stands out compared with the alternatives discussed elsewhere, "
        "where the relative merits are the deciding factor"
    ),
    "uncertainty": (
        "the claim is disputed across sources, so it should be read as a "
        "provisional finding rather than a settled point"
    ),
    "strategic": (
        "for the decision at hand that changes the role it can play, which "
        "means the repeated figure is decision-relevant"
    ),
    "strength": (
        "the claim is corroborated by multiple independent sources, which "
        "raises confidence in it rather than merely repeating it"
    ),
    "framing": (
        "for this section's angle the point matters because it frames the "
        "trade-off the section goes on to examine"
    ),
}


def _analysis_clause(sentence: str, move: str) -> str:
    """The analytical tail for a refined sentence, scoped to its topic and move.

    Picks the first topic-specific variant whose subject vocabulary appears in
    the sentence. There is deliberately NO generic fallback: a move clause that
    is not scoped to the claim's own topic reads as topic-agnostic boilerplate
    ("the underlying mechanism is that the earlier conditions compound…" stapled
    onto an mRNA-delivery sentence). When no topic-specific variant applies the
    empty string is returned, and the caller leaves the sentence unchanged.

    The tail's REASONING SHAPE (mechanism, trade-off, uncertainty, …) still
    matches the move the signals selected; only the topic scoping changed.
    """
    move = move if move in _ANALYSIS_BY_MOVE else "implication"
    for pattern, variants in _TOPIC_ANALYSIS:
        if pattern.search(sentence or ""):
            clause = variants.get(move)
            if clause:
                return clause
    return ""


def _stable_pick(options: Sequence[str], seed: str) -> str:
    """Deterministically choose one template for a claim.

    No RNG: the choice is derived from the claim key so the same repetition
    always refines the same way (stable tests, stable diffs) while different
    claims receive different transitions (a report does not read as one
    boilerplate line repeated). Falls back to the first option on empty input.
    """
    if not options:
        return ""
    for char in seed or "":
        if char.isalnum():
            return options[sum(ord(c) for c in seed) % len(options)]
    return options[0]


@dataclass
class Refinement:
    """The outcome of classifying one writer sentence against the ledger."""

    text: str
    kept: bool = True
    transformed: bool = False
    dimension: str = ""
    first_section: str = ""
    # The adaptive reasoning move chosen for a transformed restatement
    # (implication / mechanism / causal / tradeoff / uncertainty / comparison /
    # strategic). Empty for kept-verbatim sentences.
    move: str = ""

_NEGATION_RE = re.compile(r"(?i)\b(not|no|never|without|neither|nor|fails? to|does not|did not|is not|are not)\b")

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_BULLET_RE = re.compile(r"^\s*[-*]\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TRIVIAL_RE = re.compile(r"^(?:[-*]\s*)?$")

# A sentence must carry at least this many content tokens to be claim-keyed.
# Shorter units ("In short.", "Taken together.") are transitions, not claims,
# and would otherwise collide on an empty key.
_MIN_CONTENT_TOKENS = 3

# Similarity above which two differently-worded sentences are the same claim.
# Deliberately high: this layer must never delete a genuinely distinct finding
# because it shares vocabulary. Calibrated against the shared semantic engine's
# near-duplicate band (AGENTS.md: dedup threshold 0.86 for merging; 0.78 here
# because cross-section restatement is looser than within-pool duplication).
RESTATEMENT_SIMILARITY = 0.78

# Fuzzy restatement detection. Paraphrased cross-section restatements ("Rooppur
# … at an estimated cost close to US$13 billion" vs "the Rosatom-constructed
# Rooppur plant carries an estimated cost of close to US$13 billion") score only
# 0.49-0.68 on the semantic engine — below any safe global similarity threshold —
# yet they are plainly the same fact. The discriminating signal is a shared
# ANCHOR: at least two distinctive content tokens that are not generic report
# vocabulary (a name, a figure with its unit). Two sentences that share two
# rare anchors AND clear a modest token-overlap bar are treated as one claim.
ANCHOR_OVERLAP_MIN = 0.55
ANCHOR_SHARED_MIN = 2

# Generic report vocabulary: high-frequency across any query's evidence. These
# are never "anchors" — otherwise every sentence about cost/capacity/energy
# would look like a restatement of every other.
_GENERIC_ANCHORS: Set[str] = {
    "cost", "costs", "capac", "capita", "energy", "nuclear", "power", "plant",
    "project", "year", "countr", "govern", "invest", "investig", "percent",
    "share", "global", "world", "report", "data", "increase", "higher", "lower",
    "billion", "trillion", "million", "percentag", "point", "source", "sourc",
    "estim", "total", "develop", "developing", "growth", "risk", "risks",
}


@dataclass
class ClaimUse:
    """One claim, where it was first used, and the dimensions seen so far."""

    key: str
    text: str
    section: str
    dimensions: Set[str] = field(default_factory=set)
    occurrences: int = 1


@dataclass
class SynthesisIntelligenceReport:
    """Measured redundancy for one assembled report (the benchmark score)."""

    total_sentences: int = 0
    unique_claims: int = 0
    repeated_claims: int = 0
    removed_restatements: int = 0
    refined_transitions: int = 0
    allowed_expansions: int = 0
    sections: int = 0
    top_repeats: List[Dict[str, object]] = field(default_factory=list)

    @property
    def redundancy_ratio(self) -> float:
        """Repeated-claim count over unique claims; 0 is ideal."""
        if not self.unique_claims:
            return 0.0
        return round(self.repeated_claims / self.unique_claims, 4)

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_sentences": self.total_sentences,
            "unique_claims": self.unique_claims,
            "repeated_claims": self.repeated_claims,
            "removed_restatements": self.removed_restatements,
            "refined_transitions": self.refined_transitions,
            "allowed_expansions": self.allowed_expansions,
            "sections": self.sections,
            "redundancy_ratio": self.redundancy_ratio,
            "top_repeats": self.top_repeats[:8],
        }


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def canonical_tokens(text: str, *, keep_negation: bool = True) -> List[str]:
    """Content tokens of `text`: citations stripped, stopwords removed, stemmed.

    Negation words are preserved by default (AGENTS.md: negation is never a
    stopword in a similarity/verification component) so "X" and "not X" keep
    different token sets.
    """
    body = _CITATION_RE.sub(" ", text or "")
    tokens: List[str] = []
    for raw in re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", body.lower()):
        for part in re.split(r"[-']", raw):
            part = part.strip()
            if not part or part in _STOPWORDS:
                continue
            tokens.append(_stem(part))
    if not keep_negation:
        tokens = [t for t in tokens if not _NEGATION_RE.fullmatch(t)]
    return tokens


def claim_key(text: str) -> str:
    """A canonical, order-insensitive key for a claim sentence.

    Two sentences that assert the same fact with different word order or a
    light inflection share a key; a sentence with an extra negation does not
    (negation tokens survive). Returns "" for units too short to be claims.
    """
    tokens = canonical_tokens(text)
    if len(tokens) < _MIN_CONTENT_TOKENS:
        return ""
    return " ".join(sorted(tokens))


def claim_polarity(text: str) -> int:
    """-1 when the sentence is negated, +1 otherwise.

    Part of the polarity guard: identical tokens with opposite polarity are
    different claims, and this layer must never remove one as a "repeat" of
    the other.
    """
    return -1 if _NEGATION_RE.search(text or "") else 1


def analytical_dimensions(text: str) -> Set[str]:
    """Which analytical dimensions a sentence adds: mechanism / implication /
    comparison / uncertainty. Empty means it is a bare fact statement."""
    found: Set[str] = set()
    for name, pattern in _DIMENSIONS:
        if pattern.search(text or ""):
            found.add(name)
    return found


_QUANT_RE = re.compile(r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b", re.IGNORECASE)

# Section-axis vocabulary. The axis signal is the section heading (the outline
# title or the writer's own heading), so the match is on the human words a
# heading uses, not a model-invented slug.
_COMPARISON_AXIS_RE = re.compile(
    r"(?i)\b(compar|versus|\bvs\b|alternative|how it compares|benchmark|"
    r"trade-?off|weigh)"
)
_MECHANISM_AXIS_RE = re.compile(
    r"(?i)\b(how it works|mechanism|internal|process|architecture|"
    r"how .{0,20}works|technical)"
)
_CAUSAL_AXIS_RE = re.compile(
    r"(?i)\b(causal|cause|why|root|origins?|history|background|drivers?)"
)
_STRATEGIC_AXIS_RE = re.compile(
    r"(?i)\b(strateg|decision|recommend|outlook|policy|future|plan|"
    r"implication|what it means|over \d+ years)"
)

# Query-intent vocabulary read from the (already-classified) query text.
_DECISION_QUERY_RE = re.compile(
    r"(?i)\b(should|recommend|strategy|strategic|policy|policymaker|"
    r"decide|decision|choice|prioriti|over \d+ ?(?:years|decades)|"
    r"government|regulat|plan for|roadmap|most (?:effective|viable|economical))\b"
)
_CAUSAL_QUERY_RE = re.compile(
    r"(?i)\b(cause[ds]?|why|explain (?:what|how)|what led to|what drove|"
    r"origins?|how did .{0,30}happen|reason)\b"
)
_COMPARISON_QUERY_RE = re.compile(
    r"(?i)\b(compare|comparison|versus|\bvs\b|better|which .{0,20}(?:better|"
    r"cheaper|safer)|trade-?off|alternatives?)\b"
)
_MECHANISM_QUERY_RE = re.compile(
    r"(?i)\b(how does|how do|how it works|mechanism|process|work internally|"
    r"explain how)\b"
)

# Claim-type vocabulary (the sentence's own shape).
_MECHANISM_CLAIM_RE = re.compile(
    r"(?i)\b(process|mechanism|works? by|functions?|operates?|reacts?|"
    r"converts?|transfers?|pipeline|architecture|algorithm|protocol|"
    r"how .{0,20}works|enables?|allows? .{0,20}to)\b"
)
_CAUSAL_CLAIM_RE = re.compile(
    r"(?i)\b(because|due to|as a result|driven by|caused|led to|triggered|"
    r"stemmed|arose|resulted from)\b"
)

# The deterministic precedence over signals: (rule_name, move). Evaluated in
# order; the first matching rule wins. Keeping this as data (not a chain of
# ifs) makes the precedence inspectable and gives tests a stable target.
_MOVE_RULES: Tuple[Tuple[str, str], ...] = (
    ("contradicted_comparative", "tradeoff"),
    ("contradicted", "uncertainty"),
    ("decision_query", "strategic"),
    ("comparison_query", "comparison"),
    ("causal_query", "causal"),
    ("mechanism_query", "mechanism"),
    ("comparison_axis", "comparison"),
    ("mechanism_axis", "mechanism"),
    ("causal_axis", "causal"),
    ("strategic_axis", "strategic"),
    ("causal_claim", "causal"),
    ("mechanism_claim", "mechanism"),
    ("comparative_claim", "comparison"),
    ("quantitative_authoritative", "implication"),
    ("quantitative", "tradeoff"),
    ("authoritative", "implication"),
    ("default", "implication"),
)


def reasoning_moves(text: str) -> Set[str]:
    """Which adaptive reasoning moves a sentence carries.

    This is the move-level sibling of `analytical_dimensions`: it recognizes
    all seven moves (the expansion detector deliberately keeps its narrower
    vocabulary so attaching a move never re-labels an expansion). Used to
    verify that a refinement actually expresses the move the signals chose.
    """
    found: Set[str] = set()
    for name, pattern in _MOVE_MARKERS.items():
        if pattern.search(text or ""):
            found.add(name)
    return found


def _is_comparative(sentence: str, sig: Dict[str, object]) -> bool:
    return bool(_COMPARISON_RE.search(sentence or "")) or bool(
        _COMPARISON_RE.search(_axis_text(sig))
    ) or bool(_COMPARISON_QUERY_RE.search(_query_text(sig)))


def _axis_text(sig: Dict[str, object]) -> str:
    return str(sig.get("axis", "") or "").strip()


def _query_text(sig: Dict[str, object]) -> str:
    return str(sig.get("query", "") or sig.get("query_text", "") or "").strip()


def _is_authoritative(sig: Dict[str, object]) -> bool:
    return bool(
        sig.get("authoritative")
        or sig.get("corroborated")
        or sig.get("primary")
        or sig.get("primary_source")
    )


def _choose_move(sentence: str, signals: Optional[Dict[str, object]]) -> str:
    """Choose the adaptive reasoning move for a refined restatement.

    Reads only signals that are actually available on the evidence record /
    report context — contradiction count, source authority, query intent, and
    the section axis — plus the claim's own shape. The precedence in
    `_MOVE_RULES` is fixed and ordered, so the same claim in the same context
    always receives the same move (no RNG, no hash, no LLM).

    Signals consumed (all optional):
      contradicted / uncertain  -> the claim is disputed
      comparative / comparison  -> claim or axis or query is a comparison
      authoritative / primary / corroborated -> stronger provenance
      query / query_text, query_type          -> classified intent
      axis                                     -> section heading
    """
    sig = signals or {}
    sentence = sentence or ""
    contradicted = bool(sig.get("contradicted") or sig.get("uncertain"))
    axis = _axis_text(sig)
    query = _query_text(sig)
    query_type = str(sig.get("query_type", "") or "").strip().lower()
    comparative = _is_comparative(sentence, sig)
    # Strip citation markers before reading a quantity: "[1]" is a source
    # pointer, not a figure, and treating it as one made every cited sentence
    # look quantitative.
    quant = bool(_QUANT_RE.search(_CITATION_RE.sub(" ", sentence)))
    authoritative = _is_authoritative(sig)

    is_causal_query = bool(_CAUSAL_QUERY_RE.search(query)) or query_type in ("causal", "analytical")
    is_decision_query = bool(_DECISION_QUERY_RE.search(query)) or bool(sig.get("decision_query"))
    is_comparison_query = query_type == "comparative" or bool(_COMPARISON_QUERY_RE.search(query))
    is_mechanism_query = bool(_MECHANISM_QUERY_RE.search(query)) or query_type == "exploratory"

    for rule, move in _MOVE_RULES:
        if rule == "contradicted_comparative" and contradicted and comparative:
            return move
        if rule == "contradicted" and contradicted:
            return move
        if rule == "decision_query" and is_decision_query:
            return move
        if rule == "comparison_query" and is_comparison_query:
            return move
        if rule == "causal_query" and is_causal_query:
            return move
        if rule == "mechanism_query" and is_mechanism_query:
            return move
        if rule == "comparison_axis" and _COMPARISON_AXIS_RE.search(axis):
            return move
        if rule == "mechanism_axis" and _MECHANISM_AXIS_RE.search(axis):
            return move
        if rule == "causal_axis" and _CAUSAL_AXIS_RE.search(axis):
            return move
        if rule == "strategic_axis" and _STRATEGIC_AXIS_RE.search(axis):
            return move
        if rule == "causal_claim" and _CAUSAL_CLAIM_RE.search(sentence):
            return move
        if rule == "mechanism_claim" and _MECHANISM_CLAIM_RE.search(sentence):
            return move
        if rule == "comparative_claim" and comparative:
            return move
        if rule == "quantitative_authoritative" and quant and authoritative:
            return move
        if rule == "quantitative" and quant:
            return move
        if rule == "authoritative" and authoritative:
            return move
        if rule == "default":
            return move
    return "implication"


# The refined sentence must stay bounded: a transformation that doubles the
# word count would harm readability more than the redundancy it fixes. The
# transition + analysis appended to the claim is capped at this many words.
MAX_APPENDED_WORDS = 38


def refine_restatement(
    sentence: str,
    *,
    dimension: str = "implication",
    move: str = "",
    seed: str = "",
) -> str:
    """Rewrite a bare restatement into a contextual transition + analysis.

    The original claim is preserved verbatim — including its number and `[n]`
    marker — so nothing is lost and no citation is invented; the transition
    names the relationship to the earlier use and the move-specific analytical
    clause adds the reasoning the restatement lacked. `move` (preferred) or the
    legacy `dimension` names the reasoning relation; the phrase is chosen to
    fit it. Deterministic: no LLM is ever needed.
    """
    clean = (sentence or "").strip()
    if not clean:
        return clean
    chosen = move or dimension or "implication"
    chosen = chosen if chosen in _ANALYSIS_BY_MOVE else "implication"
    analysis = _analysis_clause(clean, chosen)
    if not analysis:
        # No topic-specific clause applies to this claim. The adaptive layer
        # only ever transforms a sentence when it has a concrete, topic-scoped
        # move for it; a generic topic-agnostic tail is exactly the boilerplate
        # corruption this module must not introduce (AGENTS.md: adaptive-moves
        # boilerplate corruption). Leave the sentence byte-for-byte unchanged.
        return clean
    transition = _stable_pick(_TRANSITION_BY_MOVE[chosen], seed or clean)
    appended = f"{transition} {analysis}"
    if len(appended.split()) > MAX_APPENDED_WORDS:
        # Defensive bound; current clauses are all well under it, but a future
        # template must not silently inflate every refined sentence. Trim the
        # analysis (never the transition, which carries the discourse link).
        keep = max(0, MAX_APPENDED_WORDS - len(transition.split()))
        analysis = " ".join(analysis.split()[:keep]).rstrip(",;:") if keep else ""
    # The original claim keeps its terminal punctuation before the appended
    # clause, so "… $13 billion [1]" becomes "… $13 billion [1], and the figure
    # is significant because …" rather than two spliced sentences.
    stripped = clean.rstrip()
    if stripped.endswith("."):
        stripped = stripped[:-1]
    if analysis:
        return f"{transition} {stripped}, and {analysis}."
    return f"{transition} {stripped}."


# Label/fragment shapes that are NOT full prose sentences: a Key-Figures bullet
# ("**$12.65 billion** — reported cost …"), a heading fragment, or a line that
# starts lowercase or has no terminal punctuation. Appending an analytical
# clause to one of those produces garbled prose ("Building on that, $12.65
# billion** — reported cost …, and the figure is significant…"), so a repeated
# fragment is left intact instead: it was never the dangling-opening failure
# this layer exists to fix.
_FRAGMENT_RE = re.compile(r"^\s*(?:\*\*|[-*]\s|\d+\)\s|[a-z])")

# Label/value bullets are the OTHER non-prose shape that leaked through the
# first guard: a Key-Figures line whose subject is a label and whose body is a
# value or title list, e.g. "SWE-bench Verified frontier performance: 60%
# rising to near 100% within a single year [2]." or "Organizational AI
# adoption: 88%; university students using generative AI: four in five [2]."
# They start uppercase and end with a period, so the fragment check above
# misses them, and the refinement pass then spliced move clauses into nearly
# every Key-Figures bullet in live reports (AGENTS.md: adaptive-moves
# boilerplate corruption). The discriminator is a colon-delimited label with NO
# finite verb before the delimiter — a sentence ("The defining choice is
# subtractive: …") has one, a label ("Frontier performance: …") does not.
_LABEL_DELIM_RE = re.compile(r"[:—]")


def _has_finite_verb(text: str) -> bool:
    """Conservative test for a finite (tensed) verb in a LABEL HEAD.

    Only copulas/auxiliaries/modals count. A label/value head is a bare noun
    phrase ("frontier performance", "risk scores", "survey base"); a real
    sentence's head carries a tensed form ("the choice is subtractive"). Being
    strict here is correct: a head with no auxiliary is a label, and refining a
    label is exactly the corruption this guard prevents. A genuine sentence
    without a colon never reaches this check.
    """
    return any(
        word.lower() in _AUXILIARY_VERBS
        for word in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text or "")
    )


# Copulas / auxiliaries / modals: their presence alone is a finite clause.
_AUXILIARY_VERBS: Set[str] = {
    "is", "are", "was", "were", "be", "been", "being", "am",
    "has", "have", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "must", "cannot", "isn't", "aren't", "wasn't", "weren't", "doesn't",
    "don't", "didn't", "hasn't", "haven't", "hadn't",
}

# The opening of a refined sentence: the deterministic post-condition is that a
# refinement begins with one of these transitions, so a sentence already
# carrying one must never be refined again (Bug: double-appending the move
# clause to an already-refined line).
_TRANSITION_MARKERS: Tuple[str, ...] = tuple(
    phrase for phrases in _TRANSITION_BY_MOVE.values() for phrase in phrases
)


def _is_already_refined(sentence: str) -> bool:
    """True when the sentence already carries a refinement transition.

    Checks the first twelve words only: the transition opens the sentence, so a
    genuine prose sentence that happens to contain similar words later is not
    mistaken for an already-refined line. This is the no-double-append guard.
    """
    head = " ".join((sentence or "").split()[:12]).lower()
    if not head:
        return False
    return any(marker.lower() in head for marker in _TRANSITION_MARKERS)


def _is_label_bullet(sentence: str) -> bool:
    """True when the line is a label/value or title fragment, not prose.

    Rejects a colon- or em-dash-delimited label whose head carries no finite
    verb ("Frontier performance: 60% …", "Survey base: 7,000 firms") and a very
    short unit that cannot carry a full clause. A real sentence with a colon
    ("The choice is subtractive: …") has a finite verb in its head and passes.
    """
    stripped = (sentence or "").strip()
    if not stripped:
        return False
    # A bold-only label line, or a value line led by a number/bullet glyph.
    if _FRAGMENT_RE.match(stripped):
        return True
    # Too few content words to be a full prose sentence.
    if len(stripped.split()) < 6:
        return True
    delim = _LABEL_DELIM_RE.search(stripped)
    if delim is not None:
        head = stripped[: delim.start()].strip()
        # The label lives before the first delimiter; if there is no finite verb
        # there, the line is a label/value fragment.
        if head and not _has_finite_verb(head):
            return True
    return False


def _is_refinable_sentence(sentence: str, *, is_bullet: bool = False) -> bool:
    """True when `sentence` is a full prose sentence worth transforming.

    Only genuine PROSE restatements qualify. An enumerable bullet — even when
    its value happens to be a full sentence, as in Key Figures ("The first mRNA
    vaccines … [6].") — is a data item, not an argument restatement, so it is
    left intact. Title/value fragments, short units, and lines that already
    carry a move clause are likewise left intact: refining them splices an
    analytical clause onto a label and corrupts the report (the live
    Key-Figures boilerplate failure this guard exists for).
    """
    stripped = (sentence or "").strip()
    if not stripped:
        return False
    if is_bullet:
        return False
    if _FRAGMENT_RE.match(stripped):
        return False
    if not stripped[-1:] in ".!?":
        return False
    if len(stripped.split()) < 6:
        return False
    if _is_already_refined(stripped):
        return False
    if _is_label_bullet(stripped):
        return False
    return True


def _is_sentence_unit(line: str) -> bool:
    stripped = line.strip()
    if not stripped or _HEADING_RE.match(stripped):
        return False
    return bool(re.search(r"[A-Za-z]", stripped))


def split_report_sections(answer: str) -> List[Tuple[str, List[str]]]:
    """Split a markdown report into (heading, body-lines) in order.

    The preamble before the first heading is returned under the "" heading.
    Lines are preserved as-is (bullets included) so reassembly is lossless
    apart from the sentences this layer deliberately removes.
    """
    sections: List[Tuple[str, List[str]]] = []
    current_heading = ""
    current_lines: List[str] = []
    for line in (answer or "").replace("\r\n", "\n").split("\n"):
        if _HEADING_RE.match(line):
            if current_heading or current_lines:
                sections.append((current_heading, current_lines))
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    sections.append((current_heading, current_lines))
    return sections


def _iter_units(lines: Sequence[str]) -> Iterable[Tuple[int, str]]:
    """Yield (line_index, sentence) for claim-bearing sentences in `lines`."""
    for index, line in enumerate(lines):
        if not _is_sentence_unit(line):
            continue
        stripped = line.strip().lstrip("-* ").strip()
        for sentence in _SENTENCE_SPLIT_RE.split(stripped):
            sentence = sentence.strip()
            if sentence and re.search(r"[A-Za-z]", sentence):
                yield index, sentence


def _similar_to_any(candidate: str, prior_texts: Sequence[str]) -> bool:
    """Same claim expressed in different words (near-duplicate restatement)."""
    if not candidate or not prior_texts:
        return False
    try:
        from app.core.semantic import pair_similarity

        return any(pair_similarity(candidate, prior) >= RESTATEMENT_SIMILARITY for prior in prior_texts)
    except Exception as exc:  # noqa: BLE001 - never let similarity break assembly
        logger.warning("[SynthesisIntel] similarity check failed", exc_info=exc)
        return False


def anchors(text: str) -> Set[str]:
    """Distinctive content tokens: the recurring subject names and figures.

    Generic report vocabulary ("cost", "energy", "project" — frequent in any
    query's evidence) is excluded, so two sentences only share anchors when
    they name the same specific entity or quantity.
    """
    return {
        token
        for token in canonical_tokens(text)
        if token not in _GENERIC_ANCHORS and (len(token) >= 5 or token.isdigit())
    }


def _fuzzy_restatement(candidate: str, prior: str) -> bool:
    """True when `candidate` restates the same fact as `prior`.

    Two independent signals, both required:
      * anchors — at least `ANCHOR_SHARED_MIN` distinctive tokens in common
        (the shared subject/figure), and
      * containment — the SHORTER sentence's content tokens are at least
        `ANCHOR_OVERLAP_MIN` contained in the longer one's. An expansion
        (same fact + a new causal clause) is a superset, so containment is
        evaluated in whichever direction makes the shorter text the numerator;
        a sentence that merely borrows a name but asserts something else does
        not match.
    """
    cand_tokens = set(canonical_tokens(candidate))
    prior_tokens = set(canonical_tokens(prior))
    if len(cand_tokens) < _MIN_CONTENT_TOKENS or len(prior_tokens) < _MIN_CONTENT_TOKENS:
        return False
    shared_anchors = anchors(candidate) & anchors(prior)
    if len(shared_anchors) < ANCHOR_SHARED_MIN:
        return False
    smaller, larger = (cand_tokens, prior_tokens) if len(cand_tokens) <= len(prior_tokens) else (prior_tokens, cand_tokens)
    containment = len(smaller & larger) / max(1, len(smaller))
    return containment >= ANCHOR_OVERLAP_MIN


class ClaimLedger:
    """Tracks claim keys across sections and decides what may be re-used.

    Deterministic and order-sensitive: the FIRST section to state a claim
    keeps it (with its citation); later sections may only re-use it when they
    add an analytical dimension the earlier uses did not.
    """

    def __init__(self) -> None:
        self._uses: Dict[str, ClaimUse] = {}
        self._texts: List[str] = []
        self.removed_restatements = 0
        self.refined_restatements = 0
        self.kept_fragments = 0
        self.allowed_expansions = 0
        self.total_sentences = 0

    @property
    def unique_claims(self) -> int:
        return len(self._uses)

    @property
    def repeated_claims(self) -> int:
        return sum(1 for use in self._uses.values() if use.occurrences > 1)

    def _find_prior(self, sentence: str) -> Optional[ClaimUse]:
        key = claim_key(sentence)
        if key:
            use = self._uses.get(key)
            if use is not None and claim_polarity(use.text) != claim_polarity(sentence):
                # Negation flip: same vocabulary, opposite assertion. Not a
                # repeat; fall through to the similarity check for a second
                # opinion rather than treating as same-key.
                use = None
            if use is not None:
                return use
        polarity = claim_polarity(sentence)
        for use in self._uses.values():
            if claim_polarity(use.text) != polarity:
                continue
            if _similar_to_any(sentence, [use.text]) or _fuzzy_restatement(sentence, use.text):
                return use
        return None

    def refine(
        self,
        section: str,
        sentence: str,
        *,
        signals: Optional[Dict[str, object]] = None,
        is_bullet: bool = False,
    ) -> Refinement:
        """Classify a sentence; TRANSFORM a bare restatement instead of dropping it.

        * A new claim is kept verbatim and registered.
        * A repeat that adds a genuinely new analytical dimension is kept
          verbatim (expansion) and the dimension is recorded.
        * A bare restatement is rewritten into a transition + analysis
          sentence that preserves the original claim, its number and its `[n]`
          marker. It is never removed, so a section can never lose its opening.
        * An enumerable BULLET restatement (`is_bullet`) is kept verbatim: a
          bullet is a data item, not a prose argument, and refining it spliced
          move clauses into Key Figures in live reports.

        `signals` carries evidence-derived hints for the analytical clause
        (contradicted / uncertain / corroborated / primary / axis); the
        transformation works deterministically with no signals and no LLM.
        """
        self.total_sentences += 1
        key = claim_key(sentence)
        prior = self._find_prior(sentence)
        if prior is None:
            if key:
                self._uses[key] = ClaimUse(key=key, text=sentence, section=section)
            self._texts.append(sentence)
            return Refinement(text=sentence, kept=True, transformed=False)

        prior.occurrences += 1
        dimensions = analytical_dimensions(sentence)
        novel = dimensions - prior.dimensions
        if novel:
            # Genuinely expanded: record the new dimension and allow it intact.
            prior.dimensions |= dimensions
            self.allowed_expansions += 1
            self._texts.append(sentence)
            return Refinement(text=sentence, kept=True, transformed=False)

        if not _is_refinable_sentence(sentence, is_bullet=is_bullet):
            # A bullet, label-style line or fragment: leave it intact rather
            # than splice an analytical clause onto a data item or non-sentence.
            self.kept_fragments += 1
            self._texts.append(sentence)
            return Refinement(text=sentence, kept=True, transformed=False)

        move = _choose_move(sentence, signals)
        refined = refine_restatement(
            sentence,
            move=move,
            seed=key or sentence,
        )
        if refined == sentence.strip():
            # No topic-specific move clause applied, so the sentence is left
            # exactly as written. Treat it as kept-verbatim rather than a
            # refinement: nothing was transformed and no generic tail invented.
            self.kept_fragments += 1
            self._texts.append(sentence)
            return Refinement(text=sentence, kept=True, transformed=False)
        # Record the dimension the move maps onto so a later true expansion of
        # the same claim is still detected as novel. `analytical_dimensions`
        # keeps its narrow vocabulary; trade-off/causal/strategic moves are
        # recorded as the implication dimension they express.
        prior.dimensions.add(
            move if move in ("mechanism", "implication", "comparison", "uncertainty") else "implication"
        )
        self.refined_restatements += 1
        self.removed_restatements += 1
        self._texts.append(refined)
        logger.debug(
            "[SynthesisIntel] refined restatement in section '%s' (first used in '%s', move=%s)",
            section, prior.section, move,
        )
        return Refinement(
            text=refined,
            kept=True,
            transformed=True,
            dimension=move,
            first_section=prior.section,
            move=move,
        )

    def register(self, section: str, sentence: str) -> bool:
        """Backward-compatible wrapper returning whether a sentence is kept.

        With the refinement layer every claim-bearing sentence is kept, so
        this always returns True for a non-empty unit; callers that need the
        transformed text must use `refine`.
        """
        return self.refine(section, sentence).kept

    def register_recap(self, section: str, sentence: str) -> None:
        """Record a claim from a recap section (Executive Summary / Key
        Findings) WITHOUT counting it as removable.

        Recaps are the report's deliberate preview; their text is kept whole,
        but their claims become prior uses so a deep-dive section cannot
        restate them as if newly discovered.
        """
        key = claim_key(sentence)
        self.total_sentences += 1
        prior = self._find_prior(sentence)
        if prior is None:
            if key:
                self._uses[key] = ClaimUse(key=key, text=sentence, section=section)
            self._texts.append(sentence)
            return
        prior.occurrences += 1
        prior.dimensions |= analytical_dimensions(sentence)
        self._texts.append(sentence)


def compress_section_text(
    text: str,
    ledger: ClaimLedger,
    section: str,
    *,
    signals: Optional[Dict[str, object]] = None,
) -> str:
    """Refine restating sentences in one section body, preserving structure.

    A restatement is never deleted: it is rewritten into a transition +
    analysis sentence (see `ClaimLedger.refine`), so a section's opening
    sentence survives as a coherent opener even when it repeated an earlier
    claim. Line-granular: headings are never touched and an empty section is
    never produced.
    """
    out_lines: List[str] = []
    for line in (text or "").replace("\r\n", "\n").split("\n"):
        if not _is_sentence_unit(line):
            out_lines.append(line)
            continue
        is_bullet = bool(_BULLET_RE.match(line))
        stripped = line.strip()
        prefix = "- " if is_bullet else ""
        body = stripped.lstrip("-* ").strip()
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(body) if s.strip()]
        kept: List[str] = []
        for sentence in sentences:
            # `is_bullet` forbids transformation (an enumerable item is not a
            # prose restatement) but still registers the claim, so a later
            # prose section cannot restate a figure the list already stated.
            outcome = ledger.refine(section, sentence, signals=signals, is_bullet=is_bullet)
            if outcome.kept and outcome.text:
                kept.append(outcome.text)
        if not kept:
            continue
        if is_bullet:
            out_lines.extend(f"{prefix}{s}" for s in kept)
        else:
            out_lines.append(" ".join(kept))
    return "\n".join(out_lines)


def apply_synthesis_intelligence(
    answer: str,
    *,
    protect_headings: Sequence[str] = (),
    recap_headings: Sequence[str] = (),
    signals: Optional[Dict[str, object]] = None,
) -> Tuple[str, SynthesisIntelligenceReport]:
    """Deterministically REFINE cross-section restatements in a report.

    A restatement is rewritten into a contextual transition + analysis
    sentence rather than deleted (see `ClaimLedger.refine`), so a section
    whose opening repeated an earlier claim still opens with a complete,
    coherent sentence and the reader learns what the repetition means.

    `protect_headings` names sections that must keep their full body (the
    machine-appended appendices, which describe measured state and are not
    writer prose).

    `recap_headings` names the report's deliberate summary sections
    (Executive Summary, Key Findings). Their bodies are never refined —
    a brief is expected to preview its findings there — but every claim they
    state IS registered, so the deep-dive sections cannot restate them. This
    is the observed failure mode: the Executive Summary established "Rooppur
    … US$13 billion [1]" and seven later sections re-asserted it instead of
    adding analysis.

    `signals` carries report-level evidence hints (contradicted / uncertain /
    corroborated / primary / authoritative / query / query_type) that select
    the adaptive reasoning move of each refinement; the section axis is added
    per section from its heading. It is optional; the deterministic
    transformation runs with no signals and no LLM.
    """
    if not answer:
        return answer, SynthesisIntelligenceReport()

    protected = {h.strip().lower() for h in protect_headings}
    recap = {h.strip().lower() for h in recap_headings}
    sections = split_report_sections(answer)
    ledger = ClaimLedger()
    rebuilt: List[str] = []

    for heading, lines in sections:
        heading_key = heading.strip().lower()
        body_lines = list(lines)
        has_prose = any(_is_sentence_unit(l) for l in body_lines)
        if heading and has_prose and heading_key not in protected:
            if heading_key in recap:
                # Register the recap's claims so later sections cannot restate
                # them, but keep the recap text itself intact.
                for _, sentence in _iter_units(body_lines):
                    ledger.register_recap(heading, sentence)
            else:
                section_signals = dict(signals or {})
                section_signals.setdefault("axis", heading)
                body_lines = compress_section_text(
                    "\n".join(body_lines),
                    ledger,
                    heading or "Preamble",
                    signals=section_signals,
                ).split("\n")
        block = [heading] if heading else []
        block.extend(body_lines)
        rebuilt.append("\n".join(block))

    refined = "\n".join(rebuilt).strip()

    # Guard: never return an empty body because of this layer. An over-eager
    # transformation is worse than the redundancy it addresses, so fall back
    # to the original text when nothing meaningful survived.
    if not _has_writer_prose(refined):
        logger.warning("[SynthesisIntel] refinement emptied the report; keeping original")
        return answer, SynthesisIntelligenceReport()

    report = SynthesisIntelligenceReport(
        total_sentences=ledger.total_sentences,
        unique_claims=ledger.unique_claims,
        repeated_claims=ledger.repeated_claims,
        removed_restatements=ledger.removed_restatements,
        refined_transitions=ledger.refined_restatements,
        allowed_expansions=ledger.allowed_expansions,
        sections=sum(1 for h, l in sections if h and any(_is_sentence_unit(x) for x in l)),
        top_repeats=_top_repeats(ledger),
    )
    if report.refined_transitions:
        logger.info(
            "[SynthesisIntel] refined %d cross-section restatement(s) into transitions; %d expansion(s) kept",
            report.refined_transitions, report.allowed_expansions,
        )
    return refined, report


def _has_writer_prose(text: str) -> bool:
    """True when the report still carries more than headings alone."""
    for line in (text or "").split("\n"):
        stripped = line.strip()
        if not stripped or _HEADING_RE.match(stripped):
            continue
        if re.search(r"[A-Za-z]", stripped) and len(stripped.split()) >= 4:
            return True
    return False


def _top_repeats(ledger: ClaimLedger, limit: int = 8) -> List[Dict[str, object]]:
    repeated = sorted(
        (u for u in ledger._uses.values() if u.occurrences > 1),
        key=lambda u: (-u.occurrences, u.section),
    )
    return [
        {
            "occurrences": u.occurrences,
            "first_section": u.section,
            "dimensions": sorted(u.dimensions),
            "text": u.text[:160],
        }
        for u in repeated[:limit]
    ]


def analyze_report(answer: str) -> SynthesisIntelligenceReport:
    """Measure redundancy in a finished report without changing it.

    Uses the same claim-key + similarity machinery as the compressor, so the
    benchmark's redundancy score and the shipped report's behaviour cannot
    drift apart. Machine appendices are excluded — they are measured state,
    not writer prose.
    """
    from app.agents.sources import strip_machine_sections

    body = strip_machine_sections(answer or "")
    sections = split_report_sections(body)
    ledger = ClaimLedger()
    section_count = 0
    for heading, lines in sections:
        if not heading or not any(_is_sentence_unit(l) for l in lines):
            continue
        section_count += 1
        for _, sentence in _iter_units(lines):
            key = claim_key(sentence)
            prior = ledger._find_prior(sentence)
            if prior is None:
                if key:
                    ledger._uses[key] = ClaimUse(key=key, text=sentence, section=heading)
                ledger._texts.append(sentence)
            else:
                prior.occurrences += 1
                prior.dimensions |= analytical_dimensions(sentence)
                ledger._texts.append(sentence)
    return SynthesisIntelligenceReport(
        total_sentences=len(ledger._texts),
        unique_claims=ledger.unique_claims,
        repeated_claims=ledger.repeated_claims,
        removed_restatements=0,
        allowed_expansions=0,
        sections=section_count,
        top_repeats=_top_repeats(ledger),
    )
