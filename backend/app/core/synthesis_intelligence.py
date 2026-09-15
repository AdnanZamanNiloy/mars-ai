"""Synthesis intelligence layer — repetition tracking and expansion enforcement.

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

GPT Researcher avoids this differently (a single writer over compressed
context); MARS writes section-wise for provider-size reasons, so the fix has
to be explicit cross-section bookkeeping. This module is that bookkeeping,
and it is deterministic — no LLM, no network, no new model. It does not touch
retrieval, grading, corroboration or contradiction; it only decides which
already-written sentences survive assembly.

What it does
------------
1. CLAIM-KEY TRACKING — every writer sentence is reduced to a canonical
   `claim_key` (citations stripped, stopwords removed, tokens stemmed and
   sorted). The same key seen in an earlier section is a REPEAT.
2. DIMENSION-AWARE PENALTY — a repeat is only allowed when it adds a new
   analytical dimension the earlier use did not: a mechanism ("because",
   "driven by"), an implication/trade-off ("therefore", "at the cost of"),
   a comparison ("compared with", "whereas") or an explicit uncertainty
   ("unresolved", "not established"). A verbatim restatement of the same
   fact in a new section is REMOVED, keeping the first occurrence and its
   citation. This is the deterministic fallback: it runs whether or not the
   LLM obeyed the expansion instruction.
3. CITATION PRESERVATION — retained sentences keep their exact `[n]` markers;
   removed sentences take their markers with them; no marker is ever
   invented, renumbered or moved between sentences.
4. STRUCTURE PRESERVATION — headings, section order and the required
   sections are never touched; a section reduced to a heading alone is
   dropped only when it had no unique content at all.

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
    r"which forces|forcing|trigger(?:s|ed|ing)?|explains? why|"
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


def _plural_keys(texts: Sequence[str]) -> Set[str]:
    return {claim_key(t) for t in texts if claim_key(t)}


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

    def register(self, section: str, sentence: str) -> bool:
        """Register a sentence; return True when it should be KEPT.

        A new claim is always kept. An existing claim is kept only when this
        occurrence introduces an analytical dimension not seen before; a
        verbatim restatement is rejected (and its citation goes with it).
        """
        self.total_sentences += 1
        key = claim_key(sentence)
        prior = self._find_prior(sentence)
        if prior is None:
            if key:
                self._uses[key] = ClaimUse(key=key, text=sentence, section=section)
            self._texts.append(sentence)
            return True

        prior.occurrences += 1
        dimensions = analytical_dimensions(sentence)
        novel = dimensions - prior.dimensions
        if novel:
            # Genuinely expanded: record the new dimension and allow it.
            prior.dimensions |= dimensions
            self.allowed_expansions += 1
            self._texts.append(sentence)
            return True

        self.removed_restatements += 1
        logger.debug(
            "[SynthesisIntel] removed restatement in section '%s' (first used in '%s')",
            section, prior.section,
        )
        return False

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


def compress_section_text(text: str, ledger: ClaimLedger, section: str) -> str:
    """Remove restating sentences from one section body, preserving structure.

    Line-granular: a bullet whose only sentence is a restatement is dropped
    whole; a prose line keeps only its non-restating sentences. Headings are
    never touched.
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
            if ledger.register(section, sentence):
                kept.append(sentence)
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
) -> Tuple[str, SynthesisIntelligenceReport]:
    """Deterministically remove cross-section restatements from a report.

    `protect_headings` names sections that must keep their full body (the
    machine-appended appendices, which describe measured state and are not
    writer prose).

    `recap_headings` names the report's deliberate summary sections
    (Executive Summary, Key Findings). Their bodies are never compressed —
    a brief is expected to preview its findings there — but every claim they
    state IS registered, so the deep-dive sections cannot restate them. This
    is the observed failure mode: the Executive Summary established "Rooppur
    … US$13 billion [1]" and seven later sections re-asserted it instead of
    adding analysis.
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
                body_lines = compress_section_text(
                    "\n".join(body_lines), ledger, heading or "Preamble"
                ).split("\n")
        block = [heading] if heading else []
        block.extend(body_lines)
        rebuilt.append("\n".join(block))

    compressed = "\n".join(rebuilt).strip()

    # Guard: never return an empty body because of this layer. An over-eager
    # compression is worse than the redundancy it removes, so fall back to the
    # original text when nothing meaningful survived.
    if not _has_writer_prose(compressed):
        logger.warning("[SynthesisIntel] compression emptied the report; keeping original")
        return answer, SynthesisIntelligenceReport()

    report = SynthesisIntelligenceReport(
        total_sentences=ledger.total_sentences,
        unique_claims=ledger.unique_claims,
        repeated_claims=ledger.repeated_claims,
        removed_restatements=ledger.removed_restatements,
        allowed_expansions=ledger.allowed_expansions,
        sections=sum(1 for h, l in sections if h and any(_is_sentence_unit(x) for x in l)),
        top_repeats=_top_repeats(ledger),
    )
    if report.removed_restatements:
        logger.info(
            "[SynthesisIntel] removed %d cross-section restatement(s); %d expansion(s) kept",
            report.removed_restatements, report.allowed_expansions,
        )
    return compressed, report


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
