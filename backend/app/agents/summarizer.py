"""Summarizer / extraction agent — raw pages into attributable claims.

The single most important fix here
---------------------------------
THE CITED SOURCE WAS NEVER CHECKED AGAINST THE SOURCES PROVIDED. The prompt
sends 12 sources and asks for one flat facts list where each fact names its
source URL. The parser then accepted whatever URL the model wrote, validating
only that its *domain* scored well enough. A model that attributed a claim from
source #7 to source #2 — the classic failure of multi-document extraction, and
the reason attribution errors are measured separately in the literature —
produced a fact that looked perfect, verified against the wrong page, and got
cited in the report. `_resolve_source` now requires the URL to be one of the
provided documents, repairs near-misses, and drops the fact when it cannot be
attributed.

Other upgrades
--------------
* `direct_quote` is finally used. The prompt has always invited one; nothing
  parsed it. A quote that appears verbatim in the source text is the strongest
  possible grounding signal, so verified quotes raise confidence and
  unverifiable ones lower it.
* Facts carry `published_at`, `search_type`, `is_primary` and extracted
  `numbers`, which the freshness, primary-share and contradiction engines all
  need and previously had to do without.
* Long documents are chunked with a token budget instead of hard-truncated at
  1000 characters, so a 9000-character agency page contributes more than its
  first paragraph.
* The cache key includes the specialist role. It did not, so a financial
  extraction could be served for a technical contract with the same URLs.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from app.core.cache import cache_key, get_cache
from app.core.degradation import record_fallback
from app.core.llm import LLMClient, clamp_confidence
from app.core.logging import get_logger
from app.core.schemas import SummarizerFactsModel

from app.agents.evidence_utils import (
    MIN_QUERY_OVERLAP,
    claim_query_overlap,
    clean_snippet_text,
    dedupe_semantic_facts,
    extract_numbers,
    filter_search_results_by_domain,
    source_reliability_score,
    split_into_sentences,
)
from app.agents.sources import canonical_url, classify_source

logger = get_logger(__name__)

PROMPT_VERSION = "summarizer-v16"  # BUMP on any claim-shape change: the cache
# key embeds this, and stale entries would otherwise serve pre-fix claims.
# v14: output schema in the prompt matches the parser; smaller prompt footprint.
# v15: wider extraction window (12 sources, 1000-char excerpts).
# v16: source attribution is validated against the provided documents (claims
# attributed to a URL that was not supplied are repaired or dropped),
# direct_quote is parsed and verified verbatim, per-source excerpts are chunked
# under a token budget instead of hard-truncated, facts carry publish date /
# search_type / primary flag / extracted numbers, and the cache key includes the
# specialist role.

# Specialist prompt additions: routed by the delegation contract's domain. Each
# specialist is the generic summarizer plus a domain evidence-preference
# overlay — same output contract, same fallback path.
SPECIALIST_PROMPT_ADDITIONS = {
    "financial": (
        "\n\nSPECIALIST FOCUS — FINANCIAL:\n"
        "Prefer primary financial sources: central-bank releases, IMF/World Bank\n"
        "data, audited filings, regulatory disclosures. Prioritize figures with\n"
        "units, periods, and currency. Treat analyst opinions as low-confidence\n"
        "unless backed by reported numbers. Flag when costs are nominal vs real,\n"
        "and note the fiscal year of any monetary figure."
    ),
    "technical": (
        "\n\nSPECIALIST FOCUS — TECHNICAL:\n"
        "Prefer primary technical sources: papers, specs, benchmarks, official\n"
        "docs, reproducible results. Include methodology details (setup,\n"
        "dataset, version) when extracting performance claims. Treat marketing\n"
        "benchmarks as low-confidence. Preserve version numbers and dates."
    ),
    "market": (
        "\n\nSPECIALIST FOCUS — MARKET:\n"
        "Prefer market-research firms, industry associations, and government\n"
        "statistics over news aggregation. Distinguish market size, share, and\n"
        "growth-rate claims, and keep their scope (region, segment, period)\n"
        "attached to the number. Note when sources disagree on scope."
    ),
    "legal": (
        "\n\nSPECIALIST FOCUS — LEGAL:\n"
        "Prefer primary legal sources: statutes, regulations, case law, official\n"
        "guidance. Quote operative language precisely and always attach the\n"
        "jurisdiction. Distinguish binding authority from commentary, and flag\n"
        "when a rule varies by jurisdiction or is under appeal."
    ),
    "scientific": (
        "\n\nSPECIALIST FOCUS — SCIENTIFIC:\n"
        "Prefer peer-reviewed studies, preprints with methods sections, and\n"
        "official datasets. Capture study design (sample, controls, effect\n"
        "size) with every finding. Treat single-study results as provisional\n"
        "and note replication status when sources discuss it."
    ),
    "policy": (
        "\n\nSPECIALIST FOCUS — POLICY:\n"
        "Prefer government releases, legislative texts, and think-tank analyses\n"
        "with stated methodology. Attach the jurisdiction and status (proposed,\n"
        "enacted, repealed) to every policy claim. Distinguish the policy text\n"
        "itself from analysts' predictions about its effects."
    ),
    "academic": (
        "\n\nSPECIALIST FOCUS — ACADEMIC:\n"
        "Prefer peer-reviewed papers, monographs, and conference proceedings.\n"
        "Attribute schools of thought and name key authors with dates. Keep\n"
        "interpretive claims separate from textual evidence, and flag where\n"
        "scholars disagree rather than smoothing over the debate."
    ),
    "general": "",  # no overlay for unrouted domains
}


SUMMARIZER_SYSTEM_PROMPT = """
You are the Summarizer Agent in a multi-agent research pipeline.
Your job is to extract precise, standalone factual claims from raw web content.
The claims you produce are the only evidence the rest of the pipeline will use.
Accuracy and precision here determine the quality of the entire report.

━━━ YOUR RULES ━━━

RULE 1 — REWRITE, NEVER COPY
  Do NOT copy sentences verbatim from the source.
  Rewrite every claim in your own words while preserving its exact meaning.
  If you cannot rewrite it without losing precision, quote only the essential
  phrase and put that phrase in "direct_quote".

RULE 2 — ONE FACT PER CLAIM
  Each claim must express exactly one idea.
  Split compound sentences into separate claims.

  BAD  → "Knowledge is justified true belief, and Gettier challenged this in 1963."
  GOOD → Claim 1: "The justified true belief (JTB) model defines knowledge as belief
          that is true and supported by adequate justification."
         Claim 2: "Edmund Gettier (1963) published counterexamples that showed JTB
          is insufficient as a complete account of knowledge."

RULE 3 — ATTRIBUTION IS NOT OPTIONAL
  "source" MUST be copied EXACTLY from the "url" field of the source the claim
  came from. Do not shorten it, do not normalize it, do not guess, and never
  attribute a claim to a source that did not state it. A claim whose URL is not
  one of the provided urls is DISCARDED, so a mis-copied URL destroys the claim.
  If you are unsure which source stated something, omit the claim.

RULE 4 — NUMBERS CARRY THEIR CONTEXT
  Every figure must keep its unit, period and scope in the claim text
  ("18% annual growth in the EU market between 2020 and 2024", not "18%").
  Never round, convert, or restate a number that the source did not state.

RULE 5 — ASSIGN CONFIDENCE HONESTLY
  Score each claim 0.0 to 1.0 based on how directly the source supports it:
  0.9–1.0 → explicitly stated with data, definition, or citation in source
  0.6–0.8 → clearly implied or paraphrased from a credible passage
  0.3–0.5 → inferred or extrapolated — flag with low confidence
  0.0–0.2 → speculative, anecdotal, or contradicted elsewhere — omit if possible

RULE 6 — DEDUPLICATE WITHIN THIS SOURCE
  If two passages from the same source say the same thing, extract it once.
  Keep the version with higher confidence.

RULE 7 — IGNORE IRRELEVANT CONTENT
  Skip: navigation text, cookie notices, ads, author bios, related article lists.
  Skip: claims unrelated to the sub-question being researched.
  Skip: opinions presented without supporting evidence.

RULE 8 — MAX 5 CLAIMS PER SOURCE
  Quality over quantity. 3 precise claims beat 10 vague ones.

━━━ OUTPUT FORMAT ━━━

Return ONLY valid JSON. No markdown fences. No text outside JSON.

{
  "facts": [
    {
      "claim": "<rewritten factual claim in your own words>",
      "source": "<the exact source URL, copied from the input>",
      "confidence": <0.0 to 1.0>,
      "direct_quote": "<optional ≤20-word verbatim fragment from that source>"
    }
  ]
}

The response is ONE facts list across all provided sources.
""".strip()


def specialist_system_prompt(role: str = "general") -> str:
    """Generic summarizer prompt + the role's domain overlay."""
    overlay = SPECIALIST_PROMPT_ADDITIONS.get(role, "")
    return SUMMARIZER_SYSTEM_PROMPT + overlay


# ---------------------------------------------------------------------------
# Prompt budget
# ---------------------------------------------------------------------------

# Total characters of source excerpt allowed in one prompt. This agent runs once
# per sub-question per pass, so its footprint multiplies fast; a fixed total
# budget divided across sources is a far better trade than a fixed per-source
# truncation, which wasted budget on thin sources and starved rich ones.
EXCERPT_CHAR_BUDGET = 22_000
MAX_SOURCES_PER_CALL = 12
MIN_EXCERPT_CHARS = 600
MAX_EXCERPT_CHARS = 3_500


def _allocate_excerpts(
    results: Sequence[Dict[str, Any]], budget: int = EXCERPT_CHAR_BUDGET
) -> List[Dict[str, Any]]:
    """Split the excerpt budget across sources proportionally to what they have.

    A source with 400 characters of text gets 400; a source with 30,000 gets a
    fair share of the remainder rather than the same 1000 as everything else.
    """
    chosen = list(results)[:MAX_SOURCES_PER_CALL]
    if not chosen:
        return []
    available = [
        len(str(item.get("content", "") or "") or str(item.get("snippet", "") or ""))
        for item in chosen
    ]
    total_available = sum(available) or 1
    out: List[Dict[str, Any]] = []
    for item, have in zip(chosen, available):
        share = int(budget * (have / total_available))
        allowance = max(MIN_EXCERPT_CHARS, min(MAX_EXCERPT_CHARS, share))
        content = str(item.get("content", "") or "")
        out.append({
            "title": str(item.get("title", "") or "")[:200],
            "url": str(item.get("url", "") or ""),
            "snippet": str(item.get("snippet", "") or "")[:400],
            "content": _head_and_tail(content, allowance),
            "sub_question": str(item.get("sub_question", "") or ""),
            "published": str(item.get("published_at", "") or ""),
        })
    return out


def _head_and_tail(text: str, allowance: int) -> str:
    """Keep the opening and the closing of a long document.

    Pure head truncation systematically loses conclusions, results tables and
    "limitations" sections — exactly the passages a criticism or evidence axis
    needs. 70/30 head/tail keeps the framing and the findings.
    """
    text = text or ""
    if len(text) <= allowance:
        return text
    head = int(allowance * 0.7)
    tail = allowance - head
    return f"{text[:head]}\n…\n{text[-tail:]}"


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

def _build_source_index(results: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """canonical URL -> source record, for attribution checks."""
    index: Dict[str, Dict[str, Any]] = {}
    for item in results or ():
        url = str(item.get("url", "") or "")
        key = canonical_url(url)
        if key and key not in index:
            index[key] = item
    return index


def _resolve_source(
    raw_source: str, index: Dict[str, Dict[str, Any]]
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Map a model-written URL onto one of the provided documents.

    Three outcomes, in order of preference:
      exact/canonical match  -> the provided URL (trusted attribution)
      unique host match      -> that host's provided URL, with a note; models
                                routinely drop a path segment or a query string
      no match               -> ("", None); the claim is dropped, because an
                                unattributable claim is not evidence
    """
    raw = str(raw_source or "").strip()
    if not raw:
        return "", None
    key = canonical_url(raw)
    if key in index:
        record = index[key]
        return str(record.get("url", "") or raw), record

    from app.agents.sources import extract_domain

    host = extract_domain(raw)
    if host:
        candidates = [
            record for candidate_key, record in index.items()
            if extract_domain(candidate_key) == host
        ]
        if len(candidates) == 1:
            record = candidates[0]
            logger.debug(
                "[Summarizer] repaired attribution %s -> %s", raw[:70], str(record.get("url"))[:70]
            )
            return str(record.get("url", "") or ""), record
    return "", None


_QUOTE_NORMALIZE_RE = re.compile(r"[^a-z0-9 ]+")


def _quote_supported(quote: str, source_text: str) -> Optional[bool]:
    """Is the quoted fragment actually present in the source?

    None means "no quote offered", which is neutral. A quote that is offered and
    absent is a strong negative signal: the model invented a verbatim fragment.
    Comparison is whitespace/punctuation-insensitive because extraction mangles
    both.
    """
    text = re.sub(r"\s+", " ", str(quote or "")).strip()
    if len(text) < 12:
        return None
    haystack = _QUOTE_NORMALIZE_RE.sub(" ", (source_text or "").lower())
    haystack = re.sub(r"\s+", " ", haystack)
    needle = _QUOTE_NORMALIZE_RE.sub(" ", text.lower())
    needle = re.sub(r"\s+", " ", needle).strip()
    if not needle:
        return None
    return needle in haystack


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

async def summarizer_agent(
    llm: LLMClient,
    query: str,
    search_results: List[Dict[str, str]],
    specialist_role: str = "general",
) -> List[Dict[str, Any]]:
    quality_results = filter_search_results_by_domain(search_results)
    if not quality_results:
        return []

    system_prompt = specialist_system_prompt(specialist_role)
    compact_results = _allocate_excerpts(quality_results)
    source_index = _build_source_index(quality_results)
    text_by_url = {
        str(item.get("url", "")): (
            str(item.get("content", "") or "") + " " + str(item.get("snippet", "") or "")
        )
        for item in quality_results
        if item.get("url")
    }
    meta_by_url = {
        str(item.get("url", "")): item for item in quality_results if item.get("url")
    }

    user_prompt = (
        f"Research query: {query}\n\n"
        f"Sources ({len(compact_results)}):\n{compact_results}\n\n"
        "Extract high-quality claims as JSON in this schema: "
        '{"facts": [{"claim": "...", "source": "https://...", "confidence": 0.0, '
        '"direct_quote": "..."}]}\n'
        "The \"source\" value MUST be one of the url values above, copied exactly. "
        "Prefer the full page content over the snippet when both are present. "
        "Keep every number's unit, period and scope inside the claim text. "
        "Ignore weak, promotional, or opinion-blog sources."
    )

    cache = get_cache(llm.settings)
    key = cache_key(
        "summarize_facts",
        PROMPT_VERSION,
        specialist_role,          # role changes the prompt, so it must key the cache
        query,
        tuple(sorted(item.get("url", "") for item in compact_results)),
    )
    try:
        cached = cache.get(key)
    except Exception as exc:
        logger.warning("[Summarizer] cache read failed: %s", exc, exc_info=exc)
        cached = None

    if cached is not None:
        logger.info("[Summarizer] cache hit (%s)", PROMPT_VERSION)
        facts = cached
    else:
        logger.info("[Summarizer] cache miss (%s)", PROMPT_VERSION)
        try:
            payload = await llm.generate_json(
                system_prompt,
                user_prompt,
                response_model=SummarizerFactsModel,
            )
            facts = payload.get("facts", []) if isinstance(payload, dict) else []
        except Exception as exc:
            logger.warning("[Summarizer] LLM call failed, using heuristic fallback", exc_info=exc)
            facts = []
        # Cache only successful, non-empty extractions: caching the empty list on
        # provider failure poisoned the key for a full TTL, so the summarizer kept
        # "recovering" from cache after the provider was healthy again.
        if facts:
            try:
                cache.set(key, facts, expire=llm.settings.cache_ttl_sec)
            except Exception as exc:
                logger.warning("[Summarizer] cache write failed, continuing uncached: %s", exc, exc_info=exc)

    cleaned: List[Dict[str, Any]] = []
    attribution_drops = 0
    for fact in facts or []:
        if not isinstance(fact, dict):
            continue
        # Model output gets the same fragment hygiene as snippets: echoed
        # truncations must not enter the pool.
        claim = clean_snippet_text(str(fact.get("claim", "")), min_chars=25)
        if not claim:
            continue

        source, record = _resolve_source(fact.get("source", ""), source_index)
        if not source:
            attribution_drops += 1
            continue

        source_score = source_reliability_score(source)
        if source_score < 0.55:
            continue
        if claim_query_overlap(query, claim) < MIN_QUERY_OVERLAP:
            continue

        profile = classify_source(source)
        model_confidence = clamp_confidence(fact.get("confidence", 0.0))
        confidence = (0.70 * model_confidence) + (0.25 * source_score)

        quote = str(fact.get("direct_quote", "") or "").strip()
        quote_state = _quote_supported(quote, text_by_url.get(source, ""))
        if quote_state is True:
            # A verbatim fragment located in the source is the strongest
            # grounding signal available before verification runs.
            confidence += 0.06
        elif quote_state is False:
            confidence -= 0.12
            logger.debug("[Summarizer] unverifiable quote on %s", source[:70])
        if profile.is_primary:
            confidence += 0.03

        meta = meta_by_url.get(source, {})
        numbers = [q.to_dict() for q in extract_numbers(claim, limit=6)]
        cleaned.append({
            "claim": claim,
            "source": source,
            "confidence": clamp_confidence(confidence),
            "agent": specialist_role,
            "sub_question": str(
                (record or {}).get("sub_question", "") or meta.get("sub_question", "") or ""
            ),
            "direct_quote": quote[:240],
            "quote_verified": quote_state,
            "published_at": str(meta.get("published_at", "") or ""),
            "search_type": str(meta.get("search_type", "") or ""),
            "is_primary": profile.is_primary,
            "source_tier": profile.tier,
            "numbers": numbers,
            "extraction": "llm",
        })

    if attribution_drops:
        logger.warning(
            "[Summarizer] dropped %d claim(s) attributed to URLs that were not provided",
            attribution_drops,
        )

    if cleaned:
        return dedupe_semantic_facts(cleaned)

    # ------------------------------------------------------------------
    # Heuristic fallback: the model contributed nothing usable.
    # ------------------------------------------------------------------
    # Works sentence-by-sentence from full fetched content first (far richer
    # than snippets). Candidates are scored against BOTH the top-level query and
    # the source's own sub-question, ranked, and only the best per source kept:
    # taking the first sentences over a bare 0.15 overlap let off-topic
    # passages through as top claims.
    record_fallback("summarizer")
    MIN_FALLBACK_OVERLAP = 0.2
    fallback: List[Dict[str, Any]] = []
    for item in quality_results[:MAX_SOURCES_PER_CALL]:
        raw = (item.get("content", "") or "")[:4000] or item.get("snippet", "")
        if not raw:
            continue
        url = str(item.get("url", "") or "")
        profile = classify_source(url)
        sub_q = str(item.get("sub_question", "") or "")
        scored: List[Tuple[float, str]] = []
        for sent in split_into_sentences(raw, max_sentences=20):
            claim = clean_snippet_text(sent)
            if not claim:
                continue
            relevance = max(
                claim_query_overlap(query, claim),
                claim_query_overlap(sub_q, claim) if sub_q else 0.0,
            )
            if relevance < MIN_FALLBACK_OVERLAP:
                continue
            scored.append((relevance, claim))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        for relevance, claim in scored[:3]:
            fallback.append({
                "claim": claim,
                "source": url,
                # Extractive claims are quoted from the page, so their grounding
                # is the page's authority plus their own topical relevance.
                "confidence": clamp_confidence(
                    (0.30 * max(0.45, relevance)) + (0.70 * profile.authority)
                ),
                "agent": specialist_role,
                "sub_question": sub_q,
                "direct_quote": claim[:240],
                "quote_verified": True,
                "published_at": str(item.get("published_at", "") or ""),
                "search_type": str(item.get("search_type", "") or ""),
                "is_primary": profile.is_primary,
                "source_tier": profile.tier,
                "numbers": [q.to_dict() for q in extract_numbers(claim, limit=6)],
                "extraction": "heuristic",
            })
    return dedupe_semantic_facts(fallback)
