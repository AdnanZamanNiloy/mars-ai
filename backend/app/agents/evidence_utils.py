from __future__ import annotations

import re
from datetime import datetime
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Set
from urllib.parse import urlparse


LOW_QUALITY_DOMAINS: Set[str] = {
    "reddit.com",
    "quora.com",
    "zhihu.com",
    "baidu.com",
    "sohu.com",
    "csdn.net",
    "medium.com",
    "blogspot.com",
    "substack.com",
    "wordpress.com",
    "youtube.com",
    "youtu.be",
    "tiktok.com",
    "pinterest.com",
    "whatfix.com",
}


HIGH_AUTHORITY_DOMAINS: Set[str] = {
    "stanford.edu",
    "plato.stanford.edu",
    "iep.utm.edu",
    "britannica.com",
    "routledge.com",
    "nature.com",
    "science.org",
    "arxiv.org",
    "huggingface.co",
    "paperswithcode.com",
    "github.com",
    "openml.org",
    "mlcommons.org",
    "who.int",
    "oecd.org",
    "worldbank.org",
    "imf.org",
    "un.org",
}


def extract_domain(url: str) -> str:
    parsed = urlparse((url or "").strip())
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def is_high_quality_domain(url: str, blocked_domains: Iterable[str] = LOW_QUALITY_DOMAINS) -> bool:
    domain = extract_domain(url)
    if not domain:
        return False
    for blocked in blocked_domains:
        blocked_value = blocked.lower().strip()
        if domain == blocked_value or domain.endswith(f".{blocked_value}"):
            return False
    return True


def source_reliability_score(url: str) -> float:
    domain = extract_domain(url)
    if not domain:
        return 0.0
    if not is_high_quality_domain(url):
        return 0.0

    if domain in HIGH_AUTHORITY_DOMAINS or any(domain.endswith(f".{d}") for d in HIGH_AUTHORITY_DOMAINS):
        return 0.95
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 0.92
    if domain.endswith(".org"):
        return 0.75
    if domain.endswith(".com"):
        return 0.58
    if domain.endswith(".co"):
        return 0.62
    return 0.55


def is_reliable_source(url: str, min_score: float = 0.62) -> bool:
    return source_reliability_score(url) >= min_score


def normalize_claim_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    cleaned = cleaned.strip("-:;,. ")
    if not cleaned:
        return ""
    if len(cleaned) > 260:
        cleaned = cleaned[:257].rstrip() + "..."
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def filter_search_results_by_domain(
    results: List[Dict[str, str]],
    min_score: float = 0.60,
    fallback_min_score: float = 0.55,
    fallback_limit: int = 8,
) -> List[Dict[str, str]]:
    strong: List[Dict[str, str]] = []
    fallback: List[Dict[str, Any]] = []

    for item in results:
        url = str(item.get("url", "")).strip()
        if not url or not is_high_quality_domain(url):
            continue

        score = source_reliability_score(url)
        if score >= min_score:
            strong.append(item)
            continue
        if score >= fallback_min_score:
            fallback.append({"score": score, "item": item})

    if strong:
        strong_domains = {extract_domain(str(item.get("url", ""))) for item in strong if item.get("url")}
        strong_domains.discard("")
        if len(strong) >= 3 and len(strong_domains) >= 2:
            return strong

        supplemented = list(strong)
        seen_urls = {str(item.get("url", "")).strip() for item in strong if item.get("url")}
        fallback.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        for entry in fallback:
            item = entry.get("item")
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            if not url or url in seen_urls:
                continue
            supplemented.append(item)
            seen_urls.add(url)
            if len(supplemented) >= max(3, min(fallback_limit, 10)):
                break
        return supplemented

    fallback.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return [entry["item"] for entry in fallback[: max(1, fallback_limit)]]


def filter_facts_by_domain(
    facts: List[Dict[str, Any]],
    min_score: float = 0.62,
    fallback_min_score: float = 0.55,
) -> List[Dict[str, Any]]:
    strong: List[Dict[str, Any]] = []
    fallback: List[Dict[str, Any]] = []

    for item in facts:
        source = str(item.get("source", "")).strip()
        claim = normalize_claim_text(str(item.get("claim", "")))
        if not source or not claim or not is_high_quality_domain(source):
            continue

        score = source_reliability_score(source)
        if score >= min_score:
            strong.append(item)
            continue
        if score >= fallback_min_score:
            fallback.append(item)

    if strong:
        strong_domains = {extract_domain(str(item.get("source", ""))) for item in strong if item.get("source")}
        strong_domains.discard("")
        if len(strong) >= 3 and len(strong_domains) >= 2:
            return strong

        supplemented = list(strong)
        seen_sources = {str(item.get("source", "")).strip() for item in strong if item.get("source")}
        fallback.sort(key=lambda x: float(x.get("confidence", 0.0) or 0.0), reverse=True)

        for item in fallback:
            source = str(item.get("source", "")).strip()
            if not source or source in seen_sources:
                continue
            supplemented.append(item)
            seen_sources.add(source)
            if len(supplemented) >= 8:
                break
        return supplemented

    fallback.sort(key=lambda x: float(x.get("confidence", 0.0) or 0.0), reverse=True)
    return fallback[:8]


def _tokenize(text: str) -> Set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token}


def _semantic_similarity(a: str, b: str) -> float:
    a_norm = normalize_claim_text(a).lower()
    b_norm = normalize_claim_text(b).lower()
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0

    seq_ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    tok_a = _tokenize(a_norm)
    tok_b = _tokenize(b_norm)
    jaccard = len(tok_a & tok_b) / len(tok_a | tok_b) if (tok_a or tok_b) else 0.0
    return 0.6 * seq_ratio + 0.4 * jaccard


def dedupe_semantic_facts(facts: List[Dict[str, Any]], threshold: float = 0.86) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []

    for item in facts:
        claim = normalize_claim_text(str(item.get("claim", "")))
        source = str(item.get("source", "")).strip()
        try:
            confidence = float(item.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if not claim or not source:
            continue

        candidate = {"claim": claim, "source": source, "confidence": max(0.0, min(1.0, confidence)),
                     "agent": str(item.get("agent", "") or "")}

        merge_index = -1
        for idx, kept in enumerate(deduped):
            score = _semantic_similarity(claim, str(kept.get("claim", "")))
            if score >= threshold:
                merge_index = idx
                break

        if merge_index == -1:
            deduped.append(candidate)
            continue

        existing_conf = float(deduped[merge_index].get("confidence", 0.0) or 0.0)
        if confidence > existing_conf:
            deduped[merge_index] = candidate

    return deduped


CITATION_RE = re.compile(r"\[(\d+)\]")
LEGEND_RE = re.compile(r"^\[(\d+)\]\s+\S.*?—\s*(\S+)\s*$")
SUPPORT_THRESHOLD = 0.30


def verify_answer_support(
    answer: str,
    facts: List[Dict[str, Any]],
    threshold: float = SUPPORT_THRESHOLD,
) -> Dict[str, Any]:
    """Post-synthesis verification (report-contract pattern): every cited
    sentence must overlap verified evidence from its cited source.

    The legend is parsed back out of the answer itself, so numbering can
    never drift from what was actually emitted. Sentences without markers
    are counted as uncited (not failed) — only cited-but-unsupported
    sentences count against the rate.
    """
    body, _, legend_block = (answer or "").partition("\nSources:")
    legend_urls: Dict[int, str] = {}
    for line in legend_block.splitlines():
        match = LEGEND_RE.match(line.strip())
        if match:
            try:
                legend_urls[int(match.group(1))] = match.group(2)
            except (TypeError, ValueError):
                continue

    verified_by_url: Dict[str, List[str]] = {}
    for fact in facts or []:
        if not fact.get("verified"):
            continue
        url = str(fact.get("source", "") or "")
        claim = str(fact.get("claim", "") or "")
        if url and claim:
            verified_by_url.setdefault(url, []).append(claim)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    cited = supported = uncited = 0
    unsupported: List[str] = []
    for sentence in sentences:
        numbers = [int(n) for n in CITATION_RE.findall(sentence)]
        if not numbers:
            if len(sentence.split()) >= 8:
                uncited += 1
            continue
        cited += 1
        hit = False
        for n in numbers:
            url = legend_urls.get(n, "")
            for claim in verified_by_url.get(url, []):
                if _semantic_similarity(sentence, claim) >= threshold:
                    hit = True
                    break
            if hit:
                break
        if hit:
            supported += 1
        else:
            unsupported.append(sentence[:160])

    return {
        "sentences": len(sentences),
        "cited": cited,
        "supported": supported,
        "uncited": uncited,
        "unsupported": unsupported,
        # None (not 0.0) when nothing was cited — unmeasured, not failed.
        "rate": (supported / cited) if cited else None,
    }


# Leading date stamps search engines prepend to snippets ("Mar 17, 2026 ·",
# "2 days ago ·"). They leak into fallback answers as garbage prefixes.
DATE_STAMP_RE = re.compile(
    r"^(?:[A-Z][a-z]{2,8}\s+\d{1,2},\s+\d{4}|\d+\s+(?:day|hour|minute|second)s?\s+ago)"
    r"\s*[·\-–|]\s*",
    re.IGNORECASE,
)

# UI cruft glued into snippets ("Learn more", often truncated to "Learn mor"
# by length cuts). Stripped anywhere they occur, not just at edges.
LINK_TEXT_RE = re.compile(r"\b(?:Learn|Read|Show|See|Click|Continue)\s+mo(?:r(?:e)?)?\b\.?", re.IGNORECASE)

MIN_CLEAN_CLAIM_CHARS = 50


def looks_truncated(text: str) -> bool:
    """True when a claim ends in a probable mid-word cut ("...from a large
    dat"): no sentence-ending punctuation anywhere and a stub tail token.
    Conservative by design — only flags the clear shape, never judges
    content. (Rare casualty: legit ends like "...the US".)"""
    stripped = re.sub(r"\s+", " ", (text or "")).strip()
    if not stripped:
        return True
    if re.search(r"[.!?](?=\s|$)", stripped):
        return False
    return len(stripped.rsplit(None, 1)[-1]) <= 3


def clean_snippet_text(snippet: str, max_chars: int = 300, min_chars: int = MIN_CLEAN_CLAIM_CHARS) -> str:
    """Turn a raw search snippet into a presentable claim sentence.

    Strips engine date stamps, cuts to the last complete sentence within
    max_chars, and rejects stumps shorter than min_chars. Returns ""
    when nothing salvageable remains.
    """
    text = re.sub(r"\s+", " ", (snippet or "")).strip()
    text = DATE_STAMP_RE.sub("", text).strip()
    text = LINK_TEXT_RE.sub("", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < min_chars:
        return ""
    # Trim to the last complete sentence; fall back to a comma break so a
    # mid-sentence cut ("...and effica") never survives as a claim.
    working = text[:max_chars]
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", working)]
    if ends:
        text = working[: ends[-1]].strip()
    else:
        alt = [m.end() for m in re.finditer(r"[,;:](?=\s)", working)]
        if alt:
            text = (working[: alt[-1]].rstrip(",;:") + ".").strip()
        elif len(working.rsplit(None, 1)[-1]) <= 3:
            # No sentence end and a stub tail ("...from a large dat"):
            # a mid-word cut. (Rare casualty: legit ends like "...the US".)
            return ""
    text = text.strip()
    if len(text) < min_chars:
        return ""
    return text


def claim_query_overlap(query: str, claim: str) -> float:
    """Word overlap between the research query and a claim (0-1). Guards
    the evidence pool against off-topic drift (crypto tips in a transfer
    learning run): zero shared vocabulary means unrelated."""
    q_words = _tokenize(query or "")
    c_words = _tokenize(claim or "")
    if not q_words:
        return 0.0
    return len(q_words & c_words) / len(q_words)


MIN_QUERY_OVERLAP = 0.15


def select_diverse(
    claims: List[Dict[str, Any]], k: int = 3, max_similarity: float = 0.40
) -> List[Dict[str, Any]]:
    """Greedy maximal-marginal-relevance pick: highest confidence first,
    then each next claim must add novelty vs everything already picked.
    Stops fallback answers reading the same definition three times.

    Novelty uses the overlap coefficient (|A∩B|/min(|A|,|B|)), not raw
    similarity: it is robust to claim length, where blended similarity
    dilutes (long paraphrases scored 0.36, cross-sense pairs ~0.14)."""
    ranked = sorted(claims or [], key=lambda f: float(f.get("confidence", 0.0) or 0.0), reverse=True)
    selected: List[Dict[str, Any]] = []
    for candidate in ranked:
        text = str(candidate.get("claim", "") or "")
        if not text:
            continue
        candidate_tokens = _tokenize(text)
        if not candidate_tokens:
            continue
        novel = True
        for kept in selected:
            kept_tokens = _tokenize(str(kept.get("claim", "") or ""))
            if not kept_tokens:
                continue
            overlap = len(candidate_tokens & kept_tokens) / min(len(candidate_tokens), len(kept_tokens))
            if overlap >= max_similarity:
                novel = False
                break
        if novel:
            selected.append(candidate)
        if len(selected) >= k:
            break
    return selected


def parse_published_date(value: str) -> str:
    """Best-effort parse of provider/fetch date strings to YYYY-MM-DD.

    Accepts ISO 8601 (with Z), HTTP Last-Modified, and common short forms.
    Returns "" when unparseable — unknown stays unknown, never guessed.
    """
    text = (value or "").strip()
    if not text:
        return ""
    candidates = [text, text.replace("Z", "+00:00")]
    for candidate in candidates:
        try:
            return datetime.fromisoformat(candidate).date().isoformat()
        except (ValueError, TypeError):
            pass
    try:
        return parsedate_to_datetime(text).date().isoformat()
    except (TypeError, ValueError):
        pass
    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%d %b %Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except (TypeError, ValueError):
            continue
    return ""
