from __future__ import annotations

import re
from difflib import SequenceMatcher
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

        candidate = {"claim": claim, "source": source, "confidence": max(0.0, min(1.0, confidence))}

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
