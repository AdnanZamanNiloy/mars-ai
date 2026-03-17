from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Set
from urllib.parse import urlparse


LOW_QUALITY_DOMAINS: Set[str] = {
    "reddit.com",
    "quora.com",
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


def filter_search_results_by_domain(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [item for item in results if is_reliable_source(item.get("url", ""), min_score=0.60)]


def filter_facts_by_domain(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [item for item in facts if is_reliable_source(str(item.get("source", "")), min_score=0.62)]


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
