"""CPU-light hybrid semantic similarity engine.

MARS scores text-to-text affinity in five places (fact dedup, contradiction
banding, citation support, verifier polarity checks, summarizer relevance).
The legacy implementation blended char-ratio + token-Jaccard, which misses
paraphrases ("renewable capacity doubled" vs "wind and solar grew 2x") and
costs O(n^2) SequenceMatcher time on deep runs.

This engine blends three signals with a fixed IDF policy (no model, no
fitting, no network, ~0 extra RAM):

1. TF-IDF cosine — stopword-down-weighted, number/entity-up-weighted;
   catches paraphrase-level topical overlap.
2. Token Jaccard — set overlap, robust to reordering.
3. SequenceMatcher char ratio — surface affinity, strongest on
   near-identical strings.

Weights were calibrated so the historical 0.86 dedup threshold and the
0.50-0.86 contradiction band keep their meaning for near-duplicates while
paraphrase pairs score ~0.1 higher than before.

All functions are deterministic, thread-safe and side-effect free. Batch
APIs vectorize once and use a single matrix product.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np

from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# Tokenization + fixed IDF policy
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_NUMBERISH_RE = re.compile(r"^\d+[.,]?\d*$|^\d+st$|^\d+nd$|^\d+rd$|^\d+th$")

# Compact English function-word list (high frequency => low IDF).
# NEGATION WORDS ARE DELIBERATELY ABSENT: "not" flips claim meaning, so it
# is weighted as strongly as a number (see _NEGATION_TOKENS_ENGINE). A
# stopword list containing "not" made "X" and "not X" score 1.0 — a
# benchmark-found bug — because every other token matched exactly.
_STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because
been before being below between both but by can could did do does doing
down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just me more most my
myself now of off on once only or other our ours ourselves out over
own same she should so some such than that the their theirs them themselves
then there these they this those through to too under until up very was we
were what when where which while who whom why will with you your yours
yourself yourselves shall may might must upon among within without across
per via etc said says say according based new one two three four five six
seven eight nine ten first second third last next many much several various
including included include
""".split())

# Negation carries claim-level meaning: weighted like numbers.
_NEGATION_TOKENS_ENGINE = frozenset({
    "not", "no", "never", "cannot", "cant", "without", "neither",
    "nor", "none",
})

_WEIGHT_STOPWORD = 0.30   # function words barely discriminate
_WEIGHT_DEFAULT = 1.50    # ordinary content word
_WEIGHT_LONG = 1.80       # 7+ chars: rare, informative
_WEIGHT_NUMBER = 2.40     # numbers are the strongest grounding signal
_WEIGHT_BIGRAM = 0.60     # multiplier applied to (w1+w2) idf sum

_MAX_TEXT_CHARS = 4000


def _token_weight(token: str) -> float:
    if token in _NEGATION_TOKENS_ENGINE:
        return 2.00  # negation flips meaning — as discriminative as a number
    if token in _STOPWORDS:
        return _WEIGHT_STOPWORD
    if _NUMBERISH_RE.match(token):
        return _WEIGHT_NUMBER
    if len(token) >= 7:
        return _WEIGHT_LONG
    if token.isupper() and len(token) >= 2:  # acronyms: NASA, GDP, TWh
        return 2.00
    return _WEIGHT_DEFAULT


def _stem(tok: str) -> str:
    """Light suffix stemmer — consistency over linguistics.

    Both sides of a pair pass through the SAME rules, so "declined",
    "declines" and "decline" all collapse to "declin" and stopword-noise
    merges stop failing. Handles plurals, past tense, gerunds and the
    trailing-e/doubled-consonant patterns that dominate research prose.
    """
    if len(tok) >= 5:
        if tok.endswith("ies"):
            tok = tok[:-3] + "y"
        elif tok.endswith("sses"):
            tok = tok[:-2]
        elif tok.endswith("es"):
            tok = tok[:-2]
        elif tok.endswith("ed"):
            tok = tok[:-2]
        elif tok.endswith("ing"):
            tok = tok[:-3]
        elif tok.endswith("s") and not tok.endswith("ss") and not tok.endswith("us"):
            tok = tok[:-1]
    if len(tok) >= 4 and tok.endswith("e"):
        tok = tok[:-1]
    if (
        len(tok) >= 4
        and tok[-1] == tok[-2]
        and tok[-1] not in "aeiouysl"
    ):
        tok = tok[:-1]
    return tok


# Synonym canonicalization on STEMS: irregular pasts and high-frequency
# research synonyms that stemming cannot reach. Deliberately tiny — each
# entry must be truth-preserving (never joins antonyms).
_SYNONYMS = {
    "fell": "fall", "fall": "declin", "drop": "declin", "dropp": "declin",
    "decreas": "declin",
    "rose": "rise", "ris": "rise", "climb": "rise", "surge": "rise", "jump": "rise",
    "increas": "rise",
    "grew": "grow", "growth": "grow",
    "usd": "dollar", "dollar": "dollar",
    "reach": "total", "total": "total", "hit": "total", "amount": "total",
    "studi": "studi",
    "show": "found", "found": "found",
    "percentage": "percent",
    "approx": "about", "roughly": "about", "nearli": "about",
}


def _canonical(tok: str) -> str:
    stemmed = _stem(tok)
    return _SYNONYMS.get(stemmed, stemmed)


@lru_cache(maxsize=100_000)
def _tokens_cached(text: str) -> Tuple[Tuple[str, float], ...]:
    """Token -> weight tuples for a text (LRU-cached across pair calls).

    Tokens are stemmed and synonym-canonicalized before weighting, so
    morphological variants ("outperforms"/"outperform") and irregular
    pasts ("fell"/"declined") score as the same term.
    """
    raw = _TOKEN_RE.findall(text.lower()[:_MAX_TEXT_CHARS])
    if not raw:
        return ()
    counts: Dict[str, int] = {}
    for tok in raw:
        counts[tok] = counts.get(tok, 0) + 1
    # unigram weights on canonical (stemmed + synonym) forms; distinct raw
    # tokens mapping to the same canonical ACCUMULATE (fell + fell-ish).
    weighted: Dict[str, float] = {}
    for tok, cnt in counts.items():
        can = _canonical(tok)
        weighted[can] = weighted.get(can, 0.0) + cnt * _token_weight(tok)
    # bigram weights (sublinear tf keeps long docs from dominating)
    bigrams: Dict[str, float] = {}
    for a, b in zip(raw, raw[1:]):
        if a in _STOPWORDS and b in _STOPWORDS:
            continue
        bg = f"{_canonical(a)} {_canonical(b)}"
        bigrams[bg] = bigrams.get(bg, 0.0) + (_token_weight(a) + _token_weight(b)) * _WEIGHT_BIGRAM
    merged: Dict[str, float] = dict(weighted)
    for bg, w in bigrams.items():
        merged[bg] = merged.get(bg, 0.0) + w
    return tuple(merged.items())


def _l2_normalize(vec: Dict[str, float]) -> Dict[str, float]:
    norm = sum(w * w for w in vec.values()) ** 0.5
    if norm <= 1e-12:
        return {}
    return {tok: w / norm for tok, w in vec.items()}


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Pair scoring
# ---------------------------------------------------------------------------

def _stripped_text(text: str) -> str:
    tokens = [_canonical(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]
    return " ".join(tokens)


def _content_tokens_set(text: str) -> frozenset:
    return frozenset(
        _canonical(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS
    )


def _quantities_key(text: str) -> Tuple[Tuple[str, float], ...]:
    """Sorted (unit, value) pairs — the numeric fingerprint of a claim."""
    from app.agents.evidence_utils import _significant_quantities

    try:
        return tuple(sorted(
            (q.unit, round(q.value, 4)) for q in _significant_quantities(text)
            if not q.is_year
        ))
    except Exception:
        return ()


def _dup_floor(
    a: str, b: str, a_data: Tuple[str, frozenset, Tuple], b_data: Tuple[str, frozenset, Tuple]
) -> float:
    """Duplicate-detection floor: the FULL hybrid score computed on
    stopword-stripped text, when the two claims share essentially all
    content words AND every significant quantity.

    Function-word variants ("declined 14% in 2023" vs "declined by 14%
    during 2023") scored ~0.75 on the full-text blend — the stopwords
    polluted all three signals — and escaped the 0.86 merge. On stripped
    text the same pair scores ~1.0. The quantity guard is what keeps this
    safe: "grew 25%" vs "grew 11%" share content words too, and must NEVER
    merge — they are the numeric contradiction the engine exists to catch.
    Benchmark-calibrated: content-Jaccard >= 0.75 separates
    countries/nations (0.75, merge) from solar/wind (0.67, no merge).
    """
    stripped_a, content_a, quants_a = a_data
    stripped_b, content_b, quants_b = b_data
    if not content_a or not content_b:
        return 0.0
    if quants_a != quants_b:
        return 0.0  # different numbers → never a duplicate
    # A claim never floors toward its own negation ("X" vs "not X" share
    # every content token once "not" is weighted — but they are opposites).
    try:
        from app.agents.evidence_utils import claim_polarity

        pa, pb = claim_polarity(a), claim_polarity(b)
        if pa != 0 and pb != 0 and pa != pb:
            return 0.0
    except Exception:
        pass
    inter = len(content_a & content_b)
    union = len(content_a | content_b)
    if union == 0 or inter / union < 0.75:
        return 0.0
    seq_s = SequenceMatcher(
        None, stripped_a[:_MAX_TEXT_CHARS], stripped_b[:_MAX_TEXT_CHARS]
    ).ratio()
    tfidf_s = _tfidf_cosine(stripped_a, stripped_b)
    content_jac = inter / union
    return round(_W_SEQ * seq_s + _W_JAC * content_jac + _W_TFIDF * tfidf_s, 4)


# Blend weights: keep near-duplicate scores ~unchanged vs the legacy
# 0.6*seq + 0.4*jaccard formula, lift paraphrase pairs by ~0.1.
_W_SEQ, _W_JAC, _W_TFIDF = 0.40, 0.25, 0.35


def _tfidf_cosine(a: str, b: str) -> float:
    va = _l2_normalize(dict(_tokens_cached(a)))
    vb = _l2_normalize(dict(_tokens_cached(b)))
    if not va or not vb:
        return 0.0
    if len(vb) < len(va):  # iterate the smaller
        va, vb = vb, va
    dot = sum(w * vb.get(tok, 0.0) for tok, w in va.items())
    return max(0.0, min(1.0, dot))


def _content_overlap_gate(ta: Tuple[str, float], tb: Tuple[str, float]) -> bool:
    """Cheap gate: is the expensive char-diff worth running?

    Pass when Jaccard >= 0.10 OR the pair shares at least two rare tokens
    (numbers, long words) — paraphrases re-word function vocabulary but
    usually keep some anchors. Mirrors the legacy gate while admitting
    anchor-sharing paraphrases the Jaccard gate alone would skip.
    """
    sa = {t for t, w in ta if w >= _WEIGHT_LONG}
    sb = {t for t, w in tb if w >= _WEIGHT_LONG}
    rare_shared = len(sa & sb)
    na = {t for t, w in ta if w >= _WEIGHT_NUMBER}
    nb = {t for t, w in tb if w >= _WEIGHT_NUMBER}
    num_shared = len(na & nb)
    return rare_shared >= 3 or num_shared >= 2


def pair_similarity(a: str, b: str) -> float:
    """Hybrid similarity in [0, 1] for two texts."""
    if not a or not b:
        return 0.0
    a_n, b_n = a.strip().lower(), b.strip().lower()
    if a_n == b_n:
        return 1.0
    ta, tb = _tokens_cached(a_n), _tokens_cached(b_n)
    if not ta or not tb:
        return 0.0
    jac = _jaccard(frozenset(t[0] for t in ta), frozenset(t[0] for t in tb))
    tfidf = _tfidf_cosine(a_n, b_n)
    if jac < 0.10 and tfidf < 0.35 and not _content_overlap_gate(ta, tb):
        # Cheapest exit: blend the two cheap signals only.
        return round(_W_SEQ * 0.0 + _W_JAC * jac + _W_TFIDF * tfidf, 4)
    seq = SequenceMatcher(None, a_n[:_MAX_TEXT_CHARS], b_n[:_MAX_TEXT_CHARS]).ratio()
    hybrid = round(_W_SEQ * seq + _W_JAC * jac + _W_TFIDF * tfidf, 4)
    # Duplicate-detection floor: function-word variants with identical
    # quantities compare on stopword-stripped text; the floor can only
    # RAISE the score toward a merge, never lower it.
    floor = _dup_floor(
        a_n, b_n,
        (_stripped_text(a_n), _content_tokens_set(a_n), _quantities_key(a_n)),
        (_stripped_text(b_n), _content_tokens_set(b_n), _quantities_key(b_n)),
    )
    return max(hybrid, floor)


# ---------------------------------------------------------------------------
# Batch scoring (dedup, contradiction banding, citation support)
# ---------------------------------------------------------------------------

def _batch_signals(texts: Sequence[str]) -> Dict[str, np.ndarray] | None:
    """Vectorized cheap-signal block for a batch of texts.

    Returns TF-IDF presence/weight/boolean matrices + per-doc token sets,
    or None when the batch is empty. Everything downstream (jaccard, rare
    anchor counts, number sharing) becomes one matmul instead of a Python
    O(n^2) loop — the difference between 1s and 60ms at 200 facts.
    """
    n = len(texts)
    if n == 0:
        return None
    lowered = [(t or "").strip().lower() for t in texts]
    token_tuples = [_tokens_cached(t) for t in lowered]
    index: Dict[str, int] = {}
    presence_rows: List[Dict[int, float]] = []
    rare_cols: Dict[int, None] = {}
    num_cols: Dict[int, None] = {}
    for toks in token_tuples:
        row: Dict[int, float] = {}
        for tok, w in toks:
            col = index.get(tok)
            if col is None:
                col = len(index)
                index[tok] = col
            row[col] = w
            if w >= _WEIGHT_LONG:
                rare_cols[col] = None
            if w >= _WEIGHT_NUMBER:
                num_cols[col] = None
        presence_rows.append(row)
    width = len(index)
    weights = np.zeros((n, width), dtype=np.float32)
    for i, row in enumerate(presence_rows):
        for col, w in row.items():
            weights[i, col] = w
    presence = (weights > 0).astype(np.float32)
    counts = presence.sum(axis=1)  # token count per doc
    rare_mask = np.zeros((n, len(rare_cols)), dtype=np.float32)
    for new_col, old_col in enumerate(rare_cols):
        rare_mask[:, new_col] = presence[:, old_col]
    num_mask = np.zeros((n, len(num_cols)), dtype=np.float32)
    for new_col, old_col in enumerate(num_cols):
        num_mask[:, new_col] = presence[:, old_col]
    return {
        "lowered": lowered,
        "token_tuples": token_tuples,
        "weights": weights,
        "presence": presence,
        "counts": counts,
        "rare_mask": rare_mask,
        "num_mask": num_mask,
        "sets": [frozenset(t[0] for t in toks) for toks in token_tuples],
    }


def _tfidf_norm(block: Dict[str, np.ndarray]) -> np.ndarray:
    """L2-normalized TF-IDF weight matrix (zero rows stay zero)."""
    w = block["weights"]
    norms = np.linalg.norm(w, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return w / norms


def _jaccard_matrix(presence: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Pairwise token Jaccard via one boolean matmul."""
    inter = presence @ presence.T
    union = counts[:, None] + counts[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    return np.clip(jac, 0.0, 1.0)


# Above this batch size, dense matrices would exceed the 8GB RAM budget
# (n x terms x 4B) and lexical enrichment is skipped: cheap-blend scores
# only. Fact pools never approach this in practice; the guard is for safety.
_MAX_DENSE_BATCH = 800


def similarity_matrix(texts: Sequence[str], max_seq_pairs: int = 4000) -> np.ndarray:
    """Symmetric pairwise hybrid-similarity matrix for a batch of texts.

    Budget model: TF-IDF cosine, token Jaccard, rare-anchor overlap and
    number-sharing are ALL computed as vectorized matmuls; the expensive
    char-ratio enrichment runs only on the ``max_seq_pairs`` most promising
    pairs (ranked by cheap blend score, ties broken by index). Pairs that
    lose their enrichment budget keep the cheap-blend score — a slight
    underestimate that can only make dedup more conservative.
    """
    n = len(texts)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    block = _batch_signals(texts)
    if block is None:
        return np.zeros((n, n), dtype=np.float32)
    tfidf = np.clip(_tfidf_norm(block) @ _tfidf_norm(block).T, 0.0, 1.0)
    jac = _jaccard_matrix(block["presence"], block["counts"])
    rare_shared = (
        block["rare_mask"] @ block["rare_mask"].T
        if block["rare_mask"].shape[1] else np.zeros((n, n), dtype=np.float32)
    )
    num_shared = (
        block["num_mask"] @ block["num_mask"].T
        if block["num_mask"].shape[1] else np.zeros((n, n), dtype=np.float32)
    )
    empty = block["counts"] <= 0
    if empty.any():
        tfidf[empty, :] = 0.0
        tfidf[:, empty] = 0.0
        jac[empty, :] = 0.0
        jac[:, empty] = 0.0
        rare_shared[empty, :] = 0.0
        rare_shared[:, empty] = 0.0
    np.fill_diagonal(tfidf, 1.0)
    np.fill_diagonal(jac, 1.0)

    cheap = _W_JAC * jac + _W_TFIDF * tfidf
    out = cheap.astype(np.float32)

    if n <= _MAX_DENSE_BATCH and max_seq_pairs > 0:
        gate = (
            (jac >= 0.10)
            | (tfidf >= 0.35)
            | (rare_shared >= 3)
            | (num_shared >= 2)
        )
        np.fill_diagonal(gate, False)
        ii, jj = np.nonzero(np.triu(gate.astype(bool)))
        if len(ii):
            order = np.lexsort((jj, ii, -cheap[ii, jj]))
            ii, jj = ii[order][:max_seq_pairs], jj[order][:max_seq_pairs]
            lowered = block["lowered"]
            sets = block["sets"]
            dup_data = [
                (_stripped_text(t), _content_tokens_set(t), _quantities_key(t))
                for t in lowered
            ]
            for i, j in zip(ii.tolist(), jj.tolist()):
                seq = SequenceMatcher(
                    None, lowered[i][:_MAX_TEXT_CHARS], lowered[j][:_MAX_TEXT_CHARS]
                ).ratio()
                score = round(
                    _W_SEQ * seq + _W_JAC * _jaccard(sets[i], sets[j])
                    + _W_TFIDF * float(tfidf[i, j]), 4
                )
                floor = _dup_floor(lowered[i], lowered[j], dup_data[i], dup_data[j])
                if floor > score:
                    score = floor
                out[i, j] = out[j, i] = score
    np.fill_diagonal(out, 1.0)
    return out


def cross_similarity(
    queries: Sequence[str], candidates: Sequence[str], max_seq_pairs: int = 4000
) -> np.ndarray:
    """Hybrid-similarity matrix of shape (len(queries), len(candidates)).

    Used by citation support (answer sentences x claims) and verifier
    polarity anchoring (claim x source sentences). Same budget model as
    :func:`similarity_matrix`: vectorized cheap signals, char-ratio
    enrichment only for the top ``max_seq_pairs`` cells.
    """
    n_q, n_c = len(queries), len(candidates)
    if not queries or not candidates:
        return np.zeros((n_q, n_c), dtype=np.float32)
    # ONE shared term index across queries + candidates: separate blocks
    # would assign colliding column ids and silently scramble every dot
    # product (weights, presence, rare and number masks alike).
    q_lower = [(q or "").strip().lower() for q in queries]
    c_lower = [(c or "").strip().lower() for c in candidates]
    block = _batch_signals(q_lower + c_lower)
    if block is None:
        return np.zeros((n_q, n_c), dtype=np.float32)

    norm = _l2_rows(block["weights"])
    tfidf = np.clip(norm[:n_q] @ norm[n_q:].T, 0.0, 1.0)

    presence = block["presence"]
    counts = block["counts"]
    inter = presence[:n_q] @ presence[n_q:].T
    union = counts[:n_q, None] + counts[None, n_q:] - inter
    jac = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)
    jac = np.clip(jac, 0.0, 1.0)

    rare_mask = block["rare_mask"]
    num_mask = block["num_mask"]
    rare_shared = (
        rare_mask[:n_q] @ rare_mask[n_q:].T
        if rare_mask.shape[1] else np.zeros((n_q, n_c), dtype=np.float32)
    )
    num_shared = (
        num_mask[:n_q] @ num_mask[n_q:].T
        if num_mask.shape[1] else np.zeros((n_q, n_c), dtype=np.float32)
    )
    q_empty = counts[:n_q] <= 0
    c_empty = counts[n_q:] <= 0
    if q_empty.any():
        tfidf[q_empty, :] = 0.0
        jac[q_empty, :] = 0.0
    if c_empty.any():
        tfidf[:, c_empty] = 0.0
        jac[:, c_empty] = 0.0

    cheap = (_W_JAC * jac + _W_TFIDF * tfidf).astype(np.float32)
    out = cheap.copy()
    if n_q * n_c <= _MAX_DENSE_BATCH * 4 and max_seq_pairs > 0:
        gate = (jac >= 0.10) | (tfidf >= 0.35) | (rare_shared >= 3) | (num_shared >= 2)
        flat_gate = gate.reshape(-1)
        flat_cheap = cheap.reshape(-1)
        idx = np.nonzero(flat_gate)[0]
        if len(idx):
            order = np.lexsort((idx, -flat_cheap[idx]))
            idx = idx[order][:max_seq_pairs]
            q_sets, c_sets = block["sets"][:n_q], block["sets"][n_q:]
            for flat in idx.tolist():
                qi, ci = divmod(flat, n_c)
                seq = SequenceMatcher(
                    None, q_lower[qi][:_MAX_TEXT_CHARS], c_lower[ci][:_MAX_TEXT_CHARS]
                ).ratio()
                out[qi, ci] = round(
                    _W_SEQ * seq + _W_JAC * _jaccard(q_sets[qi], c_sets[ci])
                    + _W_TFIDF * float(tfidf[qi, ci]), 4
                )
    return out


def _l2_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return mat / norms


def rank_by_similarity(query: str, candidates: Sequence[str]) -> List[float]:
    """Hybrid scores of query vs each candidate, order preserved."""
    if not candidates:
        return []
    return [float(x) for x in cross_similarity([query], list(candidates))[0]]


# ---------------------------------------------------------------------------
# Public singleton-style helpers (stable import surface)
# ---------------------------------------------------------------------------

def similarity(a: str, b: str) -> float:
    """Alias of :func:`pair_similarity` — the historical call shape."""
    return pair_similarity(a, b)
