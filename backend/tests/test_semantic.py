"""Semantic engine: hybrid TF-IDF + lexical similarity."""

import time

import numpy as np

from app.core.semantic import (
    cross_similarity,
    pair_similarity,
    rank_by_similarity,
    similarity,
    similarity_matrix,
    top_match,
)

IDENTICAL = ("Solar capacity grew 40% in 2024", "Solar capacity grew 40% in 2024")
NEAR_DUP = (
    "Global solar capacity grew by 40% in 2024",
    "Global solar capacity grew 40% in 2024",
)
PARAPHRASE = (
    "Wind and solar capacity roughly doubled over the past decade",
    "Capacity from wind and solar nearly doubled in the last ten years",
)
UNRELATED = (
    "The central bank raised interest rates by 75 basis points",
    "Pigeons are birds commonly found in cities",
)
NEGATION_PAIR = (
    "The model outperforms the baseline",
    "The model does not outperform the baseline",
)


def test_identical_scores_one():
    assert pair_similarity(*IDENTICAL) == 1.0


def test_near_duplicate_scores_high():
    assert pair_similarity(*NEAR_DUP) >= 0.85


def test_paraphrase_scores_moderate():
    # The TF-IDF term must lift paraphrases above the unrelated floor.
    assert pair_similarity(*PARAPHRASE) >= 0.30


def test_unrelated_scores_low():
    assert pair_similarity(*UNRELATED) <= 0.25


def test_negation_stays_similar():
    # Contradiction detection depends on negated pairs remaining in the
    # 0.50-0.86 similarity band: polarity flips, topical content identical.
    score = pair_similarity(*NEGATION_PAIR)
    assert 0.5 <= score


def test_empty_inputs():
    assert pair_similarity("", "something") == 0.0
    assert pair_similarity(None, "x") == 0.0  # type: ignore[arg-type]


def test_matrix_symmetric_and_diagonal_one():
    texts = [IDENTICAL[0], NEAR_DUP[1], PARAPHRASE[0], UNRELATED[1]]
    m = similarity_matrix(texts)
    assert m.shape == (4, 4)
    assert np.allclose(m, m.T, atol=1e-4)
    assert all(m[i, i] == 1.0 for i in range(4))


def test_matrix_consistent_with_pair_scoring():
    texts = [NEAR_DUP[0], NEAR_DUP[1], UNRELATED[0], UNRELATED[1]]
    m = similarity_matrix(texts)
    assert abs(float(m[0, 1]) - pair_similarity(*NEAR_DUP)) < 0.02
    assert float(m[2, 3]) <= 0.25


def test_cross_similarity_consistent():
    queries = [NEAR_DUP[0], UNRELATED[0]]
    candidates = [NEAR_DUP[1], UNRELATED[1], PARAPHRASE[0]]
    m = cross_similarity(queries, candidates)
    assert m.shape == (2, 3)
    assert float(m[0, 0]) >= 0.85
    assert float(m[1, 1]) <= 0.25
    assert float(m[0, 2]) >= 0.25  # near-dup vs paraphrase shares topic


def test_cross_similarity_matches_pair_within_tolerance():
    a, b = PARAPHRASE
    m = cross_similarity([a], [b])
    assert abs(float(m[0, 0]) - pair_similarity(a, b)) < 0.02


def test_top_match_finds_best():
    query = NEAR_DUP[0]
    candidates = [UNRELATED[1], NEAR_DUP[1], PARAPHRASE[0]]
    idx, score = top_match(query, candidates)
    assert idx == 1
    assert score >= 0.85


def test_rank_preserves_order():
    scores = rank_by_similarity(NEAR_DUP[0], [UNRELATED[1], NEAR_DUP[1]])
    assert len(scores) == 2
    assert scores[1] > scores[0]


def test_similarity_alias():
    assert similarity(NEAR_DUP[0], NEAR_DUP[1]) == pair_similarity(*NEAR_DUP)


def test_matrix_performance_300_texts():
    rng = np.random.default_rng(7)
    vocab = ["solar", "wind", "capacity", "grew", "percent", "gw", "installed",
             "cost", "battery", "storage", "grid", "deployment", "2024", "china"]
    texts = []
    for _ in range(300):
        k = rng.integers(6, 14)
        words = [vocab[int(rng.integers(0, len(vocab)))] for _ in range(int(k))]
        texts.append(" ".join(words))
    start = time.perf_counter()
    m = similarity_matrix(texts)
    elapsed = time.perf_counter() - start
    assert m.shape == (300, 300)
    # Must stay well under the 2s budget on CPU-only hardware; historically
    # the SequenceMatcher path needed seconds for this size.
    assert elapsed < 2.0, f"matrix took {elapsed:.2f}s"


def test_numbers_weighted_higher_than_stopwords():
    # A shared number should outweigh shared function words.
    with_num = pair_similarity("Costs rose 4.2 billion dollars", "Revenues hit 4.2 billion dollars")
    without = pair_similarity("Costs rose a lot of money", "Revenues hit a lot of money")
    assert with_num > without
