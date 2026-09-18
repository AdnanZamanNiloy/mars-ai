# MARS-AI Benchmark Results

*Suite: mars-offline-benchmarks v2.0 — generated 2026-09-18T08:44:19Z*

Deterministic, network-free measurement of the intelligence and performance
properties of the upgraded pipeline. All metrics run on hand-labeled
fixtures (see `backend/bench/datasets.py`); every component result is
reproducible with `python bench/run_offline.py` from `backend/`.

## Executive summary

| Component | Headline metrics |
|---|---|
| Verification accuracy | P=1.0000 R=1.0000 F1=1.0000 (16 labeled cases) |
| Hallucination rejection | leak rate 0.0000 (6/6 fabricated claims rejected) |
| Citation support | sentence accuracy 1.0000, numeric grounding 1.0000 |
| Contradiction detection | P=1.0000 R=1.0000 F1=1.0000 across 3 kinds ['numeric', 'polarity', 'temporal'] |
| Confidence calibration | 100% in expected band, monotonic ordering: True |
| Semantic engine | Spearman 0.7094 vs labels, dedup F1 0.7500 @ 0.86 |
| LLM response cache | 66% hit ratio on repeat-heavy workload, 199.39x repeat-pass speedup |
| End-to-end pipeline (mocked LLM) | 3 iterations (intelligent stop), 2 dependency waves, support rate 1.0000 |

Raw numbers: `benchmark_results.json` alongside this file.

## Intelligence detail

### Verification (claim vs source)

8 true positives, 0 false accepts, 0 false rejects, 8 true rejects.
Hard checks: weighted lexical overlap, source authority, unit-aware numeric grounding,
polarity consistency vs the most-similar source sentence, direct-quote location.
Per-claim latency 243.0 µs. Misses: 0.

### Hallucination adversarial set

Fabricated numbers, invented facts, and retraction claims against clean sources.
Rejected 6/6; leak rate 0.0000.

### Contradiction engine v2

Numeric (unit-aware), polarity and temporal detectors over the shared semantic
engine. Kinds detected this run: {'numeric': 2, 'polarity': 3, 'temporal': 1}. Pairwise scan latency
5.6200 ms for 12 labeled pairs.

### Confidence engine v2

Scores: {'strong': 0.815, 'moderate': 0.698, 'weak': 0.213, 'conflicted': 0.74}. Contradictions apply a pool-size-scaled penalty;
citation support and axis coverage blend in when present.

## Performance detail

| Operation | Latency |
|---|---|
| Similarity matrix, 60 facts | 217.9 ms |
| Similarity matrix, 200 facts | 564.6 ms |
| Cross-similarity, 40 sentences x 200 claims | 10.4 ms |
| Contradiction scan, 60 facts | 256.1 ms |
| Dedup pass, 60 facts | 240.1 ms |
| Answer-support check, 40 sentences | 0.2500 ms |
| Pair similarity (single) | 185.8 µs |

Peak-RSS delta across a 300-fact matrix workload: 20648 KiB —
the engine fits comfortably in the 8 GB RAM budget with the whole stack.

## Known limitations (measured, not hidden)

- Embedding-level paraphrases ("doubled over the past decade" vs "nearly
  doubled in the last ten years") score below the dedup threshold: TF-IDF
  cannot map decade↔ten years. Dedup recall on the labeled set is
 0.6000 (precision 1.0000 — it never merges what it should not).
- The semantic engine is deliberately lexical (CPU-light, 8 GB RAM
  constraint): synonym-level paraphrases are partially handled by the
  stemmer + curated synonym map; deeper equivalence needs embeddings.
- Offline benchmarks exercise deterministic components and a scripted
  end-to-end pipeline. Live-model quality (planner/synthesizer wording)
  is measured by `bench/run_live.py` with real provider keys.

