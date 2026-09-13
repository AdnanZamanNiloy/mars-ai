# Changelog

## v2.0 — Production upgrade

The upgrade from the v1 vision implementation: every dead-ended component
wired live, four new subsystems, and a benchmark suite that found (and
fixed) six real correctness bugs before shipping.

### Wired: previously-built-but-inert components

- **Budget governor** (`agents/budget.py`) — was complete but had zero call
  sites. Now: every LLM call records provider-reported token usage into a
  per-run ledger; searches record provider units; the depth controller
  consults `can_afford_pass` before every expansion; runs persist real
  `estimated_cost` (was hardcoded `None`).
- **Dependency waves** (planner's `execution_waves`) — was computed and
  discarded. Now: the summarizer executes wave-by-wave, and dependent
  contracts receive earlier-wave findings as bounded grounding context.
- **Feature-11 stopping** (`agents/stopping.py` marginal-gain logic) —
  merged into the live depth controller: no-novel-query memory (never
  re-runs a search the run already issued), min-iterations per mode,
  mode confidence targets (quick 0.60 … audit 0.85).
- **Confidence v3 signals** — contradictions (pool-size-scaled penalty),
  citation support (10% weight carved from citation coverage) and axis
  coverage (5% from source diversity) now feed the live engine.

### New subsystems

- **Semantic engine** (`core/semantic.py`): TF-IDF hybrid with stemming,
  synonym canonicalization, negation weighting, and fully vectorized batch
  scoring (jaccard/rare-anchor/number gates via presence matmuls).
  Replaces per-pair SequenceMatcher loops: 40×200 batch scoring 865ms→3ms,
  300-fact matrix 5s→0.5s.
- **LLM response cache** (`core/llm_cache.py`): exact-prompt disk cache;
  repeated critic re-evals and re-runs served in ~1ms with the dollar cost
  refunded while token accounting stays visible.
- **Run usage ledger** (`core/usage.py`): ContextVar ledger bridging the
  app-scoped LLM client to per-run budget accounting; streams live to the
  UI on every critic event.
- **Citation health check** (`agents/citation_check.py`): live URL
  re-validation of the final answer's legend (HEAD → 2KB ranged GET,
  bounded concurrency/timeout/count, never fatal) fused with sentence
  support into ok/warn/broken/bad verdicts surfaced in the report, the
  API events and the UI.

### Upgraded components

- **Contradiction engine v2**: numeric (now unit-aware — "%" vs "GW" is no
  longer a conflict, years excluded), polarity (state-assertion vocabulary,
  detected outside the similarity band), temporal (period mismatch),
  severity ranking, additive `kind`/`severity`/`values` fields with legacy
  keys preserved.
- **Answer support**: batched cross-similarity scoring; per-sentence
  `sentence_details` (status + support score) for the UI and benchmarks.
- **Deduplication**: matrix-based greedy merge; stopword-stripped dup floor
  for function-word variants with identical quantities; polarity guard so a
  claim can never corroborate its own negation.
- **Confidence engine v2**: see above; degraded-run cap retained.
- **Provider chain**: usage parsed from responses (`CompletionResult`),
  pre-flight probe results cached 30s on success (failures always re-probe).
- **Persistence**: schema auto-initialization on first use; `file:` /
  `sqlite://` DATABASE_URL forms; parent directories created;
  `save_report` wrapped so persistence can never kill a completed run
  (both were fresh-install failure modes).
- **API events**: `plan` carries `waves`; `critic` and `final_report`
  carry `budget`; `final_report` carries `wave_report` and
  `citation_health`.

### Frontend

- Intelligence panel: **Cost & budget** meter (calls, tokens, USD,
  utilization bar, cache hit rate), **Execution waves** strip,
  **Citation health** row, answer-support health row, and the two new
  confidence signal labels.
- Run state and missions persist real cost.

### Benchmark-driven bug fixes

Found by `bench/run_offline.py` on its first run, fixed, and re-measured:

1. `claim_polarity` counted bare "up"/"down" as direction tokens — the "up"
   inside "follow-up" canceled "reduction" and a polarity-inverted claim
   verified as true (verification F1 0.94 → 1.00).
2. "approved"/"banned"-class state assertions were absent from the polarity
   vocabulary — "approved" vs "not approved" was undetectable.
3. Dedup merged "X" with "not X" and counted the negating source as
   corroboration — now blocked by a polarity guard.
4. Antonym-swap pairs ("contracted" vs "expanded") fell below the
   similarity-band floor; polarity conflicts now need only a 0.45 topical
   floor (contradiction F1 0.80 → 1.00).
5. Contradiction penalty was pool-size-blind — one severe conflict among 3
   facts now costs up to 2x its per-conflict penalty (calibration 75% →
   100% in band).
6. Function-word variants ("declined 14% in 2023" vs "declined by 14%
   during 2023") escaped the 0.86 dedup merge — the stopword-stripped
   dup floor restores them while a quantity guard keeps "grew 25%" and
   "grew 11%" unmergeable.

### Tests

343 tests (was 282), all offline: +15 semantic engine, +9 depth controller
v2, +6 LLM cache/usage, +3 wave execution, +8 contradiction v2, +9 citation
health, +10 confidence v2, +5 sqlite robustness. Autouse fixtures isolate
the LLM cache and citation URL checks so no test ever hits the network.

### Benchmarks

See `BENCHMARK_RESULTS.md`. Headlines: verification F1 1.00, hallucination
leak 0%, citation support 100%, contradiction F1 1.00 (3 kinds), confidence
calibration 100% in-band and monotonic, dedup precision 1.00, LLM cache
6.7x repeat-pass speedup.

### Housekeeping

- `requirements.txt`: numpy added explicitly (the semantic engine is the
  first direct numpy use).
- `.env.example` documents the new knobs (budget note updated from
  "enforced once ported" — it is enforced now; LLM cache; citation check).
- Docs set added: `SETUP.md`, `CONFIG.md`, `ARCHITECTURE.md`,
  `BENCHMARK_RESULTS.md`, this changelog.
