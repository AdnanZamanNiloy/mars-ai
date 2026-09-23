# Changelog

## v2.4 — Question-Driven Coverage & Research (Phase 10)

Measured finding: MARS could rank evidence by relevance (v2.3 §1), but a broad
question was still researched and answered through whatever the first retrieval
returned — required dimensions the search never covered simply vanished from the
plan, so partial coverage was presented as the whole landscape.

### Question-driven coverage (`core/synthesis_planner.py`)

- `build_synthesis_plan(required_dimensions=...)`: the plan now plans against the
  QUESTION's required dimensions (the research contracts' own dimension labels),
  not only the axes the retrieved facts happened to fill.
- `required_dimensions_from_plan()`: derives that set deterministically from the
  research contracts, so no new LLM call is spent.
- `SynthesisPlan.uncovered`: a required dimension with no evidence is now named
  (and rendered to the writer as "REQUIRED DIMENSIONS WITH NO EVIDENCE") instead
  of disappearing. Distinct from `under_researched` (thin-but-present).
- `SynthesisPlan.required_total` / `covered_total` / `coverage_ratio`: an explicit
  coverage summary. Partial coverage adds a writer caution that the evidence-
  backed findings are what the research established, not the complete landscape.
- `PlannedDimension.required`: marks dimensions the question demanded.
- Themes are now ordered by the centrality of the evidence behind each section,
  so a peripheral axis with facts (unrelated history/Wikipedia material) cannot
  become a headline theme merely because it was retrieved.

### Coverage-aware follow-up (`core/investigation_planner.py`)

- The existing `dimension_coverage` channel now also targets `uncovered`
  required dimensions, ranked ABOVE merely thin ones (`GAP_UNCOVERED_DIMENSION =
  1.2` vs `GAP_UNDER_RESEARCHED_DIMENSION = 0.9`). Reuses the same candidate kind
  and the existing per-pass budget; no new detector, no new LLM call, no new
  channel.

### Forecasting questions (`core/temporal.py`, `agents/synthesizer.py`, `agents/outline.py`)

- `query_targets_future()`: clock-relative detection of a future year in the
  question, so "the most demanding jobs in 2027" is a forecast even without the
  verb "will". A historical year never triggers the forecast shape.
- The forecast writer guidance now requires four strictly separated statement
  kinds — OFFICIAL PROJECTIONS, CURRENT INDICATORS, CROSS-SOURCE INFERENCE,
  UNKNOWNS — and forbids inventing a probability, rank, percentage or date the
  evidence does not contain.

### Coverage-aware AnalystBrief (`agents/analyst.py`)

- The analyst prompt now receives coverage (required vs covered), the uncovered
  dimensions, the intended interpretation and under-specified readings — not
  just a ranked fact list — so it reasons over the whole landscape.

### Modes (unchanged by design)

- Deep/executive already raise breadth, source requirements and iterations;
  quick lowers them. No per-mode answer templates were added.

## v2.3 — LLM Analytical Synthesis (analyst, not report generator)

Measured finding: the deterministic SynthesisPlan (v2.2) organized the evidence
but did not synthesize it — it had no thesis field, restated convergent claims
as "conclusions", and on a broad query ("current trends in AI") narrowed the
central question onto the first sub-dimension. An LLM stage between plan and
writer was warranted.

### New stage: `agents/analyst.py`

- `analytical_synthesis()`: one small LLM call over the SAME verified evidence
  + deterministic plan, producing a structured `AnalyticalBrief` — central
  thesis, major insights, relationships, counter-evidence, implications,
  cross-source conclusions, uncertainties. Rendered into the writer prompt
  after the plan; no chain-of-thought ever leaves the stage.
- **No new facts**: every statement cites evidence numbers; a deterministic
  guard drops any statement carrying a number absent from the evidence pool
  (citation markers stripped first, so `[1]` is never read as a figure).
- **Verification unchanged**: the writer still cites `[n]` against verified
  facts and `verify_answer_support` still runs. This stage only shapes what the
  writer reasons from.
- **Fallback (AGENTS.md 4.7)**: LLM failure / empty output / disabled
  (`synthesis_analyst_enabled`) degrades to a brief built from the
  deterministic plan (established → dominant → conclusions), never empty when
  the plan is non-empty, never a crash.

### Deterministic plan fix

- `central_question` is now the USER'S QUERY, not the first planned dimension.
  Using the first sub-question narrowed a broad question onto one angle — the
  exact collapse the blueprint exists to prevent.

## v2.2 — Evidence Synthesis Planning

Reasoning over the evidence landscape BEFORE writing, not just better prose.

### New stage: `core/synthesis_planner.py` (deterministic, LLM-free)

- Builds a single explicit `SynthesisPlan` answering the twelve planning
  questions (what is asked, needed dimensions, dominant vs incidental
  findings, conflicts, under-researched dimensions, established/inferred,
  prioritisation, what to omit, structure) from artifacts the pipeline
  already computes — no new detection logic, no new LLM call.
- **Dominant vs incidental** finding ranking by centrality (reuses
  `evidence_completion.claim_impact` plus the grade weight). A finding clears
  the dominant bar when it is quantitative, summary-used or A-grade
  corroborated; a pool with nothing above the bar still gets its single best
  finding as the lead.
- **Under-researched dimensions**: a dimension whose most central finding is
  important but which carries fewer than two findings is named — a
  coverage-vs-importance judgement the existing channels did not make.
- Renders into the writer prompt (`_render_context_block`) alongside the
  reasoning structure; wired in `workflow.synthesizer_node`.

### Synthesis-plan-driven follow-up research

- New allocator channel `KIND_DIMENSION_COVERAGE` in
  `core/investigation_planner.py`: the under-researched dimensions the plan
  names become targeted searches using the dimension's own question text.
  Every other channel fires on an evidence *deficiency*; this is the first
  that fires on a synthesis-level judgement.

### Under-specified (non-homonymous) questions

- `IntentReport.interpretations`: a curated detector for terms with two
  materially useful readings ("most demanding jobs" = high-demand OR
  high-strain). The writer is told to answer both briefly, never to spend the
  answer explaining that the term is ambiguous. Emitted on the `intent`
  event.

### Cross-source synthesis is attributed, not rejected

- `verify_answer_support` classified a genuine synthesis sentence ("taken
  together, these findings indicate…") as *unsupported* because it matched no
  single cited source verbatim. A sentence citing two or more distinct
  verified sources is now `synthesis` — counted separately, never as
  supported fact and never as contamination. Fabricated numbers in such a
  sentence still fail as `numeric_failure`. `citation_check` counts synthesis
  sentences toward the URL's support.

## v2.1 — Intent & Answer-Quality layer


The upgrade against the "What is transformer?" failure class: the pipeline
used to treat the raw query as a search string, mix electrical-transformer
statistics into ML answers, and ship whatever the synthesizer produced.
Understand-before-searching and a pre-delivery quality gate are now
architectural stages, not prompt hopes.

### New pipeline stage: Intent Classification (`agents/intent.py`)

- START → intent → planner: one small, cache-friendly LLM call resolves the
  question BEFORE research is shaped — ambiguity into ranked senses with
  probabilities, the research domain, and the explanation level
  (basic/practical/expert).
- Deterministic fallback (same contract as every agent): curated homonym
  hints + the orchestrator's lexical classifiers, including self-resolution
  when the user already disambiguated ("python snake feeding habits").
- MARS never blocks on a clarifying question: `recommended_action` decides
  whether research targets the dominant sense or structures both, and the
  answer itself opens with the disambiguation.
- The grounding search on the raw query moved into the intent node and is
  shared with the planner (it was the mechanism pulling wrong-sense pages
  into the plan).
- Planner: intent overrides the model's domain classification; ambiguous
  plans tag every contract with a `sense` label; `engineering` added to the
  domain vocabulary (technical specialist).
- Summarizer: sense constraint discards other-sense claims at extraction
  time; facts carry `sense` end to end; cache keyed by sense.
- Synthesizer: ambiguous reports open with a numbered disambiguation
  (`1) **Sense** — explanation`), keep each sense's evidence in its own
  sections, and the citation audit exempts the disambiguation lines; basic
  level forces a plain-language explanation with an analogy.
- New `intent` NDJSON event (+ frontend case) and an Intent row in the
  intelligence panel.

### Provider resilience: LLM-written answers on flaky free tiers

- **Payload ladders**: Groq rejects requests over ~21-41KB (HTTP 413, measured
  live) — exactly where the summarizer prompt lands. Both writer stages now
  shrink their evidence view (22k→12k→6k excerpt budgets; 40→24→14 fact caps)
  and stay LLM-written instead of degrading to extraction. Rate-limit walls
  (429/TPM exhaustion) shrink too; timeouts never do (slowness isn't a size
  signal — shrink-retrying a slow provider multiplies 90s stalls).
- **Permanent-400 fail-fast**: a provider with no credits 400s every call;
  400s with balance/credit/key/model signatures and 404s fail fast instead of
  burning 4 retries per LLM call.
- **Patient Retry-After**: the first retry of a call honors the provider's
  own wait hint up to 20s (one patient wait clears a rolling TPM window);
  later retries cap at 3s. Body-style hints ('try again in X.XXs') parsed.
- **ACTIVE_PROVIDER_FALLBACK** (default false): a FAILING UI-selected active
  provider falls through to the env chain instead of degrading the run —
  built for flaky free proxies; strict exclusivity stays available.

### New pre-delivery stage: Answer Quality Optimizer (`agents/answer_quality.py`)

- Every synthesized answer is scored 0-100 on Accuracy / Relevance /
  Evidence / Clarity / Reasoning from measured pipeline state only (support
  verdicts, verification flags, source registry, citation health) — no LLM.
- Hard floors bind: an ambiguous-query report without the mandatory
  disambiguation fails relevance regardless of its other scores; data-dump
  bullets, missing limitations, unsurfaced conflicts and off-band length
  feed actionable failure strings.
- A failing draft gets exactly ONE re-synthesis with the failures fed back
  (never a loop; skipped when the synthesizer itself is degraded); the
  better draft ships and the scores are disclosed in the report's
  `# Answer Quality` section and the `final_report` event.
- Settings: `INTENT_ENABLED`, `QUALITY_GATE_ENABLED`, `QUALITY_THRESHOLD`.

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
