# MARS Offline Golden Evaluation (v1)

*Generated 2026-09-16T10:41:42Z — deterministic, network-free.*

Golden set: `bench/golden/queries_v1.json`  
Thresholds: `bench/golden/thresholds_v1.json`  
Queries: 26 (scored 25, failed 1)

**Result: PASS** (0 threshold violation(s))

## Aggregate metrics

| Metric | Actual | Floor | Status |
|---|---|---|---|
| query_type_accuracy | 1.0000 | 1.0000 | ok |
| dimension_hit_rate | 0.8800 | 0.8000 | ok |
| forbidden_clean_rate | 1.0000 | 1.0000 | ok |
| section_presence_rate | 1.0000 | 1.0000 | ok |
| citation_resolution_rate | 1.0000 | 1.0000 | ok |
| verified_claims_mean | 2.5200 | 1.8000 | ok |
| corroborated_claims_mean | 1.8000 | 1.2000 | ok |
| distinct_domains_mean | 1.9600 | 1.4000 | ok |
| primary_share_mean | 0.8133 | 0.6500 | ok |
| grade_ab_share | 1.0000 | 0.9500 | ok |
| answer_quality_mean | 69.5600 | 62.0000 | ok |
| answer_relevance_mean | 0.4099 | 0.3000 | ok |
| support_rate_mean | 0.5053 | 0.4000 | ok |
| minimums_pass_rate | 1.0000 | 0.9500 | ok |
| depth_routing_pass_rate | 1.0000 |  | ok |
| detection_precision | 1.0000 |  | ok |
| detection_recall | 1.0000 |  | ok |
| detection_f1 | 1.0000 |  | ok |
| resolution_precision | 1.0000 |  | ok |
| resolution_recall | 1.0000 |  | ok |
| resolution_f1 | 1.0000 |  | ok |
| classification_accuracy | 1.0000 |  | ok |
| provider_classification_pass_rate | 1.0000 |  | ok |

## Per category

| Category | Queries | Quality | Support | Citation res. | Min-pass |
|---|---|---|---|---|---|
| ambiguous_term | 3 | 68.000 | 0.444 | 1.000 | 1.000 |
| causal | 4 | 67.250 | 0.442 | 1.000 | 1.000 |
| comparison | 4 | 67.250 | 0.483 | 1.000 | 1.000 |
| current_trend | 4 | 72.333 | 0.500 | 1.000 | 1.000 |
| decision_policy | 3 | 66.000 | 0.556 | 1.000 | 1.000 |
| factual_explanation | 5 | 73.800 | 0.587 | 1.000 | 1.000 |
| quantitative | 3 | 71.000 | 0.500 | 1.000 | 1.000 |

## Contradiction detection / resolution

Labeled cases: 13 (correct 13, incorrect 0)

| id | expected | observed | ok |
|---|---|---|---|
| true_same_measure_conflict | contradiction | contradiction | Y |
| true_same_measure_conflict_market_growth | contradiction | contradiction | Y |
| true_same_measure_conflict_primary_vs_secondary | contradiction | contradiction | Y |
| different_period_same_measure | resolved_explained | resolved_explained | Y |
| different_period_capacity | resolved_explained | resolved_explained | Y |
| different_scope_global_vs_us | resolved_explained | resolved_explained | Y |
| different_scope_regional_segments | resolved_explained | resolved_explained | Y |
| unrelated_shared_numbers_and_year | not_contradiction | not_contradiction | Y |
| unrelated_shared_number_different_metric | not_contradiction | not_contradiction | Y |
| unit_mismatch_percent_vs_capacity | not_contradiction | not_contradiction | Y |
| polarity_direct_negation | contradiction | contradiction | Y |
| polarity_opposite_directions_different_years | contradiction | contradiction | Y |
| same_value_incidental_context_year | not_contradiction | not_contradiction | Y |

## Per query

| id | cat | type ok | dims | sections | cit. res | verified | corrob | domains | quality | min ok |
|---|---|---|---|---|---|---|---|---|---|---|
| factual-rag | factual_explanation | Y | 3/3 | 5/5 | 1.00 | 5 | 3 | 4 | 75 | Y |
| factual-mrna | factual_explanation | Y | 3/3 | 5/5 | 1.00 | 3 | 2 | 2 | 73 | Y |
| factual-transformer-ml | factual_explanation | Y | 2/3 | 5/5 | 1.00 | 3 | 1 | 3 | 74 | Y |
| factual-solid-state | factual_explanation | Y | 3/3 | 5/5 | 1.00 | 2 | 2 | 2 | 76 | Y |
| trend-renewables | current_trend | Y | 3/3 | 5/5 | 1.00 | 4 | 3 | 3 | 70 | Y |
| trend-ai-adoption | current_trend | Y | 2/3 | 5/5 | 1.00 | 1 | 1 | 1 | 75 | Y |
| trend-storage | current_trend | ERROR | | | | | | | | UnboundLocalError: cannot access local variable 'fallback_re |
| comparison-nuclear-solar | comparison | Y | 3/3 | 5/5 | 1.00 | 3 | 2 | 2 | 62 | Y |
| comparison-lithium-solid | comparison | Y | 3/3 | 5/5 | 1.00 | 3 | 3 | 2 | 72 | Y |
| comparison-rag-finetune | comparison | Y | 3/3 | 5/5 | 1.00 | 4 | 3 | 4 | 71 | Y |
| decision-nuclear-investment | decision_policy | Y | 4/4 | 5/5 | 1.00 | 1 | 1 | 1 | 62 | Y |
| decision-carbon-policy | decision_policy | Y | 4/4 | 5/5 | 1.00 | 3 | 3 | 2 | 73 | Y |
| ambiguous-transformer | ambiguous_term | Y | 2/3 | 5/5 | 1.00 | 3 | 1 | 2 | 67 | Y |
| ambiguous-apple | ambiguous_term | Y | 2/2 | 5/5 | 1.00 | 2 | 1 | 1 | 65 | Y |
| ambiguous-bank | ambiguous_term | Y | 2/2 | 5/5 | 1.00 | 1 | 1 | 1 | 72 | Y |
| causal-rag-hallucination | causal | Y | 1/3 | 5/5 | 1.00 | 5 | 3 | 4 | 76 | Y |
| causal-battery-costs | causal | Y | 2/3 | 5/5 | 1.00 | 3 | 2 | 2 | 68 | Y |
| causal-solar-growth | causal | Y | 2/3 | 5/5 | 1.00 | 1 | 1 | 1 | 62 | Y |
| quant-renewable | quantitative | Y | 2/2 | 5/5 | 1.00 | 3 | 2 | 3 | 70 | Y |
| quant-battery-price | quantitative | Y | 2/2 | 5/5 | 1.00 | 2 | 2 | 1 | 73 | Y |
| quant-ai-investment | quantitative | Y | 2/2 | 5/5 | 1.00 | 1 | 1 | 1 | 70 | Y |
| factual-heat-pumps | factual_explanation | Y | 3/3 | 5/5 | 1.00 | 2 | 2 | 1 | 71 | Y |
| trend-ev-adoption | current_trend | Y | 3/3 | 5/5 | 1.00 | 2 | 1 | 1 | 72 | Y |
| comparison-wind-solar | comparison | Y | 3/3 | 5/5 | 1.00 | 3 | 2 | 2 | 64 | Y |
| causal-nuclear-cost | causal | Y | 2/3 | 5/5 | 1.00 | 2 | 1 | 2 | 63 | Y |
| decision-ai-regulation | decision_policy | Y | 2/3 | 5/5 | 1.00 | 1 | 1 | 1 | 63 | Y |

## Determinism / separation from live judging

- Every number here comes from deterministic, LLM-free scorers and a
  scripted offline pipeline. No model is called and the network is not
  touched. Live/model-judged quality stays in `bench/run_live.py`.
- Citation and numeric metrics are computed over the writer body only:
  `sources.strip_machine_sections` removes the machine-appended sections
  (Evidence integrity, Source ledger, Sources, Limitations, ...) so the
  accounting is never scored as if it were unsupported prose.
