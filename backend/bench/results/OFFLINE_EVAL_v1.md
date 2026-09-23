# MARS Offline Golden Evaluation (v1)

*Generated 2026-09-23T09:08:42Z — deterministic, network-free.*

Golden set: `bench/golden/queries_v1.json`  
Thresholds: `bench/golden/thresholds_v1.json`  
Queries: 26 (scored 26, failed 0)

**Result: PASS** (0 threshold violation(s))

## Aggregate metrics

| Metric | Actual | Floor | Status |
|---|---|---|---|
| query_type_accuracy | 1.0000 | 1.0000 | ok |
| dimension_hit_rate | 0.8846 | 0.8000 | ok |
| forbidden_clean_rate | 1.0000 | 1.0000 | ok |
| section_presence_rate | 0.4000 |  | ok |
| citation_resolution_rate | 1.0000 | 1.0000 | ok |
| verified_claims_mean | 2.4615 | 1.8000 | ok |
| corroborated_claims_mean | 1.7308 | 1.2000 | ok |
| distinct_domains_mean | 1.9231 | 1.4000 | ok |
| primary_share_mean | 0.8205 | 0.6500 | ok |
| grade_ab_share | 1.0000 | 0.9500 | ok |
| answer_quality_mean | 60.1154 | 57.0000 | ok |
| answer_relevance_mean | 0.3945 | 0.3000 | ok |
| support_rate_mean | 0.5051 | 0.4000 | ok |
| minimums_pass_rate | 1.0000 | 0.9000 | ok |

## Per category

| Category | Queries | Quality | Support | Citation res. | Min-pass |
|---|---|---|---|---|---|
| ambiguous_term | 3 | 58.667 | 0.444 | 1.000 | 1.000 |
| causal | 4 | 58.750 | 0.442 | 1.000 | 1.000 |
| comparison | 4 | 58.500 | 0.483 | 1.000 | 1.000 |
| current_trend | 4 | 58.000 | 0.500 | 1.000 | 1.000 |
| decision_policy | 3 | 58.667 | 0.556 | 1.000 | 1.000 |
| factual_explanation | 5 | 65.800 | 0.587 | 1.000 | 1.000 |
| quantitative | 3 | 60.333 | 0.500 | 1.000 | 1.000 |

## Per query

| id | cat | type ok | dims | sections | cit. res | verified | corrob | domains | quality | min ok |
|---|---|---|---|---|---|---|---|---|---|---|
| factual-rag | factual_explanation | Y | 3/3 | 2/5 | 1.00 | 5 | 3 | 4 | 66 | Y |
| factual-mrna | factual_explanation | Y | 3/3 | 2/5 | 1.00 | 3 | 2 | 2 | 65 | Y |
| factual-transformer-ml | factual_explanation | Y | 2/3 | 2/5 | 1.00 | 3 | 1 | 3 | 67 | Y |
| factual-solid-state | factual_explanation | Y | 3/3 | 2/5 | 1.00 | 2 | 2 | 2 | 69 | Y |
| trend-renewables | current_trend | Y | 3/3 | 2/5 | 1.00 | 4 | 3 | 3 | 61 | Y |
| trend-ai-adoption | current_trend | Y | 2/3 | 2/5 | 1.00 | 1 | 1 | 1 | 64 | Y |
| trend-storage | current_trend | Y | 3/3 | 2/5 | 1.00 | 1 | 0 | 1 | 47 | Y |
| comparison-nuclear-solar | comparison | Y | 3/3 | 2/5 | 1.00 | 3 | 2 | 2 | 53 | Y |
| comparison-lithium-solid | comparison | Y | 3/3 | 2/5 | 1.00 | 3 | 3 | 2 | 64 | Y |
| comparison-rag-finetune | comparison | Y | 3/3 | 2/5 | 1.00 | 4 | 3 | 4 | 62 | Y |
| decision-nuclear-investment | decision_policy | Y | 4/4 | 2/5 | 1.00 | 1 | 1 | 1 | 54 | Y |
| decision-carbon-policy | decision_policy | Y | 4/4 | 2/5 | 1.00 | 3 | 3 | 2 | 66 | Y |
| ambiguous-transformer | ambiguous_term | Y | 2/3 | 2/5 | 1.00 | 3 | 1 | 2 | 59 | Y |
| ambiguous-apple | ambiguous_term | Y | 2/2 | 2/5 | 1.00 | 2 | 1 | 1 | 55 | Y |
| ambiguous-bank | ambiguous_term | Y | 2/2 | 2/5 | 1.00 | 1 | 1 | 1 | 62 | Y |
| causal-rag-hallucination | causal | Y | 1/3 | 2/5 | 1.00 | 5 | 3 | 4 | 67 | Y |
| causal-battery-costs | causal | Y | 2/3 | 2/5 | 1.00 | 3 | 2 | 2 | 59 | Y |
| causal-solar-growth | causal | Y | 2/3 | 2/5 | 1.00 | 1 | 1 | 1 | 54 | Y |
| quant-renewable | quantitative | Y | 2/2 | 2/5 | 1.00 | 3 | 2 | 3 | 60 | Y |
| quant-battery-price | quantitative | Y | 2/2 | 2/5 | 1.00 | 2 | 2 | 1 | 63 | Y |
| quant-ai-investment | quantitative | Y | 2/2 | 2/5 | 1.00 | 1 | 1 | 1 | 58 | Y |
| factual-heat-pumps | factual_explanation | Y | 3/3 | 2/5 | 1.00 | 2 | 2 | 1 | 62 | Y |
| trend-ev-adoption | current_trend | Y | 3/3 | 2/5 | 1.00 | 2 | 1 | 1 | 60 | Y |
| comparison-wind-solar | comparison | Y | 3/3 | 2/5 | 1.00 | 3 | 2 | 2 | 55 | Y |
| causal-nuclear-cost | causal | Y | 2/3 | 2/5 | 1.00 | 2 | 1 | 2 | 55 | Y |
| decision-ai-regulation | decision_policy | Y | 2/3 | 2/5 | 1.00 | 1 | 1 | 1 | 56 | Y |

## Determinism / separation from live judging

- Every number here comes from deterministic, LLM-free scorers and a
  scripted offline pipeline. No model is called and the network is not
  touched. Live/model-judged quality stays in `bench/run_live.py`.
- Citation and numeric metrics are computed over the writer body only:
  `sources.strip_machine_sections` removes the machine-appended sections
  (Evidence integrity, Source ledger, Sources, Limitations, ...) so the
  accounting is never scored as if it were unsupported prose.
