# Configuration Reference

Every environment variable, its default, and what it actually controls.
All variables live in `backend/.env` (see `.env.example`); real environment
variables override `.env` values, which override `.env.example`.

## LLM providers

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | `""` | Primary provider key (free tier works). Values starting with `your_` count as unset. |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Groq model id. Check console.groq.com when a model is decommissioned. |
| `HUGGINGFACE_API_KEY` | `""` | Fallback provider key. |
| `HUGGINGFACE_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | HF Inference model (3-model fallback list is built in). |
| `CUSTOM_LLM_API_KEY` | `""` | OpenAI-compatible provider key — all three CUSTOM_* must be set together. |
| `CUSTOM_LLM_BASE_URL` | `""` | Provider base URL (with or without `/chat/completions`). |
| `CUSTOM_LLM_MODEL` | `""` | Provider model id. |
| `CUSTOM_LLM_TIMEOUT_SEC` | `90` | Per-call timeout for the custom provider only (slow hosts need 30-60s on planner-sized prompts). |
| `MARS_SECRET_KEY` | auto | Fernet key material for encrypting UI-added provider keys at rest. Unset falls back to an auto-created `backend/.mars_secret` file. |

Provider chain order: UI-selected active provider (exclusive) →
`CUSTOM_LLM_*` trio → Groq → HuggingFace, with per-provider circuit
breakers (timeouts open immediately; fast failures trip after 3; 60s cooldown).

## Search providers

| Variable | Default | Description |
|---|---|---|
| `TAVILY_API_KEY` | `""` | Optional richer web search (free tier available). |
| `TAVILY_SEARCH_DEPTH` | `basic` | `basic` or `advanced`. |

DuckDuckGo, Wikipedia, arXiv and Crossref need no keys. Provider fan-out is
type-driven: academic queries add arXiv + Crossref; encyclopedia/statistical
add Wikipedia; news uses the Tavily news topic.

## Runtime limits (8 GB RAM profile)

| Variable | Default | Description |
|---|---|---|
| `MAX_PARALLEL_SEARCH` | `2` | Concurrent whole-contract searches. |
| `MAX_PARALLEL_FETCH` | `4` | Concurrent page downloads (separate bulkhead). |
| `MAX_PARALLEL_AGENTS` | `3` | Hardware cap on planned agents (deep/executive may exceed via mode presets). |
| `MAX_PARALLEL_LLM` | `2` | Concurrent LLM calls pipeline-wide. Deliberately low: concurrent large prompts are what burns free-tier TPM/TPD quotas. |
| `MAX_ITERATIONS` | `3` | Planner→search→summarize→critic loop ceiling (mode presets override). |
| `SEARCH_FETCH_TOP_N` | `8` | Top-ranked results per sub-question whose full page is fetched. |
| `SEARCH_MAX_QUERIES_PER_PASS` | `8` | Query fan-out cap per pass (questions + variants). |
| `MAX_QUERIES_PER_CONTRACT` | `3` | Query variants per contract, including primary-source fan-out. |
| `SEARCH_MAX_RESULTS` | `10` | Results kept per provider search before dedup/rank. |
| `SEARCH_RETRY_ATTEMPTS` | `3` | Transient-error retries per provider search (timeouts fail fast). |

## Timeouts

| Variable | Default | Description |
|---|---|---|
| `LLM_TIMEOUT_SEC` | `25` | Per-attempt timeout for Groq/HF calls. |
| `SEARCH_TIMEOUT_SEC` | `20` | Per-request timeout for search providers and page fetches. |
| `RESEARCH_TIMEOUT_SEC` | `300` | Outer ceiling on one streamed research run. |

## Research budget governor (v2: now enforced)

| Variable | Default | Description |
|---|---|---|
| `MAX_BUDGET_USD` | `0.50` | Dollar ceiling per run (mode multipliers: quick 0.35x … deep 2.5x). |
| `MAX_BUDGET_TOKENS` | `400000` | Token ceiling per run. |
| `MAX_LLM_CALLS` | `60` | LLM call ceiling per run. |
| `MAX_RESEARCH_SECONDS` | `300` | Wall-clock ceiling per run. |
| `STRICT_BUDGET` | `false` | `true` raises BudgetExceeded instead of gracefully finalizing. |

Every LLM call records provider-reported token usage (character estimates
for HF); searches record provider units; the depth controller consults the
ledger before every expansion and finalizes when a pass cannot be paid for.
Cache hits refund their dollar cost while keeping token accounting visible.

## Dynamic depth / intelligent stopping

| Variable | Default | Description |
|---|---|---|
| `SUFFICIENCY_THRESHOLD` | `0.75` | Base confidence target (mode targets override: quick 0.60 … audit 0.85). |
| `MIN_MARGINAL_GAIN` | `0.03` | Per-iteration confidence gain below which a pass counts as stalled (two stalls stop). |
| `MAX_RESEARCH_DEPTH` | `0` | Hard expansion-depth ceiling; `0` means use `MAX_ITERATIONS`. |

Additional stopping rules need no configuration: the no-novel-query memory
compares the critic's follow-ups against every already-run search; budget
walls apply from the ledger; `min_iterations` per mode (audit ≥ 2) prevents
premature stops.

## Caching

| Variable | Default | Description |
|---|---|---|
| `CACHE_SIZE_LIMIT_BYTES` | `250000000` | Hard disk-cache size cap (search + LLM caches share the budget). |
| `CACHE_TTL_SEC` | `3600` | Search-result TTL. |
| `LLM_CACHE_ENABLED` | `true` | Exact-prompt LLM response cache. Tests disable it automatically. |
| `LLM_CACHE_TTL_SEC` | `21600` | LLM response TTL (6h — sources age out). |

## Citation validation (v2)

| Variable | Default | Description |
|---|---|---|
| `CITATION_CHECK_ENABLED` | `true` | Live URL re-check of the sources the final answer cites. |
| `CITATION_CHECK_TIMEOUT_SEC` | `5.0` | Per-URL probe timeout (HEAD, then 2KB ranged GET). |
| `CITATION_CHECK_MAX` | `10` | Top cited sources re-checked per report (bounded, never fatal). |

## Intent classification (v2.1)

| Variable | Default | Description |
|---|---|---|
| `INTENT_ENABLED` | `true` | One small LLM call before planning: resolves ambiguity into ranked senses, sets the research domain and explanation level. Deterministic fallback (curated homonym hints + lexical classifiers) when the call fails. |

## Answer quality gate (v2.1)

| Variable | Default | Description |
|---|---|---|
| `QUALITY_GATE_ENABLED` | `true` | Score every synthesized answer 0-100 on accuracy/relevance/evidence/clarity/reasoning from measured state. |
| `QUALITY_THRESHOLD` | `70` | Passing line. A failing draft gets exactly one corrective re-synthesis; scores are disclosed in the report either way. |

## API behavior

| Variable | Default | Description |
|---|---|---|
| `RATE_LIMIT` | `5/minute` | slowapi limit on `/api/research/stream`. |
| `DATABASE_URL` | `./research.db` | SQLite path. Accepts plain paths, `file:` and `sqlite://` URI forms; missing parent directories are created. |
