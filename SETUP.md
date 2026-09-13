# Setup Guide

Everything needed to run MARS on a fresh machine. Target profile: any
Linux/macOS/Windows host with **Python 3.10+**, **Node 18+**, 8 GB RAM and
internet access. No GPU, no local models, no paid services required.

---

## 1. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies (all pure-Python or wheels — nothing compiles):
FastAPI, LangGraph, httpx, ddgs (DuckDuckGo search), tenacity, diskcache,
aiosqlite, cryptography, structlog, slowapi, pydantic-settings, numpy.
`pymupdf` is optional and only enables PDF extraction from arXiv results.

### Configure providers

```bash
cp .env.example .env
```

Open `.env` and set **at least one** LLM provider:

```env
# Option A — Groq free tier (recommended: fast, generous free quota)
GROQ_API_KEY=gsk_...
GROQ_MODEL=openai/gpt-oss-20b

# Option B — HuggingFace Inference (fallback)
HUGGINGFACE_API_KEY=hf_...
HUGGINGFACE_MODEL=Qwen/Qwen2.5-7B-Instruct

# Option C — ANY OpenAI-compatible provider (OpenRouter free models,
# Together, a self-hosted vLLM/LM Studio/Ollama endpoint, ...)
CUSTOM_LLM_API_KEY=...
CUSTOM_LLM_BASE_URL=https://openrouter.ai/api/v1
CUSTOM_LLM_MODEL=meta-llama/llama-3.1-8b-instruct:free
```

Notes:

- All three `CUSTOM_LLM_*` values must be set together; when present, the
  custom provider **leads** the chain and Groq/HF remain as fallbacks.
- Values starting with `your_` are treated as unset (the `.env.example`
  placeholders are safe to leave in place).
- Alternatively, add providers at runtime in the UI's **Providers** tab —
  keys are stored Fernet-encrypted in the SQLite DB, and an explicitly
  selected provider becomes exclusive (no other key is spent).
- `TAVILY_API_KEY` is optional (free tier available) and adds a richer
  search provider; DuckDuckGo, Wikipedia, arXiv and Crossref need no keys.

### Start the server

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Verify:

```bash
curl http://127.0.0.1:8000/api/health
# {"status":"ok"}
```

Your first research run, without the UI:

```bash
curl -N -X POST http://127.0.0.1:8000/api/research/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the evidence that retrieval augmented generation reduces hallucinations?", "mode": "standard"}'
```

Each line of the response is a JSON event (`plan`, `search_progress`,
`findings`, `critic`, `final_report`, ...). The full event schema is in
`ARCHITECTURE.md`.

---

## 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://127.0.0.1:5173**. The Vite dev server proxies `/api` to the
backend on port 8000.

For a production build:

```bash
npm run build          # outputs frontend/dist
npx vite preview       # serve the built bundle
```

---

## 3. Modes

Select the research mode in the composer:

| Mode | Agents | Max passes | Confidence target | Use for |
|---|---|---|---|---|
| quick | 2 | 1 | 0.60 | fast lookups |
| standard | 4 | 3 | 0.75 | default research |
| deep | 5 | 5 | 0.80 | multi-angle investigations |
| executive | 5 | 4 | 0.78 | decision-ready briefs |
| audit | 3 | 4 | 0.85 | strict verification passes |
| redteam | 3 | 2 | 0.72 | adversarial review |

---

## 4. Benchmarks

Offline (deterministic, no keys needed):

```bash
cd backend
python bench/run_offline.py            # full suite
python bench/run_offline.py --quick    # skip the e2e pipeline
```

Writes `bench/results/benchmark_results.json` and
`bench/results/BENCHMARK_RESULTS.md`.

Live (spends real free-tier quota — one research run per query):

```bash
cd backend
python bench/run_live.py
MARS_LIVE_QUERIES="question one|question two" python bench/run_live.py
```

---

## 5. Tests

```bash
cd backend
python -m pytest tests/ -q             # 343 tests, all offline
```

Every test runs without network or API keys: HTTP is mocked with respx,
the LLM cache and citation URL checks are force-disabled by `conftest.py`,
and the provider store is isolated from your real database.

---

## 6. Troubleshooting

**"No LLM provider configured" at startup**
Set `GROQ_API_KEY` (or the `CUSTOM_LLM_*` trio) in `backend/.env`, or add a
provider in the UI's Providers tab. Values starting with `your_` count as
unset.

**A run ends with "No LLM provider is reachable right now"**
All configured providers failed their pre-flight probe — the error lists
per-provider reasons. Usual causes: exhausted free-tier daily quota
(resets daily), wrong key (401), or a slow custom endpoint
(raise `CUSTOM_LLM_TIMEOUT_SEC`).

**Runs feel slow**
- The first run of a query pays full search + LLM latency; re-runs hit the
  disk caches (search: 1h TTL; LLM prompts: 6h TTL).
- `MAX_PARALLEL_LLM=2` is deliberate — raising it on free tiers causes
  429s that cost more time than they save.
- `deep` mode runs up to 5 passes; use `standard` for interactive work.

**Groq 404 on model**
Models get decommissioned. Set `GROQ_MODEL` to a current model from
`console.groq.com/docs/models`. `openai/gpt-oss-20b` is the verified
default at the time of writing.

**Database locked / fresh-install errors (v2 fix)**
The persistence layer now auto-creates the schema on first use and accepts
`file:` / `sqlite://` DATABASE_URL forms. If you still see lock errors,
ensure only one backend process uses the same `research.db`.

**DuckDuckGo rate limiting**
DDG is unauthenticated and occasionally throttles. The circuit breaker
skips it for 45s and falls back to other providers; setting a (free)
`TAVILY_API_KEY` removes the dependency entirely.
