<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-FFD43B?style=flat&labelColor=306998)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128-white?style=flat&labelColor=009485)
![LangGraph](https://img.shields.io/badge/LangGraph-latest-white?style=flat&labelColor=333)
![React](https://img.shields.io/badge/React-18-white?style=flat&labelColor=222&color=149eca)
![SQLite](https://img.shields.io/badge/SQLite-aiosqlite-white?style=flat&labelColor=003B57&color=0a6a8a)
![License](https://img.shields.io/badge/License-MIT-2da44e?style=flat&labelColor=555)

# MARS — Multi-Agent Research System v2

</div>

MARS is a **deep research assistant** that decomposes complex questions,
runs parallel investigations across five search providers, verifies every
claim against its cited source, detects contradictions, re-checks cited URLs,
and produces an auditable report — while tracking exactly what each run
costs in tokens, calls, dollars and time.

It is built to run on modest hardware (8 GB RAM, i5-1240P-class CPU, no GPU)
against **free-tier remote LLM APIs**, and to beat standard chatbots on the
things that matter for research: verification, citations, contradiction
handling and honest confidence — see `BENCHMARK_RESULTS.md`.

---

## What v2 adds over a standard research chatbot

| Capability | What it does | Where |
|---|---|---|
| **Verification** | Every claim is checked against its source text: weighted lexical overlap, source authority, unit-aware numeric grounding (2% tolerance, scale folding), polarity consistency, quote location | `app/agents/verifier.py` |
| **Citation validation** | Sentences are checked against the evidence of the source they cite, and the cited URLs are **re-fetched live** (HEAD → ranged GET) — dead or moved links are flagged in the report | `app/agents/citation_check.py` |
| **Contradiction detection** | Three detectors: numeric (unit-aware divergence), polarity (affirms vs negates), temporal (same measure, different periods) — with severity ranking and resolution follow-ups fed back into research | `app/core/contradictions.py` |
| **Confidence engine** | 9 signals including citation support, axis coverage and a pool-size-scaled contradiction penalty; degraded runs are capped so extraction fallbacks can never report "high confidence" | `app/core/confidence.py` |
| **Intelligent stopping** | Marginal-gain analysis, no-novel-query memory (never re-runs a search it already ran), mode-aware confidence targets, budget walls | `app/core/depth_controller.py` |
| **Cost-aware reasoning** | Every LLM call records tokens/USD from provider usage fields; the budget governor refuses passes that cannot be paid for; the run ledger streams live to the UI | `app/core/usage.py`, `app/agents/budget.py` |
| **Wave-based parallelism** | Plans carry dependency waves: independent contracts run concurrently, dependent ones wait and receive earlier-wave findings as grounding context | `app/graph/workflow.py` |
| **LLM response cache** | Exact-prompt disk cache — repeated critic re-evals and re-runs are served in ~1ms with zero provider spend (6.7x measured speedup on repeat-heavy workloads) | `app/core/llm_cache.py` |
| **Semantic engine** | CPU-light TF-IDF hybrid (stemming + synonym canonicalization + negation weighting) powering dedup, contradiction banding and citation support — ~70µs per pair, 3ms for 40×200 batch scoring | `app/core/semantic.py` |

The full component map and data flow live in `ARCHITECTURE.md`.

---

## Architecture at a glance

```
Query → Orchestrator → Planner (waves, contracts)
      → Search (Tavily/DDG/Wikipedia/arXiv/Crossref, dedup, circuit breakers)
      → Summarizer (per-contract specialists, wave-ordered, prerequisite context)
      → Verifier (deterministic checks, memory released after pass)
      → Critic + Contradictions + Red Team + Confidence
      → (sufficient? → Synthesizer : expand / budget stop / stall stop)
      → Synthesizer (cited answer) → Citation health check → Finalize
```

Streaming NDJSON events carry the live plan, wave progress, findings,
verification flags, budget telemetry and citation health to the React UI.

---

## Quick start

```bash
# 1. Backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add GROQ_API_KEY (free tier works)

# 2. Run
uvicorn main:app --host 127.0.0.1 --port 8000

# 3. Frontend (separate terminal)
cd frontend && npm install && npm run dev
# open http://127.0.0.1:5173
```

Full instructions (including the OpenAI-compatible custom provider,
benchmarks, and troubleshooting): `SETUP.md`.

---

## Benchmarks

Deterministic offline suite (no network, no API keys):

```bash
cd backend && python bench/run_offline.py
```

Headline results (v2, labeled fixtures in `backend/bench/datasets.py`):

- Verification accuracy **F1 = 1.00** (16 labeled claim/source cases)
- Hallucination rejection: **0% leak** (fabricated numbers, invented facts)
- Citation support: **100% sentence accuracy**
- Contradiction detection: **F1 = 1.00** across numeric, polarity, temporal
- Confidence calibration: **100% in expected band**, monotonic ordering
- LLM cache: **6.7x** repeat-pass speedup, 66.7% hit ratio on repeat-heavy workloads

Full tables, latency profiles and known limitations: `BENCHMARK_RESULTS.md`.
Live-provider benchmarks: `cd backend && python bench/run_live.py`.

---

## Test suite

```bash
cd backend && python -m pytest tests/ -q    # 343 tests, fully offline
```

---

## Project structure

```
mars/
├── backend/
│   ├── app/
│   │   ├── agents/       # planner, search, summarizer, verifier, critic,
│   │   │                 # synthesizer, contradictions, citation health,
│   │   │                 # budget, red team, reliability
│   │   ├── core/         # llm + cache, semantic engine, usage ledger,
│   │   │                 # confidence, depth controller, config, providers
│   │   ├── graph/        # LangGraph workflow (nodes, waves, routing)
│   │   ├── api/          # FastAPI routes (stream, resume, trace, providers)
│   │   └── db/           # SQLite persistence (14 tables, WAL)
│   ├── bench/            # offline + live benchmark suites and datasets
│   ├── scripts/          # scenarios, self-diagnosis
│   └── tests/            # 343 offline tests
├── frontend/             # React 18 + Vite mission console
├── ARCHITECTURE.md       # component map and data flow
├── SETUP.md              # installation, providers, troubleshooting
├── CONFIG.md             # every environment variable
├── BENCHMARK_RESULTS.md  # measured quality and performance
└── CHANGELOG.md          # v2.0 changes
```

---

## Resource profile

Designed for an 8 GB RAM / i5-1240P host:

- No local model serving — all generation on remote free-tier APIs
- Bounded concurrency everywhere: `MAX_PARALLEL_SEARCH=2`, `MAX_PARALLEL_LLM=2`
  (concurrent large prompts are what exhausts free-tier TPM/TPD quotas)
- Page content is released after verification; iteration ceilings and the
  budget governor bound every run
- Disk caches are size-capped (250 MB default)

## Security

- Provider keys are Fernet-encrypted at rest (`MARS_SECRET_KEY` or an
  auto-created `.mars_secret` file); keys are never serialized to the UI
- Never commit real API keys; keep `.env` local (already in `.gitignore`)
- The service binds to `127.0.0.1` by default — do not expose it directly
  to the internet without an auth layer
