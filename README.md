# MARS: Multi Agent Research System

MARS is a lightweight, asynchronous research assistant that plans a query, searches the web, extracts evidence, critiques completeness, and produces a final markdown report.

It is designed for low-resource environments by delegating model inference to remote APIs and keeping local infrastructure minimal.

## Highlights

- Multi-agent workflow orchestrated with LangGraph
- Streaming research progress via NDJSON events
- SQLite persistence for final reports
- Provider fallback for LLM and web search
- React + Vite frontend with no heavy UI dependencies

## System Architecture

Pipeline:

`Query -> Planner -> Search -> Summarizer -> Critic -> (loop or Finalize)`

Core components:

- `app/agents/planner.py`: Generates focused sub-questions
- `app/agents/search.py`: Runs bounded async search with fallback providers
- `app/agents/summarizer.py`: Extracts structured facts from search snippets
- `app/agents/critic.py`: Evaluates sufficiency and suggests refinement
- `app/graph/workflow.py`: Defines graph transitions and loop routing
- `app/api/routes.py`: Exposes health and streaming research endpoints
- `app/db/sqlite.py`: Initializes and writes report records

## Technology Stack

- Backend: FastAPI, LangGraph, httpx, aiosqlite, pydantic
- Frontend: React 18, Vite 5
- Storage: SQLite

## Project Layout

```text
main.py
app/
  agents/
  api/
  core/
  db/
  graph/
ui/
  src/
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm 9+

## Quick Start

### 1. Backend setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Configure environment

Set at least one LLM provider key in `.env`:

- `GROQ_API_KEY` (recommended)
- `HUGGINGFACE_API_KEY` (fallback)

Optional:

- `TAVILY_API_KEY` for richer search results

### 3. Run backend

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 4. Run frontend

```bash
cd ui
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Configuration Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | No* | `""` | Primary LLM API key |
| `GROQ_MODEL` | No | `llama-3.1-8b-instant` | Groq model name |
| `HUGGINGFACE_API_KEY` | No* | `""` | Fallback LLM API key |
| `HUGGINGFACE_MODEL` | No | `mistralai/Mistral-7B-Instruct-v0.3` | HuggingFace model |
| `TAVILY_API_KEY` | No | `""` | Optional search provider |
| `DATABASE_URL` | No | `./research.db` | SQLite file path |
| `MAX_PARALLEL_SEARCH` | No | `2` | Max concurrent search operations |
| `MAX_ITERATIONS` | No | `3` | Max planner-search-summarize-critic loops |
| `LLM_TIMEOUT_SEC` | No | `25` | LLM request timeout |
| `SEARCH_TIMEOUT_SEC` | No | `20` | Search request timeout |

`*` At least one of `GROQ_API_KEY` or `HUGGINGFACE_API_KEY` must be set.

## API Endpoints

### `GET /`

Service metadata and available endpoints.

### `GET /api/health`

Health check.

Response:

```json
{ "status": "ok" }
```

### `POST /api/research/stream`

Streams research progress as newline-delimited JSON (`application/x-ndjson`).

Request body:

```json
{ "query": "What are the latest open-source small language model benchmarks in 2026?" }
```

Validation:

- `query` length: 5 to 500 characters

Event schema:

- `progress`: `{ "type": "progress", "request_id": "...", "message": "..." }`
- `plan`: `{ "type": "plan", "items": ["..."] }`
- `search_progress`: `{ "type": "search_progress", "snippets": 12 }`
- `critic`: `{ "type": "critic", "iteration": 1, "reason": "..." }`
- `findings`: `{ "type": "findings", "items": [{"claim":"...","source":"..."}] }`
- `final_report`: `{ "type": "final_report", "report": "...", "confidence": 0.73 }`
- `error`: `{ "type": "error", "message": "..." }`

Example:

```bash
curl -N -X POST http://127.0.0.1:8000/api/research/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"Compare efficient open-source speech-to-text models for CPU inference"}'
```

## Persistence

Final reports are stored in SQLite table `research_reports` with:

- `query`
- `report`
- `confidence`
- `created_at` (UTC ISO-8601)

## Reliability and Resource Profile

- Async I/O across API, search, and workflow operations
- Bounded search concurrency (`MAX_PARALLEL_SEARCH`)
- Iteration ceiling (`MAX_ITERATIONS`) to avoid unbounded loops
- Fallback behavior for missing provider responses
- Minimal local footprint: no Redis, no external queue, no local model serving

## Security Notes

- Do not commit real API keys to source control.
- Keep `.env` local and secret.
- Rotate keys immediately if they are exposed.

## Development Notes

- Frontend proxy is configured in `ui/vite.config.js` for `/api -> http://127.0.0.1:8000`
- CORS allows `http://127.0.0.1:5173` and `http://localhost:5173`

## License

Add your project license here (for example: MIT, Apache-2.0, or Proprietary).
