# MARS Implementation Manual — Agent Execution Guide

Target repo: `https://github.com/AdnanZamanNiloy/mars-ai`
Companion doc: `MARS-vision-v2.md` (feature rationale, UI mockups, data model, brand). This manual does not repeat feature descriptions — it tells an implementing agent *what to build, in which files, in what order, and how to know it's done*. Where a task implements a feature from the vision doc, this manual points to it by name instead of re-explaining it.

This manual was written after cloning and reading the actual repository, not from the README alone. Section 2 lists confirmed, verified bugs with file:line references — these are real, not hypothetical.

---

## 0. How to use this manual (agent operating rules)

If you are an autonomous coding agent executing this manual, follow these rules without exception:

```text
1. Work phase by phase, in order: Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4.
   Do not start a later phase until every task in the current phase has passed
   its verification step. Phase 0 and Phase 1 are not optional and not
   reorderable — every later phase assumes they're done.

2. One task = one commit. Use the task ID as the commit prefix, e.g.:
     git commit -m "[0.1] fix undefined PLANNER_SYSTEM_PROMPT"
   Small, reviewable commits. Never squash multiple tasks into one commit.

3. Create one branch per phase off the repo's default branch:
     git checkout -b phase-0-critical-fixes
   Do not merge a phase branch until all its tasks are done and verified.
   Never force-push. Never rewrite history that's already pushed.

4. Run the task's verification command before moving to the next task.
   If verification fails, fix it before proceeding — do not continue with a
   known-broken step and "fix it later."

5. Never commit secrets. .env stays untracked (it already is, per .gitignore).
   If a task needs a new env var, add it to .env.example with a placeholder
   value, not .env.

6. If a task requires a product decision this manual doesn't cover (e.g. "what
   exact wording should the executive report use"), make the smallest
   reasonable choice, note it in the commit message, and move on. Do not
   block on it. If a task requires a decision that changes architecture (e.g.
   "should we drop SQLite for Postgres"), stop and surface the question
   instead of deciding unilaterally — that's out of scope for this manual,
   which assumes SQLite throughout.

7. Do not delete or weaken an existing working code path to implement a new
   one unless the task explicitly says to replace it. Several agents
   (planner, critic, synthesizer) already have deterministic fallback paths
   that keep the pipeline working when the LLM call fails — preserve that
   pattern in every new agent you add.

8. After finishing all tasks in a phase, update the checklist in Appendix A
   (check the boxes) in the same commit as the last task, and open a PR from
   the phase branch with a summary of what changed and the verification
   output.
```

---

## 1. Repo & environment orientation

Actual structure (verified by cloning the repo):

```text
mars-ai/
├── main.py                       # FastAPI app, CORS, lifespan init
├── requirements.txt               # fastapi, uvicorn, langgraph, httpx,
│                                   # pydantic, aiosqlite, duckduckgo-search,
│                                   # python-dotenv — no tests, no retry libs,
│                                   # no rate limiting, no caching libs yet
├── .env.example
├── app/
│   ├── agents/
│   │   ├── planner.py             # 226 lines
│   │   ├── search.py              # 292 lines — DDG + Wikipedia, no Tavily
│   │   ├── summarizer.py          # 145 lines
│   │   ├── critic.py              # 178 lines
│   │   ├── synthesizer.py         # 108 lines — exists, not in README
│   │   └── evidence_utils.py      # 247 lines — source scoring, dedup,
│   │                               # domain filtering — exists, not in README
│   ├── api/routes.py              # 122 lines — the /api/research/stream route
│   ├── core/
│   │   ├── config.py              # 44 lines — dataclass Settings
│   │   └── llm.py                 # 133 lines — Groq + HF client, JSON extraction
│   ├── db/sqlite.py                # 30 lines — single research_reports table
│   └── graph/workflow.py          # 268 lines — LangGraph state machine
└── ui/
    └── src/
        ├── App.jsx                # NDJSON stream consumer, per-event UI
        └── components/            # PipelineBar, FinalAnswerCard, SectionCard
```

Setup:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in GROQ_API_KEY at minimum
cd ui && npm install && cd ..
```

Run:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
cd ui && npm run dev   # http://127.0.0.1:5173
```

Smoke test:

```bash
curl -s http://127.0.0.1:8000/api/health
curl -N -X POST http://127.0.0.1:8000/api/research/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval augmented generation?"}'
```

---

## 2. Current state assessment — confirmed issues

These were found by reading the actual code, not inferred. Fix all of them in Phase 0 before building anything new.

```text
[BUG-1 · CRITICAL] app/agents/planner.py:177
  `PLANNER_SYSTEM_PROMPT` is referenced but never defined anywhere in the
  file or the codebase (confirmed via grep across app/). Every LLM planning
  call raises NameError, which is caught by the surrounding `except
  Exception`, and silently falls back to `fallback_plan()`. Net effect: the
  Planner has never actually used the LLM. Every research run — regardless
  of query complexity — gets the same fixed 4-question definition/mechanism/
  application/limitations template. This is the single highest-impact fix
  in the codebase.

[BUG-2 · HIGH] app/agents/critic.py:7-101 vs 120-131
  `CRITIC_SYSTEM_PROMPT` instructs the model to return one JSON schema
  (passed / checks / failed_count / issues / suggested_queries /
  confidence_score / limitations_note), but the `user_prompt` built inside
  `critic_agent()` asks for a different schema (is_sufficient / reason /
  improved_queries / confidence), and the parsing code only reads the
  second schema. The model receives contradictory instructions.
  `CRITIC_USER_TEMPLATE` is defined but never used — dead code.

[BUG-3 · HIGH] app/graph/workflow.py:58, app/api/routes.py:65,107
  `RUNTIME_STATE` is a module-level dict keyed by `request_id`. Every
  request writes a full LangGraph state snapshot into it (all search
  results, all facts, full report) and nothing ever deletes the entry.
  This is an unbounded memory leak — on an 8GB host this will degrade or
  crash the process under sustained use.

[BUG-4 · MEDIUM] app/api/routes.py — stream_research()
  No timeout wraps the whole request. `LLM_TIMEOUT_SEC` / `SEARCH_TIMEOUT_SEC`
  bound individual HTTP calls, but a pathological loop (up to
  MAX_ITERATIONS re-runs of planner→search→summarize→critic) has no outer
  ceiling on total request time.

[GAP-5 · MEDIUM] app/agents/search.py
  Tavily is documented in README/.env.example as an optional enriched
  search provider but is never called anywhere in the code. Search is
  DuckDuckGo (`duckduckgo-search` package — verify this still resolves on
  PyPI; it has had naming churn upstream toward `ddgs`) plus the raw
  Wikipedia API, with no retry/backoff. `_wiki()` and `_fetch_content()`
  both use bare `except: return ""`/`except: return []`, which swallows
  real errors a retry could have fixed.

[GAP-6 · LOW] app/agents/search.py:219-224
  Content fetching for the top 3 ranked results per sub-question is
  sequential (`for r in ranked[:3]: await _fetch_content(...)`), not
  concurrent — avoidable latency.

[GAP-7 · HIGH, PRODUCT-LEVEL] app/agents/synthesizer.py:19
  The Synthesizer's system prompt explicitly says "Do NOT include source
  links or citations inline." Despite the pipeline computing real source-
  reliability scores and deduped claims (evidence_utils.py), none of that
  provenance reaches the synthesized answer text itself — only a generic
  "Supporting Evidence" list is appended afterward, disconnected from
  specific sentences. This is the gap the Verification Agent and Confidence
  Engine (Phase 2) are meant to close.

[GAP-8 · LOW] app/graph/workflow.py:77,248
  `MAX_ITERATIONS` is silently floored at 3 in two places
  (`build_initial_state`: `max(3, int(max_iterations))`, and
  `route_after_critic`: `max(max_iterations, 3)`). Setting
  `MAX_ITERATIONS=1` in `.env` currently has no effect. Not necessarily
  wrong, but undocumented, and it costs 3x the API calls a user configuring
  a lower value would expect.

[GAP-9 · LOW] No tests exist anywhere in the repo. No `tests/` directory,
  no test runner in requirements.txt.

[GAP-10 · LOW] `research.db` (a real SQLite file with prior research
  history, ~330KB) is committed to git despite being listed in
  `.gitignore` — it was tracked before the ignore rule was added, so the
  rule doesn't retroactively untrack it.

[GAP-11 · MEDIUM] No rate limiting, no structured logging, no `/metrics`,
  no auth. CORS is hardcoded to localhost dev origins only.
```

The frontend (`ui/src/App.jsx`) is further along than expected: it already renders per-event-type UI (plan, critique loops, findings cards with confidence bars and source-trust badges) rather than a single spinner. What it lacks is reconnect-on-drop handling — a dropped `fetch` just ends the run with an error.

---

## Phase 0 — Critical bug fixes

Do this before anything else. Branch: `phase-0-critical-fixes`. Estimated effort: small, mechanical fixes.

### 0.1 — Fix BUG-1: define `PLANNER_SYSTEM_PROMPT`

File: `app/agents/planner.py`. Add a module-level `PLANNER_SYSTEM_PROMPT` string above `planner_agent()`, matching the output contract already defined by the `PlannerOutput`/`SubQuestion` TypedDicts and the constants `VALID_SEARCH_TYPES`/`VALID_DOMAINS` that the post-processing code already enforces. Use this as the concrete schema (write the prompt in the same style as `SUMMARIZER_SYSTEM_PROMPT` in `summarizer.py` — numbered rules, then an explicit output schema):

```text
Required JSON schema:
{
  "query_type": "<factual|comparative|analytical|exploratory>",
  "query_scope": "<narrow|broad>",
  "dominant_domain": "<machine_learning|software|philosophy|economics|science|general>",
  "sub_questions": [
    {
      "id": 1,
      "question": "<specific, search-ready sub-question, not a topic label>",
      "axis": "<definition|mechanism|application|criticism|comparison|...>",
      "search_type": "<encyclopedia|academic|statistical|news|comparison>",
      "priority": 1,
      "depends_on": [],
      "coverage_goal": "<what this sub-question should establish>",
      "domain": "<same enum as dominant_domain>"
    }
  ],
  "coverage_note": "<one sentence: what would full coverage of this query require>"
}
Generate 3-5 sub_questions. Cover at least 2 distinct axes. If
critique_feedback is provided, generate sub_questions that specifically
close the gaps it describes rather than repeating the original plan.
```

Definition of done: the `NameError` is gone; `planner_agent()` for a non-`what is/define/explain` query calls the LLM and returns sub-questions that are not identical to `fallback_plan()`'s fixed 4 items.

Verification:

```bash
python -c "
import asyncio
from app.core.config import get_settings
from app.core.llm import LLMClient
from app.agents.planner import planner_agent, fallback_plan

async def main():
    llm = LLMClient(get_settings())
    result = await planner_agent(llm, 'Compare the economics of nuclear vs solar energy in Bangladesh')
    fallback = fallback_plan('Compare the economics of nuclear vs solar energy in Bangladesh')
    assert result != fallback, 'Planner is still falling back to the fixed template'
    print('OK — planner used the LLM:', [q['question'] for q in result])

asyncio.run(main())
"
```

### 0.2 — Fix BUG-2: reconcile the Critic's prompt and parsing schema

File: `app/agents/critic.py`. The code that parses the response only reads `is_sufficient` / `reason` / `improved_queries` / `confidence` — keep that as the contract for now (Phase 2 task 2.4 replaces this whole scoring approach with the Confidence Engine, so don't over-invest here). Rewrite `CRITIC_SYSTEM_PROMPT` to describe exactly that schema instead of the unused checks/issues/suggested_queries schema. Delete the dead `CRITIC_USER_TEMPLATE` (or wire it into `critic_agent()` if you'd rather use string formatting instead of the inline f-string — either is fine, just don't leave it unused).

Definition of done: the system prompt and the actual parsed schema match; no dead template left behind.

Verification: `grep -n "CRITIC_USER_TEMPLATE" app/agents/critic.py` returns either zero matches (deleted) or a match inside `critic_agent()` (wired in) — not both defined-and-unused.

### 0.3 — Fix BUG-3: stop the `RUNTIME_STATE` memory leak

File: `app/api/routes.py`. Minimal safe fix: after reading `final_state` (line ~107), remove the entry:

```python
final_state: Dict[str, Any] = RUNTIME_STATE.pop(request_id, {})
```

Also pop it in the `except` branch before `return`, so a failed run doesn't leak either. This is the Phase-0-safe fix. Note for Phase 1: the better long-term fix is to stop using a global mutable dict entirely — capture the last snapshot in a local variable in `event_stream()` and pass it forward, since each request already runs in its own coroutine. Leave a `# TODO(phase-1): replace RUNTIME_STATE global with local snapshot capture` comment for task 1.x if you don't do the full refactor now.

Definition of done: `RUNTIME_STATE` never grows unboundedly across requests — verify by running 20 sequential research requests and confirming `len(RUNTIME_STATE)` stays at 0 or 1 between requests, not 20.

### 0.4 — Fix GAP-10: untrack `research.db`

```bash
git rm --cached research.db
git commit -m "[0.4] stop tracking research.db (already gitignored)"
```

Definition of done: `git status` no longer shows `research.db` as tracked; the file still exists locally (untracked) and the app still writes to it fine.

### 0.5 — Verify GAP-5's dependency: `duckduckgo-search`

```bash
pip index versions duckduckgo-search 2>&1 || true
python -c "from duckduckgo_search import DDGS; print('OK')"
```

If the import fails or the package is deprecated in favor of `ddgs`, migrate: update `requirements.txt` (`ddgs` instead of `duckduckgo-search`) and the import line in `app/agents/search.py` (`from ddgs import DDGS`) — the API is a drop-in match at the time this manual was written, but confirm against the installed version's actual interface before committing.

Definition of done: `_ddg_text` and `_ddg_news` in `search.py` return results in a fresh venv install, not just one that happened to have an old cached version installed.

---

## Phase 1 — Tier 0 Foundation (reliability & guardrails)

Branch: `phase-1-foundation`. Nothing in Phase 2+ should be built on top of a pipeline that still fails silently, leaks memory, or has no tests. Add to `requirements.txt` as you go: `tenacity`, `pydantic-settings`, `diskcache`, `structlog`, `slowapi`, `pytest`, `pytest-asyncio`, `respx` (for mocking httpx in tests).

### 1.1 — Retry + circuit breaker around LLM calls

File: `app/core/llm.py`. Wrap `_call_groq` and `_call_huggingface` with `tenacity.retry` (exponential backoff + jitter, 2-3 attempts, respecting `settings.llm_timeout_sec` as the per-attempt timeout — don't let retries multiply the total wait past a sane ceiling). Add a small circuit breaker: a module-level or `LLMClient`-level failure counter with a cooldown window (e.g. after 3 consecutive Groq failures, skip straight to the HuggingFace fallback for the next 60 seconds instead of trying Groq first every time). Keep the existing try/except fallback structure in `_generate_with_fallback` — just make each leg smarter, don't replace the fallback pattern.

Verification: unit test in `tests/test_llm_client.py` using `respx` to mock a failing-then-succeeding Groq endpoint and assert the retry succeeds; mock a persistently-failing Groq and assert HF fallback is used without exhausting all retries against Groq first once the breaker is open.

### 1.2 — Structured, validated LLM outputs

New file: `app/core/schemas.py`. Define Pydantic models mirroring what each agent already expects to parse: `PlannerOutputModel`, `SummarizerFactsModel`, `CriticVerdictModel`, `SynthesizerAnswerModel`. In `LLMClient.generate_json`, add an optional `response_model: Type[BaseModel] | None` parameter; when provided, validate the parsed JSON against it and raise (triggering the existing retry loop) on validation failure instead of only checking "is this valid JSON at all." Update each agent (`planner.py`, `summarizer.py`, `critic.py`, `synthesizer.py`) to pass its corresponding model.

Definition of done: a deliberately malformed mock LLM response (e.g. missing `sub_questions` key) is rejected and retried, not silently accepted as an empty plan.

### 1.3 — Per-request timeout

File: `app/api/routes.py`. Add `RESEARCH_TIMEOUT_SEC` to `Settings` (default 90). Wrap the `async for snapshot in workflow.astream(...)` loop with `asyncio.wait_for` or `asyncio.timeout()` (Python 3.11+; since the repo pins nothing above what ships with 3.10+, check the Python version in use before choosing the API — `asyncio.wait_for` works on 3.10). On timeout, emit an `error` event with a clear message, `pop` the `RUNTIME_STATE` entry, and return — don't leave the connection hanging.

Definition of done: a query that would loop past the timeout (simulate by temporarily setting `RESEARCH_TIMEOUT_SEC=1`) produces a clean `error` NDJSON event instead of a hung connection.

### 1.4 — Bounded caching

New file: `app/core/cache.py` wrapping `diskcache.Cache` with a size limit (e.g. `size_limit=250_000_000`, 250MB) and a helper for TTL'd get/set. Wire in two places:
- `app/agents/search.py`: cache each sub-question's ranked results keyed by a normalized-query hash, short TTL (e.g. 1 hour) — repeated sub-questions across runs are common.
- `app/agents/summarizer.py`: cache extracted facts keyed by `(source_url_hash, content_hash, prompt_version)` so retries and duplicate content don't re-spend LLM tokens.

Add a `PROMPT_VERSION` constant near each agent's system prompt so cache keys naturally invalidate when you edit a prompt later.

Definition of done: running the same query twice in a row is measurably faster on the second run (log cache hit/miss counts) and the disk cache directory stays under its size limit under repeated load.

### 1.5 — SQLite WAL mode + schema versioning

File: `app/db/sqlite.py`. In `init_db`, add `PRAGMA journal_mode=WAL;` and `PRAGMA busy_timeout=5000;` before creating tables. Add a `schema_version` table (`version INTEGER`) so Phase 2's schema extension (task 2.7) can check "have I already added the new tables" idempotently instead of relying only on `CREATE TABLE IF NOT EXISTS` scattered everywhere.

Definition of done: `sqlite3 research.db "PRAGMA journal_mode;"` reports `wal`; concurrent read+write from two connections doesn't error with "database is locked" under a quick manual test.

### 1.6 — Structured logging

New file: `app/core/logging.py` configuring `structlog` for JSON output. Generate a `request_id` in `routes.py` (already exists — `str(uuid.uuid4())`) and thread it through every agent call, either as an explicit parameter or via `structlog.contextvars.bind_contextvars(request_id=request_id)` at the top of `event_stream()`. Replace the ad hoc `logging.getLogger(__name__)` calls in `planner.py` (and add equivalents in the other agents, which currently have none) with the structured logger.

Definition of done: a single research run's logs can be filtered by `request_id` and show the full planner→search→summarize→critic trace for that run.

### 1.7 — Rate limiting

File: `main.py`. Add `slowapi`, configure a limiter (e.g. `5/minute` per IP by default, overridable via a `RATE_LIMIT` env var), apply it to the `/api/research/stream` route specifically — this is the expensive one.

Definition of done: firing 6 requests in under a minute from the same client gets a 429 on the 6th.

### 1.8 — Tests

New `tests/` directory with `pytest` + `pytest-asyncio`. Minimum coverage for this phase:
- `tests/test_evidence_utils.py` — pure functions (`source_reliability_score`, `dedupe_semantic_facts`, `filter_facts_by_domain`), no mocking needed, fast and high-value.
- `tests/test_planner.py` — using a mocked `LLMClient` (respx or a hand-rolled fake), assert `planner_agent` returns LLM-driven output for non-trivial queries (this is the regression test for BUG-1 — it must never silently pass again).
- `tests/test_llm_client.py` — retry/circuit-breaker behavior from 1.1.
- `tests/test_api_health.py` — `httpx.AsyncClient` hitting `/api/health`.

Definition of done: `pytest tests/ -v` passes; add a `pytest` line to a simple CI workflow (`.github/workflows/tests.yml`) if one doesn't exist, running on push.

### 1.9 — Settings hardening

File: `app/core/config.py`. Replace the hand-rolled `@dataclass` + manual `os.getenv` calls with `pydantic_settings.BaseSettings`, which gives you automatic `.env` loading, type coercion, and validation. Add a model validator that raises at startup if neither `groq_api_key` nor `huggingface_api_key` is set (fail fast, matching the existing runtime error in `llm.py` but surfaced before the server even starts accepting requests). Add the new settings introduced in this phase (`RESEARCH_TIMEOUT_SEC`, cache size/TTL, rate limit) here too.

Definition of done: starting the app with no LLM key configured fails immediately with a clear message, not on the first request.

---

## Phase 2 — Tier 1 features

Branch: `phase-2-tier1`. Implements Features 01, 04, 07, 08, 10, 11, 12 from `MARS-vision-v2.md` (Adaptive Orchestrator, Isolated Agent Context, Verification Layer, Critic/Red Team, Confidence Engine, Dynamic Research Depth, Cost/Token Governor), plus Delegation Contracts, the first slice of the Research Memory schema, and the durable event log Feature 14 (Full Observability) needs beyond live logging. Build 3-4 tasks at a time, not all at once, and keep the Phase 1 test suite green throughout.

Architecture note — "parallel agent missions" vs "parallel searches": `MARS-vision-v2.md` Feature 03 is explicit that the target extends parallel searches into parallel **agent missions**, not just differently-prompted calls sharing one pipeline. Task 3.1 (Phase 3) starts by routing specialist prompts inside the existing `search_node`/`summarizer_node` as an MVP — that's an acceptable starting architecture, but it does not by itself satisfy Feature 02/03. It only satisfies them once each specialist path also has the isolation properties from task 2.9 below (own contract, own scoped context, own stop condition, own tool permissions) — at that point a "specialist" is a logically independent mission that happens to share infrastructure, not just a different prompt string. Do not report Feature 02/03 as complete based on 3.1 alone.

### 2.1 — Adaptive Orchestrator with hardware-capped concurrency

New file: `app/agents/orchestrator.py`. Implement a complexity scorer — start heuristic, not another LLM call: word count, presence of comparative/superlative language ("compare", "should", "vs", "best"), number of distinct entities/domains implied, presence of numeric/time-bound scope ("over 20 years"). Map the score to `low | medium | high | very_high`, then to a target agent/sub-question count, **clamped by a new `MAX_PARALLEL_AGENTS` setting (default 3)** regardless of what the raw score suggests — this is the hardware guardrail from `MARS-vision-v2.md` Feature 01. `very_high` complexity should require an explicit `deep_research: true` flag on the request to exceed the default cap; without it, clamp silently and note the clamp in the `plan` NDJSON event's metadata.

Definition of done: a deliberately "very high complexity" query without `deep_research: true` still runs with ≤3 concurrent sub-question searches; the plan event indicates when clamping occurred.

### 2.2 — Cost / Token Governor

New file: `app/core/budget.py`. Track approximate token usage per LLM call (character count / 4 is an acceptable estimate if the provider response doesn't include a usage field — check Groq's response payload for a `usage` block first and prefer real numbers over the estimate when available). Accumulate a running cost estimate per `request_id` using a configurable per-model $/token rate (`GROQ_COST_PER_1K_TOKENS` etc. in settings, since these change — don't hardcode). Add a `RESEARCH_MAX_COST_USD` setting (default something conservative, e.g. $0.50) and stop the critic loop early (route straight to synthesizer) if it's exceeded, same as hitting `MAX_ITERATIONS`. Emit a new `budget` NDJSON event type (`{"type": "budget", "estimated_cost": 0.12, "limit": 0.50}`) after each LLM call so the frontend can show it live.

Definition of done: setting an artificially low `RESEARCH_MAX_COST_USD` causes the pipeline to stop early and still produce a report, with a limitations note explaining the budget cutoff (reuse the pattern already in `build_markdown_report`'s limitations list).

### 2.3 — Verification Agent

New file: `app/agents/verifier.py`. For each `(claim, source)` pair coming out of the summarizer, check whether the claim's key terms actually appear in the source's fetched content or snippet (you already have `content`/`snippet` from `search.py` — use it; this doesn't require a new fetch). Combine that lexical-overlap check with the existing `source_reliability_score` to produce a `verified: bool` and `verification_reason: str` per fact. Add this as a new graph node between `summarizer` and `critic` in `workflow.py` (`graph.add_node("verifier", verifier_node)`, rewire the edges: `summarizer → verifier → critic`). Facts that fail verification should still be kept in state (for transparency) but excluded from what the synthesizer treats as usable evidence.

Definition of done: a fact whose claim text has near-zero lexical overlap with its cited source's content is flagged `verified: false`; the synthesizer only uses `verified: true` facts.

### 2.4 — Confidence Engine

New file: `app/core/confidence.py`. Replace the inline `overall_conf` formula currently living in `critic_node` (`app/graph/workflow.py`) with a proper multi-signal function implementing the breakdown from `MARS-vision-v2.md` Feature 10: source quality (already have `source_reliability_score`), source diversity (distinct domains / total facts), citation coverage (verified facts / total facts, from 2.3), cross-source agreement (you'll need a simple signal here — e.g. how many distinct sources support the same claim cluster after `dedupe_semantic_facts`), freshness (best-effort — you don't currently capture publish dates; either add a `published_at` best-guess extraction in `search.py`'s content fetch, or explicitly mark freshness as "not yet measured" and weight it at 0 rather than faking a number), claim verification (from 2.3), critic survival (did the critic pass it without a loop). Return both the overall score and the per-signal breakdown so the frontend can eventually render the confidence breakdown UI from the vision doc.

Definition of done: `critic_node` calls `compute_confidence(...)` instead of the inline formula; the per-signal breakdown is available in the state for later UI work.

### 2.5 — Critic → Red Team upgrade

File: `app/agents/critic.py` (already fixed in 0.2). Extend the (now-consistent) critic prompt with 2-3 explicit adversarial questions from the Red Team list in `MARS-vision-v2.md` Feature 08 ("what assumption is weakest," "what would invalidate this conclusion") appended to the existing evaluation criteria, and have the model's `reason` field explicitly address them when `is_sufficient` is false. This is a prompt change, not a new agent — don't build a separate Critic and RedTeam agent, extend the existing one.

Definition of done: critic `reason` text for a rejected research run names a specific weak assumption or missing alternative explanation, not just a generic "insufficient coverage."

### 2.6 — Delegation Contracts

The planner (fixed in 0.1) already emits `question/axis/search_type/priority/depends_on/coverage_goal/domain` per sub-question — that's most of a delegation contract already. Formalize it: make `SubQuestion` (already a `TypedDict` in `planner.py`) the single contract type imported by `search.py` and `summarizer.py` instead of each accessing raw dict keys ad hoc. Add `minimum_sources: int` and `stop_condition: str` fields to match the contract shape in the vision doc, defaulting to sane values (`minimum_sources=2`, `stop_condition="sufficient evidence for this axis"`) and validated via `PlannerOutputModel` from 1.2.

Definition of done: every sub-question flowing through the pipeline has a fully-typed contract, checked by `PlannerOutputModel`, not a loosely-shaped dict.

### 2.7 — Research Memory schema extension (first slice)

File: `app/db/sqlite.py`. Add tables, additive only — keep `research_reports` for backward compatibility with anything already reading it:

```sql
CREATE TABLE IF NOT EXISTS research_runs (
    id TEXT PRIMARY KEY,            -- the request_id, reused as run id
    query TEXT NOT NULL,
    complexity TEXT,
    agent_count INTEGER,
    status TEXT NOT NULL,           -- running | completed | failed | timeout
    estimated_cost REAL,
    confidence REAL,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS agent_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES research_runs(id),
    question TEXT NOT NULL,
    axis TEXT,
    search_type TEXT,
    priority INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES research_runs(id),
    url TEXT NOT NULL,
    reliability_score REAL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES research_runs(id),
    claim TEXT NOT NULL,
    source_url TEXT NOT NULL,
    confidence REAL,
    verified INTEGER,               -- 0/1, from 2.3
    created_at TEXT NOT NULL
);
```

Write to these incrementally as the workflow runs (a row in `research_runs` at start, `agent_tasks` after planning, `sources`/`claims` after summarize+verify), not only at the very end — this is what makes Research Replay (Phase 3) possible later without redesigning persistence again.

Definition of done: after a research run, all four tables have rows joinable by `run_id`; `research_reports` still works unchanged for any existing readers.

Note: this is the first Research Memory slice, not the complete model from `MARS-vision-v2.md` Section 29. It deliberately excludes evidence (distinct from claims), per-iteration critic reviews, decisions, and final-report metadata as separate durable rows. Task 3.8 extends this schema to close that gap. Do not consider Feature 06 (Research Memory) fully implemented until 3.8 is done.

### 2.8 — Dynamic Research Depth / Adaptive Expansion

New file: `app/core/depth_controller.py`. Implements Feature 11 explicitly. This is distinct from both the existing iteration loop (`route_after_critic` in `workflow.py`, which only checks `is_sufficient` or the `MAX_ITERATIONS` ceiling) and the Cost Governor (2.2, which only checks budget) — Dynamic Research Depth is the decision, made every iteration, of whether to expand, contract, or stop, based on evidence rather than iteration count alone. Depends on 2.2, 2.4, and 2.6 already being in place.

Define and implement explicitly:

```text
Spawn trigger — a new sub-question is spawned when the critic
  identifies a specific coverage gap (already returns
  improved_queries) AND axis coverage from the Confidence Engine's
  diversity signal (2.4) is below threshold (fewer than 2 axes
  covered, or one axis > 60% of facts).

Evidence-sufficiency threshold — sufficient when the Confidence
  Engine's overall score (2.4) >= a configurable SUFFICIENCY_THRESHOLD
  (default 0.75) AND every planned axis has at least minimum_sources
  (from the Delegation Contract, 2.6) verified facts.

Coverage-gap detection — compare the set of `axis` values covered by
  *verified* facts against the axes the planner originally scoped
  (PlannerOutput.sub_questions[].axis); any axis with zero or
  below-threshold verified facts is a gap.

Marginal information gain — track confidence delta between this
  iteration and the previous one. If delta < MIN_MARGINAL_GAIN
  (default 0.03) for two consecutive iterations, stop even if
  MAX_ITERATIONS and budget both allow more. This is the actual
  "know when to stop" signal an iteration-count ceiling alone can't
  give you.

Confidence/budget interaction — if the Cost Governor (2.2) reports
  remaining budget below a safety margin, disable expansion (no new
  sub-questions) even when evidence is insufficient; degrade to
  "finalize with limitations" instead of exceeding budget.

Maximum recursion/depth — a hard MAX_RESEARCH_DEPTH setting (default
  = MAX_ITERATIONS, tracked separately since depth != iteration count
  once expansion can add sub-questions mid-run).

Early stop — the pipeline must be able to reach `synthesizer` before
  MAX_ITERATIONS when sufficiency + low marginal gain both hold. This
  is currently impossible: route_after_critic only checks two
  conditions today.
```

Wire `depth_controller.decide(state) -> Literal["expand", "finalize"]` into `route_after_critic` in `workflow.py`, replacing the current two-condition check with a call into this module.

Definition of done: a query that reaches high confidence with clearly diminishing returns (assert on confidence deltas across iterations in a test) finalizes before `MAX_ITERATIONS`, and the report's limitations section states this was an early stop on marginal gain, not a forced ceiling.

### 2.9 — Agent Context Isolation

New file: `app/core/isolation.py` (or fold into `orchestrator.py` from 2.1). Implements Feature 04 explicitly — currently only an implicit side effect of Delegation Contracts (2.6), not an enforced boundary. Depends on 2.6.

Rules to enforce for every agent invocation (search, summarizer, specialist variants added in 3.1, verifier):

```text
- Each agent call receives only its own contract (question / axis /
  search_type / priority / coverage_goal / domain / minimum_sources /
  stop_condition) — never the full ResearchState or another
  sub-question's raw search results/content.
- Cross-agent information enters only through the structured
  facts/claims list after summarization + verification — never as raw
  source text passed between sub-question workers.
- Each agent's tool access is explicit and scoped (e.g. a Financial
  specialist's allowed source domains, from 3.1, is enforced here —
  not just suggested by prompt wording).
- Raw fetched content (SearchResult.content in search.py) is
  discarded once that sub-question's summarization completes — it
  must not persist in shared state past that point (this also
  reinforces the Phase 0/1 memory guardrails already in this manual).
- Each agent enforces its own stop_condition independently — a
  specialist finishing early because its contract's minimum_sources
  is met is not blocked or extended by another agent's state.
```

Implement a lightweight `AgentContext` dataclass/pydantic model that is the *only* thing passed into `search_node`/`summarizer_node`/specialist functions — not the full `ResearchState`. Refactor the relevant call sites in `workflow.py` to construct one `AgentContext` per sub-question.

Definition of done: a unit test asserts a specialist function's call signature only receives its own `AgentContext`, never the full `ResearchState` or another sub-question's results; a second test confirms raw `content` fields never reach `claims`/`sources` beyond the single summarization call that used them.

### 2.10 — Full Observability Persistence (agent_events)

Extends 1.6 (structured logging) and 2.7 (memory schema) — live logs alone aren't a durable, queryable history, which is what Feature 14 and the Phase 3.4 Research Replay endpoint actually need.

Add to `app/db/sqlite.py`:

```sql
CREATE TABLE IF NOT EXISTS agent_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES research_runs(id),
    node TEXT NOT NULL,             -- planner | search | summarizer |
                                     -- verifier | critic | synthesizer
    event_type TEXT NOT NULL,       -- start | end | retry | failure |
                                     -- tool_call | budget | verification
    payload TEXT,                   -- JSON: model used, tokens, retry
                                     -- count, error message, etc.
    started_at TEXT,
    ended_at TEXT
);
```

Emit a row from every graph node's start and end (wrap each node function in `workflow.py` with a small `record_event(run_id, node, event_type, payload)` helper), every LLM retry/failure from 1.1's circuit breaker, every budget check from 2.2, and every verification decision from 2.3. This is the table the `/trace` endpoint (3.4) actually needs to read from — task 3.4 is updated below to join `agent_events` with `agent_tasks`/`sources`/`claims`, not just the latter three, or the trace is missing tool calls, retries, failures, prompt/model metadata, token usage, and node timing.

Definition of done: after one research run, `agent_events` contains a chronological row for every node transition, every LLM retry (if any occurred), and every budget check; `/api/research/{run_id}/trace` (3.4) includes all of it, not only the plan/sources/claims summary.

---

## Phase 3 — Tier 2 features

Branch: `phase-3-tier2`. Implements Features 02 (full specialist roster), 09 (Contradiction Engine), 13 (Durable Execution), 15 (Research Replay), 18 (Executive Decision Intelligence, completed via the expanded 3.5 below), the Evidence Explorer / claim inspector UI (Section 25 of the vision doc), Research Modes (Section 28), and the remaining Research Memory tables (Feature 06, completed via 3.8) that 2.7 deliberately deferred. Only start this phase once Phase 2's tests are green and you've run at least a handful of real research queries against the Phase 2 build without regressions.

### 3.1 — Specialized Agent Workforce (expand beyond generic search)

The current pipeline treats every sub-question identically regardless of domain. Add 2-3 specialist prompt variants first (Financial, Technical, Market — pick based on what queries you actually see, not the full 7-role roster at once), each a thin wrapper around the existing `summarizer_agent` with a domain-specific system prompt addition (e.g. "prefer primary financial filings and central-bank data over news summaries") selected by the `domain` field already present in each sub-question's delegation contract. Don't build a new orchestration layer for this — route by `domain` in the existing `search_node`/`summarizer_node`.

Architecture requirement (resolves a target/manual inconsistency): `MARS-vision-v2.md` Feature 03 requires parallel *agent missions*, not just parallel searches wearing different prompts. Routing by `domain` inside the shared `search_node`/`summarizer_node` is an acceptable MVP implementation **only if** each specialist path also applies the isolation rules from task 2.9 (own `AgentContext`, own tool/domain permissions, own stop condition). At that point each specialist is a logically independent mission that happens to share infrastructure, which satisfies the target. A specialist that is merely a different prompt string sharing full state with every other sub-question does not satisfy Feature 02/03.

Definition of done: a financial sub-question and a general sub-question produce measurably different source preferences (check via `provider`/domain distribution in the resulting facts), **and** each specialist path is provably using its own `AgentContext` (2.9), not the shared `ResearchState` — verify by asserting the specialist function's call signature never receives the full state object.

### 3.2 — Contradiction Engine

New file: `app/core/contradictions.py`. After `dedupe_semantic_facts` groups near-duplicate claims, add a pass that instead flags claims which are *topically* similar (same entity/axis) but *numerically or directionally* different (e.g. two claims both about "market growth rate" with different percentages). Simple heuristic to start: extract numbers via regex from claims sharing a semantic-similarity band just below the dedup merge threshold (i.e. similar enough to be about the same thing, not similar enough to be the same claim), and surface them as a `contradictions` list in state, each with both claims + both sources. Feed this into the critic's evaluation (contradiction acknowledged vs. silently dropped) and into the final report as an explicit section.

Definition of done: a query with genuinely conflicting sources (test with something like population or GDP figures which vary by source) produces at least one entry in the contradictions list, visible in the final report.

### 3.3 — Durable Checkpointing

Use the `research_runs`/`agent_tasks`/`sources`/`claims` tables from 2.7 as the checkpoint store — don't add a separate checkpoint mechanism. After each graph node completes, persist its output to the relevant table (this should already be mostly true from 2.7; this task is about making resume actually work). Add a `resume_research(run_id)` path: if a `research_runs` row has `status='failed'` or `status='timeout'`, a resume request should rebuild `ResearchState` from the persisted rows instead of starting from an empty state, and re-enter the graph at the appropriate node (skip `planner`/`search` if `agent_tasks`/`sources` already exist for that run). Add `POST /api/research/{run_id}/resume` to `routes.py`.

Definition of done: killing the server mid-run (simulate with a manual process kill during a slow query) and then calling the resume endpoint continues from persisted state instead of re-running the whole pipeline.

### 3.4 — Research Replay

`GET /api/research/{run_id}/trace` in `routes.py`: read `agent_events` (2.10), `agent_tasks`, `sources`, and `claims` for the run and return them in chronological order as a single JSON structure (node start/end times and retries → plan → sources found per sub-question → claims extracted → verification results → critique iterations → budget checks → final confidence). Joining `agent_events` in is required, not optional — without it the trace is missing tool calls, retries, failures, prompt/model metadata, and token usage, which is what "full observability" (Feature 14) actually promises. This is a read-only reconstruction from the Phase 2/3 persistence — no new write path needed.

Definition of done: the endpoint returns a complete, ordered trace for any completed `run_id` — including retries/failures if any occurred — sufficient to answer "why did this report reach this confidence score," not just "what was the final answer."

### 3.5 — Executive Report Builder (Decision Intelligence Layer)

Extend `build_markdown_report` in `workflow.py` (or extract it into `app/core/report_builder.py` if it's grown unwieldy) to match the Executive Report structure from `MARS-vision-v2.md` Section 26: Executive Summary, Key Findings (bullet list derived from the highest-confidence, verified claims), Evidence, Contradictions (from 3.2), Risks/Limitations (already exists), Confidence (already exists, now with the Phase 2.4 breakdown), Sources. Keep the existing "Final Answer / Supporting Evidence / Limitations / Confidence Score" structure as the baseline and add sections around it rather than rewriting from scratch — the frontend's `extractFinalAnswer` in `App.jsx` parses by markdown heading text, so preserve the `# Final Answer` heading exactly or update the frontend parser in the same commit.

That reporting shell is not the same thing as Feature 18 (Executive Decision Intelligence). `MARS-vision-v2.md` Section 21 is explicit that Options → Recommendation → Rationale are the differentiating layer, not cosmetic formatting — it shows a concrete Option A/B/C example with one recommended option and a stated reason. Implement this as a distinct `build_decision_layer(state) -> List[DecisionOption]` function, required alongside the descriptive sections above:

```text
- Strategic Options — derive 2-4 named options (A/B/C...) from the
  synthesized findings and contradictions. For purely factual/
  definitional queries this may collapse to a single "no material
  decision" option, which is fine. For comparative/analytical queries
  (detected via the Orchestrator's query_type from 2.1) it must
  produce genuine alternatives, not the same answer restated three
  ways.
- Option comparison — each option gets an explicit trade-off
  statement, not just a label.
- Recommendation — exactly one option marked is_recommended, chosen
  by an explicit, code-level rule (e.g. highest expected value under
  the Confidence Engine's signals, or lowest downside risk when
  overall confidence is low) — don't leave the choice to unstructured
  LLM judgment alone.
- Evidence-backed rationale — the rationale text must reference
  specific verified claims (2.3) that support the choice, not generic
  language.
- Downside/risk per option — at minimum one risk statement per
  option, pulled from contradictions (3.2) or low-confidence axes
  where relevant.
- Confidence in the recommendation itself — distinct from the
  overall research confidence score. This is "how sure are we THIS
  option is right," which can be lower than evidence-quality
  confidence even when the underlying facts are solid (e.g.
  high-confidence facts, but a genuinely close call between options).
- Evidence vs. judgment separation — the report must structurally
  separate "what the evidence shows" (Key Findings/Evidence) from
  "what we recommend and why" (Decision Layer). Never blend a
  recommendation into the findings section as if it were itself a
  fact.
```

Persist each option as a row in `decisions` (3.8).

Definition of done: a comparative/analytical query (e.g. "Should Bangladesh invest in nuclear vs. solar energy?") produces a report with 2+ distinct, genuinely different strategic options, exactly one marked recommended, a rationale that references specific verified claims, and a risk note per option — not just a single-paragraph synthesized answer with a findings list appended. A purely factual query still produces all the descriptive sections with the Decision Layer correctly collapsed to "no material decision," not broken or omitted.

### 3.6 — Evidence Explorer (claim inspector) UI

New component `ui/src/components/ClaimInspector.jsx`. A simple detail panel — no graph rendering — showing, for a selected finding: the claim, its source(s), the verification result from 2.3, and the confidence breakdown from 2.4 (matching the list/tree format in `MARS-vision-v2.md` Section 25, not a rendered graph). Wire it to open when a user clicks a finding card in the existing `App.jsx` findings grid. Needs a new API response shape or reuse of the `/trace` endpoint from 3.4 filtered to one claim.

Definition of done: clicking a finding card shows its full verification/confidence detail without a page reload.

### 3.7 — Research Modes

Add a `mode` field to `ResearchRequest` in `routes.py` (`quick | standard | deep`, default `standard`), each mapping to a preset `(max_agents, max_iterations, deep_research_flag)` tuple consumed by the orchestrator (2.1) and iteration logic. Surface as a mode selector in the frontend next to the query input.

Definition of done: `mode: "quick"` visibly produces fewer agents/iterations than `mode: "deep"` for the same query, and `deep` is the only mode that can exceed the `MAX_PARALLEL_AGENTS` default cap from 2.1.

### 3.8 — Complete Research Memory Persistence

Closes the gap flagged after 2.7: `MARS-vision-v2.md` Section 29's full model includes evidence, critiques, and decisions as durable rows, not just claims and sources. Extend `app/db/sqlite.py` further, additive to 2.7/2.10:

```sql
CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES research_runs(id),
    source_id INTEGER REFERENCES sources(id),
    raw_snippet TEXT,                -- the underlying material a claim
                                      -- was derived from — distinct
                                      -- from the rewritten claim itself
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS critic_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES research_runs(id),
    iteration INTEGER NOT NULL,
    is_sufficient INTEGER,
    reason TEXT,
    confidence REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES research_runs(id),
    option_label TEXT NOT NULL,       -- "A", "B", "C"...
    description TEXT NOT NULL,
    is_recommended INTEGER,
    rationale TEXT,
    risk_note TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS final_reports (
    run_id TEXT PRIMARY KEY REFERENCES research_runs(id),
    report_markdown TEXT NOT NULL,
    confidence REAL,
    generated_at TEXT NOT NULL
);
```

Distinguish `evidence` from `claims` explicitly: a `claim` is the summarizer's rewritten, deduplicated assertion (already in the 2.7 schema); `evidence` is the raw material it was derived from, kept separately so Verification (2.3) and Audit research mode (3.7) can show "here is exactly what the model read," independent of how it was rewritten. Populate `critic_reviews` on **every** critic iteration, not just the final one — this is what makes Replay (3.4) show the actual back-and-forth, not only the outcome. Populate `decisions` in the same commit as 3.5 — one row per strategic option. Populate `final_reports` once, at `finalize_node`, keyed by `run_id`; keep writing `research_reports` too for backward compatibility, but treat `final_reports` as canonical going forward since it joins cleanly with everything else.

Definition of done: every table in the full memory model (`research_runs`, `agent_tasks`, `sources`, `claims`, `agent_events`, `evidence`, `critic_reviews`, `decisions`, `final_reports`) has real rows after one research run; a single query joining all of them on `run_id` reconstructs the complete mission end to end.

---

## Phase 4 — Tier 3 (optional, advanced)

Branch: `phase-4-tier3`. Only build these if the product is proving itself in real use after Phases 0-3. Treated more lightly here since they're genuinely optional — goals and entry points, not full task breakdowns.

```text
Evaluation Lab
  Goal: a fixed set of 15-20 test queries with expected coverage, run
  on demand (not nightly — you don't have infra for a scheduler yet),
  scoring confidence/citation-count/contradiction-detection trend over
  time. Entry point: new `scripts/run_eval.py` reading
  `tests/eval_queries.json`, hitting the real pipeline, writing results
  to a new `evaluation_runs` table.

Self-Diagnosis
  Goal: inspect agent_tasks/claims/sources across recent research_runs
  for patterns (a specialist consistently returning low-confidence
  facts, a search provider consistently failing). Entry point: a
  read-only analysis script over the Phase 2.7 tables, not a live
  agent — start as a report you run manually, not an autonomous loop.

Scenario Engine
  Goal: given a completed research run, spawn targeted research under
  explicit, named alternative assumptions and present a genuine
  comparison — not just re-answers of the same query with different
  wording. Must explicitly define:
    - Scenario assumptions: each scenario states its differing input
      assumption in one sentence (e.g. "EV adoption reaches 60% by
      2035" vs. "25% by 2035"), stored as data, not implicit in a
      reworded query.
    - Scenario-specific research questions: each scenario gets its
      own delegation-contract sub-questions (2.6), derived from the
      base plan but scoped to that assumption — not the identical
      sub-question list rerun with different phrasing.
    - Outputs/metrics compared: define which specific fields from the
      base report (confidence, key findings, recommended option from
      3.5) are placed side by side across scenarios in a comparison
      table, not three separate, unlinked reports.
    - Scenario-sensitive recommendation: state explicitly whether the
      Decision Layer's recommended option (3.5) changes across
      scenarios — if the same option wins in every scenario, say so
      ("robust across scenarios"); if it changes, name which
      assumption flips it.
    - Scenario impact on confidence: confidence should vary per
      scenario (thinner evidence under a less-studied assumption
      should show lower confidence, not the same score copy-pasted
      across scenarios).
    - Explicit distinction from a plain rerun: a scenario differs from
      resubmitting the query by (a) the stated assumption constraining
      sub-question generation and source selection, and (b) reusing
      the base run's already-verified, assumption-neutral facts (e.g.
      current battery costs) rather than re-researching everything
      from zero — only assumption-dependent claims are re-researched
      per scenario.
  Entry point: reuses the Phase 0-3 pipeline as a subroutine, called
  once per scenario with an assumption-scoped delegation contract set
  — not just a reworded top-level query.

Mission Control UI
  Goal: the full visual redesign from MARS-vision-v2.md Section 36.
  Do this last. It's a multi-week frontend project and depends on
  every API surface above already being stable (trace endpoint,
  budget events, confidence breakdown, contradictions) — building it
  earlier means redesigning it again once those land.

Multi-tenant (users/projects)
  Goal: only pursue this if there's an actual second user. Don't add
  auth/multi-tenancy speculatively — it's real scope (users table,
  session handling, per-user rate limits) with no payoff until needed.
```

---

## Appendix A — Full task checklist

```text
Phase 0 — Critical fixes
[ ] 0.1  Fix undefined PLANNER_SYSTEM_PROMPT
[ ] 0.2  Reconcile Critic prompt/schema mismatch
[ ] 0.3  Fix RUNTIME_STATE memory leak
[ ] 0.4  Untrack research.db from git
[ ] 0.5  Verify/migrate duckduckgo-search dependency

Phase 1 — Tier 0 Foundation
[ ] 1.1  Retry + circuit breaker on LLM calls
[ ] 1.2  Structured/validated LLM outputs (Pydantic schemas)
[ ] 1.3  Per-request timeout
[ ] 1.4  Bounded caching (search + summarizer)
[ ] 1.5  SQLite WAL mode + schema_version table
[ ] 1.6  Structured logging with request_id
[ ] 1.7  Rate limiting on /api/research/stream
[ ] 1.8  Test suite (pytest) — planner regression test is mandatory
[ ] 1.9  Settings hardening (pydantic-settings, fail-fast validation)

Phase 2 — Tier 1 features
[ ] 2.1  Adaptive Orchestrator + MAX_PARALLEL_AGENTS hardware cap
[ ] 2.2  Cost / Token Governor + budget NDJSON event
[ ] 2.3  Verification Agent (new graph node)
[ ] 2.4  Confidence Engine (multi-signal, replaces inline formula)
[ ] 2.5  Critic → Red Team prompt upgrade
[ ] 2.6  Delegation Contracts formalized via PlannerOutputModel
[ ] 2.7  Research Memory schema extension — FIRST SLICE ONLY
         (research_runs, agent_tasks, sources, claims); completed by 3.8
[ ] 2.8  Dynamic Research Depth / Adaptive Expansion (Feature 11)
[ ] 2.9  Agent Context Isolation (Feature 04)
[ ] 2.10 Full Observability Persistence — agent_events table (Feature 14)

Phase 3 — Tier 2 features
[ ] 3.1  Specialized Agent Workforce (2-3 roles to start) — requires 2.9
         isolation to actually satisfy Feature 02/03, not just prompts
[ ] 3.2  Contradiction Engine
[ ] 3.3  Durable Checkpointing + resume endpoint
[ ] 3.4  Research Replay (/trace endpoint) — now joins agent_events too
[ ] 3.5  Executive Report Builder + Decision Intelligence Layer
         (options / recommendation / rationale / risk — Feature 18)
[ ] 3.6  Evidence Explorer / claim inspector UI
[ ] 3.7  Research Modes (quick/standard/deep)
[ ] 3.8  Complete Research Memory Persistence (evidence, critic_reviews,
         decisions, final_reports) — closes out Feature 06

Phase 4 — Tier 3 (optional)
[ ] 4.1  Evaluation Lab
[ ] 4.2  Self-Diagnosis
[ ] 4.3  Scenario Engine — full spec: assumptions, scenario-scoped
         questions, compared metrics, scenario-sensitive recommendation
[ ] 4.4  Mission Control UI redesign
[ ] 4.5  Multi-tenant (only if actually needed)
```

---

## Appendix B — Verification commands cheat sheet

```bash
# Backend smoke test
curl -s http://127.0.0.1:8000/api/health

# Full pipeline test
curl -N -X POST http://127.0.0.1:8000/api/research/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare open-source speech-to-text models for CPU inference"}'

# Test suite (from Phase 1 onward)
pytest tests/ -v

# Check WAL mode (from Phase 1.5 onward)
sqlite3 research.db "PRAGMA journal_mode;"

# Check no runaway RUNTIME_STATE growth (from Phase 0.3 onward)
python -c "from app.graph.workflow import RUNTIME_STATE; print(len(RUNTIME_STATE))"

# Frontend build
cd ui && npm run build
```

---

## Appendix C — Guardrails (what not to do)

```text
- Do not add Redis, a message queue, or a self-hosted vector database.
  Nothing in this manual needs them; they cost RAM you don't have on an
  8GB host and duplicate work the LLM API or SQLite already does.
- Do not raise MAX_PARALLEL_AGENTS or MAX_PARALLEL_SEARCH above what
  you've actually load-tested on the target machine. Default to 3 and
  raise it only after measuring real memory headroom.
- Do not build the Mission Control UI (Phase 4.4) before Phases 0-3 are
  stable. It's a large frontend investment that depends on API surfaces
  that don't exist yet.
- Do not remove the deterministic fallback paths in planner.py,
  critic.py, or synthesizer.py when adding LLM-driven behavior on top of
  them — they're what keeps the pipeline answering something when the
  LLM call fails.
- Do not skip Phase 0 or Phase 1 to get to "the interesting features"
  faster. Every Phase 2+ feature assumes the pipeline doesn't leak
  memory, doesn't hang, and has tests catching regressions.
- Do not invent cost/latency numbers for the Cost Governor (2.2) or
  Evaluation Lab (4.1) — use real provider pricing and measured
  latencies, not the placeholder figures ($1.84, 2m 14s) from the
  vision document, which were illustrative only.
- Do not report Feature 02/03 (Specialized Agent Workforce / Parallel
  Execution) as complete based on 3.1 alone. They require the
  isolation properties from 2.9 — own AgentContext, own tool/domain
  permissions, own stop condition — to count as independent agent
  missions rather than shared-state prompt variants.
- Do not report Feature 06 (Research Memory) as complete based on 2.7
  alone. 2.7 is explicitly the first slice; 3.8 adds evidence,
  critic_reviews, decisions, and final_reports, which Replay, Resume,
  Audit, and the Decision Layer all depend on.
- Do not report Feature 14 (Full Observability) as complete based on
  structured logging (1.6) alone. Logs are live-only; 2.10's
  agent_events table is what makes the history durable and queryable
  by the /trace endpoint (3.4).
```
