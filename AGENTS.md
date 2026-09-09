# AGENTS.md — Operating Rules for AI Coding Agents

This file is the persistent rulebook for any AI coding agent (opencode or otherwise) working in this repository. It applies to **every** task, not just the phased roadmap in `manual.md`. Read it once at the start of a session and hold yourself to it for every change you make.

Two companion documents give context this file doesn't repeat:
- `MARS-vision-v2.md` — what the system should eventually do and why (feature rationale, UI mockups, data model).
- `manual.md` — what to build, in what order, with file-level tasks and acceptance criteria.

This file exists because the codebase already shipped real, silent bugs that a bit more discipline would have caught — see Section 2. Its job is to stop that from happening again, not to describe features.

---

## 1. Project snapshot

```text
Stack:      FastAPI + LangGraph (async Python 3.10+), SQLite (aiosqlite),
            React 18 + Vite 5 frontend, httpx for outbound calls.
Generation: remote LLM APIs only (Groq primary, HuggingFace fallback) —
            no local model inference.
Hardware:   target host has 8GB RAM. Concurrency and memory footprint
            are real constraints, not theoretical ones — see Section 5.

Entrypoint:      main.py (FastAPI app, lifespan init)
Agents:          app/agents/*.py (planner, search, summarizer, critic,
                 synthesizer, evidence_utils — one file per pipeline
                 stage or shared utility)
Orchestration:   app/graph/workflow.py (LangGraph StateGraph)
API:             app/api/routes.py (the /api/research/stream route)
Core utilities:  app/core/*.py (config, llm client, and whatever you add
                 for caching/budget/confidence/isolation per manual.md)
Persistence:     app/db/sqlite.py
Frontend:        ui/src/App.jsx consumes the NDJSON stream; components
                 in ui/src/components/
```

Commands:

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY at minimum
cd ui && npm install && cd ..

# Run
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
cd ui && npm run dev

# Test (once tests/ exists per manual.md 1.8 — add this the moment it does)
pytest tests/ -v

# Lint/format — repo has none configured yet. If you add ruff/black/mypy
# (recommended, not yet done), wire the exact commands here so future
# sessions don't have to rediscover them.

# Smoke test
curl -s http://127.0.0.1:8000/api/health
curl -N -X POST http://127.0.0.1:8000/api/research/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval augmented generation?"}'
```

---

## 2. Confirmed bug history (read this before touching agent code)

These are real bugs found by reading the code, not hypotheticals. Each one below is turned into a standing rule in Section 4. Keep this list updated — when you fix a bug or find a new one, add it here with the same format, so the next session doesn't reintroduce it.

```text
[FIXED — Phase 0 task 0.1, commit f225b3c] app/agents/planner.py
  PLANNER_SYSTEM_PROMPT was referenced but never defined. Every LLM
  planning call raised NameError, caught by a broad `except Exception`,
  silently falling back to a fixed 4-question template. The planner
  never used the LLM, for the entire history of the project, and
  nothing detected it. Root cause: no import-time check, no test, and
  an exception handler broad enough to hide a NameError.

[FIXED — Phase 0 task 0.2, commit d878d07] app/agents/critic.py
  CRITIC_SYSTEM_PROMPT described one JSON schema; the code that parsed
  the LLM's response read a different, incompatible schema. Root
  cause: the prompt and the parser were edited independently and never
  cross-checked against each other.

[FIXED — Phase 0 task 0.3; fully removed in Phase 1] app/graph/workflow.py, app/api/routes.py
  RUNTIME_STATE, a module-level dict keyed by request_id, was written
  to on every request and never cleaned up — an unbounded memory leak.
  Root cause: global mutable state used for what should have been a
  local/scoped value. Phase 1 replaced it with a local `last_snapshot`
  variable in event_stream(); the global no longer exists.

[FIXED — Phase 1 task 1.4] app/agents/search.py
  Multiple bare `except: return []` / `except: return ""` blocks
  swallowed all errors, including transient ones a retry would fix.
  Root cause: exception handling written for "never crash" without
  distinguishing expected-empty from actually-broken. All bare excepts
  now log with context; provider failures degrade to empty results
  instead of killing the run.

[FIXED — Phase 2 audit] app/core/config.py
  `SettingsConfigDict(env_file=(".env", ".env.example"))` — in
  pydantic-settings the LAST file wins, so the placeholder
  `your_groq_api_key_here` from .env.example silently overrode the real
  key from .env. Every LLM call 401'd and the pipeline "worked" via
  deterministic fallbacks, hiding the misconfiguration for two phases.
  Root cause: wrong env_file precedence plus fallbacks that mask
  provider auth failures. Fixed by putting .env last and adding a
  regression test. Secondary finding: the configured Groq model
  (llama-3.1-8b-instant) had been decommissioned upstream — always
  verify model IDs against the provider when auth succeeds but calls
  still fail.
```

If you find a new instance of any of these patterns anywhere in the codebase while working on something else, fix it or flag it in your commit message — don't leave it for later just because it's outside your current task's file scope.

---

## 3. Non-negotiable workflow rules

```text
1. One task = one commit, small and reviewable. Reference the manual.md
   task ID in the message when applicable: "[2.3] add Verification Agent".

2. Branch per phase/feature, off the default branch. Never force-push.
   Never rewrite already-pushed history.

3. Run the task's verification command before moving on. A task isn't
   done because the code was written — it's done because you checked
   it does what it claims.

4. Never commit .env, API keys, or database files with real data.
   research.db was already committed once despite being gitignored —
   check `git status` before every commit, not just the first one.

5. Every new setting goes through Settings (app/core/config.py) and
   gets a placeholder in .env.example. Never call os.getenv() ad hoc
   in a new module — it's how settings drift out of sync with what's
   documented.

6. If a task requires an architecture-level decision not covered by
   manual.md or MARS-vision-v2.md (e.g. changing the DB engine,
   removing an existing working code path, changing the public API
   contract in a breaking way), stop and surface the question instead
   of deciding unilaterally. Small implementation choices within a
   task's stated scope don't need escalation — make the smallest
   reasonable choice and note it in the commit message.

7. Do not remove or weaken an existing deterministic fallback path
   (planner.py, critic.py, synthesizer.py already have this pattern:
   try LLM → on failure, fall back to a rule-based default) when adding
   new LLM-driven behavior. Every new agent must have the same shape —
   see Section 6.
```

---

## 4. Bug-class prevention rules

Each rule below exists because of a specific, real bug (Section 2). Apply all of them to every change you make, not just new files.

### 4.1 — Never reference a constant/prompt without defining it in scope

The exact failure mode that broke the planner. Before finishing any task that adds or edits an agent module:

```bash
python -c "import app.agents.<module_you_touched>"
```

This alone would have caught the undefined `PLANNER_SYSTEM_PROMPT` immediately. Run it for every agent file you touch, every time — it costs one second and it is the single highest-value check in this repo.

### 4.2 — A system prompt's declared schema and the parser must match, always

If a `*_SYSTEM_PROMPT` documents an output JSON shape, the function that calls `llm.generate_json(...)` and reads `payload.get(...)` must read exactly those keys — not a different, more convenient shape written separately. Before considering any prompt "final":

```text
1. List every key the system prompt's schema promises.
2. Grep the parsing code in the same function for every .get("key")
   call it makes.
3. The two lists must be identical. If they're not, fix the prompt or
   the parser — don't leave both and hope the model guesses right.
```

If there's a separate `*_USER_TEMPLATE` constant, it must actually be used — a defined-but-unused template (as `CRITIC_USER_TEMPLATE` was) is a sign the prompt and the code were edited independently and drifted.

### 4.3 — No module-level mutable state without an explicit cleanup path

Any `dict`/`list`/`set` declared at module scope that gets written to per-request or per-run (keyed by `request_id`, `run_id`, etc.) is a leak waiting to happen. Prefer a local variable scoped to the request's coroutine. If module-level state is genuinely necessary, every write path must have a matching cleanup — `pop()` in a `finally` block, a TTL, or a bounded size with eviction. State that only grows is not acceptable at any scale on an 8GB host.

### 4.4 — No bare or silently-swallowing exception handlers

```text
Not acceptable:  except: return []
Not acceptable:  except Exception: return ""
Minimum bar:     except Exception as exc: logger.warning(..., exc_info=exc); return fallback_value
```

If a fallback is genuinely intended behavior (not just "don't crash"), say so in a comment the way `summarizer.py`'s "Heuristic fallback when the model output is malformed or empty" comment already does well — that's the standard to match, not the bare `except:` blocks in `search.py`'s `_wiki()`/`_fetch_content()`. A silent catch is exactly what let the planner bug survive undetected — every broad exception handler is a place a future bug can hide the same way.

### 4.5 — No dead code

If you define a constant, template, or function and don't call it in the same commit, either wire it in or delete it. Before finishing a task:

```bash
grep -rn "YOUR_NEW_NAME" app/ --include="*.py" | grep -v __pycache__
```

If it only appears once (the definition), it's dead. This is also how `CRITIC_USER_TEMPLATE` was caught.

### 4.6 — Concurrent by default for independent I/O

```text
Not acceptable:  for item in items: await do_io(item)   # when calls
                                                          # are independent
Preferred:       await asyncio.gather(*(do_io(item) for item in items))
```

`SearchClient.run_search` already does this correctly across sub-questions. `_search`'s top-3 content-fetch loop does not — don't copy that pattern into new code; fix it if you touch that function, and don't introduce the sequential version elsewhere.

### 4.7 — Every LLM-calling agent needs a deterministic fallback

Match the existing shape in `planner.py`/`summarizer.py`/`synthesizer.py`/`critic.py`:

```python
NAME_SYSTEM_PROMPT = """...""".strip()   # defined in this file, verified importable

async def name_agent(llm: LLMClient, ...) -> ...:
    try:
        payload = await llm.generate_json(NAME_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:
        logger.warning("[Name] LLM failed, using fallback", exc_info=exc)
        return deterministic_fallback(...)

    result = extract_and_validate(payload)
    if not result:
        logger.warning("[Name] empty/invalid LLM output, using fallback")
        return deterministic_fallback(...)
    return result
```

New agents added per `manual.md` (verifier, orchestrator, depth controller, etc.) must follow this shape. A new agent with no non-LLM fallback is a regression relative to the rest of the codebase, even if it "works" when the API is up.

### 4.8 — DB schema changes are additive and idempotent

`CREATE TABLE IF NOT EXISTS` only. Never drop or destructively alter an existing table (`research_reports` in particular must keep working for any existing reader). New tables get added, old ones stay.

### 4.9 — New NDJSON event types need a frontend case in the same commit

`app/api/routes.py`'s `event_line()` and `ui/src/App.jsx`'s `applyEvent()` switch are an implicit contract. `applyEvent` has a `default: break` — a backend event type with no matching `case` is silently dropped, not an error. If you add an event type (e.g. `budget`, per `manual.md` 2.2), add the frontend case in the same commit, or it will look like the feature does nothing.

---

## 5. Hardware-aware defaults

The target host has 8GB RAM and no local model inference — every constraint here is about orchestration memory and concurrency, not compute:

```text
- Default concurrency caps (MAX_PARALLEL_SEARCH, and MAX_PARALLEL_AGENTS
  once it exists) start low (2-3) and only get raised after you've
  actually measured memory headroom under load — not because a feature
  spec suggests a bigger number.
- Release raw fetched content (SearchResult.content) from memory once
  it's been summarized. Don't let full source text accumulate across a
  whole research run's state.
- Any new cache (diskcache or otherwise) must have an explicit size
  cap. An unbounded cache is a slower way to hit the same OOM as
  RUNTIME_STATE was causing.
- No Redis, no message queue, no self-hosted vector DB, no local
  embedding models. Nothing in this project's roadmap needs them, and
  they cost RAM this host doesn't have.
```

---

## 6. Definition of Done

Apply this checklist to every task, whether it's from `manual.md` or ad hoc:

```text
[ ] python -c "import <every module you touched>" succeeds
[ ] No bare `except:` / silent `except Exception: return <empty>`
    introduced — failures are logged with context
[ ] No new dead code (grep for anything you defined but never call)
[ ] Any new module-level mutable state has an explicit cleanup path
[ ] Any new LLM-calling function has a deterministic fallback (4.7)
[ ] Any new setting is in Settings + .env.example, not a raw os.getenv()
[ ] Any new DB table uses CREATE TABLE IF NOT EXISTS and doesn't break
    existing readers
[ ] Any new NDJSON event type has a matching case in ui/src/App.jsx
[ ] Independent I/O calls use asyncio.gather, not a sequential loop,
    unless there's a stated reason otherwise
[ ] Tests added/updated and passing (pytest tests/ -v, once tests/
    exists per manual.md 1.8 — add coverage for what you built)
[ ] Manual smoke test run against the real running server, not just
    unit tests in isolation
[ ] git status shows nothing unexpected (no .env, no *.db, no
    accidentally-included data) before committing
[ ] Commit message references the manual.md task ID if one applies
```

If any box can't be checked, the task isn't done — say so explicitly rather than reporting completion.

---

## 7. When unsure

```text
- Feature intent/rationale unclear -> check MARS-vision-v2.md first.
- Build order/what's next unclear -> check manual.md first.
- Whether something counts as "done" for a specific feature -> check
  manual.md's "Definition of done" for that task and Appendix C's
  guardrails — several features (Feature 02/03, Feature 06, Feature 14)
  have explicit rules about not being marked complete on a partial
  implementation.
- Still unclear, or the question is architectural rather than
  implementation-level -> stop and ask (Section 3, rule 6). Don't
  guess on anything that would be expensive to undo.
```
