# Architecture

How the v2 system fits together — components, data flow, and the reasoning
behind the design decisions that differ from a standard RAG chatbot.

---

## Pipeline

```mermaid
flowchart TD
    Q[User query + mode] --> IN[Intent<br/>ambiguity, senses, domain,<br/>explanation level]
    IN --> ORCH[Orchestrator<br/>complexity + plan targets]
    ORCH --> PL[Planner<br/>delegation contracts: axis, search_type,<br/>minimum_sources, wave, variants]
    PL -->|search-informed: one grounding search on the raw query| SE
    SE[Search<br/>Tavily / DuckDuckGo / Wikipedia / arXiv / Crossref<br/>circuit breakers, canonical-URL dedup,<br/>domain diversity caps, disk cache]
    SE --> SU[Summarizer<br/>wave-ordered specialists:<br/>wave N receives wave N-1 findings<br/>as bounded grounding context]
    SU --> VE[Verifier<br/>lexical overlap, source authority,<br/>numeric grounding, polarity,<br/>quote location, freshness]
    VE --> CR[Critic + Contradiction Engine + Red Team + Confidence v2]
    CR -->|sufficient| SY[Synthesizer<br/>sense-separated cited answer<br/>with mandatory disambiguation]
    CR -->|expand: novel queries + budget + no stall| PL
    CR -->|stop: budget wall / stall / no novel queries / ceiling| SY
    SY --> QG[Answer quality gate<br/>accuracy/relevance/evidence/<br/>clarity/reasoning 0-100<br/>one bounded re-synthesis]
    QG --> CH[Citation health check<br/>live URL re-validation +<br/>sentence-support fusion]
    CH --> FIN[Finalize<br/>report: answer, evidence,<br/>contradictions, decision layer,<br/>limitations, confidence]
```

One LangGraph instance drives the loop (`app/graph/workflow.py`); the
resume endpoint re-enters at the critic node with state rebuilt from SQLite.

---

## Component map

### Intelligence

| Module | Responsibility |
|---|---|
| `app/agents/intent.py` | Intent classification before research: ambiguity → ranked senses, domain, explanation level; deterministic fallback with curated homonym hints; never blocks — the answer disambiguates. |
| `app/agents/answer_quality.py` | Pre-delivery gate: five-axis 0-100 scoring from measured state; one bounded re-synthesis with failures fed back; below-threshold answers disclosed. |
| `app/agents/planner.py` | Contracts with axis/search_type/minimum_sources/variants/wave/sense; axis-coverage enforcement; intent domain override; dependency waves (max 3). |
| `app/agents/search.py` | 5 providers, per-provider circuit breakers + retry policies, fetch bulkhead, PDF extraction, canonical-URL + near-dup snippet dedup, domain diversity caps, freshness half-lives, disk cache. |
| `app/agents/summarizer.py` | Per-contract specialists (financial/technical/…), source attribution validated against provided documents, direct-quote parsing, token-budgeted chunking, per-URL cache keyed by role + prerequisite digest. |
| `app/agents/verifier.py` | Deterministic per-fact checks; blanks raw content after each pass (memory hygiene). |
| `app/core/contradictions.py` | Three detectors (polarity first — outside the similarity band; then temporal; then unit-aware numeric), severity ordering, cap 5. |
| `app/core/confidence.py` | 7 base signals + citation support + axis coverage + pool-size-scaled contradiction penalty; degraded-run cap 0.55. |
| `app/agents/citation_check.py` | Live URL re-validation (HEAD → 2KB ranged GET, bounded, never fatal) fused with sentence support into per-source verdicts (ok/warn/broken/bad). |
| `app/core/semantic.py` | TF-IDF hybrid engine: stemming, synonym canonicalization, negation weighting, vectorized batch scoring, stopword-stripped dup floor. |
| `app/agents/redteam.py` | Heuristic adversarial review; survival score gates the critic. |
| `app/core/decision.py` | Decision layer (options → recommendation → rationale → risk) for strategic queries. |

### Control & efficiency

| Module | Responsibility |
|---|---|
| `app/core/depth_controller.py` | Intelligent stopping: sufficiency, marginal-gain stalls, ceiling, no-novel-query memory, budget walls, mode confidence targets, min iterations. |
| `app/core/usage.py` | Per-run ledger (ContextVar): LLM tokens/cost, searches, cache hits; feeds the budget governor, the stopping rule, the API events and the UI. |
| `app/agents/budget.py` | Four ceilings (USD/tokens/calls/seconds), mode multipliers, can-afford-pass protocol. |
| `app/core/llm.py` | Provider chain with breakers, fail-fast auth/quota errors, Retry-After honoring (capped 3s), JSON mode, usage parsing, probe caching (30s success). |
| `app/core/llm_cache.py` | Exact-prompt disk cache (TTL 6h, size-capped) — cache hits refund USD while keeping token accounting. |
| `app/core/isolation.py` | AgentContext per sub-question: workers never see the full state or another contract's raw content. |

### Surfaces

| Module | Responsibility |
|---|---|
| `app/api/routes.py` | NDJSON stream (events below), resume, trace, provider CRUD + latency probe. |
| `app/db/sqlite.py` | 14 tables, WAL, auto-initialized schema, `file:`/`sqlite://` URL forms. |
| `frontend/` | React mission console: thread, answer card, claim drawer, intelligence panel (budget meter, wave strip, citation health, confidence breakdown). |

---

## NDJSON event schema (`POST /api/research/stream`)

| Event | Payload highlights |
|---|---|
| `progress` | `request_id`, `message` |
| `plan` | `items` (sub-questions), `orchestration`, `waves` (dependency-wave shape) |
| `search_progress` | `snippets` |
| `critic` | `iteration`, `reason`, `breakdown` (confidence signals + weights + notes), `redteam`, `budget` (live ledger snapshot) |
| `findings` | `items` (claims with verification flags/scores/reasons); re-emitted once with `verified_update` after the verifier pass |
| `final_report` | `report`, `confidence`, `degraded`, `answer_support`, `budget`, `wave_report`, `citation_health` |
| `decisions` | `items` (decision layer options) |
| `error` | `message` (actionable: key/quota/timeout causes) |

---

## Design decisions worth knowing

**Why verification is deterministic (no LLM).** A verifier that costs a
model call is a verifier you consult less often. Lexical + numeric +
polarity checks run in ~145µs per claim and catch the failure modes that
matter (fabricated numbers, negation inversions, topical mismatches) —
measured F1 1.0 on the labeled set.

**Why the LLM cache is exact-prompt.** Semantic caching returns subtly
wrong answers to slightly different questions. Research pipelines re-issue
*identical* prompts (critic re-evals after partial expansion, re-runs) —
an exact-hit cache captures that traffic with zero correctness risk.

**Why contradictions penalize instead of blending into the average.**
A weighted average lets three strong sources mask one direct lie. The
penalty is pool-size scaled: one conflict among 3 facts means a third of
the evidence disagrees; among 30 it is one stale page.

**Why "not" is not a stopword.** The negation token weighted like a number
is what keeps "X" and "not X" from merging in dedup, keeps them inside the
contradiction band, and keeps citation support from counting a negation as
an affirmation. (Found by the benchmark suite, not by intuition.)

**Why waves exist even though search is question-level.** The value is
context chaining in the summarizer: "compare X vs Y" runs after "what is X"
and receives its findings as bounded grounding — dependent extraction sees
references it could not resolve from raw pages alone.

**Why MAX_PARALLEL_LLM=2.** Concurrent large prompts are exactly what
exhausts free-tier TPM/TPD quotas (observed live: Groq 200k daily tokens
burned by 3 parallel summarizer calls). The semaphore bounds in-flight
prompts, not threads.

---

## Data lifecycles

- **Raw page content** is blanked by the verifier after each pass — the
  last consumer of full text is verification; snippets persist for
  transparency.
- **Facts** accumulate across passes with append-only plan growth; dedup
  merges restatements and records corroboration (distinct domains only —
  self-syndication is not corroboration).
- **Caches**: search results 1h, LLM responses 6h, both size-capped at
  250MB and LRU-evicted.
- **SQLite**: every run persists agent events, sources, claims,
  verification results, contradictions, citations, critic reviews and
  decisions — the trace endpoint reconstructs the full timeline.
