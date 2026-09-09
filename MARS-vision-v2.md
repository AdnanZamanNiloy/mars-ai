# MARS — Multi-Agent Research System

> **Autonomous Research Intelligence for Complex Questions and High-Stakes Decisions**

MARS is an AI-powered research intelligence platform that decomposes complex questions, intelligently delegates research to specialized agents, runs investigations in parallel, verifies evidence, challenges conclusions, preserves research state, and produces decision-ready outputs with traceable evidence.

The current MARS already provides autonomous planning, bounded asynchronous search, iterative critique, streaming research events, SQLite persistence, provider fallback, and a LangGraph workflow.

The top-tier version evolves that foundation into a **research operating system**.

Section 0 below is a candid feasibility check against a solo developer running this on an i5-1240P / 8GB RAM machine with remote LLM APIs for generation. Read it before Section 34's build order — the rest of the document is the long-term vision, but Section 34 is the actual sequence to build in.

---

# 0. Feasibility & Sequencing Reality Check

**Is the rest of this document "perfect planning"? No — not as originally written.** It's a strong, complete product vision. It is not an execution plan calibrated to your actual constraints. Specifically:

```text
1. No phasing tied to capacity
   The original "Tier 1 — Must Build" list bundled 10 major
   subsystems (orchestrator, full agent roster, parallel execution,
   contracts, memory, verification, critic, evidence graph,
   confidence engine, Mission Control UI) as if they'd ship together.
   That's 6-12 months of team effort, not a next sprint.

2. Agent counts and data model are enterprise defaults
   "5-10 parallel agents" and a ~20-table schema aren't derived from
   your hardware. Nothing in the original doc caps concurrency
   against 8GB of RAM the way your existing MAX_PARALLEL_SEARCH
   already caps search concurrency.

3. Zero reliability content
   No retries, circuit breakers, timeouts, rate limiting, caching,
   or tests anywhere in this document. Every feature below assumes
   a rock-solid base pipeline that doesn't currently exist. Adding
   a Critic/Red Team or Confidence Engine on top of a pipeline with
   no retry logic just gives you a more elaborate way to fail.

4. Illustrative numbers, not targets
   "$1.84 estimated," "2m 14s runtime" are placeholder flavor text,
   not figures derived from real Groq/HuggingFace pricing. Don't
   plan a budget governor around them without measuring your own
   actual token costs first.

5. UI ambition front-loaded
   The NASA-Mission-Control-meets-Bloomberg-Terminal redesign
   (Section 36) is a multi-week frontend project. It was originally
   listed as a Tier 1 item, competing for time against backend
   correctness before the backend has even proven itself.
```

**What changed in this version of the document:**

```text
- Removed: Feature 07 "Evidence Graph" (spec + feature table row)
- Removed: the "Research Graph" UI screen (it was the same graph,
  just visualized — removing one without the other would leave a
  UI screen with nothing behind it)
- Kept: "Evidence Explorer" as a plain list/detail view for a single
  claim's sources and verification status — no graph rendering
  required, it's just structured text under Verification + Confidence
- Added: Section 0 (this section)
- Added: hardware guardrails to Feature 01 (Orchestrator) and
  Feature 03 (Parallel Research), capping default concurrency
  against your 8GB RAM instead of leaving it unbounded
- Rewrote: Section 34 "Recommended Build Priority" into four tiers,
  with a new Tier 0 (reliability foundation) that has to exist
  before anything else in this document
- Updated: the "Five Signature Features" in Section 35 to four,
  since Evidence Intelligence Graph is gone
- Noted: Section 36 (visual design direction) is explicitly a later
  phase, after the backend is stable
```

**Bottom line:** treat this document as the north star, not the sprint plan. What you actually implement first is the resilience/observability roadmap (retries, caching, structured logging, rate limiting, SQLite WAL, tests) — none of which lives in this vision doc because it's a separate, more boring, more urgent layer. Once that foundation exists, Section 34 below tells you what to layer on top of it and in what order.

---

# 1. Product Vision

## Core Principle

MARS should not behave like:

```text
User → Chatbot → Answer
```

It should behave like:

```text
User
  ↓
Understand the question
  ↓
Create research strategy
  ↓
Select the right research workforce
  ↓
Run independent investigations
  ↓
Collect evidence
  ↓
Challenge findings
  ↓
Verify claims and citations
  ↓
Synthesize knowledge
  ↓
Generate strategic conclusions
  ↓
User receives an auditable answer
```

## Product Promise

> **Ask a difficult question. MARS builds the research team, investigates the problem, verifies the evidence, and explains the conclusion.**

---

# 2. Top-Tier System Architecture

```text
                               ┌─────────────────────┐
                               │       USER          │
                               │ Question / Objective│
                               └──────────┬──────────┘
                                          ↓
                               ┌─────────────────────┐
                               │ MARS ORCHESTRATOR   │
                               │                     │
                               │ Complexity Analysis │
                               │ Strategy Planning   │
                               │ Agent Selection     │
                               │ Budget Allocation   │
                               │ Stop Conditions     │
                               └──────────┬──────────┘
                                          ↓
                 ┌────────────────────────┼────────────────────────┐
                 ↓                        ↓                        ↓
        ┌────────────────┐       ┌────────────────┐       ┌────────────────┐
        │ SCOUT AGENTS   │       │ ANALYST AGENTS │       │ SPECIALISTS    │
        │                │       │                │       │                │
        │ Web discovery  │       │ Deep analysis  │       │ Finance        │
        │ Source hunting │       │ Comparisons    │       │ Legal          │
        │ Fact discovery │       │ Calculations   │       │ Technical      │
        └────────┬───────┘       └────────┬───────┘       │ Market         │
                 │                        │               │ Academic       │
                 └────────────────────────┼───────────────┘
                                          ↓
                               ┌─────────────────────┐
                               │ RESEARCH MEMORY     │
                               │                     │
                               │ Plans               │
                               │ Tasks               │
                               │ Sources             │
                               │ Evidence            │
                               │ Claims              │
                               │ Agent state         │
                               │ Intermediate work   │
                               └──────────┬──────────┘
                                          ↓
                               ┌─────────────────────┐
                               │ CRITIC / RED TEAM   │
                               │                     │
                               │ Contradictions      │
                               │ Weak assumptions    │
                               │ Missing evidence    │
                               │ Bias                │
                               └──────────┬──────────┘
                                          ↓
                               ┌─────────────────────┐
                               │ VERIFICATION ENGINE │
                               │                     │
                               │ Claim checking      │
                               │ Citation checking   │
                               │ Source quality      │
                               │ Evidence alignment  │
                               └──────────┬──────────┘
                                          ↓
                               ┌─────────────────────┐
                               │ SYNTHESIS ENGINE    │
                               │                     │
                               │ Findings → Insights │
                               │ Conflicts → Resolve │
                               └──────────┬──────────┘
                                          ↓
                               ┌─────────────────────┐
                               │ DECISION ENGINE     │
                               │                     │
                               │ Options             │
                               │ Risks               │
                               │ Scenarios           │
                               │ Recommendations     │
                               └──────────┬──────────┘
                                          ↓
                               ┌─────────────────────┐
                               │ EXECUTIVE OUTPUT    │
                               │                     │
                               │ Report              │
                               │ Sources             │
                               │ Confidence          │
                               └─────────────────────┘
```

---

# 3. Feature Fingerprint

The following features define the **top-tier MARS fingerprint**.

| #  | Feature                     | What it does                                         | Why it matters                              |
| -- | --------------------------- | ---------------------------------------------------- | ------------------------------------------- |
| 01 | Adaptive Orchestrator       | Decides research strategy and agent count            | Prevents unnecessary agent usage            |
| 02 | Specialized Agent Workforce | Assigns different research roles                     | Enables deeper parallel research            |
| 03 | Parallel Execution          | Runs independent investigations simultaneously       | Reduces research latency                    |
| 04 | Isolated Agent Context      | Each agent works independently                       | Reduces cross-agent contamination           |
| 05 | Delegation Contracts        | Gives every agent explicit objectives and boundaries | Prevents duplication and vague work         |
| 06 | Research Memory             | Stores plan, evidence, claims, and state             | Enables long-running and resumable research |
| 07 | Citation Verification       | Independently verifies citations and claims          | Reduces unsupported conclusions             |
| 08 | Critic / Red Team           | Attempts to disprove findings                        | Improves robustness                         |
| 09 | Contradiction Engine        | Detects conflicting evidence                         | Prevents false consensus                    |
| 10 | Confidence Engine           | Calculates evidence-based confidence                 | Makes uncertainty visible                   |
| 11 | Dynamic Research Depth      | Expands or contracts investigation based on evidence | Uses resources intelligently                |
| 12 | Cost / Token Governor       | Controls agent and tool budgets                      | Makes the system economically practical     |
| 13 | Durable Execution           | Checkpoints research state                           | Allows recovery after failure               |
| 14 | Full Agent Trace            | Records actions, decisions, and tool calls           | Enables debugging and trust                 |
| 15 | Research Replay             | Reconstructs how a result was produced               | Supports evaluation and auditing            |
| 16 | Evaluation Lab               | Tests research quality systematically                | Enables continuous improvement              |
| 17 | Scenario Engine              | Runs what-if analysis                                | Turns research into strategic intelligence  |
| 18 | Executive Decision Layer     | Converts research into options and recommendations   | Makes results actionable                    |
| 19 | Self-Diagnosis               | Finds weaknesses in tools/prompts/workflows          | Enables system improvement                  |

---

# 4. Feature 01 — Adaptive Orchestrator

## Purpose

The Orchestrator is the **brain of MARS**.

Instead of always using the same number of agents, it first determines how difficult the research problem is.

## How it works

```text
Question
   ↓
Intent detection
   ↓
Complexity scoring
   ↓
Research dimensions
   ↓
Required expertise
   ↓
Agent count
   ↓
Research budget
   ↓
Execution plan
```

## Example

### Simple query

```text
"What is the latest version of Python?"
```

MARS:

```text
Complexity: Low
Agents: 1
Sources: 2–3
Depth: Quick
Budget: Low
```

### Complex query

```text
"Should Bangladesh significantly increase nuclear energy investment
over the next 20 years?"
```

MARS:

```text
Complexity: Very High
Agents: 7

Research dimensions:
- Energy economics
- Nuclear technology
- Regulation
- Financing
- Geopolitics
- Environmental impact
- Long-term scenarios
```

## Hardware guardrail (8GB RAM, API-only generation)

```text
Even for "Very High" complexity, cap the default agent count at 3-4
concurrent agents unless you've measured real headroom under load.
Treat 7+ agent fan-outs as an explicit "Deep Research" opt-in, gated
behind the Cost/Token Governor (Feature 12), not the default path.
The complexity score should still be computed as above — just clamp
the resulting agent count against a MAX_PARALLEL_AGENTS ceiling
before dispatch.
```

---

# 5. Feature 02 — Specialized Agent Workforce

MARS should operate as an **AI research organization**.

## Core Agents

### Orchestrator

Plans and coordinates the entire mission.

### Scout

Discovers sources, reports, datasets, papers, and relevant information.

### Analyst

Interprets evidence, compares findings, calculates trends, and performs reasoning.

### Specialist

Handles domain-specific research.

Examples:

```text
Financial Specialist
Legal Specialist
Technical Specialist
Market Specialist
Scientific Specialist
Policy Specialist
Academic Specialist
```

### Critic

Challenges assumptions and searches for weaknesses.

### Verification Agent

Checks claims and citation accuracy.

### Synthesizer

Combines all validated findings.

### Strategist

Turns findings into decision options.

---

# 6. Feature 03 — Parallel Research

Instead of:

```text
Agent A → finish
Agent B → finish
Agent C → finish
```

MARS executes:

```text
Agent A ────────┐
Agent B ────────┤
Agent C ────────┼──→ Evidence Layer
Agent D ────────┤
Agent E ────────┘
```

The current MARS already uses bounded asynchronous search and a configurable concurrency limit.

The top-tier version extends this from **parallel searches** to **parallel agent missions**.

## Hardware guardrail (8GB RAM, API-only generation)

```text
This needs an explicit ceiling, same pattern as your existing
MAX_PARALLEL_SEARCH. Add MAX_PARALLEL_AGENTS (default 3) and enforce
it in the orchestrator dispatch step, not just as documentation.

Also: once an agent's task completes, release the raw source text
it collected from memory and keep only its structured findings
(claims + citations). Holding full text for every source across
every agent for the whole mission is the fastest way to exhaust
8GB under a "Deep Research" run with 60+ sources.
```

---

# 7. Feature 04 — Isolated Agent Context

Every research agent gets:

```text
Own context
Own objective
Own task
Own tool access
Own evidence
Own stopping condition
```

Example:

```text
Market Agent
Objective:
Analyze global EV battery market growth.

Do:
- Find market forecasts
- Compare credible sources
- Identify major trends

Do not:
- Analyze regulatory policy
- Make investment recommendations
```

This makes specialization meaningful.

---

# 8. Feature 05 — Delegation Contracts

Every agent receives a structured task contract.

```json
{
  "agent": "financial_researcher",
  "objective": "Analyze the economics of nuclear expansion",
  "scope": [
    "capital expenditure",
    "operating costs",
    "financing",
    "ROI"
  ],
  "tools": [
    "web_search",
    "documents",
    "calculator"
  ],
  "minimum_sources": 6,
  "output_format": "structured_findings",
  "stop_condition": "Sufficient evidence"
}
```

This prevents agents from duplicating one another.

---

# 9. Feature 06 — Research Memory

Current MARS persists completed reports in SQLite with the query, report, confidence, and timestamp.

Top-tier MARS should persist **the entire research mission**.

## Research Memory Model

```text
Project
 ├── Objective
 ├── Research strategy
 ├── Agent tasks
 ├── Agent outputs
 ├── Sources
 ├── Evidence
 ├── Claims
 ├── Contradictions
 ├── Critiques
 ├── Verification results
 ├── Decisions
 └── Final report
```

This makes MARS resumable.

---

# 10. Feature 07 — Dedicated Verification Layer

Never allow:

```text
Agent → User
```

Use:

```text
Research
   ↓
Verification
   ↓
Synthesis
   ↓
User
```

The verification engine checks:

```text
Claim exists?
      ↓
Source supports claim?
      ↓
Citation points to correct source?
      ↓
Source quality acceptable?
      ↓
Evidence current enough?
      ↓
Contradictory evidence?
```

---

# 11. Feature 08 — Critic + Red Team

The Critic should do more than ask:

> “Is the research sufficient?”

It should actively try to break the conclusion.

## Red Team Questions

```text
What assumptions are weak?

Which sources disagree?

What evidence is missing?

Could this conclusion be wrong?

What alternative explanation exists?

What would invalidate the recommendation?
```

The result becomes:

```text
Original conclusion
        ↓
Red-team attack
        ↓
Weakness discovered
        ↓
Additional research
        ↓
Updated conclusion
```

---

# 12. Feature 09 — Contradiction Engine

MARS should explicitly surface disagreement.

```text
SOURCE A
Market growth: 18%

SOURCE B
Market growth: 11%

SOURCE C
Market growth: 7%
```

Instead of hiding this, MARS reports:

```text
⚠ CONTRADICTION DETECTED

Forecast range:
7% – 18%

Primary reason:
Different assumptions regarding EV adoption.

Resolution:
Use scenario-based forecast rather than a single number.
```

This turns disagreement into intelligence.

---

# 13. Feature 10 — Evidence-Based Confidence

Do not rely solely on a model-generated confidence value.

Calculate confidence using signals such as:

```text
Source quality
Source diversity
Citation coverage
Cross-source agreement
Evidence strength
Freshness
Claim verification
Critic survival
```

## UI

```text
RESEARCH CONFIDENCE

Source Quality          92%
Evidence Coverage       96%
Cross-source Agreement  74%
Claim Verification      90%
Source Freshness        88%
Critic Survival         93%

────────────────────────

OVERALL CONFIDENCE      87%
```

---

# 14. Feature 11 — Dynamic Research Depth

MARS should know when to stop.

```text
Initial research
       ↓
Evidence sufficient?
     ↙       ↘
   YES        NO
    ↓          ↓
Finalize   Spawn more research
               ↓
          Re-evaluate
```

Possible stopping signals:

```text
Enough high-quality evidence
No major unanswered sub-question
Confidence above threshold
Budget threshold reached
Marginal information gain too low
Maximum depth reached
```

---

# 15. Feature 12 — Cost & Token Governor

Top-tier MARS must understand economics.

Every research mission gets a budget.

```text
Research Budget

Estimated: $1.84

Spent:     $1.21

Remaining: $0.63
```

MARS can decide:

```text
Use another agent?
Run another search?
Search deeper?
Stop now?
```

This prevents runaway research. Given that generation runs entirely on paid LLM APIs, this is one of the cheapest features on this list to build and one of the highest-value — it directly protects your budget as agent concurrency increases. Consider building it alongside Tier 1, not after it.

---

# 16. Feature 13 — Durable Execution

If an agent fails at step 45 of a long research mission, MARS should not restart from step 1.

Use checkpoints:

```text
Checkpoint 1
Plan saved

Checkpoint 2
Sources saved

Checkpoint 3
Agent findings saved

Checkpoint 4
Critique saved

Checkpoint 5
Verification saved
```

Failure:

```text
Agent timeout
     ↓
Restore latest checkpoint
     ↓
Retry only failed task
     ↓
Continue mission
```

---

# 17. Feature 14 — Full Observability

Every important action should be observable.

```text
14:03:12
Orchestrator created strategy

14:03:14
Spawned 6 agents

14:03:19
Scout collected 14 sources

14:03:24
Analyst generated 8 claims

14:03:27
Critic challenged 3 claims

14:03:31
Verification started

14:03:37
2 unsupported claims rejected

14:03:42
Synthesis complete
```

Your existing streaming API already exposes progress, plan, search progress, critic, findings, final report, and error events.

The top-tier UI should visualize these events as a **live mission timeline**.

---

# 18. Feature 15 — Research Replay

Every research run should be replayable.

```text
Research Run #1042

[Replay]

→ Orchestrator
→ Agent allocation
→ Searches
→ Findings
→ Critic
→ Verification
→ Synthesis
```

This is extremely valuable for:

```text
Debugging
Research auditing
Evaluation
Prompt testing
Demonstrations
```

---

# 19. Feature 16 — Evaluation Lab

MARS should evaluate itself.

## Evaluation Dashboard

```text
MARS Evaluation Lab

Test Questions               100

Research Accuracy             91%
Citation Accuracy             95%
Source Quality                89%
Completeness                  92%
Contradiction Detection       87%
Recommendation Quality        88%
Average Cost                  $1.42
Average Runtime               2m 14s
```

Use representative research questions and compare MARS against simpler baselines.

---

# 20. Feature 17 — Scenario Engine

Research should not stop at:

> “Here is what is happening.”

MARS should answer:

> “What happens next?”

Example:

```text
BASE CASE
Moderate EV adoption

OPTIMISTIC
Rapid EV adoption

PESSIMISTIC
Slow adoption + supply constraints

REGULATORY
Major policy changes

HIGH-COST
Capital costs remain elevated
```

Each scenario can trigger targeted agent research.

---

# 21. Feature 18 — Executive Decision Intelligence

The final output should not look like raw AI prose.

It should answer:

```text
What happened?
Why does it matter?
What are the major risks?
What opportunities exist?
What are the options?
What should we do?
How confident are we?
```

## Executive Output

```text
EXECUTIVE CONCLUSION

The evidence supports moderate expansion,
but immediate aggressive investment is not
recommended.

CONFIDENCE
87%

KEY FINDINGS
01 — Demand is growing
02 — Supply risk remains material
03 — Regulation is uncertain

STRATEGIC OPTIONS

A — Invest aggressively
B — Partner with suppliers
C — Stage investment

RECOMMENDED

Option C

WHY

Lower downside while preserving upside.

[View Evidence]
[Challenge Conclusion]
[Open Full Research]
```

---

# 22. Feature 19 — Self-Diagnosis

An advanced MARS can improve its own research process.

The system should inspect:

```text
Agent failures
Poor search queries
Repeated sources
Weak outputs
Bad tool descriptions
High latency
High token consumption
```

Then recommend improvements:

```text
Problem detected:

Financial Agent repeatedly returns
low-quality market forecasts.

Suggested change:

Add primary-source requirement
and financial database preference.
```

This creates a feedback loop:

```text
Run
 ↓
Observe
 ↓
Diagnose
 ↓
Improve
 ↓
Run again
```

---

# 23. Top-Tier UI Design

The interface should have **five major surfaces**.

## 01 — Command Center

The home screen.

```text
MARS
Research Intelligence

What decision are you trying to make?

┌─────────────────────────────────────────────┐
│ Ask a research question...                  │
└─────────────────────────────────────────────┘

Quick Research      Deep Research
                     ↓
                 7 agents
                 64 sources
                 ~$1.84
```

### Main sections

```text
New Research
Recent Missions
Active Research
Saved Knowledge
Agent Library
```

---

# 24. Mission Control

This is the most visually impressive screen.

```text
MARS / MISSION CONTROL

EV BATTERY SUPPLY CHAIN 2030

ORCHESTRATOR
Strategy complete
7 agents deployed

SCOUT              ████████████ 100%
MARKET             █████████░░░  82%
FINANCE            ████████░░░░  71%
REGULATORY         ██████░░░░░░  55%
TECHNICAL          █████████░░░  81%
CRITIC             ○ Waiting
VERIFIER           ○ Waiting
```

Right panel:

```text
RESEARCH HEALTH

Sources       64
Claims        42
Conflicts      6
Verified      37
Confidence    87%

Budget

$1.21 / $1.84
```

Bottom:

```text
LIVE AGENT TRACE
```

---

# 25. Evidence Explorer

A dedicated claim-inspection view — a structured list/detail panel, not a graph renderer. It reads directly from the Verification and Confidence data you're already collecting; no separate graph engine is needed.

```text
CLAIM

"EV battery costs will continue declining."

SUPPORTED BY
├── Source A
├── Source B
└── Source C

ANALYZED BY
├── Market Agent
└── Financial Agent

CHALLENGED BY
└── Critic

VERIFICATION
✓ Citation valid
✓ Source supports claim
✓ No major contradiction

CONFIDENCE
91%
```

---

# 26. Executive Report

The final result should look closer to a premium strategy report than a chatbot conversation.

```text
MARS INTELLIGENCE REPORT

Research Question

Executive Summary

Key Findings

Evidence

Market Analysis

Risks

Contradictions

Scenarios

Strategic Options

Recommendation

Confidence

Sources

Research Trace
```

---

# 27. Agent Library

Allow users to inspect the MARS workforce.

```text
AGENT LIBRARY

Scout
Source discovery

Analyst
Deep reasoning

Financial
Economic analysis

Regulatory
Policy analysis

Critic
Red-team verification

Verifier
Citation validation

Strategist
Decision intelligence
```

Each agent should show:

```text
Objective
Tools
Capabilities
Performance
Average latency
Average cost
Success rate
```

---

# 28. Research Modes

MARS should provide explicit modes.

| Mode      | Use                              | Agents | Output           |
| --------- | --------------------------------- | -----: | ---------------- |
| Quick     | Simple factual questions         |      1 | Direct answer    |
| Standard  | Moderate research                |    2–4 | Research brief   |
| Deep      | Complex investigation            |    5–8 | Full report      |
| Executive | Decisions and strategy           |   5–10 | Decision brief   |
| Audit     | Verification                     |    2–5 | Evidence audit   |
| Red Team  | Challenge an existing conclusion |    3–6 | Counter-analysis |

Note: "Deep" and "Executive" modes exceed the default `MAX_PARALLEL_AGENTS` guardrail from Feature 01/03. Gate them behind an explicit user opt-in and a Cost/Token Governor check, not as silently-triggered defaults.

---

# 29. Updated MARS Data Model

The system should grow beyond a single `research_reports` table.

```text
users
projects
research_runs
research_plans
agent_tasks
agent_runs
agent_events

sources
evidence
claims
citations
contradictions
critic_reviews
verification_results

research_memory
scenarios
decisions
final_reports

evaluation_runs
evaluation_results
```

Build this incrementally, not as a single migration. A reasonable first slice is `research_runs`, `agent_tasks`, `sources`, `claims` — enough to support Tier 1 below — with the rest added as each feature actually needs it.

---

# 30. Updated API Philosophy

Your existing `/api/research/stream` endpoint already streams structured progress events.

The top-tier API should extend this into mission-oriented APIs.

```text
POST   /api/research
POST   /api/research/{id}/pause
POST   /api/research/{id}/resume
POST   /api/research/{id}/cancel

GET    /api/research/{id}
GET    /api/research/{id}/events
GET    /api/research/{id}/sources
GET    /api/research/{id}/claims
GET    /api/research/{id}/evidence
GET    /api/research/{id}/trace

POST   /api/research/{id}/challenge
POST   /api/research/{id}/verify
POST   /api/research/{id}/scenario

GET    /api/projects
GET    /api/agents
GET    /api/evaluations
```

---

# 31. Top-Tier Research Lifecycle

The canonical MARS lifecycle should be:

```text
01  RECEIVE
    User submits research objective

02  UNDERSTAND
    MARS determines intent and constraints

03  SCORE
    Complexity and research depth calculated

04  PLAN
    Orchestrator creates research strategy

05  DELEGATE
    Specialized agents receive contracts

06  EXPLORE
    Agents research independently

07  COLLECT
    Evidence enters research memory

08  ANALYZE
    Agents interpret evidence

09  CHALLENGE
    Critic / Red Team attacks findings

10  VERIFY
    Claims and citations are checked

11  RESOLVE
    Contradictions are analyzed

12  SYNTHESIZE
    Findings become coherent conclusions

13  SIMULATE
    Optional scenarios are generated

14  DECIDE
    Strategic recommendations are formed

15  DELIVER
    Executive report is produced

16  TRACE
    Full evidence and agent history remain inspectable
```

---

# 32. Example: A Complete MARS Mission

## User

```text
Should our company invest $20M in AI infrastructure over the next
three years?
```

## Step 1 — Orchestrator

```text
Complexity: Very High

Required dimensions:
- Financial
- Technical
- Market
- Competitive
- Security
- Workforce
```

## Step 2 — Agent Allocation

```text
1 × Scout
1 × Financial Analyst
1 × Technical Analyst
1 × Market Analyst
1 × Competitive Analyst
1 × Security Analyst
1 × Critic
```

## Step 3 — Parallel Research

All independent agents investigate their assignments.

## Step 4 — Evidence Collection

MARS records:

```text
72 sources
48 claims
13 major findings
7 contradictions
```

## Step 5 — Critic

The Critic finds:

```text
3 assumptions too optimistic
2 forecasts based on weak sources
1 major market risk
```

## Step 6 — Verification

```text
44 claims verified
2 citations rejected
2 claims rewritten
```

## Step 7 — Synthesis

MARS produces:

```text
Recommended investment:
$12M initially

Stage additional:
$8M based on performance triggers
```

## Step 8 — Scenario Analysis

```text
Bull Case
Base Case
Bear Case
```

## Step 9 — Executive Report

The user receives:

```text
Recommendation
Financial rationale
Technical rationale
Risks
Scenario outcomes
Evidence
Confidence
Sources
Full research trace
```

---

# 33. What Makes MARS Top-Tier

MARS becomes genuinely differentiated when these capabilities work together:

```text
ADAPTIVE
     +
PARALLEL
     +
SPECIALIZED
     +
PERSISTENT
     +
VERIFIABLE
     +
CRITICAL
     +
OBSERVABLE
     +
DECISION-ORIENTED
```

The strongest product identity is:

> **MARS does not simply generate answers. It constructs and manages a temporary AI research organization around each difficult question.**

---

# 34. Recommended Build Priority

This replaces the original flat "Tier 1/2/3" list with a sequence that assumes one developer, 8GB RAM, and paid LLM APIs.

## Tier 0 — Foundation (build before anything else in this document)

```text
Retries + circuit breaker around every LLM and search call
Structured, schema-validated LLM outputs (reject and retry malformed
  output instead of letting it propagate into the graph)
Per-request timeouts, decoupled from client disconnects
Bounded, TTL'd caching for search results and summaries
SQLite WAL mode + a basic backup job
Structured (JSON) logging with a request_id threaded through the
  pipeline
Rate limiting on the research endpoint
A basic pytest suite covering the existing planner → search →
  critic loop
```

Why first: every feature below assumes the base pipeline doesn't fall over, doesn't leak memory, and doesn't silently burn API budget. None of Tier 0 is in the rest of this document — it's the substrate everything else stands on.

## Tier 1 — Must build (extends what already exists; build 3-4 at a time, not all at once)

```text
Adaptive Orchestrator, with the MAX_PARALLEL_AGENTS hardware cap
  from Feature 01/03 enforced, not just documented
Parallel Agent Execution, bounded (default 3 concurrent agents)
Delegation Contracts — formalize what each agent call already
  implicitly does
Cost / Token Governor — cheap to build, protects your budget as
  concurrency increases, so build it alongside the orchestrator
  rather than after it
Verification Agent
Evidence-Based Confidence Engine
Critic → Red Team upgrade (extends your existing critic agent with
  adversarial prompts rather than building a new one)
Research Memory schema extension — start with research_runs,
  agent_tasks, sources, claims from Section 29, not the full 20-table
  model
```

## Tier 2 — High value (after Tier 1 is stable and covered by tests)

```text
Specialized Agent Workforce — expand from 2-3 roles (Scout, Analyst,
  Critic) to the full roster (Financial, Legal, Technical, Market,
  etc.) as real questions actually demand each specialty
Contradiction Engine
Durable Checkpointing
Research Replay
Executive Report Builder
Evidence Explorer / claim inspector (Section 25)
Research Modes (Quick / Standard / Deep presets, Section 28)
```

## Tier 3 — Advanced / optional (only if the product proves itself in use)

```text
Evaluation Lab
Self-Diagnosis
Scenario Engine
Mission Control UI — the full NASA/Bloomberg-style redesign
  (Section 36)
Multi-tenant users/projects, org-wide knowledge memory
```

---

# 35. MARS Final Product Identity

## Name

**MARS**

### Multi-Agent Research System

## Tagline

> **Research beyond the obvious.**

Alternative:

> **Complex questions. Coordinated intelligence.**

## Positioning

> **MARS is an autonomous research intelligence platform that orchestrates specialized AI agents to investigate complex questions, verify evidence, challenge assumptions, and transform research into decision-ready intelligence.**

## Core Loop

```text
QUESTION
   ↓
ORCHESTRATE
   ↓
RESEARCH
   ↓
ANALYZE
   ↓
CHALLENGE
   ↓
VERIFY
   ↓
SYNTHESIZE
   ↓
DECIDE
```

## The Four Signature Features

```text
01  Adaptive Agent Orchestration
02  Independent Research Workforce
03  Verification + Red Team
04  Executive Decision Intelligence
```

These four features should define the MARS brand, architecture, and UI rather than simply adding more agents for the sake of complexity.

---

# 36. Final Design Direction

This is the long-term visual identity — build it in Tier 3, once the backend (Tier 0-2) is stable and trustworthy. Frontend polish shouldn't compete with reliability and agent-quality work early on.

The visual language should feel like:

```text
NASA Mission Control
        +
Bloomberg Terminal
        +
Premium AI Workspace
        +
Strategy Consulting Platform
```

Use:

* Dark graphite interface
* Soft atmospheric gradients
* Subtle Mars-inspired copper/red accents
* Thin borders and glass-like surfaces
* High-contrast editorial typography
* Dense but controlled information hierarchy
* Live agent status indicators
* Timelines and structured evidence panels
* Minimal decorative noise
* Strong use of whitespace around major decisions

The interface should communicate one message immediately:

> **Something intelligent is happening behind the screen.**

That is the design goal for the top-tier MARS experience — once there's a reliable system behind it to represent.
