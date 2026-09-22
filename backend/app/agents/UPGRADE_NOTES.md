# Epistemics upgrade — integration notes

Four files changed, 20 untouched byte-for-byte. Everything is additive: no
existing signature, return type or dict key was removed.

| File | Change |
|---|---|
| `epistemics.py` | **NEW** — conflict adjudication, claim-type evidence standards, coverage balance |
| `confidence.py` | Four previously-dead signals wired in as named caps/penalties |
| `synthesizer.py` | Adjudicates conflicts before rendering; suppresses fake ranges |
| `__init__.py` | Lazy exports for the new names |

---

## 1. Confidence was epistemically inert

`compute_confidence` accepted `contradictions`, `citation_support`,
`planned_axes` and `target_domains`, then appended a note saying they were
*"recorded but not wired into the live engine yet"*. The system could detect
five severe source conflicts and still publish **High (0.82)**.

Measured, same pool of 8 verified facts:

| Situation | Before | After |
|---|---|---|
| Clean pool | 0.95 | 0.95 |
| 3 unresolved source conflicts | 0.95 | **0.55** |
| 4 of 6 planned angles empty | 0.95 | **0.71** |
| 55% citation support | 0.95 | **0.80** |
| One-sided comparison | 0.95 | **0.70** |

The live engine's weights are **untouched**, so historical scores stay
comparable. Adjustments are applied afterwards by
`apply_epistemic_adjustments`, and every one is recorded in `caps_applied`
plus `stats["epistemic_adjustments"]`. `stats["engine_overall"]` keeps the
pre-adjustment score, so any movement can be explained to a reader.

A cap only fires when it *lowers* the score — it can never inflate a weak pool.

## 2. Most "contradictions" were not contradictions

`find_contradictions` compares magnitude and topic similarity, so it fires
on `"revenue was $2bn"` vs `"revenue was $3bn"`. If those are 2022 and 2024
that is growth, and the synthesizer was printing *"sources disagree; the range
is $2–3bn"* — a factual error the pipeline invented on its own.

`classify_conflict` separates:

- **time_series** — same time-varying quantity, different periods
- **scope_mismatch** — US vs global, annual vs cumulative, gross vs net
- **unit_mismatch** — the comparison was never valid
- **genuine** — everything that survives

Polarity conflicts (`"does reduce"` vs `"does not reduce"`) are always genuine
by construction and can never be suppressed.

## 3. Genuine conflicts are now adjudicated

"Sources disagree" is the right answer only when the sources are comparable.
When a regulator's filing contradicts a trade-press estimate, presenting both
as equally weighted is not neutrality.

Rules fire in this order, and the **order is the policy** — a primary source
outranks a newer secondary one, because recency cannot fix a source that was
never authoritative:

1. `primary_source`
2. `measured_over_estimated`
3. `verified`
4. `recency_for_time_varying` — gated on the quantity actually changing over
   time, so a 2025 blog never supersedes a 2019 measurement of a constant
5. `independent_corroboration`
6. `source_authority`

Every verdict carries the rule that produced it, so a reader can reject the
reasoning instead of having to trust it. When no rule fires, the conflict
stays open and the range is then the honest answer rather than the default.

## 4. Evidence standards by claim type

"X is defined as Y" and "X causes Y" cleared the same bar. Now:

| Claim type | Independent sources | Notes |
|---|---|---|
| definitional / descriptive / attributive | 1 | |
| statistical | 2 **or** a primary source | |
| causal | 2 | one source asserting causation is usually reporting correlation |
| evaluative | 2 | contestable by construction |
| predictive | 1 | **unverifiable** — can only be attributed, never established |

Statistical claims accept a primary source *in place of* corroboration. That
is the bar a careful analyst applies, and demanding both would mark almost
every real figure unmet — a check that fires on everything is a check nobody
reads.

## 5. Coverage asymmetry

`coverage_gaps` passes a comparison with nine sources on option A and one on
option B, because both angles are non-empty. That report will confidently
recommend A having barely looked at B. `coverage_asymmetry` measures the split
across the entities the *question* names, which is the axis bias travels along.

---

## Wiring

Wired on the **live path**. `app/core/confidence.compute_confidence` (the
function `workflow.py` calls) now accepts `epistemics` and `query`, and
applies `apply_epistemic_adjustments` itself via a deferred import of
`app.agents.confidence`. Previously the epistemic wrapper existed but was
never imported by the workflow, so every conflict/asymmetry/staleness cap was
dead on real runs.

`workflow.py` builds the report once and caches it in state:

```python
epistemics = state.get("epistemics") or assess_epistemics(query, facts, contradictions)
state["epistemics"] = epistemics          # the synthesizer reuses it, no recompute
breakdown = compute_confidence(..., epistemics=epistemics, query=query)
```

**Applied exactly once.** When `epistemics` is supplied it is authoritative and
the core engine's raw `_contradiction_penalty` is skipped — otherwise a single
conflict would be penalized twice, once by the raw count and once by the
classifier. Without `epistemics`, core behavior is unchanged. The adjustment
trigger is `epistemics is not None`; callers that do not supply it (existing
tests, older call sites) see identical scores.

Verified: a genuine unresolved conflict scores 0.757 without epistemics and
0.607 with — and the 4 contradiction-penalty regression tests remain green.

**Live caveat:** on a real run the mechanism is reachable (`engine_version`
and `epistemic_adjustments` appear on the wire), but a run whose pool has no
genuine conflict, staleness or coverage asymmetry correctly shows
`caps_applied: []` — no adjustment fires. The live path is wired; whether a
cap lands depends on the evidence.

---

## Risks and tuning


- **`_TIME_VARYING_RE`** is a vocabulary list (revenue, price, adoption,
  emissions…). A domain-specific time-varying quantity outside it will be
  treated as a genuine conflict — a *safe* failure (you get the old behaviour),
  but worth extending for your domains.
- **`_SCOPE_GROUPS`** likewise. The asymmetric branch (one side scope-qualified,
  the other bare) is the weakest signal here; if it over-suppresses, delete that
  branch and keep only the both-sides-qualified case.
- **`coverage_asymmetry`** entity extraction is deliberately crude. A false
  entity finds no evidence and is dropped; the `min_ratio=3.0` default is the
  knob to tune.
- **`_find_fact`** matches contradiction claim strings back to facts at 0.6
  similarity. If your contradiction payloads truncate claims hard, lower it or
  attach fact ids to contradictions upstream.

Every check is wrapped: a failing check costs its own finding and nothing else.
An epistemic check must never be the reason a report fails to ship.
