/* Per-message pipeline trace isolation regression tests.
 *
 * The bug: one App-level traceLog array was shared by EVERY run card
 * (ThreadMessage received steps={traceLog}), so sending a new message in a
 * session appended to (or, via the reset band-aids, rewrote) the trace shown
 * on previous messages — no message had its own independent trace.
 *
 * The fix stores steps ON each message and routes every event through
 * pushTrace(tempId, entry) → applyTraceEntry(message.steps, entry). These
 * tests lock the helper contract the UI relies on: immutable appends, keyed
 * upserts, the cap, and — most importantly — that two messages never share
 * step state.
 *
 * Runs on Node's built-in test runner; no browser required.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

const { applyTraceEntry, TRACE_STEP_CAP } = await import("../src/lib.js");

test("applyTraceEntry appends a stamped step and does not mutate the input", () => {
  const before = [];
  const after = applyTraceEntry(before, { text: "Query received" });
  assert.equal(before.length, 0, "input array must not be mutated");
  assert.equal(after.length, 1);
  assert.equal(after[0].text, "Query received");
  assert.equal(after[0].kind, "active", "entries default to kind=active");
  assert.ok(after[0].at, "entries get a timestamp");
});

test("an explicit kind on the entry wins over the default", () => {
  const out = applyTraceEntry([], { text: "done step", kind: "done" });
  assert.equal(out[0].kind, "done");
});

test("keyed entry upserts in place instead of appending duplicates", () => {
  let steps = [];
  steps = applyTraceEntry(steps, { key: "search", text: "Evidence gathered (10 sources)" });
  steps = applyTraceEntry(steps, { key: "search", text: "Evidence gathered (80 sources)" });
  assert.equal(steps.length, 1, "same key must update, not duplicate");
  assert.equal(steps[0].text, "Evidence gathered (80 sources)");
});

test("a keyed entry with no prior match still appends", () => {
  let steps = applyTraceEntry([], { text: "Query received" });
  steps = applyTraceEntry(steps, { key: "findings", text: "Claims extracted" });
  assert.equal(steps.length, 2);
});

test(`steps are capped at ${TRACE_STEP_CAP} entries`, () => {
  let steps = [];
  for (let i = 0; i < TRACE_STEP_CAP + 25; i++) {
    steps = applyTraceEntry(steps, { text: `step ${i}` });
  }
  assert.equal(steps.length, TRACE_STEP_CAP);
  assert.equal(steps[steps.length - 1].text, `step ${TRACE_STEP_CAP + 24}`);
});

test("applyTraceEntry tolerates a non-array / undefined steps value", () => {
  assert.equal(applyTraceEntry(undefined, { text: "a" }).length, 1);
  assert.equal(applyTraceEntry(null, { text: "a" }).length, 1);
  assert.equal(applyTraceEntry("junk", { text: "a" }).length, 1);
});

/* The isolation guarantee itself: each message owns its steps array. */
test("each message keeps an independent trace — new message never touches previous", () => {
  // Message A (first run) accumulates its trace.
  let stepsA = [];
  stepsA = applyTraceEntry(stepsA, { text: "Query received" });
  stepsA = applyTraceEntry(stepsA, { text: "Understood: machine_learning" });
  const snapshotA = [...stepsA];

  // Message B (second run in the same session) starts from ITS OWN empty list.
  let stepsB = [];
  stepsB = applyTraceEntry(stepsB, { text: "Query received" });
  stepsB = applyTraceEntry(stepsB, { text: "Router: external evidence required" });

  // B's growth left A untouched (same contents, independent array).
  assert.deepEqual(stepsA.map((s) => s.text), snapshotA.map((s) => s.text));
  assert.equal(stepsA.length, 2, "message A must keep only its own 2 steps");
  assert.equal(stepsB.length, 2, "message B has only its own 2 steps");
  assert.notStrictEqual(stepsA, stepsB);

  // A's further growth (e.g. a late resume event) does not touch B either.
  stepsA = applyTraceEntry(stepsA, { text: "Final report delivered" });
  assert.equal(stepsA.length, 3);
  assert.equal(stepsB.length, 2);
  assert.ok(!stepsB.some((s) => s.text === "Final report delivered"));
});
