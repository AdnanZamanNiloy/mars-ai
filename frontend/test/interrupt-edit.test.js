/* Interrupt-and-edit regression tests (frontend helpers).
 *
 * The bug class: after stopping generation and editing an earlier message,
 * the thread must truncate from that message so the resend replaces it,
 * without touching the chat's session identity.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

const store = new Map();
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
  clear: () => store.clear(),
};

const { truncateFromMessage } = await import("../src/lib.js");

const thread = () => ([
  { id: "u1", kind: "user", text: "first" },
  { id: "r1", kind: "run", run: { tempId: "t1", query: "first" } },
  { id: "u2", kind: "user", text: "second" },
  { id: "r2", kind: "run", run: { tempId: "t2", query: "second", aborted: true } },
]);

test("editing the interrupted message drops it and its run", () => {
  const out = truncateFromMessage(thread(), "u2");
  assert.deepEqual(out.map((m) => m.id), ["u1", "r1"]);
});

test("editing an earlier message drops everything after it", () => {
  const out = truncateFromMessage(thread(), "u1");
  assert.deepEqual(out, []);
});

test("unknown id leaves the thread untouched (nothing to edit)", () => {
  const before = thread();
  const out = truncateFromMessage(before, "does-not-exist");
  assert.equal(out, before);
});

test("non-array input degrades to an empty thread, never throws", () => {
  assert.deepEqual(truncateFromMessage(null, "u1"), []);
});

test("a cancelled partial run is inside the truncated region", () => {
  // The stopped run sits after the edited user turn, so it is removed and
  // cannot linger as a half-answer in the resent thread.
  const out = truncateFromMessage(thread(), "u2");
  assert.equal(out.some((m) => m.run?.aborted), false);
});
