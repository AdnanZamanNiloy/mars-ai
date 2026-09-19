/* Chat-session persistence regression tests (frontend helpers).
 *
 * The bug: every question created a new session. These lock in the helper
 * contract that App.jsx relies on — one stable active session id, a chat
 * list keyed by sessionId (not runId), and New Chat minting a new id.
 *
 * Runs on Node's built-in test runner with a tiny localStorage stub, so no
 * new dependency and no browser is required.
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

const {
  newSessionId,
  loadActiveSessionId,
  saveActiveSessionId,
  upsertMission,
  updateMission,
  removeMission,
  loadMissions,
} = await import("../src/lib.js");

test("active session id persists across reload", () => {
  store.clear();
  assert.equal(loadActiveSessionId(), "");
  saveActiveSessionId("chat-1");
  // Simulate reload: read straight from the stub storage.
  assert.equal(loadActiveSessionId(), "chat-1");
});

test("newSessionId is non-empty and unique", () => {
  const a = newSessionId();
  const b = newSessionId();
  assert.ok(a && typeof a === "string");
  assert.notEqual(a, b);
});

test("two runs in one chat produce ONE mission keyed by sessionId", () => {
  store.clear();
  upsertMission({ sessionId: "chat-1", runId: "run-a", query: "q1", status: "running" });
  upsertMission({ sessionId: "chat-1", runId: "run-b", query: "q2", status: "completed" });
  const list = loadMissions();
  assert.equal(list.length, 1, "follow-up must not create a second chat");
  assert.equal(list[0].sessionId, "chat-1");
  // Latest run is recorded on the same session.
  assert.equal(list[0].runId, "run-b");
});

test("a different sessionId creates a separate chat (New Chat)", () => {
  store.clear();
  upsertMission({ sessionId: "chat-1", runId: "run-a", query: "q1" });
  upsertMission({ sessionId: "chat-2", runId: "run-b", query: "q2" });
  const ids = loadMissions().map((m) => m.sessionId).sort();
  assert.deepEqual(ids, ["chat-1", "chat-2"]);
});

test("updateMission and removeMission target the session, not the run", () => {
  store.clear();
  upsertMission({ sessionId: "chat-1", runId: "run-a", query: "q1" });
  upsertMission({ sessionId: "chat-2", runId: "run-b", query: "q2" });
  updateMission("chat-1", { title: "renamed", pinned: true });
  const afterUpdate = loadMissions();
  const c1 = afterUpdate.find((m) => m.sessionId === "chat-1");
  assert.equal(c1.title, "renamed");
  assert.equal(c1.pinned, true);
  // Removing chat-1 leaves chat-2 untouched.
  const afterRemove = removeMission("chat-1");
  assert.deepEqual(afterRemove.map((m) => m.sessionId), ["chat-2"]);
});
