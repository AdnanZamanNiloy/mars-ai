/* Conversation auto-scroll regression tests (frontend helper).
 *
 * Contract: following the stream continues while the viewport is near the
 * bottom, and STOPS once the user scrolls up to read older messages.
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

const { shouldAutoScroll } = await import("../src/lib.js");

test("at the very bottom: keep following", () => {
  assert.equal(
    shouldAutoScroll({ scrollHeight: 1000, scrollTop: 400, clientHeight: 600 }),
    true,
  );
});

test("within the threshold (chunk just arrived): keep following", () => {
  // distance = 100 < 160
  assert.equal(
    shouldAutoScroll({ scrollHeight: 1000, scrollTop: 300, clientHeight: 600 }),
    true,
  );
});

test("user scrolled up to read older messages: do NOT force-scroll", () => {
  // distance = 400 >= 160
  assert.equal(
    shouldAutoScroll({ scrollHeight: 1000, scrollTop: 0, clientHeight: 600 }),
    false,
  );
});

test("exactly at the threshold boundary is not following", () => {
  // distance = 160, not < 160
  assert.equal(
    shouldAutoScroll({ scrollHeight: 1000, scrollTop: 240, clientHeight: 600 }),
    false,
  );
});

test("short response that fits: always at bottom, keep following", () => {
  assert.equal(
    shouldAutoScroll({ scrollHeight: 500, scrollTop: 0, clientHeight: 600 }),
    true,
  );
});
