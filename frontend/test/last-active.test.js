/* Session last-active label tests.
 *
 * The sidebar shows each chat's latest activity as a compact, human phrase
 * (never a raw percentage). These lock the format the UI depends on,
 * including calendar-aware "yesterday" and the short-date fallback.
 *
 * Runs on Node's built-in test runner with a fixed `now`, so it is
 * deterministic and timezone-independent.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

const { lastActiveLabel } = await import("../src/lib.js");

const NOW = new Date("2026-09-20T12:00:00").getTime();
const ago = (ms) => new Date(NOW - ms).toISOString();
const MIN = 60_000;
const HOUR = 60 * MIN;
const DAY = 24 * HOUR;

test("under a minute reads 'just now'", () => {
  assert.equal(lastActiveLabel(ago(30_000), NOW), "just now");
});

test("minutes are pluralised correctly", () => {
  assert.equal(lastActiveLabel(ago(2 * MIN), NOW), "2 min ago");
  assert.equal(lastActiveLabel(ago(59 * MIN), NOW), "59 min ago");
});

test("hours within the same calendar day", () => {
  assert.equal(lastActiveLabel(ago(HOUR), NOW), "1 hour ago");
  assert.equal(lastActiveLabel(ago(3 * HOUR), NOW), "3 hours ago");
});

test("the previous calendar day reads 'yesterday'", () => {
  assert.equal(lastActiveLabel(new Date("2026-09-19T12:00:00").toISOString(), NOW), "yesterday");
});

test("a few days ago counts days", () => {
  assert.equal(lastActiveLabel(new Date("2026-09-17T12:00:00").toISOString(), NOW), "3 days ago");
});

test("older than a week shows a short date", () => {
  assert.equal(lastActiveLabel(new Date("2026-09-10T12:00:00").toISOString(), NOW), "Sep 10");
});

test("invalid or missing input yields an empty label", () => {
  assert.equal(lastActiveLabel("", NOW), "");
  assert.equal(lastActiveLabel(null, NOW), "");
  assert.equal(lastActiveLabel("not-a-date", NOW), "");
});
