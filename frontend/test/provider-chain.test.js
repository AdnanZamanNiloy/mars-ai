/* Provider fallback chain helper tests (frontend).
 *
 * Locks the contract ProvidersView/ChainsSection rely on:
 *   - member #1 is always the "Primary", later members are "Fallback N"
 *   - chain membership resolves in ORDER and drops dangling ids
 *   - providers not yet in a chain are offered as "available"
 *
 * Runs on Node's built-in test runner; no DOM or new dependency required.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

const { chainRole, resolveChainMembers } = await import("../src/lib.js");

const P = (id, name) => ({ id, name, model: `${name}-model`, base_url: `https://${name}.example/v1` });
const PROVIDERS = [P(1, "alpha"), P(2, "beta"), P(3, "gamma")];

test("first chain member is the primary; later members are ordered fallbacks", () => {
  assert.equal(chainRole(0).label, "Primary");
  assert.equal(chainRole(0).tone, "primary");
  assert.equal(chainRole(1).label, "Fallback 1");
  assert.equal(chainRole(2).label, "Fallback 2");
  assert.equal(chainRole(2).tone, "fallback");
});

test("members resolve in the configured order, not provider order", () => {
  const { members } = resolveChainMembers(PROVIDERS, [3, 1, 2]);
  assert.deepEqual(members.map((m) => m.name), ["gamma", "alpha", "beta"]);
});

test("providers not in the chain are offered as available", () => {
  const { members, available } = resolveChainMembers(PROVIDERS, [2]);
  assert.deepEqual(members.map((m) => m.id), [2]);
  assert.deepEqual(available.map((m) => m.id).sort(), [1, 3]);
});

test("a dangling member id (deleted provider) is dropped, not rendered blank", () => {
  const { members, available } = resolveChainMembers(PROVIDERS, [1, 999, 3]);
  assert.deepEqual(members.map((m) => m.id), [1, 3]);
  assert.equal(available.length, 1);
});

test("an empty chain has no members and all providers available", () => {
  const { members, available } = resolveChainMembers(PROVIDERS, []);
  assert.deepEqual(members, []);
  assert.equal(available.length, 3);
});

test("the same provider can never appear twice in a resolved chain", () => {
  const { members } = resolveChainMembers(PROVIDERS, [1, 1, 2]);
  assert.deepEqual(members.map((m) => m.id), [1, 2]);
});
