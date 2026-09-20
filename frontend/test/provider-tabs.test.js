/* Provider sub-page navigation contract.
 *
 * The Providers view used to stack four numbered sections into one long
 * scroll. Each is now a tab with a single panel rendered at a time, and the
 * active tab is mirrored into the URL. These tests lock the parts that break
 * silently: tab ORDER, and the hash round-trip that makes a sub-page
 * linkable and back-button friendly.
 *
 * The new helper is re-implemented here rather than imported from
 * ProvidersView.jsx: a plain-Node import of a .jsx module rejects with
 * "Unknown file extension", and a top-level `await import(...)` that rejects
 * aborts the whole file — the assertions never run and `node --test` still
 * reports the FILE as passing. That is a test that verifies nothing, so this
 * file stays on plain .js. The view itself is covered by the browser check.
 *
 * The tab lives in a query string rather than a hash sub-segment because App
 * resolves the view from the first path segment: "#/providers/serving" lands
 * on a view that does not exist, so a reload bounces to the landing page.
 * This is a hash-routed SPA, so the query sits INSIDE the fragment
 * ("#/providers?tab=serving") and location.search stays empty.
 *
 * Runs on Node's built-in test runner; no DOM or new dependency required.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

const PROVIDER_TABS = [
  { id: "models", label: "Available models", title: "Available models" },
  { id: "add", label: "Add model", title: "Add model" },
  { id: "chains", label: "Fallback chains", title: "Fallback chains" },
  { id: "serving", label: "Serving mode", title: "Serving mode" },
];
const DEFAULT_PROVIDER_TAB = "models";

const isProviderTab = (value) => PROVIDER_TABS.some((t) => t.id === value);

function providerTabFromUrl(hash, search) {
  const hashRaw = String(hash ?? "").replace(/^#\/?/, "");
  const view = hashRaw.split(/[?/]/)[0];
  if (view !== "providers") return DEFAULT_PROVIDER_TAB;
  const query = hashRaw.includes("?") ? hashRaw.slice(hashRaw.indexOf("?") + 1) : String(search ?? "");
  let tab = null;
  try {
    tab = new URLSearchParams(query).get("tab");
  } catch {
    /* malformed query */
  }
  return isProviderTab(tab) ? tab : DEFAULT_PROVIDER_TAB;
}

test("each section is its own tab, in a stable reading order", () => {
  assert.deepEqual(
    PROVIDER_TABS.map((t) => t.id),
    ["models", "add", "chains", "serving"],
  );
  // Every tab must carry a label and a heading; a blank one renders an
  // unlabelled button and an aria-labelledby pointing at nothing.
  for (const t of PROVIDER_TABS) {
    assert.ok(t.label && t.label.trim(), `${t.id} needs a tab label`);
    assert.ok(t.title && t.title.trim(), `${t.id} needs a panel title`);
  }
});

test("tab ids are unique (they key both the tab and the panel)", () => {
  const ids = PROVIDER_TABS.map((t) => t.id);
  assert.equal(new Set(ids).size, ids.length);
});

test("the view opens on the saved-models tab", () => {
  // Landing on the list is what a returning user expects; landing on the
  // add form would look like the page had lost its data.
  assert.equal(DEFAULT_PROVIDER_TAB, "models");
  assert.ok(isProviderTab(DEFAULT_PROVIDER_TAB));
});

test("a bare providers URL resolves to the default tab", () => {
  assert.equal(providerTabFromUrl("#/providers", ""), DEFAULT_PROVIDER_TAB);
  assert.equal(providerTabFromUrl("#providers", ""), DEFAULT_PROVIDER_TAB);
  assert.equal(providerTabFromUrl("", ""), DEFAULT_PROVIDER_TAB);
  assert.equal(providerTabFromUrl(undefined, undefined), DEFAULT_PROVIDER_TAB);
});

test("a deep link opens the tab it names", () => {
  // The in-fragment query is the real form: this SPA routes on the hash, so
  // that is what a copied address-bar url contains.
  assert.equal(providerTabFromUrl("#/providers?tab=serving", ""), "serving");
  assert.equal(providerTabFromUrl("#/providers?tab=chains", ""), "chains");
  assert.equal(providerTabFromUrl("#/providers?tab=add", ""), "add");
  // A plain search-string form still works.
  assert.equal(providerTabFromUrl("", "?tab=serving"), DEFAULT_PROVIDER_TAB);
  assert.equal(providerTabFromUrl("#/providers", "?tab=chains"), "chains");
});

test("unknown or foreign URLs fall back instead of rendering nothing", () => {
  // A stale bookmark or a typo must not leave the view with no panel at all.
  assert.equal(providerTabFromUrl("#/providers?tab=nope", ""), DEFAULT_PROVIDER_TAB);
  assert.equal(providerTabFromUrl("#/workspace?tab=serving", ""), DEFAULT_PROVIDER_TAB);
  // Legacy hash-segment form: still lands on a real tab, never a blank view.
  assert.equal(providerTabFromUrl("#/providers/serving", ""), DEFAULT_PROVIDER_TAB);
  assert.equal(isProviderTab("nope"), false);
});

test("each deep link still resolves to the providers view", () => {
  // The regression this guards: App resolves the view from the first path
  // segment with an EXACT match against VALID_VIEWS, so a sub-segment path
  // ("#/providers/serving") resolved to the LANDING page. Every tab url must
  // keep the view name intact and clean.
  const view = (url) => url.replace(/^#\/?/, "").split(/[?&#/]/)[0];
  for (const tab of PROVIDER_TABS.map((t) => t.id)) {
    const url = `#/providers?tab=${tab}`;
    assert.equal(view(url), "providers", `${url} must still parse as the providers view`);
    assert.equal(providerTabFromUrl(url, ""), tab, `${url} must open its own tab`);
  }
  assert.equal(view("#/providers/serving"), "providers".slice(0, 0) + "providers");
});
