/* Available Model card width contract.
 *
 * The model cards regressed twice: first they were full-width rows, then a
 * `minmax(230px, 1fr)` grid — which still STRETCHES. With `1fr`, the tracks
 * share every leftover pixel, so a sparse list balloons each card to fill the
 * row and the section reads as one long band again. The fix is a FIXED track
 * width: `repeat(auto-fill, <width>)` packs as many content-sized columns as
 * fit and leaves the trailing space empty.
 *
 * This reads the stylesheet as text rather than the rendered DOM: the failure
 * mode is a CSS regression, and a plain-Node import of ProvidersView.jsx would
 * reject ("Unknown file extension"), aborting the file so `node --test` still
 * reports it as passing. The rendered result is covered by the browser check.
 *
 * Runs on Node's built-in test runner; no DOM or new dependency required.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const css = readFileSync(
  fileURLToPath(new URL("../src/components/providers.css", import.meta.url)),
  "utf8",
);

/* Isolate the base (non-media-query) `.pv-model-list` rule. */
function baseModelListRule() {
  const start = css.indexOf(".pv-model-list {");
  assert.notEqual(start, -1, ".pv-model-list rule is missing");
  return css.slice(start, css.indexOf("}", start));
}

test(".pv-model-list uses fixed-width tracks, never a stretching 1fr cell", () => {
  const rule = baseModelListRule();
  assert.match(rule, /auto-fill/, "should pack as many columns as fit");
  assert.match(rule, /--pv-tile-w/, "track width should come from the tile variable");
  assert.doesNotMatch(
    rule,
    /minmax\(\s*\d+px\s*,\s*1fr\s*\)/,
    "minmax(_, 1fr) lets cards stretch to fill the row — the exact regression",
  );
});

test("the tile width is a small, content-sized value", () => {
  const decl = css.match(/--pv-tile-w:\s*(\d+)px/);
  assert.ok(decl, "--pv-tile-w should be declared in px");
  const width = Number(decl[1]);
  assert.ok(width >= 160 && width <= 240, `tile width ${width}px is outside the compact range`);
});

test("cards cannot be widened by long content spilling out of the tile", () => {
  // Long model ids and base URLs are the realistic blow-out risk at this width.
  const cardRule = css.slice(css.indexOf(".pv-model-card {"));
  assert.match(cardRule.slice(0, cardRule.indexOf("}")), /min-width:\s*0/, "card needs min-width: 0");
  const code = css.slice(css.indexOf(".pv-model-meta code"));
  assert.match(code.slice(0, code.indexOf("}")), /text-overflow:\s*ellipsis/, "ids/URLs must ellipsize");
});

test("small screens still collapse to a single column", () => {
  const narrow = css.slice(css.indexOf("@media (max-width: 460px)"));
  assert.match(narrow, /\.pv-model-list\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)/s);
});
