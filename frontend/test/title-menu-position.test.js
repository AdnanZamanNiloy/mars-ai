/* Regression: the title-bar dropdowns must be absolutely positioned so that
 * opening them can never stretch the header or move its bottom border.
 *
 * Root cause of the bug this locks: `.title-menu` shared a surface rule that
 * set z-index but no `position`, so it rendered as a static flex child of the
 * header and pushed the border down. These assertions are intentionally
 * text-level (no DOM): they fail if the positioning contract is ever removed.
 *
 * Runs on Node's built-in test runner; no dependency or browser required.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const rawCss = readFileSync(join(here, "..", "src", "layout.css"), "utf8");
// Strip /* ... */ comments first: a comment that mentions a selector (e.g.
// "its .title-menu-wrap anchor") would otherwise be merged into the next
// rule's selector list and break exact matching.
const css = rawCss.replace(/\/\*[\s\S]*?\*\//g, "");

/* Extract the bodies of all top-level rules whose selector list contains
 * `sel` as an exact selector (so `.title-menu` never matches
 * `.title-menu-wrap`). */
function ruleBody(sel) {
  const out = [];
  const re = /([^{}]+)\{([^}]*)\}/g;
  let match;
  while ((match = re.exec(css)) !== null) {
    const selectors = match[1].split(",").map((s) => s.trim());
    if (selectors.includes(sel)) out.push(match[2]);
  }
  return out.join("\n");
}

test(".title-menu is absolutely positioned (never a flow block)", () => {
  const body = ruleBody(".title-menu");
  assert.match(body, /position\s*:\s*absolute/, ".title-menu must be position: absolute");
});

test(".title-menu has a z-index so it paints above the header", () => {
  const body = ruleBody(".title-menu");
  assert.match(body, /z-index\s*:\s*\d+/, ".title-menu must declare a z-index");
});

test(".title-menu stays inside the header for short titles (left-anchored)", () => {
  // Left-anchored to the wrap (which begins at the header's left padding), so
  // a short title's menu cannot extend left past the header into the sidebar.
  const body = ruleBody(".title-menu:not(.library-menu)");
  assert.match(body, /left\s*:\s*0/, "title menu must left-anchor to stay in the header");
  assert.match(body, /right\s*:\s*auto/, "title menu must not right-anchor past the header edge");
});

test(".title-menu-wrap is a positioned anchor that claims free space", () => {
  const body = ruleBody(".title-menu-wrap");
  assert.match(body, /position\s*:\s*relative/, "wrap must establish the containing block");
  // flex-grow: the wrap must claim the bar's free space so a short title is
  // not clipped; it keeps min-width:0 so the text can still ellipsize.
  assert.match(body, /flex\s*:\s*1\s+1\s+auto/, "wrap must grow to give the title room");
});

test(".library-btn and .library-menu are styled and positioned", () => {
  assert.ok(ruleBody(".library-btn").length > 0, ".library-btn must have styling");
  const menu = ruleBody(".library-menu");
  assert.match(menu, /left\s*:\s*auto/, "Library menu anchors to the right to avoid overflow");
  assert.match(menu, /right\s*:\s*0/);
});

test(".page-title-bar is the relative anchor for the header", () => {
  const body = ruleBody(".page-title-bar");
  assert.match(body, /position\s*:\s*relative/, "header must be a positioned anchor");
});
