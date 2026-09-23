/* parseReport: the primary answer is now the synthesizer's own prose, with
 * pipeline metadata in a separate audit document. These tests lock in both
 * shapes: the adaptive report (no legacy wrapper) renders whole, and legacy
 * persisted reports still split on the historical H1 headings.
 */
import { test } from "node:test";
import assert from "node:assert/strict";

import { parseReport } from "../src/lib.js";

test("adaptive report renders the whole document as the answer", () => {
  const report = "AI is moving toward agentic systems [1].\n\n## Infrastructure\nThe compute build-out ... [2].";
  const sections = parseReport(report);
  assert.equal(sections.finalAnswer, report);
  assert.equal(sections.evidence, "");
});

test("legacy wrapper still splits on the historical headings", () => {
  const report = [
    "# Final Answer",
    "The answer text [1].",
    "",
    "# Supporting Evidence",
    "- a claim (https://a.example)",
    "",
    "# Contradictions",
    "- a vs b",
  ].join("\n");
  const sections = parseReport(report);
  assert.equal(sections.finalAnswer, "The answer text [1].");
  assert.match(sections.evidence, /a claim/);
  assert.match(sections.contradictions, /a vs b/);
});

test("empty report yields an empty answer, not a crash", () => {
  assert.deepEqual(parseReport(""), {
    finalAnswer: "", evidence: "", contradictions: "", decisionLayer: "",
  });
});
