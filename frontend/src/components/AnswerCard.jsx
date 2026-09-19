import { confidenceLabel, parseReport } from "../lib";
import {
  IconAlert, IconChart, IconCheckCircle, IconDoc,
} from "./icons";
/* Final report card — renders ONLY backend-produced content:
 * report markdown sections, findings events, decisions events. */

/* Inline markdown renderer: the report body is markdown, so `**bold**`,
 * `*italic*`, `` `code` ``, and `[text](url)` must render as elements
 * instead of printing their markers literally. Returns an array of nodes so
 * it can nest inside <p>/<li> without wrapping in a block element. */
function renderInline(text) {
  const src = String(text ?? "");
  if (!src) return src;
  const nodes = [];
  // Order matters: links, then bold, then italic, then code.
  const re = /(\[([^\]]+)\]\((https?:\/\/[^\s)]+)\))|(\*\*([^*]+)\*\*)|(\*([^*\n]+)\*)|(`([^`]+)`)/g;
  let last = 0;
  let m;
  let k = 0;
  while ((m = re.exec(src)) !== null) {
    if (m.index > last) nodes.push(src.slice(last, m.index));
    if (m[2] && m[3]) {
      nodes.push(
        <a key={k++} href={m[3]} target="_blank" rel="noreferrer noopener">{m[2]}</a>
      );
    } else if (m[5] !== undefined) {
      nodes.push(<strong key={k++}>{m[5]}</strong>);
    } else if (m[7] !== undefined) {
      nodes.push(<em key={k++}>{m[7]}</em>);
    } else if (m[9] !== undefined) {
      nodes.push(<code key={k++}>{m[9]}</code>);
    }
    last = re.lastIndex;
  }
  if (last < src.length) nodes.push(src.slice(last));
  return nodes;
}

/* Minimal rich renderer for the report body: `## ` section headers,
 * `- `/`1. ` list items, and inline markdown become real hierarchy.
 * Backend owns the words — this only maps markers to elements. */
function renderRichText(body) {
  const blocks = String(body || "")
    .split(/\n{2,}/)
    .map((b) => b.trim())
    .filter(Boolean);
  // Line-oriented parser. A numbered/bulleted block is a SINGLE list whose
  // items each start on their own line; continuation lines (indented or
  // wrapped) append to the current item instead of being merged into one
  // paragraph. This fixes "1. a 2. b 3. c" collapsing into a single line when
  // the items are not separated by blank lines.
  const out = [];
  let list = [];
  let ordered = false;
  let key = 0;
  const flushList = () => {
    if (!list.length) return;
    const Tag = ordered ? "ol" : "ul";
    out.push(
      <Tag key={`list-${key++}`} className="answer-list">
        {list.map((item, j) => (
          <li key={j}>{renderInline(item)}</li>
        ))}
      </Tag>
    );
    list = [];
    ordered = false;
  };

  const BULLET_RE = /^\s*[-*+]\s+(.*)$/;
  const NUMBER_RE = /^\s*\d+[.)]\s+(.*)$/;

  blocks.forEach((block) => {
    // A blank-line-separated block may itself contain several lines (list
    // items and/or plain wrapped text). Walk them in order.
    const lines = block.split("\n");
    // Whole-block headings/paragraphs (single logical block).
    const first = block.trim();
    if (list.length === 0) {
      if (first.startsWith("### ")) {
        out.push(<h5 key={`h-${key++}`} className="answer-subh">{renderInline(first.slice(4).trim())}</h5>);
        return;
      }
      if (first.startsWith("## ")) {
        out.push(<h4 key={`h-${key++}`} className="answer-h">{renderInline(first.slice(3).trim())}</h4>);
        return;
      }
      if (first.startsWith("# ")) {
        out.push(<h4 key={`h-${key++}`} className="answer-h">{renderInline(first.slice(2).trim())}</h4>);
        return;
      }
    }
    if (/^\s*\[\d+\]\s/.test(block)) {
      flushList();
      out.push(<p key={`src-${key++}`} className="source-line">{renderInline(block)}</p>);
      return;
    }

    let sawListLine = false;
    for (const raw of lines) {
      const line = raw.replace(/\s+$/, "");
      const num = line.match(NUMBER_RE);
      const bul = line.match(BULLET_RE);
      if (num || bul) {
        // New item. If the current list type changes, close and start fresh.
        const lineOrdered = Boolean(num);
        if (list.length && lineOrdered !== ordered) flushList();
        if (!list.length) ordered = lineOrdered;
        list.push((num ? num[1] : bul[1]).trim());
        sawListLine = true;
      } else if (list.length && line.trim() && /^\s+/.test(line)) {
        // Indented continuation of the current item.
        list[list.length - 1] = `${list[list.length - 1]} ${line.trim()}`;
      } else {
        // Plain text line: close any open list, then emit a paragraph.
        flushList();
        if (line.trim()) {
          out.push(<p key={`p-${key++}`} className="answer-para">{renderInline(line.trim())}</p>);
        }
      }
    }
    if (sawListLine) flushList();
  });
  flushList();
  return out;
}

/* Describe a degraded run honestly from the agents that actually fell back.
 * The old banner hardcoded "LLM providers were unavailable ... extractive,
 * uncited ... confidence is capped" for EVERY degraded agent, which is wrong:
 * an agent can fall back for a bad prompt, a size rejection or an invalid
 * model response, not only a provider outage; the synthesizer's fallback
 * DOES cite its claims; and only summarizer/synthesizer fallbacks cap
 * confidence. Names below mirror the backend's EXTRACTIVE_FALLBACK_AGENTS. */
const EXTRACTIVE_AGENTS = new Set(["summarizer", "synthesizer"]);
const AGENT_LABELS = {
  intent: "intent classification",
  planner: "planning",
  summarizer: "evidence extraction",
  verifier: "verification",
  critic: "critique",
  synthesizer: "synthesis",
  redteam: "red-team review",
};
// Backend reason codes (app/core/degradation.py + agent-local codes): WHY a
// stage degraded. Provider causes and evidence causes must read differently —
// a provider outage is not thin evidence.
const REASON_LABELS = {
  "provider-transient": "the LLM provider was temporarily unavailable (rate limit, timeout or outage)",
  "provider-hard": "the LLM provider rejected the request permanently (auth, quota or size)",
  provider_timeout: "the LLM provider timed out",
  providers_unavailable: "no LLM provider could serve the request",
  llm_error: "the LLM call failed",
  "weak-evidence": "the extracted evidence was too thin or unusable",
  no_facts_parsed: "no usable claims could be parsed from the sources",
  llm_returned_no_facts: "the model returned no usable claims",
  payload_too_large: "the source payload was too large for the provider",
};

function reasonText(reason) {
  return REASON_LABELS[reason] || (reason ? String(reason).replace(/_/g, " ") : "");
}

function describeDegradation(degraded, reasons = {}, providerDegraded = false) {
  const labels = degraded.map((a) => AGENT_LABELS[a] || a);
  const extractive = degraded.filter((a) => EXTRACTIVE_AGENTS.has(a));
  const parts = [];
  // Name the concrete causes first, deduped — this is what distinguishes a
  // provider-transient run from a genuinely weak-evidence run.
  const causes = [];
  for (const agent of degraded) {
    const text = reasonText(reasons[agent]);
    if (text && !causes.includes(text)) causes.push(text);
  }
  if (providerDegraded && !causes.some((c) => /provider/.test(c))) {
    causes.push("an LLM provider failed during the run");
  }
  if (extractive.length) {
    parts.push(
      `${labels.join(", ")} ran on deterministic extraction instead of a model-generated response`
    );
    parts.push(
      "claims are unrewritten source text, so the answer is less polished and confidence is capped"
    );
  } else {
    parts.push(
      `the model response for ${labels.join(", ")} was unavailable or unusable, so a rule-based fallback was used`
    );
    parts.push("treat those sections as lower-assurance");
  }
  if (causes.length) parts.push(`cause: ${causes.join("; ")}`);
  return parts.join(" — ") + ".";
}

export default function AnswerCard({ run }) {
  const findings = Array.isArray(run.findings) ? run.findings : [];
  const sections = parseReport(run.report || "");
  const total = findings.length;
  const verified = findings.filter((f) => f && f.verified === true).length;
  const conflicts = sections.contradictions ? sections.contradictions.split(/\n+/).filter((l) => l.trim()).length : 0;
  const degraded = Array.isArray(run.degraded) ? run.degraded.filter(Boolean) : [];
  const degradedNotice = degraded.length
    ? describeDegradation(degraded, run.degradedReasons || {}, run.providerDegraded === true)
    : "";
  const decisions = Array.isArray(run.decisions) ? run.decisions : [];
  const support = typeof run.answerSupport === "number" ? Math.round(run.answerSupport * 100) : null;
  const outlineSections = run.outline && Array.isArray(run.outline.sections) ? run.outline.sections : [];

  return (
    <div className="answer anim-rise">
      {(degraded.length || run.providerDegraded === true) ? (
        <div className="degraded-banner" role="alert">
          <IconAlert size={16} />
          <div>
            <strong>Degraded run — treat with caution.</strong>{" "}
            {degradedNotice ||
              "an LLM provider failed during the run, so some output came from deterministic fallbacks."}
          </div>
        </div>
      ) : null}

      {run.directAnswer ? (
        <div className="answer-lead">{renderRichText(run.directAnswer.answer || "")}</div>
      ) : sections.finalAnswer ? (
        <div className="answer-lead">{renderRichText(sections.finalAnswer)}</div>
      ) : (
        <p className="answer-lead">The final report did not include an executive summary.</p>
      )}

      {outlineSections.length ? (
        <div className="outline-block">
          <h3 style={{ fontSize: 15.5, fontWeight: 650, margin: "0 0 8px" }}>
            Report outline{run.sectionWise ? " (section-wise synthesis)" : ""}
          </h3>
          <ol className="outline-list">
            {outlineSections.map((s) => (
              <li key={`${s.axis}-${s.title}`}>
                <b>{s.title}</b>
                {s.coverage_goal ? <span className="sub2"> — {s.coverage_goal}</span> : null}
              </li>
            ))}
          </ol>
        </div>
      ) : null}

      {sections.contradictions ? (
        <div className="contradiction">
          <strong>Contradictions detected</strong>
          {"\n"}{sections.contradictions}
        </div>
      ) : null}

      {decisions.length ? (
        <div className="decisions-block">
          <h3 style={{ fontSize: 15.5, fontWeight: 650, margin: "0 0 8px" }}>Decision layer</h3>
          {decisions.map((d) => (
            <div className={`decision-option${d.is_recommended ? " recommended" : ""}`} key={d.option_label}>
              <div className="decision-head">
                <b>Option {d.option_label}{d.is_recommended ? " — recommended" : ""}</b>
              </div>
              {d.description ? <p>{d.description}</p> : null}
              {d.rationale ? <p className="sub2">Rationale: {d.rationale}</p> : null}
              {d.risk_note ? <p className="sub2">Risk: {d.risk_note}</p> : null}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function ReplayAnswerCard({ trace }) {
  /* Read-only replay built from the persisted trace — same card, zero new fetches. */
  const run = {
    report: trace.final_report?.report_markdown || "",
    confidence: typeof trace.final_report?.confidence === "number" ? trace.final_report.confidence : null,
    findings: (trace.claims || []).map((c) => ({
      claim: c.claim,
      source: c.source_url,
      verified: c.verified === 1 || c.verified === true,
      confidence: c.confidence,
      agent: c.agent || "",
      challenged: c.challenged === 1 || c.challenged === true,
    })),
    decisions: trace.decisions || [],
    snippets: (trace.sources || []).length,
  };
  return <AnswerCard run={run} />;
}
