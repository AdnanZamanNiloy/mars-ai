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
  const out = [];
  let list = [];
  let ordered = false;
  const flushList = (key) => {
    if (!list.length) return;
    const Tag = ordered ? "ol" : "ul";
    out.push(
      <Tag key={key} className="answer-list">
        {list.map((item, j) => (
          <li key={j}>{renderInline(item)}</li>
        ))}
      </Tag>
    );
    list = [];
    ordered = false;
  };
  blocks.forEach((block, i) => {
    const bulletLines = block.split("\n").map((l) => l.trim()).filter(Boolean);
    const isBullets = bulletLines.length > 0 && bulletLines.every((l) => /^[-*]\s+/.test(l));
    const isNumbered = bulletLines.length > 0 && bulletLines.every((l) => /^\d+[.)]\s+/.test(l));

    if (block.startsWith("### ")) {
      flushList(`list-${i}`);
      out.push(
        <h5 key={i} className="answer-subh">
          {renderInline(block.slice(4).trim())}
        </h5>
      );
    } else if (block.startsWith("## ")) {
      flushList(`list-${i}`);
      out.push(
        <h4 key={i} className="answer-h">
          {renderInline(block.slice(3).trim())}
        </h4>
      );
    } else if (block.startsWith("# ")) {
      flushList(`list-${i}`);
      out.push(
        <h4 key={i} className="answer-h">
          {renderInline(block.slice(2).trim())}
        </h4>
      );
    } else if (isBullets || isNumbered) {
      // A whole block of "- "/"* "/"1." lines is one list; a mixed line
      // (e.g. "- **Label** — text") still renders as a single item.
      flushList(`mix-${i}`);
      ordered = isNumbered;
      for (const line of bulletLines) {
        list.push(line.replace(/^(?:[-*]|\d+[.)])\s+/, ""));
      }
    } else if (/^\[\d+\]\s/.test(block)) {
      flushList(`list-${i}`);
      out.push(
        <p key={i} className="source-line">
          {renderInline(block)}
        </p>
      );
    } else {
      flushList(`list-${i}`);
      out.push(
        <p key={i} className="answer-para">
          {renderInline(block)}
        </p>
      );
    }
  });
  flushList("list-end");
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

function describeDegradation(degraded) {
  const labels = degraded.map((a) => AGENT_LABELS[a] || a);
  const extractive = degraded.filter((a) => EXTRACTIVE_AGENTS.has(a));
  const parts = [];
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
  return parts.join(" — ") + ".";
}

export default function AnswerCard({ run }) {
  const findings = Array.isArray(run.findings) ? run.findings : [];
  const sections = parseReport(run.report || "");
  const total = findings.length;
  const verified = findings.filter((f) => f && f.verified === true).length;
  const conflicts = sections.contradictions ? sections.contradictions.split(/\n+/).filter((l) => l.trim()).length : 0;
  const degraded = Array.isArray(run.degraded) ? run.degraded.filter(Boolean) : [];
  const degradedNotice = describeDegradation(degraded);
  const decisions = Array.isArray(run.decisions) ? run.decisions : [];
  const support = typeof run.answerSupport === "number" ? Math.round(run.answerSupport * 100) : null;

  const stats = [
    { icon: IconDoc, value: String(total), label: "Total claims", sub: `From ${run.snippets || "—"} sources`, tone: "muted" },
    { icon: IconCheckCircle, value: String(verified), label: "Verified claims", sub: total ? `${Math.round((verified / total) * 100)}% verification rate` : "No claims yet", tone: "good" },
    { icon: IconAlert, value: String(conflicts), label: "Conflicts", sub: conflicts ? "Requires attention" : "None detected", tone: conflicts ? "warn" : "muted" },
    { icon: IconChart, value: typeof run.confidence === "number" ? `${Math.round(run.confidence * 100)}%` : "—", label: "Overall confidence", sub: `${confidenceLabel(run.confidence)} confidence`, tone: "warn" },
  ];
  if (support !== null) {
    stats.push({
      icon: IconCheckCircle, value: `${support}%`, label: "Citation support",
      sub: "Cited sentences backed by verified evidence",
      tone: support >= 70 ? "good" : "warn",
    });
  }

  return (
    <div className="answer anim-rise">
      {degraded.length ? (
        <div className="degraded-banner" role="alert">
          <IconAlert size={16} />
          <div>
            <strong>Degraded run — treat with caution.</strong>{" "}
            {degradedNotice}
          </div>
        </div>
      ) : null}

      {sections.finalAnswer ? (
        <div className="answer-lead">{renderRichText(sections.finalAnswer)}</div>
      ) : (
        <p className="answer-lead">The final report did not include an executive summary.</p>
      )}

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

      <div className="evidence-block">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h3 style={{ fontSize: 15.5, fontWeight: 650, margin: 0 }}>Evidence summary</h3>
        </div>
        <p className="sub">
          {run.snippets > 0
            ? `Based on analysis from ${run.snippets} source${run.snippets === 1 ? "" : "s"} across the research axes.`
            : "Source statistics were not reported for this run."}
        </p>
        <div className="stat-grid">
          {stats.map((s) => (
            <div className="stat" key={s.label}>
              <span className="stat-ic"><s.icon size={17} className={`tone-${s.tone}`} /></span>
              <div>
                <b>{s.value}</b>
                <div className="lbl">{s.label}</div>
                <div className="sub2">{s.sub}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
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
