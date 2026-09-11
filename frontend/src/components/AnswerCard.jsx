import { confidenceLabel, parseReport } from "../lib";
import {
  IconAlert, IconChart, IconCheckCircle, IconDoc,
} from "./icons";
/* Final report card — renders ONLY backend-produced content:
 * report markdown sections, findings events, decisions events. */

/* Minimal rich renderer for the report body: `## ` section headers and
 * `- ` finding bullets become real hierarchy; everything else stays a
 * paragraph. Backend owns the words — this only maps markers to elements. */
function renderRichText(body) {
  const blocks = String(body || "")
    .split(/\n{2,}/)
    .map((b) => b.trim())
    .filter(Boolean);
  const out = [];
  let list = [];
  const flushList = (key) => {
    if (!list.length) return;
    out.push(
      <ul key={key} className="answer-list">
        {list.map((item, j) => (
          <li key={j}>{item}</li>
        ))}
      </ul>
    );
    list = [];
  };
  blocks.forEach((block, i) => {
    if (block.startsWith("### ")) {
      flushList(`list-${i}`);
      out.push(
        <h5 key={i} className="answer-subh">
          {block.slice(4).trim()}
        </h5>
      );
    } else if (block.startsWith("## ")) {
      flushList(`list-${i}`);
      out.push(
        <h4 key={i} className="answer-h">
          {block.slice(3).trim()}
        </h4>
      );
    } else if (block.startsWith("- ")) {
      // A bullet block may hold several "- " items joined by single
      // newlines — render each as its own list item.
      for (const item of block.split("\n")) {
        const text = item.trim();
        if (text.startsWith("- ")) list.push(text.slice(2).trim());
      }
    } else if (/^\[\d+\]\s/.test(block)) {
      flushList(`list-${i}`);
      out.push(
        <p key={i} className="source-line">
          {block}
        </p>
      );
    } else {
      flushList(`list-${i}`);
      out.push(
        <p key={i} className="answer-para">
          {block}
        </p>
      );
    }
  });
  flushList("list-end");
  return out;
}

export default function AnswerCard({ run }) {
  const findings = Array.isArray(run.findings) ? run.findings : [];
  const sections = parseReport(run.report || "");
  const total = findings.length;
  const verified = findings.filter((f) => f && f.verified === true).length;
  const conflicts = sections.contradictions ? sections.contradictions.split(/\n+/).filter((l) => l.trim()).length : 0;
  const degraded = Array.isArray(run.degraded) ? run.degraded.filter(Boolean) : [];
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
            LLM providers were unavailable, so {degraded.join(", ")} produced
            deterministic (extractive, uncited) output instead of model-written
            analysis. Confidence is capped accordingly.
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
