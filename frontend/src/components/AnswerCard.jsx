import { confidenceLabel, parseReport } from "../lib";
import {
  IconAlert, IconChart, IconCheckCircle, IconDoc,
} from "./icons";

/* Final report card — renders ONLY backend-produced content:
 * report markdown sections, findings events, decisions events. */

export default function AnswerCard({ run }) {
  const findings = Array.isArray(run.findings) ? run.findings : [];
  const sections = parseReport(run.report || "");
  const total = findings.length;
  const verified = findings.filter((f) => f && f.verified === true).length;
  const conflicts = sections.contradictions ? sections.contradictions.split(/\n+/).filter((l) => l.trim()).length : 0;

  const stats = [
    { icon: IconDoc, value: String(total), label: "Total claims", sub: `From ${run.snippets || "—"} sources`, tone: "muted" },
    { icon: IconCheckCircle, value: String(verified), label: "Verified claims", sub: total ? `${Math.round((verified / total) * 100)}% verification rate` : "No claims yet", tone: "good" },
    { icon: IconAlert, value: String(conflicts), label: "Conflicts", sub: conflicts ? "Requires attention" : "None detected", tone: conflicts ? "warn" : "muted" },
    { icon: IconChart, value: typeof run.confidence === "number" ? `${Math.round(run.confidence * 100)}%` : "—", label: "Overall confidence", sub: `${confidenceLabel(run.confidence)} confidence`, tone: "warn" },
  ];

  return (
    <div className="answer anim-rise">
      {sections.finalAnswer ? (
        <p className="answer-lead">{sections.finalAnswer}</p>
      ) : (
        <p className="answer-lead">The final report did not include an executive summary.</p>
      )}

      {sections.contradictions ? (
        <div className="contradiction">
          <strong>Contradictions detected</strong>
          {"\n"}{sections.contradictions}
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
    })),
    decisions: trace.decisions || [],
    snippets: (trace.sources || []).length,
  };
  return <AnswerCard run={run} />;
}
