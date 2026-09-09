import { confidenceLabel, extractDomain, parseReport, trustOf } from "../lib";
import {
  IconAlert, IconArrowRight, IconChart, IconCheckCircle, IconDoc,
} from "./icons";

/* Final report card — renders ONLY backend-produced content:
 * report markdown sections, findings events, decisions events. */

function FindingRow({ item, index, selected, onSelect }) {
  const claim = item?.claim || "";
  const domain = extractDomain(item?.source || "");
  const trust = trustOf(domain);
  return (
    <div
      className={`finding clickable${selected ? " selected" : ""}`}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === "Enter") onSelect(); }}
      title="Open verification detail"
    >
      <span className="finding-n">{index + 1}</span>
      <div style={{ flex: 1, minWidth: 0 }}>
        <p className="finding-text">{claim}</p>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginTop: 6 }}>
          {item?.source ? (
            <a
              className="src-link"
              href={item.source}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
            >
              {domain || "source"}
            </a>
          ) : (
            <span className="src-link" style={{ border: 0 }}>unsourced</span>
          )}
          <span className={`tag tone-${trust.tone === "good" ? "good" : trust.tone === "bad" ? "bad" : "muted"}`}>
            {trust.label}
          </span>
          {typeof item?.verified === "boolean" ? (
            <span className={`tag tone-${item.verified ? "good" : "bad"}`}>
              {item.verified ? "verified" : "unverified"}
            </span>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export default function AnswerCard({ run, selectedFinding, onSelectFinding }) {
  const decisions = Array.isArray(run.decisions) ? run.decisions : [];
  const findings = Array.isArray(run.findings) ? run.findings : [];
  const sections = parseReport(run.report || "");
  const recommended = decisions.find((d) => d && d.is_recommended);
  const risks = decisions.flatMap((d) => (d && d.risk_note ? [{ option: d.option_label, note: d.risk_note }] : []));

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

      <div className="answer-cols">
        <div>
          <h3>Key findings</h3>
          {findings.length > 0 ? (
            findings.slice(0, 8).map((f, i) => (
              <FindingRow
                key={`${i}-${(f?.claim || "").slice(0, 24)}`}
                item={f}
                index={i}
                selected={selectedFinding === f}
                onSelect={() => onSelectFinding(selectedFinding === f ? null : f)}
              />
            ))
          ) : (
            <p className="empty">No findings were extracted for this run.</p>
          )}
        </div>
        <div className="right">
          <h3>Risks</h3>
          {risks.length > 0 ? (
            risks.map((r, i) => (
              <div className="risk-row" key={i}>
                <IconAlert size={17} />
                <p>Option {r.option}: {r.note}</p>
              </div>
            ))
          ) : (
            <p className="empty">No explicit risks were recorded.</p>
          )}

          <h3 style={{ marginTop: 22 }}>Recommended next step</h3>
          <div className="risk-row">
            <span className="finding-n" style={{ width: 34, height: 34 }}>
              <IconArrowRight size={15} />
            </span>
            <p style={{ marginTop: 5 }}>
              {recommended
                ? `Proceed with Option ${recommended.option_label} — ${recommended.description}`
                : "Review the findings and evidence, then run a follow-up to go deeper."}
            </p>
          </div>
        </div>
      </div>

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
  return <AnswerCard run={run} selectedFinding={null} onSelectFinding={() => {}} />;
}
