import { useMemo, useState } from "react";
import { extractDomain, trustOf } from "../lib";

/* Evidence explorer — tabulates REAL claims from the thread's runs
 * (live findings events or replayed trace claims). Filter only; no synthesis. */

export function collectEvidence(messages) {
  const rows = [];
  for (const m of messages) {
    if (m.kind === "run" && m.run.findings.length > 0) {
      for (const f of m.run.findings) {
        rows.push({ ...f, runQuery: m.run.query, runId: m.run.runId });
      }
    } else if (m.kind === "replay") {
      for (const c of m.trace.claims || []) {
        rows.push({
          claim: c.claim,
          source: c.source_url,
          verified: c.verified === 1 || c.verified === true,
          confidence: c.confidence,
          runQuery: m.query,
          runId: m.runId,
        });
      }
    }
  }
  return rows;
}

export default function EvidenceView({ messages, onInspect }) {
  const [filter, setFilter] = useState("all");
  const rows = useMemo(() => collectEvidence(messages), [messages]);
  const shown = rows.filter((r) => {
    if (filter === "verified") return r.verified === true;
    if (filter === "unverified") return r.verified !== true;
    return true;
  });
  const verifiedCount = rows.filter((r) => r.verified === true).length;

  return (
    <div className="view anim-rise">
      <div className="view-head">
        <h2>Evidence</h2>
        <p>{rows.length} claims · {verifiedCount} verified · across this session</p>
      </div>
      {rows.length === 0 ? (
        <p className="empty">No evidence yet — run a mission or replay one to inspect its claims here.</p>
      ) : (
        <>
          <div className="chip-row" style={{ marginBottom: 16 }}>
            {[["all", `All (${rows.length})`], ["verified", `Verified (${verifiedCount})`], ["unverified", `Unverified (${rows.length - verifiedCount})`]].map(([k, label]) => (
              <button
                key={k}
                type="button"
                className="chip"
                style={filter === k ? { borderColor: "#52525b", color: "#fff", background: "rgba(255,255,255,.05)" } : undefined}
                onClick={() => setFilter(k)}
              >
                {label}
              </button>
            ))}
          </div>
          {shown.map((r, i) => {
            const domain = extractDomain(r.source || "");
            const trust = trustOf(domain);
            return (
              <div key={`${r.runId}-${i}`} className="mission-card" onClick={() => onInspect(r)} role="button" tabIndex={0}
                onKeyDown={(e) => { if (e.key === "Enter") onInspect(r); }}>
                <span className={`dot ${r.verified === true ? "done" : r.verified === false ? "warn" : "idle"}`} style={{ marginTop: 6 }} />
                <span className="body">
                  <p className="q" style={{ fontWeight: 500 }}>{r.claim}</p>
                  <span className="meta">
                    {r.source ? (
                      <a href={r.source} target="_blank" rel="noopener noreferrer" className="src-link"
                        onClick={(e) => e.stopPropagation()}>{domain || "source"}</a>
                    ) : <span>unsourced</span>}
                    <span className={`tag tone-${trust.tone === "good" ? "good" : trust.tone === "bad" ? "bad" : "muted"}`}>{trust.label}</span>
                    {typeof r.verified === "boolean" ? (
                      <span className={`tag tone-${r.verified ? "good" : "bad"}`}>{r.verified ? "verified" : "unverified"}</span>
                    ) : null}
                    {r.agent ? <span className="tag tone-muted">{r.agent}</span> : null}
                    {r.challenged ? <span className="tag tone-bad">challenged</span> : null}
                    {typeof r.confidence === "number" ? <span>{Math.round(r.confidence * 100)}% confident</span> : null}
                  </span>
                </span>
              </div>
            );
          })}
        </>
      )}
    </div>
  );
}
