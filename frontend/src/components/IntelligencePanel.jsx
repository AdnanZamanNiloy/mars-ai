import { formatTime } from "../lib";
import { Planet } from "./Sidebar";
import {
  IconAgents, IconAlert, IconChart, IconCheck, IconCheckCircle, IconChevronUp,
  IconCoins, IconDoc, IconRefresh, IconSearch, IconShield, IconShieldCheck, IconTarget,
} from "./icons";

/* Mission Intelligence — every number comes from streamed run state.
 * Props: run (live run object or null), traceLog (timestamped event lines),
 * onCollapse/onExpand, collapsed, onResume, onReplay. */

function AgentRow({ icon: Ic, name, desc, status, meta, progress }) {
  const label = status === "done" ? "Complete" : status === "active" ? "Analyzing" : "Waiting";
  const color = status === "done" ? "var(--mint)" : status === "active" ? "var(--mars-soft)" : "var(--t3)";
  return (
    <div className="agent-row">
      <span className="agent-ic"><Ic size={17} /></span>
      <div className="agent-main">
        <div className="agent-top">
          <span className="agent-name">{name}</span>
          <span className="agent-status" style={{ color }}>{status === "waiting" ? "Waiting ⋯" : label}</span>
        </div>
        <div className="agent-desc">{desc}</div>
        <div className="bar"><div className={status === "done" ? "green" : ""} style={{ width: `${progress}%` }} /></div>
        {meta ? <div className="agent-desc" style={{ marginTop: 5, marginBottom: 0 }}>{meta}</div> : null}
      </div>
    </div>
  );
}

function TraceNode({ kind }) {
  if (kind === "done") {
    return (
      <span className="trace-check"><IconCheck size={9} /></span>
    );
  }
  if (kind === "warn") return <span className="trace-node"><span className="trace-dot" /></span>;
  if (kind === "active") return <span className="trace-node"><span className="trace-ring" /></span>;
  return <span className="trace-node"><span className="trace-blink" /></span>;
}

export default function IntelligencePanel({
  run, traceLog, collapsed, onCollapse, onExpand, onResume, onReplay, replaying,
}) {
  if (collapsed) return null;

  const agents = run ? deriveAgents(run) : [];
  const health = run ? deriveHealth(run) : [];
  const working = run && !run.done && !run.error ? agents.filter((a) => a.status === "active").length : 0;

  return (
    <aside className="intel">
      <div className="intel-head">
        <div>
          <h2>Mission Intelligence</h2>
          <p>
            {run ? (
              <><strong>{working} agents working</strong> · Real-time research</>
            ) : (
              "No active mission"
            )}
          </p>
        </div>
        <button className="icon-btn" onClick={onCollapse} title="Collapse panel" aria-label="Collapse panel">
          <IconChevronUp size={15} />
        </button>
      </div>

      <section className="intel-section">
        <h3>Agents</h3>
        {agents.length > 0 ? (
          agents.map((a) => <AgentRow key={a.name} {...a} />)
        ) : (
          <p className="empty">Agents appear here once a run starts.</p>
        )}
      </section>

      <section className="intel-section">
        <h3>Research health</h3>
        {health.length > 0 ? (
          health.map((r) => (
            <div className="health-row" key={r.label}>
              <r.icon size={15} className={`tone-${r.tone}`} />
              <span className="k">{r.label}</span>
              <span className="v" style={r.hot ? { color: "var(--mars-soft)" } : undefined}>{r.value}</span>
            </div>
          ))
        ) : (
          <p className="empty">No health signals yet.</p>
        )}
      </section>

      <section className="intel-section">
        <div className="budget-line">
          <span className="k">Budget usage</span>
          {run?.budget ? (
            <span className="v">
              ${run.budget.cost.toFixed(4)}{" "}
              <small>/ ${run.budget.limit != null ? run.budget.limit.toFixed(2) : "—"}</small>
            </span>
          ) : (
            <span className="v"><small>No spend yet</small></span>
          )}
        </div>
        <div className="bar" style={{ marginTop: 11 }}>
          <div style={{ width: `${budgetPct(run)}%` }} className={run?.budget?.overBudget ? "" : "green"} />
        </div>
        <div className="budget-sub">
          {run?.budget
            ? `${budgetPct(run)}%${run.budget.calls != null ? ` · ${run.budget.calls} LLM calls` : ""}${run.budget.overBudget ? " · budget reached" : ""}`
            : "Cost accrues as the model is called"}
        </div>
      </section>

      {run?.error && run?.resumable ? (
        <section className="intel-section">
          <button className="btn" onClick={onResume} disabled={run?.resuming} style={{ width: "100%" }}>
            <IconRefresh size={14} /> {run?.resuming ? "Resuming…" : "Resume from checkpoint"}
          </button>
        </section>
      ) : null}

      <section className="intel-section">
        <div className="intel-section-head">
          <h3>Live trace</h3>
          {run?.runId ? (
            <button className="link-btn" onClick={() => onReplay(run.runId)} disabled={replaying}>
              {replaying ? "Loading…" : "View replay →"}
            </button>
          ) : null}
        </div>
        <div className="trace">
          {traceLog.length > 0 ? (
            traceLog.map((t, i) => (
              <div className="trace-row" key={i}>
                <span className="trace-time">{formatTime(t.at)}</span>
                <TraceNode kind={t.kind} />
                <span className="trace-text">{t.text}</span>
              </div>
            ))
          ) : (
            <p className="empty">Pipeline events will stream here live.</p>
          )}
        </div>
      </section>

      <section className="intel-section">
        <div className="health-row">
          <IconRefresh size={15} className="tone-muted" />
          <span className="k">Estimated completion</span>
          <span className="v">{run && !run.done ? "in progress" : run?.done ? "done" : "—"}</span>
        </div>
      </section>
    </aside>
  );
}

export function IntelExpandButton({ onExpand }) {
  return (
    <button className="icon-btn intel-expand" onClick={onExpand} title="Expand panel" aria-label="Expand panel">
      <IconAgents size={15} />
    </button>
  );
}

/* ---------- derivation: stream state → panel widgets ---------- */

function deriveAgents(run) {
  const planned = run.plan.length;
  const hasSearch = run.snippets > 0;
  const hasFacts = run.findings.length > 0;
  const critiques = run.critiques.length;
  const done = run.done;

  const st = (isDone, isActive) => (isDone ? "done" : isActive ? "active" : "waiting");
  const pct = (isDone, isActive, base) => (isDone ? 100 : isActive ? base : 8);

  const active = !done && !run.error;
  return [
    {
      icon: IconTarget, name: "Planner", desc: planned ? "Strategy created" : "Decomposing your question…",
      status: st(planned > 0, active), meta: planned ? `${planned} sub-questions` : null, progress: pct(planned > 0, active, 55),
    },
    {
      icon: IconSearch, name: "Search", desc: hasSearch ? "Gathering sources…" : "Waiting for plan",
      status: st(hasFacts, active && planned > 0), meta: hasSearch ? `${run.snippets} sources` : null,
      progress: pct(hasFacts, active && planned > 0, hasSearch ? 65 : 30),
    },
    {
      icon: IconDoc, name: "Summarizer", desc: hasFacts ? "Extracting claims…" : "Waiting for evidence",
      status: st(hasFacts && critiques > 0, active && hasSearch), meta: hasFacts ? `${run.findings.length} claims` : null,
      progress: pct(hasFacts && critiques > 0, active && hasSearch, hasFacts ? 70 : 25),
    },
    {
      icon: IconShieldCheck, name: "Verifier", desc: run.verifiedCount > 0 ? "Checking claims…" : "Waiting for claims",
      status: st(critiques > 0, active && hasFacts), meta: run.verifiedCount > 0 ? `${run.verifiedCount} verified` : null,
      progress: pct(critiques > 0, active && hasFacts, run.verifiedCount > 0 ? 60 : 20),
    },
    {
      icon: IconShield, name: "Critic", desc: critiques > 0 ? "Stress-testing conclusions…" : "Waiting for verified claims",
      status: st(done && critiques > 0, active && hasFacts), meta: critiques > 0 ? `${critiques} review pass${critiques === 1 ? "" : "es"}` : null,
      progress: pct(done && critiques > 0, active && hasFacts, critiques > 0 ? 55 : 15),
    },
    {
      icon: IconChart, name: "Synthesizer", desc: done ? "Report delivered" : "Waiting for approval",
      status: st(done, active && critiques > 0), meta: null, progress: pct(done, active && critiques > 0, 40),
    },
  ];
}

function deriveHealth(run) {
  const verified = run.findings.filter((f) => f.verified === true).length;
  return [
    { icon: IconDoc, label: "Sources analyzed", value: String(run.snippets), tone: "muted" },
    { icon: IconCheckCircle, label: "Total claims", value: String(run.findings.length), tone: "muted" },
    { icon: IconCheckCircle, label: "Verified claims", value: String(verified), tone: "good" },
    { icon: IconAlert, label: "Critic passes", value: String(run.critiques.length), tone: run.critiques.length > 1 ? "warn" : "muted", hot: run.critiques.length > 1 },
    { icon: IconTarget, label: "Research confidence", value: typeof run.confidence === "number" ? `${Math.round(run.confidence * 100)}%` : "—", tone: "muted" },
  ];
}

function budgetPct(run) {
  const b = run?.budget;
  if (!b || b.limit == null || b.limit <= 0 || b.cost == null) return 0;
  return Math.max(0, Math.min(100, Math.round((b.cost / b.limit) * 100)));
}
