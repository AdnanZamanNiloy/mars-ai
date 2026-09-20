import {
  IconAlert, IconChart, IconCheckCircle, IconChevronRight,
  IconDoc, IconRefresh, IconRoute, IconSearch, IconShield, IconShieldCheck, IconTarget,
} from "./icons";

/* Mission Intelligence — the run's instrument rail.
 *
 * PRINCIPLE: every number here is a value the stream actually delivered.
 * There are no invented progress percentages and no per-agent "estimated"
 * bars. An agent is done, working, or queued — decided by which artifacts
 * exist — and that is all we claim to know. When a section has no data we
 * hide it; we never pad the rail with placeholders that fill the column but
 * carry no signal.
 *
 * Props: run (live run object, replay-shaped run, or null),
 *        collapsed, onCollapse, onResume.
 */

const STATE_LABEL = { done: "Complete", active: "Working", waiting: "Queued" };

/* Canonical pipeline — same seven agents, same order, as AgentsView and the
 * live RunProgress strip, so "agent 3" means the same thing everywhere. */
const PIPELINE = [
  { key: "orchestrator", icon: IconTarget, name: "Orchestrator" },
  { key: "planner", icon: IconTarget, name: "Planner" },
  { key: "search", icon: IconSearch, name: "Search" },
  { key: "summarizer", icon: IconDoc, name: "Summarizer" },
  { key: "verifier", icon: IconShieldCheck, name: "Verifier" },
  { key: "critic", icon: IconShield, name: "Critic" },
  { key: "synthesizer", icon: IconChart, name: "Synthesizer" },
];

function AgentRow({ icon: Ic, name, desc, status, index }) {
  const label = STATE_LABEL[status] || "Queued";
  return (
    <li className={`spine-node s-${status}`}>
      <span className="spine-idx">{status === "done" ? "\u2713" : (index ?? "")}</span>
      <span className="spine-body">
        <span className="spine-title">
          <Ic size={13} className="spine-ic" />
          {name}
          <span className="spine-state">{label}</span>
        </span>
        <span className="spine-desc">{desc}</span>
      </span>
    </li>
  );
}

export default function IntelligencePanel({
  run, collapsed, onCollapse, onResume,
}) {
  if (collapsed) return null;

  const agents = run ? deriveAgents(run) : [];
  const health = run ? deriveHealth(run) : [];
  const working = agents.filter((a) => a.status === "active").length;

  return (
    <aside className="intel" aria-label="Research intelligence">
      <div className="intel-head">
        <div>
          <h2>Intelligence</h2>
          <p aria-live="polite" aria-atomic="true">
            {run ? (
              run.replay ? (
                <><strong>Session replay</strong> · figures from the stored trace</>
              ) : run.done ? (
                <><strong>Run complete</strong> · figures from the finished trace</>
              ) : (
                <><strong>{working} agent{working === 1 ? "" : "s"} working</strong> · streaming live</>
              )
            ) : (
              "No active run"
            )}
          </p>
        </div>
        <button className="icon-btn" onClick={onCollapse} title="Collapse panel" aria-label="Collapse the intelligence panel">
          <IconChevronRight size={15} />
        </button>
      </div>

      {!run ? (
        <IdleState />
      ) : (
        <>
          <section className="intel-section">
            <h3>Pipeline</h3>
            <ol className="spine">
              {agents.map((a, i) => <AgentRow key={a.name} index={i + 1} {...a} />)}
            </ol>
          </section>

          <section className="intel-section">
            <h3>Research health</h3>
            {health.map((r) => (
              <div className="health-row" key={r.label}>
                <r.icon size={15} className={`tone-${r.tone}`} />
                <span className="k">{r.label}</span>
                <span className="v" style={r.hot ? { color: "var(--mars-soft)" } : undefined}>{r.value}</span>
              </div>
            ))}
            <CitationHealthRow health={run.citationHealth} />
          </section>

          <ConfidenceBreakdown breakdown={run.breakdown} />
          <RedTeamPanel redteam={run.redteam} />
          <EvidenceGrades distribution={run.evidenceDistribution} />
          <BudgetMeter budget={run.budget} />

          {run.error && run.resumable ? (
            <section className="intel-section">
              <button className="btn" onClick={onResume} disabled={run.resuming} style={{ width: "100%" }}>
                <IconRefresh size={14} /> {run.resuming ? "Resuming…" : "Resume from checkpoint"}
              </button>
            </section>
          ) : null}
        </>
      )}
    </aside>
  );
}

/* The idle rail is orientation, not a column of "nothing yet". It explains
 * what this panel becomes once a run starts, in one compact block. */
function IdleState() {
  return (
    <section className="intel-section intel-idle">
      <p className="intel-lede">
        Ask a question and this rail becomes the run's instrument panel:
      </p>
      <ul className="intel-idle-list">
        <li><IconTarget size={14} /> The seven-agent pipeline as it advances</li>
        <li><IconChart size={14} /> A confidence breakdown from measured signals</li>
        <li><IconShieldCheck size={14} /> Claim verification and citation health</li>
        <li><IconAlert size={14} /> Cost, budget and adversarial review</li>
      </ul>
    </section>
  );
}

/* ---------- derivation: stream state → panel widgets ---------- */

const SIGNAL_LABELS = {
  source_quality: "Source quality",
  source_diversity: "Source diversity",
  citation_coverage: "Claims verified",
  citation_support: "Citation support",
  axis_coverage: "Axis coverage",
  claim_verification_strength: "Source-term overlap",
  claim_evidence_quality: "Claim evidence quality",
  cross_source_agreement: "Cross-source agreement",
  critic_survival: "Critic survival",
  freshness: "Freshness",
};

/* Unknown keys are humanised rather than printed raw, so a new backend
 * signal never shows up as snake_case in the console. */
function signalLabel(key) {
  if (SIGNAL_LABELS[key]) return SIGNAL_LABELS[key];
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function ConfidenceBreakdown({ breakdown }) {
  const signals = breakdown?.signals;
  if (!signals || typeof signals !== "object") return null;
  const overall = typeof breakdown.overall === "number" ? Math.round(breakdown.overall * 100) : null;
  return (
    <section className="intel-section">
      <h3>Confidence breakdown</h3>
      {overall !== null ? (
        <div className="health-row">
          <IconChart size={15} className="tone-muted" />
          <span className="k"><strong>Overall</strong></span>
          <span className="v">{overall}%</span>
        </div>
      ) : null}
      {Object.entries(signals).map(([key, value]) => {
        const pct = typeof value === "number" ? Math.round(value * 100) : null;
        return (
          <div key={key}>
            <div className="health-row" style={{ paddingBottom: 2 }}>
              <span className="k">{signalLabel(key)}</span>
              <span className="v">{pct !== null ? `${pct}%` : "—"}</span>
            </div>
            <div className="bar" style={{ marginBottom: 8 }}>
              <div style={{ width: `${pct ?? 0}%` }} />
            </div>
            {key === "freshness" && pct === 0 ? (
              <div className="budget-sub note" style={{ marginBottom: 8 }}>
                publish dates not captured yet
              </div>
            ) : null}
          </div>
        );
      })}
    </section>
  );
}

function BudgetMeter({ budget }) {
  /* Rendered only when the run actually reported telemetry, so a replay
   * (which carries none) does not show a dead "arrives later" section. */
  if (!budget || typeof budget !== "object") return null;
  const tokens = budget.spent_tokens ?? 0;
  const calls = budget.llm_calls ?? 0;
  const usd = typeof budget.spent_usd === "number" ? budget.spent_usd : 0;
  const util = typeof budget.utilization === "number" ? Math.round(budget.utilization * 100) : null;
  const hits = budget.cache_hits ?? 0;
  const hitRate = budget.cache_hit_rate ?? 0;
  const fmtTokens = tokens >= 1000 ? `${(tokens / 1000).toFixed(1)}k` : String(tokens);
  return (
    <section className="intel-section">
      <h3>Cost &amp; budget</h3>
      <div className="health-row">
        <span className="k">LLM calls</span>
        <span className="v">{calls}{hits > 0 ? ` (${hits} cached)` : ""}</span>
      </div>
      <div className="health-row">
        <span className="k">Tokens spent</span>
        <span className="v">{fmtTokens}</span>
      </div>
      <div className="health-row">
        <span className="k">Estimated cost</span>
        <span className="v">${usd < 0.01 && usd > 0 ? usd.toFixed(4) : usd.toFixed(3)}</span>
      </div>
      {util !== null ? (
        <>
          <div className="health-row" style={{ paddingBottom: 2 }}>
            <span className="k">Budget utilization</span>
            <span className="v" style={util >= 80 ? { color: "var(--mars-soft)" } : undefined}>{util}%</span>
          </div>
          <div className="bar" style={{ marginBottom: 8 }}>
            <div style={{ width: `${util}%`, background: util >= 80 ? "var(--mars-soft)" : undefined }} />
          </div>
        </>
      ) : null}
      {hits > 0 ? (
        <div className="budget-sub note">
          cache hit rate {Math.round(hitRate * 100)}% — repeated prompts served from disk, free
        </div>
      ) : null}
    </section>
  );
}

function CitationHealthRow({ health }) {
  const summary = health && typeof health.summary === "object" ? health.summary : null;
  if (!summary) return null;
  const broken = (summary.broken || 0) + (summary.bad || 0);
  const warn = summary.warn || 0;
  const ok = summary.ok || 0;
  const total = ok + warn + broken + (summary.unchecked || 0);
  if (!total) return null;
  const tone = broken ? "warn" : warn ? "muted" : "good";
  const label = broken
    ? `${broken} broken of ${total}`
    : warn ? `${warn} flagged of ${total}`
    : `${ok}/${total} verified live`;
  return (
    <div className="health-row">
      <IconShield size={15} className={`tone-${tone}`} />
      <span className="k">Citation health</span>
      <span className="v" style={broken ? { color: "var(--mars-soft)" } : undefined}>{label}</span>
    </div>
  );
}

function EvidenceGrades({ distribution }) {
  if (!distribution || typeof distribution !== "object") return null;
  const counts = ["A", "B", "C", "D"].map((g) => ({
    grade: g,
    count: Number(distribution[g]) || 0,
  }));
  const total = counts.reduce((sum, c) => sum + c.count, 0);
  if (!total) return null;
  return (
    <section className="intel-section">
      <h3>Evidence grades</h3>
      <div className="health-row">
        <IconShieldCheck size={15} className="tone-muted" />
        <span className="k">A/B/C/D</span>
        <span className="v">{counts.map((c) => `${c.grade}${c.count}`).join(" · ")}</span>
      </div>
    </section>
  );
}

function RedTeamPanel({ redteam }) {
  /* Adversarial review: only rendered when the critic actually reported
   * attacks. The survival score is computed from the findings, never taken
   * from the model. */
  const findings = Array.isArray(redteam?.findings) ? redteam.findings : [];
  const score = typeof redteam?.survival_score === "number"
    ? Math.round(redteam.survival_score * 100) : null;
  if (score === null && !findings.length) return null;
  const top = [...findings].sort((a, b) => (b.severity || 0) - (a.severity || 0)).slice(0, 4);
  const weak = score !== null && score < 60;
  return (
    <section className="intel-section">
      <h3>Red team</h3>
      <div className="health-row">
        <IconShield size={15} className={weak ? "tone-warn" : "tone-good"} />
        <span className="k">Evidence survival</span>
        <span className="v" style={weak ? { color: "var(--mars-soft)" } : undefined}>
          {score !== null ? `${score}%` : "—"}
        </span>
      </div>
      {top.map((f, i) => (
        <div key={`${f.kind}-${i}`} className="budget-sub note">
          <strong>{f.kind || "weakness"}</strong>
          {typeof f.severity === "number" ? ` (${Math.round(f.severity * 100)}%)` : ""} — {f.statement}
        </div>
      ))}
    </section>
  );
}

/* ---------- agent state ----------
 * Status is decided by which artifacts exist — the same monotonic ladder the
 * live pipeline strip uses — and each agent gets ONE honest descriptor drawn
 * from real counts. No estimated per-agent percentages. */
function deriveAgents(run) {
  const planned = run.plan.length;
  const sources = run.snippets || 0;
  const claims = run.findings.length;
  const verified = run.verifiedCount || 0;
  const passes = run.critiques.length;
  const done = !!run.done;
  const live = !done && !run.error;

  // How far along the pipeline the run has actually reached (0..7).
  let reached = 0;
  if (run.intent || run.route || planned) reached = 2;
  if (sources) reached = 3;
  if (claims) reached = 4;
  if (verified) reached = 5;
  if (passes) reached = 6;
  if (done) reached = 7;

  const desc = {
    orchestrator: run.modeLabel ? `${run.modeLabel} strategy` : "complexity scored",
    planner: planned ? `${planned} sub-question${planned === 1 ? "" : "s"}` : "decomposing the question",
    search: sources ? `${sources} source${sources === 1 ? "" : "s"}` : "retrieving and ranking sources",
    summarizer: claims ? `${claims} claim${claims === 1 ? "" : "s"} extracted` : "extracting claims",
    verifier: verified ? `${verified} verified` : "checking claims against sources",
    critic: passes ? `${passes} review pass${passes === 1 ? "" : "es"}` : "judging sufficiency",
    synthesizer: done ? "report delivered" : "writing the cited answer",
  };

  return PIPELINE.map((stage, i) => {
    const index = i + 1;
    const isDone = reached > index;
    const isActive = reached === index && live;
    return {
      icon: stage.icon,
      name: stage.name,
      status: isDone ? "done" : isActive ? "active" : "waiting",
      desc: desc[stage.key],
    };
  });
}

function deriveHealth(run) {
  const verified = run.findings.filter((f) => f.verified === true).length;
  const rows = [
    { icon: IconDoc, label: "Sources analyzed", value: String(run.snippets), tone: "muted" },
    { icon: IconCheckCircle, label: "Total claims", value: String(run.findings.length), tone: "muted" },
    { icon: IconCheckCircle, label: "Verified claims", value: String(verified), tone: "good" },
    { icon: IconAlert, label: "Critic passes", value: String(run.critiques.length), tone: run.critiques.length > 1 ? "warn" : "muted", hot: run.critiques.length > 1 },
    { icon: IconTarget, label: "Research confidence", value: typeof run.confidence === "number" ? `${Math.round(run.confidence * 100)}%` : "—", tone: "muted" },
  ];
  if (run.intent && run.intent.domain) {
    rows.unshift({
      icon: IconTarget,
      label: `Intent · ${run.intent.level || "practical"}`,
      value: `${run.intent.domain}${run.intent.ambiguity ? " · ambiguous" : ""}`,
      tone: run.intent.ambiguity ? "warn" : "good",
      hot: run.intent.ambiguity,
    });
  }
  if (run.route && run.route.path) {
    rows.push({
      icon: IconRoute,
      label: "Router",
      value: run.route.path === "direct" ? "direct answer" : "research",
      tone: run.route.path === "direct" ? "warn" : "muted",
      hot: run.route.path === "direct",
    });
  }
  if (typeof run.answerSupport === "number") {
    rows.push({
      icon: IconCheckCircle, label: "Answer support", tone: run.answerSupport >= 0.8 ? "good" : "warn",
      value: `${Math.round(run.answerSupport * 100)}%`, hot: run.answerSupport < 0.8,
    });
  }
  if (run.quality && typeof run.quality.overall === "number") {
    rows.push({
      icon: IconChart, label: "Answer quality",
      tone: run.quality.passed ? "good" : "warn",
      value: `${run.quality.overall}/100`, hot: !run.quality.passed,
    });
  }
  if (run.error) {
    rows.push({ icon: IconAlert, label: "Run status", value: "interrupted", tone: "bad", hot: true });
  }
  return rows;
}
