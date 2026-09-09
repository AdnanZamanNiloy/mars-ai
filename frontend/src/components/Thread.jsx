import { formatTime } from "../lib";
import { Planet } from "./Sidebar";
import { IconCheck, IconClock } from "./icons";

/* Live pipeline card driven entirely by streamed backend events. */

export function UserMessage({ text, time }) {
  return (
    <div className="msg anim-rise">
      <div className="msg-avatar">YOU</div>
      <div className="msg-body">
        <div className="msg-head"><b>You</b><time>{time}</time></div>
        <div className="msg-user-card"><p>{text}</p></div>
      </div>
    </div>
  );
}

export function MarsMessageShell({ time, children }) {
  return (
    <div className="msg anim-rise">
      <Planet size={40} />
      <div className="msg-body">
        <div className="msg-head"><b>MARS</b><time>{time}</time></div>
        {children}
      </div>
    </div>
  );
}

function StageRow({ state, label, meta }) {
  return (
    <div className={`stage ${state}`}>
      {state === "done" ? (
        <span className="trace-check" style={{ width: 16, height: 16 }}>
          <IconCheck size={10} />
        </span>
      ) : state === "active" ? (
        <span className="trace-ring" />
      ) : (
        <span className="trace-blink" />
      )}
      <span>{label}</span>
      {meta ? <span className="meta">{meta}</span> : null}
    </div>
  );
}

export function LiveRunCard({ run }) {
  const stages = [
    {
      key: "plan",
      label: "Planning research strategy",
      state: run.plan.length > 0 ? "done" : run.started ? "active" : "waiting",
      meta: run.plan.length > 0 ? `${run.plan.length} agents` : null,
    },
    {
      key: "search",
      label: "Gathering evidence",
      state: run.snippets > 0 ? (run.findings.length > 0 ? "done" : "active") : run.plan.length > 0 ? "active" : "waiting",
      meta: run.snippets > 0 ? `${run.snippets} sources` : null,
    },
    {
      key: "analyze",
      label: "Analyzing claims",
      state: run.findings.length > 0 ? (run.critiques.length > 0 ? "done" : "active") : run.snippets > 0 ? "active" : "waiting",
      meta: run.findings.length > 0 ? `${run.findings.length} claims` : null,
    },
    {
      key: "critic",
      label: "Critic review",
      state: run.critiques.length > 0 ? (run.done ? "done" : "active") : run.findings.length > 0 ? "active" : "waiting",
      meta: run.critiques.length > 0 ? `pass ${run.critiques.length}` : null,
    },
    {
      key: "report",
      label: "Synthesizing report",
      state: run.done ? "done" : run.critiques.length > 0 ? "active" : "waiting",
      meta: null,
    },
  ];

  return (
    <div className="live">
      <div className="live-head">
        <span className="dot live" />
        <h2>Research in progress</h2>
        <span className="spacer" />
        <span className="tag tone-muted">{run.modeLabel}</span>
      </div>
      <div className="stage-list">
        {stages.map((s) => <StageRow key={s.key} {...s} />)}
      </div>
      {run.plan.length > 0 ? (
        <ul className="plan-mini">
          {run.plan.map((q, i) => <li key={i}>{q}</li>)}
        </ul>
      ) : null}
    </div>
  );
}

export function TypingRow() {
  return (
    <span className="typing-pill">
      <i /><i /><i />
      Agents reviewing…
    </span>
  );
}

export function TraceEventLine({ at, children }) {
  return (
    <div className="trace-row">
      <span className="trace-time">{formatTime(at)}</span>
      <span className="trace-node"><span className="trace-blink" /></span>
      <span className="trace-text">{children}</span>
    </div>
  );
}

export function ErrorCard({ message, resumable, onResume, resuming }) {
  const noKey = /LLM key|GROQ_API_KEY|HUGGINGFACE/i.test(message || "");
  return (
    <div className="error-box anim-rise" role="alert">
      <div style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
        <IconClock size={16} style={{ flex: "none", marginTop: 2 }} />
        <div style={{ flex: 1 }}>
          <div><strong>{noKey ? "Model access not configured" : "Research interrupted"}</strong></div>
          <div style={{ marginTop: 5 }}>{message}</div>
          {resumable && !noKey ? (
            <div style={{ marginTop: 11 }}>
              <button className="btn" onClick={onResume} disabled={resuming}>
                {resuming ? "Resuming…" : "Resume from checkpoint"}
              </button>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
