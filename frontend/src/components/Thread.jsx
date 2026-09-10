import { useState } from "react";
import { formatTime } from "../lib";
import { IconCheck, IconClock, IconCopy, IconRefresh, IconSpark, IconSpeaker, IconThumbDown, IconThumbUp } from "./icons";

/* Chat-style thread: right-aligned user bubbles, plain MARS responses
 * with a working action row (copy, read aloud, feedback, regenerate). */

export function UserMessage({ text, time }) {
  return (
    <div className="msg-user anim-rise">
      <div className="bubble" title={time}>{text}</div>
    </div>
  );
}

const FEEDBACK_KEY = "mars.feedback.v1";

function readFeedback(id) {
  try {
    return (JSON.parse(localStorage.getItem(FEEDBACK_KEY) || {}))[id] || null;
  } catch {
    return null;
  }
}

function writeFeedback(id, value) {
  try {
    const all = JSON.parse(localStorage.getItem(FEEDBACK_KEY) || {});
    if (value) all[id] = value; else delete all[id];
    localStorage.setItem(FEEDBACK_KEY, JSON.stringify(all));
  } catch {
    /* ignore */
  }
}

export function MessageActions({ messageId, text, onRegenerate, canRegenerate }) {
  const [vote, setVote] = useState(() => readFeedback(messageId));
  const [copied, setCopied] = useState(false);
  const [speaking, setSpeaking] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const area = document.createElement("textarea");
      area.value = text;
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };

  const toggleSpeak = () => {
    if (!("speechSynthesis" in window)) return;
    if (speaking) {
      window.speechSynthesis.cancel();
      setSpeaking(false);
      return;
    }
    const utterance = new SpeechSynthesisUtterance(text.slice(0, 2000));
    utterance.onend = () => setSpeaking(false);
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
    setSpeaking(true);
  };

  const rate = (value) => {
    const next = vote === value ? null : value;
    setVote(next);
    writeFeedback(messageId, next);
  };

  return (
    <div className="msg-actions" role="toolbar" aria-label="Message actions">
      <button className="icon-btn" onClick={copy} title={copied ? "Copied" : "Copy"} aria-label="Copy">
        {copied ? <IconCheck size={15} /> : <IconCopy size={15} />}
      </button>
      <button className="icon-btn" onClick={toggleSpeak} title={speaking ? "Stop" : "Read aloud"} aria-label="Read aloud">
        <IconSpeaker size={15} />
      </button>
      <button className={`icon-btn${vote === "up" ? " is-active" : ""}`} onClick={() => rate("up")} title="Good response" aria-label="Good response" aria-pressed={vote === "up"}>
        <IconThumbUp size={15} />
      </button>
      <button className={`icon-btn${vote === "down" ? " is-active" : ""}`} onClick={() => rate("down")} title="Bad response" aria-label="Bad response" aria-pressed={vote === "down"}>
        <IconThumbDown size={15} />
      </button>
      {canRegenerate ? (
        <button className="icon-btn" onClick={onRegenerate} title="Regenerate" aria-label="Regenerate">
          <IconRefresh size={15} />
        </button>
      ) : null}
    </div>
  );
}

export function MarsMessageShell({ id, text, onRegenerate, canRegenerate, children }) {
  return (
    <div className="msg-mars anim-rise">
      <span className="sparkle" aria-hidden="true"><IconSpark size={22} /></span>
      <div className="msg-body">
        {children}
        {text ? (
          <MessageActions messageId={id} text={text} onRegenerate={onRegenerate} canRegenerate={canRegenerate} />
        ) : null}
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
