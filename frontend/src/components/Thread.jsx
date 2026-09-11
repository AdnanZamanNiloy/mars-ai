import { useState } from "react";
import { formatTime } from "../lib";
import { IconCheck, IconChevronDown, IconClock, IconCopy, IconRefresh } from "./icons";

/* Chat-style thread: right-aligned user bubbles, plain MARS responses
 * with a working action row (copy, read aloud, feedback, regenerate). */

export function UserMessage({ text, time }) {
  return (
    <div className="msg-user anim-rise">
      <div className="bubble" title={time}>{text}</div>
    </div>
  );
}

export function MessageActions({ messageId, text, onRegenerate, canRegenerate }) {
  const [copied, setCopied] = useState(false);

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

  return (
    <div className="msg-actions" role="toolbar" aria-label="Message actions">
      <button className="icon-btn" onClick={copy} title={copied ? "Copied" : "Copy"} aria-label="Copy">
        {copied ? <IconCheck size={15} /> : <IconCopy size={15} />}
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
      <div className="msg-body">
        {children}
        {text ? (
          <MessageActions messageId={id} text={text} onRegenerate={onRegenerate} canRegenerate={canRegenerate} />
        ) : null}
      </div>
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

export function ThinkingSteps({ steps }) {
  const [open, setOpen] = useState(true);
  if (!steps || steps.length === 0) return null;
  return (
    <div className="steps-card anim-rise">
      <button
        className="steps-head"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <IconChevronDown size={14} className={open ? "flip" : ""} />
        {open ? "Less steps" : `${steps.length} steps`}
      </button>
      {open ? (
        <div className="steps-list">
          {steps.map((t, i) => (
            <div className="steps-row" key={i}>
              <span className="steps-rail"><span className={`steps-dot kind-${t.kind || "wait"}`} /></span>
              <span className="steps-text">{t.text}</span>
              <span className="steps-time">{formatTime(t.at)}</span>
            </div>
          ))}
        </div>
      ) : null}
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
