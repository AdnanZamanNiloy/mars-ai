import { useState } from "react";
import { formatTime } from "../lib";
import { IconCheck, IconChevronDown, IconCopy, IconInfo, IconPencil, IconRefresh } from "./icons";

/* Chat thread: right-aligned user bubbles, MARS responses, working action row. */

export function UserMessage({
  text, time, onEdit, editing, onCancelEdit, onSubmitEdit, disabled,
  onRegenerate, canRegenerate, messageId,
}) {
  const [draft, setDraft] = useState(text || "");
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text || "");
    } catch {
      const area = document.createElement("textarea");
      area.value = text || "";
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };

  if (editing) {
    const autoGrow = (el) => {
      if (!el) return;
      el.style.height = "48px";
      el.style.height = `${Math.max(48, Math.min(el.scrollHeight, 180))}px`;
    };
    return (
      <div className="msg-user anim-rise" data-message-id={messageId}>
        <div className="bubble bubble-edit">
          <label className="eyebrow" htmlFor={`edit-${messageId}`}>Editing question</label>
          <textarea
            id={`edit-${messageId}`}
            className="msg-edit-input"
            autoFocus
            rows={1}
            value={draft}
            disabled={disabled}
            ref={autoGrow}
            onChange={(e) => { setDraft(e.target.value); autoGrow(e.target); }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                onSubmitEdit?.(draft);
              }
              if (e.key === "Escape") onCancelEdit?.();
            }}
          />
          <div className="msg-edit-actions">
            <span className="msg-edit-info" title="Resending replaces this turn and everything after it">
              <IconInfo size={12} />
            </span>
            <span style={{ flex: 1 }} />
            <button className="msg-edit-cancel" onClick={onCancelEdit} disabled={disabled}>Cancel</button>
            <button
              className="msg-edit-save"
              onClick={() => onSubmitEdit?.(draft)}
              disabled={disabled || (draft || "").trim().length < 5}
            >
              Resend
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="msg-user anim-rise" data-message-id={messageId}>
      <div className="bubble" title={time}>{text}</div>
      <div className="user-actions" role="toolbar" aria-label="Message actions">
        <span className="user-time">{formatTime(time)}</span>
        {onRegenerate ? (
          <button
            className="icon-btn user-action-btn"
            onClick={() => onRegenerate()}
            disabled={disabled || !canRegenerate}
            title="Run this question again"
            aria-label="Run this question again"
          >
            <IconRefresh size={13} />
          </button>
        ) : null}
        {onEdit ? (
          <button
            className="icon-btn user-action-btn"
            onClick={() => onEdit()}
            disabled={disabled}
            title="Edit and resend"
            aria-label="Edit this question"
          >
            <IconPencil size={13} />
          </button>
        ) : null}
        <button
          className="icon-btn user-action-btn"
          onClick={copy}
          title={copied ? "Copied" : "Copy"}
          aria-label="Copy question"
        >
          {copied ? <IconCheck size={13} /> : <IconCopy size={13} />}
        </button>
      </div>
    </div>
  );
}

export function MessageActions({ text, onRegenerate, canRegenerate }) {
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
    <div className="msg-actions" role="toolbar" aria-label="Report actions">
      <button className="icon-btn" onClick={copy} title={copied ? "Copied" : "Copy report"} aria-label="Copy report">
        {copied ? <IconCheck size={15} /> : <IconCopy size={15} />}
      </button>
      {canRegenerate ? (
        <button className="icon-btn" onClick={onRegenerate} title="Regenerate report" aria-label="Regenerate report">
          <IconRefresh size={15} />
        </button>
      ) : null}
    </div>
  );
}

export function MarsMessageShell({ text, onRegenerate, canRegenerate, children }) {
  return (
    <div className="msg-mars anim-rise">
      <div className="msg-body">
        {children}
        {text ? (
          <MessageActions text={text} onRegenerate={onRegenerate} canRegenerate={canRegenerate} />
        ) : null}
      </div>
    </div>
  );
}

export function TypingRow() {
  return (
    <span className="typing-pill" role="status">
      <i /><i /><i />
      Agents reviewing…
    </span>
  );
}

export function ThinkingSteps({ steps }) {
  const [open, setOpen] = useState(false);
  if (!steps || steps.length === 0) return null;
  // Collapsed by default: the report is the product, the trace is support.
  return (
    <div className="steps-card">
      <button
        className="steps-head"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <IconChevronDown size={13} className={open ? "flip" : ""} />
        {open ? "Hide pipeline trace" : `Pipeline trace · ${steps.length} step${steps.length === 1 ? "" : "s"}`}
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
      <div style={{ display: "flex", gap: 9, alignItems: "flex-start" }}>
        <IconInfo size={15} style={{ flex: "none", marginTop: 2 }} />
        <div style={{ flex: 1 }}>
          <strong>{noKey ? "Model access not configured" : "Research interrupted"}</strong>
          <div style={{ marginTop: 5, color: "var(--t2)" }}>{message}</div>
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
