import { MODE_META, confidenceLabel, timeAgo } from "../lib";
import { STATUS_DOT } from "./Sidebar";
import { IconClock, IconX } from "./icons";

/* Research = this browser's real run history (localStorage). Opening one
 * replays its persisted backend trace — never fabricated. */

export function statusLabel(status) {
  switch (status) {
    case "running": return "Researching";
    case "resumable": return "Interrupted · resumable";
    case "completed": return "Completed";
    case "failed": return "Failed";
    case "aborted": return "Aborted";
    case "cancelled": return "Stopped";
    default: return status || "Unknown";
  }
}

export function statusTone(status) {
  switch (status) {
    case "completed": return "good";
    case "running": return "blue";
    case "resumable": return "warn";
    case "failed": return "bad";
    default: return "muted";
  }
}

export { STATUS_DOT };

export default function MissionsView({ missions, onOpen, onRemove, onNew }) {
  if (missions.length === 0) {
    return (
      <div className="view anim-rise">
        <div className="view-head">
          <h2>Research</h2>
          <p>Every research chat started from this browser</p>
        </div>
        <p className="empty">
          No research yet. Ask a question in the console and it will be listed here with
          its mode, status, confidence and trace.
        </p>
        <div style={{ marginTop: 14 }}>
          <button className="btn-primary" onClick={onNew}>Start research</button>
        </div>
      </div>
    );
  }

  const completed = missions.filter((m) => m.status === "completed").length;

  return (
    <div className="view anim-rise">
      <div className="view-head">
        <h2>Research</h2>
        <p>
          {missions.length} chat{missions.length === 1 ? "" : "s"} · {completed} completed ·
          select any chat to replay its persisted trace
        </p>
      </div>
      {missions.map((m) => {
        const parent = m.parentRunId ? missions.find((p) => p.sessionId === m.parentRunId) : null;
        return (
          <div
            key={m.sessionId}
            className="mission-card"
            onClick={() => onOpen(m.sessionId)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onOpen(m.sessionId); }
            }}
            role="button"
            tabIndex={0}
            aria-label={`Open ${m.title || m.query}`}
          >
            <span className={`dot ${STATUS_DOT[m.status] || "idle"}`} style={{ marginTop: 6 }} />
            <span className="body">
              <p className="q">{m.title || m.query}</p>
              {parent ? (
                <span className="meta">
                  <span>↳ follow-up on “{(parent.query || "").slice(0, 70)}”</span>
                </span>
              ) : null}
              <span className="meta">
                <span className="tag tone-muted">{MODE_META[m.mode]?.label || m.mode}</span>
                <span className={`tag tone-${statusTone(m.status)}`}>{statusLabel(m.status)}</span>
                {typeof m.confidence === "number" ? (
                  <span>Confidence {Math.round(m.confidence * 100)}% ({confidenceLabel(m.confidence)})</span>
                ) : null}
                <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                  <IconClock size={12} /> {timeAgo(m.updatedAt)}
                </span>
              </span>
            </span>
            <button
              type="button"
              className="icon-btn"
              title="Remove from history"
              aria-label={`Remove ${m.title || m.query} from history`}
              onClick={(e) => { e.stopPropagation(); onRemove(m.sessionId); }}
            >
              <IconX size={14} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
