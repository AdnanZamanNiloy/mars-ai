import { MODE_META, confidenceLabel, timeAgo } from "../lib";
import { IconClock, IconX } from "./icons";

/* Missions = this browser's real run history (localStorage). Opening one
 * replays its persisted backend trace — never fabricated. */

const DOT = {
  running: "live",
  resumable: "warn",
  completed: "done",
  failed: "bad",
  aborted: "idle",
};

export function statusLabel(status) {
  switch (status) {
    case "running": return "Research active";
    case "resumable": return "Interrupted — resumable";
    case "completed": return "Completed";
    case "failed": return "Failed";
    case "aborted": return "Aborted";
    default: return status || "Unknown";
  }
}

export default function MissionsView({ missions, onOpen, onRemove, onNew }) {
  if (missions.length === 0) {
    return (
      <div className="view anim-rise">
        <div className="view-head">
          <h2>Missions</h2>
          <p>Every research run started from this browser</p>
        </div>
        <p className="empty">No missions yet. Start your first research run to see it here.</p>
        <div style={{ textAlign: "center", marginTop: 16 }}>
          <button className="btn-primary" onClick={onNew}>Start new research</button>
        </div>
      </div>
    );
  }
  return (
    <div className="view anim-rise">
      <div className="view-head">
        <h2>Missions</h2>
        <p>{missions.length} run{missions.length === 1 ? "" : "s"} · click any mission to replay its trace</p>
      </div>
      {missions.map((m) => {
        const parent = m.parentRunId ? missions.find((p) => p.runId === m.parentRunId) : null;
        return (
        <div key={m.runId} className="mission-card" onClick={() => onOpen(m.runId)} role="button" tabIndex={0}
          onKeyDown={(e) => { if (e.key === "Enter") onOpen(m.runId); }}>
          <span className={`dot ${DOT[m.status] || "idle"}`} style={{ marginTop: 6 }} />
          <span className="body">
            <p className="q">{m.query}</p>
            {parent ? (
              <span className="meta">
                <span>↳ challenge of “{(parent.query || "").slice(0, 70)}”</span>
              </span>
            ) : null}
            <span className="meta">
              <span className="tag tone-muted">{MODE_META[m.mode]?.label || m.mode}</span>
              <span className={`tag tone-${m.status === "completed" ? "good" : m.status === "running" ? "blue" : m.status === "resumable" ? "bad" : "muted"}`}>
                {statusLabel(m.status)}
              </span>
              {typeof m.confidence === "number" ? (
                <span>Confidence {Math.round(m.confidence * 100)}% ({confidenceLabel(m.confidence)})</span>
              ) : null}
              <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                <IconClock size={12} /> {timeAgo(m.updatedAt)}
              </span>
            </span>
          </span>
          <button
            className="icon-btn"
            title="Remove from history"
            aria-label="Remove from history"
            onClick={(e) => { e.stopPropagation(); onRemove(m.runId); }}
          >
            <IconX size={14} />
          </button>
        </div>
        );
      })}
    </div>
  );
}
