import { timeAgo } from "../lib";
import { IconCompass, IconMissions, IconLayers, IconPlus } from "./icons";

export function Planet({ size = 40, ring = false }) {
  return (
    <span className="planet" style={{ width: size, height: size }} aria-hidden="true">
      {ring ? <span className="planet-ring" /> : null}
    </span>
  );
}

const NAV = [
  { id: "workspace", label: "Command Center", icon: IconCompass },
  { id: "missions", label: "Missions", icon: IconMissions },
  { id: "evidence", label: "Evidence", icon: IconLayers },
];

const DOT = {
  running: "live",
  resumable: "warn",
  completed: "done",
  failed: "bad",
  aborted: "idle",
};

export default function Sidebar({ view, onNavigate, missions, activeRunId, onOpenMission, onNew, open, onClose }) {
  return (
    <aside className={`sidebar${open ? " open" : ""}`}>
      <div className="brand">
        <Planet size={44} />
        <div>
          <div className="brand-name">MARS</div>
          <div className="brand-sub">Multi-Agent Research System</div>
        </div>
      </div>

      <nav className="side-nav" aria-label="Primary">
        <button className="new-btn" onClick={() => { onNew(); onClose?.(); }}>
          <IconPlus size={16} /> New Research
        </button>
        {NAV.map((item) => (
          <button
            key={item.id}
            className={view === item.id ? "active" : ""}
            onClick={() => { onNavigate(item.id); onClose?.(); }}
          >
            <item.icon size={17} />
            {item.label}
            {item.id === "missions" && missions.length > 0 ? (
              <span className="count">{missions.length}</span>
            ) : null}
          </button>
        ))}
      </nav>

      <div className="side-section">
        <div className="side-section-head">
          <b>Recent Missions</b>
        </div>
        {missions.length === 0 ? (
          <p className="empty">No missions yet — your runs will appear here.</p>
        ) : (
          missions.slice(0, 6).map((m) => (
            <button
              key={m.runId}
              className={`mission-row${m.runId === activeRunId ? " active" : ""}`}
              onClick={() => { onOpenMission(m.runId); onClose?.(); }}
              title={m.query}
            >
              <span className={`dot ${DOT[m.status] || "idle"}`} />
              <span className="body">
                <span className="name">{m.query}</span>
                <span className="sub">
                  <span>{statusLabel(m.status)}</span>
                  <span>{timeAgo(m.updatedAt)}</span>
                </span>
              </span>
            </button>
          ))
        )}
      </div>

      <div className="side-foot">
        <div className="ver">MARS console · v2.0</div>
      </div>
    </aside>
  );
}

function statusLabel(status) {
  switch (status) {
    case "running": return "Research active";
    case "resumable": return "Interrupted — resumable";
    case "completed": return "Completed";
    case "failed": return "Failed";
    case "aborted": return "Aborted";
    default: return status || "Unknown";
  }
}
