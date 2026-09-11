import { timeAgo } from "../lib";
import { IconAgents, IconCompass, IconMissions, IconPlus } from "./icons";

export function Planet({ size = 40, ring = false }) {
  return (
    <span className="planet" style={{ width: size, height: size }} aria-hidden="true">
      {ring ? <span className="planet-ring" /> : null}
    </span>
  );
}

const NAV = [
  { id: "missions", label: "Missions", icon: IconMissions },
  { id: "providers", label: "Providers", icon: IconCompass },
  { id: "agents", label: "Agents", icon: IconAgents },
];

const DOT = {
  running: "live",
  resumable: "warn",
  completed: "done",
  failed: "bad",
  aborted: "idle",
};

export default function Sidebar({ view, onNavigate, missions, activeRunId, onOpenMission, onNew, open, onClose }) {
  const ordered = [...missions].sort((a, b) => Number(!!b.pinned) - Number(!!a.pinned));
  return (
    <aside className={`sidebar${open ? " open" : ""}`}>
      <div
        className="brand brand-home"
        onClick={() => { onNavigate("workspace"); onClose?.(); }}
        onKeyDown={(e) => { if (e.key === "Enter") { onNavigate("workspace"); onClose?.(); } }}
        role="button"
        tabIndex={0}
        title="Back to research console"
      >
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
          ordered.slice(0, 6).map((m) => {
            const label = m.title || m.query;
            return (
              <button
                key={m.runId}
                className={`mission-row${m.runId === activeRunId ? " active" : ""}${m.pinned ? " pinned" : ""}`}
                onClick={() => { onOpenMission(m.runId); onClose?.(); }}
                title={label}
              >
                <span className={`dot ${DOT[m.status] || "idle"}`} />
                <span className="body">
                  <span className="name">{label}</span>
                  <span className="sub">
                    <span>{statusLabel(m.status)}</span>
                    {m.pinned ? <span>Pinned</span> : null}
                    <span>{timeAgo(m.updatedAt)}</span>
                  </span>
                </span>
              </button>
            );
          })
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
