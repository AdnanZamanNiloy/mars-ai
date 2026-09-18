import { useEffect, useState } from "react";
import { IconAgents, IconCompass, IconMissions, IconMore, IconPencil, IconPin, IconPlus, IconTrash } from "./icons";

export function Planet({ size = 40, ring = false }) {
  return (
    <span className="planet" style={{ width: size, height: size }} aria-hidden="true">
      {ring ? <span className="planet-ring" /> : null}
    </span>
  );
}

const NAV = [
  { id: "missions", label: "Research", icon: IconMissions },
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

export default function Sidebar({ view, onNavigate, missions, activeRunId, onOpenMission, onNew, open, onClose, onRename, onTogglePin, onDelete }) {
  const ordered = [...missions].sort((a, b) => Number(!!b.pinned) - Number(!!a.pinned));
  const [menuRunId, setMenuRunId] = useState(null);
  const [editing, setEditing] = useState(null);

  const pinned = ordered.filter((m) => m.pinned);
  const recent = ordered.filter((m) => !m.pinned).slice(0, 6);

  const renderRow = (m) => {
    const label = m.title || m.query;
    const isEditing = editing === m.runId;
    return (
      <div key={m.runId} className={`mission-menu-wrap${menuRunId === m.runId ? " open" : ""}`}>
        <button
          className={`mission-row${m.runId === activeRunId ? " active" : ""}${m.pinned ? " pinned" : ""}`}
          onClick={() => { onOpenMission(m.runId); onClose?.(); }}
          title={label}
        >
          <span className={`dot ${DOT[m.status] || "idle"}`} />
          <span className="body">
            {isEditing ? (
              <input
                className="mission-rename-input"
                defaultValue={label}
                autoFocus
                onClick={(e) => e.stopPropagation()}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    const clean = e.currentTarget.value.trim();
                    if (clean) onRename?.(m.runId, clean);
                    setEditing(null);
                  } else if (e.key === "Escape") {
                    setEditing(null);
                  }
                }}
                onBlur={(e) => {
                  const clean = e.currentTarget.value.trim();
                  if (clean && clean !== label) onRename?.(m.runId, clean);
                  setEditing(null);
                }}
              />
            ) : (
              <span className="name">{label}</span>
            )}
          </span>
        </button>
        <button
          type="button"
          className="mission-menu-btn"
          aria-label={`Actions for ${label}`}
          aria-haspopup="menu"
          aria-expanded={menuRunId === m.runId}
          onClick={(e) => {
            e.stopPropagation();
            setMenuRunId((cur) => (cur === m.runId ? null : m.runId));
          }}
        >
          <IconMore size={15} />
        </button>
        {menuRunId === m.runId ? (
          <div className="mission-menu" role="menu">
            <button
              type="button"
              className="mission-menu-item"
              role="menuitem"
              onClick={() => { onTogglePin?.(m.runId); setMenuRunId(null); }}
            >
              <IconPin size={15} /> {m.pinned ? "Unpin" : "Pin"}
            </button>
            <button
              type="button"
              className="mission-menu-item"
              role="menuitem"
              onClick={() => { setEditing(m.runId); setMenuRunId(null); }}
            >
              <IconPencil size={15} /> Rename
            </button>
            <button
              type="button"
              className="mission-menu-item danger"
              role="menuitem"
              onClick={() => { onDelete?.(m.runId); setMenuRunId(null); }}
            >
              <IconTrash size={15} /> Delete
            </button>
          </div>
        ) : null}
      </div>
    );
  };

  useEffect(() => {
    if (menuRunId == null) return;
    const onDown = (e) => {
      if (!e.target.closest(".mission-menu-wrap")) setMenuRunId(null);
    };
    const onKey = (e) => {
      if (e.key === "Escape") setMenuRunId(null);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [menuRunId]);

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
        {pinned.length > 0 ? (
          <>
            <div className="side-section-head">
              <b>Pinned</b>
            </div>
            {pinned.map(renderRow)}
          </>
        ) : null}

        <div className={`side-section-head${pinned.length > 0 ? " has-above" : ""}`}>
          <b>Recent Research</b>
        </div>
        {missions.length === 0 ? (
          <p className="empty">No research yet — your runs will appear here.</p>
        ) : recent.length === 0 ? (
          <p className="empty">No recent research — everything is pinned.</p>
        ) : (
          recent.map(renderRow)
        )}
      </div>

      <div className="side-foot">
        <div className="ver">MARS console · v2.0</div>
      </div>
    </aside>
  );
}
