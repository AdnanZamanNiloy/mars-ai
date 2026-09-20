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
  cancelled: "idle",
};

export default function Sidebar({ view, onNavigate, missions, activeSessionId, onOpenMission, onNew, open, onClose, onRename, onTogglePin, onDelete }) {
  const ordered = [...missions].sort((a, b) => Number(!!b.pinned) - Number(!!a.pinned));
  const [menuSessionId, setMenuSessionId] = useState(null);
  const [editing, setEditing] = useState(null);

  const pinned = ordered.filter((m) => m.pinned);
  const recent = ordered.filter((m) => !m.pinned).slice(0, 6);

  const renderRow = (m) => {
    const label = m.title || m.query;
    const isEditing = editing === m.sessionId;
    return (
      <div key={m.sessionId} className={`mission-menu-wrap${menuSessionId === m.sessionId ? " open" : ""}`}>
        <button
          className={`mission-row${m.sessionId === activeSessionId ? " active" : ""}${m.pinned ? " pinned" : ""}`}
          onClick={() => { onOpenMission(m.sessionId); onClose?.(); }}
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
                    if (clean) onRename?.(m.sessionId, clean);
                    setEditing(null);
                  } else if (e.key === "Escape") {
                    setEditing(null);
                  }
                }}
                onBlur={(e) => {
                  const clean = e.currentTarget.value.trim();
                  if (clean && clean !== label) onRename?.(m.sessionId, clean);
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
          aria-expanded={menuSessionId === m.sessionId}
          onClick={(e) => {
            e.stopPropagation();
            setMenuSessionId((cur) => (cur === m.sessionId ? null : m.sessionId));
          }}
        >
          <IconMore size={15} />
        </button>
        {menuSessionId === m.sessionId ? (
          <div className="mission-menu" role="menu">
            <button
              type="button"
              className="mission-menu-item"
              role="menuitem"
              onClick={() => { onTogglePin?.(m.sessionId); setMenuSessionId(null); }}
            >
              <IconPin size={15} /> {m.pinned ? "Unpin" : "Pin"}
            </button>
            <button
              type="button"
              className="mission-menu-item"
              role="menuitem"
              onClick={() => { setEditing(m.sessionId); setMenuSessionId(null); }}
            >
              <IconPencil size={15} /> Rename
            </button>
            <button
              type="button"
              className="mission-menu-item danger"
              role="menuitem"
              onClick={() => { onDelete?.(m.sessionId); setMenuSessionId(null); }}
            >
              <IconTrash size={15} /> Delete
            </button>
          </div>
        ) : null}
      </div>
    );
  };

  useEffect(() => {
    if (menuSessionId == null) return;
    const onDown = (e) => {
      if (!e.target.closest(".mission-menu-wrap")) setMenuSessionId(null);
    };
    const onKey = (e) => {
      if (e.key === "Escape") setMenuSessionId(null);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [menuSessionId]);

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
