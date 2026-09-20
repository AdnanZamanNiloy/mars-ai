import { useEffect, useState } from "react";
import { lastActiveLabel } from "../lib";
import {
  IconAgents, IconCompass, IconMissions, IconMore, IconPencil, IconPin,
  IconPlus, IconTrash,
} from "./icons";
import ThemeToggle from "./ThemeToggle";

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

export const STATUS_DOT = {
  running: "live",
  resumable: "warn",
  completed: "done",
  failed: "bad",
  aborted: "idle",
  cancelled: "idle",
};

const RECENT_LIMIT = 6;

export default function Sidebar({
  view, onNavigate, missions, activeSessionId, onOpenMission, onNew,
  open, onClose, onRename, onTogglePin, onDelete,
}) {
  const ordered = [...missions].sort((a, b) => Number(!!b.pinned) - Number(!!a.pinned));
  const [menuSessionId, setMenuSessionId] = useState(null);
  const [editing, setEditing] = useState(null);
  const [showAll, setShowAll] = useState(false);

  const pinned = ordered.filter((m) => m.pinned);
  const unpinned = ordered.filter((m) => !m.pinned);
  // History is no longer silently capped at 6 with no way out.
  const recent = showAll ? unpinned : unpinned.slice(0, RECENT_LIMIT);
  const hidden = Math.max(0, unpinned.length - RECENT_LIMIT);

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

  const go = (id) => { onNavigate(id); onClose?.(); };

  const renderRow = (m) => {
    const label = m.title || m.query;
    const isEditing = editing === m.sessionId;
    const isActive = m.sessionId === activeSessionId;
    return (
      <div key={m.sessionId} className={`mission-menu-wrap${menuSessionId === m.sessionId ? " open" : ""}`}>
        <button
          type="button"
          className={`mission-row${isActive ? " active" : ""}`}
          aria-current={isActive ? "true" : undefined}
          onClick={() => { onOpenMission(m.sessionId); onClose?.(); }}
          title={label}
        >
          <span className={`dot ${STATUS_DOT[m.status] || "idle"}`} />
          <span className="body">
            {isEditing ? (
              <input
                className="mission-rename-input"
                defaultValue={label}
                autoFocus
                aria-label={`Rename ${label}`}
                onClick={(e) => e.stopPropagation()}
                onKeyDown={(e) => {
                  e.stopPropagation();
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
            <span className="sub">
              <span>{m.mode || "standard"}</span>
              {m.updatedAt || m.lastActive ? (
                <span>{lastActiveLabel(m.updatedAt || m.lastActive)}</span>
              ) : null}
            </span>
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
          <div className="mission-menu" role="menu" aria-label={`Actions for ${label}`}>
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

  return (
    <aside className={`sidebar${open ? " open" : ""}`} aria-label="Chats and navigation">
      {/* A real <button>: Space now works, and the branding is keyboard
          reachable without a role/tabIndex workaround. */}
      <button type="button" className="brand" onClick={() => go("workspace")} title="Back to the research console">
        <Planet size={38} />
        <span className="brand-copy">
          <span className="brand-name">MARS</span>
          <span className="brand-sub">Multi-Agent Research System</span>
        </span>
      </button>

      <nav className="side-nav" aria-label="Primary">
        <button type="button" className="new-btn" onClick={() => { onNew(); onClose?.(); }}>
          <IconPlus size={15} /> New research
        </button>
        {NAV.map((item) => (
          <button
            type="button"
            key={item.id}
            className={view === item.id ? "active" : ""}
            aria-current={view === item.id ? "page" : undefined}
            onClick={() => go(item.id)}
          >
            <item.icon size={16} />
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
            <div className="side-section-head"><b>Pinned</b></div>
            {pinned.map(renderRow)}
          </>
        ) : null}

        <div className={`side-section-head${pinned.length > 0 ? " has-above" : ""}`}>
          <b>Recent</b>
          {hidden > 0 ? (
            <button type="button" onClick={() => setShowAll((s) => !s)}>
              {showAll ? "Show less" : `+${hidden} more`}
            </button>
          ) : null}
        </div>
        {missions.length === 0 ? (
          <p className="empty">No research yet — start a question and the chat appears here.</p>
        ) : unpinned.length === 0 ? (
          <p className="empty">Everything is pinned.</p>
        ) : (
          recent.map(renderRow)
        )}
      </div>

      <div className="side-foot">
        <span className="ver">MARS console · v2.1</span>
        <ThemeToggle />
      </div>
    </aside>
  );
}
