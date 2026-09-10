import { timeAgo } from "../lib";
import { IconX } from "./icons";

/* Saved Knowledge — claims the user bookmarked from the inspector or
 * evidence explorer. Local to this browser, like missions. */

export default function KnowledgeView({ items, onRemove, onInspect }) {
  if (items.length === 0) {
    return (
      <div className="view anim-rise">
        <div className="view-head">
          <h2>Knowledge</h2>
          <p>Claims you saved for later</p>
        </div>
        <p className="empty">Nothing saved yet — open any claim in the inspector or evidence list to save it here.</p>
      </div>
    );
  }
  return (
    <div className="view anim-rise">
      <div className="view-head">
        <h2>Knowledge</h2>
        <p>{items.length} saved claim{items.length === 1 ? "" : "s"}</p>
      </div>
      {items.map((k, i) => (
        <div key={`${k.savedAt}-${i}`} className="mission-card" onClick={() => onInspect(k)} role="button" tabIndex={0}
          onKeyDown={(e) => { if (e.key === "Enter") onInspect(k); }}>
          <span className="dot done" style={{ marginTop: 6 }} />
          <span className="body">
            <p className="q" style={{ fontWeight: 500 }}>{k.claim}</p>
            <span className="meta">
              {k.source ? <span>{k.source}</span> : <span>unsourced</span>}
              {typeof k.verified === "boolean" ? (
                <span className={`tag tone-${k.verified ? "good" : "bad"}`}>{k.verified ? "verified" : "unverified"}</span>
              ) : null}
              <span>saved {timeAgo(k.savedAt)}</span>
            </span>
          </span>
          <button className="icon-btn" title="Remove" aria-label="Remove"
            onClick={(e) => { e.stopPropagation(); onRemove(k); }}>
            <IconX size={14} />
          </button>
        </div>
      ))}
    </div>
  );
}
