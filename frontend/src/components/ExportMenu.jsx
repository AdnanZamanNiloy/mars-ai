import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { downloadReport } from "../api";

/* Report export menu: Markdown, Word, PDF.
 *
 * The file is produced by the backend from the canonical stored report data
 * (never scraped from this rendered card), so an export always matches the
 * persisted report exactly. The menu is always available; if the run has no
 * id yet, selecting a format surfaces a clear message instead of a dead
 * control, and a run without a completed report surfaces the backend's
 * message rather than saving a blank file.
 *
 * Layering: the menu is rendered in a PORTAL to document.body with
 * position: fixed, anchored to the button's rect. Report content lives inside
 * scrolling/animated ancestors (.thread overflow, .msg-mars anim-rise) that
 * create stacking contexts and clip descendants — an absolutely-positioned
 * dropdown inside them is trapped no matter how high its z-index. A fixed
 * portal escapes every ancestor's overflow and stacking context. */
const FORMATS = [
  { id: "md", label: "Markdown (.md)" },
  { id: "docx", label: "Word (.docx)" },
  { id: "pdf", label: "PDF (.pdf)" },
];

const MENU_WIDTH = 190;

export default function ExportMenu({ runId }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const wrapRef = useRef(null);
  const menuRef = useRef(null);

  const placeMenu = useCallback(() => {
    const btn = wrapRef.current;
    if (!btn) return;
    const r = btn.getBoundingClientRect();
    // Right-align to the button; keep inside the viewport with an 8px margin.
    const left = Math.max(8, Math.min(r.right - MENU_WIDTH, window.innerWidth - MENU_WIDTH - 8));
    setPos({ top: r.bottom + 4, left });
  }, []);

  useEffect(() => {
    if (!open) return;
    placeMenu();
    const onDown = (e) => {
      const inButton = wrapRef.current && wrapRef.current.contains(e.target);
      const inMenu = menuRef.current && menuRef.current.contains(e.target);
      if (!inButton && !inMenu) setOpen(false);
    };
    const onKey = (e) => {
      if (e.key === "Escape") setOpen(false);
    };
    const onReflow = () => placeMenu();
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    window.addEventListener("resize", onReflow);
    window.addEventListener("scroll", onReflow, true);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", onReflow);
      window.removeEventListener("scroll", onReflow, true);
    };
  }, [open, placeMenu]);

  const download = async (format) => {
    setError("");
    if (!runId) {
      setError("This run is not saved yet — export becomes available once it has a run id.");
      return;
    }
    setBusy(format);
    try {
      await downloadReport(runId, format);
      setOpen(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  };

  const menu = open
    ? createPortal(
        <div
          ref={menuRef}
          className="title-menu export-menu"
          role="menu"
          style={{ position: "fixed", top: pos.top, left: pos.left, right: "auto" }}
        >
          {FORMATS.map((f) => (
            <button
              key={f.id}
              className="title-menu-item"
              onClick={() => download(f.id)}
              disabled={busy === f.id}
              role="menuitem"
            >
              {busy === f.id ? "Preparing…" : f.label}
            </button>
          ))}
          {error ? <div className="export-error" role="alert">{error}</div> : null}
        </div>,
        document.body,
      )
    : null;

  return (
    <div className="export-menu-wrap" ref={wrapRef}>
      <button
        type="button"
        className="btn export-btn"
        onClick={() => setOpen((o) => !o)}
        aria-label="Export report"
        aria-expanded={open}
        aria-haspopup="menu"
        title="Export this report as Markdown, Word or PDF"
      >
        Export
        <span className={`nav-chev${open ? " open" : ""}`} aria-hidden="true">▾</span>
      </button>
      {menu}
    </div>
  );
}
