import { useEffect, useRef, useState } from "react";
import { MODE_META } from "../lib";
import { IconCheck, IconChevronDown, IconSend, IconStop } from "./icons";

const MODES = ["quick", "standard", "deep", "executive", "audit", "redteam"];

export default function Composer({
  value, onChange, onSubmit, running, onAbort, mode, onModeChange, placeholder,
}) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef(null);
  const areaRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) setOpen(false);
    };
    const onKey = (e) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  useEffect(() => {
    const el = areaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 150)}px`;
    }
  }, [value]);

  const canSend = value.trim().length >= 5 && !running;

  const submit = () => {
    const v = value.trim();
    if (v.length < 5 || running) return;
    onSubmit(v);
  };

  return (
    <div className="composer">
      <textarea
        ref={areaRef}
        rows={1}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
        placeholder={placeholder || "Ask a research question…"}
        disabled={running}
        aria-label="Research question"
      />
      <div className="composer-toolbar">
        <div className="mode-wrap" ref={menuRef}>
          <button
            type="button"
            className="pill-btn"
            onClick={() => setOpen((o) => !o)}
            disabled={running}
            aria-expanded={open}
            aria-haspopup="menu"
            title="Research mode — changes agent count and iteration budget"
          >
            {MODE_META[mode]?.label || mode}
            <IconChevronDown size={13} />
          </button>
          {open && !running ? (
            <div className="mode-menu" role="menu" aria-label="Research mode">
              {MODES.map((m) => (
                <button
                  key={m}
                  type="button"
                  role="menuitemradio"
                  aria-checked={m === mode}
                  className={m === mode ? "picked" : ""}
                  onClick={() => { onModeChange(m); setOpen(false); }}
                >
                  <span>
                    {MODE_META[m].label}
                    <small>{MODE_META[m].hint}</small>
                  </span>
                  {m === mode ? <IconCheck size={13} /> : null}
                </button>
              ))}
            </div>
          ) : null}
        </div>
        <span className="toolbar-spacer" />
        {running ? (
          <button type="button" className="send-btn is-abort" onClick={onAbort} title="Stop this run" aria-label="Stop this run">
            <IconStop size={14} />
          </button>
        ) : (
          <button
            type="button"
            className="send-btn"
            onClick={submit}
            disabled={!canSend}
            title={canSend ? "Send (Enter)" : "Type at least 5 characters"}
            aria-label="Send research question"
          >
            <IconSend size={13} />
          </button>
        )}
      </div>
    </div>
  );
}
