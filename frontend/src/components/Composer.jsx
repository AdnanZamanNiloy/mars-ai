import { useEffect, useRef, useState } from "react";
import { MODE_META } from "../lib";
import { IconCheck, IconChevronDown, IconSend, IconStop } from "./icons";

const MODES = ["quick", "standard", "deep", "executive", "audit", "redteam"];

export default function Composer({
  value,
  onChange,
  onSubmit,
  running,
  onAbort,
  mode,
  onModeChange,
  placeholder,
}) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef(null);
  const areaRef = useRef(null);

  useEffect(() => {
    const close = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  useEffect(() => {
    const el = areaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
    }
  }, [value]);

  const submit = () => {
    const v = value.trim();
    if (v.length < 5 || running) return;
    onSubmit(v);
  };

  return (
    <div>
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
          placeholder={placeholder || "Ask the research…"}
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
              title="Research mode"
            >
              {MODE_META[mode]?.label || mode}
              <IconChevronDown size={13} />
            </button>
            {open && !running ? (
              <div className="mode-menu" role="menu">
                {MODES.map((m) => (
                  <button
                    key={m}
                    type="button"
                    role="menuitem"
                    className={m === mode ? "picked" : ""}
                    onClick={() => { onModeChange(m); setOpen(false); }}
                  >
                    <span>{MODE_META[m].label}<small>{MODE_META[m].hint}</small></span>
                    {m === mode ? <IconCheck size={13} /> : null}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
          <span className="toolbar-spacer" />
          {running ? (
            <button type="button" className="send-btn is-abort" onClick={onAbort} title="Abort run" aria-label="Abort run">
              <IconStop size={15} />
            </button>
          ) : (
            <button type="button" className="send-btn" onClick={submit} disabled={value.trim().length < 5} title="Send" aria-label="Send">
              <IconSend size={14} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
