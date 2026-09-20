import { useEffect, useRef, useState } from "react";

/* Destructive-action guard. Replaces window.confirm, which is unstyleable and
 * untunable to the theme, with an inline two-step confirm that matches the
 * rest of the console. Clicking away or pressing Escape cancels. */

export default function ConfirmButton({
  onConfirm, label, confirmLabel = "Confirm", className = "btn", disabled = false, title,
}) {
  const [armed, setArmed] = useState(false);
  const wrapRef = useRef(null);

  useEffect(() => {
    if (!armed) return;
    const onDown = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setArmed(false);
    };
    const onKey = (e) => { if (e.key === "Escape") setArmed(false); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [armed]);

  return (
    <span className="confirm-btn" ref={wrapRef}>
      <button
        type="button"
        className={className}
        disabled={disabled}
        title={title}
        aria-expanded={armed}
        onClick={() => {
          if (!armed) { setArmed(true); return; }
          setArmed(false);
          onConfirm();
        }}
        onBlur={() => setArmed(false)}
      >
        {armed ? confirmLabel : label}
      </button>
    </span>
  );
}
