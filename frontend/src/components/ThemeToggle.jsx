/* Global theme toggle. Flips `data-theme` on <html> and persists to the
 * shared `mars-docs-theme` key, so the choice survives reloads and is picked
 * up by the docs iframe (which reads the same key on boot).
 *
 * Mounted in BOTH the landing nav and the console sidebar footer — previously
 * it appeared only on the landing page, so there was no way to switch theme
 * once inside the console. */

import { useState } from "react";
import { applyTheme, getStoredTheme } from "../theme";

export default function ThemeToggle({ className = "" }) {
  const [mode, setMode] = useState(() => getStoredTheme());
  const next = mode === "light" ? "dark" : "light";

  const toggle = () => {
    setMode(next);
    applyTheme(next);
  };

  return (
    <button
      type="button"
      className={`theme-toggle${className ? ` ${className}` : ""}`}
      onClick={toggle}
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
    >
      {mode === "light" ? (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="4.5" />
          <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" />
        </svg>
      )}
    </button>
  );
}
