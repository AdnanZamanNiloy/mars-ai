/* Global theme state — shared by the console and the docs iframe.
 *
 * The single source of truth is `data-theme` on <html> plus the
 * `mars-docs-theme` localStorage key (the key docs.app.js already uses).
 * Docs read the same key on boot, so persistence is automatic; the
 * postMessage hop exists only to live-sync an already-open docs iframe. */

export const THEME_KEY = "mars-docs-theme";

/* Resolve once, in this order: an explicit stored choice, then the OS
   preference, then dark. This must match the inline bootstrap in index.html
   exactly, or first paint and React state can disagree. */
export function getStoredTheme() {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === "light" || stored === "dark") return stored;
    if (window.matchMedia("(prefers-color-scheme: light)").matches) return "light";
    return "dark";
  } catch (e) {
    return "dark";
  }
}

export function applyTheme(mode) {
  document.documentElement.setAttribute("data-theme", mode);
  try { localStorage.setItem(THEME_KEY, mode); } catch (e) { /* ignore */ }
}

export function initTheme() {
  const mode = getStoredTheme();
  document.documentElement.setAttribute("data-theme", mode);
  return mode;
}
