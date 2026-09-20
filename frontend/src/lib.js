/* Pure helpers: report parsing, formatting, session mission log.
 * No backend calls here (see api.js). No fake research data anywhere. */

export function extractSection(markdown, heading, nextHeadings) {
  if (!markdown) return "";
  const start = markdown.indexOf(heading);
  if (start === -1) return "";
  const bodyStart = start + heading.length;
  let end = -1;
  for (const h of nextHeadings) {
    const i = markdown.indexOf(h, bodyStart);
    if (i !== -1 && (end === -1 || i < end)) end = i;
  }
  return (end === -1 ? markdown.slice(bodyStart) : markdown.slice(bodyStart, end)).trim();
}

const REPORT_HEADINGS = ["# Final Answer", "# Supporting Evidence", "# Contradictions", "# Decision Layer"];

export function parseReport(markdown) {
  return {
    finalAnswer: extractSection(markdown, "# Final Answer", REPORT_HEADINGS.filter((h) => h !== "# Final Answer")),
    evidence: extractSection(markdown, "# Supporting Evidence", REPORT_HEADINGS.filter((h) => h !== "# Supporting Evidence")),
    contradictions: extractSection(markdown, "# Contradictions", REPORT_HEADINGS.filter((h) => h !== "# Contradictions")),
    decisionLayer: extractSection(markdown, "# Decision Layer", REPORT_HEADINGS.filter((h) => h !== "# Decision Layer")),
  };
}

export function extractDomain(url) {
  if (!url) return "";
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

export function trustOf(domain) {
  const host = (domain || "").toLowerCase();
  if (!host) return { label: "unknown", tone: "muted" };
  if (host === "wikipedia.org" || host.endsWith(".wikipedia.org") || host === "arxiv.org" || host.endsWith(".arxiv.org")) {
    return { label: "high trust", tone: "good" };
  }
  if (["reddit.com", "medium.com", "quora.com", "facebook.com", "tiktok.com"].some((d) => host === d || host.endsWith(`.${d}`))) {
    return { label: "low trust", tone: "bad" };
  }
  return { label: "medium trust", tone: "muted" };
}

export function formatTime(isoOrDate) {
  try {
    const d = isoOrDate instanceof Date ? isoOrDate : new Date(isoOrDate);
    return d.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
  } catch {
    return "";
  }
}

export function timeAgo(iso) {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const secs = Math.max(0, Math.floor((Date.now() - then) / 1000));
  if (secs < 60) return "just now";
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins} min ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  const days = Math.floor(hours / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

export function confidenceLabel(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "unknown";
  if (value >= 0.8) return "high";
  if (value >= 0.5) return "moderate";
  return "low";
}

const MISSIONS_KEY = "mars.missions.v1";
const MAX_MISSIONS = 30;

/* Interrupt-and-edit: drop the edited user turn and everything after it, so
 * the resend can append a fresh turn in its place. Returns the original list
 * unchanged when the id isn't found (nothing to edit). Pure and testable. */
export function truncateFromMessage(messages, messageId) {
  if (!Array.isArray(messages)) return [];
  const idx = messages.findIndex((m) => m && m.id === messageId);
  if (idx === -1) return messages;
  return messages.slice(0, idx);
}

/* Auto-scroll intent: should the stream keep the newest content in view?
 * True while the viewport is within `threshold` px of the bottom. Once the
 * user scrolls further up to read older turns, this is false and streaming
 * must not yank them back down. Pure so it can be unit-tested. */
export function shouldAutoScroll({ scrollHeight, scrollTop, clientHeight }, threshold = 160) {
  const distance = scrollHeight - scrollTop - clientHeight;
  return distance < threshold;
}

/* The active chat's stable session id. One chat = one id, reused for every
 * follow-up question; only "New Chat" mints a fresh one. Kept in its own key
 * so an interrupted write to the session list can't lose the active chat. */
const ACTIVE_SESSION_KEY = "mars.activeSession.v1";

export function newSessionId() {
  try {
    if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID();
  } catch {
    /* fall through to timestamp id */
  }
  return `s${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export function loadActiveSessionId() {
  try {
    return localStorage.getItem(ACTIVE_SESSION_KEY) || "";
  } catch {
    return "";
  }
}

export function saveActiveSessionId(sessionId) {
  try {
    if (sessionId) localStorage.setItem(ACTIVE_SESSION_KEY, sessionId);
    else localStorage.removeItem(ACTIVE_SESSION_KEY);
  } catch {
    /* storage unavailable — the chat just won't survive a reload */
  }
  return sessionId || "";
}

export function loadMissions() {
  try {
    const raw = localStorage.getItem(MISSIONS_KEY);
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}

/* A mission is a record of one REAL chat session this browser started: its
 * sessionId groups every question asked in the same chat, and runId points at
 * the most recent run so the backend can still supply that run's trace. */
export function upsertMission(mission) {
  const list = loadMissions();
  const existing = list.find((m) => m.sessionId === mission.sessionId);
  // The chat is named by its FIRST message. Later follow-ups must not rename
  // it, so the stored title (and the query it falls back to) is preserved;
  // only an explicitly-supplied title, or a brand-new session, sets it.
  const merged = {
    ...existing,
    ...mission,
    title: mission.title || existing?.title || "",
    query: existing?.query || mission.query,
  };
  const rest = list.filter((m) => m.sessionId !== mission.sessionId);
  rest.unshift({ ...merged, updatedAt: new Date().toISOString() });
  try {
    localStorage.setItem(MISSIONS_KEY, JSON.stringify(rest.slice(0, MAX_MISSIONS)));
  } catch {
    /* storage full or unavailable — missions just won't persist */
  }
  return rest.slice(0, MAX_MISSIONS);
}

export function removeMission(sessionId) {
  const list = loadMissions().filter((m) => m.sessionId !== sessionId);
  try {
    localStorage.setItem(MISSIONS_KEY, JSON.stringify(list));
  } catch {
    /* ignore */
  }
  return list;
}

/* Patch fields (title, pinned) on one mission, preserving order. */
export function updateMission(sessionId, patch) {
  const list = loadMissions().map((m) =>
    m.sessionId === sessionId ? { ...m, ...patch, updatedAt: new Date().toISOString() } : m);
  try {
    localStorage.setItem(MISSIONS_KEY, JSON.stringify(list.slice(0, MAX_MISSIONS)));
  } catch {
    /* ignore */
  }
  return list.slice(0, MAX_MISSIONS);
}

const KNOWLEDGE_KEY = "mars.knowledge.v1";
const MAX_SAVED = 100;

export function loadKnowledge() {
  try {
    const raw = localStorage.getItem(KNOWLEDGE_KEY);
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}

export function saveKnowledgeItem(item) {
  const key = `${item.claim || ""}||${item.source || ""}`;
  const list = loadKnowledge().filter((k) => `${k.claim || ""}||${k.source || ""}` !== key);
  list.unshift({ claim: item.claim || "", source: item.source || "",
                 verified: item.verified ?? null, savedAt: new Date().toISOString() });
  try {
    localStorage.setItem(KNOWLEDGE_KEY, JSON.stringify(list.slice(0, MAX_SAVED)));
  } catch {
    /* ignore */
  }
  return list.slice(0, MAX_SAVED);
}

export function removeKnowledgeItem(item) {
  const key = `${item.claim || ""}||${item.source || ""}`;
  const list = loadKnowledge().filter((k) => `${k.claim || ""}||${k.source || ""}` !== key);
  try {
    localStorage.setItem(KNOWLEDGE_KEY, JSON.stringify(list));
  } catch {
    /* ignore */
  }
  return list;
}

export const MODE_META = {
  quick: { label: "Quick scan", hint: "2 agents · 1 pass · fastest" },
  standard: { label: "Standard", hint: "3 agents · up to 3 passes" },
  deep: { label: "Deep research", hint: "5 agents · up to 5 passes" },
  executive: { label: "Executive brief", hint: "5 agents · up to 4 passes · decision-focused" },
  audit: { label: "Evidence audit", hint: "3 agents · up to 4 passes · verification-heavy" },
  redteam: { label: "Red team", hint: "3 agents · up to 2 passes · challenges conclusions" },
};
