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

export function loadMissions() {
  try {
    const raw = localStorage.getItem(MISSIONS_KEY);
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}

/* A mission is a record of a REAL run this browser started: its run_id lets
 * us replay the full trace from the backend at any time. */
export function upsertMission(mission) {
  const list = loadMissions().filter((m) => m.runId !== mission.runId);
  list.unshift({ ...mission, updatedAt: new Date().toISOString() });
  try {
    localStorage.setItem(MISSIONS_KEY, JSON.stringify(list.slice(0, MAX_MISSIONS)));
  } catch {
    /* storage full or unavailable — missions just won't persist */
  }
  return list.slice(0, MAX_MISSIONS);
}

export function removeMission(runId) {
  const list = loadMissions().filter((m) => m.runId !== runId);
  try {
    localStorage.setItem(MISSIONS_KEY, JSON.stringify(list));
  } catch {
    /* ignore */
  }
  return list;
}

export const MODE_META = {
  quick: { label: "Quick scan", hint: "2 agents · 1 pass · fastest" },
  standard: { label: "Standard", hint: "3 agents · up to 3 passes" },
  deep: { label: "Deep research", hint: "5 agents · up to 5 passes" },
};
