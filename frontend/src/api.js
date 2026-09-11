/* Backend client — the single place that talks to the FastAPI backend.
 * Event protocol mirrors app/api/routes.py exactly:
 *   progress | plan | search_progress | critic | findings | budget |
 *   decisions | final_report | error
 */

export async function streamNDJSON(url, { method = "POST", body, signal, onEvent, onHttpError }) {
  const response = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
    signal,
  });

  if (!response.ok || !response.body) {
    let detail = `Request failed (${response.status})`;
    try {
      const data = await response.json();
      if (data && data.detail) detail = String(data.detail);
    } catch {
      /* keep generic message */
    }
    const err = new Error(detail);
    err.status = response.status;
    if (onHttpError) onHttpError(err);
    throw err;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  const pump = (chunk, done) => {
    buffer += chunk;
    const lines = buffer.split("\n");
    buffer = done ? "" : lines.pop() || "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        onEvent(JSON.parse(trimmed));
      } catch {
        /* ignore malformed chunks to keep the stream resilient */
      }
    }
    if (done && buffer.trim()) {
      try {
        onEvent(JSON.parse(buffer.trim()));
      } catch {
        /* ignore */
      }
      buffer = "";
    }
  };

  for (;;) {
    const { value, done } = await reader.read();
    pump(decoder.decode(value || new Uint8Array(), { stream: !done }), done);
    if (done) break;
  }
}

export function startResearch({ query, mode, signal, onEvent }) {
  return streamNDJSON("/api/research/stream", {
    method: "POST",
    body: { query, mode },
    signal,
    onEvent,
  });
}

export function resumeResearch({ runId, signal, onEvent }) {
  return streamNDJSON(`/api/research/${encodeURIComponent(runId)}/resume`, {
    method: "POST",
    signal,
    onEvent,
  });
}

export async function fetchTrace(runId, signal) {
  const response = await fetch(`/api/research/${encodeURIComponent(runId)}/trace`, { signal });
  if (!response.ok) {
    let detail = `Trace request failed (${response.status})`;
    try {
      const data = await response.json();
      if (data && data.detail) detail = String(data.detail);
    } catch {
      /* keep generic */
    }
    throw new Error(detail);
  }
  return response.json();
}

/* ---------- LLM providers ---------- */

async function apiJSON(url, { method = "GET", body } = {}) {
  const response = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try {
    data = await response.json();
  } catch {
    /* non-JSON error body */
  }
  if (!response.ok) {
    throw new Error((data && data.detail) || `Request failed (${response.status})`);
  }
  return data;
}

export function listProviders() {
  return apiJSON("/api/providers");
}

export function createProvider(payload) {
  return apiJSON("/api/providers", { method: "POST", body: payload });
}

export function updateProvider(id, payload) {
  return apiJSON(`/api/providers/${encodeURIComponent(id)}`, { method: "PUT", body: payload });
}

export function deleteProvider(id) {
  return apiJSON(`/api/providers/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export function setActiveProvider(id) {
  return apiJSON(`/api/providers/${encodeURIComponent(id)}/active`, { method: "POST" });
}

export function clearActiveProvider() {
  return apiJSON("/api/providers/active/clear", { method: "POST" });
}

export function testProvider(id, timeoutSec = null) {
  const body = timeoutSec === null || timeoutSec === undefined ? undefined : { timeout_sec: timeoutSec };
  return apiJSON(`/api/providers/${encodeURIComponent(id)}/test`, { method: "POST", body });
}
