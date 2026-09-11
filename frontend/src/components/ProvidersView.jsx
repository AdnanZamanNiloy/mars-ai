import { useCallback, useEffect, useState } from "react";
import {
  clearActiveProvider,
  createProvider,
  deleteProvider,
  listProviders,
  setActiveProvider,
  testProvider,
  updateProvider,
} from "../api";

/* Providers tab: user-managed OpenAI-compatible LLM endpoints. Exactly one
 * may be active — the active model is the ONLY one research uses (no
 * fallback chain while one is selected). Keys are encrypted server-side;
 * the UI only ever sees a last-4 hint. */
const EMPTY = { name: "", base_url: "", api_key: "", model: "" };

export default function ProvidersView() {
  const [providers, setProviders] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [form, setForm] = useState(EMPTY);
  const [editingId, setEditingId] = useState(null);
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState("");
  const [probes, setProbes] = useState({});
  const [timeouts, setTimeouts] = useState({});

  const refresh = useCallback(async () => {
    try {
      const data = await listProviders();
      setProviders(data.providers || []);
      setActiveId(data.active_id ?? null);
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh ]);

  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));

  const startEdit = (p) => {
    setEditingId(p.id);
    setForm({ name: p.name, base_url: p.base_url, api_key: "", model: p.model });
    setFormError("");
  };

  const cancelEdit = () => {
    setEditingId(null);
    setForm(EMPTY);
    setFormError("");
  };

  const submit = async (e) => {
    e.preventDefault();
    setSaving(true);
    setFormError("");
    try {
      if (editingId) {
        const payload = { name: form.name, base_url: form.base_url, model: form.model };
        if (form.api_key.trim()) payload.api_key = form.api_key;
        await updateProvider(editingId, payload);
      } else {
        await createProvider(form);
      }
      cancelEdit();
      await refresh();
    } catch (err) {
      setFormError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  const remove = async (p) => {
    if (!window.confirm(`Delete provider "${p.name}"? Saved runs keep working; future research falls back to the default chain.`)) return;
    try {
      await deleteProvider(p.id);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const activate = async (id) => {
    try {
      await setActiveProvider(id);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const deactivate = async () => {
    try {
      await clearActiveProvider();
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const probeTimeout = (id) => {
    const raw = parseFloat(timeouts[id]);
    return Number.isFinite(raw) ? raw : 15;
  };

  const probe = async (id) => {
    setProbes((prev) => ({ ...prev, [id]: { state: "testing", msg: "Testing connection…" } }));
    try {
      const result = await testProvider(id, probeTimeout(id));
      setProbes((prev) => ({
        ...prev,
        [id]: result.ok
          ? { state: "ok", msg: `Connected in ${result.latency_ms} ms` }
          : { state: "bad", msg: result.error || "Connection failed" },
      }));
    } catch (err) {
      setProbes((prev) => ({
        ...prev, [id]: { state: "bad", msg: err instanceof Error ? err.message : String(err) },
      }));
    }
  };

  const active = providers.find((p) => p.id === activeId) || null;

  return (
    <div className="view anim-rise">
      <div className="view-head">
        <h2>Providers</h2>
        <p>{providers.length} saved · {active ? `active: ${active.name}` : "no active model — default chain in use"}</p>
      </div>

      {loading ? <p className="empty">Loading providers…</p> : null}
      {error ? <div className="error-box">{error}</div> : null}

      {active ? (
        <div className="provider-active">
          <span className="dot live" />
          <span>
            <b>{active.name}</b> · {active.model} — all research, agents and synthesis run on this model only.
          </span>
          <button className="btn" onClick={deactivate}>Use default chain</button>
        </div>
      ) : null}

      <form className="provider-form" onSubmit={submit}>
        <h3>{editingId ? "Edit provider" : "Add provider"}</h3>
        <label>
          Provider name
          <input value={form.name} onChange={set("name")} placeholder="e.g. openai" maxLength={60} required />
        </label>
        <label>
          Base URL
          <input
            value={form.base_url} onChange={set("base_url")}
            placeholder="https://api.openai.com/v1" inputMode="url" required
          />
        </label>
        <label>
          API key
          <input
            value={form.api_key} onChange={set("api_key")} type="password"
            autoComplete="new-password"
            placeholder={editingId ? "Leave blank to keep the stored key" : "Stored encrypted, never displayed"}
            required={!editingId}
          />
        </label>
        <label>
          Model ID
          <input value={form.model} onChange={set("model")} placeholder="e.g. gpt-4o-mini" required />
        </label>
        {formError ? <div className="error-box">{formError}</div> : null}
        <div className="provider-form-actions">
          <button className="btn-primary" type="submit" disabled={saving}>
            {saving ? "Saving…" : editingId ? "Save changes" : "Add provider"}
          </button>
          {editingId ? (
            <button className="btn" type="button" onClick={cancelEdit}>Cancel</button>
          ) : null}
        </div>
      </form>

      {providers.map((p) => {
        const probeState = probes[p.id];
        const isActive = p.id === activeId;
        return (
          <div key={p.id} className={`provider-card${isActive ? " active" : ""}`}>
            <span className={`dot ${isActive ? "live" : "idle"}`} style={{ marginTop: 6 }} />
            <span className="body">
              <p className="q">
                {p.name}
                {isActive ? <span className="tag tone-blue">Active model</span> : null}
              </p>
              <p className="meta-line">{p.model} · {p.base_url}</p>
              <p className="meta-line sub">{p.has_key ? `key stored (${p.key_hint})` : "no key stored"}</p>
              {probeState ? (
                <p className={`meta-line probe-${probeState.state}`}>{probeState.msg}</p>
              ) : null}
              <span className="meta">
                {!isActive ? (
                  <button className="btn" onClick={() => activate(p.id)}>Use this model</button>
                ) : null}
                <button className="btn" onClick={() => probe(p.id)}>
                  {probeState?.state === "testing" ? "Testing…" : "Test connection"}
                </button>
                <label className="timeout-field" title="Probe timeout in seconds (5–120)">
                  <input
                    type="number" min={5} max={120} step={1}
                    value={timeouts[p.id] ?? 15}
                    onChange={(e) => setTimeouts((prev) => ({ ...prev, [p.id]: e.target.value }))}
                    aria-label={`Test timeout in seconds for ${p.name}`}
                  />
                  <span>s</span>
                </label>
                <button className="btn" onClick={() => startEdit(p)}>Edit</button>
                <button className="btn btn-danger" onClick={() => remove(p)}>Delete</button>
              </span>
            </span>
          </div>
        );
      })}
      {!loading && providers.length === 0 ? (
        <p className="empty">No providers yet — add your first OpenAI-compatible endpoint above.</p>
      ) : null}
    </div>
  );
}
