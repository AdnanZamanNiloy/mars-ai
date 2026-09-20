import { useCallback, useEffect, useMemo, useState } from "react";
import {
  clearActiveProvider, clearProviderChain, createProvider, createProviderChain,
  deleteProvider, deleteProviderChain, listProviderChains, listProviders,
  renameProviderChain, reorderProviderChain, setActiveProvider, setProviderChainEnabled,
  setProviderChainMembers, testProvider, updateProvider,
} from "../api";
import { chainRole, resolveChainMembers } from "../lib";
import ConfirmButton from "./ConfirmButton";

/* Providers — production control surface for the model layer.
 *
 * Four sections, one job each:
 *   1. Add model        — register an OpenAI-compatible endpoint.
 *   2. Available models — saved endpoints, with status + actions.
 *   3. Fallback chains  — ordered primary→fallback lists (drag to reorder).
 *   4. Serving mode     — exactly one of: single model, or one fallback chain.
 *
 * Serving selection is mutually exclusive server-side: choosing a single
 * model disables a chain and vice versa. Keys are encrypted at rest; the UI
 * only ever sees a last-4 hint. */

const EMPTY = { name: "", base_url: "", api_key: "", model: "" };

/* ---- 1. Add / edit model ------------------------------------------------ */
function AddModelSection({ editing, onSaved, onCancel }) {
  const [form, setForm] = useState(EMPTY);
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState("");

  useEffect(() => {
    setForm(editing
      ? { name: editing.name, base_url: editing.base_url, api_key: "", model: editing.model }
      : EMPTY);
    setFormError("");
  }, [editing]);

  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));

  const submit = async (e) => {
    e.preventDefault();
    setSaving(true);
    setFormError("");
    try {
      if (editing) {
        const payload = { name: form.name, base_url: form.base_url, model: form.model };
        if (form.api_key.trim()) payload.api_key = form.api_key;
        await updateProvider(editing.id, payload);
      } else {
        await createProvider(form);
      }
      onSaved();
    } catch (err) {
      setFormError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="pv-section">
      <div className="pv-section-head">
        <span className="pv-index">1</span>
        <div>
          <h3>Add model</h3>
          <p>Register an OpenAI-compatible endpoint. The key is encrypted at rest.</p>
        </div>
      </div>
      <form className="provider-form pv-form" onSubmit={submit}>
        <div className="pv-grid">
          <label>
            Provider name
            <input value={form.name} onChange={set("name")} placeholder="e.g. openai" maxLength={60} required />
          </label>
          <label>
            Model ID
            <input value={form.model} onChange={set("model")} placeholder="e.g. gpt-4o-mini" required />
          </label>
          <label className="pv-col-2">
            Base URL
            <input
              value={form.base_url} onChange={set("base_url")}
              placeholder="https://api.openai.com/v1" inputMode="url" required
            />
          </label>
          <label className="pv-col-2">
            API key
            <input
              value={form.api_key} onChange={set("api_key")} type="password"
              autoComplete="new-password"
              placeholder={editing ? "Leave blank to keep the stored key" : "Stored encrypted, never displayed"}
              required={!editing}
            />
          </label>
        </div>
        {formError ? <div className="error-box" role="alert">{formError}</div> : null}
        <div className="provider-form-actions">
          <button className="btn-primary" type="submit" disabled={saving}>
            {saving ? "Saving…" : editing ? "Save changes" : "Add model"}
          </button>
          {editing ? <button className="btn" type="button" onClick={onCancel}>Cancel</button> : null}
        </div>
      </form>
    </section>
  );
}

/* ---- 2. Available models ------------------------------------------------ */
function ModelsSection({ providers, activeId, probes, timeouts, busyId, onProbe, onTimeout, onEdit, onDelete }) {
  return (
    <section className="pv-section">
      <div className="pv-section-head">
        <span className="pv-index">2</span>
        <div>
          <h3>Available models</h3>
          <p>{providers.length} saved endpoint{providers.length === 1 ? "" : "s"}.</p>
        </div>
      </div>

      {providers.length === 0 ? (
        <p className="empty">No models yet — add an OpenAI-compatible endpoint above.</p>
      ) : null}

      <div className="pv-model-list">
        {providers.map((p) => {
          const probeState = probes[p.id];
          const isActive = p.id === activeId;
          const dotState = probeState?.state === "ok" ? "live" : probeState?.state === "bad" ? "bad" : "idle";
          return (
            <div key={p.id} className={`provider-card pv-model-card${isActive ? " active" : ""}`}>
              <span className={`dot ${dotState}`} aria-hidden="true" style={{ marginTop: 6 }} />
              <span className="body">
                <p className="q">
                  {p.name}
                  {isActive ? <span className="tag tone-good">Serving</span> : null}
                  {probeState && probeState.state !== "testing" ? (
                    <span className={`tag tone-${probeState.state === "ok" ? "good" : "bad"}`}>
                      {probeState.state === "ok" ? "reachable" : "unreachable"}
                    </span>
                  ) : null}
                </p>
                <p className="meta-line">{p.model} · {p.base_url}</p>
                <p className="meta-line sub">{p.has_key ? `key stored (${p.key_hint})` : "no key stored"}</p>
                {probeState && probeState.state !== "testing" ? (
                  <p className={`meta-line probe-${probeState.state}`} role="status">{probeState.msg}</p>
                ) : null}
                <span className="meta">
                  <button className="btn" onClick={() => onProbe(p.id)} disabled={busyId === p.id}>
                    {probeState?.state === "testing" ? "Testing…" : "Test"}
                  </button>
                  <label className="timeout-field" title="Probe timeout in seconds (5–120)">
                    <input
                      type="number" min={5} max={120} step={1}
                      value={timeouts[p.id] ?? 15}
                      onChange={(e) => onTimeout(p.id, e.target.value)}
                      aria-label={`Test timeout in seconds for ${p.name}`}
                    />
                    <span>s</span>
                  </label>
                  <button className="btn" onClick={() => onEdit(p)}>Edit</button>
                  <ConfirmButton
                    className="btn btn-danger"
                    label="Delete"
                    confirmLabel="Confirm delete"
                    title={`Delete provider ${p.name}`}
                    onConfirm={() => onDelete(p)}
                  />
                </span>
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

/* ---- 3. Fallback chains ------------------------------------------------- */
function ChainsSection({ providers, chains, loading, busy, onError, onReload }) {
  const [newName, setNewName] = useState("");
  const [editingId, setEditingId] = useState(null);
  const [editName, setEditName] = useState("");
  const [dragKey, setDragKey] = useState(null);

  const run = async (fn) => {
    try {
      await fn();
      await onReload();
    } catch (e) {
      if (onError) onError(e instanceof Error ? e.message : String(e));
    }
  };

  const add = (e) => {
    e.preventDefault();
    const name = newName.trim();
    if (!name) return;
    run(async () => { await createProviderChain(name); setNewName(""); });
  };

  const saveName = (chain) =>
    run(async () => { await renameProviderChain(chain.id, editName.trim()); setEditingId(null); });

  const setMembers = (chain, ids) => run(() => setProviderChainMembers(chain.id, ids));
  const addMember = (chain, providerId) =>
    setMembers(chain, [...chain.members.map((m) => m.provider_id), providerId]);
  const removeMember = (chain, providerId) =>
    setMembers(chain, chain.members.map((m) => m.provider_id).filter((id) => id !== providerId));

  // Drag-to-reorder: drop member (fromIndex) onto (toIndex), persist order.
  const dropAt = (chain, fromIndex, toIndex) => {
    if (fromIndex === toIndex || fromIndex < 0 || toIndex < 0) return;
    const ids = chain.members.map((m) => m.provider_id);
    const next = [...ids];
    const [item] = next.splice(fromIndex, 1);
    next.splice(toIndex, 0, item);
    run(() => reorderProviderChain(chain.id, next));
  };

  return (
    <section className="pv-section">
      <div className="pv-section-head">
        <span className="pv-index">3</span>
        <div>
          <h3>Fallback chains</h3>
          <p>
            Member 1 is tried first; on a provider/transient failure the runtime continues to
            member 2, then 3. Drag to reorder priority.
          </p>
        </div>
      </div>

      <form className="chain-add" onSubmit={add}>
        <input
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder="New chain name, e.g. production"
          maxLength={60}
          aria-label="New fallback chain name"
        />
        <button className="btn-primary" type="submit" disabled={busy || !newName.trim()}>
          Add chain
        </button>
      </form>

      {loading ? <p className="empty">Loading chains…</p> : null}
      {!loading && chains.length === 0 ? (
        <p className="empty">No chains yet — create one, then order your saved models.</p>
      ) : null}

      {chains.map((chain) => {
        const { members, available } = resolveChainMembers(
          providers, chain.members.map((m) => m.provider_id),
        );
        return (
          <div key={chain.id} className={`chain-card${chain.is_enabled ? " enabled" : ""}`}>
            <div className="chain-card-head">
              {editingId === chain.id ? (
                <span className="chain-rename">
                  <input
                    value={editName}
                    onChange={(e) => setEditName(e.target.value)}
                    maxLength={60}
                    aria-label={`Rename chain ${chain.name}`}
                    autoFocus
                  />
                  <button className="btn-primary" onClick={() => saveName(chain)} disabled={busy || !editName.trim()}>
                    Save
                  </button>
                  <button className="btn" onClick={() => setEditingId(null)}>Cancel</button>
                </span>
              ) : (
                <>
                  <span className="chain-title">
                    {chain.name}
                    {chain.is_enabled ? <span className="tag tone-good">Serving</span> : null}
                  </span>
                  <span className="chain-actions">
                    <button
                      className="btn" disabled={busy}
                      onClick={() => { setEditingId(chain.id); setEditName(chain.name); }}
                    >
                      Rename
                    </button>
                    <ConfirmButton
                      className="btn btn-danger"
                      label="Delete"
                      confirmLabel="Confirm delete"
                      disabled={busy}
                      onConfirm={() => run(() => deleteProviderChain(chain.id))}
                    />
                  </span>
                </>
              )}
            </div>

            <ol className="chain-members">
              {members.map((p, index) => {
                const role = chainRole(index);
                const key = `${chain.id}:${index}`;
                return (
                  <li
                    key={p.id}
                    className={`chain-member role-${role.tone}${dragKey === key ? " dragging" : ""}`}
                    draggable={!busy}
                    onDragStart={() => setDragKey(key)}
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={() => {
                      const from = dragKey && dragKey.startsWith(`${chain.id}:`)
                        ? Number(dragKey.split(":")[1]) : -1;
                      dropAt(chain, from, index);
                      setDragKey(null);
                    }}
                    onDragEnd={() => setDragKey(null)}
                  >
                    <span className="chain-grip" aria-hidden="true" title="Drag to reorder">⠿</span>
                    <span className="chain-order">{index + 1}</span>
                    <span className="chain-member-body">
                      <span className="chain-member-name">
                        {p.name}
                        <span className={`tag tone-${index === 0 ? "warn" : "muted"}`}>{role.label}</span>
                      </span>
                      <span className="chain-member-meta">{p.model} · {p.base_url}</span>
                    </span>
                    <span className="chain-member-actions">
                      <button
                        type="button" className="btn btn-sm" title="Move up"
                        onClick={() => dropAt(chain, index, index - 1)}
                        disabled={busy || index === 0}
                        aria-label={`Move ${p.name} earlier`}
                      >
                        ↑
                      </button>
                      <button
                        type="button" className="btn btn-sm" title="Move down"
                        onClick={() => dropAt(chain, index, index + 1)}
                        disabled={busy || index === members.length - 1}
                        aria-label={`Move ${p.name} later`}
                      >
                        ↓
                      </button>
                      <button
                        type="button" className="btn btn-sm"
                        onClick={() => removeMember(chain, p.id)}
                        disabled={busy}
                        aria-label={`Remove ${p.name}`}
                      >
                        Remove
                      </button>
                    </span>
                  </li>
                );
              })}
              {members.length === 0 ? (
                <li className="empty">No models in this chain yet — add one below.</li>
              ) : null}
            </ol>

            <div className="chain-add-member">
              <span className="eyebrow">
                Add {members.length === 0 ? "primary" : `fallback ${members.length}`}
              </span>
              {available.length === 0 ? (
                <p className="empty" style={{ marginTop: 8 }}>Every saved model is already in this chain.</p>
              ) : (
                <div className="chain-picker-list">
                  {available.map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      className="chain-picker-btn"
                      disabled={busy}
                      onClick={() => addMember(chain, p.id)}
                    >
                      <span className="chain-picker-name">{p.name}</span>
                      <span className="chain-picker-model">{p.model}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </section>
  );
}

/* ---- 4. Serving mode ---------------------------------------------------- */
function ServingModeSection({ providers, activeId, chains, busy, onSelectSingle, onSelectChain, onUseDefault }) {
  const enabledChain = chains.find((c) => c.is_enabled) || null;
  const mode = enabledChain ? "chain" : activeId ? "single" : "default";
  const active = providers.find((p) => p.id === activeId) || null;

  return (
    <section className="pv-section">
      <div className="pv-section-head">
        <span className="pv-index">4</span>
        <div>
          <h3>Serving mode</h3>
          <p>Exactly one mode runs research. Choosing one disables the other.</p>
        </div>
      </div>

      <div className="serving-modes">
        <div className={`serving-card${mode === "single" ? " active" : ""}`}>
          <div className="serving-card-head">
            <span className="serving-title">Single model</span>
            {mode === "single" ? (
              <button className="btn btn-sm" onClick={onUseDefault} disabled={busy}>
                Disable
              </button>
            ) : (
              <span className="tag tone-muted">Inactive</span>
            )}
          </div>
          <p className="serving-desc">
            {mode === "single"
              ? <>Active — every agent runs on <b>{active?.name}</b>. Pick another to switch.</>
              : <>Every agent runs on one model.{active ? ` Not active (chain is serving).` : ""}</>}
          </p>
          <div className="serving-picker">
            {providers.length === 0 ? (
              <p className="empty">Add a model first.</p>
            ) : (
              providers.map((p) => (
                <button
                  key={p.id}
                  type="button"
                  className={`serving-option${mode === "single" && p.id === activeId ? " picked" : ""}`}
                  onClick={() => onSelectSingle(p.id)}
                  disabled={busy}
                >
                  <span className="serving-option-name">{p.name}</span>
                  <span className="serving-option-meta">
                    {mode === "single" && p.id === activeId ? "Serving" : "Enable"}
                  </span>
                </button>
              ))
            )}
          </div>
        </div>

        <div className={`serving-card${mode === "chain" ? " active" : ""}`}>
          <div className="serving-card-head">
            <span className="serving-title">Fallback chain</span>
            {mode === "chain" ? (
              <button className="btn btn-sm" onClick={onUseDefault} disabled={busy}>
                Disable
              </button>
            ) : (
              <span className="tag tone-muted">Inactive</span>
            )}
          </div>
          <p className="serving-desc">
            {mode === "chain"
              ? <>Active — trying <b>{enabledChain?.name}</b> in order. Pick another to switch.</>
              : <>Try the primary, then fall back in order.{enabledChain ? "" : ""}</>}
          </p>
          <div className="serving-picker">
            {chains.length === 0 ? (
              <p className="empty">Create a chain first.</p>
            ) : (
              chains.map((c) => (
                <button
                  key={c.id}
                  type="button"
                  className={`serving-option${c.is_enabled ? " picked" : ""}`}
                  onClick={() => onSelectChain(c.id)}
                  disabled={busy || c.members.length === 0}
                  title={c.members.length === 0 ? "Add models to this chain first" : undefined}
                >
                  <span className="serving-option-name">{c.name}</span>
                  <span className="serving-option-meta">
                    {c.members.length} model{c.members.length === 1 ? "" : "s"}
                    {c.is_enabled ? " · Serving" : " · Enable"}
                  </span>
                </button>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="serving-default-row">
        <span className="serving-hint">
          {mode === "default"
            ? "Default env chain in use (Groq / HuggingFace / CUSTOM_LLM_*)."
            : "Selecting a mode automatically disables the other."}
        </span>
      </div>
    </section>
  );
}

/* ---- Page --------------------------------------------------------------- */
export default function ProvidersView() {
  const [providers, setProviders] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [chains, setChains] = useState([]);
  const [chainsLoading, setChainsLoading] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [editing, setEditing] = useState(null);
  const [probes, setProbes] = useState({});
  const [timeouts, setTimeouts] = useState({});
  const [busy, setBusy] = useState(false);
  const [busyId, setBusyId] = useState(null);

  const refreshProviders = useCallback(async () => {
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

  const refreshChains = useCallback(async () => {
    try {
      const data = await listProviderChains();
      setChains(Array.isArray(data?.chains) ? data.chains : []);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setChainsLoading(false);
    }
  }, []);

  const refreshAll = useCallback(async () => {
    await Promise.all([refreshProviders(), refreshChains()]);
  }, [refreshProviders, refreshChains]);

  useEffect(() => { refreshAll(); }, [refreshAll]);

  const onSaved = async () => { setEditing(null); await refreshAll(); };

  const remove = async (p) => {
    try {
      await deleteProvider(p.id);
      await refreshAll();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const probeTimeout = (id) => {
    const raw = parseFloat(timeouts[id]);
    return Number.isFinite(raw) ? raw : 15;
  };

  const probe = async (id) => {
    setBusyId(id);
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
    } finally {
      setBusyId(null);
    }
  };

  // Serving-mode selection is mutually exclusive server-side; refresh both
  // lists so the two cards always reflect the single active mode.
  const selectSingle = async (id) => {
    setBusy(true);
    try { await setActiveProvider(id); await refreshAll(); }
    catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setBusy(false); }
  };

  const selectChain = async (id) => {
    setBusy(true);
    try { await setProviderChainEnabled(id, true); await refreshAll(); }
    catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setBusy(false); }
  };

  const useDefault = async () => {
    setBusy(true);
    try {
      await Promise.all([clearActiveProvider(), clearProviderChain()]);
      await refreshAll();
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setBusy(false); }
  };

  const enabledChain = useMemo(() => chains.find((c) => c.is_enabled) || null, [chains]);
  const activeProvider = providers.find((p) => p.id === activeId) || null;
  const serving = enabledChain
    ? `chain "${enabledChain.name}"`
    : activeProvider ? activeProvider.name : "the default chain";

  return (
    <div className="view anim-rise">
      <div className="view-head">
        <h2>Providers</h2>
        <p>
          {providers.length} model{providers.length === 1 ? "" : "s"} ·{" "}
          {chains.length} chain{chains.length === 1 ? "" : "s"} · serving:{" "}
          <strong style={{ color: "var(--t2)" }}>{serving}</strong>
        </p>
      </div>

      {error ? <div className="error-box" role="alert">{error}</div> : null}

      <AddModelSection editing={editing} onSaved={onSaved} onCancel={() => setEditing(null)} />

      {loading ? <p className="empty">Loading models…</p> : (
        <ModelsSection
          providers={providers}
          activeId={activeId}
          probes={probes}
          timeouts={timeouts}
          busyId={busyId}
          onProbe={probe}
          onTimeout={(id, v) => setTimeouts((prev) => ({ ...prev, [id]: v }))}
          onEdit={setEditing}
          onDelete={remove}
        />
      )}

      <ChainsSection
        providers={providers}
        chains={chains}
        loading={chainsLoading}
        busy={busy}
        onError={setError}
        onReload={refreshChains}
      />

      <ServingModeSection
        providers={providers}
        activeId={activeId}
        chains={chains}
        busy={busy}
        onSelectSingle={selectSingle}
        onSelectChain={selectChain}
        onUseDefault={useDefault}
      />
    </div>
  );
}
