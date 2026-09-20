import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  clearActiveProvider, clearProviderChain, createProvider, createProviderChain,
  deleteProvider, deleteProviderChain, listProviderChains, listProviders,
  renameProviderChain, reorderProviderChain, setActiveProvider, setProviderChainEnabled,
  setProviderChainMembers, testProvider, updateProvider,
} from "../api";
import { chainRole, resolveChainMembers } from "../lib";
import ConfirmButton from "./ConfirmButton";
import {
  IconCheck, IconChevronDown, IconMore, IconPencil,
  IconPlus, IconRefresh, IconX,
} from "./icons";
import "./providers.css";

/* Providers — the model-layer control surface.
 *
 * Four sections, one job each, top to bottom:
 *   1. Add model         register an OpenAI-compatible endpoint.
 *   2. Available models  saved endpoints, live status, per-model actions.
 *   3. Fallback chains   named, ordered primary → fallback lists.
 *   4. Serving mode      exactly one of: one model, or one chain.
 *
 * Field mapping (matches the persisted schema):
 *   Provider name → `name`        human label for the endpoint
 *   Model ID      → `model`       the id sent to the provider API
 *   Model name    → `model_name`  display label; defaults to the model id
 *
 * Serving selection is mutually exclusive server-side: picking a single model
 * disables any chain and vice versa. Keys are encrypted at rest; the UI only
 * ever shows a last-4 hint. */

const EMPTY = { name: "", base_url: "", api_key: "", model: "", model_name: "" };

/* Tab order, exported for the test that locks this contract. The four
 * sections used to stack into one long scroll; each is now its own sub-page
 * reached from this bar, so the whole view fits without scrolling. */
export const PROVIDER_TABS = [
  { id: "models", label: "Available models", title: "Available models" },
  { id: "add", label: "Add model", title: "Add model" },
  { id: "chains", label: "Fallback chains", title: "Fallback chains" },
  { id: "serving", label: "Serving mode", title: "Serving mode" },
];

export const DEFAULT_PROVIDER_TAB = "models";

export function isProviderTab(value) {
  return PROVIDER_TABS.some((t) => t.id === value);
}

const HASH_PREFIX = "#/providers";

/* The active tab rides in the url's QUERY STRING, deliberately not as a hash
 * sub-segment: App.viewFromHash() resolves the view from the first path
 * segment, so "#/providers/serving" would land on a view that does not exist
 * and a reload would bounce to the landing page. "#/providers?tab=serving"
 * keeps the view parseable while making each sub-page linkable and shareable.
 *
 * This is a hash-routed SPA, so the query lives INSIDE the fragment
 * ("#/providers?tab=serving") and location.search stays empty — the fragment
 * is what a user copies out of the address bar. location.search is still read
 * as a fallback so a plain "?tab=" url works too. */
export function providerTabFromUrl(hash, search) {
  const hashRaw = String(hash ?? "").replace(/^#\/?/, "");
  const view = hashRaw.split(/[?/]/)[0];
  if (view !== "providers") return DEFAULT_PROVIDER_TAB;
  const query = hashRaw.includes("?") ? hashRaw.slice(hashRaw.indexOf("?") + 1) : String(search ?? "");
  let tab = null;
  try {
    tab = new URLSearchParams(query).get("tab");
  } catch {
    /* malformed query — fall through to the default */
  }
  return isProviderTab(tab) ? tab : DEFAULT_PROVIDER_TAB;
}

const PROVIDER_PRESETS = [
  { label: "OpenAI", base_url: "https://api.openai.com/v1" },
  { label: "Groq", base_url: "https://api.groq.com/openai/v1" },
  { label: "OpenRouter", base_url: "https://openrouter.ai/api/v1" },
  { label: "Together", base_url: "https://api.together.xyz/v1" },
];

/* ---- section shell ------------------------------------------------------
 * One tab renders at a time, so the old numbered badge would have restarted
 * at 1 on every tab. The panel is a real tabpanel: it owns the accessible
 * name (aria-labelledby → the tab button) and is focusable so keyboard users
 * land in the content they just selected. */
function Section({ id, title, desc, aside, children }) {
  return (
    <section className="pv-panel" role="tabpanel" id={`pv-panel-${id}`} aria-labelledby={`pv-tab-${id}`} tabIndex={0}>
      <header className="pv-panel-head">
        <div className="pv-head-text">
          <h3>{title}</h3>
          {desc ? <p>{desc}</p> : null}
        </div>
        {aside ? <div className="pv-head-aside">{aside}</div> : null}
      </header>
      <div className="pv-section-body">{children}</div>
    </section>
  );
}

/* ---- 1. Add / edit model ------------------------------------------------ */
function AddModelSection({ editing, onSaved, onCancel }) {
  const [form, setForm] = useState(EMPTY);
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState("");
  const [showKey, setShowKey] = useState(false);

  useEffect(() => {
    setForm(editing
      ? {
        name: editing.name, base_url: editing.base_url, api_key: "",
        model: editing.model, model_name: editing.model_name || "",
      }
      : EMPTY);
    setFormError("");
    setShowKey(false);
  }, [editing]);

  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));

  const submit = async (e) => {
    e.preventDefault();
    setSaving(true);
    setFormError("");
    try {
      const label = form.model_name.trim();
      if (editing) {
        // Blank label explicitly clears it back to the model ID; omitting the
        // key entirely would instead preserve the previously stored label.
        const payload = {
          name: form.name, base_url: form.base_url,
          model: form.model, model_name: label,
        };
        if (form.api_key.trim()) payload.api_key = form.api_key;
        await updateProvider(editing.id, payload);
      } else {
        await createProvider({
          name: form.name, base_url: form.base_url,
          api_key: form.api_key, model: form.model, model_name: label,
        });
      }
      onSaved();
    } catch (err) {
      setFormError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Section
      id="add"
      title={editing ? "Edit model" : "Add model"}
      desc="Register an OpenAI-compatible endpoint. The API key is encrypted at rest and never displayed again."
      aside={editing
        ? <span className="pv-chip tone-warn">Editing “{editing.name}”</span>
        : <span className="pv-chip tone-muted">New endpoint</span>}
    >
      <form className="pv-form" onSubmit={submit}>
        <div className="pv-field-grid">
          <div className="pv-field">
            <label htmlFor="pv-name">Provider name</label>
            <input id="pv-name" value={form.name} onChange={set("name")} placeholder="e.g. openai" maxLength={60} required />
          </div>
          <div className="pv-field">
            <label htmlFor="pv-model">Model ID</label>
            <input id="pv-model" value={form.model} onChange={set("model")} placeholder="e.g. gpt-4o-mini" required />
            <span className="pv-hint">The id sent to the provider API.</span>
          </div>
          <div className="pv-field">
            <label htmlFor="pv-model-name">Model name</label>
            <input
              id="pv-model-name" value={form.model_name} onChange={set("model_name")}
              placeholder="e.g. GPT-4o mini" maxLength={200}
            />
            <span className="pv-hint">Display label. Defaults to the model ID.</span>
          </div>
          <div className="pv-field pv-span-2">
            <label htmlFor="pv-url">Base URL</label>
            <input
              id="pv-url" value={form.base_url} onChange={set("base_url")}
              placeholder="https://api.openai.com/v1" inputMode="url" required
            />
            <div className="pv-presets" role="group" aria-label="Base URL presets">
              {PROVIDER_PRESETS.map((p) => (
                <button
                  key={p.label} type="button" className="pv-preset"
                  onClick={() => setForm((f) => ({ ...f, base_url: p.base_url }))}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>
          <div className="pv-field pv-span-2">
            <label htmlFor="pv-key">API key</label>
            <div className="pv-key-row">
              <input
                id="pv-key" value={form.api_key} onChange={set("api_key")}
                type={showKey ? "text" : "password"} autoComplete="new-password"
                placeholder={editing ? "Leave blank to keep the stored key" : "Stored encrypted, never displayed"}
                required={!editing}
              />
              <button
                type="button" className="pv-key-toggle"
                onClick={() => setShowKey((s) => !s)}
                aria-label={showKey ? "Hide API key" : "Show API key"}
                aria-pressed={showKey}
              >
                {showKey ? <IconX size={14} /> : <IconCheck size={14} />}
              </button>
            </div>
          </div>
        </div>
        {formError ? <div className="error-box" role="alert">{formError}</div> : null}
        <div className="pv-form-actions">
          <button className="btn-primary" type="submit" disabled={saving}>
            <IconPlus size={14} /> {saving ? "Saving…" : editing ? "Save changes" : "Add model"}
          </button>
          {editing ? (
            <button className="btn" type="button" onClick={onCancel}>
              <IconX size={14} /> Cancel
            </button>
          ) : null}
        </div>
      </form>
    </Section>
  );
}

/* ---- 2. Available models ------------------------------------------------ */
function StatusPill({ probe, isActive, inChain }) {
  if (isActive) {
    return <span className="pv-status tone-good"><span className="pv-status-dot" />Serving</span>;
  }
  if (probe?.state === "testing") {
    return <span className="pv-status tone-muted"><span className="pv-status-dot spin" />Testing…</span>;
  }
  if (probe?.state === "ok") {
    return <span className="pv-status tone-good"><span className="pv-status-dot" />Reachable</span>;
  }
  if (probe?.state === "bad") {
    return <span className="pv-status tone-bad"><span className="pv-status-dot" />Unreachable</span>;
  }
  if (inChain) {
    return <span className="pv-status tone-muted">In chain</span>;
  }
  return <span className="pv-status tone-muted">Idle</span>;
}

function ModelsSection({ providers, activeId, probes, timeouts, busyId, chainUseByProvider, onProbe, onTimeout, onEdit, onDelete }) {
  return (
    <Section
      id="models"
      title="Available models"
      desc="Every saved endpoint. Test reaches the provider live; Edit opens it in the form above."
      aside={<span className="pv-chip tone-muted">{providers.length} saved</span>}
    >
      {providers.length === 0 ? (
        <div className="pv-empty">
          <IconPlus size={18} />
          <p>No models yet — add an OpenAI-compatible endpoint above.</p>
        </div>
      ) : (
        <div className="pv-model-list">
          {providers.map((p) => {
            const probeState = probes[p.id];
            const isActive = p.id === activeId;
            const chainNames = chainUseByProvider.get(p.id) || [];
            return (
              <article key={p.id} className={`pv-model-card${isActive ? " active" : ""}`}>
                <div className="pv-model-main">
                  <div className="pv-model-title-row">
                    <h4 className="pv-model-name">{p.name}</h4>
                    <StatusPill probe={probeState} isActive={isActive} inChain={chainNames.length > 0} />
                    {chainNames.length > 0 ? (
                      <span className="pv-chip tone-muted" title={chainNames.join(", ")}>
                        {chainNames.length === 1 ? chainNames[0] : `${chainNames.length} chains`}
                      </span>
                    ) : null}
                  </div>
                  {p.model_name && p.model_name !== p.model ? (
                    <p className="pv-model-label">{p.model_name}</p>
                  ) : null}
                  <dl className="pv-model-meta">
                    <div>
                      <dt>Model ID</dt>
                      <dd><code>{p.model}</code></dd>
                    </div>
                    <div>
                      <dt>Base URL</dt>
                      <dd><code>{p.base_url}</code></dd>
                    </div>
                    <div>
                      <dt>API key</dt>
                      <dd>{p.has_key ? <code>{p.key_hint || "•••• stored"}</code> : <span className="pv-muted">none stored</span>}</dd>
                    </div>
                  </dl>
                  <p
                    className={`pv-probe${probeState ? ` probe-${probeState.state}` : ""}`}
                    role="status"
                  >
                    {probeState && probeState.state !== "testing" ? probeState.msg : "\u00a0"}
                  </p>
                </div>
                <div className="pv-model-actions">
                  <button className="btn" onClick={() => onProbe(p.id)} disabled={busyId === p.id}>
                    <IconRefresh size={14} /> {probeState?.state === "testing" ? "Testing…" : "Test"}
                  </button>
                  <label className="pv-timeout" title="Probe timeout in seconds (5–120)">
                    <input
                      type="number" min={5} max={120} step={1}
                      value={timeouts[p.id] ?? 15}
                      onChange={(e) => onTimeout(p.id, e.target.value)}
                      aria-label={`Test timeout in seconds for ${p.name}`}
                    />
                    <span>s</span>
                  </label>
                  <button className="btn" onClick={() => onEdit(p)}>
                    <IconPencil size={14} /> Edit
                  </button>
                  <ConfirmButton
                    className="btn btn-danger"
                    label="Delete"
                    confirmLabel="Confirm delete"
                    title={`Delete provider ${p.name}`}
                    onConfirm={() => onDelete(p)}
                  />
                </div>
              </article>
            );
          })}
        </div>
      )}
    </Section>
  );
}

/* ---- 3. Fallback chains ------------------------------------------------- */
function ChainsSection({ providers, chains, loading, busy, onError, onReload }) {
  const [newName, setNewName] = useState("");
  const [editingId, setEditingId] = useState(null);
  const [editName, setEditName] = useState("");
  const [drag, setDrag] = useState(null); // { chainId, index }
  const [overIndex, setOverIndex] = useState(null);
  const nameRef = useRef(null);

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

  const saveName = (chain) => {
    const clean = editName.trim();
    if (!clean) return;
    run(async () => { await renameProviderChain(chain.id, clean); setEditingId(null); });
  };

  const addMember = (chain, providerId) =>
    run(() => setProviderChainMembers(chain.id, [...chain.members.map((m) => m.provider_id), providerId]));
  const removeMember = (chain, providerId) =>
    run(() => setProviderChainMembers(chain.id, chain.members.map((m) => m.provider_id).filter((id) => id !== providerId)));

  // Move an item, persist the new order.
  const move = (chain, fromIndex, toIndex) => {
    if (fromIndex === toIndex || fromIndex < 0 || toIndex < 0) return;
    const ids = chain.members.map((m) => m.provider_id);
    const next = [...ids];
    const [item] = next.splice(fromIndex, 1);
    next.splice(toIndex, 0, item);
    run(() => reorderProviderChain(chain.id, next));
  };

  return (
    <Section
      id="chains"
      title="Fallback chains"
      desc="An ordered list of saved models. Priority 1 is the primary; on a provider failure the runtime advances to 2, then 3. Drag a row to reprioritise."
      aside={<span className="pv-chip tone-muted">{chains.length} chain{chains.length === 1 ? "" : "s"}</span>}
    >
      <form className="pv-chain-add" onSubmit={add}>
        <input
          ref={nameRef}
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder="New chain name, e.g. production"
          maxLength={60}
          aria-label="New fallback chain name"
        />
        <button className="btn-primary" type="submit" disabled={busy || !newName.trim()}>
          <IconPlus size={14} /> Add chain
        </button>
      </form>

      {loading ? <p className="pv-muted-note">Loading chains…</p> : null}
      {!loading && chains.length === 0 ? (
        <div className="pv-empty">
          <IconPlus size={18} />
          <p>No chains yet — create one, then order your saved models.</p>
        </div>
      ) : null}

      <div className="pv-chain-list">
        {chains.map((chain) => {
          const { members, available } = resolveChainMembers(
            providers, chain.members.map((m) => m.provider_id),
          );
          const isDragging = drag?.chainId === chain.id;
          return (
            <article key={chain.id} className={`pv-chain-card${chain.is_enabled ? " enabled" : ""}`}>
              <header className="pv-chain-head">
                {editingId === chain.id ? (
                  <span className="pv-chain-rename">
                    <input
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      maxLength={60}
                      aria-label={`Rename chain ${chain.name}`}
                      autoFocus
                      onKeyDown={(e) => { if (e.key === "Enter") saveName(chain); if (e.key === "Escape") setEditingId(null); }}
                    />
                    <button className="btn-primary btn-sm" onClick={() => saveName(chain)} disabled={busy || !editName.trim()}>Save</button>
                    <button className="btn btn-sm" onClick={() => setEditingId(null)}>Cancel</button>
                  </span>
                ) : (
                  <>
                    <span className="pv-chain-title">
                      <h4>{chain.name}</h4>
                      {chain.is_enabled
                        ? <span className="pv-status tone-good"><span className="pv-status-dot" />Serving</span>
                        : <span className="pv-chip tone-muted">{members.length} model{members.length === 1 ? "" : "s"}</span>}
                    </span>
                    <span className="pv-chain-actions">
                      <button className="btn btn-sm" disabled={busy} onClick={() => { setEditingId(chain.id); setEditName(chain.name); }}>
                        <IconPencil size={13} /> Rename
                      </button>
                      <ConfirmButton
                        className="btn btn-danger btn-sm"
                        label="Delete"
                        confirmLabel="Confirm delete"
                        disabled={busy}
                        onConfirm={() => run(() => deleteProviderChain(chain.id))}
                      />
                    </span>
                  </>
                )}
              </header>

              <ol className="pv-members">
                {members.map((p, index) => {
                  const role = chainRole(index);
                  const isOver = isDragging && overIndex === index;
                  return (
                    <li
                      key={p.id}
                      className={`pv-member role-${role.tone}${isDragging && drag.index === index ? " dragging" : ""}${isOver ? " drop-target" : ""}`}
                      draggable={!busy}
                      onDragStart={() => setDrag({ chainId: chain.id, index })}
                      onDragEnter={() => setOverIndex(index)}
                      onDragOver={(e) => e.preventDefault()}
                      onDrop={(e) => {
                        e.preventDefault();
                        if (drag && drag.chainId === chain.id) move(chain, drag.index, index);
                        setDrag(null); setOverIndex(null);
                      }}
                      onDragEnd={() => { setDrag(null); setOverIndex(null); }}
                    >
                      <span className="pv-grip" aria-hidden="true" title="Drag to reorder"><IconMore size={15} /></span>
                      <span className="pv-priority" aria-hidden="true">{index + 1}</span>
                      <span className="pv-member-body">
                        <span className="pv-member-name">
                          {p.name}
                          <span className={`pv-role tone-${role.tone}`}>{role.label}</span>
                        </span>
                        <span className="pv-member-meta"><code>{p.model}</code> · {p.base_url}</span>
                      </span>
                      <span className="pv-member-actions">
                        <button
                          type="button" className="pv-icon-btn" title="Move earlier"
                          onClick={() => move(chain, index, index - 1)}
                          disabled={busy || index === 0}
                          aria-label={`Move ${p.name} earlier`}
                        >
                          <IconChevronDown size={15} className="rot-180" />
                        </button>
                        <button
                          type="button" className="pv-icon-btn" title="Move later"
                          onClick={() => move(chain, index, index + 1)}
                          disabled={busy || index === members.length - 1}
                          aria-label={`Move ${p.name} later`}
                        >
                          <IconChevronDown size={15} />
                        </button>
                        <button
                          type="button" className="pv-icon-btn danger" title={`Remove ${p.name}`}
                          onClick={() => removeMember(chain, p.id)}
                          disabled={busy}
                          aria-label={`Remove ${p.name}`}
                        >
                          <IconX size={15} />
                        </button>
                      </span>
                    </li>
                  );
                })}
                {members.length === 0 ? (
                  <li className="pv-members-empty">No models in this chain yet — add one below.</li>
                ) : null}
              </ol>

              <div className="pv-chain-add-member">
                <span className="pv-eyebrow">
                  Add {members.length === 0 ? "primary" : `fallback ${members.length}`}
                </span>
                {available.length === 0 ? (
                  <p className="pv-muted-note">Every saved model is already in this chain.</p>
                ) : (
                  <div className="pv-picker">
                    {available.map((p) => (
                      <button
                        key={p.id} type="button" className="pv-picker-btn" disabled={busy}
                        onClick={() => addMember(chain, p.id)}
                      >
                        <IconPlus size={13} />
                        <span className="pv-picker-name">{p.name}</span>
                        <span className="pv-picker-model">{p.model}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </article>
          );
        })}
      </div>
    </Section>
  );
}

/* ---- 4. Serving mode ---------------------------------------------------- */
function ServingModeSection({ providers, activeId, chains, busy, onSelectSingle, onSelectChain, onUseDefault }) {
  const enabledChain = chains.find((c) => c.is_enabled) || null;
  const mode = enabledChain ? "chain" : activeId ? "single" : "default";
  const active = providers.find((p) => p.id === activeId) || null;

  return (
    <Section
      id="serving"
      title="Serving mode"
      desc="Choose exactly one: a single model, or one fallback chain. Selecting one automatically disables the other."
      aside={mode === "default"
        ? <span className="pv-chip tone-muted">Env default</span>
        : <span className="pv-chip tone-good">Active</span>}
    >
      <div className="pv-serving-grid" role="radiogroup" aria-label="Serving mode">
        <div className={`pv-serving-card${mode === "single" ? " active" : ""}`} role="radio" aria-checked={mode === "single"} tabIndex={-1}>
          <div className="pv-serving-head">
            <span className="pv-serving-title">
              <span className="pv-radio" aria-hidden="true">{mode === "single" ? <span className="pv-radio-dot" /> : null}</span>
              Single model
            </span>
            {mode === "single"
              ? <button className="btn btn-sm" onClick={onUseDefault} disabled={busy}>Disable</button>
              : <span className="pv-chip tone-muted">Inactive</span>}
          </div>
          <p className="pv-serving-desc">
            {mode === "single"
              ? <>Active — every agent runs on <b>{active?.name}</b>. Pick another to switch.</>
              : "Every agent runs on one model. Selecting a model disables the chain."}
          </p>
          <div className="pv-serving-picker">
            {providers.length === 0 ? (
              <p className="pv-muted-note">Add a model first.</p>
            ) : providers.map((p) => {
              const picked = mode === "single" && p.id === activeId;
              return (
                <button
                  key={p.id} type="button"
                  className={`pv-serving-option${picked ? " picked" : ""}`}
                  onClick={() => onSelectSingle(p.id)}
                  disabled={busy}
                  aria-pressed={picked}
                >
                  <span className="pv-serving-option-name">{p.name}</span>
                  <span className="pv-serving-option-meta">{picked ? "Serving" : "Enable"}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className={`pv-serving-card${mode === "chain" ? " active" : ""}`} role="radio" aria-checked={mode === "chain"} tabIndex={-1}>
          <div className="pv-serving-head">
            <span className="pv-serving-title">
              <span className="pv-radio" aria-hidden="true">{mode === "chain" ? <span className="pv-radio-dot" /> : null}</span>
              Fallback chain
            </span>
            {mode === "chain"
              ? <button className="btn btn-sm" onClick={onUseDefault} disabled={busy}>Disable</button>
              : <span className="pv-chip tone-muted">Inactive</span>}
          </div>
          <p className="pv-serving-desc">
            {mode === "chain"
              ? <>Active — trying <b>{enabledChain?.name}</b> in priority order. Pick another to switch.</>
              : "Try the primary, then fall back in order. Selecting a chain disables the single model."}
          </p>
          <div className="pv-serving-picker">
            {chains.length === 0 ? (
              <p className="pv-muted-note">Create a chain first.</p>
            ) : chains.map((c) => (
              <button
                key={c.id} type="button"
                className={`pv-serving-option${c.is_enabled ? " picked" : ""}`}
                onClick={() => onSelectChain(c.id)}
                disabled={busy || c.members.length === 0}
                aria-pressed={c.is_enabled}
                title={c.members.length === 0 ? "Add models to this chain first" : undefined}
              >
                <span className="pv-serving-option-name">{c.name}</span>
                <span className="pv-serving-option-meta">
                  {c.members.length} model{c.members.length === 1 ? "" : "s"}
                  {c.is_enabled ? " · Serving" : " · Enable"}
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="pv-serving-foot">
        <span className="pv-hint">
          {mode === "default"
            ? "Default env chain in use (Groq / HuggingFace / CUSTOM_LLM_*)."
            : "Selecting a mode automatically disables the other."}
        </span>
      </div>
    </Section>
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
  const [tab, setTab] = useState(
    () => providerTabFromUrl(window.location.hash, window.location.search),
  );

  /* Follow the URL when it changes underneath us (back/forward, or a link
   * into a specific sub-page). go() uses pushState, which fires neither
   * popstate nor hashchange here, so this cannot fight the click handler. */
  useEffect(() => {
    const sync = () => setTab(providerTabFromUrl(window.location.hash, window.location.search));
    window.addEventListener("hashchange", sync);
    window.addEventListener("popstate", sync);
    return () => {
      window.removeEventListener("hashchange", sync);
      window.removeEventListener("popstate", sync);
    };
  }, []);

  /* pushState (not location.hash) so the tab is a real history entry and
   * "back" leaves the tab rather than only rewriting the fragment. The
   * search-stripping fallback covers non-browser/test environments. */
  const goTab = useCallback((next) => {
    if (!isProviderTab(next)) return;
    setTab(next);
    const url = next === DEFAULT_PROVIDER_TAB ? HASH_PREFIX : `${HASH_PREFIX}?tab=${next}`;
    try {
      window.history.pushState({ view: "providers", tab: next }, "", url);
    } catch {
      try { window.location.hash = url.slice(1); } catch { /* no DOM */ }
    }
  }, []);

  /* Left/right arrows move between tabs, per the ARIA tabs pattern. The tab
   * list is a single tab stop, so the arrows are the only way across. */
  const onTabKeyDown = useCallback((e) => {
    const dir = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!dir) return;
    e.preventDefault();
    const i = PROVIDER_TABS.findIndex((t) => t.id === tab);
    const next = PROVIDER_TABS[(i + dir + PROVIDER_TABS.length) % PROVIDER_TABS.length];
    goTab(next.id);
    document.getElementById(`pv-tab-${next.id}`)?.focus();
  }, [tab, goTab]);

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
    ? `chain “${enabledChain.name}”`
    : activeProvider ? activeProvider.name : "the default chain";

  // Map provider id → chain names it belongs to (for the model cards).
  const chainUseByProvider = useMemo(() => {
    const map = new Map();
    for (const c of chains) {
      for (const m of c.members || []) {
        const list = map.get(m.provider_id) || [];
        list.push(c.name);
        map.set(m.provider_id, list);
      }
    }
    return map;
  }, [chains]);

  return (
    <div className="view anim-rise pv-view">
      <div className="view-head pv-view-head">
        <div>
          <h2>Providers</h2>
          <p>
            {providers.length} model{providers.length === 1 ? "" : "s"} ·{" "}
            {chains.length} chain{chains.length === 1 ? "" : "s"} · serving:{" "}
            <strong>{serving}</strong>
          </p>
        </div>
        <div className="pv-summary" aria-label="Serving summary">
          <span className={`pv-summary-item${enabledChain || activeProvider ? " on" : ""}`}>
            <span className="pv-summary-label">Mode</span>
            <span className="pv-summary-value">
              {enabledChain ? "Fallback chain" : activeProvider ? "Single model" : "Env default"}
            </span>
          </span>
        </div>
      </div>

      {error ? <div className="error-box" role="alert">{error}</div> : null}

      {/* Tab list. Counts ride on the tabs so the collapsed panels stay
          legible without opening them. */}
      <div className="pv-tabs" role="tablist" aria-label="Provider settings sections" onKeyDown={onTabKeyDown}>
        {PROVIDER_TABS.map((t) => {
          const count = t.id === "models" ? providers.length
            : t.id === "chains" ? chains.length
            : null;
          return (
            <button
              key={t.id}
              id={`pv-tab-${t.id}`}
              type="button"
              role="tab"
              className={`pv-tab${tab === t.id ? " active" : ""}`}
              aria-selected={tab === t.id}
              aria-controls={`pv-panel-${t.id}`}
              tabIndex={tab === t.id ? 0 : -1}
              onClick={() => goTab(t.id)}
            >
              {t.label}
              {count !== null ? <span className="pv-tab-count">{count}</span> : null}
            </button>
          );
        })}
      </div>

      <div className="pv-sections">
        {tab === "add" ? (
          <AddModelSection editing={editing} onSaved={onSaved} onCancel={() => setEditing(null)} />
        ) : null}

        {tab === "models" ? (
          loading ? <p className="pv-muted-note">Loading models…</p> : (
            <ModelsSection
              providers={providers}
              activeId={activeId}
              probes={probes}
              timeouts={timeouts}
              busyId={busyId}
              chainUseByProvider={chainUseByProvider}
              onProbe={probe}
              onTimeout={(id, v) => setTimeouts((prev) => ({ ...prev, [id]: v }))}
              onEdit={(p) => { setEditing(p); goTab("add"); }}
              onDelete={remove}
            />
          )
        ) : null}

        {tab === "chains" ? (
          <ChainsSection
            providers={providers}
            chains={chains}
            loading={chainsLoading}
            busy={busy}
            onError={setError}
            onReload={refreshChains}
          />
        ) : null}

        {tab === "serving" ? (
          <ServingModeSection
            providers={providers}
            activeId={activeId}
            chains={chains}
            busy={busy}
            onSelectSingle={selectSingle}
            onSelectChain={selectChain}
            onUseDefault={useDefault}
          />
        ) : null}
      </div>
    </div>
  );
}
