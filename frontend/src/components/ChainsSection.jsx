import { useCallback, useEffect, useState } from "react";
import {
  clearProviderChain, createProviderChain, deleteProviderChain, listProviderChains,
  renameProviderChain, reorderProviderChain, setProviderChainEnabled, setProviderChainMembers,
} from "../api";
import { chainRole, resolveChainMembers } from "../lib";
import ConfirmButton from "./ConfirmButton";

/* Provider fallback chains: ordered primary → fallback lists built from the
 * saved providers above. At most one chain is enabled; the runtime tries
 * member #1 first and fails over in order on provider/transient errors.
 * Chains only reference existing providers — no keys or config are entered
 * here, so this never duplicates provider logic. */

export default function ChainsSection({ providers, onError }) {
  const [chains, setChains] = useState([]);
  const [loading, setLoading] = useState(true);
  const [newName, setNewName] = useState("");
  const [busy, setBusy] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [editName, setEditName] = useState("");
  const [addingTo, setAddingTo] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const data = await listProviderChains();
      setChains(Array.isArray(data?.chains) ? data.chains : []);
    } catch (e) {
      if (onError) onError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [onError]);

  useEffect(() => { refresh(); }, [refresh]);

  const fail = (e) => {
    if (onError) onError(e instanceof Error ? e.message : String(e));
  };

  const run = async (fn) => {
    setBusy(true);
    try {
      await fn();
      await refresh();
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  };

  const add = (e) => {
    e.preventDefault();
    if (!newName.trim()) return;
    run(async () => {
      await createProviderChain(newName.trim());
      setNewName("");
    });
  };

  const toggle = (chain) =>
    run(async () => {
      if (chain.is_enabled) await clearProviderChain();
      else await setProviderChainEnabled(chain.id, true);
    });

  const remove = (chain) => run(() => deleteProviderChain(chain.id));

  const saveName = (chain) =>
    run(async () => {
      await renameProviderChain(chain.id, editName.trim());
      setEditingId(null);
    });

  const move = (chain, index, target) => {
    const ids = chain.members.map((m) => m.provider_id);
    const next = [...ids];
    const [item] = next.splice(index, 1);
    next.splice(target, 0, item);
    run(() => reorderProviderChain(chain.id, next));
  };

  const addMember = (chain, providerId) =>
    run(() => setProviderChainMembers(
      chain.id, [...chain.members.map((m) => m.provider_id), providerId],
    ));

  const removeMember = (chain, providerId) =>
    run(() => setProviderChainMembers(
      chain.id, chain.members.map((m) => m.provider_id).filter((id) => id !== providerId),
    ));

  return (
    <section className="chain-section">
      <div className="view-head chain-head">
        <h3>Fallback chains</h3>
        <p>
          Provider 1 is tried first. On a provider or transient failure the runtime continues
          to provider 2, then 3 — so one outage does not end a run. Only one chain is enabled
          at a time.
        </p>
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
        <p className="empty">
          No fallback chains yet. Add one, then order your saved providers as primary and
          fallbacks.
        </p>
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
                    <span className={`tag tone-${chain.is_enabled ? "good" : "muted"}`}>
                      {chain.is_enabled ? "Serving traffic" : "Disabled"}
                    </span>
                  </span>
                  <span className="chain-actions">
                    <button className="btn" onClick={() => toggle(chain)} disabled={busy}>
                      {chain.is_enabled ? "Disable" : "Enable"}
                    </button>
                    <button
                      className="btn"
                      onClick={() => { setEditingId(chain.id); setEditName(chain.name); }}
                      disabled={busy}
                    >
                      Rename
                    </button>
                    <ConfirmButton
                      className="btn btn-danger"
                      label="Delete"
                      confirmLabel="Confirm delete"
                      onConfirm={() => remove(chain)}
                      disabled={busy}
                    />
                  </span>
                </>
              )}
            </div>

            <ol className="chain-members">
              {members.map((p, index) => {
                const role = chainRole(index);
                return (
                  <li key={p.id} className={`chain-member role-${role.tone}`}>
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
                        onClick={() => move(chain, index, index - 1)}
                        disabled={busy || index === 0}
                        aria-label={`Move ${p.name} earlier in the chain`}
                      >
                        ↑
                      </button>
                      <button
                        type="button" className="btn btn-sm" title="Move down"
                        onClick={() => move(chain, index, index + 1)}
                        disabled={busy || index === members.length - 1}
                        aria-label={`Move ${p.name} later in the chain`}
                      >
                        ↓
                      </button>
                      <button
                        type="button" className="btn btn-sm"
                        onClick={() => removeMember(chain, p.id)}
                        disabled={busy}
                        aria-label={`Remove ${p.name} from this chain`}
                      >
                        Remove
                      </button>
                    </span>
                  </li>
                );
              })}
              {members.length === 0 ? (
                <li className="empty">No providers in this chain yet — add one below.</li>
              ) : null}
            </ol>

            {/* Button-based picker with the role made explicit, replacing the
                <select> that silently labelled the first member "Primary". */}
            <div className="chain-add-member">
              <div className="chain-picker">
                <span className="eyebrow">
                  Add {members.length === 0 ? "primary" : `fallback ${members.length + 1}`}
                </span>
                {available.length === 0 ? (
                  <p className="empty" style={{ marginTop: 8 }}>
                    Every saved provider is already in this chain.
                  </p>
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
          </div>
        );
      })}
    </section>
  );
}
