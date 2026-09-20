import { useCallback, useEffect, useState } from "react";
import {
  clearProviderChain,
  createProviderChain,
  deleteProviderChain,
  listProviderChains,
  renameProviderChain,
  reorderProviderChain,
  setProviderChainEnabled,
  setProviderChainMembers,
} from "../api";
import { chainRole, resolveChainMembers } from "../lib";

/* Provider fallback chains: ordered primary → fallback lists built from the
 * saved providers above. At most one chain is enabled; the runtime tries
 * member #1 first and fails over in order on provider/transient errors.
 * Chains only reference existing providers — no keys or config are entered
 * here, so this never duplicates provider logic. */

export default function ChainsSection({ providers, onError }) {
  const [chains, setChains] = useState([]);
  const [enabledId, setEnabledId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [newName, setNewName] = useState("");
  const [busy, setBusy] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [editName, setEditName] = useState("");

  const refresh = useCallback(async () => {
    try {
      const data = await listProviderChains();
      setChains(Array.isArray(data?.chains) ? data.chains : []);
      setEnabledId(data?.enabled_id ?? null);
    } catch (e) {
      if (onError) onError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [onError]);

  useEffect(() => {
    refresh();
  }, [refresh]);

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

  const remove = (chain) => {
    if (!window.confirm(`Delete fallback chain "${chain.name}"? Providers themselves are kept.`)) return;
    run(() => deleteProviderChain(chain.id));
  };

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
          Try provider 1 first; on a provider/transient failure, continue to
          provider 2, then 3. Only one chain runs at a time.
        </p>
      </div>

      <form className="chain-add" onSubmit={add}>
        <input
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder="New chain name (e.g. production)"
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
          No fallback chains yet. Add one, then order saved providers as primary and fallbacks.
        </p>
      ) : null}

      {chains.map((chain) => {
        const { members, available } = resolveChainMembers(providers, chain.members.map((m) => m.provider_id));
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
                    {chain.is_enabled ? (
                      <span className="tag tone-blue">Enabled</span>
                    ) : (
                      <span className="tag">Disabled</span>
                    )}
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
                    <button className="btn btn-danger" onClick={() => remove(chain)} disabled={busy}>
                      Delete
                    </button>
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
                        <span className={`tag tone-${role.tone}`}>{role.label}</span>
                      </span>
                      <span className="chain-member-meta">{p.model} · {p.base_url}</span>
                    </span>
                    <span className="chain-member-actions">
                      <button
                        className="btn" title="Move up"
                        onClick={() => move(chain, index, index - 1)}
                        disabled={busy || index === 0}
                        aria-label={`Move ${p.name} up`}
                      >
                        ↑
                      </button>
                      <button
                        className="btn" title="Move down"
                        onClick={() => move(chain, index, index + 1)}
                        disabled={busy || index === members.length - 1}
                        aria-label={`Move ${p.name} down`}
                      >
                        ↓
                      </button>
                      <button
                        className="btn btn-danger"
                        onClick={() => removeMember(chain, p.id)}
                        disabled={busy}
                        aria-label={`Remove ${p.name} from chain`}
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

            <div className="chain-add-member">
              <label>
                Add provider
                <select
                  value=""
                  onChange={(e) => { if (e.target.value) addMember(chain, Number(e.target.value)); }}
                  disabled={busy || available.length === 0}
                  aria-label={`Add provider to ${chain.name}`}
                >
                  <option value="">
                    {available.length === 0 ? "All providers already in this chain" : "Select a provider…"}
                  </option>
                  {available.map((p) => (
                    <option key={p.id} value={p.id}>
                      {members.length === 0 ? "Primary — " : "Fallback — "}{p.name} ({p.model})
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>
        );
      })}
    </section>
  );
}
