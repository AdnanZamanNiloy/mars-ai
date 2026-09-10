import { useEffect, useState } from "react";
import { fetchEvalBatches } from "../api";

/* Evaluation Lab dashboard — read-only view over persisted eval batches.
 * Degraded rows are excluded from summaries server-side (count shown). */

function pct(value) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "—";
}

export default function EvalView() {
  const [batches, setBatches] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    fetchEvalBatches(5, controller.signal)
      .then((data) => setBatches(data.batches || []))
      .catch((err) => {
        if (err.name !== "AbortError") setError(err.message || "Could not load eval batches");
      });
    return () => controller.abort();
  }, []);

  if (error) {
    return (
      <div className="view anim-rise">
        <div className="view-head"><h2>Evaluations</h2></div>
        <div className="error-box">{error}</div>
      </div>
    );
  }
  if (batches === null) {
    return (
      <div className="view anim-rise">
        <div className="view-head"><h2>Evaluations</h2><p>Loading batches…</p></div>
        <p className="empty">Fetching persisted evaluation history.</p>
      </div>
    );
  }
  if (batches.length === 0) {
    return (
      <div className="view anim-rise">
        <div className="view-head"><h2>Evaluations</h2><p>No batches yet</p></div>
        <p className="empty">Run scripts/run_eval.py to produce the first batch.</p>
      </div>
    );
  }
  return (
    <div className="view anim-rise">
      <div className="view-head">
        <h2>Evaluations</h2>
        <p>{batches.length} batch{batches.length === 1 ? "" : "es"} · degraded rows excluded from trends</p>
      </div>
      {batches.map((b) => (
        <div key={b.batch} style={{ marginBottom: 22 }}>
          <div className="intel-section-head" style={{ marginBottom: 10 }}>
            <h3 style={{ fontSize: 14 }}>Batch {b.batch}</h3>
            <span className="tag tone-muted">{b.summary.queries} queries</span>
          </div>
          <div className="stat-grid" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
            <div className="stat"><div><b>{pct(b.summary.pass_rate)}</b><div className="lbl">Pass rate</div></div></div>
            <div className="stat"><div><b>{b.summary.avg_confidence?.toFixed(2) ?? "—"}</b><div className="lbl">Avg confidence</div></div></div>
            <div className="stat"><div><b>{b.summary.avg_verified?.toFixed(1) ?? "—"}</b><div className="lbl">Avg verified</div></div></div>
            <div className="stat"><div><b>{b.summary.avg_cost != null ? `$${b.summary.avg_cost.toFixed(4)}` : "—"}</b><div className="lbl">Avg cost</div></div></div>
          </div>
          {b.degraded_excluded > 0 ? (
            <p className="sub" style={{ marginTop: 8 }}>{b.degraded_excluded} degraded row{b.degraded_excluded === 1 ? "" : "s"} excluded</p>
          ) : null}
          <div style={{ marginTop: 12 }}>
            {b.rows.map((r) => (
              <div key={`${b.batch}-${r.query_id}`} className="mission-card" style={{ cursor: "default" }}>
                <span className={`dot ${r.passed ? "done" : "bad"}`} style={{ marginTop: 6 }} />
                <span className="body">
                  <p className="q">{r.query || r.query_id}</p>
                  <span className="meta">
                    <span className="tag tone-muted">{r.mode}</span>
                    {typeof r.confidence === "number" ? <span>{Math.round(r.confidence * 100)}% conf</span> : null}
                    <span>{r.claims ?? "—"} claims · {r.verified ?? "—"} verified</span>
                    {r.degraded?.length > 0 ? <span className="tag tone-bad">degraded: {r.degraded.join(", ")}</span> : null}
                    {typeof r.judge_score === "number" ? <span>judge {r.judge_score.toFixed(1)}/5</span> : null}
                  </span>
                </span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
