import { extractDomain, trustOf } from "../lib";
import { IconX } from "./icons";

/* Verification detail for one finding — every field comes from the
 * backend findings stream (claim, source, verified, verification_score,
 * verification_reason, confidence). Nothing is synthesized. */

export default function ClaimDrawer({ finding, onClose, onSave, saved }) {
  if (!finding) return null;
  const domain = extractDomain(finding.source || "");
  const trust = trustOf(domain);

  return (
    <>
      <button className="drawer-scrim" onClick={onClose} aria-label="Close detail" />
      <div className="drawer" role="dialog" aria-label="Claim verification detail">
        <div className="close-row">
          <h2>Claim inspector</h2>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            <IconX size={16} />
          </button>
        </div>
        {onSave ? (
          <div style={{ marginTop: 12 }}>
            <button className="btn" onClick={() => onSave(finding)} disabled={saved}>
              {saved ? "Saved to knowledge" : "Save to knowledge"}
            </button>
          </div>
        ) : null}

        <div className="tag-row">
          {typeof finding.verified === "boolean" ? (
            <span className={`tag tone-${finding.verified ? "good" : "bad"}`}>
              {finding.verified ? "Verified" : "Unverified"}
            </span>
          ) : (
            <span className="tag tone-muted">Verification pending</span>
          )}
          <span className={`tag tone-${trust.tone === "good" ? "good" : trust.tone === "bad" ? "bad" : "muted"}`}>
            {trust.label} source
          </span>
          {finding.challenged ? <span className="tag tone-bad">Challenged by critic</span> : null}
        </div>

        <div className="kv">
          <div className="k">Claim</div>
          <div className="v">{finding.claim || "—"}</div>
        </div>

        <div className="kv">
          <div className="k">Analyzed by</div>
          <div className="v">{finding.agent || "unknown specialist"}</div>
        </div>

        <div className="kv">
          <div className="k">Source</div>
          <div className="v">
            {finding.source ? (
              <a href={finding.source} target="_blank" rel="noopener noreferrer" className="src-link">
                {domain || finding.source}
              </a>
            ) : (
              "No source URL recorded"
            )}
          </div>
        </div>

        {typeof finding.confidence === "number" ? (
          <div className="kv">
            <div className="k">Claim confidence</div>
            <div className="v">{Math.round(finding.confidence * 100)}%</div>
            <div className="bar" style={{ marginTop: 8 }}>
              <div style={{ width: `${Math.round(finding.confidence * 100)}%` }} />
            </div>
          </div>
        ) : null}

        {typeof finding.verification_score === "number" ? (
          <div className="kv">
            <div className="k">Verification score</div>
            <div className="v">{Math.round(finding.verification_score * 100)}%</div>
            <div className="bar" style={{ marginTop: 8 }}>
              <div className={finding.verified ? "green" : ""} style={{ width: `${Math.round(finding.verification_score * 100)}%` }} />
            </div>
          </div>
        ) : null}

        {finding.verification_reason ? (
          <div className="kv">
            <div className="k">Why this verdict</div>
            <div className="v">{finding.verification_reason}</div>
          </div>
        ) : null}
      </div>
    </>
  );
}
