import { useState } from "react";

/**
 * Claim Inspector (Phase 3.6, Evidence Explorer MVP).
 *
 * A simple detail panel — no graph rendering — showing, for a selected
 * finding: the claim, its source, the verification result, and confidence.
 * Opens when a user clicks a finding card.
 */
export default function ClaimInspector({ finding, onClose }) {
  const [dismissed, setDismissed] = useState(false);

  if (!finding || dismissed) return null;

  const verified = finding.verified;
  const hasVerification = typeof verified === "boolean";
  const score = typeof finding.verification_score === "number" ? finding.verification_score : null;
  const confidence = typeof finding.confidence === "number" ? finding.confidence : null;

  let hostname = finding.source || "";
  try {
    hostname = new URL(finding.source).hostname;
  } catch {
    /* keep raw string */
  }

  return (
    <div className="claim-inspector" role="dialog" aria-label="Claim detail">
      <div className="claim-inspector-head">
        <strong>Claim detail</strong>
        <button
          type="button"
          className="claim-inspector-close"
          onClick={() => {
            setDismissed(true);
            if (onClose) onClose();
          }}
          aria-label="Close claim detail"
        >
          ×
        </button>
      </div>

      <p className="claim-inspector-claim">{finding.claim}</p>

      {finding.source ? (
        <p className="meta">
          Source: <a href={finding.source} target="_blank" rel="noreferrer">{hostname}</a>
        </p>
      ) : null}

      {hasVerification ? (
        <p className={`claim-verification ${verified ? "is-verified" : "is-unverified"}`}>
          {verified ? "Verified" : "Not verified"}
          {score != null ? ` · lexical overlap ${score.toFixed(2)}` : ""}
        </p>
      ) : (
        <p className="meta">Verification pending…</p>
      )}

      {finding.verification_reason ? (
        <p className="meta">{finding.verification_reason}</p>
      ) : null}

      {confidence != null ? (
        <p className="meta">Confidence: {(confidence * 100).toFixed(0)}%</p>
      ) : null}
    </div>
  );
}
