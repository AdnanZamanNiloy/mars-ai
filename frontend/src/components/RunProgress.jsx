/* Live run panel — the console's instrument readout while the pipeline works.
 *
 * This is the ONE place the in-flight state is shown, replacing the old
 * three-dancing-dots `TypingRow`. It renders the REAL 7-agent MARS pipeline in
 * execution order (matching AgentsView and the sidebar mission spine) and
 * advances each node off data the stream has actually delivered — never a
 * timer, never a fabricated percentage.
 *
 * Why this shape: a research run can take minutes, and the honest signal is
 * "which stage is happening, and what has it produced so far". Progress is
 * therefore derived from observable artifacts (plan items, snippet counts,
 * verified claims, critic iterations) rather than a fake bar.
 */

import { IconCheck, IconShield } from "./icons";

/* Execution order — kept in the same sequence as AgentsView so the number
 * beside a stage here means the same thing everywhere in the app. */
const STAGES = [
  { id: "orchestrator", label: "Orchestrator", desc: "Scoring complexity and setting strategy" },
  { id: "planner", label: "Planner", desc: "Decomposing the question into contracts" },
  { id: "search", label: "Search", desc: "Retrieving, ranking and fetching sources" },
  { id: "summarizer", label: "Summarizer", desc: "Extracting claims in isolated contexts" },
  { id: "verifier", label: "Verifier", desc: "Checking each claim against its source" },
  { id: "critic", label: "Critic", desc: "Judging sufficiency and red-teaming assumptions" },
  { id: "synthesizer", label: "Synthesizer", desc: "Writing the cited final answer" },
];

/* Map the run's observable fields onto a stage index. Returns the index of
 * the stage currently in progress; every earlier stage counts as done, and a
 * later stage has not started. This is deliberately monotonic-friendly: each
 * signal only ever moves the marker forward as data arrives. */
function stageIndex(run) {
  if (run.plan && run.plan.length) {
    if (run.report) return STAGES.length; // everything finished
    if (run.critiques && run.critiques.length) return 6; // synthesizing
    if (typeof run.verifiedCount === "number" && run.verifiedCount > 0) return 5; // critic
    if (run.findings && run.findings.length) return 4; // verifying
    if (run.snippets > 0) return 3; // summarizing began
    return 2; // plan received → searching
  }
  if (run.intent || run.route) return 1;
  return 0;
}

/* One honest, stage-specific figure where the run has actually produced one.
 * Returns a plain string (not an element) so callers can fall back to the
 * stage description with `||`. Nothing invented: every branch reads a field
 * the event stream has really populated. */
function stageStat(run) {
  if (run.critiques && run.critiques.length) {
    return `${run.findings?.length || 0} claims · ${run.verifiedCount || 0} verified · pass ${run.critiques.length}`;
  }
  if (run.findings && run.findings.length) {
    return `${run.findings.length} claims · ${run.verifiedCount || 0} verified`;
  }
  if (typeof run.snippets === "number" && run.snippets > 0) {
    return `${run.snippets} sources gathered`;
  }
  return "";
}

export default function RunProgress({ run, onAbort }) {
  const active = Math.min(stageIndex(run), STAGES.length - 1);
  const mode = run.modeLabel || run.mode || "standard";
  const agentCount = run.plan?.length || 0;

  return (
    <div className="run-progress anim-rise" aria-live="polite">
      <div className="run-progress-head">
        <span className="run-progress-title">
          <span className="dot live" aria-hidden="true" />
          Researching
        </span>
        <span className="run-progress-meta">
          {agentCount > 0 ? `${agentCount} agents · ` : ""}
          {mode}
        </span>
        {onAbort ? (
          <button
            type="button"
            className="btn btn-sm run-abort"
            onClick={onAbort}
            title="Stop this run"
          >
            <IconShield size={13} />
            Stop
          </button>
        ) : null}
      </div>

      <ul className="spine run-spine run-spine-compact">
        {STAGES.map((stage, i) => {
              const state = i < active ? "s-done" : i === active ? "s-active" : "";
          // Live figure for the running stage, or its description when that
          // stage has not produced a countable artifact yet. Computed here
          // rather than via `<StageMeta/> || desc`, which could never fall
          // back because a rendered element is always truthy.
          const detail = i === active ? (stageStat(run) || stage.desc) : stage.desc;
          return (
            <li className={`spine-node ${state}`} key={stage.id}>
              <span className="spine-idx" aria-hidden="true">
                {i < active ? <IconCheck size={12} /> : i + 1}
              </span>
              <div className="spine-body">
                <div className="spine-title">
                  {stage.label}
                  <span className="spine-state">
                    {i < active ? "done" : i === active ? "running" : "queued"}
                  </span>
                </div>
                <div className="spine-desc">{detail}</div>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
