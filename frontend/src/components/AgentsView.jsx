/* Agent Library — the MARS workforce as it actually exists. Objectives and
 * tools are stated from the code, not marketing copy. Live per-agent latency
 * and cost are not instrumented yet — said plainly rather than faked. */

import { IconChart, IconCheckCircle, IconDoc, IconSearch, IconShield, IconShieldCheck, IconTarget } from "./icons";

const AGENTS = [
  {
    name: "Orchestrator", icon: IconTarget,
    objective: "Scores query complexity and sets strategy, agent count and iteration caps.",
    tools: ["complexity scorer", "mode presets"],
    outputs: ["strategy selection", "hardware-cap clamping"],
  },
  {
    name: "Planner", icon: IconTarget,
    objective: "Decomposes the question into delegation contracts with search types and priorities.",
    tools: ["LLM planning", "methodology prompt"],
    outputs: ["sub-questions", "diversity contracts", "query variants"],
  },
  {
    name: "Search", icon: IconSearch,
    objective: "Retrieves and ranks sources per sub-question, then fetches full content for the top hits.",
    tools: ["web_search", "fetch_content"],
    outputs: ["dedup + ranking", "variant fan-out", "already-searched skip"],
  },
  {
    name: "Summarizer", icon: IconDoc,
    objective: "Extracts standalone factual claims in isolated per-question contexts.",
    tools: ["LLM extraction", "heuristic fallback"],
    outputs: ["7 domain overlays", "snippet cleaning"],
  },
  {
    name: "Verifier", icon: IconCheckCircle,
    objective: "Checks every claim against its cited source; flags contradictions and truncations.",
    tools: ["lexical overlap", "contradiction engine"],
    outputs: ["verified flags", "answer-support scoring"],
  },
  {
    name: "Critic", icon: IconShield,
    objective: "Judges sufficiency, red-teams assumptions, and forces expansion within the depth ceiling.",
    tools: ["LLM critique", "synthesis gate"],
    outputs: ["confidence breakdown", "gap queries"],
  },
  {
    name: "Synthesizer", icon: IconChart,
    objective: "Writes the cited final answer with a numbered source legend.",
    tools: ["LLM synthesis", "deterministic fallback"],
    outputs: ["inline [n] citations", "section-wise structure"],
  },
];

export default function AgentsView() {
  return (
    <div className="view anim-rise">
      <div className="view-head">
        <h2>Agents</h2>
        <p>
          {AGENTS.length} pipeline agents in execution order · specialist overlays ride the
          summarizer
        </p>
      </div>

      {/* The same numbered spine used in the intelligence rail: agent order
          is a real, stable sequence, not decoration. */}
      <ol className="spine" aria-label="Pipeline execution order">
        {AGENTS.map((a, i) => (
          <li key={a.name} className="spine-node">
            <span className="spine-idx">{i + 1}</span>
            <span className="spine-body">
              <span className="spine-title">
                <a.icon size={14} style={{ color: "var(--t3)", flex: "none" }} />
                {a.name}
              </span>
              <span className="spine-desc">{a.objective}</span>
              <span className="spine-meta">
                tools: {a.tools.join(" · ")} — outputs: {a.outputs.join(" · ")}
              </span>
            </span>
          </li>
        ))}
      </ol>

      <p className="empty" style={{ marginTop: 18 }}>
        Per-agent latency, cost and success rate need tool-call tracing, which is not
        instrumented yet. Nothing above is estimated.
      </p>
    </div>
  );
}
