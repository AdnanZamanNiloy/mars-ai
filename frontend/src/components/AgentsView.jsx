/* Agent Library — the MARS workforce as it actually exists: the seven
 * pipeline agents plus specialist overlays. Objectives and tools are
 * stated from the code, not marketing copy. Live per-agent latency and
 * cost need tool-call tracing (not yet instrumented) — said plainly
 * below instead of faked. */

const AGENTS = [
  { name: "Orchestrator", objective: "Scores query complexity and sets strategy, agent count, and caps.",
    tools: ["complexity scorer", "mode presets"], capabilities: ["strategy selection", "hardware-cap clamping"] },
  { name: "Planner", objective: "Decomposes the query into delegation contracts with search types and priorities.",
    tools: ["LLM planning", "methodology prompt"], capabilities: ["sub-questions", "diversity contracts", "query variants"] },
  { name: "Search", objective: "Retrieves and ranks sources per sub-question, fetches full content for the top hits.",
    tools: ["web_search", "fetch_content"], capabilities: ["dedup + ranking", "variant fan-out", "skip-searched"] },
  { name: "Summarizer", objective: "Extracts standalone factual claims in isolated per-question contexts.",
    tools: ["LLM extraction", "heuristic fallback"], capabilities: ["financial/technical/market/legal/scientific/policy/academic overlays", "snippet cleaning"] },
  { name: "Verifier", objective: "Checks every claim against its cited source; flags contradictions and truncations.",
    tools: ["lexical overlap", "contradiction engine"], capabilities: ["verified flags", "answer-support scoring"] },
  { name: "Critic", objective: "Judges sufficiency, red-teams assumptions, and forces expansion within the depth ceiling.",
    tools: ["LLM critique", "synthesis gate"], capabilities: ["confidence breakdown", "gap queries"] },
  { name: "Synthesizer", objective: "Writes the cited final answer with a numbered source legend.",
    tools: ["LLM synthesis", "deterministic fallback"], capabilities: ["inline [n] citations", "paragraph structure"] },
];

export default function AgentsView() {
  return (
    <div className="view anim-rise">
      <div className="view-head">
        <h2>Agents</h2>
        <p>{AGENTS.length} pipeline agents · specialist overlays ride the summarizer</p>
      </div>
      {AGENTS.map((a) => (
        <div key={a.name} className="mission-card" style={{ cursor: "default" }}>
          <span className="dot done" style={{ marginTop: 6 }} />
          <span className="body">
            <p className="q">{a.name}</p>
            <span className="meta" style={{ display: "block", marginTop: 6 }}>
              <span style={{ display: "block", marginBottom: 6 }}>{a.objective}</span>
              <span style={{ display: "block" }}>Tools: {a.tools.join(" · ")}</span>
              <span style={{ display: "block" }}>Capabilities: {a.capabilities.join(" · ")}</span>
            </span>
          </span>
        </div>
      ))}
      <p className="empty">Live per-agent latency, cost, and success rates need tool-call tracing, which is not instrumented yet.</p>
    </div>
  );
}
