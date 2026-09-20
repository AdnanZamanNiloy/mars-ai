/* MARS landing — a working front door.
 *
 * Design direction: the landing does the same job as the console's welcome
 * screen, at full width. Instead of a stock AI-marketing composition (three
 * blurred orbs, eight twinkling stars, a giant letter-spaced wordmark), the
 * right column shows the product's ACTUAL shape: the 7-agent pipeline as a
 * numbered spine, and a real evidence ledger with verification outcomes.
 * That pipeline is the product; showing it beats describing it.
 *
 * Pure presentational — no data fetching, no fabricated run state. The
 * pipeline order and the claim ledger are the same facts the Agents view and
 * the intelligence rail render from code. */

import { Planet } from "./Sidebar";
import ThemeToggle from "./ThemeToggle";
import { IconDoc, IconGithub } from "./icons";

const GITHUB_URL = "https://github.com/AdnanZamanNiloy/mars-ai";

const PIPELINE = [
  { name: "Orchestrator", desc: "Scores complexity, sets agent count and iteration caps" },
  { name: "Planner", desc: "Decomposes the question into delegation contracts" },
  { name: "Search", desc: "Retrieves, ranks and fetches sources per sub-question" },
  { name: "Summarizer", desc: "Extracts standalone factual claims with domain overlays" },
  { name: "Verifier", desc: "Checks each claim against its own cited source" },
  { name: "Critic", desc: "Red-teams assumptions and forces expansion on gaps" },
  { name: "Synthesizer", desc: "Writes the cited report with a numbered source legend" },
];

const LEDGER = [
  { claim: "Claim checked against its cited source", tone: "good", label: "verified" },
  { claim: "Two sources disagree on the figure", tone: "warn", label: "contradiction" },
  { claim: "Claim has no retrievable source", tone: "bad", label: "unsupported" },
];

export default function Landing({ onStart, onDocs }) {
  return (
    <div className="landing">
      <header className="landing-nav">
        <a
          className="landing-brand"
          href="#/"
          onClick={(e) => e.preventDefault()}
          aria-label="MARS — Multi-Agent Research System"
        >
          <Planet size={36} />
          <span className="brand-copy">
            <span className="brand-name">MARS</span>
            <span className="brand-sub">Multi-Agent Research System</span>
          </span>
        </a>
        <nav className="landing-links" aria-label="Primary">
          <a className="landing-link" href="#/docs" onClick={(e) => { e.preventDefault(); onDocs?.(); }}>
            <IconDoc size={15} />
            <span className="landing-link-label">Documentation</span>
          </a>
          <a className="landing-link" href={GITHUB_URL} target="_blank" rel="noreferrer" aria-label="GitHub repository">
            <IconGithub size={15} />
            <span className="landing-link-label">GitHub</span>
          </a>
          <ThemeToggle />
        </nav>
      </header>

      <main className="landing-main">
        <div className="landing-copy">
          <span className="landing-eyebrow">
            <span className="dot live" aria-hidden="true" />
            7 coordinated agents
          </span>
          <h1 className="landing-title">
            Answers you can <em>audit</em>, not just answers you can read.
          </h1>
          <p className="landing-desc">
            MARS decomposes a hard question, dispatches it across a coordinated team of
            specialist agents, verifies every claim against the source that produced it, and
            red-teams its own conclusions before writing a cited report — live, to your console.
          </p>

          <button type="button" className="landing-cta" onClick={onStart}>
            Start research
          </button>
          <span className="landing-cta-note">
            No signup. Bring your own OpenAI-compatible key, or use the default chain.
          </span>

          <div className="landing-metrics">
            <div className="landing-metric">
              <div className="m-val">7</div>
              <div className="m-lab">pipeline agents</div>
            </div>
            <div className="landing-metric">
              <div className="m-val">6</div>
              <div className="m-lab">research modes</div>
            </div>
            <div className="landing-metric">
              <div className="m-val">1</div>
              <div className="m-lab">claim, 1 cited source</div>
            </div>
          </div>
        </div>

        {/* The evidence panel: the real pipeline, then what verification
            actually outputs. This is the product, shown. */}
        <div className="landing-panel" aria-label="How a MARS run is structured">
          <div className="landing-panel-head">
            <div>
              <div className="t">Run structure</div>
              <p className="q">Every question follows this order, and the console streams each step as it happens.</p>
            </div>
          </div>

          <ol className="spine" aria-label="Pipeline execution order">
            {PIPELINE.map((node, i) => (
              <li key={node.name} className="spine-node">
                <span className="spine-idx">{i + 1}</span>
                <span className="spine-body">
                  <span className="spine-title">{node.name}</span>
                  <span className="spine-desc">{node.desc}</span>
                </span>
              </li>
            ))}
          </ol>

          <div className="landing-ledger">
            <div className="eyebrow" style={{ marginBottom: 9 }}>What verification produces</div>
            {LEDGER.map((row) => (
              <div className="landing-ledger-row" key={row.label}>
                <span className={`dot ${row.tone === "good" ? "done" : row.tone === "warn" ? "warn" : "bad"}`} />
                <span>{row.claim}</span>
                <span className={`tag tone-${row.tone}`}>{row.label}</span>
              </div>
            ))}
          </div>
        </div>
      </main>

      <footer className="landing-foot">
        <span>MARS console · v2.1 · Multi-Agent Research System</span>
        <span className="foot-links">
          <a href="#/docs" onClick={(e) => { e.preventDefault(); onDocs?.(); }}>Documentation</a>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer">Source</a>
        </span>
      </footer>
    </div>
  );
}
