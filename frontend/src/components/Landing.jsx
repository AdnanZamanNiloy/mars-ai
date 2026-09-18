/* MARS landing — full-screen cinematic entry. Pure presentational:
 * no data fetching, no backend calls. onStart enters the workspace. */

import { Planet } from "./Sidebar";
import ThemeToggle from "./ThemeToggle";

const GITHUB_URL = "https://github.com/AdnanZamanNiloy/mars-ai";

export default function Landing({ onStart, onDocs }) {
  return (
    <div className="landing">
      <div className="landing-atmosphere" aria-hidden="true">
        <span className="orb orb-a" />
        <span className="orb orb-b" />
        <span className="orb orb-c" />
        <span className="star s1" />
        <span className="star s2" />
        <span className="star s3" />
        <span className="star s4" />
        <span className="star s5" />
        <span className="star s6" />
        <span className="star s7" />
        <span className="star s8" />
      </div>

      <header className="landing-nav">
        <a
          className="landing-brand brand"
          href="#/"
          onClick={(e) => { e.preventDefault(); }}
          aria-label="MARS — Multi-Agent Research System, home"
        >
          <Planet size={44} />
          <div>
            <div className="brand-name">MARS</div>
            <div className="brand-sub">Multi-Agent Research System</div>
          </div>
        </a>
        <nav className="landing-links" aria-label="Primary">
          <a className="landing-link" href="#/docs" onClick={(e) => { e.preventDefault(); onDocs?.(); }}>
            <span className="link-dot" aria-hidden="true" /> Docs
          </a>
          <a className="landing-link" href={GITHUB_URL} target="_blank" rel="noreferrer">
            <span className="link-dot" aria-hidden="true" /> GitHub
          </a>
          <ThemeToggle className="landing-theme-toggle" />
        </nav>
      </header>

      <main className="landing-hero">
        <div className="landing-planet" aria-hidden="true">
          <span className="planet-ring" />
          <span className="planet-glow" />
        </div>
        <h1 className="landing-title">MARS</h1>
        <p className="landing-subtitle">Multi-Agent Research System</p>
        <p className="landing-tagline">Complex questions. Coordinated intelligence.</p>
        <p className="landing-desc">
          MARS decomposes hard research questions and dispatches them across a
          coordinated team of specialist agents: planning, searching,
          verifying, and challenging every claim before synthesizing a cited,
          decision ready report.
        </p>
        <button type="button" className="landing-cta" onClick={onStart}>
          Start Research
        </button>
      </main>

      <footer className="landing-foot">
        <span className="foot-agents">
          <span>Planner</span>
          <span>Search</span>
          <span>Verifier</span>
          <span>Critic</span>
          <span>Synthesizer</span>
        </span>
      </footer>
    </div>
  );
}
