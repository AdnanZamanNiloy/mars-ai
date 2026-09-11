/* MARS landing — full-screen cinematic entry. Pure presentational:
 * no data fetching, no backend calls. onStart enters the workspace. */

const GITHUB_URL = "https://github.com/AdnanZamanNiloy/mars-ai";
const DOCS_URL = "https://github.com/AdnanZamanNiloy/mars-ai#readme";

export default function Landing({ onStart }) {
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
        <span className="landing-brand">MARS</span>
        <nav className="landing-links">
          <a href={DOCS_URL} target="_blank" rel="noreferrer">Docs</a>
          <a href={GITHUB_URL} target="_blank" rel="noreferrer">GitHub</a>
        </nav>
      </header>

      <main className="landing-hero">
        <div className="landing-planet" aria-hidden="true" />
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
        <span>Planner · Search · Verifier · Critic · Synthesizer</span>
      </footer>
    </div>
  );
}
