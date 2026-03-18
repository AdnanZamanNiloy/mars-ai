import { useMemo, useRef, useState } from "react";
import FinalAnswerCard from "./components/FinalAnswerCard";
import PipelineBar from "./components/PipelineBar";
import SectionCard from "./components/SectionCard";

const EXAMPLES = [
  "What are the most credible small-language-model benchmarks in 2026?",
  "Compare open-source speech-to-text models that run efficiently on CPU.",
  "What are the practical limits of free-tier LLM APIs for research automation?",
];

export default function App() {
  const [query, setQuery] = useState(EXAMPLES[0]);
  const [requestId, setRequestId] = useState("");
  const [progress, setProgress] = useState([]);
  const [plan, setPlan] = useState([]);
  const [loops, setLoops] = useState([]);
  const [findings, setFindings] = useState([]);
  const [searchSnippets, setSearchSnippets] = useState(0);
  const [report, setReport] = useState("");
  const [confidence, setConfidence] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [activeStep, setActiveStep] = useState("planner");
  const activeController = useRef(null);

  const canSubmit = useMemo(() => query.trim().length >= 5 && !running, [query, running]);

  const topFindings = useMemo(() => findings.slice(0, 7), [findings]);

  const finalAnswer = useMemo(() => extractFinalAnswer(report), [report]);

  const abortRun = () => {
    if (activeController.current) {
      activeController.current.abort();
      activeController.current = null;
    }
  };

  const runResearch = async (event) => {
    event.preventDefault();
    if (!canSubmit) return;

    const controller = new AbortController();
    activeController.current = controller;

    setRunning(true);
    setError("");
    setRequestId("");
    setProgress(["Connecting..."]);
    setPlan([]);
    setLoops([]);
    setFindings([]);
    setSearchSnippets(0);
    setReport("");
    setConfidence(null);
    setActiveStep("planner");

    try {
      const response = await fetch("/api/research/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query.trim() }),
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        throw new Error(`Request failed (${response.status})`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      const applyEvent = (evt) => {
        switch (evt.type) {
          case "progress":
            if (evt.request_id) {
              setRequestId(evt.request_id);
            }
            if (evt.message) {
              setProgress((prev) => [...prev, evt.message]);
            }
            break;
          case "plan":
            setPlan(Array.isArray(evt.items) ? evt.items : []);
            setActiveStep("planner");
            break;
          case "search_progress":
            if (typeof evt.snippets === "number") {
              setSearchSnippets(evt.snippets);
            }
            setActiveStep("search");
            break;
          case "critic":
            if (evt.iteration) {
              setLoops((prev) => [...prev, { iteration: evt.iteration, reason: evt.reason || "" }]);
            }
            setActiveStep("critic");
            break;
          case "findings":
            if (Array.isArray(evt.items)) {
              setFindings((prev) => [...prev, ...evt.items]);
            }
            setActiveStep("summarizer");
            break;
          case "final_report":
            setReport(evt.report || "");
            if (typeof evt.confidence === "number") {
              setConfidence(evt.confidence);
            }
            setActiveStep("synthesizer");
            break;
          case "error":
            setError(evt.message || "Unknown stream error");
            break;
          default:
            break;
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          buffer += decoder.decode();
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            applyEvent(JSON.parse(trimmed));
          } catch {
            // Ignore malformed chunks to keep stream resilient.
          }
        }
      }

      const trailing = buffer.trim();
      if (trailing) {
        try {
          applyEvent(JSON.parse(trailing));
        } catch {
          // Ignore malformed trailing chunk.
        }
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setProgress((prev) => [...prev, "Mission aborted by user."]);
      } else {
        setError(err instanceof Error ? err.message : "Unknown stream error");
      }
    } finally {
      activeController.current = null;
      setRunning(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="hero">
        <p className="eyebrow">MARS</p>
        <h1>Multi Agent Research System</h1>
        <p className="sub">Fast, low-resource, citation-grounded research streaming.</p>
      </header>

      <main className="layout-grid">
        <div className="main-column">
          <section className="card input-panel">
            <form onSubmit={runResearch}>
              <label htmlFor="query" className="label-title">Research query</label>
              <textarea
                id="query"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a research question..."
                rows={4}
              />

              <div className="row">
                <button type="submit" className="run-search-btn" disabled={!canSubmit}>
                  Run Search
                </button>
                <button type="button" className="abort-mission-btn" onClick={abortRun} disabled={!running}>
                  Abort Mission
                </button>
              </div>
            </form>

            <div className="chips">
              {EXAMPLES.map((item) => (
                <button key={item} type="button" className="chip" onClick={() => setQuery(item)} disabled={running}>
                  {item}
                </button>
              ))}
            </div>

            {error ? <p className="error">Error: {error}</p> : null}
          </section>

          <PipelineBar activeStep={activeStep} />

          <FinalAnswerCard answer={finalAnswer} confidence={confidence} running={running} />

          <SectionCard title="Findings" rightMeta={`${topFindings.length} shown`}>
            {topFindings.length > 0 ? (
              <div className="findings-grid">
                {topFindings.map((item, idx) => {
                  const domain = extractDomain(item.source || "");
                  const trust = sourceTrustLevel(domain);
                  const rowConfidence = clampPercent((item.confidence ?? confidence ?? 0.62) * 100 - idx * 4);

                  return (
                    <article key={`finding-${idx}`} className="finding-card">
                      <p className="finding-claim">{item.claim}</p>
                      <div className="finding-meta">
                        {item.source ? (
                          <a href={item.source} target="_blank" rel="noopener noreferrer" className="finding-domain finding-link">
                            {domain || "unknown source"}
                          </a>
                        ) : (
                          <span className="finding-domain">{domain || "unknown source"}</span>
                        )}
                        <span className={`source-badge ${trust.className}`}>{trust.label}</span>
                      </div>
                      <div className="confidence-bar" aria-label="confidence">
                        <div className="confidence-bar-fill" style={{ width: `${rowConfidence}%` }} />
                      </div>
                    </article>
                  );
                })}
              </div>
            ) : (
              <p className="empty">No findings extracted yet.</p>
            )}
          </SectionCard>
        </div>

        <aside className="stream-column">
          <SectionCard
            title="Status"
            rightMeta={requestId ? `Request ${requestId.slice(0, 8)}` : null}
          >
            <p className="status-line">
              <span className={`status-dot ${running ? "is-running" : "is-idle"}`} />
              {running ? "Running" : "Idle"}
            </p>
            {searchSnippets > 0 ? <p className="meta">Snippets retrieved: {searchSnippets}</p> : null}
          </SectionCard>

          <SectionCard title="Progress" rightMeta={`${progress.length} updates`}>
            {progress.length > 0 ? (
              <ul className="list-plain">
                {progress.map((item, idx) => (
                  <li key={`progress-${idx}`}>{item}</li>
                ))}
              </ul>
            ) : (
              <p className="empty">No progress events yet.</p>
            )}
          </SectionCard>

          <SectionCard title="Plan" initiallyCollapsed rightMeta={`${plan.length} items`}>
            {plan.length > 0 ? (
              <ul className="list-plain">
                {plan.map((item, idx) => (
                  <li key={`plan-${idx}`}>{item}</li>
                ))}
              </ul>
            ) : (
              <p className="empty">No plan generated yet.</p>
            )}
          </SectionCard>

          <SectionCard title="Critique" initiallyCollapsed rightMeta={`${loops.length} loops`}>
            {loops.length > 0 ? (
              <ul className="list-plain">
                {loops.map((item, idx) => (
                  <li key={`loop-${idx}`}>
                    Iteration {item.iteration}: {item.reason}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="empty">No critique loops yet.</p>
            )}
          </SectionCard>

          {report ? (
            <SectionCard title="Final Report (Raw)" rightMeta={typeof confidence === "number" ? `Confidence ${confidence.toFixed(2)}` : null}>
              {typeof confidence === "number" ? <p className="meta">Confidence: {confidence.toFixed(2)}</p> : null}
              <pre className="report-raw">{report}</pre>
            </SectionCard>
          ) : null}

          {!report && findings.length === 0 && !running ? <p className="empty">No output yet. Submit a query to begin.</p> : null}
        </aside>
      </main>
    </div>
  );
}

function extractFinalAnswer(reportText) {
  if (!reportText) return "";
  const marker = "# Final Answer";
  const evidenceMarker = "# Supporting Evidence";

  const start = reportText.indexOf(marker);
  if (start === -1) return "";
  const bodyStart = start + marker.length;
  const end = reportText.indexOf(evidenceMarker, bodyStart);
  const section = end === -1 ? reportText.slice(bodyStart) : reportText.slice(bodyStart, end);
  return section.trim();
}

function extractDomain(url) {
  if (!url) return "";
  try {
    const parsed = new URL(url);
    return parsed.hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

function sourceTrustLevel(domain) {
  const host = (domain || "").toLowerCase();
  const highTrust = ["wikipedia.org", "arxiv.org"];
  const lowTrust = ["reddit.com", "medium.com", "quora.com"];

  if (highTrust.some((d) => host === d || host.endsWith(`.${d}`))) {
    return { label: "high", className: "source-badge-high" };
  }
  if (lowTrust.some((d) => host === d || host.endsWith(`.${d}`))) {
    return { label: "low", className: "source-badge-low" };
  }
  return { label: "medium", className: "source-badge-medium" };
}

function clampPercent(value) {
  const n = Number(value);
  if (Number.isNaN(n)) return 0;
  return Math.max(8, Math.min(100, Math.round(n)));
}
