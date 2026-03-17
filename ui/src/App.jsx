import { useMemo, useState } from "react";

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

  const canSubmit = useMemo(() => query.trim().length >= 5 && !running, [query, running]);

  const runResearch = async (event) => {
    event.preventDefault();
    if (!canSubmit) return;

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

    try {
      const response = await fetch("/api/research/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query.trim() }),
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
            break;
          case "search_progress":
            if (typeof evt.snippets === "number") {
              setSearchSnippets(evt.snippets);
            }
            break;
          case "critic":
            if (evt.iteration) {
              setLoops((prev) => [...prev, { iteration: evt.iteration, reason: evt.reason || "" }]);
            }
            break;
          case "findings":
            if (Array.isArray(evt.items)) {
              setFindings((prev) => [...prev, ...evt.items]);
            }
            break;
          case "final_report":
            setReport(evt.report || "");
            if (typeof evt.confidence === "number") {
              setConfidence(evt.confidence);
            }
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
      setError(err instanceof Error ? err.message : "Unknown stream error");
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="page">
      <header className="hero">
        <p className="eyebrow">MARS</p>
        <h1>Multi Agent Research System</h1>
        <p className="sub">Fast, low-resource, citation-grounded research streaming.</p>
      </header>

      <main className="grid">
        <section className="panel input-panel">
          <form onSubmit={runResearch}>
            <label htmlFor="query">Research query</label>
            <textarea
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a research question..."
              rows={6}
            />

            <div className="row">
              <button type="submit" disabled={!canSubmit}>
                {running ? "Running..." : "Run Research"}
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

        <section className="panel output-panel">
          <div className="stream-title">Research Stream</div>

          <div className="event-block">
            <h3>Status</h3>
            <p>{running ? "Running" : "Idle"}</p>
            {requestId ? <p className="meta">Request ID: {requestId}</p> : null}
            {searchSnippets > 0 ? <p className="meta">Snippets retrieved: {searchSnippets}</p> : null}
          </div>

          {progress.length > 0 ? (
            <div className="event-block">
              <h3>Progress</h3>
              <ul>
                {progress.map((item, idx) => (
                  <li key={`progress-${idx}`}>{item}</li>
                ))}
              </ul>
            </div>
          ) : null}

          {plan.length > 0 ? (
            <div className="event-block">
              <h3>Plan</h3>
              <ul>
                {plan.map((item, idx) => (
                  <li key={`plan-${idx}`}>{item}</li>
                ))}
              </ul>
            </div>
          ) : null}

          {loops.length > 0 ? (
            <div className="event-block">
              <h3>Critic Loops</h3>
              <ul>
                {loops.map((item, idx) => (
                  <li key={`loop-${idx}`}>
                    Iteration {item.iteration}: {item.reason}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          {findings.length > 0 ? (
            <div className="event-block">
              <h3>Findings</h3>
              <ul>
                {findings.map((item, idx) => (
                  <li key={`finding-${idx}`}>
                    {item.claim}
                    {item.source ? <span className="source"> ({item.source})</span> : null}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          {report ? (
            <div className="event-block report-block">
              <h3>Final Report</h3>
              {typeof confidence === "number" ? <p className="meta">Confidence: {confidence.toFixed(2)}</p> : null}
              <pre>{report}</pre>
            </div>
          ) : null}

          {!report && findings.length === 0 && !running ? <p className="empty">No output yet. Submit a query to begin.</p> : null}
        </section>
      </main>
    </div>
  );
}
