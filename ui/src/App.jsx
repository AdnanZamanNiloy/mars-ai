import { useCallback, useEffect, useRef, useState } from "react";
import { fetchTrace, resumeResearch, startResearch } from "./api";
import { MODE_META, loadMissions, removeMission, upsertMission } from "./lib";
import Sidebar, { Planet } from "./components/Sidebar";
import Composer from "./components/Composer";
import { ErrorCard, LiveRunCard, MarsMessageShell, TypingRow, UserMessage } from "./components/Thread";
import AnswerCard, { ReplayAnswerCard } from "./components/AnswerCard";
import ClaimDrawer from "./components/ClaimDrawer";
import IntelligencePanel from "./components/IntelligencePanel";
import MissionsView from "./components/MissionsView";
import EvidenceView from "./components/EvidenceView";
import { IconChevronLeft, IconMenu } from "./components/icons";

const EXAMPLE_QUERIES = [
  "Should Bangladesh invest in nuclear or solar energy?",
  "What are the most credible small-language-model benchmarks in 2026?",
  "Compare open-source speech-to-text models that run efficiently on CPU.",
];

let seq = 1;
const nid = () => `m${Date.now()}-${seq++}`;

function blankRun(query, mode) {
  return {
    tempId: nid(),
    runId: null,
    query,
    mode,
    modeLabel: MODE_META[mode]?.label || mode,
    startedAt: new Date().toISOString(),
    started: true,
    plan: [],
    orchestration: {},
    snippets: 0,
    findings: [],
    verifiedCount: 0,
    critiques: [],
    budget: null,
    decisions: [],
    report: "",
    confidence: null,
    done: false,
    error: "",
    resumable: false,
    resuming: false,
    aborted: false,
  };
}

export default function App() {
  const [view, setView] = useState("workspace");
  const [missions, setMissions] = useState(() => loadMissions());
  const [messages, setMessages] = useState([]);
  const [composer, setComposer] = useState("");
  const [mode, setMode] = useState("standard");
  const [running, setRunning] = useState(false);
  const [selectedFinding, setSelectedFinding] = useState(null);
  const [traceLog, setTraceLog] = useState([]);
  const [intelCollapsed, setIntelCollapsed] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [replaying, setReplaying] = useState(false);

  const controllerRef = useRef(null);
  const threadRef = useRef(null);
  const runRef = useRef(null); // tempId of the in-flight run

  const saveMissions = useCallback((list) => setMissions(list), []);

  useEffect(() => {
    const el = threadRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const patchRun = useCallback((tempId, patch) => {
    setMessages((prev) =>
      prev.map((m) => (m.kind === "run" && m.run.tempId === tempId ? { ...m, run: { ...m.run, ...patch } } : m))
    );
  }, []);

  const pushTrace = useCallback((entry) => {
    setTraceLog((prev) => {
      if (entry.key) {
        const idx = prev.findIndex((t) => t.key === entry.key);
        if (idx !== -1) {
          const next = [...prev];
          next[idx] = { ...next[idx], ...entry };
          return next;
        }
      }
      return [...prev.slice(-60), { at: new Date().toISOString(), kind: "active", ...entry }];
    });
  }, []);

  const handleEvent = useCallback((tempId, evt) => {
    switch (evt.type) {
      case "progress":
        if (evt.request_id) {
          patchRun(tempId, { runId: evt.request_id });
          setMessages((prev) => {
            const msg = prev.find((m) => m.kind === "run" && m.run.tempId === tempId);
            if (msg) {
              saveMissions(upsertMission({
                runId: evt.request_id, query: msg.run.query, mode: msg.run.mode,
                status: msg.run.resuming ? "running" : "running", confidence: null, cost: null,
              }));
            }
            return prev;
          });
        }
        if (evt.message) pushTrace({ text: evt.message, kind: "active" });
        break;
      case "plan":
        patchRun(tempId, {
          plan: Array.isArray(evt.items) ? evt.items : [],
          orchestration: evt.orchestration || {},
        });
        pushTrace({ text: `Strategy created (${(evt.items || []).length} agents)`, kind: "done" });
        break;
      case "search_progress":
        if (typeof evt.snippets === "number") patchRun(tempId, { snippets: evt.snippets });
        pushTrace({ key: "search", text: `Evidence gathered (${evt.snippets ?? 0} sources)`, kind: "active" });
        break;
      case "critic":
        if (evt.iteration) {
          setMessages((prev) => prev.map((m) =>
            m.kind === "run" && m.run.tempId === tempId
              ? { ...m, run: { ...m.run, critiques: [...m.run.critiques, { iteration: evt.iteration, reason: evt.reason || "" }] } }
              : m
          ));
          pushTrace({ text: `Critic pass ${evt.iteration}: ${evt.reason || "reviewed"}`, kind: "done" });
        }
        break;
      case "findings":
        if (Array.isArray(evt.items)) {
          setMessages((prev) => prev.map((m) => {
            if (m.kind !== "run" || m.run.tempId !== tempId) return m;
            const findings = [...m.run.findings, ...evt.items];
            return { ...m, run: { ...m.run, findings, verifiedCount: findings.filter((f) => f.verified === true).length } };
          }));
          pushTrace({ key: "findings", text: "Claims extracted", kind: "active" });
        }
        break;
      case "budget":
        patchRun(tempId, {
          budget: {
            cost: typeof evt.estimated_cost === "number" ? evt.estimated_cost : null,
            limit: typeof evt.limit === "number" ? evt.limit : null,
            calls: typeof evt.llm_calls === "number" ? evt.llm_calls : null,
            overBudget: Boolean(evt.over_budget),
          },
        });
        pushTrace({
          key: "budget",
          text: `Budget $${(evt.estimated_cost ?? 0).toFixed(4)}${evt.limit != null ? ` / $${evt.limit.toFixed(2)}` : ""}`,
          kind: evt.over_budget ? "warn" : "active",
        });
        break;
      case "decisions":
        if (Array.isArray(evt.items)) patchRun(tempId, { decisions: evt.items });
        pushTrace({ text: `${(evt.items || []).length} options evaluated`, kind: "done" });
        break;
      case "final_report":
        setMessages((prev) => prev.map((m) => {
          if (m.kind !== "run" || m.run.tempId !== tempId) return m;
          const run = {
            ...m.run,
            report: evt.report || "",
            confidence: typeof evt.confidence === "number" ? evt.confidence : null,
            done: true,
            resuming: false,
          };
          if (run.runId) {
            saveMissions(upsertMission({
              runId: run.runId, query: run.query, mode: run.mode,
              status: "completed", confidence: run.confidence, cost: run.budget?.cost ?? null,
            }));
          }
          return { ...m, run };
        }));
        pushTrace({ key: "findings", text: "Claims extracted", kind: "done" });
        pushTrace({ text: "Final report delivered", kind: "done" });
        break;
      case "error": {
        const message = evt.message || "Unknown stream error";
        const noKey = /LLM key|GROQ_API_KEY|HUGGINGFACE/i.test(message);
        const resumable = !noKey && /timed out|failed/i.test(message);
        setMessages((prev) => prev.map((m) => {
          if (m.kind !== "run" || m.run.tempId !== tempId) return m;
          const run = { ...m.run, error: message, resuming: false, resumable };
          if (run.runId) {
            saveMissions(upsertMission({
              runId: run.runId, query: run.query, mode: run.mode,
              status: resumable ? "resumable" : "failed", confidence: null, cost: run.budget?.cost ?? null,
            }));
          }
          return { ...m, run };
        }));
        pushTrace({ text: message.slice(0, 90), kind: "warn" });
        break;
      }
      default:
        break;
    }
  }, [patchRun, pushTrace, saveMissions]);

  const launch = useCallback(async (queryText, { resumeRun = null } = {}) => {
    if (running) return;
    const controller = new AbortController();
    controllerRef.current = controller;

    let tempId;
    if (resumeRun) {
      tempId = resumeRun.tempId;
      patchRun(tempId, {
        error: "", resumable: false, resuming: true, done: false,
        findings: [], verifiedCount: 0, budget: null,
      });
      pushTrace({ text: `Resuming run ${resumeRun.runId.slice(0, 8)} from checkpoint`, kind: "active" });
    } else {
      const run = blankRun(queryText, mode);
      tempId = run.tempId;
      const at = new Date().toISOString();
      setMessages((prev) => [
        ...prev,
        { id: nid(), kind: "user", text: queryText, at },
        { id: nid(), kind: "run", run, at },
      ]);
      setTraceLog([]);
      setSelectedFinding(null);
      setView("workspace");
    }

    runRef.current = tempId;
    setRunning(true);
    try {
      if (resumeRun) {
        await resumeResearch({ runId: resumeRun.runId, signal: controller.signal, onEvent: (e) => handleEvent(tempId, e) });
      } else {
        await startResearch({ query: queryText, mode, signal: controller.signal, onEvent: (e) => handleEvent(tempId, e) });
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        setMessages((prev) => prev.map((m) => {
          if (m.kind !== "run" || m.run.tempId !== tempId) return m;
          const run = { ...m.run, aborted: true, resuming: false };
          if (run.runId) {
            saveMissions(upsertMission({
              runId: run.runId, query: run.query, mode: run.mode,
              status: "aborted", confidence: null, cost: run.budget?.cost ?? null,
            }));
          }
          return { ...m, run };
        }));
        pushTrace({ text: "Mission aborted by user", kind: "warn" });
      } else {
        handleEvent(tempId, { type: "error", message: err instanceof Error ? err.message : "Unknown stream error" });
      }
    } finally {
      controllerRef.current = null;
      runRef.current = null;
      setRunning(false);
    }
  }, [running, mode, patchRun, pushTrace, handleEvent, saveMissions]);

  const submitQuery = useCallback((text) => {
    const v = text.trim();
    if (v.length < 5 || running) return;
    setComposer("");
    launch(v);
  }, [launch, running]);

  const abortRun = useCallback(() => {
    controllerRef.current?.abort();
  }, []);

  const resumeRun = useCallback((run) => {
    if (running || !run.runId) return;
    launch("", { resumeRun: run });
  }, [launch, running]);

  const openReplay = useCallback(async (runId) => {
    if (replaying) return;
    setReplaying(true);
    try {
      const trace = await fetchTrace(runId);
      const mission = loadMissions().find((m) => m.runId === runId);
      setMessages((prev) => [...prev, { id: nid(), kind: "replay", runId, trace, query: mission?.query || trace.query || "Replay", at: new Date().toISOString() }]);
      const events = (trace.events || []).map((e) => ({
        at: e.ended_at || e.started_at || new Date().toISOString(),
        kind: e.event_type === "end" ? "done" : "active",
        text: `${e.node} · ${e.event_type}`,
      }));
      setTraceLog(events.length > 0 ? events : [{ at: new Date().toISOString(), kind: "done", text: "Trace loaded — no node events recorded" }]);
      setView("workspace");
    } catch (err) {
      setMessages((prev) => [...prev, {
        id: nid(), kind: "notice", text: `Could not load replay: ${err instanceof Error ? err.message : "unknown error"}`,
        at: new Date().toISOString(),
      }]);
    } finally {
      setReplaying(false);
    }
  }, [replaying]);

  const activeRun = [...messages].reverse().find((m) => m.kind === "run")?.run || null;

  return (
    <div className="shell">
      <Sidebar
        view={view}
        onNavigate={setView}
        missions={missions}
        activeRunId={activeRun?.runId}
        onOpenMission={openReplay}
        onNew={() => { setView("workspace"); setComposer(""); }}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      {sidebarOpen ? <button className="scrim" onClick={() => setSidebarOpen(false)} aria-label="Close menu" /> : null}

      <div className="workspace-wrap">
        <main className="workspace">
          <header className="topbar">
            <button className="icon-btn menu-btn" onClick={() => setSidebarOpen(true)} aria-label="Open menu">
              <IconMenu size={17} />
            </button>
            <div>
              <h1>{view === "missions" ? "Missions" : view === "evidence" ? "Evidence" : "Command Center"}</h1>
              <div className="crumb">
                {activeRun ? `${activeRun.query.slice(0, 64)}${activeRun.query.length > 64 ? "…" : ""}` : "Multi-Agent Research System"}
              </div>
            </div>
            <span className="spacer" />
            {activeRun?.runId ? <span className="tag tone-muted">run {activeRun.runId.slice(0, 8)}</span> : null}
            {running ? <span className="tag tone-blue"><span className="dot live" /> live</span> : null}
          </header>

          <div className="thread" ref={threadRef}>
            <div className="thread-inner">
              {view === "missions" ? (
                <MissionsView
                  missions={missions}
                  onOpen={openReplay}
                  onRemove={(runId) => saveMissions(removeMission(runId))}
                  onNew={() => setView("workspace")}
                />
              ) : view === "evidence" ? (
                <EvidenceView messages={messages} onInspect={setSelectedFinding} />
              ) : messages.length === 0 ? (
                <WelcomeHero onPick={setComposer} />
              ) : (
                messages.map((m) => <ThreadMessage
                  key={m.id}
                  message={m}
                  running={running}
                  selectedFinding={selectedFinding}
                  onSelectFinding={setSelectedFinding}
                  onResume={resumeRun}
                />)
              )}
            </div>
          </div>

          {view === "workspace" ? (
            <div className="composer-zone">
              <div className="composer-inner">
                <Composer
                  value={composer}
                  onChange={setComposer}
                  onSubmit={submitQuery}
                  running={running}
                  onAbort={abortRun}
                  mode={mode}
                  onModeChange={setMode}
                  placeholder={messages.length === 0 ? "Ask a research question… (Enter to send)" : "Ask a follow-up or challenge the conclusion…"}
                />
              </div>
            </div>
          ) : null}
        </main>

        {view === "workspace" ? (
          <>
            <IntelligencePanel
              run={activeRun}
              traceLog={traceLog}
              collapsed={intelCollapsed}
              onCollapse={() => setIntelCollapsed(true)}
              onExpand={() => setIntelCollapsed(false)}
              onResume={() => activeRun && resumeRun(activeRun)}
              onReplay={openReplay}
              replaying={replaying}
            />
            {intelCollapsed ? (
              <button className="icon-btn intel-expand" onClick={() => setIntelCollapsed(false)} title="Expand panel" aria-label="Expand panel">
                <IconChevronLeft size={15} />
              </button>
            ) : null}
          </>
        ) : null}
      </div>

      <ClaimDrawer finding={selectedFinding} onClose={() => setSelectedFinding(null)} />
    </div>
  );
}

function ThreadMessage({ message, running, selectedFinding, onSelectFinding, onResume }) {
  if (message.kind === "user") {
    return <UserMessage text={message.text} time={fmtTime(message.at)} />;
  }
  if (message.kind === "run") {
    const { run } = message;
    return (
      <MarsMessageShell time={fmtTime(message.at)}>
        {!run.done && !run.error && !run.aborted ? <LiveRunCard run={run} /> : null}
        {run.aborted && !run.done ? (
          <div className="error-box" style={{ borderColor: "var(--line)", background: "var(--card)" }}>
            Mission aborted by user before completion.
          </div>
        ) : null}
        {run.error ? (
          <ErrorCard message={run.error} resumable={run.resumable} resuming={run.resuming} onResume={() => onResume(run)} />
        ) : null}
        {run.done && run.report ? (
          <AnswerCard run={run} selectedFinding={selectedFinding} onSelectFinding={onSelectFinding} />
        ) : null}
        {run.done && !run.report && !run.error ? (
          <div className="error-box">The run finished without producing a report.</div>
        ) : null}
        {running && !run.done && !run.error ? (
          <div style={{ marginTop: 14 }}><TypingRow /></div>
        ) : null}
      </MarsMessageShell>
    );
  }
  if (message.kind === "replay") {
    const { trace } = message;
    return (
      <MarsMessageShell time={fmtTime(message.at)}>
        <div className="replay-banner">
          <span className="tag tone-blue">replay</span>
          <span>Read-only record of “{message.query}” · status: {trace.status}{(trace.plan || []).length ? ` · ${(trace.plan || []).length} planned questions` : ""}</span>
        </div>
        {trace.final_report ? (
          <ReplayAnswerCard trace={trace} />
        ) : (
          <div className="error-box">This run has no final report recorded.</div>
        )}
      </MarsMessageShell>
    );
  }
  if (message.kind === "notice") {
    return (
      <MarsMessageShell time={fmtTime(message.at)}>
        <div className="error-box">{message.text}</div>
      </MarsMessageShell>
    );
  }
  return null;
}

function WelcomeHero({ onPick }) {
  return (
    <div className="hero-card anim-rise">
      <Planet size={88} ring />
      <h2>What should MARS <span className="accent">investigate</span>?</h2>
      <p>
        A team of research agents plans the inquiry, gathers sources, verifies claims,
        challenges conclusions and synthesizes a cited report — streaming live to this console.
      </p>
      <div className="hero-examples">
        {EXAMPLE_QUERIES.map((q) => (
          <button key={q} type="button" onClick={() => onPick(q)}>{q}</button>
        ))}
      </div>
    </div>
  );
}

function fmtTime(iso) {
  try {
    return new Date(iso).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
  } catch {
    return "";
  }
}
