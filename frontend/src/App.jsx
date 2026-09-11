import { useCallback, useEffect, useRef, useState } from "react";
import { fetchTrace, resumeResearch, startResearch } from "./api";
import { MODE_META, loadKnowledge, loadMissions, parseReport, removeKnowledgeItem, removeMission, saveKnowledgeItem, updateMission, upsertMission } from "./lib";
import Sidebar, { Planet } from "./components/Sidebar";
import Composer from "./components/Composer";
import { ErrorCard, MarsMessageShell, ThinkingSteps, TypingRow, UserMessage } from "./components/Thread";
import AnswerCard, { ReplayAnswerCard } from "./components/AnswerCard";
import ClaimDrawer from "./components/ClaimDrawer";
import ErrorBoundary from "./components/ErrorBoundary";
import IntelligencePanel from "./components/IntelligencePanel";
import MissionsView from "./components/MissionsView";
import EvidenceView from "./components/EvidenceView";
import KnowledgeView from "./components/KnowledgeView";
import AgentsView from "./components/AgentsView";
import ProvidersView from "./components/ProvidersView";
import Landing from "./components/Landing";
import { IconChevronDown, IconChevronLeft, IconDoc, IconFolder, IconLayers, IconMenu, IconPencil, IconPin, IconTrash } from "./components/icons";

let seq = 1;
const nid = () => `m${Date.now()}-${seq++}`;

const VALID_VIEWS = ["landing", "workspace", "missions", "evidence", "knowledge", "agents", "providers"];

const LIBRARY = [
  { id: "evidence", label: "Evidence", icon: IconLayers },
  { id: "knowledge", label: "Knowledge", icon: IconDoc },
];

/* Library dropdown in the title bar: the research library pages live here
 * instead of the sidebar. Opens on click, highlights the open page. */
function LibraryMenu({ view, onNavigate }) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
    };
    const onKey = (e) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open ]);

  const active = LIBRARY.some((item) => item.id === view);
  return (
    <div className="title-menu-wrap library-menu-wrap" ref={wrapRef}>
      <button
        className={`library-btn${active ? " active" : ""}`}
        onClick={() => setOpen((o) => !o)}
        aria-label="Library"
        aria-expanded={open}
        aria-haspopup="menu"
      >
        <IconFolder size={15} /> Library
        <IconChevronDown size={14} className={`nav-chev${open ? " open" : ""}`} />
      </button>
      {open ? (
        <div className="title-menu library-menu" role="menu">
          {LIBRARY.map((item) => (
            <button
              key={item.id}
              className={`title-menu-item${view === item.id ? " active" : ""}`}
              onClick={() => { onNavigate(item.id); setOpen(false); }}
            >
              <item.icon size={15} /> {item.label}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function viewFromHash() {
  try {
    const h = window.location.hash.replace(/^#\/?/, "");
    return VALID_VIEWS.includes(h) ? h : "landing";
  } catch {
    return "landing";
  }
}

/* Reconstruct a panel-shaped run from a persisted session trace, so the
 * intelligence panel has real numbers on replay screens. Read-only — the
 * trace already carries everything; nothing is re-fetched. */
function replayPanelRun(trace, query, runId) {
  const claims = Array.isArray(trace?.claims) ? trace.claims : [];
  const reviews = Array.isArray(trace?.critic_reviews) ? trace.critic_reviews : [];
  const findings = claims.map((c) => ({
    claim: c.claim,
    source: c.source_url,
    verified: c.verified === 1 || c.verified === true,
    confidence: c.confidence,
    agent: c.agent || "",
    challenged: c.challenged === 1 || c.challenged === true,
  }));
  const lastReview = reviews.length ? reviews[reviews.length - 1] : null;
  const completed = trace?.status === "completed";
  return {
    runId,
    query,
    replay: true,
    plan: (trace?.plan || []).map((t) => t.question).filter(Boolean),
    snippets: (trace?.sources || []).length,
    findings,
    verifiedCount: findings.filter((f) => f.verified).length,
    critiques: reviews.map((r) => ({ iteration: r.iteration, reason: r.reason || "" })),
    breakdown: lastReview?.breakdown || null,
    confidence: typeof trace?.confidence === "number" ? trace.confidence : null,
    done: completed,
    error: completed ? "" : `Run ${trace?.status || "unknown"}`,
    resumable: false,
    resuming: false,
  };
}

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
    breakdown: null,
    decisions: [],
    report: "",
    confidence: null,
    degraded: [],
    answerSupport: null,
    done: false,
    error: "",
    resumable: false,
    resuming: false,
    aborted: false,
  };
}

export default function App() {
  const [view, setViewState] = useState(viewFromHash);
  const viewRef = useRef(viewFromHash());

  /* Browser back/forward support: every view is a real history entry, so
   * the arrows move between landing and the console views. */
  const go = useCallback((v) => {
    const next = VALID_VIEWS.includes(v) ? v : "workspace";
    if (viewRef.current === next) return;
    viewRef.current = next;
    try {
      window.history.pushState({ view: next }, "", next === "landing" ? "#/" : `#/${next}`);
    } catch {
      /* non-browser/test environment — state still updates */
    }
    setViewState(next);
  }, []);

  useEffect(() => {
    const onPop = (e) => {
      const v = (e.state && e.state.view) || viewFromHash();
      const next = VALID_VIEWS.includes(v) ? v : "landing";
      viewRef.current = next;
      setViewState(next);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);
  const [missions, setMissions] = useState(() => loadMissions());
  const [knowledge, setKnowledge] = useState(() => loadKnowledge());
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
  const parentRef = useRef(null); // parent runId for challenge runs

  const saveMissions = useCallback((list) => setMissions(list), []);

  useEffect(() => {
    const el = threadRef.current;
    if (!el) return;
    // Follow the stream only while the user is already near the bottom;
    // yanking the view on every event fights anyone scrolling back up.
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 160;
    if (nearBottom) el.scrollTop = el.scrollHeight;
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
                parentRunId: parentRef.current,
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
              ? { ...m, run: { ...m.run,
                  critiques: [...m.run.critiques, { iteration: evt.iteration, reason: evt.reason || "" }],
                  breakdown: evt.breakdown && typeof evt.breakdown === "object" ? evt.breakdown : m.run.breakdown,
                } }
              : m
          ));
          pushTrace({ text: `Critic pass ${evt.iteration}: ${evt.reason || "reviewed"}`, kind: "done" });
        }
        break;
      case "findings":
        if (Array.isArray(evt.items)) {
          setMessages((prev) => prev.map((m) => {
            if (m.kind !== "run" || m.run.tempId !== tempId) return m;
            // verified_update re-emits annotated facts (same length): upsert
            // by claim text instead of appending duplicates.
            let findings;
            if (evt.verified_update) {
              const byClaim = new Map(m.run.findings.map((f) => [f.claim, f]));
              for (const item of evt.items) byClaim.set(item.claim, item);
              findings = [...byClaim.values()];
            } else {
              // Key on claim+source: two distinct claims can share text, and
              // empty-claim items must not collapse into one "" key.
              const byKey = new Map(m.run.findings.map((f) => [`${f.claim}||${f.source}`, f]));
              for (const item of evt.items) byKey.set(`${item.claim}||${item.source}`, item);
              findings = [...byKey.values()];
            }
            return { ...m, run: { ...m.run, findings,
              verifiedCount: findings.filter((f) => f.verified === true).length } };
          }));
          pushTrace({ key: "findings", text: "Claims extracted", kind: "active" });
        }
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
            degraded: Array.isArray(evt.degraded) ? evt.degraded : [],
            answerSupport: typeof evt.answer_support === "number" ? evt.answer_support : null,
            done: true,
            resuming: false,
          };
          if (run.runId) {
            saveMissions(upsertMission({
              runId: run.runId, query: run.query, mode: run.mode,
              status: "completed", confidence: run.confidence, cost: null,
              degraded: run.degraded,
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
              status: resumable ? "resumable" : "failed", confidence: null, cost: null,
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

  const launch = useCallback(async (queryText, { resumeRun = null, parentRunId = null } = {}) => {
    if (running) return;
    const controller = new AbortController();
    controllerRef.current = controller;
    parentRef.current = parentRunId;

    let tempId;
    if (resumeRun) {
      tempId = resumeRun.tempId;
      patchRun(tempId, {
        error: "", resumable: false, resuming: true, done: false,
        findings: [], verifiedCount: 0, degraded: [],
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
      go("workspace");
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
              status: "aborted", confidence: null, cost: null,
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
      parentRef.current = null;
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

  const startNew = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    runRef.current = null;
    setMessages((prev) => {
      const active = [...prev].reverse().find((m) => m.kind === "run");
      const run = active?.run;
      if (run?.runId && !run.done && !run.error) {
        saveMissions(upsertMission({
          runId: run.runId, query: run.query, mode: run.mode,
          status: "aborted", confidence: null, cost: null,
        }));
      }
      return [];
    });
    setTraceLog([]);
    setSelectedFinding(null);
    setComposer("");
    setRunning(false);
    go("workspace");
  }, []);

  const resumeRun = useCallback((run) => {
    if (running || !run.runId) return;
    launch("", { resumeRun: run });
  }, [launch, running]);

  const openReplay = useCallback(async (runId) => {
    if (replaying || running) return;
    setReplaying(true);
    try {
      const trace = await fetchTrace(runId);
      const mission = loadMissions().find((m) => m.runId === runId);
      // Selecting a session REPLACES the thread: only that session's record
      // is shown, never a mix of every session in this browser tab.
      setMessages([{
        id: nid(), kind: "replay", runId, trace,
        query: mission?.query || trace.query || "Replay", at: new Date().toISOString(),
      }]);
      const events = (trace.events || []).map((e) => ({
        at: e.ended_at || e.started_at || new Date().toISOString(),
        kind: e.event_type === "end" ? "done" : "active",
        text: `${e.node} · ${e.event_type}`,
      }));
      setTraceLog(events.length > 0 ? events : [{ at: new Date().toISOString(), kind: "done", text: "Trace loaded — no node events recorded" }]);
      setSelectedFinding(null);
      go("workspace");
    } catch (err) {
      setMessages([{
        id: nid(), kind: "notice", text: `Could not load replay: ${err instanceof Error ? err.message : "unknown error"}`,
        at: new Date().toISOString(),
      }]);
    } finally {
      setReplaying(false);
    }
  }, [replaying, running]);

  const activeRun = [...messages].reverse().find((m) => m.kind === "run")?.run || null;

  // When the thread shows a replayed session, the intelligence panel reads
  // a run-shaped view reconstructed from that session's persisted trace —
  // otherwise it renders blank ("No active mission") on replay screens.
  // Declared up here because the page title below also needs lastMessage.
  const lastMessage = messages.length ? messages[messages.length - 1] : null;
  const panelRun = lastMessage?.kind === "replay"
    ? replayPanelRun(lastMessage.trace, lastMessage.query, lastMessage.runId)
    : activeRun;

  /* Fixed page title: the current research question on the workspace view,
   * plain view names elsewhere. The menu acts on the displayed mission. */
  const VIEW_TITLES = { missions: "Missions", evidence: "Evidence", knowledge: "Knowledge", agents: "Agents", providers: "Providers" };
  const titleMessage = [...messages].reverse().find((m) =>
    (m.kind === "run" && m.run?.query) ||
    (m.kind === "replay" && m.query) ||
    (m.kind === "user" && m.text));
  const displayedRunId = activeRun?.runId
    || (lastMessage?.kind === "replay" ? lastMessage.runId : null)
    || null;
  const activeMission = displayedRunId
    ? (missions.find((m) => m.runId === displayedRunId) || null)
    : null;
  const pageTitle = view === "workspace"
    ? (activeMission
      ? (activeMission.title || activeMission.query || "New Research")
      : (!titleMessage ? "New Research"
        : titleMessage.kind === "user" ? titleMessage.text
        : titleMessage.kind === "run" ? titleMessage.run.query
        : titleMessage.query))
    : (VIEW_TITLES[view] || "Command Center");

  const renameMission = useCallback((runId, title) => {
    saveMissions(updateMission(runId, { title }));
  }, [saveMissions]);
  const toggleMissionPin = useCallback((runId) => {
    saveMissions(updateMission(runId, { pinned: !(missions.find((m) => m.runId === runId)?.pinned) }));
  }, [missions, saveMissions]);
  const deleteMission = useCallback((runId) => {
    if (!window.confirm("Delete this research session? This cannot be undone.")) return;
    saveMissions(removeMission(runId));
    if (displayedRunId === runId) startNew();
  }, [displayedRunId, saveMissions, startNew]);

  return (
    view === "landing" ? (
      <Landing onStart={() => go("workspace")} />
    ) : (
    <div className="shell">
      <ErrorBoundary>
      <Sidebar
        view={view}
        onNavigate={go}
        missions={missions}
        activeRunId={panelRun?.runId}
        onOpenMission={openReplay}
        onNew={startNew}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      {sidebarOpen ? <button className="scrim" onClick={() => setSidebarOpen(false)} aria-label="Close menu" /> : null}
      </ErrorBoundary>

      <div className="workspace-wrap">
        <ErrorBoundary>
        <main className="workspace">
          <PageTitle
            title={pageTitle}
            mission={view === "workspace" ? activeMission : null}
            view={view}
            onNavigate={go}
            showLibrary={view === "workspace" && messages.some((m) => m.kind === "run" || m.query || m.run?.query)}
            onRename={renameMission}
            onTogglePin={toggleMissionPin}
            onDelete={deleteMission}
          />
          <button className="icon-btn menu-btn" onClick={() => setSidebarOpen(true)} aria-label="Open menu">
            <IconMenu size={17} />
          </button>
          <div
            className={`thread${view === "workspace" && messages.length === 0 ? " thread-center" : ""}`}
            ref={threadRef}
          >
            <div className="thread-inner">
              {view === "missions" ? (
                <MissionsView
                  missions={missions}
                  onOpen={openReplay}
                  onRemove={(runId) => saveMissions(removeMission(runId))}
                  onNew={startNew}
                />
              ) : view === "evidence" ? (
                <EvidenceView messages={messages} onInspect={setSelectedFinding} />
              ) : view === "knowledge" ? (
                <KnowledgeView
                  items={knowledge}
                  onRemove={(item) => setKnowledge(removeKnowledgeItem(item))}
                  onInspect={setSelectedFinding}
                />
              ) : view === "agents" ? (
                <AgentsView />
              ) : view === "providers" ? (
                <ProvidersView />
              ) : messages.length === 0 ? (
                <WelcomeHero
                  composer={
                    <Composer
                      value={composer}
                      onChange={setComposer}
                      onSubmit={submitQuery}
                      running={running}
                      onAbort={abortRun}
                      mode={mode}
                      onModeChange={setMode}
                      placeholder="Ask a research question… (Enter to send)"
                    />
                  }
                />
              ) : (
                messages.map((m) => <ThreadMessage
                  key={m.id}
                  message={m}
                  running={running}
                  onResume={resumeRun}
                  onRegenerate={submitQuery}
                  steps={traceLog}
                />)
              )}
            </div>
          </div>

          {view === "workspace" && messages.length > 0 ? (
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
                  placeholder="Ask a follow-up question…"
                />
              </div>
            </div>
          ) : null}
        </main>
        </ErrorBoundary>

        {view === "workspace" ? (
          <ErrorBoundary>
            <IntelligencePanel
              run={panelRun}
              collapsed={intelCollapsed}
              onCollapse={() => setIntelCollapsed(true)}
              onResume={() => activeRun && resumeRun(activeRun)}
            />
            {intelCollapsed ? (
              <button className="icon-btn intel-expand" onClick={() => setIntelCollapsed(false)} title="Expand panel" aria-label="Expand panel">
                <IconChevronLeft size={15} />
              </button>
            ) : null}
          </ErrorBoundary>
        ) : null}
      </div>

      <ClaimDrawer
        finding={selectedFinding}
        onClose={() => setSelectedFinding(null)}
        onSave={(finding) => setKnowledge(saveKnowledgeItem(finding))}
        saved={selectedFinding ? knowledge.some(
          (k) => `${k.claim || ""}||${k.source || ""}` ===
                 `${selectedFinding.claim || ""}||${selectedFinding.source || ""}`) : false}
      />
    </div>
    )
  );
}

function ThreadMessage({ message, running, onResume, onRegenerate, steps }) {
  if (message.kind === "user") {
    return <UserMessage text={message.text} />;
  }
  if (message.kind === "run") {
    const { run } = message;
    return (
      <MarsMessageShell
        id={message.id}
        text={run.report || ""}
        canRegenerate={run.done && !running && !run.error && run.query.length > 0}
        onRegenerate={() => onRegenerate(run.query)}
      >
        <ThinkingSteps steps={steps} />
        {run.aborted && !run.done ? (
          <div className="error-box" style={{ borderColor: "var(--line)", background: "var(--card)" }}>
            Mission aborted by user before completion.
          </div>
        ) : null}
        {run.error ? (
          <ErrorCard message={run.error} resumable={run.resumable} resuming={run.resuming} onResume={() => onResume(run)} />
        ) : null}
        {run.done && run.report ? (
          <AnswerCard run={run} />
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
    const report = trace.final_report?.report_markdown || "";
    return (
      <>
      <UserMessage text={message.query} />
      <MarsMessageShell
        id={message.id}
        text={report}
        canRegenerate={!running && message.query.length > 0}
        onRegenerate={() => onRegenerate(message.query)}
      >
        <ThinkingSteps steps={steps} />
        {trace.final_report ? (
          <ReplayAnswerCard trace={trace} />
        ) : (
          <div className="error-box">This run has no final report recorded.</div>
        )}
      </MarsMessageShell>
      </>
    );
  }
  if (message.kind === "notice") {
    return (
      <MarsMessageShell id={message.id} text={message.text}>
        <div className="error-box">{message.text}</div>
      </MarsMessageShell>
    );
  }
  return null;
}

function PageTitle({ title, mission, view, onNavigate, showLibrary, onRename, onTogglePin, onDelete }) {
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const wrapRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) {
        setOpen(false);
      }
    };
    const onKey = (e) => {
      if (e.key === "Escape") { setOpen(false); }
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open ]);

  const close = () => { setOpen(false); setEditing(false); };

  const commitRename = () => {
    const clean = draft.trim().slice(0, 80);
    if (clean && mission) onRename(mission.runId, clean);
    setEditing(false);
  };

  if (editing && mission) {
    return (
      <div className="page-title-bar">
        <input
          className="title-rename-input"
          autoFocus
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commitRename}
          onKeyDown={(e) => {
            if (e.key === "Enter") commitRename();
            if (e.key === "Escape") setEditing(false);
          }}
          aria-label="Rename research session"
        />
      </div>
    );
  }

  return (
    <div className="page-title-bar">
      <span className="page-title-text" title={title}>{title}</span>
      {mission ? (
        <div className="title-menu-wrap" ref={wrapRef}>
          <button
            className="icon-btn title-menu-btn"
            onClick={() => setOpen((o) => !o)}
            aria-label="Session actions"
            aria-expanded={open}
            aria-haspopup="menu"
          >
            <IconChevronDown size={15} />
          </button>
          {open ? (
            <div className="title-menu" role="menu">
              <button
                className="title-menu-item"
                onClick={() => { setDraft(mission.title || mission.query || ""); setEditing(true); setOpen(false); }}
              >
                <IconPencil size={14} /> Rename
              </button>
              <button className="title-menu-item" onClick={() => { onTogglePin(mission.runId); close(); }}>
                <IconPin size={14} /> {mission.pinned ? "Unpin" : "Pin"}
              </button>
              <button className="title-menu-item danger" onClick={() => { onDelete(mission.runId); close(); }}>
                <IconTrash size={14} /> Delete
              </button>
            </div>
          ) : null}
        </div>
      ) : null}
      {showLibrary ? <LibraryMenu view={view} onNavigate={onNavigate} /> : null}
    </div>
  );
}

function WelcomeHero({ composer }) {
  return (
    <div className="hero-card anim-rise">
      <Planet size={88} ring />
      <h2>What should MARS <span className="accent">investigate</span>?</h2>
      <p>
        A team of research agents plans the inquiry, gathers sources, verifies claims,
        challenges conclusions and synthesizes a cited report — streaming live to this console.
      </p>
      {composer ? <div className="hero-composer">{composer}</div> : null}
    </div>
  );
}
