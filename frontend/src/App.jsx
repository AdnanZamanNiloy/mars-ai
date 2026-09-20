import { useCallback, useEffect, useRef, useState } from "react";
import { fetchSession, fetchTrace, listSessions, resumeResearch, startResearch } from "./api";
import { MODE_META, loadActiveSessionId, loadKnowledge, loadMissions, newSessionId, parseReport, removeKnowledgeItem, removeMission, saveActiveSessionId, saveKnowledgeItem, shouldAutoScroll, truncateFromMessage, updateMission, upsertMission } from "./lib";
import Sidebar from "./components/Sidebar";
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
import DocsView from "./components/DocsView";
import { IconChevronDown, IconChevronLeft, IconDoc, IconFolder, IconLayers, IconMenu, IconPencil, IconPin, IconTrash } from "./components/icons";

let seq = 1;
const nid = () => `m${Date.now()}-${seq++}`;

const VALID_VIEWS = ["landing", "workspace", "missions", "evidence", "knowledge", "agents", "providers", "docs"];

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
    degradedReasons: {},
    providerDegraded: false,
    providerKinds: [],
    answerSupport: null,
    // v2 signals: dependency-wave plan shape, live budget ledger,
    // per-source citation health, contradiction count.
    waves: [],
    waveReport: [],
    budget: null,
    citationHealth: null,
    contradictions: 0,
    // Red team (adversarial review): survival score + findings from the
    // critic pass. Streamed on every `critic` event; previously dropped.
    redteam: null,
    // v2.1: resolved intent (senses, domain, explanation level).
    intent: null,
    // Query router: direct-vs-research decision (path, reason, confidence).
    route: null,
    // Direct-answer path: the delivered answer, when the router answered
    // without research.
    directAnswer: null,
    quality: null,
    evidenceDistribution: null,
    // Answer-first outline: the section shape the writer targeted.
    outline: null,
    sectionWise: false,
    done: false,
    error: "",
    resumable: false,
    resuming: false,
    aborted: false,
  };
}

/* Rebuild the ordered replay thread for a persisted session.

 * The session row already carries one run entry per question, but the replay
 * card reads the REAL per-run trace (claims, sources, decisions). Fetching it
 * here — rather than synthesizing a stub from the stored report — is what
 * makes a restored answer render its evidence instead of a blank card.
 * The session's redundant `user` rows are dropped: the replay card renders
 * its own question, so keeping both showed every question twice. */
async function buildReplayMessages(session, signal) {
  const runRows = (session.messages || []).filter((m) => m.role !== "user");
  return Promise.all(runRows.map(async (meta) => {
    let trace;
    try {
      trace = await fetchTrace(meta.run_id, signal);
    } catch {
      // Trace unavailable (e.g. purged): degrade to the session row's report
      // so the chat still restores instead of erroring out.
      trace = {
        run_id: meta.run_id,
        query: meta.query,
        status: meta.status,
        confidence: meta.confidence,
        claims: [],
        sources: [],
        decisions: [],
        events: [],
        final_report: meta.report
          ? { report_markdown: meta.report, confidence: meta.confidence }
          : null,
      };
    }
    return {
      id: meta.id,
      kind: "replay",
      runId: meta.run_id,
      trace,
      query: trace.query || meta.query || "Replay",
      at: meta.created_at,
      // Map the persisted node trail to ThinkingSteps' shape so a restored
      // chat shows the same pipeline steps it showed while running, instead
      // of a single synthetic "Restored chat" line.
      steps: (trace.events || []).map((e) => ({
        at: e.ended_at || e.started_at || meta.created_at,
        kind: e.event_type === "end" ? "done" : "active",
        text: `${e.node} · ${e.event_type}`,
      })),
    };
  }));
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
  /* Plain anchor navigation (landing "Docs" link, docs-iframe "Console"
   * exit targeting the top window) changes the hash without pushState,
   * which fires `hashchange` but not `popstate` — sync the view here.
   * `go()` uses pushState, which never fires hashchange, so no loop. */
  useEffect(() => {
    const onHash = () => {
      const next = viewFromHash();
      if (viewRef.current !== next) {
        viewRef.current = next;
        setViewState(next);
      }
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const [missions, setMissions] = useState(() => loadMissions());
  const [sessionId, setSessionId] = useState(() => loadActiveSessionId());
  const [knowledge, setKnowledge] = useState(() => loadKnowledge());
  const [messages, setMessages] = useState([]);
  const [composer, setComposer] = useState("");
  const [mode, setMode] = useState("standard");
  const [running, setRunning] = useState(false);
  const [selectedFinding, setSelectedFinding] = useState(null);
  const [traceLog, setTraceLog] = useState([]);
  const [intelCollapsed, setIntelCollapsed] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [replaying, setReplaying] = useState(false);
  const [editingMessageId, setEditingMessageId] = useState(null);

  const controllerRef = useRef(null);
  const threadRef = useRef(null);
  const runRef = useRef(null); // tempId of the in-flight run
  const parentRef = useRef(null); // parent runId for challenge runs
  const sessionIdRef = useRef(sessionId); // stable chat id for async handlers
  const abortedRef = useRef(new Set()); // tempIds of runs the user stopped
  const autoScrollRef = useRef(true); // follow the stream unless user scrolled up

  const setActiveSession = useCallback((id) => {
    sessionIdRef.current = id || "";
    saveActiveSessionId(id || "");
    setSessionId(id || "");
  }, []);

  const saveMissions = useCallback((list) => setMissions(list), []);

  /* Refresh the sidebar chat list from the backend on mount so chats created
   * on another tab or before a localStorage clear are still discoverable. */
  useEffect(() => {
    const controller = new AbortController();
    (async () => {
      try {
        const remote = await listSessions(controller.signal);
        if (!Array.isArray(remote) || remote.length === 0) return;
        setMissions((prev) => {
          const bySession = new Map(remote.map((s) => [s.id, s]));
          const merged = remote.map((s) => {
            const local = prev.find((m) => m.sessionId === s.id) || {};
            return {
              ...local,
              sessionId: s.id,
              runId: local.runId || (s.run_ids && s.run_ids[s.run_ids.length - 1]) || null,
              query: local.query || s.query || s.title,
              title: local.title || "",
              status: s.status || local.status || "completed",
              confidence: s.confidence ?? local.confidence ?? null,
              runCount: s.run_count || 0,
            };
          });
          // Keep local-only pinned order for chats the backend hasn't seen.
          for (const m of prev) {
            if (m.sessionId && !bySession.has(m.sessionId)) merged.push(m);
          }
          return merged.slice(0, 30);
        });
      } catch {
        /* offline or backend not ready — localStorage remains the source */
      }
    })();
    return () => controller.abort();
  }, []);

  /* Restore the active chat's full history on reload. Only runs when there
   * is a persisted session and the thread is still empty. */
  useEffect(() => {
    const id = sessionIdRef.current;
    if (!id) return;
    const controller = new AbortController();
    (async () => {
      try {
        const session = await fetchSession(id, controller.signal);
        const restored = await buildReplayMessages(session, controller.signal);
        if (restored.length === 0) return;
        setMessages((prev) => {
          if (prev.length > 0) return prev; // user already started typing
          return restored;
        });
        const latestSteps = restored[restored.length - 1]?.steps || [];
        if (latestSteps.length > 0) setTraceLog(latestSteps);
      } catch {
        /* no persisted history — start fresh */
      }
    })();
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* Navigation helper: scroll a thread to its bottom, honoring the user's
   * intent. `autoScrollRef` is true unless the user scrolled up to read —
   * then we leave the viewport alone so streaming doesn't yank it back. */
  const scrollThreadToBottom = useCallback((behavior = "auto") => {
    const el = threadRef.current;
    if (!el) return;
    try {
      el.scrollTo({ top: el.scrollHeight, behavior });
    } catch {
      el.scrollTop = el.scrollHeight;
    }
  }, []);

  /* After sending, bring the NEW user turn into view with clearance for the
   * composer below. Scrolling to raw scrollHeight left the new message and
   * the first processing steps hidden under the fixed composer; anchoring on
   * the user bubble (minus an offset) keeps both visible. */
  const scrollToLatestTurn = useCallback((messageId, behavior = "smooth") => {
    const el = threadRef.current;
    if (!el) return;
    const node = messageId
      ? el.querySelector(`[data-message-id="${messageId}"]`)
      : null;
    const COMPOSER_CLEARANCE = 130;
    if (!node) {
      scrollThreadToBottom(behavior);
      return;
    }
    const target = node.offsetTop - COMPOSER_CLEARANCE;
    try {
      el.scrollTo({ top: Math.max(0, target), behavior });
    } catch {
      el.scrollTop = Math.max(0, target);
    }
  }, [scrollThreadToBottom]);

  useEffect(() => {
    const el = threadRef.current;
    if (!el) return;
    // A near-bottom position means the user is following the stream; a
    // deliberate scroll up flips intent off until they return to the bottom.
    const onScroll = () => {
      autoScrollRef.current = shouldAutoScroll(el);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (!autoScrollRef.current) return;
    // Smooth-follow the newest message as the run streams; a raw jump here
    // fights in-flight layout and causes visible jitter.
    scrollThreadToBottom("smooth");
  }, [messages, scrollThreadToBottom]);

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
    // Ignore any late event from a run the user stopped or replaced. Without
    // this, a buffered chunk could patch an orphaned run and even persist a
    // partial report after the interrupt.
    if (abortedRef.current.has(tempId)) return;
    switch (evt.type) {
      case "progress":
        if (evt.request_id) {
          patchRun(tempId, { runId: evt.request_id });
          // The backend confirms the chat id it grouped this run under; adopt
          // it so follow-ups keep reusing exactly the same session.
          const confirmedSession = evt.session_id || sessionIdRef.current;
          if (confirmedSession && confirmedSession !== sessionIdRef.current) {
            setActiveSession(confirmedSession);
          }
          if (evt.request_id || confirmedSession) {
            setMessages((prev) => {
              const msg = prev.find((m) => m.kind === "run" && m.run.tempId === tempId);
              if (msg) {
                saveMissions(upsertMission({
                  sessionId: confirmedSession, runId: evt.request_id, query: msg.run.query,
                  mode: msg.run.mode, status: "running", confidence: null, cost: null,
                  parentRunId: parentRef.current,
                }));
              }
              return prev;
            });
          }
        }
        if (evt.message) pushTrace({ text: evt.message, kind: "active" });
        break;
      case "intent":
        patchRun(tempId, {
          intent: {
            domain: evt.domain, queryType: evt.query_type, level: evt.explanation_level,
            ambiguity: evt.ambiguity, senses: Array.isArray(evt.senses) ? evt.senses : [],
            action: evt.recommended_action, origin: evt.origin,
          },
        });
        pushTrace({
          text: `Understood: ${evt.domain || "?"}${evt.ambiguity ? " — ambiguous, will disambiguate" : ""}`,
          kind: "done",
        });
        break;
      case "route":
        patchRun(tempId, {
          route: {
            path: evt.path, reason: evt.reason || "",
            confidence: typeof evt.confidence === "number" ? evt.confidence : null,
            origin: evt.origin || "", signals: evt.signals && typeof evt.signals === "object" ? evt.signals : {},
          },
        });
        pushTrace({
          text: evt.path === "direct"
            ? "Router: answerable directly — answering from general knowledge"
            : evt.path === "conversation"
              ? "Router: conversational turn"
              : "Router: external evidence required",
          kind: "done",
        });
        break;
      case "direct_answer":
        patchRun(tempId, {
          directAnswer: {
            answer: evt.answer || "",
            confidence: typeof evt.confidence === "number" ? evt.confidence : null,
            selfConfidence: typeof evt.self_confidence === "number" ? evt.self_confidence : null,
            reason: evt.reason || "",
            kind: evt.kind || "",
          },
        });
        pushTrace({
          text: evt.kind
            ? "Conversational reply"
            : "Answered directly (no sources consulted)",
          kind: "done",
        });
        break;
      case "plan":
        patchRun(tempId, {
          plan: Array.isArray(evt.items) ? evt.items : [],
          orchestration: evt.orchestration || {},
          waves: Array.isArray(evt.waves) ? evt.waves : [],
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
                  budget: evt.budget && typeof evt.budget === "object" ? evt.budget : m.run.budget,
                  redteam: evt.redteam && typeof evt.redteam === "object" ? evt.redteam : m.run.redteam,
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
            degradedReasons: evt.degraded_reasons && typeof evt.degraded_reasons === "object"
              ? evt.degraded_reasons : {},
            providerDegraded: evt.provider_degraded === true,
            providerKinds: Array.isArray(evt.provider_kinds) ? evt.provider_kinds : [],
            answerSupport: typeof evt.answer_support === "number" ? evt.answer_support : null,
            budget: evt.budget && typeof evt.budget === "object" ? evt.budget : m.run.budget,
            waveReport: Array.isArray(evt.wave_report) ? evt.wave_report : m.run.waveReport,
            citationHealth: evt.citation_health && typeof evt.citation_health === "object"
              ? evt.citation_health : m.run.citationHealth,
            quality: evt.quality && typeof evt.quality === "object" ? evt.quality : m.run.quality,
            evidenceDistribution: evt.evidence_distribution && typeof evt.evidence_distribution === "object"
              ? evt.evidence_distribution : m.run.evidenceDistribution,
            outline: evt.outline && typeof evt.outline === "object" ? evt.outline : m.run.outline,
            sectionWise: typeof evt.section_wise === "boolean" ? evt.section_wise : m.run.sectionWise,
            done: true,
            resuming: false,
          };
          if (run.runId) {
            saveMissions(upsertMission({
              sessionId: sessionIdRef.current, runId: run.runId, query: run.query, mode: run.mode,
              status: "completed", confidence: run.confidence,
              cost: run.budget && typeof run.budget.spent_usd === "number" ? run.budget.spent_usd : null,
              degraded: run.degraded,
            }));
          }
          return { ...m, run };
        }));
        pushTrace({ key: "findings", text: "Claims extracted", kind: "done" });
        pushTrace({ text: "Final report delivered", kind: "done" });
        // The finished answer can be much taller than the stream snapshot;
        // settle at the bottom so the full response is visible.
        if (autoScrollRef.current) {
          requestAnimationFrame(() => scrollThreadToBottom("smooth"));
        }
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
              sessionId: sessionIdRef.current, runId: run.runId, query: run.query, mode: run.mode,
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
  }, [patchRun, pushTrace, saveMissions, scrollThreadToBottom]);

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
        degradedReasons: {}, providerDegraded: false, providerKinds: [],
      });
      pushTrace({ text: `Resuming run ${resumeRun.runId.slice(0, 8)} from checkpoint`, kind: "active" });
    } else {
      const run = blankRun(queryText, mode);
      tempId = run.tempId;
      // One chat = one session id for its whole life. Mint one only when this
      // chat has none yet (first message, or after "New Chat").
      if (!sessionIdRef.current) setActiveSession(newSessionId());
      const at = new Date().toISOString();
      const userId = nid();
      setMessages((prev) => [
        ...prev,
        { id: userId, kind: "user", text: queryText, at },
        { id: nid(), kind: "run", run, at },
      ]);
      setTraceLog([]);
      setSelectedFinding(null);
      // Sending re-engages following and brings the new turn + the start of
      // its processing into view, clear of the composer.
      autoScrollRef.current = true;
      requestAnimationFrame(() => scrollToLatestTurn(userId, "smooth"));
      go("workspace");
    }

    runRef.current = tempId;
    setRunning(true);
    setIntelCollapsed(false);
    try {
      if (resumeRun) {
        await resumeResearch({ runId: resumeRun.runId, signal: controller.signal, onEvent: (e) => handleEvent(tempId, e) });
      } else {
        await startResearch({ query: queryText, mode, sessionId: sessionIdRef.current, signal: controller.signal, onEvent: (e) => handleEvent(tempId, e) });
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") {
        // Register the stop so any late in-flight event is dropped.
        abortedRef.current.add(tempId);
        setMessages((prev) => prev.map((m) => {
          if (m.kind !== "run" || m.run.tempId !== tempId) return m;
          const run = { ...m.run, aborted: true, resuming: false };
          if (run.runId) {
            saveMissions(upsertMission({
              sessionId: sessionIdRef.current, runId: run.runId, query: run.query, mode: run.mode,
              status: "cancelled", confidence: null, cost: null,
            }));
          }
          return { ...m, run };
        }));
        pushTrace({ text: "Research stopped by user", kind: "warn" });
      } else {
        handleEvent(tempId, { type: "error", message: err instanceof Error ? err.message : "Unknown stream error" });
      }
    } finally {
      controllerRef.current = null;
      runRef.current = null;
      parentRef.current = null;
      setRunning(false);
    }
  }, [running, mode, patchRun, pushTrace, handleEvent, saveMissions, setActiveSession, scrollToLatestTurn]);

  const submitQuery = useCallback((text) => {
    const v = text.trim();
    if (v.length < 5 || running) return;
    setComposer("");
    launch(v);
  }, [launch, running]);

  const abortRun = useCallback(() => {
    controllerRef.current?.abort();
  }, []);

  /* Interrupt-and-edit: stop any active run, drop everything from the edited
   * user turn onward, then resend the new text in the SAME session so the
   * chat's history stays coherent. The truncated turns are removed from the
   * thread; the backend already recorded the interrupted run as 'cancelled'
   * with no report, so nothing partial is shown as an answer. */
  const editAndResend = useCallback((messageId, newText) => {
    const v = (newText || "").trim();
    if (v.length < 5) return;
    controllerRef.current?.abort();
    setMessages((prev) => truncateFromMessage(prev, messageId));
    setEditingMessageId(null);
    setComposer("");
    setTraceLog([]);
    setSelectedFinding(null);
    // launch() appends the fresh user + run pair and reuses the session id.
    setTimeout(() => launch(v), 0);
  }, [launch]);

  /* Single entry point from the thread: open the editor for a user turn, or
   * submit it. A null id closes the editor without resending. */
  const handleEditMessage = useCallback((messageId, newText) => {
    if (messageId == null) {
      setEditingMessageId(null);
      return;
    }
    if (newText === undefined) {
      // Opening the editor: stop nothing yet — the user may cancel. If a run
      // is active, the composer's Stop remains the way to halt it.
      setEditingMessageId(messageId);
      return;
    }
    editAndResend(messageId, newText);
  }, [editAndResend]);

  const startNew = useCallback(() => {
    controllerRef.current?.abort();
    controllerRef.current = null;
    runRef.current = null;
    setMessages((prev) => {
      const active = [...prev].reverse().find((m) => m.kind === "run");
      const run = active?.run;
      if (run?.runId && !run.done && !run.error) {
        saveMissions(upsertMission({
          sessionId: sessionIdRef.current, runId: run.runId, query: run.query, mode: run.mode,
          status: "aborted", confidence: null, cost: null,
        }));
      }
      return [];
    });
    // "New Chat" is the ONLY action that mints a fresh session id.
    setActiveSession(newSessionId());
    setTraceLog([]);
    setSelectedFinding(null);
    setComposer("");
    setRunning(false);
    setIntelCollapsed(true);
    go("workspace");
  }, [setActiveSession]);

  const resumeRun = useCallback((run) => {
    if (running || !run.runId) return;
    launch("", { resumeRun: run });
  }, [launch, running]);

  const openReplay = useCallback(async (targetSessionId) => {
    if (replaying || running) return;
    setReplaying(true);
    try {
      const session = await fetchSession(targetSessionId);
      // Opening a chat adopts its session id, so a follow-up asked from a
      // replayed chat appends to THAT chat rather than starting a new one.
      setActiveSession(targetSessionId);
      const restored = await buildReplayMessages(session);
      if (restored.length === 0) {
        restored.push({
          id: nid(), kind: "notice",
          text: "This chat has no messages yet.",
          at: new Date().toISOString(),
        });
      }
      setMessages(restored);
      // Show the latest run's real pipeline steps in the panel; fall back to
      // a single notice only when the trace genuinely recorded no events.
      const latestSteps = restored[restored.length - 1]?.steps || [];
      setTraceLog(latestSteps.length > 0
        ? latestSteps
        : [{ at: new Date().toISOString(), kind: "done", text: "Trace loaded — no node events recorded" }]);
      setSelectedFinding(null);
      setIntelCollapsed(false);
      go("workspace");
    } catch (err) {
      setMessages([{
        id: nid(), kind: "notice", text: `Could not load chat: ${err instanceof Error ? err.message : "unknown error"}`,
        at: new Date().toISOString(),
      }]);
    } finally {
      setReplaying(false);
    }
  }, [replaying, running, setActiveSession]);

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
   * plain view names elsewhere. The menu acts on the displayed chat. */
  const VIEW_TITLES = { missions: "Research", evidence: "Evidence", knowledge: "Knowledge", agents: "Agents", providers: "Providers" };
  const titleMessage = [...messages].reverse().find((m) =>
    (m.kind === "run" && m.run?.query) ||
    (m.kind === "replay" && m.query) ||
    (m.kind === "user" && m.text));
  // The displayed chat is the active session; only after a run has been
  // persisted does a mission record exist for it.
  const activeMission = sessionId
    ? (missions.find((m) => m.sessionId === sessionId) || null)
    : null;
  const pageTitle = view === "workspace"
    ? (activeMission
      ? (activeMission.title || activeMission.query || "New Research")
      : (!titleMessage ? "New Research"
        : titleMessage.kind === "user" ? titleMessage.text
        : titleMessage.kind === "run" ? titleMessage.run.query
        : titleMessage.query))
    : (VIEW_TITLES[view] || "Command Center");

  /* Browser tab title follows the session: "query — MARS" while research
   * is on screen, plain "MARS" everywhere else. Favicon stays put. */
  useEffect(() => {
    try {
      const q = (pageTitle && pageTitle !== "New Research" && view === "workspace")
        ? pageTitle
        : "";
      document.title = q ? `${q} — MARS` : "MARS";
    } catch {
      /* non-DOM environment */
    }
  }, [pageTitle, view]);

  const renameMission = useCallback((targetSessionId, title) => {
    saveMissions(updateMission(targetSessionId, { title }));
  }, [saveMissions]);
  const toggleMissionPin = useCallback((targetSessionId) => {
    saveMissions(updateMission(targetSessionId, { pinned: !(missions.find((m) => m.sessionId === targetSessionId)?.pinned) }));
  }, [missions, saveMissions]);
  const deleteMission = useCallback((targetSessionId) => {
    if (!window.confirm("Delete this research chat? This cannot be undone.")) return;
    saveMissions(removeMission(targetSessionId));
    if (sessionIdRef.current === targetSessionId) startNew();
  }, [saveMissions, startNew]);

  return (
    view === "landing" ? (
      <Landing onStart={() => go("workspace")} onDocs={() => go("docs")} />
    ) : view === "docs" ? (
      <DocsView />
    ) : (
    <div className="shell">
      <ErrorBoundary>
      <Sidebar
        view={view}
        onNavigate={go}
        missions={missions}
        activeSessionId={sessionId}
        onOpenMission={openReplay}
        onNew={startNew}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onRename={renameMission}
        onTogglePin={toggleMissionPin}
        onDelete={deleteMission}
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
                  editingMessageId={editingMessageId}
                  onEditMessage={handleEditMessage}
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

function ThreadMessage({ message, running, onResume, onRegenerate, steps, editingMessageId, onEditMessage }) {
  // A restored replay carries its OWN pipeline steps (from its trace), so a
  // multi-question chat shows each run's real steps instead of the global
  // log's most-recent run duplicated across every card.
  const stepList = message.kind === "replay" ? (message.steps || []) : steps;
  if (message.kind === "user") {
    const isEditing = editingMessageId === message.id;
    return (
      <UserMessage
        text={message.text}
        time={message.at}
        messageId={message.id}
        editing={isEditing}
        disabled={running}
        canRegenerate={!running}
        onRegenerate={onRegenerate ? () => onRegenerate(message.text) : undefined}
        onEdit={onEditMessage ? () => onEditMessage(message.id) : undefined}
        onCancelEdit={() => onEditMessage?.(null)}
        onSubmitEdit={(txt) => onEditMessage?.(message.id, txt)}
      />
    );
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
        <ThinkingSteps steps={stepList} />
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
        <ThinkingSteps steps={stepList} />
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
      <MarsMessageShell text={message.text}>
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
      <h2>What should MARS <span className="accent">investigate</span>?</h2>
      <p>
        A team of research agents plans the inquiry, gathers sources, verifies claims,
        challenges conclusions and synthesizes a cited report live to this console.
      </p>
      {composer ? <div className="hero-composer">{composer}</div> : null}
    </div>
  );
}
