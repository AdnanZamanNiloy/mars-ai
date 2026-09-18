/* ============================================================================
   MARS Documentation — content model.
   Every claim below is grounded in this repository's source, README.md,
   ARCHITECTURE.md, BENCHMARK_RESULTS.md and CHANGELOG.md.
   ========================================================================== */
(function () {
  "use strict";

  function icon(path, extra) {
    return '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"' + (extra || "") + ">" + path + "</svg>";
  }

  var I = {
    overview: icon('<path d="M4 5h16M4 12h16M4 19h10"/>'),
    brain: icon('<path d="M12 5a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V8a3 3 0 0 0-3-3Z"/><path d="M9 9a3 3 0 0 1-3-3M15 15a3 3 0 0 0 3 3"/>'),
    flow: icon('<circle cx="6" cy="6" r="2.5"/><circle cx="18" cy="18" r="2.5"/><path d="M8.5 6H16a2 2 0 0 1 2 2v7.5M6 8.5V16a2 2 0 0 0 2 2h7.5"/>'),
    agents: icon('<circle cx="12" cy="8" r="3"/><path d="M5 20a7 7 0 0 1 14 0"/><path d="M19 8h2M3 8h2"/>'),
    shield: icon('<path d="M12 3 5 6v6c0 4.2 2.9 7.5 7 9 4.1-1.5 7-4.8 7-9V6l-7-3Z"/><path d="m9 12 2 2 4-4"/>'),
    gauge: icon('<path d="M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z"/><path d="M12 12 15 8"/><path d="M4.5 19a9 9 0 1 1 15 0"/>'),
    quote: icon('<path d="M7 7h4v4H9a3 3 0 0 0-3 3V7ZM13 7h4v4h-2a3 3 0 0 0-3 3V7Z"/>'),
    layers: icon('<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 13 9 5 9-5"/>'),
    control: icon('<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M8 9h8M8 13h5M8 17h3"/>'),
    api: icon('<path d="M8 8 4 12l4 4M16 8l4 4-4 4M13 5l-2 14"/>'),
    bolt: icon('<path d="M13 3 5 13h6l-1 8 8-10h-6l1-8Z"/>'),
    db: icon('<ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6"/><path d="M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3"/>'),
    check: icon('<path d="m5 12 4.5 4.5L19 7"/>'),
    warn: icon('<path d="M12 4 3 19h18L12 4Z"/><path d="M12 10v4M12 17h.01"/>'),
    info: icon('<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>'),
    copy: icon('<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h8"/>'),
    target: icon('<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3.4"/>'),
    search: icon('<circle cx="11" cy="11" r="7"/><path d="m20 20-3.2-3.2"/>'),
    book: icon('<path d="M5 4h11a2 2 0 0 1 2 2v14H7a2 2 0 0 1-2-2V4Z"/><path d="M5 18a2 2 0 0 1 2-2h11"/>'),
    scale: icon('<path d="M12 4v16M7 8l-3 6h6L7 8ZM17 8l-3 6h6l-3-6Z"/><path d="M6 4h12"/>'),
    arrow: icon('<path d="M12 5v14M6 13l6 6 6-6"/>')
  };

  var NAV = [
    { group: "Getting Started", items: [
      { id: "introduction", title: "Introduction", icon: I.overview },
      { id: "capabilities", title: "Capabilities", icon: I.layers },
      { id: "quickstart", title: "Quick Start", icon: I.bolt },
      { id: "modes", title: "Research Modes", icon: I.control }
    ]},
    { group: "Architecture", items: [
      { id: "architecture", title: "System Architecture", icon: I.flow },
      { id: "workflow", title: "Research Workflow", icon: I.target },
      { id: "orchestration", title: "Orchestration & Waves", icon: I.brain },
      { id: "streaming", title: "Streaming Events", icon: I.api }
    ]},
    { group: "Agents", items: [
      { id: "agents", title: "Agent Roster", icon: I.agents },
      { id: "search", title: "Search & Retrieval", icon: I.search },
      { id: "summarizer", title: "Specialist Extraction", icon: I.book }
    ]},
    { group: "Evidence & Verification", items: [
      { id: "verification", title: "Claim Verification", icon: I.shield },
      { id: "contradictions", title: "Contradictions", icon: I.scale },
      { id: "citations", title: "Citations & Health", icon: I.quote },
      { id: "confidence", title: "Confidence Engine", icon: I.gauge }
    ]},
    { group: "Operations", items: [
      { id: "budget", title: "Cost & Budget", icon: I.gauge },
      { id: "stopping", title: "Depth & Stopping", icon: I.target },
      { id: "reliability", title: "Reliability", icon: I.warn },
      { id: "persistence", title: "Persistence & Trace", icon: I.db },
      { id: "api", title: "API Reference", icon: I.api },
      { id: "benchmarks", title: "Benchmarks", icon: I.check }
    ]}
  ];

  var C = {};
  var ARROW = '<div class="pipe-arrow">' + I.arrow + "</div>";

  /* Code-block helpers: keep shell content readable without quote escaping. */
  var COPY_BTN = '<button class="copy-btn" type="button">' + I.copy + " Copy</button>";
  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function shBlock(rows) {
    var body = rows.join("\n");
    return (
      '<div class="code-block term" data-lang="bash">' +
      '<div class="code-head"><span class="lang">' +
      '<span class="term-dots"><i></i><i></i><i></i></span> bash</span>' +
      COPY_BTN +
      "</div><pre><code>" + body + "</code></pre></div>"
    );
  }
  /* t() = plain shell text (HTML-escaped); c()/g()/y() = coloured tokens. */
  function t(text) { return esc(text); }
  function c(comment) { return '<span class="tok-c">' + esc(comment) + "</span>"; }
  function g(text) { return '<span class="tok-g">' + esc(text) + "</span>"; }
  function y(text) { return '<span class="tok-y">' + esc(text) + "</span>"; }

  /* ============================ Introduction ============================ */
  C.introduction = [
    '<div class="hero">',
      '<p class="eyebrow">Multi-Agent Research System · v2.1</p>',
      '<h1>Research beyond the obvious.</h1>',
      '<p class="lede">MARS is an autonomous research intelligence platform. It decomposes a complex question, plans a research strategy, dispatches specialized agents in parallel, verifies every claim against its cited source, challenges its own conclusions, and delivers a cited report with an honest, evidence-derived confidence score.</p>',
      '<div class="hero-cta">',
        '<a class="btn primary" href="#quickstart">', I.bolt, " Quick start</a>",
        '<a class="btn" href="#architecture">', I.flow, " Architecture</a>",
        '<a class="btn" href="#benchmarks">', I.check, " Benchmarks</a>",
      "</div>",
      '<div class="hero-stats">',
        '<div class="hero-stat"><div class="v">1.00</div><div class="k">Verification F1 · 16 labeled cases</div></div>',
        '<div class="hero-stat"><div class="v">0%</div><div class="k">Hallucination leak · 6/6 rejected</div></div>',
        '<div class="hero-stat"><div class="v">1.00</div><div class="k">Contradiction detection F1</div></div>',
        '<div class="hero-stat"><div class="v">343</div><div class="k">Offline tests passing</div></div>',
      "</div>",
    "</div>",

    "<h2 id=\"what-is-mars\">What MARS is</h2>",
    "<p>MARS is not a chatbot wrapped around a search API. It behaves like a research organization assembled for one question: it understands the question before it searches, plans a strategy, delegates work to specialist agents, runs independent investigations, verifies evidence deterministically, attacks its own findings, and synthesizes a decision-ready answer with traceable citations.</p>",
    "<p>The pipeline is a <strong>LangGraph state machine</strong> driving bounded, asynchronous agents. Generation runs on remote LLM APIs — Groq primary, HuggingFace fallback, plus any OpenAI-compatible endpoint you add. Every run persists its full timeline to SQLite, so it can be replayed, audited, or resumed after a failure.</p>",

    '<div class="callout">',
      '<span class="c-ico">', I.info, "</span>",
      '<div class="c-body"><div class="c-title">Built for modest hardware</div>',
      "<p>MARS targets an 8&nbsp;GB RAM host with no GPU and no local model inference. Concurrency is bounded everywhere — <code>MAX_PARALLEL_SEARCH=2</code>, <code>MAX_PARALLEL_LLM=2</code>, <code>MAX_PARALLEL_AGENTS=3</code> — and raw page content is released from memory after verification. These are real constraints, not aspirations.</p></div>",
    "</div>",

    "<h2 id=\"design-philosophy\">Design philosophy</h2>",
    "<p>Four principles run through every module:</p>",
    '<div class="grid cols-2">',
      '<div class="feature"><div class="f-ico">', I.shield, "</div><h4>Verification is deterministic</h4><p>A verifier that costs a model call is a verifier you consult less often. Lexical, numeric and polarity checks run in roughly 153&nbsp;µs per claim and catch fabricated numbers, negation inversions and topical mismatches.</p></div>",
      '<div class="feature"><div class="f-ico">', I.scale, "</div><h4>Disagreement is intelligence</h4><p>Contradictions are surfaced and penalized, not blended into an average. A weighted mean lets three strong sources mask one direct lie.</p></div>",
      '<div class="feature"><div class="f-ico">', I.gauge, "</div><h4>Uncertainty stays visible</h4><p>Confidence is computed from measured evidence signals — source quality, diversity, citation coverage, verification strength, critic survival. Degraded runs are capped below the sufficiency threshold so fallbacks can never read as high confidence.</p></div>",
      '<div class="feature"><div class="f-ico">', I.warn, "</div><h4>Every failure degrades gracefully</h4><p>Each LLM-calling agent has a deterministic fallback. Provider outages, quota walls and timeouts produce disclosed degradation, not silent wrong answers.</p></div>",
    "</div>"
  ];

  /* ============================ Capabilities ============================ */
  C.capabilities = [
    '<p class="eyebrow">Capabilities</p>',
    '<h2 id="capabilities">What separates MARS</h2>',
    '<p class="lede">Each capability below maps to a specific module, threshold and measured result — not a marketing bullet.</p>',

    '<h3 id="cap-intent">Understand before searching</h3>',
    "<p>Before any research is shaped, an <strong>intent classifier</strong> resolves what the question actually means. It ranks competing senses for ambiguous terms, sets the research domain and explanation level, and never silently picks the wrong meaning. The canonical failure it prevents: a question about the ML <em>transformer</em> dragging in electrical-transformer statistics.</p>",
    '<div class="code-block" data-lang="intent.json"><div class="code-head"><span class="lang">intent · resolved senses</span><button class="copy-btn" type="button">', I.copy, " Copy</button></div><pre><code>{\n  \"query\": \"What is a transformer?\",\n  \"domain\": \"machine_learning\",\n  \"explanation_level\": \"practical\",\n  \"ambiguity\": true,\n  \"senses\": [\n    { \"label\": \"ML architecture\",   \"probability\": 0.75 },\n    { \"label\": \"electrical device\", \"probability\": 0.20 }\n  ],\n  \"recommended_action\": \"research_both\"\n}</code></pre></div>",
    '<p class="small muted">When the top-two sense gap falls below <code>RESEARCH_BOTH_GAP = 0.30</code>, both senses are researched and kept separate end-to-end. Definitional queries on ambiguous terms force both. A curated homonym table — transformer, python, apple, amazon, jaguar, eclipse, windows, shell — provides a deterministic fallback when the model is unavailable.</p>',

    '<h3 id="cap-verification">Verify every claim</h3>',
    "<p>Each extracted claim is checked against the source text it cites by a deterministic verifier — no LLM, so it can run over every claim on every pass. Hard checks include weighted lexical overlap, source authority, unit-aware numeric grounding with 2% tolerance, polarity consistency, direct-quote location and freshness.</p>",

    '<h3 id="cap-citations">Re-check citations live</h3>',
    "<p>At delivery the answer’s citation legend is re-validated: each cited URL is probed live (<code>HEAD</code>, then a 2&nbsp;KB ranged <code>GET</code>), and that health signal is fused with sentence-support evidence into per-source verdicts.</p>",

    '<h3 id="cap-contradictions">Surface contradictions</h3>',
    "<p>Three detectors — polarity, temporal and unit-aware numeric — find conflicting evidence, rank it by severity, and feed resolution follow-ups back into the research loop.</p>",

    '<h3 id="cap-confidence">Compute confidence from evidence</h3>',
    "<p>The confidence engine blends nine evidence signals and subtracts a pool-size-scaled contradiction penalty. It is not a model’s self-report.</p>",

    '<h3 id="cap-cost">Account for every token</h3>',
    "<p>A per-run ledger records tokens, USD, LLM calls, searches and cache hits. A budget governor refuses research passes that cannot be paid for, and the live meter streams to the UI.</p>",

    '<div class="grid cols-3" style="margin-top:26px">',
      '<div class="metric-card"><div class="m-val acc">F1 1.00</div><div class="m-lab">Verification accuracy</div><div class="m-sub">16 labeled claim/source cases</div></div>',
      '<div class="metric-card"><div class="m-val acc">0.000</div><div class="m-lab">Hallucination leak rate</div><div class="m-sub">6/6 fabricated claims rejected</div></div>',
      '<div class="metric-card"><div class="m-val acc">100%</div><div class="m-lab">Citation sentence accuracy</div><div class="m-sub">numeric grounding 1.000</div></div>',
    "</div>"
  ];

  /* ============================ Quickstart ============================ */
  C.quickstart = [
    '<p class="eyebrow">Getting started</p>',
    '<h2 id="quickstart">Quick start</h2>',
    '<p class="lede">Two processes: a FastAPI backend and a Vite/React console. Free-tier provider keys are enough to run it.</p>',

    '<h3 id="qs-setup">1 · Backend</h3>',
    shBlock([
      c("# from the repository root"),
      t("cd backend"),
      t("python3 -m venv .venv && source .venv/bin/activate"),
      t("pip install -r requirements.txt"),
      t("cp .env.example .env") + c("   # add GROQ_API_KEY (free tier works)"),
      g("uvicorn") + t(" main:app --host 127.0.0.1 --port 8000")
    ]),

    '<h3 id="qs-frontend">2 · Frontend</h3>',
    shBlock([
      c("# in a second terminal"),
      t("cd frontend && npm install && npm run dev"),
      c("# open http://127.0.0.1:5173")
    ]),
    "<p>Set at least one provider in <code>backend/.env</code>. The app requires one of <code>GROQ_API_KEY</code>, <code>HUGGINGFACE_API_KEY</code>, or the complete <code>CUSTOM_LLM_*</code> trio, and fails to start otherwise. Additional OpenAI-compatible providers can be added at runtime from the <strong>Providers</strong> tab; keys are Fernet-encrypted at rest.</p>",

    '<h3 id="qs-health">3 · Smoke test</h3>',
    shBlock([
      g("curl") + t(" -s http://127.0.0.1:8000/api/health"),
      c('# {"status":"ok"}'),
      t(""),
      g("curl") + t(" -N -X POST http://127.0.0.1:8000/api/research/stream \\"),
      t("  -H ") + y('"Content-Type: application/json"') + t(" \\"),
      t("  -d ") + y("'{\"query\": \"What is retrieval augmented generation?\"}'")
    ]),

    '<h3 id="qs-tests">4 · Tests &amp; benchmarks</h3>',
    shBlock([
      g("pytest") + t(" tests/ -v                 ") + c("# 343 tests, fully offline"),
      g("python") + t(" bench/run_offline.py      ") + c("# deterministic, no network, no keys"),
      g("python") + t(" bench/run_live.py         ") + c("# live-provider quality suite"),
      g("ruff") + t(" check app tests bench scripts"),
      g("mypy") + t(" app")
    ]),

    '<div class="callout info">',
      '<span class="c-ico">', I.info, "</span>",
      '<div class="c-body"><div class="c-title">Configuration is centralised</div>',
      "<p>Every setting lives in <code>app/core/config.py</code> and gets a placeholder in <code>backend/.env.example</code>. The full reference is in <code>CONFIG.md</code>.</p></div>",
    "</div>"
  ];

  /* ============================ Modes ============================ */
  C.modes = [
    '<p class="eyebrow">Getting started</p>',
    '<h2 id="modes">Research modes</h2>',
    "<p class=\"lede\">A mode is an explicit budget-and-depth contract: how many agents, how many iteration passes, whether deep research is armed, and the confidence target the run must reach before it is allowed to stop.</p>",
    '<div class="table-wrap"><table>',
      "<thead><tr><th>Mode</th><th class=\"num\">Max agents</th><th class=\"num\">Max iterations</th><th>Deep research</th><th class=\"num\">Confidence target</th><th class=\"num\">Budget ×</th></tr></thead>",
      "<tbody>",
        '<tr><td><strong>quick</strong></td><td class="num">2</td><td class="num">1</td><td>—</td><td class="num">0.60</td><td class="num">0.35×</td></tr>',
        '<tr><td><strong>standard</strong></td><td class="num">4</td><td class="num">3</td><td>—</td><td class="num">0.75</td><td class="num">1.0×</td></tr>',
        '<tr><td><strong>deep</strong></td><td class="num">5</td><td class="num">5</td><td><span class="pill mint">On</span></td><td class="num">0.80</td><td class="num">2.5×</td></tr>',
        '<tr><td><strong>executive</strong></td><td class="num">5</td><td class="num">4</td><td><span class="pill mint">On</span></td><td class="num">0.78</td><td class="num">2.0×</td></tr>',
        '<tr><td><strong>audit</strong></td><td class="num">3</td><td class="num">4</td><td>—</td><td class="num">0.85</td><td class="num">1.2×</td></tr>',
        '<tr><td><strong>redteam</strong></td><td class="num">3</td><td class="num">2</td><td>—</td><td class="num">0.72</td><td class="num">1.2×</td></tr>',
      "</tbody>",
    "</table></div>",
    '<p class="small muted">Mode presets live in <code>MODE_PRESETS</code> (app/agents/orchestrator.py). <code>deep</code> and <code>executive</code> are the only modes allowed to exceed <code>MAX_PARALLEL_AGENTS</code> — an explicit opt-in, then a budget-governor check. Budget multipliers are applied to all four ceilings in <code>MODE_BUDGET_MULTIPLIER</code>.</p>',

    '<h3 id="modes-recommend">Mode recommendation</h3>',
    "<p><code>recommend_mode()</code> suggests a mode from the query shape, in priority order:</p>",
    "<ul>",
      "<li>Decision framing + high/very-high complexity → <code>executive</code></li>",
      "<li>Very-high complexity → <code>deep</code></li>",
      "<li>Low complexity + factual → <code>quick</code></li>",
      "<li>Contested claim → <code>redteam</code> (medium) or <code>deep</code></li>",
      "<li>Otherwise → <code>standard</code></li>",
    "</ul>",

    '<h3 id="modes-complexity">Complexity scoring</h3>',
    "<p>The orchestrator scores complexity with an additive, LLM-free model — spending a model call to decide how many model calls to spend is a bad trade. Signals include query length, comparative/analytical/exploratory markers, entity count, time bounds, decision framing, controversy and multi-dimensionality.</p>",
    '<div class="table-wrap"><table>',
      '<thead><tr><th>Signal</th><th class="num">Points</th></tr></thead><tbody>',
        '<tr><td>Comparative framing (<code>vs</code>, <code>compare</code>, <code>trade-offs</code>)</td><td class="num">+3</td></tr>',
        '<tr><td>Query longer than 25 words</td><td class="num">+3</td></tr>',
        "<tr><td>≥ 4 research dimensions detected</td><td class=\"num\">+3</td></tr>",
        '<tr><td>Analytical framing (<code>should</code>, <code>impact</code>, <code>risk</code>)</td><td class="num">+2</td></tr>',
        '<tr><td>Exploratory framing (<code>overview</code>, <code>trends</code>, <code>future of</code>)</td><td class="num">+2</td></tr>',
        '<tr><td>Decision framing (<code>should we</code>, <code>invest</code>, <code>recommend</code>)</td><td class="num">+2</td></tr>',
        '<tr><td>Contested (<code>debate</code>, <code>ethical</code>, <code>myth</code>)</td><td class="num">+1</td></tr>',
        '<tr><td>Entity count ≥ 3</td><td class="num">+1</td></tr>',
        '<tr><td>Time-bound (<code>2025</code>, <code>over</code>, <code>decade</code>)</td><td class="num">+1</td></tr>',
      "</tbody>",
    "</table></div>",
    "<p><strong>Levels:</strong> score ≥ 7 → <code>very_high</code>; ≥ 5 → <code>high</code>; ≥ 3 → <code>medium</code>; else <code>low</code>. Base sub-question targets by level are 3 / 4 / 5 / 6 respectively. Ten dimensions can be detected (financial, market, technical, legal, policy, scientific, academic, geopolitical, environmental, social), capped at six.</p>"
  ];

  /* ============================ Architecture ============================ */
  C.architecture = [
    '<p class="eyebrow">Architecture</p>',
    '<h2 id="architecture">System architecture</h2>',
    '<p class="lede">One LangGraph instance drives the research loop. Each node is a bounded, independently testable agent; shared intelligence lives in deterministic core modules.</p>',

    '<div class="pipeline">',
      '<div class="pipe-node"><div class="pn-ico">01</div><div class="pn-body"><div class="pn-title">Intent</div><div class="pn-desc">Resolve ambiguity, domain and explanation level before research is shaped.</div></div></div>', ARROW,
      '<div class="pipe-node"><div class="pn-ico">02</div><div class="pn-body"><div class="pn-title">Orchestrator <code>app/agents/orchestrator.py</code></div><div class="pn-desc">Complexity score → target agents, iteration ceiling, required axes, budget multiplier.</div></div></div>', ARROW,
      '<div class="pipe-node"><div class="pn-ico">03</div><div class="pn-body"><div class="pn-title">Planner <code>app/agents/planner.py</code></div><div class="pn-desc">Delegation contracts: axis, search type, minimum sources, variants, wave, sense.</div></div></div>', ARROW,
      '<div class="pipe-node"><div class="pn-ico">04</div><div class="pn-body"><div class="pn-title">Search <code>app/agents/search.py</code></div><div class="pn-desc">Five providers, circuit breakers, canonical-url dedup, domain diversity caps, disk cache.</div></div></div>', ARROW,
      '<div class="pipe-node"><div class="pn-ico">05</div><div class="pn-body"><div class="pn-title">Summarizer <code>app/agents/summarizer.py</code></div><div class="pn-desc">Wave-ordered specialist extraction; wave N receives wave N−1 findings as grounding.</div></div></div>', ARROW,
      '<div class="pipe-node"><div class="pn-ico">06</div><div class="pn-body"><div class="pn-title">Verifier <code>app/agents/verifier.py</code></div><div class="pn-desc">Deterministic per-claim checks; blanks raw content after each pass.</div></div></div>', ARROW,
      '<div class="pipe-node"><div class="pn-ico">07</div><div class="pn-body"><div class="pn-title">Critic · Contradictions · Red Team · Confidence</div><div class="pn-desc">Sufficiency gate, adversarial review, conflict detection, evidence-derived score.</div></div></div>', ARROW,
      '<div class="pipe-node"><div class="pn-ico">08</div><div class="pn-body"><div class="pn-title">Synthesizer <code>app/agents/synthesizer.py</code></div><div class="pn-desc">Sense-separated, cited answer with mandatory disambiguation.</div></div></div>', ARROW,
      '<div class="pipe-node"><div class="pn-ico">09</div><div class="pn-body"><div class="pn-title">Quality gate → Citation health → Finalize</div><div class="pn-desc">Five-axis answer score, one bounded re-synthesis, live URL validation, persisted report.</div></div></div>',
    "</div>",

    '<div class="callout info">',
      '<span class="c-ico">', I.flow, "</span>",
      '<div class="c-body"><div class="c-title">One loop, three exits</div>',
      "<p><code>route_after_critic</code> chooses <strong>expand</strong> (more research), <strong>finalize</strong> (sufficient), or a hard stop (budget wall, iteration ceiling, marginal-gain stall, or no novel queries left). The stop logic lives in <code>app/core/depth_controller.py</code>.</p></div>",
    "</div>",

    '<h3 id="arch-modules">Core intelligence modules</h3>',
    '<div class="table-wrap"><table>',
      "<thead><tr><th>Module</th><th>Responsibility</th></tr></thead><tbody>",
        "<tr><td><code>agents/intent.py</code></td><td>Ambiguity → ranked senses, domain, explanation level; deterministic fallback with a curated homonym table.</td></tr>",
        "<tr><td><code>agents/answer_quality.py</code></td><td>Pre-delivery five-axis 0–100 score from measured state; one bounded re-synthesis; sub-threshold answers disclosed.</td></tr>",
        "<tr><td><code>agents/planner.py</code></td><td>Delegation contracts with axis coverage enforcement and dependency waves (max 3).</td></tr>",
        "<tr><td><code>agents/search.py</code></td><td>Five providers, per-provider circuit breakers, canonical-url and near-dup dedup, domain caps, freshness half-lives.</td></tr>",
        "<tr><td><code>agents/verifier.py</code></td><td>Deterministic claim checks; memory hygiene.</td></tr>",
        "<tr><td><code>core/contradictions.py</code></td><td>Polarity → temporal → unit-aware numeric detectors; severity ordering; cap 5.</td></tr>",
        "<tr><td><code>core/confidence.py</code></td><td>Nine signals plus a pool-size-scaled contradiction penalty; degraded-run cap.</td></tr>",
        "<tr><td><code>agents/citation_check.py</code></td><td>Live URL re-validation fused with sentence support into per-source verdicts.</td></tr>",
        "<tr><td><code>core/semantic.py</code></td><td>CPU-light TF-IDF hybrid: stemming, synonym canonicalization, negation weighting.</td></tr>",
        "<tr><td><code>agents/redteam.py</code></td><td>Heuristic adversarial review; survival score gates the critic.</td></tr>",
        "<tr><td><code>core/depth_controller.py</code></td><td>Intelligent stopping: sufficiency, marginal-gain stalls, ceilings, budget walls.</td></tr>",
        "<tr><td><code>core/decision.py</code></td><td>Decision layer (options → recommendation → rationale → risk) for strategic queries.</td></tr>",
      "</tbody>",
    "</table></div>",

    '<h3 id="arch-lifecycles">Data lifecycles</h3>',
    "<ul>",
      "<li><strong>Raw page content</strong> is blanked by the verifier after each pass — verification is the last consumer of full text. Snippets persist for transparency.</li>",
      "<li><strong>Facts</strong> accumulate across passes with append-only plan growth; semantic dedup merges restatements and records corroboration from distinct domains only.</li>",
      "<li><strong>Caches:</strong> search results 1&nbsp;h, LLM responses 6&nbsp;h, both size-capped at 250&nbsp;MB and LRU-evicted.</li>",
      "<li><strong>SQLite:</strong> every run persists agent events, sources, claims, verification results, contradictions, citations, critic reviews and decisions — the trace endpoint reconstructs the full timeline.</li>",
    "</ul>"
  ];

  /* ============================ Workflow ============================ */
  C.workflow = [
    '<p class="eyebrow">Architecture</p>',
    '<h2 id="workflow">Research workflow</h2>',
    '<p class="lede">The sixteen-step lifecycle, from question to auditable answer. Steps 5–12 form the bounded research loop; everything is streamed live to the console.</p>',
    '<div class="table-wrap"><table>',
      '<thead><tr><th class="num">#</th><th>Stage</th><th>What happens</th></tr></thead><tbody>',
        '<tr><td class="num">01</td><td><strong>Receive</strong></td><td>User submits a query and mode.</td></tr>',
        '<tr><td class="num">02</td><td><strong>Understand</strong></td><td>Intent classifier resolves ambiguity, domain and explanation level.</td></tr>',
        '<tr><td class="num">03</td><td><strong>Score</strong></td><td>Orchestrator scores complexity and sets plan targets.</td></tr>',
        '<tr><td class="num">04</td><td><strong>Plan</strong></td><td>Planner emits delegation contracts, axis coverage and dependency waves.</td></tr>',
        '<tr><td class="num">05</td><td><strong>Search</strong></td><td>Providers run concurrently under a bounded semaphore; results are deduped and diversity-capped.</td></tr>',
        '<tr><td class="num">06</td><td><strong>Summarize</strong></td><td>Specialist agents extract claims per contract, wave by wave, with prerequisite grounding.</td></tr>',
        '<tr><td class="num">07</td><td><strong>Collect</strong></td><td>Claims enter the evidence pool; raw content is released after verification.</td></tr>',
        '<tr><td class="num">08</td><td><strong>Verify</strong></td><td>Deterministic checks annotate every claim with a score and reasons.</td></tr>',
        '<tr><td class="num">09</td><td><strong>Challenge</strong></td><td>Red team attacks the evidence base; critic rules on sufficiency.</td></tr>',
        '<tr><td class="num">10</td><td><strong>Resolve</strong></td><td>Contradiction detectors surface and rank conflicting evidence.</td></tr>',
        '<tr><td class="num">11</td><td><strong>Decide depth</strong></td><td>Depth controller chooses expand or finalize.</td></tr>',
        '<tr><td class="num">12</td><td><strong>Expand</strong></td><td>Corroboration and counter-evidence queries target the weakest claims, then loop back to search.</td></tr>',
        '<tr><td class="num">13</td><td><strong>Synthesize</strong></td><td>Cited answer, sense-separated, with a decision layer for strategic queries.</td></tr>',
        '<tr><td class="num">14</td><td><strong>Gate</strong></td><td>Five-axis quality score; one bounded re-synthesis if below threshold.</td></tr>',
        '<tr><td class="num">15</td><td><strong>Validate</strong></td><td>Live citation health check on the report legend.</td></tr>',
        '<tr><td class="num">16</td><td><strong>Deliver</strong></td><td>Report, evidence, contradictions, limitations and confidence persisted and streamed.</td></tr>',
      "</tbody>",
    "</table></div>",

    '<h3 id="wf-corrob">Targeted corroboration</h3>',
    "<p>When a claim rests on a single publisher, the workflow does not passively hope for a second source. It builds a claim-specific query that targets the claim’s own terms, <em>excludes</em> the current publisher (<code>-site:domain</code>), and aims at authoritative registries — government, education, international, agencies, journals and datasets. A per-claim investigation registry spends a bounded attempt budget (<code>max_corroboration_attempts = 2</code>) and records outcomes, so exhausted single-source gaps are disclosed as limitations rather than silently re-chased.</p>",

    '<h3 id="wf-counter">Counter-evidence by design</h3>',
    "<p>The prompt is explicit: do not only search for support. For every single-source or contradicted claim, the workflow issues a query seeking counter-evidence, limitations, or credible opposing views. These ride the normal expansion loop — no extra pipeline stage.</p>",

    '<div class="callout">',
      '<span class="c-ico">', I.search, "</span>",
      '<div class="c-body"><div class="c-title">Executed-query memory</div>',
      "<p>Every query actually issued is remembered by normalized text, so an identical query can never be re-issued across passes. A repeat would return the same evidence at full cost.</p></div>",
    "</div>"
  ];

  /* ============================ Orchestration ============================ */
  C.orchestration = [
    '<p class="eyebrow">Architecture</p>',
    '<h2 id="orchestration">Orchestration &amp; waves</h2>',
    '<p class="lede">The planner does not emit a flat list of questions. It emits contracts with dependency awareness, and the summarizer executes them in dependency order.</p>',

    '<h3 id="orch-contract">Delegation contracts</h3>',
    "<p>Every sub-question is a structured contract. This is what prevents agents from duplicating one another and keeps each investigation bounded.</p>",
    '<div class="code-block" data-lang="contract.json"><div class="code-head"><span class="lang">delegation contract</span><button class="copy-btn" type="button">', I.copy, " Copy</button></div><pre><code>{\n  \"id\": 2,\n  \"question\": \"Compare capex and levelised cost of nuclear vs solar\",\n  \"axis\": \"comparison\",\n  \"search_type\": \"comparison\",\n  \"specialist\": \"financial\",\n  \"minimum_sources\": 3,\n  \"priority\": 1,\n  \"depends_on\": [1],\n  \"wave\": 1,\n  \"sense\": \"\",\n  \"variants\": [\"nuclear vs solar cost per MWh\"],\n  \"primary_source_query\": \"site:gov OR site:edu nuclear LCOE solar LCOE\"\n}</code></pre></div>",

    '<h3 id="orch-axis">Axis coverage enforcement</h3>',
    "<p>The orchestrator derives <strong>required research axes</strong> from the query shape, and the planner is required to cover them. Missing axes are injected deterministically rather than left to the model’s discretion.</p>",
    "<ul>",
      "<li>Non-quick modes always require the <code>evidence</code> and <code>criticism</code> axes.</li>",
      "<li><code>mechanism</code> is added for analytical, comparative or exploratory queries, or when recency matters.</li>",
      "<li><code>comparison</code> for comparative queries; <code>outlook</code> when recency-sensitive; <code>definition</code> for factual or low-complexity queries.</li>",
      "<li>Required axes are capped at 5; specialists at 5.</li>",
      "<li>A <code>counter_evidence</code> axis is <strong>always on</strong>, for every query.</li>",
    "</ul>",

    '<h3 id="orch-waves">Dependency waves</h3>',
    "<p>Contracts carry explicit dependencies, sanitized deterministically: dangling deps and self-references are dropped, each node keeps at most three deps, cycles are broken, and depth is capped at two — never more than three waves, as a latency floor.</p>",
    "<p>Within a wave, contracts run concurrently with <code>asyncio.gather</code>. Dependent contracts in later waves receive the accumulated findings of earlier waves as bounded grounding context — so “compare X vs Y” runs <em>after</em> “what is X” and can resolve references it could not from raw pages alone.</p>",
    '<div class="code-block" data-lang="waves"><div class="code-head"><span class="lang">execution waves</span><button class="copy-btn" type="button">', I.copy, " Copy</button></div><pre><code>wave 0  [ definition ]  [ evidence ]        <span class=\"tok-c\"># concurrent</span>\n              \\              /\n               v            v\nwave 1  [ comparison ]  [ mechanism ]       <span class=\"tok-c\"># receive wave-0 findings</span>\n                    |\n                    v\nwave 2  [ counter_evidence ]                <span class=\"tok-c\"># adversarial close-out</span></code></pre></div>",

    '<div class="callout info">',
      '<span class="c-ico">', I.control, "</span>",
      '<div class="c-body"><div class="c-title">Hardware cap</div>',
      "<p><code>MAX_PARALLEL_AGENTS</code> defaults to 3 on the 8&nbsp;GB host and is enforced at dispatch, not merely documented. <code>deep</code> and <code>executive</code> are the only modes permitted to lift the cap, and only with the budget governor armed.</p></div>",
    "</div>"
  ];

  /* ============================ Streaming ============================ */
  C.streaming = [
    '<p class="eyebrow">Architecture</p>',
    '<h2 id="streaming">Streaming events</h2>',
    '<p class="lede"><code>POST /api/research/stream</code> returns newline-delimited JSON (NDJSON). Each line is one event; the console renders them as a live research timeline.</p>',
    '<div class="code-block" data-lang="http"><div class="code-head"><span class="lang">http · request</span><button class="copy-btn" type="button">', I.copy, " Copy</button></div><pre><code>POST /api/research/stream\nContent-Type: application/json\n\n{ \"query\": \"Should Bangladesh expand nuclear energy?\", \"mode\": \"deep\" }</code></pre></div>",

    '<h3 id="stream-schema">Event schema</h3>',
    '<div class="card" style="padding:8px 0">',
      '<div class="event-row"><div class="e-name">progress</div><div class="e-desc"><code>request_id</code>, <code>message</code> — run accepted, identity assigned.</div></div>',
      '<div class="event-row"><div class="e-name">intent</div><div class="e-desc"><code>query_type</code>, <code>domain</code>, <code>explanation_level</code>, <code>ambiguity</code>, <code>senses</code>, <code>recommended_action</code>.</div></div>',
      '<div class="event-row"><div class="e-name">plan</div><div class="e-desc"><code>items</code> (sub-questions), <code>orchestration</code> (complexity, targets, clamping), <code>waves</code>.</div></div>',
      '<div class="event-row"><div class="e-name">search_progress</div><div class="e-desc"><code>snippets</code> — cumulative source count.</div></div>',
      '<div class="event-row"><div class="e-name">critic</div><div class="e-desc"><code>iteration</code>, <code>reason</code>, <code>breakdown</code> (confidence signals), <code>redteam</code>, <code>budget</code> (live ledger).</div></div>',
      '<div class="event-row"><div class="e-name">findings</div><div class="e-desc"><code>items</code> — claims with verification flags, scores and reasons. Re-emitted once with <code>verified_update: true</code> after verification.</div></div>',
      '<div class="event-row"><div class="e-name">final_report</div><div class="e-desc"><code>report</code>, <code>confidence</code>, <code>degraded</code>, <code>answer_support</code>, <code>budget</code>, <code>wave_report</code>, <code>citation_health</code>.</div></div>',
      '<div class="event-row"><div class="e-name">decisions</div><div class="e-desc"><code>items</code> — decision-layer options for strategic queries.</div></div>',
      '<div class="event-row"><div class="e-name">error</div><div class="e-desc"><code>message</code> — actionable cause (key, quota, timeout).</div></div>',
    "</div>",
    '<p class="small muted">A backend event type with no matching frontend case is silently dropped, so the NDJSON contract and the UI are treated as one contract and changed together.</p>'
  ];

  /* ============================ Agents ============================ */
  C.agents = [
    '<p class="eyebrow">Agents</p>',
    '<h2 id="agents">Agent roster</h2>',
    '<p class="lede">MARS operates as a temporary research organization. Each agent has one job, a deterministic fallback, and a specific module.</p>',
    '<div class="grid cols-2">',
      '<div class="feature"><div class="f-ico">', I.target, "</div><h4>Intent classifier</h4><p><code>agents/intent.py</code> — resolves meaning before research. Ranked senses, domain, explanation level, deterministic homonym fallback.</p></div>",
      '<div class="feature"><div class="f-ico">', I.brain, "</div><h4>Orchestrator</h4><p><code>agents/orchestrator.py</code> — LLM-free complexity scoring, target agents, required axes, mode recommendation and budget multiplier.</p></div>",
      '<div class="feature"><div class="f-ico">', I.flow, "</div><h4>Planner</h4><p><code>agents/planner.py</code> — delegation contracts, axis-coverage enforcement, dependency waves, primary-source query hints.</p></div>",
      '<div class="feature"><div class="f-ico">', I.search, "</div><h4>Scout / retrieval</h4><p><code>agents/search.py</code> — five providers under circuit breakers, dedup, domain diversity caps, page fetch and PDF extraction.</p></div>",
      '<div class="feature"><div class="f-ico">', I.book, "</div><h4>Specialists</h4><p><code>agents/summarizer.py</code> — eight role overlays (financial, technical, market, legal, scientific, policy, academic, general) extracting structured claims.</p></div>",
      '<div class="feature"><div class="f-ico">', I.shield, "</div><h4>Verifier</h4><p><code>agents/verifier.py</code> — deterministic per-claim checks: overlap, authority, numeric grounding, polarity, quote location, freshness.</p></div>",
      '<div class="feature"><div class="f-ico">', I.warn, "</div><h4>Critic &amp; Red Team</h4><p><code>agents/critic.py</code>, <code>agents/redteam.py</code> — sufficiency gate plus adversarial attack; survival score feeds confidence.</p></div>",
      '<div class="feature"><div class="f-ico">', I.scale, "</div><h4>Contradiction engine</h4><p><code>core/contradictions.py</code> — polarity, temporal and unit-aware numeric detectors with severity ranking.</p></div>",
      '<div class="feature"><div class="f-ico">', I.gauge, "</div><h4>Synthesizer</h4><p><code>agents/synthesizer.py</code> — cites claims, separates senses, mandates disambiguation, builds an answer-first outline.</p></div>",
      '<div class="feature"><div class="f-ico">', I.check, "</div><h4>Quality gate</h4><p><code>agents/answer_quality.py</code> — five-axis score, one bounded re-synthesis, disclosure of sub-threshold drafts.</p></div>",
    "</div>",

    '<h3 id="agents-fallback">The fallback contract</h3>',
    "<p>Every LLM-calling agent follows the same shape: try the model, validate the output, and on failure fall back to a deterministic default while recording the degradation. A new agent without a non-LLM fallback is a regression relative to the rest of the system.</p>",
    '<div class="code-block" data-lang="python"><div class="code-head"><span class="lang">python · the required shape</span><button class="copy-btn" type="button">', I.copy, " Copy</button></div><pre><code>NAME_SYSTEM_PROMPT = <span class=\"tok-y\">\"\"\"...\"\"\"</span>.strip()\n\n<span class=\"tok-g\">async def</span> name_agent(llm, ...) -&gt; ...:\n    <span class=\"tok-g\">try</span>:\n        payload = <span class=\"tok-g\">await</span> llm.generate_json(NAME_SYSTEM_PROMPT, user_prompt)\n    <span class=\"tok-g\">except</span> Exception <span class=\"tok-g\">as</span> exc:\n        logger.warning(<span class=\"tok-y\">\"[Name] LLM failed, using fallback\"</span>, exc_info=exc)\n        <span class=\"tok-g\">return</span> deterministic_fallback(...)\n\n    result = extract_and_validate(payload)\n    <span class=\"tok-g\">if not</span> result:\n        logger.warning(<span class=\"tok-y\">\"[Name] empty/invalid LLM output, using fallback\"</span>)\n        <span class=\"tok-g\">return</span> deterministic_fallback(...)\n    <span class=\"tok-g\">return</span> result</code></pre></div>",
    '<p class="small muted">The critic extends this pattern with query-type-aware gating: definitional shape is required only when the query is factual or definitional, so comparative and analytical queries are never forced into guaranteed expansion loops.</p>'
  ];

  /* ============================ Search ============================ */
  C.search = [
    '<p class="eyebrow">Agents</p>',
    '<h2 id="search">Search &amp; retrieval</h2>',
    '<p class="lede">Retrieval is type-driven: the contract’s <code>search_type</code> chooses which providers run, and every provider sits behind a circuit breaker.</p>',
    '<div class="table-wrap"><table>',
      "<thead><tr><th>Provider</th><th>Type</th><th>Used for</th><th>Breaker</th></tr></thead><tbody>",
        "<tr><td><strong>Tavily</strong></td><td>Web</td><td>General and news topics (optional paid key)</td><td>3 fails · 60&nbsp;s</td></tr>",
        "<tr><td><strong>DuckDuckGo text</strong></td><td>Web</td><td>General fallback when Tavily is absent</td><td>4 fails · 45&nbsp;s</td></tr>",
        "<tr><td><strong>DuckDuckGo news</strong></td><td>News</td><td>Recency-sensitive searches</td><td>4 fails · 45&nbsp;s</td></tr>",
        "<tr><td><strong>Wikipedia</strong></td><td>Reference</td><td>Encyclopedia and statistical contracts</td><td>4 fails · 45&nbsp;s</td></tr>",
        "<tr><td><strong>arXiv</strong></td><td>Preprint</td><td>Academic contracts</td><td>3 fails · 90&nbsp;s</td></tr>",
        "<tr><td><strong>Crossref</strong></td><td>Peer-reviewed</td><td>Academic contracts</td><td>3 fails · 90&nbsp;s</td></tr>",
      "</tbody>",
    "</table></div>",

    '<h3 id="search-dedup">Dedup and diversity</h3>',
    "<ul>",
      "<li><strong>Canonical-URL dedup</strong> plus a host blocklist (social media, marketplaces, content farms).</li>",
      "<li><strong>Near-duplicate snippets</strong> removed at 0.6 same-host / 0.85 cross-host overlap.</li>",
      "<li><strong>Domain diversity caps:</strong> Wikipedia contributes at most one result per domain, all other domains at most two.</li>",
      "<li><strong>Ranking</strong> blends authority, snippet richness, query relevance, primary-source bonus and recency — with a Wikipedia penalty.</li>",
    "</ul>",

    '<h3 id="search-freshness">Freshness half-lives</h3>',
    "<p>Freshness decays by search type, so a two-year-old market figure is not treated like a two-year-old encyclopedia entry.</p>",
    '<div class="table-wrap"><table>',
      '<thead><tr><th>Search type</th><th class="num">Half-life (days)</th></tr></thead><tbody>',
        '<tr><td>News</td><td class="num">120</td></tr>',
        '<tr><td>Statistical</td><td class="num">550</td></tr>',
        '<tr><td>Comparison / default</td><td class="num">730</td></tr>',
        '<tr><td>Academic</td><td class="num">1460</td></tr>',
        '<tr><td>Encyclopedia</td><td class="num">2200</td></tr>',
      "</tbody>",
    "</table></div>",
    "<p>Undated sources receive a mid-scale score of <code>0.45</code> rather than being treated as stale. Fetching is bulwarked — <code>MAX_PARALLEL_SEARCH=2</code> for searches, <code>MAX_PARALLEL_FETCH=4</code> for content — capped at 3&nbsp;MB per page, with clear block-page detection.</p>"
  ];

  /* ============================ Summarizer ============================ */
  C.summarizer = [
    '<p class="eyebrow">Agents</p>',
    '<h2 id="summarizer">Specialist extraction</h2>',
    '<p class="lede">The summarizer is not one generic prompt. It is eight specialists sharing one disciplined output contract.</p>',
    '<h3 id="sum-roles">Roles</h3>',
    "<p>The planner routes each contract to a specialist via its domain. Each role appends a focused overlay to the base extraction prompt:</p>",
    "<p>",
      '<span class="pill mars">financial</span> <span class="pill">technical</span> <span class="pill">market</span> <span class="pill">legal</span> ',
      '<span class="pill">scientific</span> <span class="pill">policy</span> <span class="pill">academic</span> <span class="pill">general</span>',
    "</p>",
    '<h3 id="sum-rules">Extraction rules</h3>',
    "<ul>",
      "<li>Rewrite, never copy. One fact per claim, maximum five claims per source.</li>",
      "<li>The source URL is copied exactly; unattributable claims are discarded.</li>",
      "<li>Numbers keep their unit, period and scope.</li>",
      "<li>Model confidence is banded, then blended with source authority: <code>0.70 × model + 0.25 × source</code>.</li>",
      "<li>Claims whose source reliability falls below <code>0.55</code> or whose query overlap falls below <code>0.15</code> are dropped.</li>",
    "</ul>",
    '<div class="callout">',
      '<span class="c-ico">', I.layers, "</span>",
      '<div class="c-body"><div class="c-title">Wave grounding and cache correctness</div>',
      "<p>Later-wave specialists receive up to five prior findings as a “prerequisite findings already established” block, so they do not re-extract what earlier waves settled. The extraction cache is keyed by role, sense and a digest of prerequisite findings — and <code>PROMPT_VERSION</code> is bumped on any change to claim shape, so stale claims can never be served from cache.</p></div>",
    "</div>"
  ];

  /* ============================ Verification ============================ */
  C.verification = [
    '<p class="eyebrow">Evidence &amp; Verification</p>',
    '<h2 id="verification">Claim verification</h2>',
    '<p class="lede">Every claim is checked against the source text it cites. The verifier is deterministic — no model call — so it runs on every claim, every pass, in microseconds.</p>',

    '<h3 id="ver-checks">The hard checks</h3>',
    '<div class="table-wrap"><table>',
      "<thead><tr><th>Check</th><th>What it catches</th><th>Threshold</th></tr></thead><tbody>",
        "<tr><td><strong>Weighted lexical overlap</strong></td><td>Topical mismatch — claim not actually in the source</td><td>≥ 0.35 (digits ×1.6, long terms ×1.5)</td></tr>",
        "<tr><td><strong>Source authority</strong></td><td>Unreliable publisher</td><td>≥ 0.55</td></tr>",
        "<tr><td><strong>Numeric grounding</strong></td><td>Fabricated figures</td><td>2% relative tolerance, unit/scale aware</td></tr>",
        "<tr><td><strong>Polarity consistency</strong></td><td>Negation inversion (“X” vs “not X”)</td><td>best-sentence overlap ≥ 0.55</td></tr>",
        "<tr><td><strong>Quote location</strong></td><td>Fabricated direct quotes</td><td>quote must appear in source text</td></tr>",
        "<tr><td><strong>Freshness</strong></td><td>Stale evidence presented as current</td><td>flagged when score &lt; 0.25</td></tr>",
      "</tbody>",
    "</table></div>",
    "<p>Output is a per-claim <code>verification_score</code> plus <code>verified</code>, <code>verification_reason</code>, and a compact summary — total, verified, rate, top failure reasons, stale count, numeric failures, polarity failures and unreachable sources. Truncated claims are dropped, not verified.</p>",

    '<div class="callout mint">',
      '<span class="c-ico">', I.check, "</span>",
      '<div class="c-body"><div class="c-title">Why deterministic</div>',
      "<p>A verifier that costs a model call is a verifier you consult less often. The deterministic checks run at roughly 153&nbsp;µs per claim and measured <strong>F1 = 1.00</strong> with <strong>0% hallucination leak</strong> on the labeled benchmark set — fabricated numbers, invented facts and retraction claims were all rejected.</p></div>",
    "</div>",

    '<h3 id="ver-why-not">Why “not” is never a stopword</h3>',
    "<p>The negation token is weighted like a number (2.00). This single decision keeps “X” and “not X” from merging in dedup, keeps them inside the contradiction band, and keeps citation support from counting a negation as an affirmation. It was found by the benchmark suite, not by intuition.</p>"
  ];

  /* ============================ Contradictions ============================ */
  C.contradictions = [
    '<p class="eyebrow">Evidence &amp; Verification</p>',
    '<h2 id="contradictions">Contradiction detection</h2>',
    '<p class="lede">MARS does not blend disagreement into an average. It surfaces the conflict, names the values, and feeds a resolution follow-up back into research.</p>',
    "<p>Three detectors share one semantic engine and emit one output shape. They run in a deliberate order:</p>",
    '<div class="pipeline">',
      '<div class="pipe-node"><div class="pn-ico">1</div><div class="pn-body"><div class="pn-title">Polarity</div><div class="pn-desc">Checked first, outside the similarity band. Fires when both claims carry non-zero, opposing polarity. Fixed severity 0.55.</div></div></div>', ARROW,
      '<div class="pipe-node"><div class="pn-ico">2</div><div class="pn-body"><div class="pn-title">Temporal</div><div class="pn-desc">Same measure across different periods. Severity scales with divergence.</div></div></div>', ARROW,
      '<div class="pipe-node"><div class="pn-ico">3</div><div class="pn-body"><div class="pn-title">Numeric (unit-aware)</div><div class="pn-desc">Inside the similarity band, with 5% relative divergence. Disjoint scope (geography, population) is downgraded to a scope conflict rather than a false numeric one.</div></div></div>',
    "</div>",
    "<p>Results are sorted by severity descending (numeric &gt; polarity &gt; temporal, divergence-scaled) and capped at five. Same-source pairs are skipped. Resolved conflicts — where the difference is period, scope or metric rather than substance — are recorded with their explanation so the report is honest about the spread without mislabelling it as an open disagreement.</p>",
    '<div class="callout info">',
      '<span class="c-ico">', I.info, "</span>",
      '<div class="c-body"><div class="c-title">Why contradictions penalize instead of blending</div>',
      "<p>A weighted average lets three strong sources mask one direct lie. The confidence penalty is pool-size scaled: one conflict among three facts means a third of the evidence disagrees; among thirty it is one stale page.</p></div>",
    "</div>"
  ];

  /* ============================ Citations ============================ */
  C.citations = [
    '<p class="eyebrow">Evidence &amp; Verification</p>',
    '<h2 id="citations">Citations &amp; health</h2>',
    '<p class="lede">Two independent questions are asked of every citation: does the sentence match the source’s evidence, and does the link still resolve?</p>',

    '<h3 id="cit-live">Live URL re-validation</h3>',
    "<p>At delivery, each cited URL in the report’s legend is probed:</p>",
    "<ol>",
      "<li>A <code>HEAD</code> request with redirects followed.</li>",
      "<li>On failure or a <code>403/405/501</code>, a ranged <code>GET</code> for the first 2&nbsp;KB only — never a full page download.</li>",
      "<li>A <code>200–399</code> response is reachable; a changed final URL is marked redirected.</li>",
    "</ol>",
    "<p>Probes run under a concurrency cap of six with a five-second timeout and an identifying user-agent, and are never fatal to the run.</p>",

    '<h3 id="cit-verdicts">Per-source verdicts</h3>',
    "<p>Link health is fused with sentence-support evidence into a single verdict per source:</p>",
    '<div class="table-wrap"><table>',
      "<thead><tr><th>Verdict</th><th>Meaning</th></tr></thead><tbody>",
        '<tr><td><span class="pill mint">ok</span></td><td>URL reachable and all citing sentences supported.</td></tr>',
        '<tr><td><span class="pill amber">warn</span></td><td>Reachable, but some citing sentences are not supported by its evidence.</td></tr>',
        '<tr><td><span class="pill">broken</span></td><td>URL unreachable, but sentence support was fine.</td></tr>',
        '<tr><td><span class="pill bad">bad</span></td><td>Unreachable <em>and</em> under-supported.</td></tr>',
        '<tr><td><span class="pill">unchecked</span></td><td>Non-HTTP or skipped URL.</td></tr>',
      "</tbody>",
    "</table></div>",
    "<p>Broken and bad citations produce an explicit limitations line; warnings are disclosed as partially unsupported. The console shows a citation-health row with the <code>ok / warn / broken / bad / unchecked</code> counts.</p>",

    '<h3 id="cit-support">Answer support</h3>',
    "<p>Separately from link health, the synthesizer’s sentences are checked against the evidence of the source they cite using the shared semantic engine (<code>SUPPORT_THRESHOLD = 0.30</code>). The resulting support rate feeds both the confidence engine and the answer-quality gate.</p>"
  ];

  /* ============================ Confidence ============================ */
  C.confidence = [
    '<p class="eyebrow">Evidence &amp; Verification</p>',
    '<h2 id="confidence">Confidence engine</h2>',
    '<p class="lede">Confidence is computed from measured evidence, not self-reported by a model. Nine signals combine into a score, with a pool-size-scaled contradiction penalty subtracted.</p>',

    '<h3 id="conf-signals">Signals and weights</h3>',
    '<div class="card" style="padding:18px 22px">',
      '<div class="signal-row"><div class="s-label">Citation coverage</div><div class="s-bar"><div class="s-fill" style="width:100%"></div></div><div class="s-val">0.25</div></div>',
      '<div class="signal-row"><div class="s-label">Source quality</div><div class="s-bar"><div class="s-fill" style="width:80%"></div></div><div class="s-val">0.20</div></div>',
      '<div class="signal-row"><div class="s-label">Source diversity</div><div class="s-bar"><div class="s-fill" style="width:60%"></div></div><div class="s-val">0.15</div></div>',
      '<div class="signal-row"><div class="s-label">Claim verification strength</div><div class="s-bar"><div class="s-fill" style="width:60%"></div></div><div class="s-val">0.15</div></div>',
      '<div class="signal-row"><div class="s-label">Cross-source agreement</div><div class="s-bar"><div class="s-fill" style="width:60%"></div></div><div class="s-val">0.15</div></div>',
      '<div class="signal-row"><div class="s-label">Critic survival</div><div class="s-bar"><div class="s-fill" style="width:40%"></div></div><div class="s-val">0.10</div></div>',
      '<div class="signal-row"><div class="s-label">Citation support</div><div class="s-bar"><div class="s-fill" style="width:40%"></div></div><div class="s-val">carved</div></div>',
      '<div class="signal-row"><div class="s-label">Axis coverage</div><div class="s-bar"><div class="s-fill" style="width:20%"></div></div><div class="s-val">carved</div></div>',
      '<div class="signal-row"><div class="s-label">Freshness</div><div class="s-bar"><div class="s-fill" style="width:20%"></div></div><div class="s-val">carved</div></div>',
    "</div>",
    '<p class="small muted">The three conditional signals carve weight from existing ones when their inputs are present, so historical scores stay comparable when they are absent. Freshness is measured only when publish dates parse.</p>',

    '<h3 id="conf-penalty">Contradiction penalty</h3>',
    "<p>Conflicts are subtracted, not averaged, and scaled by pool size (<code>scale = min(2.0, max(1.0, 6.0 / fact_count))</code>):</p>",
    "<ul>",
      "<li>Severe conflict (severity ≥ 0.60): <strong>−0.06</strong> each</li>",
      "<li>Moderate conflict: <strong>−0.03</strong> each</li>",
      "<li>Total penalty capped at <strong>−0.18</strong></li>",
      "<li>Resolved conflicts and intra-source conflicts are skipped</li>",
    "</ul>",

    '<h3 id="conf-degraded">Degraded-run cap</h3>',
    "<p>If the summarizer or synthesizer fell back to deterministic extraction, or any provider degraded, the overall confidence is capped at <strong>0.55</strong> — below the 0.75 sufficiency threshold. This is the decisive guard against a full LLM outage silently producing a confident-looking report.</p>",
    '<div class="callout warn">',
      '<span class="c-ico">', I.warn, "</span>",
      '<div class="c-body"><div class="c-title">Why this exists</div>',
      "<p>A live run once failed every LLM call and “completed” on extractive fallbacks with off-topic sentences — yet scored 0.772, labelled “High.” The cap, the disclosed <code>degraded</code> flag and the UI banner all exist because self-verifying fallback evidence inflates every lexical signal by construction.</p></div>",
    "</div>",
    '<div class="grid cols-2" style="margin-top:18px">',
      '<div class="metric-card"><div class="m-val">0.823</div><div class="m-lab">Calibration · strong evidence</div></div>',
      '<div class="metric-card"><div class="m-val">0.283</div><div class="m-lab">Calibration · weak evidence</div></div>',
    "</div>",
    '<p class="small muted" style="margin-top:12px">Confidence calibration measured <strong>100% in expected band</strong> with monotonic ordering on the labeled set.</p>'
  ];

  /* ============================ Budget ============================ */
  C.budget = [
    '<p class="eyebrow">Operations</p>',
    '<h2 id="budget">Cost &amp; budget</h2>',
    '<p class="lede">Every run has a budget, a live ledger, and a governor that refuses work the run cannot pay for.</p>',
    '<h3 id="bud-ceilings">Four ceilings</h3>',
    '<div class="table-wrap"><table>',
      '<thead><tr><th>Ceiling</th><th class="num">Default</th><th>Setting</th></tr></thead><tbody>',
        '<tr><td>Estimated cost</td><td class="num">$0.50</td><td><code>max_budget_usd</code></td></tr>',
        '<tr><td>Tokens</td><td class="num">400,000</td><td><code>max_budget_tokens</code></td></tr>',
        '<tr><td>LLM calls</td><td class="num">60</td><td><code>max_llm_calls</code></td></tr>',
        '<tr><td>Wall-clock seconds</td><td class="num">300</td><td><code>max_research_seconds</code></td></tr>',
      "</tbody>",
    "</table></div>",
    "<p>Mode multipliers scale all four ceilings together. The governor’s <code>can_afford_pass</code> estimate uses a 3.9&nbsp;chars-per-token ratio and estimates input, output and call counts before approving an expansion pass.</p>",

    '<h3 id="bud-ledger">The run ledger</h3>',
    "<p>The ledger rides a <code>ContextVar</code>, so concurrent runs never share state. It tracks tokens and USD by stage, LLM calls, search calls, cache hits and misses, wall time, and a worst-case utilization fraction. Cache hits record tokens but charge <strong>$0</strong> — a hit refunds USD while keeping token accounting honest.</p>",
    "<p>Search cost is not negligible when it is paid: Tavily is charged at <code>$0.008</code> per search unit; the free providers are zero. LLM pricing is per-model and prefix-matched, with a conservative default of <code>(0.0004, 0.0008)</code> USD per 1K input/output tokens.</p>",
    '<div class="callout info">',
      '<span class="c-ico">', I.info, "</span>",
      '<div class="c-body"><div class="c-title">The ledger is a control input, not just a meter</div>',
      "<p>The depth controller consults the ledger before every expansion, and the live snapshot streams to the UI on each critic event. Cost-aware reasoning is part of the loop, not a post-hoc report.</p></div>",
    "</div>"
  ];

  /* ============================ Stopping ============================ */
  C.stopping = [
    '<p class="eyebrow">Operations</p>',
    '<h2 id="stopping">Depth &amp; stopping</h2>',
    '<p class="lede">MARS must know when to stop. The depth controller is pure and stateless, and decides expand versus finalize from measured state — never from a model’s optimism.</p>',
    '<h3 id="stop-order">Decision order</h3>',
    "<ol>",
      "<li><strong>Hard walls first.</strong> Budget exhausted or iteration ceiling reached → finalize immediately.</li>",
      "<li><strong>Minimum depth.</strong> Mode minimum iterations not yet reached → expand.</li>",
      "<li><strong>Evidence completeness.</strong> Uncovered axes, active high-impact uncorroborated claims, severe contradictions or thin dimensions → expand if novel follow-ups exist; otherwise finalize and disclose gaps as limitations.</li>",
      "<li><strong>Critic signal.</strong> Insufficient with novel follow-ups → expand.</li>",
      "<li><strong>Soft stops.</strong> Sufficiency (critic sufficient, or confidence at target with no weak axes); two consecutive marginal-gain stalls; or no novel queries remain.</li>",
    "</ol>",
    "<p>Marginal gain tracks confidence deltas across iterations against <code>min_marginal_gain = 0.03</code>. Novel-query memory uses Jaccard similarity ≥ 0.8, so a rephrasing of an already-issued query is recognised as not novel.</p>",
    '<div class="table-wrap"><table>',
      '<thead><tr><th>Mode</th><th class="num">Minimum iterations</th></tr></thead><tbody>',
        "<tr><td><code>quick</code></td><td class=\"num\">1</td></tr>",
        "<tr><td><code>redteam</code></td><td class=\"num\">1</td></tr>",
        "<tr><td><code>audit</code></td><td class=\"num\">2</td></tr>",
      "</tbody>",
    "</table></div>",
    "<p class=\"small muted\">When a stop is driven by a wall rather than sufficiency, the reason is written into the report’s Limitations section — an early stop is disclosed, never hidden.</p>"
  ];

  /* ============================ Reliability ============================ */
  C.reliability = [
    '<p class="eyebrow">Operations</p>',
    '<h2 id="reliability">Reliability</h2>',
    '<p class="lede">The system is designed to fail loudly, cheaply and in the right direction.</p>',

    '<h3 id="rel-provider">Provider chain</h3>',
    "<p>Generation tries providers in order: a UI-selected OpenAI-compatible provider first, then Groq, then HuggingFace. A UI-selected active provider is exclusive by default — if it fails, the run raises rather than silently falling back to a different model. HuggingFace itself walks a short internal candidate list on 404/410.</p>",

    '<h3 id="rel-breakers">Circuit breakers and timeouts</h3>',
    "<ul>",
      "<li>Three consecutive failures open a provider breaker for 60&nbsp;s; a single per-attempt timeout opens it immediately.</li>",
      "<li>A 429 does <strong>not</strong> open the breaker — throttling is not an outage.</li>",
      "<li>Auth, payment and permanent errors (<code>401/402/403</code>, permanent <code>400</code>, <code>404</code>) fail fast and are never retried.</li>",
      "<li>Timeouts are never retried; they move straight to the next provider.</li>",
      "<li>Retry-After hints are honored but capped at 3&nbsp;s.</li>",
    "</ul>",
    "<p>Two timeout budgets exist because the two worlds differ: <code>llm_timeout_sec = 25</code> for Groq and HuggingFace, and <code>custom_llm_timeout_sec = 60</code> for slower OpenAI-compatible hosts. A hard wall-clock cap at 1.5× the base guards against drip-feeding proxies.</p>",

    '<h3 id="rel-concurrency">Concurrency as a reliability feature</h3>',
    "<p><code>MAX_PARALLEL_LLM=2</code> exists because concurrent large prompts are exactly what exhausts free-tier token quotas — observed live when three parallel summarizer calls burned a 200k-token daily budget. The semaphore bounds in-flight prompts, not threads.</p>",

    '<h3 id="rel-preflight">Pre-flight probe</h3>',
    "<p>Before a run starts, the LLM client probes every configured provider in parallel. If none is reachable, the run fails in seconds with per-provider reasons instead of degrading for minutes. Successful probes are cached for 30&nbsp;s; failures are not cached, so a quota reset is noticed immediately.</p>",

    '<h3 id="rel-patterns">Failure patterns the codebase guards against</h3>',
    "<ul>",
      "<li><strong>Silent fallback hiding a real bug.</strong> Every broad exception logs with context; a <code>NameError</code> once hid for the entire history of the planner behind a bare <code>except</code>.</li>",
      "<li><strong>Mutating module-level state.</strong> Per-request state is scoped to the request coroutine; there is no unbounded request-keyed global.</li>",
      "<li><strong>Cache poisoning on failure.</strong> An agent’s empty result is never cached — the summarizer once poisoned its cache key for a full TTL.</li>",
      "<li><strong>Stale code serving a verification run.</strong> Live verifications assert the request id appears in a fresh backend log.</li>",
    "</ul>"
  ];

  /* ============================ Persistence ============================ */
  C.persistence = [
    '<p class="eyebrow">Operations</p>',
    '<h2 id="persistence">Persistence &amp; trace</h2>',
    '<p class="lede">SQLite in WAL mode, auto-initialising schema, additive migrations only. Every run leaves a complete, replayable record.</p>',
    '<div class="table-wrap"><table>',
      "<thead><tr><th>Table</th><th>Purpose</th></tr></thead><tbody>",
        "<tr><td><code>research_runs</code></td><td>One row per run: query, complexity, agent count, status, cost, confidence, iterations.</td></tr>",
        "<tr><td><code>agent_tasks</code></td><td>Planned contracts for the run.</td></tr>",
        "<tr><td><code>sources</code></td><td>Retrieved sources, deduped by URL, with reliability score.</td></tr>",
        "<tr><td><code>evidence</code></td><td>Raw snippets that claims derive from.</td></tr>",
        "<tr><td><code>claims</code></td><td>Extracted claims with verification flag, confidence, agent and challenged state.</td></tr>",
        "<tr><td><code>verification_results</code></td><td>Per-claim score and reason.</td></tr>",
        "<tr><td><code>contradictions</code></td><td>Detected conflict pairs.</td></tr>",
        "<tr><td><code>critic_reviews</code></td><td>Every critic pass, with confidence breakdown and improved queries.</td></tr>",
        "<tr><td><code>citations</code></td><td>Parsed legend markers, domains and URLs.</td></tr>",
        "<tr><td><code>decisions</code></td><td>Decision-layer options.</td></tr>",
        "<tr><td><code>final_reports</code></td><td>Canonical markdown report keyed by run id.</td></tr>",
        "<tr><td><code>agent_events</code></td><td>Node timings, retries and fallbacks — the replay timeline.</td></tr>",
        "<tr><td><code>research_reports</code></td><td>Legacy flat store, kept in sync for existing readers.</td></tr>",
        "<tr><td><code>llm_providers</code></td><td>Saved providers with Fernet-encrypted keys and key hints.</td></tr>",
      "</tbody>",
    "</table></div>",
    "<p>Schema changes are additive and idempotent — <code>CREATE TABLE IF NOT EXISTS</code> and <code>ALTER TABLE ... ADD COLUMN</code> guarded by <code>PRAGMA table_info</code>. The WAL journal allows concurrent readers alongside the writer, and a busy timeout prevents lock errors.</p>",

    '<h3 id="persist-replay">Research replay</h3>',
    "<p><code>GET /api/research/{run_id}/trace</code> reconstructs the full ordered timeline — plan, sources, claims, verification, contradictions, citations, every critic review and decisions. The UI renders a read-only research card from the trace alone, with no re-fetching. A failed or timed-out run can be resumed at the critic node from its persisted evidence rather than re-running the whole pipeline.</p>",

    '<h3 id="persist-security">Security posture</h3>',
    "<ul>",
      "<li>Provider keys are Fernet-encrypted at rest (<code>MARS_SECRET_KEY</code> or an auto-created 0600 <code>.mars_secret</code>) and never serialised to the UI.</li>",
      "<li>The service binds to <code>127.0.0.1</code> by default; do not expose it without an auth layer.</li>",
      "<li>The research endpoint is rate-limited (<code>RATE_LIMIT=5/minute</code> by default).</li>",
      "<li>Secrets are never logged; provider probe failures log only the exception type.</li>",
    "</ul>"
  ];

  /* ============================ API ============================ */
  C.api = [
    '<p class="eyebrow">Operations</p>',
    '<h2 id="api">API reference</h2>',
    '<p class="lede">A small, focused HTTP surface. The streaming endpoint is the heart of the system.</p>',

    '<h3 id="api-research">Research</h3>',
    '<div class="card" style="padding:6px 0">',
      '<div class="endpoint"><span class="method post">POST</span><span class="path">/api/research/stream</span><span class="ep-desc">Start a run; NDJSON event stream</span></div>',
      '<div class="endpoint"><span class="method post">POST</span><span class="path">/api/research/{run_id}/resume</span><span class="ep-desc">Resume a failed/timed-out run from its checkpoint</span></div>',
      '<div class="endpoint"><span class="method get">GET</span><span class="path">/api/research/{run_id}/trace</span><span class="ep-desc">Full replay reconstruction of a run</span></div>',
    "</div>",

    '<h3 id="api-providers">Providers</h3>',
    '<div class="card" style="padding:6px 0">',
      '<div class="endpoint"><span class="method get">GET</span><span class="path">/api/providers</span><span class="ep-desc">Saved providers (keys masked) + active id</span></div>',
      '<div class="endpoint"><span class="method post">POST</span><span class="path">/api/providers</span><span class="ep-desc">Add an OpenAI-compatible provider</span></div>',
      '<div class="endpoint"><span class="method put">PUT</span><span class="path">/api/providers/{id}</span><span class="ep-desc">Partial update; omit key to keep stored key</span></div>',
      '<div class="endpoint"><span class="method del">DEL</span><span class="path">/api/providers/{id}</span><span class="ep-desc">Delete a provider</span></div>',
      '<div class="endpoint"><span class="method post">POST</span><span class="path">/api/providers/{id}/active</span><span class="ep-desc">Set the single active provider</span></div>',
      '<div class="endpoint"><span class="method post">POST</span><span class="path">/api/providers/active/clear</span><span class="ep-desc">Fall back to the env/Groq/HF chain</span></div>',
      '<div class="endpoint"><span class="method post">POST</span><span class="path">/api/providers/{id}/test</span><span class="ep-desc">Single-shot latency probe</span></div>',
    "</div>",

    '<h3 id="api-health">Health</h3>',
    '<div class="card" style="padding:6px 0">',
      '<div class="endpoint"><span class="method get">GET</span><span class="path">/api/health</span><span class="ep-desc">Liveness check</span></div>',
    "</div>",

    '<h3 id="api-request">Stream request body</h3>',
    '<div class="code-block" data-lang="json"><div class="code-head"><span class="lang">json</span><button class="copy-btn" type="button">', I.copy, " Copy</button></div><pre><code>{\n  \"query\": \"Should Bangladesh expand nuclear energy?\",   <span class=\"tok-c\">// 5-500 chars</span>\n  \"mode\": \"standard\",                                     <span class=\"tok-c\">// quick|standard|deep|executive|audit|redteam</span>\n  \"deep_research\": false                                  <span class=\"tok-c\">// opt-in fan-out</span>\n}</code></pre></div>"
  ];

  /* ============================ Benchmarks ============================ */
  C.benchmarks = [
    '<p class="eyebrow">Operations</p>',
    '<h2 id="benchmarks">Benchmarks</h2>',
    '<p class="lede">Deterministic, network-free measurement against hand-labelled fixtures. Every number is reproducible with <code>python bench/run_offline.py</code>.</p>',
    '<div class="grid cols-3">',
      '<div class="metric-card"><div class="m-val acc">F1 1.00</div><div class="m-lab">Verification accuracy</div><div class="m-sub">8 TP · 0 FP · 0 FN · 8 TN</div></div>',
      '<div class="metric-card"><div class="m-val acc">0.000</div><div class="m-lab">Hallucination leak rate</div><div class="m-sub">6/6 fabricated claims rejected</div></div>',
      '<div class="metric-card"><div class="m-val acc">1.00</div><div class="m-lab">Contradiction F1</div><div class="m-sub">numeric · polarity · temporal</div></div>',
      '<div class="metric-card"><div class="m-val">100%</div><div class="m-lab">Confidence calibration</div><div class="m-sub">in expected band, monotonic</div></div>',
      '<div class="metric-card"><div class="m-val">66.7%</div><div class="m-lab">LLM cache hit ratio</div><div class="m-sub">6.26× repeat-pass speedup</div></div>',
      '<div class="metric-card"><div class="m-val">0.709</div><div class="m-lab">Semantic Spearman ρ</div><div class="m-sub">dedup F1 0.75 @ 0.86</div></div>',
    "</div>",

    '<h3 id="bench-perf">Performance</h3>',
    '<div class="table-wrap"><table>',
      '<thead><tr><th>Operation</th><th class="num">Latency</th></tr></thead><tbody>',
        '<tr><td>Single claim verification</td><td class="num">152.9 µs</td></tr>',
        '<tr><td>Pair similarity (single)</td><td class="num">140.7 µs</td></tr>',
        '<tr><td>Answer-support check (40 sentences)</td><td class="num">0.4 ms</td></tr>',
        '<tr><td>Cross-similarity (40 × 200)</td><td class="num">16.0 ms</td></tr>',
        '<tr><td>Similarity matrix (60 facts)</td><td class="num">365.7 ms</td></tr>',
        '<tr><td>Similarity matrix (200 facts)</td><td class="num">924.8 ms</td></tr>',
      "</tbody>",
    "</table></div>",
    "<p class=\"small muted\">Peak-RSS delta across a 300-fact matrix workload was 21.8&nbsp;MB — the semantic engine fits comfortably in the 8&nbsp;GB budget alongside the rest of the stack.</p>",

    '<h3 id="bench-limits">Known limitations, measured not hidden</h3>',
    "<ul>",
      "<li><strong>Lexical, not embedding-level.</strong> Paraphrases such as “doubled over the past decade” versus “nearly doubled in the last ten years” score below the dedup threshold. Dedup recall on the labeled set is 0.60 — but precision is 1.00: it never merges what it should not.</li>",
      "<li><strong>Offline suite scope.</strong> The deterministic suite exercises measurable components and a scripted pipeline; live-model wording quality is measured by <code>bench/run_live.py</code> with real provider keys.</li>",
      "<li><strong>Cache speedup.</strong> The committed run reports 6.26× on a repeat-heavy workload; README quotes 6.7× from a separate run. Both share the same 66.7% hit ratio.</li>",
    "</ul>",

    '<div class="callout mint">',
      '<span class="c-ico">', I.check, "</span>",
      '<div class="c-body"><div class="c-title">Decisions are made on measurements</div>',
      "<p>Several early bugs — the negation-word dedup merge, the electrical-transformer contamination, the off-topic limitations list — were found by writing a failing benchmark case first, then fixing the code. The benchmark suite is the regression net, not a report card.</p></div>",
    "</div>"
  ];

  window.MARS_DOCS = { nav: NAV, content: C, icon: I };
})();
