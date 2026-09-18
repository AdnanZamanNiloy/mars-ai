/* ============================================================================
   MARS Documentation — runtime.
   Renders the navigation, pages, client-side search, scroll-spy, theme
   toggle and copy buttons from the MARS_DOCS content model.
   ========================================================================== */
(function () {
  "use strict";

  var DOCS = window.MARS_DOCS || { nav: [], content: {}, icon: {} };
  var NAV = DOCS.nav || [];
  var CONTENT = DOCS.content || {};

  /* Flatten nav into pages for search + routing. */
  var PAGES = [];
  NAV.forEach(function (group) {
    group.items.forEach(function (item) {
      PAGES.push({ id: item.id, title: item.title, group: group.group, icon: item.icon });
    });
  });

  var contentInner = document.getElementById("contentInner");
  var navGroups = document.getElementById("navGroups");
  var sidenav = document.getElementById("sidenav");
  var scrim = document.getElementById("scrim");
  var searchInput = document.getElementById("searchInput");
  var searchResults = document.getElementById("searchResults");
  var menuToggle = document.getElementById("menuToggle");
  var navClose = document.getElementById("navClose");

  var activeId = null;
  var activeSearchIndex = -1;
  var searchIndexData = [];

  /* ---------------------------------------------------------------- nav */
  function renderNav() {
    var html = "";
    NAV.forEach(function (group) {
      html += '<div class="nav-group"><p class="nav-group-title">' + group.group + "</p>";
      group.items.forEach(function (item) {
        html +=
          '<a class="nav-link" data-id="' + item.id + '" href="#' + item.id + '">' +
          '<span class="n-ico">' + item.icon + "</span>" + item.title +
          "</a>";
      });
      html += "</div>";
    });
    navGroups.innerHTML = html;
    navGroups.addEventListener("click", function (e) {
      var link = e.target.closest(".nav-link");
      if (link) closeNav();
    });
  }

  /* --------------------------------------------------------------- pages */
  var PAGE_BY_ID = {};
  PAGES.forEach(function (p) { PAGE_BY_ID[p.id] = p; });

  function renderPage(id) {
    var body = CONTENT[id];
    if (!body) return;
    var page = PAGE_BY_ID[id];
    // Visually-hidden H1 keeps each page's heading hierarchy valid for
    // assistive tech without disturbing the editorial type scale.
    var h1 = page ? '<h1 class="sr-only">' + escapeHtml(page.title) + "</h1>" : "";
    contentInner.innerHTML = h1 + body.join("\n");
    bindCopyButtons();
    activeId = id;
    updateActiveNav();
    window.scrollTo(0, 0);
  }

  function updateActiveNav() {
    navGroups.querySelectorAll(".nav-link").forEach(function (link) {
      link.classList.toggle("active", link.getAttribute("data-id") === activeId);
    });
  }

  /* ------------------------------------------------------- copy buttons */
  function bindCopyButtons() {
    contentInner.querySelectorAll(".copy-btn").forEach(function (btn) {
      if (btn.dataset.bound) return;
      btn.dataset.bound = "1";
      btn.addEventListener("click", function () {
        var block = btn.closest(".code-block");
        var codeEl = block ? block.querySelector("code") : null;
        if (!codeEl) return;
        var text = codeEl.innerText;
        var done = function () {
          var original = btn.innerHTML;
          btn.classList.add("done");
          btn.innerHTML = DOCS.icon.check + " Copied";
          setTimeout(function () {
            btn.classList.remove("done");
            btn.innerHTML = original;
          }, 1600);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text, done); });
        } else {
          fallbackCopy(text, done);
        }
      });
    });
  }

  function fallbackCopy(text, done) {
    try {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      done();
    } catch (e) { /* clipboard unavailable — silently ignore */ }
  }

  /* ------------------------------------------------------- mobile nav */
  function openNav() {
    sidenav.classList.add("open");
    scrim.classList.add("on");
    scrim.tabIndex = 0;
  }
  function closeNav() {
    sidenav.classList.remove("open");
    scrim.classList.remove("on");
    scrim.tabIndex = -1;
  }
  if (menuToggle) menuToggle.addEventListener("click", openNav);
  if (navClose) navClose.addEventListener("click", closeNav);
  if (scrim) scrim.addEventListener("click", closeNav);

  /* ---------------------------------------------------------- theme */
  /* The console owns the toggle; docs just reflect the shared setting.
   * Boot reads the persisted key; postMessage live-syncs an open iframe. */
  function applyTheme(mode) {
    var next = mode === "light" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("mars-docs-theme", next); } catch (e) { /* ignore */ }
  }
  window.addEventListener("message", function (e) {
    if (e.data && e.data.type === "mars-theme") applyTheme(e.data.mode);
  });
  applyTheme(document.documentElement.getAttribute("data-theme") || "dark");

  /* --------------------------------------------------------- search */
  function buildSearchIndex() {
    PAGES.forEach(function (page) {
      var html = (CONTENT[page.id] || []).join(" ");
      var text = html
        .replace(/<[^>]*>/g, " ")
        .replace(/&nbsp;/g, " ")
        .replace(/&amp;/g, "&")
        .replace(/&rsquo;|&#39;/g, "'")
        .replace(/&ldquo;|&rdquo;|&quot;/g, '"')
        .replace(/&lt;/g, "<")
        .replace(/&gt;/g, ">")
        .replace(/\s+/g, " ")
        .trim();
      searchIndexData.push({ page: page, text: text, lower: text.toLowerCase() });
    });
  }

  function runSearch(query) {
    var q = query.trim().toLowerCase();
    if (!q) return [];
    var terms = q.split(/\s+/).filter(Boolean);
    var results = [];
    searchIndexData.forEach(function (entry) {
      var lower = entry.lower;
      var score = 0;
      var matched = true;
      terms.forEach(function (t) {
        var idx = lower.indexOf(t);
        if (idx === -1) { matched = false; return; }
        score += 10 - Math.min(9, idx / 200);
        if (entry.page.title.toLowerCase().indexOf(t) !== -1) score += 25;
      });
      if (matched) results.push({ page: entry.page, score: score });
    });
    results.sort(function (a, b) { return b.score - a.score; });
    return results.slice(0, 8);
  }

  function renderSearchResults(results, query) {
    activeSearchIndex = -1;
    if (!results.length) {
      searchResults.innerHTML = '<div class="sr-empty">No matches for “' + escapeHtml(query) + '”</div>';
      searchResults.classList.add("open");
      return;
    }
    searchResults.innerHTML = results.map(function (r) {
      return '<a class="sr-item" data-id="' + r.page.id + '">' +
        '<div class="sr-title">' + r.page.title + "</div>" +
        '<div class="sr-path">' + r.page.group + "</div></a>";
    }).join("");
    searchResults.classList.add("open");
    searchResults.querySelectorAll(".sr-item").forEach(function (item) {
      item.addEventListener("click", function () {
        closeSearch();
        renderPage(item.getAttribute("data-id"));
        history.replaceState(null, "", "#" + item.getAttribute("data-id"));
      });
    });
  }

  function moveSearchSelection(delta) {
    var items = searchResults.querySelectorAll(".sr-item");
    if (!items.length) return;
    activeSearchIndex = (activeSearchIndex + delta + items.length) % items.length;
    items.forEach(function (it, i) { it.classList.toggle("active", i === activeSearchIndex); });
    items[activeSearchIndex].scrollIntoView({ block: "nearest" });
  }

  function closeSearch() {
    searchResults.classList.remove("open");
    activeSearchIndex = -1;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  if (searchInput) {
    searchInput.addEventListener("input", function () {
      var q = searchInput.value;
      if (!q.trim()) { closeSearch(); return; }
      renderSearchResults(runSearch(q), q);
    });
    searchInput.addEventListener("keydown", function (e) {
      if (e.key === "ArrowDown") { e.preventDefault(); moveSearchSelection(1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); moveSearchSelection(-1); }
      else if (e.key === "Enter") {
        e.preventDefault();
        var items = searchResults.querySelectorAll(".sr-item");
        var target = activeSearchIndex >= 0 ? items[activeSearchIndex] : items[0];
        if (target) {
          closeSearch();
          renderPage(target.getAttribute("data-id"));
          history.replaceState(null, "", "#" + target.getAttribute("data-id"));
          searchInput.blur();
        }
      } else if (e.key === "Escape") {
        searchInput.value = "";
        closeSearch();
        searchInput.blur();
      }
    });
    document.addEventListener("click", function (e) {
      if (!e.target.closest(".search-wrap")) closeSearch();
    });
  }

  /* Keyboard shortcuts: "/" focuses search, "Esc" closes overlays. */
  document.addEventListener("keydown", function (e) {
    var tag = (e.target.tagName || "").toLowerCase();
    var typing = tag === "input" || tag === "textarea";
    if (e.key === "/" && !typing) {
      e.preventDefault();
      searchInput.focus();
    } else if (e.key === "Escape") {
      closeSearch();
      closeNav();
    }
  });

  /* ---------------------------------------------------- routing */
  function currentId() {
    var h = (window.location.hash || "").replace(/^#\/?/, "");
    return CONTENT[h] ? h : PAGES[0].id;
  }

  window.addEventListener("hashchange", function () {
    var id = currentId();
    if (id !== activeId) renderPage(id);
  });

  /* ----------------------------------------------------- boot */
  renderNav();
  buildSearchIndex();
  renderPage(currentId());
})();
