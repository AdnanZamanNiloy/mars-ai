"""Corroboration MATCHING against realistic (non-verbatim) evidence.

Confirmed live failure across 5 expansion passes: queries_issued 3-8,
results_returned 20-62, new_publishers 9-56, matched_corroboration 0 — the
only failing stage was `new_publishers -> matched_corroboration`. The
deterministic tests passed only because their mocked page contained the claim
verbatim; real corroborating pages paraphrase it and scored ~0.30 on the
shared semantic engine, far below the 0.55 band.

These tests pin the fix: a candidate is matched on the combined
title+snippet+retained-excerpt text via EITHER the semantic band OR the
numeric/rare-anchor path, while same-publisher and different-claim pages stay
excluded (precision over recall).
"""
import app.graph.workflow as wf
from app.core.config import Settings
from app.core.evidence_grade import (
    CORROBORATION_ANCHOR_MIN_CONTENT_OVERLAP,
    find_corroborating_sources,
    grade_claim,
    registrable_domain,
)

_CLAIM = "Global AI capital expenditure reached $200 billion in 2025"
_REUTERS = "https://www.reuters.com/tech/ai-capex"
_OECD = "https://www.oecd.org/ai/investment-outlook"

# Deliberately NOT verbatim: the page paraphrases the claim ("Spending ...
# hit about $200 billion last year"). Pair similarity is ~0.30, so only the
# anchor path can match it.
_OECD_TITLE = "OECD AI investment outlook"
_OECD_SNIPPET = (
    "Spending on AI infrastructure hit about $200 billion last year, "
    "per OECD figures."
)


def _settings(**kw):
    return Settings(groq_api_key="k", _env_file=None, **kw)


def _candidate(url, title, snippet, content=""):
    return {"url": url, "title": title, "snippet": snippet, "content": content}


# ---------------------------------------------------------------------------
# 1. The live-failure regression: a non-verbatim new-publisher page matches.
# ---------------------------------------------------------------------------

def test_non_verbatim_numeric_evidence_matches_via_anchor_path():
    candidates = [_candidate(_OECD, _OECD_TITLE, _OECD_SNIPPET)]
    matches = find_corroborating_sources(_CLAIM, candidates, [_REUTERS])
    assert matches == [_OECD], (
        "a topically-correct paraphrase with the matching quantity must "
        "corroborate a quantitative claim"
    )

    fact = {"claim": _CLAIM, "source": _REUTERS, "verified": True}
    if matches:
        from app.core.evidence_grade import apply_corroboration

        assert apply_corroboration(fact, matches[0]) is True
    record = grade_claim(fact)
    assert record.corroboration_count >= 2
    assert record.needs_corroboration is False


def test_retained_excerpt_alone_can_match():
    """The verifier replaces full content with a bounded excerpt; matching must
    use that excerpt, not only the (short) snippet."""
    candidates = [
        {
            "url": _OECD,
            "title": _OECD_TITLE,
            "snippet": "AI investment outlook.",
            "content": "",
            "corroboration_excerpt": _OECD_SNIPPET,
        }
    ]
    assert find_corroborating_sources(_CLAIM, candidates, [_REUTERS]) == [_OECD]


def test_title_snippet_anchor_branch_matches_headline_evidence():
    """Thin body, headline-shaped evidence: strong rare-anchor overlap in the
    title+snippet alone is enough."""
    candidates = [_candidate(
        _OECD,
        "Global AI capital expenditure $200 billion",
        "2025 figure",
    )]
    assert find_corroborating_sources(_CLAIM, candidates, [_REUTERS]) == [_OECD]


# ---------------------------------------------------------------------------
# 2. Precision: a different claim on the same topic is NOT corroboration.
# ---------------------------------------------------------------------------

def test_different_number_is_not_corroboration():
    candidates = [_candidate(
        "https://www.oecd.org/ai/other",
        "OECD AI investment outlook",
        "Spending on AI infrastructure hit about $300 billion last year.",
    )]
    assert find_corroborating_sources(_CLAIM, candidates, [_REUTERS]) == []


def test_wrong_magnitude_with_high_lexical_overlap_is_not_corroboration():
    """The numeric gate comes first: a page sharing the claim's whole
    vocabulary but quoting a DIFFERENT figure must never corroborate it — that
    is the numeric contradiction the engine exists to catch, not agreement."""
    candidates = [_candidate(
        "https://www.oecd.org/ai/other",
        "Global AI capital expenditure outlook",
        "Global AI capital expenditure reached about $300 billion last year.",
    )]
    assert find_corroborating_sources(_CLAIM, candidates, [_REUTERS]) == []


def test_number_only_page_without_topic_anchors_is_not_corroboration():
    assert CORROBORATION_ANCHOR_MIN_CONTENT_OVERLAP >= 2
    # Shares the scale phrase but a different quantity: the numeric anchor is
    # not grounded, so the page cannot corroborate the claim.
    candidates = [_candidate(
        "https://www.example.org/x",
        "Quarterly results",
        "Revenue of $500 billion was reported.",
    )]
    assert find_corroborating_sources(_CLAIM, candidates, [_REUTERS]) == []


# ---------------------------------------------------------------------------
# 3. Independence: same registrable domain is never counted.
# ---------------------------------------------------------------------------

def test_same_registrable_domain_never_matches():
    candidates = [_candidate(
        "https://news.reuters.com/ai/capex-2025", _OECD_TITLE, _OECD_SNIPPET
    )]
    assert find_corroborating_sources(_CLAIM, candidates, [_REUTERS]) == []

    existing = {
        registrable_domain(_REUTERS),
        registrable_domain("https://news.reuters.com/ai/capex-2025"),
    }
    assert len(existing) == 1


# ---------------------------------------------------------------------------
# 4. End-to-end mocked graph: non-verbatim expansion hit lifts a pending claim.
# ---------------------------------------------------------------------------

class _FakeSearch:
    """Returns a NON-verbatim new-publisher page for corroboration queries and
    the (same-publisher) primary hit otherwise."""

    def __init__(self, settings, primary_url, corroboration_url):
        self.settings = settings
        self.primary_url = primary_url
        self.corroboration_url = corroboration_url
        self.calls = []

    async def run_search(self, queries):
        out = []
        for q in queries or []:
            text = q[0] if isinstance(q, tuple) else str(q)
            self.calls.append(text)
            if "-site:reuters.com" in text.lower() or "oecd" in text.lower():
                out.append({
                    "url": self.corroboration_url,
                    "title": _OECD_TITLE,
                    "snippet": _OECD_SNIPPET,
                    "content": _OECD_SNIPPET,
                    "sub_question": text,
                })
            else:
                out.append({
                    "url": self.primary_url,
                    "title": "Reuters AI capex",
                    "snippet": f"{_CLAIM}.",
                    "content": f"{_CLAIM}.",
                    "sub_question": text,
                })
        return out


async def test_mocked_graph_reaches_corroborated_ge2_non_verbatim(monkeypatch):
    from app.graph.workflow import build_initial_state, create_workflow

    settings = _settings()
    search = _FakeSearch(settings, _REUTERS, _OECD)

    class FakeLLM:
        def __init__(self):
            self.settings = settings

        async def generate_json(self, *a, **k):
            return {}

    async def fake_planner(llm, query, critique_feedback="", today="", **kwargs):
        return [{
            "id": 1, "question": "What is the current trend of AI?",
            "axis": "evidence", "search_type": "news", "priority": 1,
            "depends_on": [], "coverage_goal": "", "domain": "general",
            "minimum_sources": 1, "stop_condition": "enough",
        }]

    async def fake_summarizer(llm, query, search_results=None,
                              specialist_role="general", prior_findings=None):
        return [{"claim": _CLAIM, "source": _REUTERS, "confidence": 0.9}]

    async def fake_critic(llm, query, facts=None, iteration=1,
                          max_iterations=3, contradictions=None, **kwargs):
        return {"is_sufficient": True, "reason": "ok",
                "improved_queries": [], "confidence": 0.9}

    async def fake_synthesizer(llm, query, facts, context=None):
        return "synthesized answer"

    monkeypatch.setattr(wf, "planner_agent", fake_planner)
    monkeypatch.setattr(wf, "summarizer_agent", fake_summarizer)
    monkeypatch.setattr(wf, "critic_agent", fake_critic)
    monkeypatch.setattr(wf, "synthesizer_agent", fake_synthesizer)
    monkeypatch.setattr(wf, "verify_facts", lambda facts, search_results: facts)

    state = build_initial_state("What is the current trend of AI?", 3, mode="standard")
    workflow = create_workflow(llm=FakeLLM(), search_client=search)

    final = None
    async for snapshot in workflow.astream(state, stream_mode="values"):
        final = {**(final or {}), **snapshot}

    assert final is not None
    facts = final.get("facts") or []
    corroborated = [f for f in facts if grade_claim(f).corroboration_count >= 2]
    assert corroborated, (
        "the non-verbatim OECD expansion hit must lift the Reuters claim to "
        f"two independent domains; facts={facts}"
    )
    assert sum(1 for f in facts if grade_claim(f).corroboration_count >= 2) > 0
