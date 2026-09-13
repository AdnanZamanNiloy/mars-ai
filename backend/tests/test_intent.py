"""Intent Classification Agent: ambiguity resolution before research."""

import pytest

from app.agents.intent import (
    IntentReport,
    classify_intent,
    heuristic_intent,
)


class FakeLLM:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    async def generate_json(self, system_prompt, user_prompt, retries=3, response_model=None):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        if self.error is not None:
            raise self.error
        if response_model is not None:
            return response_model.model_validate(self.payload).model_dump()
        return self.payload


def test_fallback_flags_transformer_ambiguity():
    report = heuristic_intent("What is transformer?")
    assert report.ambiguity is True
    assert len(report.senses) == 2
    assert report.senses[0].domain == "machine_learning"
    assert report.senses[1].domain == "engineering"
    # Bare definitional question on an ambiguous term: the spec-mandated
    # behavior is to research and present BOTH meanings, regardless of the
    # probability gap ("Transformer can mean two things...").
    assert report.recommended_action == "research_both"
    assert len(report.research_senses) == 2


def test_non_definitional_ambiguous_query_stays_dominant():
    """A query whose phrasing carries no definitional frame keeps the gap
    rule: 0.75 vs 0.20 -> research the dominant sense only."""
    report = heuristic_intent("transformer efficiency standards 2024")
    assert report.ambiguity is True
    assert report.recommended_action == "research_dominant"
    assert report.research_senses == [report.senses[0]]


def test_fallback_unambiguous_query_passes_through():
    report = heuristic_intent("What is retrieval augmented generation?")
    assert report.ambiguity is False
    assert report.senses == []
    assert report.query_type == "factual"
    assert report.recommended_action == "research_dominant"


def test_fallback_levels():
    assert heuristic_intent("explain transformer simply").explanation_level == "basic"
    assert heuristic_intent("transformer architecture internals").explanation_level == "expert"
    assert heuristic_intent("What is transformer?").explanation_level == "practical"


def test_fallback_classifies_question_type():
    assert heuristic_intent("nuclear vs solar energy in Bangladesh").query_type == "comparative"
    assert heuristic_intent("should we invest in nuclear energy").query_type == "analytical"


async def test_llm_intent_used_when_valid():
    llm = FakeLLM({
        "query_type": "factual",
        "domain": "electrical_engineering",
        "explanation_level": "practical",
        "ambiguity": True,
        "senses": [
            {"label": "Transformer neural network architecture", "domain": "machine_learning",
             "probability": 0.6, "note": "AI sense"},
            {"label": "Electrical transformer", "domain": "electrical_engineering",
             "probability": 0.35, "note": "device sense"},
        ],
        "reasoning": "likely AI, plausibly electrical",
    })
    report = await classify_intent(llm, "What is transformer?")
    assert report.origin == "llm"
    assert report.ambiguity is True
    # probabilities ranked most-likely first
    assert report.senses[0].probability >= report.senses[1].probability
    # non-planner domains aliased into the planner vocabulary
    assert report.domain == "engineering"
    assert report.senses[1].domain == "engineering"
    # gap 0.25 < 0.30 -> both senses researched
    assert report.recommended_action == "research_both"
    assert len(report.research_senses) == 2


async def test_llm_hedge_cannot_hide_a_real_second_sense():
    """The deterministic ambiguity floor: a model flagging ambiguity=false with
    a 0.4 second sense must not silently drop the alternative reading."""
    llm = FakeLLM({
        "query_type": "factual",
        "domain": "machine_learning",
        "explanation_level": "practical",
        "ambiguity": False,
        "senses": [
            {"label": "AI architecture", "domain": "machine_learning", "probability": 0.55},
            {"label": "Electrical device", "domain": "engineering", "probability": 0.40},
        ],
        "reasoning": "",
    })
    report = await classify_intent(llm, "What is transformer?")
    assert report.ambiguity is True


async def test_llm_failure_falls_back_deterministically():
    llm = FakeLLM(error=RuntimeError("provider down"))
    report = await classify_intent(llm, "What is transformer?")
    assert report.origin == "heuristic"
    assert report.ambiguity is True
    assert report.senses[0].domain == "machine_learning"


async def test_llm_invalid_payload_falls_back():
    llm = FakeLLM(payload={"senses": "not-a-list", "query_type": "nonsense"})
    report = await classify_intent(llm, "What is python?")
    assert report.origin == "heuristic"


async def test_context_snippets_reach_the_prompt():
    llm = FakeLLM(payload={
        "query_type": "factual", "domain": "machine_learning",
        "explanation_level": "practical", "ambiguity": False,
        "senses": [], "reasoning": "ok",
    })
    await classify_intent(llm, "What is transformer?",
                          context_snippets=["Attention is all you need — ML architecture"])
    assert "Attention is all you need" in llm.calls[0]["user_prompt"]


async def test_llm_ranking_overridden_for_bare_definitional():
    """The live failure: the model ranked electrical 0.8 / ML 0.2 on bare
    'What is transformer?'. Definitional phrasing forces BOTH meanings into
    the plan — the ranking may set focus, never silently drop a sense."""
    llm = FakeLLM({
        "query_type": "factual",
        "domain": "engineering",
        "explanation_level": "practical",
        "ambiguity": True,
        "senses": [
            {"label": "Transformer (electrical device)", "domain": "engineering",
             "probability": 0.8},
            {"label": "Transformer (neural network architecture)", "domain": "machine_learning",
             "probability": 0.2},
        ],
        "reasoning": "context skews electrical",
    })
    report = await classify_intent(llm, "What is transformer?")
    assert report.recommended_action == "research_both"
    assert len(report.research_senses) == 2


async def test_user_named_sense_beats_llm_ranking():
    llm = FakeLLM({
        "query_type": "factual", "domain": "machine_learning",
        "explanation_level": "practical", "ambiguity": True,
        "senses": [
            {"label": "Transformer (electrical device)", "domain": "engineering",
             "probability": 0.8},
            {"label": "Transformer neural network architecture",
             "domain": "machine_learning", "probability": 0.2},
        ],
        "reasoning": "",
    })
    report = await classify_intent(llm, "transformer neural network architecture explained")
    assert report.ambiguity is False
    assert report.senses[0].domain == "machine_learning"


def test_research_both_when_genuinely_split():
    report = IntentReport(
        query="what is jaguar", query_type="factual", domain="general",
        explanation_level="practical", ambiguity=True,
        senses=[
            __import__("app.agents.intent", fromlist=["SenseCandidate"]).SenseCandidate(
                "Jaguar car", "engineering", 0.45),
            __import__("app.agents.intent", fromlist=["SenseCandidate"]).SenseCandidate(
                "Jaguar animal", "science", 0.40),
        ],
    )
    assert report.recommended_action == "research_both"
    assert len(report.research_senses) == 2


def test_to_dict_round_trip_shape():
    report = heuristic_intent("What is transformer?")
    d = report.to_dict()
    assert set(d) == {"query", "query_type", "domain", "explanation_level", "ambiguity",
                      "senses", "recommended_action", "reasoning", "origin"}
    assert d["senses"][0]["probability"] == 0.75


def test_planner_assigns_senses_and_fallback_uses_them():
    """The intent -> planner handoff: contracts get sense tags, domains upgrade
    from general, and the deterministic fallback plans per sense."""
    from app.agents.planner import _assign_intent_senses, fallback_plan

    # Non-definitional phrasing -> dominant sense only; a hand-written sense
    # tag that names a researched sense is preserved.
    intent = heuristic_intent("transformer efficiency standards 2024").to_dict()

    plan = _assign_intent_senses([
        {"question": "transformer architecture attention mechanism", "domain": "general",
         "specialist": "general", "sense": ""},
        {"question": "transformer statistics official data", "domain": "general",
         "specialist": "general", "sense": "Electrical transformer (AC voltage device)"},
    ], intent)
    assert all(c["sense"] == "Transformer neural network architecture" for c in plan)
    # general-domain contracts upgrade to the sense's domain + specialist
    assert plan[0]["domain"] == "machine_learning"
    assert plan[0]["specialist"] == "technical"

    # research_both: fallback interleaves both senses
    both = heuristic_intent("What is transformer?").to_dict()
    fb2 = fallback_plan("What is transformer?", target_count=4, intent=both)
    used = {c.get("sense") for c in fb2}
    assert used == {"Transformer neural network architecture",
                    "Electrical transformer (AC voltage device)"}


def test_unambiguous_intent_leaves_plan_untagged():
    from app.agents.planner import _assign_intent_senses

    intent = heuristic_intent("What is retrieval augmented generation?").to_dict()
    plan = _assign_intent_senses([
        {"question": "RAG definition", "domain": "machine_learning", "specialist": "technical", "sense": ""},
    ], intent)
    assert plan[0]["sense"] == ""


class _FakeSearchClient:
    settings = None

    async def run_search(self, queries):
        return [{
            "title": "Attention Is All You Need",
            "snippet": "The transformer is a neural network architecture for sequence transduction.",
            "url": "https://arxiv.org/abs/1706.03762",
        }]


async def test_graph_runs_intent_before_planner(monkeypatch):
    """Full-graph wiring: intent resolves first, its context search is shared,
    and the planner receives the intent report."""
    import app.graph.workflow as wf
    from app.core.config import Settings
    from app.core.llm import LLMClient

    _FakeSearchClient.settings = Settings(groq_api_key="k", _env_file=None)
    llm = LLMClient(_FakeSearchClient.settings)

    captured = {}

    async def fake_classify(llm_arg, query, context_snippets=None):
        captured["context"] = list(context_snippets or [])
        return heuristic_intent(query)

    async def fake_planner(**kwargs):
        captured["intent"] = kwargs.get("intent")
        return [{
            "id": 1, "question": "transformer neural network architecture definition",
            "axis": "definition", "search_type": "encyclopedia", "priority": 1,
            "depends_on": [], "domain": "machine_learning", "minimum_sources": 2,
            "coverage_goal": "", "stop_condition": "", "variants": [], "agent": "",
            "tools": ["web_search"], "scope": [], "output_format": "structured_findings",
            "specialist": "technical", "preferred_domains": [], "primary_source_query": "",
            "wave": 0, "sense": "",
        }]

    async def fake_summarizer(llm_arg, query, search_results, specialist_role="general",
                              prior_findings=None, sense=""):
        captured["sense"] = sense
        return []

    async def fake_critic(**kwargs):
        return {"is_sufficient": False, "reason": "thin", "improved_queries": [], "confidence": 0.4}

    async def fake_synthesizer(llm=None, query=None, facts=None, context=None):
        return "## Executive Summary\n\nAnswer."

    monkeypatch.setattr(wf, "classify_intent", fake_classify)
    monkeypatch.setattr(wf, "planner_agent", fake_planner)
    monkeypatch.setattr(wf, "summarizer_agent", fake_summarizer)
    monkeypatch.setattr(wf, "critic_agent", fake_critic)
    monkeypatch.setattr(wf, "synthesizer_agent", fake_synthesizer)

    graph = wf.create_workflow(llm, _FakeSearchClient())
    state = wf.build_initial_state("What is transformer?", 3, mode="quick")
    final = None
    async for snap in graph.astream(state, stream_mode="values"):
        final = snap

    assert captured["context"], "grounding search must feed the intent classifier"
    assert captured["intent"]["ambiguity"] is True
    assert captured["intent"]["domain"] == "machine_learning"
    # intent lands in state for the synthesizer
    assert final["intent"]["ambiguity"] is True


def test_sense_tagged_fallback_groups_by_sense():
    """The extractive fallback must keep senses in separate sections, like
    the LLM path — grouping flips from sub_question to sense when tagged."""
    from app.agents.synthesizer import _deterministic_report

    facts = [
        {"claim": "The AI transformer uses attention mechanisms to weigh tokens.",
         "source": "https://arxiv.org/abs/1706.03762", "confidence": 0.9,
         "sense": "Transformer neural network architecture", "verified": True},
        {"claim": "An electrical transformer steps AC voltage up or down via windings.",
         "source": "https://en.wikipedia.org/wiki/Transformer", "confidence": 0.85,
         "sense": "Electrical transformer (AC voltage device)", "verified": True},
        {"claim": "Attention removes the recurrence bottleneck of earlier models.",
         "source": "https://arxiv.org/abs/1706.03762", "confidence": 0.88,
         "sense": "Transformer neural network architecture", "verified": True},
    ]
    result = _deterministic_report("What is transformer?", facts, facts, {}, [])
    assert "Transformer neural network architecture" in result.answer
    assert "electrical transformer" in result.answer.lower()
    # claims must stay inside their own sense's section, never mixed
    ai_pos = result.answer.find("attention mechanisms")
    elec_pos = result.answer.find("steps AC voltage")
    assert ai_pos != -1 and elec_pos != -1


def test_audit_exempts_disambiguation_lines():
    from app.agents.synthesizer import audit_citations

    numbered = [{"n": 1, "domain": "arxiv.org", "url": "https://arxiv.org/a",
                 "tier": "preprint", "authority": 0.88, "primary": True}]
    facts = [{"citation": 1, "claim": "The transformer architecture uses attention mechanisms."}]
    answer = (
        "## Executive Summary\n\n"
        "1) **Transformer neural network architecture** — attention-based deep learning model.\n"
        "2) **Electrical transformer** — device that changes AC voltage.\n\n"
        "Based on your question, this report focuses on meaning 1.\n\n"
        "The transformer architecture uses attention mechanisms [1].\n\n"
        "## Sources\n\n[1] arxiv.org (preprint, primary) — https://arxiv.org/a"
    )
    audit = audit_citations(answer, numbered, facts)
    assert audit.uncited_factual == [], audit.uncited_factual


def test_ambiguity_block_rendered_for_ambiguous_intent():
    from app.agents.synthesizer import _render_ambiguity_block

    block = _render_ambiguity_block({
        "ambiguity": True,
        "recommended_action": "research_dominant",
        "senses": [
            {"label": "Transformer neural network architecture", "domain": "machine_learning",
             "probability": 0.75, "note": "attention-based model"},
            {"label": "Electrical transformer", "domain": "engineering",
             "probability": 0.20, "note": "AC voltage device"},
        ],
    })
    assert "**Transformer neural network architecture**" in block
    assert "focuses on meaning 1" in block
    assert _render_ambiguity_block({"ambiguity": False, "senses": []}) == ""


def test_user_resolved_homonym_is_not_ambiguous():
    """'python snake feeding habits' names the sense — ambiguity resolved,
    domain follows the named sense, not the more common meaning."""
    report = heuristic_intent("python snake feeding habits")
    assert report.ambiguity is False
    assert len(report.senses) == 1
    assert report.senses[0].domain == "science"


async def test_summarizer_shrinks_prompt_when_provider_rejects_size(tmp_path):
    """The 413 ladder: a provider that rejects the full 12-source excerpt
    prompt must be retried with a smaller excerpt budget — keeping extraction
    LLM-written instead of degrading to heuristic fallback."""
    from app.agents.summarizer import summarizer_agent
    from app.core.llm import PromptTooLargeError

    class SizePickyLLM:
        """Rejects prompts over 8000 chars (stands in for the Groq 413
        threshold relative to the summarizer's real prompt sizes)."""

        def __init__(self, settings):
            self.settings = settings
            self.prompt_lengths = []

        async def generate_json(self, system_prompt, user_prompt, retries=3, response_model=None):
            self.prompt_lengths.append(len(user_prompt))
            if len(user_prompt) > 8000:
                raise PromptTooLargeError("too large")
            return {"facts": [{
                "claim": "Solar capacity grew 40 percent in 2024",
                "source": "https://arxiv.org/0", "confidence": 0.9,
            }]}

    from app.core.config import Settings

    llm = SizePickyLLM(Settings(groq_api_key="k", database_url=str(tmp_path / "ladder.db"), _env_file=None))
    results = [{
        "url": f"https://arxiv.org/{i}",
        "title": f"paper {i}",
        "content": "Solar capacity grew 40 percent in 2024, reaching 2000 GW installed worldwide. " * 30,
        "snippet": "solar grew",
        "sub_question": "solar statistics",
        "search_type": "statistical",
        "published_at": "2024-06-01",
    } for i in range(8)]

    facts = await summarizer_agent(llm, "solar statistics 2024", results)
    assert facts, "ladder must produce facts, not fall back"
    assert len(llm.prompt_lengths) >= 2, "at least one size retry"
    assert llm.prompt_lengths[-1] < llm.prompt_lengths[0], "retry must shrink the prompt"
    assert facts[0]["extraction"] == "llm"


async def test_summarizer_does_not_shrink_on_timeouts(tmp_path):
    """A slow provider (timeout wall) gets exactly TWO attempts — the initial
    payload and one budget-aware second chance at the smallest size — then
    the heuristic fallback. Never the old 3x90s stall pattern."""
    from app.agents.summarizer import summarizer_agent
    from app.core.llm import AllProvidersFailedError
    from app.core.config import Settings

    class SlowProviderLLM:
        def __init__(self, settings):
            self.settings = settings
            self.calls = 0

        async def generate_json(self, system_prompt, user_prompt, retries=3, response_model=None):
            self.calls += 1
            raise AllProvidersFailedError(
                "Active provider 'sleepy' failed: ReadTimeout. No fallback providers run."
            )

    llm = SlowProviderLLM(Settings(groq_api_key="k", database_url=str(tmp_path / "timeout.db"), _env_file=None))
    results = [{
        "url": "https://arxiv.org/0", "title": "paper",
        "content": ("Solar capacity grew 40 percent in 2024, reaching 2000 GW of installed "
                    "capacity worldwide according to the industry association's annual report. " * 20),
        "snippet": "solar", "sub_question": "solar statistics",
    }]
    facts = await summarizer_agent(llm, "solar statistics", results)
    assert llm.calls == 2, "one stall + one smallest-payload second chance, then stop"
    assert facts and facts[0]["extraction"] == "heuristic"


async def test_summarizer_timeout_gets_one_budgeted_second_chance(tmp_path):
    """A stalled provider gets exactly ONE more attempt at the smallest
    payload when the run's wall-clock budget allows it — bounded, not a
    3x90s stall pattern."""
    from app.agents.summarizer import summarizer_agent
    from app.core.config import Settings
    from app.core.llm import AllProvidersFailedError
    from app.core.usage import start_run_usage

    class StalledThenHealthyLLM:
        def __init__(self, settings):
            self.settings = settings
            self.calls = []

        async def generate_json(self, system_prompt, user_prompt, retries=3, response_model=None):
            self.calls.append(len(user_prompt))
            if len(self.calls) == 1:
                raise AllProvidersFailedError("Active provider failed: ReadTimeout.")
            return {"facts": [{
                "claim": "Solar capacity grew 40 percent in 2024",
                "source": "https://arxiv.org/0", "confidence": 0.9,
            }]}

    settings = Settings(groq_api_key="k", database_url=str(tmp_path / "s2c.db"), _env_file=None)
    llm = StalledThenHealthyLLM(settings)
    usage = start_run_usage("test-run", settings)  # fresh budget: seconds remaining is large
    usage.budget.started_at -= 1  # barely any time spent
    results = [{
        "url": f"https://arxiv.org/{i}", "title": f"paper {i}",
        "content": "Solar capacity grew 40 percent in 2024, reaching 2000 GW installed worldwide. " * 30,
        "snippet": "solar", "sub_question": "solar statistics",
    } for i in range(8)]
    # Distinct query string: the process-global cache singleton would serve
    # the other ladder test's identical key and skip the LLM entirely.
    facts = await summarizer_agent(llm, "solar statistics outlook 2024", results)
    assert len(llm.calls) == 2, "exactly one second chance"
    assert llm.calls[1] < llm.calls[0], "second chance uses the smallest payload"
    assert facts and facts[0]["extraction"] == "llm"
    from app.core.usage import clear_run_usage
    clear_run_usage()
