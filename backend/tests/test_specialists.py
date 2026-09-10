"""Specialist Workforce tests (Phase 3.1).

DoD: a financial sub-question and a general sub-question get measurably
different system prompts (observable via captured LLM calls), and each
specialist path provably uses its own AgentContext — never the full state.
"""
import asyncio

import app.graph.workflow as wf
from app.core.config import Settings
from app.core.isolation import AgentContext, build_contexts, specialist_role_for_domain
from app.core.llm import LLMClient


def test_domain_to_role_routing():
    assert specialist_role_for_domain("economics") == "financial"
    assert specialist_role_for_domain("machine_learning") == "technical"
    assert specialist_role_for_domain("software") == "technical"
    assert specialist_role_for_domain("general") == "general"
    assert specialist_role_for_domain("philosophy") == "general"


def test_specialist_prompts_differ_by_role():
    from app.agents.summarizer import specialist_system_prompt, SUMMARIZER_SYSTEM_PROMPT

    general = specialist_system_prompt("general")
    financial = specialist_system_prompt("financial")
    technical = specialist_system_prompt("technical")

    assert financial != general != technical
    assert financial.startswith(SUMMARIZER_SYSTEM_PROMPT)
    assert "FINANCIAL" in financial
    assert "TECHNICAL" in technical
    # All share the same output contract (the JSON schema block).
    for prompt in (general, financial, technical):
        assert '"claim"' in prompt
        assert "Return ONLY valid JSON" in prompt


async def test_financial_and_general_get_different_prompts_end_to_end():
    """Pipeline-level: the summarizer receives the role derived from each
    sub-question's contract, with each context scoped to its own results."""
    captured = []

    state = {
        "query": "compare energy policy and ML benchmarks",
        "sub_questions": [
            {"id": 1, "question": "energy financing costs", "axis": "comparison", "search_type": "statistical",
             "priority": 1, "depends_on": [], "coverage_goal": "", "domain": "economics",
             "minimum_sources": 2, "stop_condition": "enough"},
            {"id": 2, "question": "ML benchmark methodology", "axis": "definition", "search_type": "academic",
             "priority": 1, "depends_on": [], "coverage_goal": "", "domain": "machine_learning",
             "minimum_sources": 2, "stop_condition": "enough"},
        ],
        "search_results": [
            {"url": "https://imf.org/a", "sub_question": "energy financing costs", "content": "costs", "snippet": "s"},
            {"url": "https://arxiv.org/b", "sub_question": "ML benchmark methodology", "content": "bench", "snippet": "s"},
        ],
        "facts": [],
        "critique": {},
        "iteration": 0,
        "max_iterations": 3,
        "confidence_history": [],
    }

    async def fake_summarizer(llm, query, search_results=None, specialist_role="general"):
        captured.append({
            "role": specialist_role,
            "urls": sorted(r["url"] for r in (search_results or [])),
        })
        return []

    async def fake_critic(llm, query, facts=None, iteration=1, max_iterations=3, contradictions=None):
        return {"is_sufficient": True, "reason": "ok", "improved_queries": [], "confidence": 0.9}

    async def fake_planner(llm, query, critique_feedback="", today=""):
        return state["sub_questions"]

    async def fake_synthesizer(llm, query, facts=None):
        return "answer"

    settings = Settings(groq_api_key="k", _env_file=None)

    class Stub:
        SEARCH = state["search_results"]
        async def run_search(self, questions):
            return self.SEARCH

    monkey_patches = fake_summarizer, fake_critic, fake_planner, fake_synthesizer
    wf.summarizer_agent, wf.critic_agent, wf.planner_agent, wf.synthesizer_agent = monkey_patches

    workflow = wf.create_workflow(LLMClient(settings), Stub())
    async for _ in workflow.astream(state, stream_mode="values"):
        pass

    roles = {c["role"] for c in captured}
    assert "financial" in roles and "technical" in roles, captured
    # Each specialist only saw its own results (2.9 isolation preserved).
    for c in captured:
        if c["role"] == "financial":
            assert c["urls"] == ["https://imf.org/a"]
        if c["role"] == "technical":
            assert c["urls"] == ["https://arxiv.org/b"]


def test_context_specialist_role_accessor():
    ctx = build_contexts(
        [{"question": "q", "axis": "a", "domain": "economics"}],
        [{"url": "https://imf.org/x", "sub_question": "q"}],
    )[0]
    assert isinstance(ctx, AgentContext)
    assert ctx.specialist_role() == "financial"
    assert ctx.domain() == "economics"


def test_new_specialist_routing_and_overlays():
    from app.agents.summarizer import specialist_system_prompt, SUMMARIZER_SYSTEM_PROMPT
    from app.core.isolation import SPECIALIST_ROLES, specialist_role_for_domain

    assert specialist_role_for_domain("legal") == "legal"
    assert specialist_role_for_domain("policy") == "policy"
    assert specialist_role_for_domain("academic") == "academic"
    assert specialist_role_for_domain("science") == "scientific"
    assert specialist_role_for_domain("philosophy") == "general"
    assert SPECIALIST_ROLES == {"financial", "technical", "market", "general",
                                "legal", "scientific", "policy", "academic"}
    for role, marker in (("legal", "LEGAL"), ("scientific", "SCIENTIFIC"),
                         ("policy", "POLICY"), ("academic", "ACADEMIC")):
        prompt = specialist_system_prompt(role)
        assert marker in prompt, role
        assert prompt.startswith(SUMMARIZER_SYSTEM_PROMPT)


def test_contract_fields_validated():
    import asyncio

    from app.agents.planner import planner_agent

    class FakeLLM:
        async def generate_json(self, system_prompt, user_prompt, retries=3, response_model=None):
            payload = {
                "query_type": "factual", "query_scope": "narrow", "dominant_domain": "general",
                "sub_questions": [{
                    "id": 1, "question": "What do courts require for valid consent forms?",
                    "axis": "definition", "search_type": "academic", "priority": 1,
                    "depends_on": [], "coverage_goal": "doctrine", "domain": "legal",
                    "agent": "legal_researcher", "tools": ["web_search", "nonsense_tool", "fetch_content"],
                    "scope": ["consent", "case law", "statutes", "remedies", "extra"],
                    "output_format": "whatever",
                }],
                "coverage_note": "ok",
            }
            if response_model is not None:
                return response_model.model_validate(payload).model_dump()
            return payload

    result = asyncio.run(planner_agent(FakeLLM(), "What do courts require?"))
    item = result[0]
    assert item["agent"] == "legal_researcher"
    assert item["tools"] == ["web_search", "fetch_content"]
    assert len(item["scope"]) == 5
    assert item["output_format"] == "structured_findings"
