"""Step 5: evidence grades (§A/B/C/D) reach the UI.

Two levels are covered:
- the synthesizer node returns the measured `evidence_distribution` in its
  state update, and
- the route streams it on the `final_report` NDJSON event.
"""
import json

import httpx
import pytest
from fastapi import FastAPI

from app.api.routes import limiter, router as api_router
from app.core.config import Settings


class StubEvidenceWorkflow:
    """Emits a final_report snapshot carrying an evidence distribution."""

    async def astream(self, state, stream_mode=None):
        yield {
            "final_report": "# Final Answer\nok",
            "confidence": 0.8,
            "evidence_distribution": {"A": 2, "B": 1, "C": 0, "D": 1},
        }


def _build_app(workflow, settings: Settings) -> FastAPI:
    app = FastAPI()
    app.state.workflow = workflow
    app.state.settings = settings
    app.state.limiter = limiter
    app.include_router(api_router, prefix="/api")
    return app


@pytest.fixture(autouse=True)
def _reset_limiter():
    limiter.reset()
    yield
    limiter.reset()


async def test_final_report_event_carries_evidence_distribution(tmp_path):
    from app.db.sqlite import init_db

    db_path = str(tmp_path / "evdist.db")
    await init_db(db_path)
    settings = Settings(groq_api_key="test-key", database_url=db_path, _env_file=None)
    app = _build_app(StubEvidenceWorkflow(), settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/research/stream", json={"query": "valid research query here"}
        )
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    final = [e for e in events if e.get("type") == "final_report"]
    assert len(final) == 1, events
    assert final[0]["evidence_distribution"] == {"A": 2, "B": 1, "C": 0, "D": 1}


async def test_synthesizer_node_returns_evidence_distribution(monkeypatch):
    """The synthesizer node must put the measured distribution in its update."""
    import app.graph.workflow as wf

    usable_facts = [
        {
            "claim": "RAG combines retrieval with generation",
            "source": "https://a.com",
            "confidence": 0.9,
            "verified": True,
        },
        {
            "claim": "Dense indexes serve the retriever",
            "source": "https://b.com",
            "confidence": 0.8,
            "verified": True,
        },
    ]

    class _Answer:
        passed = True
        overall = 90
        failures = []

        def to_dict(self):
            return {"overall": 90, "passed": True}

    async def fake_synthesizer(llm, query, facts, context):
        return "answer [1]"

    async def fake_check_citations(*args, **kwargs):
        return {"checked": 0, "sources": [], "summary": {}, "enabled": False}

    monkeypatch.setattr(wf, "synthesizer_agent", fake_synthesizer)
    monkeypatch.setattr(wf, "verify_answer_support", lambda *a, **k: {"rate": 1.0, "unsupported": []})
    monkeypatch.setattr(wf, "evaluate_answer", lambda *a, **k: _Answer())
    monkeypatch.setattr(
        "app.agents.citation_check.check_citations", fake_check_citations
    )

    class _LLM:
        settings = Settings(groq_api_key="k", _env_file=None)

    graph = wf.create_workflow(llm=_LLM(), search_client=None, entry_node=None)
    node = graph.nodes["synthesizer"]
    update = await node.ainvoke({
        "query": "what is RAG",
        "facts": usable_facts,
        "confidence": 0.8,
        "mode": "standard",
    })
    dist = update.get("evidence_distribution")
    assert isinstance(dist, dict)
    assert set(dist) == {"A", "B", "C", "D"}
    assert sum(dist.values()) == len(usable_facts)
