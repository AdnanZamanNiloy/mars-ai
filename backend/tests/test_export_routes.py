"""Regression: the report-export endpoint paths and headers.

Root cause of the reported "Not Found": the export route was correct, but a
stale dev server without the route returned a generic 404. This locks the
exact path contract the frontend calls, so a future rename/prefix change is
caught in CI instead of surfacing as a dead download button.

The frontend calls (api.js):
    /api/research/{runId}/export/md
    /api/research/{runId}/export/docx
    /api/research/{runId}/export/pdf
These MUST resolve to registered routes under the /api prefix.
"""
import httpx
import pytest
from fastapi import FastAPI

from app.api.routes import limiter, router as api_router
from app.core.config import Settings
from app.db.sqlite import (
    init_db, save_citations, save_final_report, start_research_run,
)

RUN_ID = "run-regress-1"
REPORT = (
    "# Final Answer\n\nRAG retrieves then generates [1].\n\n"
    "## Sources\n\n1. [example](https://example.com/a)\n"
)


@pytest.fixture
def export_app(tmp_path):
    import asyncio

    db_path = str(tmp_path / "regress.db")
    asyncio.run(init_db(db_path))

    async def _seed():
        await start_research_run(db_path, RUN_ID, "What is RAG?", "standard", 3)
        await save_citations(db_path, RUN_ID, REPORT)
        await save_final_report(db_path, RUN_ID, REPORT, 0.8)

    asyncio.run(_seed())
    app = FastAPI()
    app.state.settings = Settings(groq_api_key="k", database_url=db_path, _env_file=None)
    app.state.limiter = limiter
    app.include_router(api_router, prefix="/api")
    return app


def test_registered_export_paths_match_frontend():
    """The router must expose exactly the path the frontend fetches."""
    paths = {r.path for r in api_router.routes if "export" in getattr(r, "path", "")}
    assert "/research/{run_id}/export/{fmt}" in paths


@pytest.mark.parametrize(
    "fmt,prefix,ctype",
    [
        ("md", b"# ", "text/markdown"),
        ("docx", b"PK", "wordprocessingml.document"),
        ("pdf", b"%PDF", "application/pdf"),
    ],
)
async def test_export_endpoint_downloads_each_format(export_app, fmt, prefix, ctype):
    transport = httpx.ASGITransport(app=export_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/api/research/{RUN_ID}/export/{fmt}")
    assert resp.status_code == 200, resp.text
    assert ctype in resp.headers["content-type"]
    assert resp.content.startswith(prefix)
    disposition = resp.headers["content-disposition"]
    assert disposition.startswith("attachment;")
    assert f".{fmt}" in disposition


async def test_export_unknown_run_is_404_not_500(export_app):
    transport = httpx.ASGITransport(app=export_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/research/missing-run/export/pdf")
    assert resp.status_code == 404


async def test_export_bad_format_is_422(export_app):
    transport = httpx.ASGITransport(app=export_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/api/research/{RUN_ID}/export/xlsx")
    assert resp.status_code == 422


async def test_markdown_export_preserves_citation_url(export_app):
    transport = httpx.ASGITransport(app=export_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/api/research/{RUN_ID}/export/md")
    assert b"https://example.com/a" in resp.content
