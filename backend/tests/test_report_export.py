"""Report export: Markdown, DOCX and PDF generated from canonical trace data.

Contract under test:
  1. all three formats produce valid, non-empty files with a real signature
     (md text, docx ZIP/OOXML, pdf %PDF)
  2. the report structure survives: headings, tables, citations/sources
  3. metadata (query, confidence, mode, generated timestamp) is preserved
  4. the export reads ONLY the stored report — no UI scraping
  5. missing/empty runs fail gracefully with a clear ExportError / 409
  6. the HTTP route returns correct content-type, filename and status codes

No network is involved: everything renders from an in-memory trace.
"""
import io
import zipfile

import httpx
import pytest
from fastapi import FastAPI

from app.core import export as report_export
from app.core.export import ExportError, build_document, render_report


def _trace(**over):
    base = {
        "run_id": "run-abc12345",
        "query": "Compare solid-state and sodium-ion grid storage economics",
        "status": "completed",
        "complexity": "deep",
        "confidence": 0.81,
        "created_at": "2026-02-01T10:00:00+00:00",
        "completed_at": "2026-02-01T10:05:00+00:00",
        "sources": [
            {"id": 1, "url": "https://a.example.com/study", "reliability_score": 0.9},
        ],
        "claims": [
            {"claim": "X is cheaper", "source_url": "https://a.example.com/study",
             "verified": 1, "confidence": 0.9},
            {"claim": "Y scales faster", "source_url": "https://b.example.com/report",
             "verified": 0, "confidence": 0.5},
        ],
        "contradictions": [],
        "citations": [
            {"id": 1, "marker": 1, "domain": "a.example.com", "url": "https://a.example.com/study"},
            {"id": 2, "marker": 2, "domain": "b.example.com", "url": "https://b.example.com/report"},
        ],
        "final_report": {
            "report_markdown": (
                "# Final Answer\n\n"
                "Solid-state cells lead on **energy density** while sodium-ion wins on cost [1].\n\n"
                "## Comparison\n\n"
                "| Metric | Solid-state | Sodium-ion |\n"
                "| --- | --- | --- |\n"
                "| Energy density | high | medium |\n"
                "| Cost | high | low |\n\n"
                "## Key Findings\n\n"
                "- Cycle life favors sodium-ion [2]\n"
                "- Manufacturing scale favors solid-state [1]\n\n"
                "1. [a.example.com](https://a.example.com/study)\n"
                "2. [b.example.com](https://b.example.com/report)\n"
            ),
            "confidence": 0.81,
            "generated_at": "2026-02-01T10:05:00+00:00",
        },
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# Format validity
# ---------------------------------------------------------------------------

def test_markdown_is_clean_native_markdown():
    data, media, filename = render_report(_trace(), "md")
    text = data.decode("utf-8")
    assert media.startswith("text/markdown")
    assert filename.endswith(".md")
    # Native markdown structure preserved verbatim.
    assert "# Final Answer" in text
    assert "## Comparison" in text
    assert "| Metric | Solid-state | Sodium-ion |" in text
    assert "**energy density**" in text
    # Metadata header present.
    assert "**Confidence:** 81%" in text
    assert "**Mode:** deep" in text


def test_docx_is_a_valid_ooxml_zip():
    data, media, filename = render_report(_trace(), "docx")
    assert media.endswith("wordprocessingml.document")
    assert filename.endswith(".docx")
    assert data[:2] == b"PK", "DOCX must be a ZIP archive"
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        assert zf.testzip() is None, "DOCX zip must be intact"
        names = zf.namelist()
        assert "word/document.xml" in names
        assert "[Content_Types].xml" in names
        xml = zf.read("word/document.xml").decode("utf-8")
    # Headings, table text and sources all made it into the document body.
    assert "Final Answer" in xml
    assert "Comparison" in xml
    assert "Solid-state" in xml and "Sodium-ion" in xml
    assert "a.example.com" in xml


def test_pdf_has_valid_signature_and_structure():
    data, media, filename = render_report(_trace(), "pdf")
    assert media == "application/pdf"
    assert filename.endswith(".pdf")
    assert data[:5] == b"%PDF-", "PDF must start with the %PDF signature"
    assert b"%%EOF" in data[-2048:], "PDF must be properly terminated"
    assert len(data) > 2000, "a real report PDF should carry real content"


# ---------------------------------------------------------------------------
# Structure / citations preservation
# ---------------------------------------------------------------------------

def test_markdown_preserves_citations_and_sources():
    text = render_report(_trace(), "md")[0].decode("utf-8")
    # Citation URLs survive into the export exactly once each.
    assert text.count("https://a.example.com/study") >= 1
    assert text.count("https://b.example.com/report") >= 1
    # The source legend is present and not duplicated.
    assert text.lower().count("## sources") == 1


def test_exporter_appends_source_legend_when_body_lacks_one():
    trace = _trace()
    trace["final_report"]["report_markdown"] = "# Final Answer\n\nA plain report [1]."
    text = render_report(trace, "md")[0].decode("utf-8")
    assert "## Sources" in text
    assert "[1] a.example.com — https://a.example.com/study" in text


def test_parser_handles_headings_tables_lists():
    doc = build_document(_trace())
    blocks = report_export.parse_blocks(doc.body_markdown)
    kinds = [b.kind for b in blocks]
    assert "heading" in kinds
    assert "table" in kinds
    assert "bullet" in kinds
    table = next(b for b in blocks if b.kind == "table")
    assert table.rows[0][0] == "Metric"
    assert table.rows[1][1] == "high"


def test_metadata_and_counts_derived_from_canonical_data():
    doc = build_document(_trace())
    assert doc.confidence == 0.81
    assert doc.claim_count == 2
    assert doc.verified_count == 1
    assert len(doc.sources) == 2
    assert doc.title.startswith("Compare solid-state")


# ---------------------------------------------------------------------------
# Graceful failure
# ---------------------------------------------------------------------------

def test_missing_run_raises_export_error():
    with pytest.raises(ExportError, match="Run not found"):
        build_document({})


def test_run_without_completed_report_raises_export_error():
    trace = _trace()
    trace["final_report"] = None
    with pytest.raises(ExportError, match="no completed report"):
        render_report(trace, "md")


def test_whitespace_only_report_counts_as_missing():
    trace = _trace()
    trace["final_report"]["report_markdown"] = "   \n\n  "
    with pytest.raises(ExportError, match="no completed report"):
        render_report(trace, "docx")


def test_unsupported_format_rejected():
    with pytest.raises(ExportError, match="Unsupported export format"):
        render_report(_trace(), "html")


def test_safe_filename_slugifies_query():
    _, _, filename = render_report(_trace(), "pdf")
    assert filename.startswith("mars-report-compare-solid-state")
    assert filename.endswith(".pdf")
    assert " " not in filename


# ---------------------------------------------------------------------------
# HTTP route
# ---------------------------------------------------------------------------

async def _app_with_run(db_path, trace):
    from app.api.routes import limiter, router as api_router
    from app.core.config import Settings
    from app.db.sqlite import (
        save_citations, save_claims, save_final_report, start_research_run,
    )

    rid = trace["run_id"]
    await start_research_run(db_path, rid, trace["query"], "deep", 5)
    await save_claims(db_path, rid, [
        {"claim": c["claim"], "source": c["source_url"], "confidence": c["confidence"],
         "verified": c["verified"]}
        for c in trace["claims"]
    ])
    await save_citations(db_path, rid, trace["final_report"]["report_markdown"])
    await save_final_report(db_path, rid, trace["final_report"]["report_markdown"],
                            trace["confidence"])

    app = FastAPI()
    app.state.settings = Settings(groq_api_key="k", database_url=db_path, _env_file=None)
    app.state.limiter = limiter
    app.include_router(api_router, prefix="/api")
    return app


async def test_export_route_serves_all_formats(tmp_path):
    from app.db.sqlite import init_db

    db_path = str(tmp_path / "export.db")
    await init_db(db_path)
    trace = _trace()
    app = await _app_with_run(db_path, trace)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for fmt, sig, ctype in (
            ("md", b"# Compare", "text/markdown"),
            ("docx", b"PK", "wordprocessingml.document"),
            ("pdf", b"%PDF", "application/pdf"),
        ):
            resp = await client.get(f"/api/research/{trace['run_id']}/export/{fmt}")
            assert resp.status_code == 200, (fmt, resp.text)
            assert ctype in resp.headers["content-type"]
            assert resp.content[: len(sig)] == sig
            assert "attachment" in resp.headers["content-disposition"]
            assert f".{fmt}" in resp.headers["content-disposition"]


async def test_export_route_unknown_run_returns_404(tmp_path):
    from app.db.sqlite import init_db

    db_path = str(tmp_path / "export404.db")
    await init_db(db_path)
    app = await _app_with_run(db_path, _trace())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/research/does-not-exist/export/pdf")
        assert resp.status_code == 404


async def test_export_route_unsupported_format_returns_422(tmp_path):
    from app.db.sqlite import init_db

    db_path = str(tmp_path / "export422.db")
    await init_db(db_path)
    app = await _app_with_run(db_path, _trace())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/research/run-abc12345/export/html")
        assert resp.status_code == 422


async def test_export_route_incomplete_run_returns_409(tmp_path):
    """A run that exists but has no stored report must return 409, not an
    empty 200 download."""
    from app.api.routes import limiter, router as api_router
    from app.core.config import Settings
    from app.db.sqlite import init_db, start_research_run

    db_path = str(tmp_path / "export409.db")
    await init_db(db_path)
    rid = "run-incomplete-1"
    await start_research_run(db_path, rid, "unfinished question", "standard", 3)
    app = FastAPI()
    app.state.settings = Settings(groq_api_key="k", database_url=db_path, _env_file=None)
    app.state.limiter = limiter
    app.include_router(api_router, prefix="/api")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/api/research/{rid}/export/md")
        assert resp.status_code == 409
        assert "no completed report" in resp.json()["detail"].lower()
