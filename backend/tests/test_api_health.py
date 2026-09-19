"""API health endpoint test (Phase 1.8)."""
import httpx

from main import app


async def test_health_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


async def test_root_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


async def test_research_stream_rejects_empty_query():
    """R5 lowered min_length to 1 so short social turns ("hey") reach the
    router; an empty query is still rejected before any pipeline work."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/research/stream", json={"query": ""})
        assert response.status_code == 422
