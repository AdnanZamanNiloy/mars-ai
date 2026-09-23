import asyncio
import uuid
import json
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.usage import clear_run_usage, start_run_usage
from app.core.config import get_settings
from app.core.degradation import (
    clear_fallbacks,
    degradation_summary,
    reset_fallbacks,
    take_fallbacks,
)
from app.db.sqlite import (
    complete_research_run,
    ensure_session,
    get_run_trace,
    get_session,
    list_sessions,
    load_state_for_resume,
    mark_challenged_claims,
    mark_run_resumable_reset,
    record_event,
    save_agent_tasks,
    save_citations,
    save_claims,
    save_contradictions,
    save_critic_review,
    save_decisions,
    save_evidence,
    save_final_report,
    save_report,
    save_sources,
    save_verification_results,
    start_research_run,
    touch_session,
)
from app.core.logging import bind_request_context, get_logger, unbind_request_context
from app.graph.workflow import build_initial_state, graph_recursion_limit


router = APIRouter()

logger = get_logger(__name__)

# Applied to the expensive research stream only; other routes stay open.
limiter = Limiter(key_func=get_remote_address)


def _finding_item(fact: Dict[str, Any]) -> Dict[str, Any]:
    """Wire shape of one findings item — shared by the stream, the verified
    re-emission and the resume path. Claim Inspector (3.6) detail fields are
    harmless when absent pre-verification."""
    return {
        "claim": fact.get("claim", ""),
        "source": fact.get("source", ""),
        "verified": fact.get("verified"),
        "verification_score": fact.get("verification_score"),
        "verification_reason": fact.get("verification_reason"),
        "agent": fact.get("agent", ""),
        "confidence": fact.get("confidence"),
    }


class ResearchRequest(BaseModel):
    # min_length=1: the router (R5) handles short social turns ("hey", "hi")
    # as conversation; an empty query is still rejected.
    query: str = Field(..., min_length=1, max_length=500)
    deep_research: bool = False
    # Research Modes (3.7 + vision §28): quick | standard | deep | executive | audit | redteam
    mode: str = Field(default="standard", pattern="^(quick|standard|deep|executive|audit|redteam)$")
    # Chat identity: all questions in one chat share this id so the backend
    # appends them to one session instead of minting a session per question.
    # Optional for backward compatibility — absent means "generate one".
    session_id: str | None = Field(default=None, max_length=120)


@router.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


class ProviderIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=60)
    base_url: str = Field(..., min_length=8, max_length=500)
    api_key: str = Field(..., min_length=1, max_length=2000)
    model: str = Field(..., min_length=1, max_length=200)
    # Optional: older clients omit it and the store falls back to `model`.
    model_name: str | None = Field(default=None, max_length=200)


class ProviderUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=60)
    base_url: str | None = Field(default=None, max_length=500)
    api_key: str | None = Field(default=None, max_length=2000)
    model: str | None = Field(default=None, max_length=200)
    model_name: str | None = Field(default=None, max_length=200)


class ProviderTestIn(BaseModel):
    timeout_sec: float | None = None


class ChainIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=60)


class ChainUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=60)


class ChainMembersIn(BaseModel):
    # Ordered provider ids: index 0 is the primary, the rest are fallbacks.
    provider_ids: list[int] = Field(default_factory=list, max_length=12)


class ChainEnabledIn(BaseModel):
    enabled: bool = True


def _providers_db(request: Request) -> str:
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(status_code=500, detail="Workflow is not initialized")
    return settings.database_url


@router.get("/providers")
async def list_llm_providers(request: Request) -> Dict[str, Any]:
    """Providers tab data: saved providers (keys masked) plus active id."""
    from app.core import providers as provider_store

    rows = await provider_store.list_providers(_providers_db(request))
    active = next((r for r in rows if r.get("is_active")), None)
    return {"providers": rows, "active_id": active["id"] if active else None}


@router.post("/providers", status_code=201)
async def create_llm_provider(body: ProviderIn, request: Request) -> Dict[str, Any]:
    from app.core import providers as provider_store

    try:
        row = await provider_store.save_provider(
            _providers_db(request), name=body.name, base_url=body.base_url,
            model=body.model, api_key=body.api_key, model_name=body.model_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"provider": row}


@router.put("/providers/{provider_id}")
async def update_llm_provider(provider_id: int, body: ProviderUpdate, request: Request) -> Dict[str, Any]:
    """Partial update; omit api_key to keep the stored key."""
    from app.core import providers as provider_store

    db_path = _providers_db(request)
    existing = await provider_store.get_provider(db_path, provider_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider id: {provider_id}")
    try:
        row = await provider_store.save_provider(
            db_path,
            provider_id=provider_id,
            name=body.name if body.name is not None else existing["name"],
            base_url=body.base_url if body.base_url is not None else existing["base_url"],
            model=body.model if body.model is not None else existing["model"],
            api_key=body.api_key,
            model_name=body.model_name,
        )
    except LookupError:
        raise HTTPException(status_code=404, detail=f"Unknown provider id: {provider_id}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"provider": row}


@router.delete("/providers/{provider_id}")
async def delete_llm_provider(provider_id: int, request: Request) -> Dict[str, Any]:
    from app.core import providers as provider_store

    db_path = _providers_db(request)
    if not await provider_store.delete_provider(db_path, provider_id):
        raise HTTPException(status_code=404, detail=f"Unknown provider id: {provider_id}")
    # Cascade through chain membership: a deleted provider must never leave a
    # dangling (and subsequently un-resolvable) member in an enabled chain.
    await provider_store.remove_provider_from_chains(db_path, provider_id)
    return {"deleted": True}


@router.post("/providers/{provider_id}/active")
async def set_active_llm_provider(provider_id: int, request: Request) -> Dict[str, Any]:
    """Serving mode = Single Model. Exactly one active provider, and it
    disables any enabled fallback chain: the two serving modes are mutually
    exclusive, so selecting a single model automatically turns off chains."""
    from app.core import providers as provider_store

    db_path = _providers_db(request)
    try:
        row = await provider_store.set_active_provider(db_path, provider_id)
    except LookupError:
        raise HTTPException(status_code=404, detail=f"Unknown provider id: {provider_id}")
    enabled = await provider_store.get_enabled_chain(db_path)
    if enabled is not None:
        await provider_store.set_chain_enabled(db_path, enabled["id"], False)
    return {"active": row}


@router.post("/providers/active/clear")
async def clear_active_llm_provider(request: Request) -> Dict[str, Any]:
    """Deselect: fall back to the legacy env/Groq/HF chain."""
    from app.core import providers as provider_store

    await provider_store.clear_active_provider(_providers_db(request))
    return {"active": None}


@router.post("/providers/{provider_id}/test")
async def test_llm_provider(
    provider_id: int, request: Request, body: ProviderTestIn | None = None
) -> Dict[str, Any]:
    """Single-shot connection probe (bypasses breakers, spends one call).
    The key never appears in logs or responses. timeout_sec is clamped to
    5..120s (default 15) — slow free-tier models need the headroom, but an
    unbounded probe could outlive the caller's patience."""
    import time as _time

    import httpx as _httpx

    from app.core import providers as provider_store

    timeout = 15.0
    if body is not None and body.timeout_sec is not None:
        timeout = min(max(float(body.timeout_sec), 5.0), 120.0)
    secret = await provider_store.get_provider_secret(_providers_db(request), provider_id)
    if secret is None:
        raise HTTPException(status_code=404, detail=f"Unknown provider id: {provider_id}")
    base = str(secret["base_url"]).rstrip("/")
    url = base if base.lower().endswith("/chat/completions") else base + "/chat/completions"
    started = _time.monotonic()
    try:
        async with _httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {secret['api_key']}", "Content-Type": "application/json"},
                json={
                    "model": secret["model"],
                    "max_tokens": 8,
                    "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
                },
            )
            response.raise_for_status()
    except Exception as exc:
        # Key material never logged: only the exception type and host-level
        # detail leave this handler.
        logger.warning("[Providers] test probe failed for id=%s: %s", provider_id, type(exc).__name__)
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    latency_ms = int((_time.monotonic() - started) * 1000)
    return {"ok": True, "latency_ms": latency_ms}


# ---------------------------------------------------------------------------
# Provider fallback chains
# ---------------------------------------------------------------------------

@router.get("/provider-chains")
async def list_provider_chains(request: Request) -> Dict[str, Any]:
    """All chains (ordered members included) plus the enabled chain id."""
    from app.core import providers as provider_store

    chains = await provider_store.list_chains(_providers_db(request))
    enabled = next((c["id"] for c in chains if c.get("is_enabled")), None)
    return {"chains": chains, "enabled_id": enabled}


@router.post("/provider-chains", status_code=201)
async def create_provider_chain(body: ChainIn, request: Request) -> Dict[str, Any]:
    from app.core import providers as provider_store

    try:
        row = await provider_store.save_chain(_providers_db(request), name=body.name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"chain": row}


@router.put("/provider-chains/{chain_id}")
async def update_provider_chain(chain_id: int, body: ChainUpdate, request: Request) -> Dict[str, Any]:
    """Rename a chain. Omit name to no-op."""
    from app.core import providers as provider_store

    db_path = _providers_db(request)
    existing = await provider_store.get_chain(db_path, chain_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Unknown chain id: {chain_id}")
    try:
        row = await provider_store.save_chain(
            db_path, chain_id=chain_id, name=body.name if body.name is not None else existing["name"],
        )
    except LookupError:
        raise HTTPException(status_code=404, detail=f"Unknown chain id: {chain_id}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"chain": row}


@router.delete("/provider-chains/{chain_id}")
async def delete_provider_chain(chain_id: int, request: Request) -> Dict[str, Any]:
    from app.core import providers as provider_store

    if not await provider_store.delete_chain(_providers_db(request), chain_id):
        raise HTTPException(status_code=404, detail=f"Unknown chain id: {chain_id}")
    return {"deleted": True}


@router.put("/provider-chains/{chain_id}/members")
async def set_provider_chain_members(
    chain_id: int, body: ChainMembersIn, request: Request
) -> Dict[str, Any]:
    """Replace the chain's ordered members (index 0 = primary). Adding,
    removing and reordering all go through here."""
    from app.core import providers as provider_store

    try:
        row = await provider_store.set_chain_members(
            _providers_db(request), chain_id, body.provider_ids
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"chain": row}


@router.post("/provider-chains/{chain_id}/reorder")
async def reorder_provider_chain(
    chain_id: int, body: ChainMembersIn, request: Request
) -> Dict[str, Any]:
    """Reorder the chain's EXISTING members (same set, new order)."""
    from app.core import providers as provider_store

    try:
        row = await provider_store.reorder_chain_members(
            _providers_db(request), chain_id, body.provider_ids
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"chain": row}


@router.post("/provider-chains/{chain_id}/enabled")
async def set_provider_chain_enabled(
    chain_id: int, request: Request, body: ChainEnabledIn | None = None
) -> Dict[str, Any]:
    """Serving mode = Fallback Chain. Enabling a chain clears any active
    single model: the two serving modes are mutually exclusive."""
    from app.core import providers as provider_store

    enabled = True if body is None else bool(body.enabled)
    db_path = _providers_db(request)
    try:
        row = await provider_store.set_chain_enabled(db_path, chain_id, enabled)
    except LookupError:
        raise HTTPException(status_code=404, detail=f"Unknown chain id: {chain_id}")
    if enabled:
        await provider_store.clear_active_provider(db_path)
    return {"chain": row}


@router.post("/provider-chains/clear")
async def clear_provider_chain(request: Request) -> Dict[str, Any]:
    """Disable whichever chain is enabled (returns to single-provider mode)."""
    from app.core import providers as provider_store

    db_path = _providers_db(request)
    enabled = await provider_store.get_enabled_chain(db_path)
    if enabled is not None:
        await provider_store.set_chain_enabled(db_path, enabled["id"], False)
    return {"enabled_id": None}


@router.get("/research/{run_id}/trace")
async def research_trace(run_id: str, request: Request) -> Dict[str, Any]:
    """Research Replay (3.4): ordered, joinable reconstruction of one run."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(status_code=500, detail="Workflow is not initialized")

    trace = await get_run_trace(settings.database_url, run_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    return trace


@router.get("/research/{run_id}/export/{fmt}")
async def export_research_report(run_id: str, fmt: str, request: Request) -> StreamingResponse:
    """Download a completed report as Markdown, DOCX or PDF.

    Renders from the SAME canonical data the trace endpoint returns (persisted
    final_reports.report_markdown + stored citations/sources), never from
    rendered UI text. A run without a completed report returns 409 so the UI
    can explain the missing report rather than serving an empty file.
    """
    from app.core import export as report_export

    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(status_code=500, detail="Workflow is not initialized")
    fmt = (fmt or "").lower()
    if not report_export.is_supported_format(fmt):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported export format '{fmt}'. Use one of: md, docx, pdf.",
        )
    trace = await get_run_trace(settings.database_url, run_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
    try:
        payload, media_type, filename = report_export.render_report(trace, fmt)
    except report_export.ExportError as exc:
        # Missing/incomplete report is a client-explainable state, not a 500.
        raise HTTPException(status_code=409, detail=str(exc))
    return StreamingResponse(
        iter([payload]),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/sessions")
async def sessions_list(request: Request) -> Dict[str, Any]:
    """Sidebar data: one row per chat session (newest activity first).

    Previously the UI derived one entry per run, which is why three
    questions in one chat produced three separate sidebar chats.
    """
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(status_code=500, detail="Workflow is not initialized")
    return {"sessions": await list_sessions(settings.database_url)}


@router.get("/sessions/{session_id}")
async def session_detail(session_id: str, request: Request) -> Dict[str, Any]:
    """Full ordered chat history for one session, so reload restores it."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(status_code=500, detail="Workflow is not initialized")
    session = await get_session(settings.database_url, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return session


@router.post("/research/stream")
@limiter.limit(get_settings().rate_limit)
async def stream_research(request: Request, payload: ResearchRequest) -> StreamingResponse:
    workflow = getattr(request.app.state, "workflow", None)
    settings = getattr(request.app.state, "settings", None)

    if workflow is None or settings is None:
        raise HTTPException(status_code=500, detail="Workflow is not initialized")

    request_id = str(uuid.uuid4())
    # Reuse the chat's session id when the client supplies one; otherwise the
    # backend owns a fresh session so old clients still get a stable grouping.
    session_id = str(payload.session_id).strip() if payload.session_id else str(uuid.uuid4())

    def event_line(event_type: str, **data: Any) -> str:
        payload_data = {"type": event_type, **data}
        return json.dumps(payload_data, ensure_ascii=True) + "\n"

    def plan_items_for_event(raw_items: Any) -> list[str]:
        if not isinstance(raw_items, list):
            return []

        items: list[str] = []
        for item in raw_items:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = str(item.get("question", "")).strip()
            else:
                text = ""
            if text:
                items.append(text)
        return items

    async def event_stream() -> AsyncGenerator[str, None]:
        bind_request_context(request_id=request_id)
        reset_fallbacks()
        # Run ledger (Feature 12): per-run budget + usage accounting. LLM
        # calls, searches and cache hits record here; the depth controller
        # consults it before every expansion; the final event persists the
        # estimated cost (previously always None).
        usage = start_run_usage(request_id, settings, mode=str(payload.mode or "standard"))
        try:
            state = build_initial_state(
                payload.query,
                settings.max_iterations,
                deep_research=payload.deep_research,
                max_parallel_agents=settings.max_parallel_agents,
                mode=payload.mode,
            )
            last_iteration = -1
            emitted_intent = False
            emitted_route = False
            emitted_direct = False
            emitted_plan = False
            emitted_findings = 0
            emitted_annotated = 0
            saved_facts = 0
            saved_source_urls: set = set()

            async def _persist(coro):
                """Memory persistence must never kill a research run — log and continue."""
                try:
                    await coro
                except Exception as exc:
                    logger.warning("persistence_failed", error=str(exc), exc_info=exc)

            async def _finish_run(status: str, confidence: float, cost: float) -> None:
                """Terminal run write + session touch in one place, so every exit
                path (completed/failed/timeout) keeps the chat's updated_at fresh."""
                await _persist(complete_research_run(
                    settings.database_url, request_id, status,
                    confidence=confidence, estimated_cost=cost,
                ))
                await _persist(touch_session(settings.database_url, session_id))

            # Node-level event trail (2.10): stream_mode="values" yields full
            # state after each node, so node completions are derived from the
            # first snapshot in which each marker appears.
            recorded_nodes: set = set()
            # Superstep boundary for per-node timing: stream_mode="values"
            # yields once per superstep, so the previous yield's timestamp is
            # an honest started_at for every node completing in this one.
            # (Before this, record_event stubbed started_at = ended_at and
            # per-stage duration was unmeasurable from the trace.)
            snapshot_boundary: list[str] = [""]

            async def _record_node_events(snapshot: Dict[str, Any], started_at: str) -> None:
                def _once(node: str) -> bool:
                    if node in recorded_nodes:
                        return False
                    recorded_nodes.add(node)
                    return True

                if snapshot.get("sub_questions") and _once("planner"):
                    payload_json = json.dumps({"sub_questions": len(snapshot["sub_questions"])})
                    await _persist(record_event(settings.database_url, request_id, "planner", "end", payload=payload_json, started_at=started_at))
                if snapshot.get("search_results") and _once("search"):
                    payload_json = json.dumps({"results": len(snapshot["search_results"])})
                    await _persist(record_event(settings.database_url, request_id, "search", "end", payload=payload_json, started_at=started_at))
                facts = snapshot.get("facts", [])
                if facts and _once("summarizer"):
                    await _persist(record_event(settings.database_url, request_id, "summarizer", "end", payload=json.dumps({"facts": len(facts)}), started_at=started_at))
                if facts and any("verified" in f for f in facts) and _once("verifier"):
                    verified_count = sum(1 for f in facts if f.get("verified"))
                    await _persist(record_event(settings.database_url, request_id, "verifier", "end", payload=json.dumps({"verified": verified_count, "total": len(facts)}), started_at=started_at))
                if snapshot.get("synthesized_answer") and _once("synthesizer"):
                    support = snapshot.get("answer_support", {}) or {}
                    await _persist(record_event(settings.database_url, request_id, "synthesizer", "end", payload=json.dumps({
                        "support_rate": support.get("rate"),
                        "cited": support.get("cited"),
                        "supported": support.get("supported"),
                    }), started_at=started_at))
                if snapshot.get("final_report") and _once("finalize"):
                    await _persist(record_event(settings.database_url, request_id, "finalize", "end", payload="", started_at=started_at))

            await _persist(ensure_session(
                settings.database_url, session_id, title=payload.query,
            ))
            await _persist(start_research_run(
                settings.database_url,
                request_id,
                payload.query,
                complexity=str(state.get("orchestration", {}).get("complexity_level", "unknown")),
                agent_count=int(state.get("orchestration", {}).get("target_agents", 0)),
                max_iterations=int(state.get("max_iterations", 3)),
                session_id=session_id,
            ))

            yield event_line("progress", request_id=request_id, session_id=session_id, message="Query received")

            # Pre-flight provider probe: a run with zero reachable LLM
            # providers is doomed to degrade to extraction — fail in seconds
            # with the per-provider reasons instead of minutes of garbage.
            llm_client = getattr(request.app.state, "llm", None)
            if llm_client is not None:
                probe_ok, probe_detail = await llm_client.probe_all()
                if not probe_ok:
                    await _finish_run("failed", 0.0, None)
                    yield event_line(
                        "error",
                        message=(
                            f"No LLM provider is reachable right now — {probe_detail}. "
                            "Add or switch providers in the Providers tab, or wait "
                            "for provider quotas to reset."
                        ),
                    )
                    return

            try:
                # Outer ceiling on total request time — individual LLM/search
                # timeouts don't bound the planner→search→summarize→critic loop.
                # `last_snapshot` is scoped to this request's coroutine — no
                # module-level state, so nothing to leak.
                last_snapshot: Dict[str, Any] = {}
                async with asyncio.timeout(settings.research_timeout_sec):
                    async for snapshot in workflow.astream(
                        state,
                        stream_mode="values",
                        config={"recursion_limit": graph_recursion_limit(state)},
                    ):
                        last_snapshot = snapshot
                        iteration = int(snapshot.get("iteration", 0))

                        # Boundary for THIS superstep's node events: the time
                        # the previous snapshot was observed (or run start).
                        _now_iso = datetime.now(timezone.utc).isoformat()
                        started_iso = snapshot_boundary[0] or _now_iso
                        snapshot_boundary[0] = _now_iso

                        await _record_node_events(snapshot, started_iso)

                        if snapshot.get("intent") and not emitted_intent:
                            # Understand-before-searching: surface the resolved
                            # intent (senses, domain, level) before the plan.
                            emitted_intent = True
                            intent_data = snapshot.get("intent") or {}
                            yield event_line("intent", **{
                                k: intent_data.get(k)
                                for k in ("query_type", "domain", "explanation_level",
                                          "ambiguity", "senses", "recommended_action", "origin")
                            })
                            await _persist(record_event(
                                settings.database_url, request_id, "intent", "end",
                                payload=json.dumps({
                                    "ambiguity": intent_data.get("ambiguity"),
                                    "domain": intent_data.get("domain"),
                                    "origin": intent_data.get("origin"),
                                }),
                                started_at=started_iso,
                            ))

                        if snapshot.get("route") and not emitted_route:
                            # Query router (R2): the direct-vs-research
                            # decision. Surfaced for the trace/UI; the path is
                            # not branched on yet (R3).
                            emitted_route = True
                            route_data = snapshot.get("route") or {}
                            yield event_line("route", **{
                                k: route_data.get(k)
                                for k in ("path", "reason", "confidence", "origin", "signals")
                            })
                            await _persist(record_event(
                                settings.database_url, request_id, "route", "end",
                                payload=json.dumps({
                                    "path": route_data.get("path"),
                                    "origin": route_data.get("origin"),
                                }),
                                started_at=started_iso,
                            ))

                        if snapshot.get("direct_answer") and not emitted_direct:
                            # Direct-answer path (R3): the query was answered
                            # without research. Surfaced separately from the
                            # final report so the UI can render an ungrounded
                            # answer with its own affordances.
                            emitted_direct = True
                            direct_meta = snapshot.get("direct_answer_meta") or {}
                            yield event_line(
                                "direct_answer",
                                answer=str(snapshot.get("direct_answer", "")),
                                confidence=snapshot.get("confidence"),
                                self_confidence=direct_meta.get("confidence"),
                                reason=direct_meta.get("reason", ""),
                                kind=direct_meta.get("conversation_kind", ""),
                            )
                            await _persist(record_event(
                                settings.database_url, request_id, "direct_answer", "end",
                                payload=json.dumps({
                                    "chars": len(str(snapshot.get("direct_answer", ""))),
                                    "self_confidence": direct_meta.get("confidence"),
                                }),
                                started_at=started_iso,
                            ))

                        if snapshot.get("sub_questions") and not emitted_plan:
                            yield event_line(
                                "plan",
                                items=plan_items_for_event(snapshot.get("sub_questions", [])),
                                orchestration=snapshot.get("orchestration", {}),
                                waves=snapshot.get("execution_waves", []) or [],
                            )
                            emitted_plan = True
                            await _persist(save_agent_tasks(
                                settings.database_url,
                                request_id,
                                snapshot.get("sub_questions", []),
                            ))

                        if snapshot.get("search_results"):
                            yield event_line(
                                "search_progress",
                                snippets=len(snapshot["search_results"]),
                            )
                            # Incremental persistence: expansion passes add new
                            # sources — save only unseen URLs, never re-insert.
                            fresh_sources = [
                                r for r in snapshot.get("search_results", [])
                                if r.get("url") and r.get("url") not in saved_source_urls
                            ]
                            if fresh_sources:
                                saved_source_urls.update(r.get("url") for r in fresh_sources)
                                await _persist(save_sources(
                                    settings.database_url,
                                    request_id,
                                    fresh_sources,
                                ))
                                await _persist(save_evidence(
                                    settings.database_url,
                                    request_id,
                                    fresh_sources,
                                ))

                        if iteration != last_iteration and iteration > 0:
                            critique = snapshot.get("critique", {})
                            reason = critique.get("reason", "No reason provided")
                            yield event_line("critic", iteration=iteration, reason=reason,
                                             breakdown=snapshot.get("confidence_breakdown") or {},
                                             redteam=snapshot.get("redteam") or {},
                                             budget=usage.snapshot())
                            last_iteration = iteration
                            await _persist(record_event(
                                settings.database_url, request_id, "critic", "end",
                                payload=json.dumps({"iteration": iteration, "is_sufficient": critique.get("is_sufficient", False)}),
                                started_at=started_iso,
                            ))
                            # Every critic iteration is durable (3.8) — Replay
                            # shows the back-and-forth, not only the outcome.
                            await _persist(save_critic_review(
                                settings.database_url, request_id, iteration, critique,
                                breakdown=snapshot.get("confidence_breakdown") or {},
                            ))

                        facts = [f for f in snapshot.get("facts", []) if isinstance(f, dict)]
                        if len(facts) > emitted_findings:
                            # Emit EVERY new fact. The old `+3` slice emitted
                            # three but marked the whole batch consumed, so
                            # facts 4..N of a large batch never streamed.
                            findings = [_finding_item(f) for f in facts[emitted_findings:]]
                            yield event_line("findings", items=findings)
                            emitted_findings = len(facts)
                        # The verifier annotates in place (same list length), so the
                        # length check above never fires for it. Re-emit the newly
                        # annotated facts whenever the annotated count grows — the
                        # old once-per-run re-emission left every expansion-pass
                        # claim stuck at "verification pending" in the UI.
                        annotated = sum(1 for f in facts if "verified" in f)
                        if annotated > emitted_annotated:
                            verified_items = [
                                _finding_item(f) for f in facts if "verified" in f
                            ][emitted_annotated:]
                            yield event_line("findings", verified_update=True, items=verified_items)
                            emitted_annotated = annotated

                        if len(facts) > saved_facts and iteration > 0:
                            # Persist new claims incrementally — but only when the
                            # appended facts carry verification flags (post-verifier
                            # snapshots). The summarizer snapshot grows the list
                            # BEFORE verification, and saving there stored
                            # expansion-pass claims with verified=0 even when they
                            # later verified; the length never changes at the
                            # verifier snapshot, so the flags were never re-saved.
                            new_facts = facts[saved_facts:]
                            if new_facts and all("verified" in f for f in new_facts):
                                await _persist(save_claims(
                                    settings.database_url,
                                    request_id,
                                    new_facts,
                                ))
                                saved_facts = len(facts)
            except TimeoutError:
                await _finish_run("timeout", 0.0, round(usage.snapshot().get("spent_usd", 0.0) or 0.0, 6))
                yield event_line(
                    "error",
                    message=(
                        f"Research timed out after {int(settings.research_timeout_sec)}s. "
                        "Try a narrower query or raise RESEARCH_TIMEOUT_SEC."
                    ),
                )
                return
            except asyncio.CancelledError:
                # Client disconnected or pressed Stop mid-stream: Starlette
                # cancels this generator. A cancelled scope cannot await, so
                # the run is marked via a DETACHED task — otherwise the row
                # sits in 'running' forever and the trace lies about the run.
                # 'cancelled' (not 'timeout') so the UI/DB tell a user stop
                # apart from a real timeout; neither is resumable, and no
                # report is saved on this path (only the normal completion
                # below writes one), so a partial answer can never persist.
                cost = round(usage.snapshot().get("spent_usd", 0.0) or 0.0, 6)
                asyncio.get_running_loop().create_task(
                    complete_research_run(
                        settings.database_url, request_id, "cancelled",
                        confidence=0.0, estimated_cost=cost,
                    )
                )
                asyncio.get_running_loop().create_task(
                    touch_session(settings.database_url, session_id)
                )
                raise
            except Exception as exc:
                await _finish_run("failed", 0.0, round(usage.snapshot().get("spent_usd", 0.0) or 0.0, 6))
                message = str(exc)
                if "No LLM provider configured" in message:
                    yield event_line(
                        "error",
                        message=(
                            "No LLM key found at runtime. Add GROQ_API_KEY or "
                            "HUGGINGFACE_API_KEY to .env, then restart the backend."
                        ),
                    )
                else:
                    yield event_line("error", message=f"Research workflow failed: {message}")
                return

            final_state: Dict[str, Any] = last_snapshot
            report = str(final_state.get("final_report", ""))
            confidence = float(final_state.get("confidence", 0.0))
            budget_snapshot = usage.snapshot()

            await _finish_run("completed", confidence, round(float(budget_snapshot.get("spent_usd", 0.0) or 0.0), 6))
            # Challenged flags land once contradictions are known (end of run).
            await _persist(mark_challenged_claims(
                settings.database_url, request_id, last_snapshot.get("contradictions") or [],
            ))
            # Verification-domain memory: contradictions, per-fact results,
            # and parsed citations persist alongside the report.
            await _persist(save_contradictions(
                settings.database_url, request_id, last_snapshot.get("contradictions") or [],
            ))
            await _persist(save_verification_results(
                settings.database_url, request_id, last_snapshot.get("facts", []),
            ))

            if report:
                # Persistence must never kill a completed run — save_report
                # was previously the one unwrapped call; a fresh install or
                # unwritable DB killed the stream right before delivery.
                await _persist(save_report(
                    database_path=settings.database_url,
                    query=payload.query,
                    report=report,
                    confidence=confidence,
                ))
                # Canonical per-run report row (3.8). The audit/trace document
                # is persisted separately from the primary answer.
                await _persist(save_final_report(
                    settings.database_url, request_id, report, confidence,
                    audit_markdown=str(last_snapshot.get("final_audit", "") or ""),
                ))
                await _persist(save_citations(settings.database_url, request_id, report))
                # Degradation flag: which agents fell back to deterministic
                # defaults (field on the existing event — no contract break).
                degraded = take_fallbacks()
                degradation = degradation_summary()
                for agent in degraded:
                    await _persist(record_event(
                        settings.database_url, request_id, agent, "fallback",
                        payload=json.dumps({"agent": agent, "reason": degradation["reasons"].get(agent, "")}),
                    ))
                support = last_snapshot.get("answer_support", {}) or {}
                yield event_line("final_report", report=report, confidence=confidence, degraded=degraded,
                                 audit=str(final_state.get("final_audit", "") or ""),
                                 degraded_reasons=degradation["reasons"],
                                 provider_degraded=degradation["provider_degraded"],
                                 provider_kinds=degradation["provider_kinds"],
                                 answer_support=support.get("rate"),
                                 budget=budget_snapshot,
                                 wave_report=last_snapshot.get("wave_report") or [],
                                 citation_health=last_snapshot.get("citation_health") or {},
                                 quality=last_snapshot.get("quality") or {},
                                 outline=last_snapshot.get("outline") or {},
                                 section_wise=bool(last_snapshot.get("section_wise")),
                                 evidence_distribution=last_snapshot.get("evidence_distribution") or {})
            else:
                _degradation = degradation_summary()
                yield event_line("final_report", report="No final report generated.", confidence=confidence,
                                 degraded=_degradation["agents"],
                                 degraded_reasons=_degradation["reasons"],
                                 provider_degraded=_degradation["provider_degraded"],
                                 provider_kinds=_degradation["provider_kinds"],
                                 budget=budget_snapshot)

            # Decision Layer rows (3.5): one per strategic option.
            decision_options = last_snapshot.get("decision_options") or []
            if decision_options:
                await _persist(save_decisions(settings.database_url, request_id, decision_options))
                yield event_line("decisions", items=[
                    {k: o.get(k) for k in ("option_label", "description", "is_recommended", "rationale", "risk_note")}
                    for o in decision_options
                ])
        finally:
            clear_fallbacks()
            clear_run_usage()
            unbind_request_context()

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.post("/research/{run_id}/resume")
@limiter.limit(get_settings().rate_limit)
async def resume_research(run_id: str, request: Request) -> StreamingResponse:
    """Durable checkpointing (3.3): resume a failed/timeout run from its
    persisted rows instead of re-running the whole pipeline."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        raise HTTPException(status_code=500, detail="Workflow is not initialized")

    # A separate graph compiled with START → critic: continues from
    # persisted evidence rather than re-running planner/search (3.3 DoD).
    from app.agents.search import SearchClient
    from app.core.llm import LLMClient
    from app.graph.workflow import create_workflow as _create_workflow

    llm = LLMClient(settings)
    search_client = SearchClient(settings)
    resume_workflow = _create_workflow(llm, search_client, entry_node="critic")

    # Rebuild state from durable rows; None means not resumable.
    state = await load_state_for_resume(settings.database_url, run_id)
    if state is None:
        raise HTTPException(
            status_code=409,
            detail=f"Run {run_id} is not resumable (unknown run_id or status is not failed/timeout)",
        )

    # If the report already exists, the run has nothing left to do.
    if state.get("final_report"):
        raise HTTPException(status_code=409, detail=f"Run {run_id} already has a final report")

    await mark_run_resumable_reset(settings.database_url, run_id)

    request_id = run_id  # resume continues the SAME run identity
    resume_session_id = state.get("session_id") or ""

    def event_line(event_type: str, **data: Any) -> str:
        return json.dumps({"type": event_type, **data}, ensure_ascii=True) + "\n"

    async def resume_stream() -> AsyncGenerator[str, None]:
        bind_request_context(request_id=request_id, resumed=True)
        # A resumed run is a fresh degradation window: earlier fallbacks are
        # already recorded against this run_id from the first attempt.
        reset_fallbacks()
        last_iteration = int(state.get("iteration", 0))
        emitted_findings = 0
        emitted_annotated = 0
        saved_facts = 0

        async def _persist(coro):
            """Memory persistence must never kill a resumed run either."""
            try:
                await coro
            except Exception as exc:
                logger.warning("persistence_failed", error=str(exc), exc_info=exc)

        async def _finish(status: str, confidence: float, cost: float) -> None:
            await _persist_complete(settings.database_url, request_id, status, confidence, cost)
            if resume_session_id:
                await _persist(touch_session(settings.database_url, resume_session_id))

        try:
            yield event_line("progress", request_id=request_id, session_id=resume_session_id,
                             message=f"Resuming run {request_id[:8]}")

            # Same pre-flight as a fresh stream: a resumed run with zero
            # reachable providers would just fail again, minutes later.
            llm_client = getattr(request.app.state, "llm", None)
            if llm_client is not None:
                probe_ok, probe_detail = await llm_client.probe_all()
                if not probe_ok:
                    await _finish("failed", 0.0, None)
                    yield event_line(
                        "error",
                        message=(
                            f"No LLM provider is reachable right now — {probe_detail}. "
                            "Add or switch providers in the Providers tab, or wait "
                            "for provider quotas to reset."
                        ),
                    )
                    return

            # Resume skips planner/search: sub-questions and sources already
            # exist for this run, so re-enter at critic with existing evidence.
            # Re-verify first — loaded facts may predate verification.
            if not any("verified" in f for f in state.get("facts", [])):
                from app.agents.verifier import verify_facts
                state["facts"] = verify_facts(
                    facts=state.get("facts", []),
                    search_results=state.get("search_results", []),
                )
                await _persist_save(settings.database_url, request_id, state.get("facts", []))

            # The frontend clears findings when resuming — re-emit the loaded
            # claims so the run card shows its evidence, not zeros.
            loaded_facts = [f for f in state.get("facts", []) if isinstance(f, dict)]
            if loaded_facts:
                yield event_line("findings", items=[_finding_item(f) for f in loaded_facts])
                emitted_findings = len(loaded_facts)
                emitted_annotated = sum(1 for f in loaded_facts if "verified" in f)
                # Loaded claims are already in the claims table from the first
                # attempt — only expansion-pass additions may be saved below.
                saved_facts = len(loaded_facts)

            last_snapshot: Dict[str, Any] = {}
            try:
                async with asyncio.timeout(settings.research_timeout_sec):
                    async for snapshot in resume_workflow.astream(
                        state,
                        stream_mode="values",
                        config={"recursion_limit": graph_recursion_limit(state)},
                    ):
                        last_snapshot = snapshot
                        iteration = int(snapshot.get("iteration", 0))

                        facts = [f for f in snapshot.get("facts", []) if isinstance(f, dict)]
                        if len(facts) > emitted_findings:
                            yield event_line(
                                "findings",
                                items=[_finding_item(f) for f in facts[emitted_findings:]],
                            )
                            emitted_findings = len(facts)
                        annotated = sum(1 for f in facts if "verified" in f)
                        if annotated > emitted_annotated:
                            verified_items = [
                                _finding_item(f) for f in facts if "verified" in f
                            ][emitted_annotated:]
                            yield event_line("findings", verified_update=True, items=verified_items)
                            emitted_annotated = annotated
                        # Expansion on resume can add claims — persist them with the
                        # same post-verifier guard as a fresh stream.
                        if len(facts) > saved_facts:
                            new_facts = facts[saved_facts:]
                            if new_facts and all("verified" in f for f in new_facts):
                                await _persist(save_claims(settings.database_url, request_id, new_facts))
                                saved_facts = len(facts)

                        if iteration != last_iteration and iteration > last_iteration:
                            critique = snapshot.get("critique", {})
                            yield event_line("critic", iteration=iteration, reason=critique.get("reason", ""),
                                             breakdown=snapshot.get("confidence_breakdown") or {},
                                             redteam=snapshot.get("redteam") or {})
                            last_iteration = iteration
                            await _persist_record(settings.database_url, request_id, iteration, critique,
                                                  breakdown=snapshot.get("confidence_breakdown") or {})

            except TimeoutError:
                await _finish("timeout", 0.0, None)
                yield event_line("error", message="Resumed run timed out. Try again or raise RESEARCH_TIMEOUT_SEC.")
                return
            except asyncio.CancelledError:
                asyncio.get_running_loop().create_task(
                    _persist_complete(settings.database_url, request_id, "cancelled", 0.0, None)
                )
                if resume_session_id:
                    asyncio.get_running_loop().create_task(
                        touch_session(settings.database_url, resume_session_id)
                    )
                raise
            except Exception as exc:
                await _finish("failed", 0.0, None)
                yield event_line("error", message=f"Resumed run failed: {exc}")
                return

            final_state = last_snapshot
            report = str(final_state.get("final_report", ""))
            confidence = float(final_state.get("confidence", 0.0))
            await _finish("completed", confidence, None)
            await _persist(mark_challenged_claims(
                settings.database_url, request_id, last_snapshot.get("contradictions") or [],
            ))
            await _persist(save_contradictions(
                settings.database_url, request_id, last_snapshot.get("contradictions") or [],
            ))
            await _persist(save_verification_results(
                settings.database_url, request_id, last_snapshot.get("facts", []),
            ))
            if report:
                await _persist_report(settings.database_url, request_id, str(state.get("query", "")), report, confidence,
                                      audit=str(final_state.get("final_audit", "") or ""))
                await _persist(save_citations(settings.database_url, request_id, report))
                degraded = take_fallbacks()
                degradation = degradation_summary()
                for agent in degraded:
                    await _persist(record_event(
                        settings.database_url, request_id, agent, "fallback",
                        payload=json.dumps({"agent": agent, "reason": degradation["reasons"].get(agent, "")}),
                    ))
                support = last_snapshot.get("answer_support", {}) or {}
                yield event_line("final_report", report=report, confidence=confidence, degraded=degraded,
                                 audit=str(final_state.get("final_audit", "") or ""),
                                 degraded_reasons=degradation["reasons"],
                                 provider_degraded=degradation["provider_degraded"],
                                 provider_kinds=degradation["provider_kinds"],
                                 answer_support=support.get("rate"))
            else:
                _degradation = degradation_summary()
                yield event_line("final_report", report="No final report generated.", confidence=confidence,
                                 degraded=_degradation["agents"],
                                 degraded_reasons=_degradation["reasons"],
                                 provider_degraded=_degradation["provider_degraded"],
                                 provider_kinds=_degradation["provider_kinds"])
        finally:
            clear_fallbacks()
            unbind_request_context()

    return StreamingResponse(resume_stream(), media_type="application/x-ndjson")


async def _persist_save(db: str, run_id: str, facts: list) -> None:
    from app.db.sqlite import save_claims as _sc
    try:
        await _sc(db, run_id, facts)
    except Exception as exc:
        logger.warning("persistence_failed", error=str(exc), exc_info=exc)


async def _persist_record(db: str, run_id: str, iteration: int, critique: dict,
                          breakdown: dict | None = None) -> None:
    try:
        await record_event(db, run_id, "critic", "end", payload=json.dumps({"iteration": iteration}))
        await save_critic_review(db, run_id, iteration, critique, breakdown=breakdown or {})
    except Exception as exc:
        logger.warning("persistence_failed", error=str(exc), exc_info=exc)


async def _persist_complete(db: str, run_id: str, status: str, confidence: float, cost: float) -> None:
    try:
        await complete_research_run(db, run_id, status, confidence=confidence, estimated_cost=cost)
    except Exception as exc:
        logger.warning("persistence_failed", error=str(exc), exc_info=exc)


async def _persist_report(
    db: str, run_id: str, query: str, report: str, confidence: float, audit: str = ""
) -> None:
    try:
        await save_report(db, query, report, confidence)  # legacy table kept in sync
        await save_final_report(db, run_id, report, confidence, audit_markdown=audit)  # canonical
    except Exception as exc:
        logger.warning("persistence_failed", error=str(exc), exc_info=exc)
