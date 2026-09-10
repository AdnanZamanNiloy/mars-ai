import asyncio
import uuid
import json
from typing import Any, AsyncGenerator, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings
from app.core.budget import BudgetTracker, current_budget
from app.core.degradation import clear_fallbacks, reset_fallbacks, take_fallbacks
from app.db.sqlite import (
    complete_research_run,
    get_run_trace,
    load_state_for_resume,
    mark_challenged_claims,
    mark_run_resumable_reset,
    record_event,
    save_agent_tasks,
    save_claims,
    save_critic_review,
    save_decisions,
    save_evidence,
    save_final_report,
    save_report,
    save_sources,
    start_research_run,
)
from app.core.logging import bind_request_context, get_logger, unbind_request_context
from app.graph.workflow import build_initial_state


router = APIRouter()

logger = get_logger(__name__)

# Applied to the expensive research stream only; other routes stay open.
limiter = Limiter(key_func=get_remote_address)


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=500)
    deep_research: bool = False
    # Research Modes (3.7 + vision §28): quick | standard | deep | executive | audit | redteam
    mode: str = Field(default="standard", pattern="^(quick|standard|deep|executive|audit|redteam)$")


@router.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


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


@router.post("/research/stream")
@limiter.limit(get_settings().rate_limit)
async def stream_research(request: Request, payload: ResearchRequest) -> StreamingResponse:
    workflow = getattr(request.app.state, "workflow", None)
    settings = getattr(request.app.state, "settings", None)

    if workflow is None or settings is None:
        raise HTTPException(status_code=500, detail="Workflow is not initialized")

    request_id = str(uuid.uuid4())

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
        try:
            state = build_initial_state(
                payload.query,
                settings.max_iterations,
                deep_research=payload.deep_research,
                max_parallel_agents=settings.max_parallel_agents,
                mode=payload.mode,
            )
            # Per-run cost governor; ContextVar-scoped so the shared LLM
            # client records usage for THIS request only.
            budget_tracker = BudgetTracker(settings)
            state["budget_tracker"] = budget_tracker
            current_budget.set(budget_tracker)
            last_budget_tokens = -1
            last_iteration = -1
            emitted_plan = False
            emitted_findings = 0
            saved_facts = 0
            saved_sources = False

            async def _persist(coro):
                """Memory persistence must never kill a research run — log and continue."""
                try:
                    await coro
                except Exception as exc:
                    logger.warning("persistence_failed", error=str(exc), exc_info=exc)

            # Node-level event trail (2.10): stream_mode="values" yields full
            # state after each node, so node completions are derived from the
            # first snapshot in which each marker appears.
            recorded_nodes: set = set()

            async def _record_node_events(snapshot: Dict[str, Any]) -> None:
                def _once(node: str) -> bool:
                    if node in recorded_nodes:
                        return False
                    recorded_nodes.add(node)
                    return True

                if snapshot.get("sub_questions") and _once("planner"):
                    payload_json = json.dumps({"sub_questions": len(snapshot["sub_questions"])})
                    await _persist(record_event(settings.database_url, request_id, "planner", "end", payload=payload_json))
                if snapshot.get("search_results") and _once("search"):
                    payload_json = json.dumps({"results": len(snapshot["search_results"])})
                    await _persist(record_event(settings.database_url, request_id, "search", "end", payload=payload_json))
                facts = snapshot.get("facts", [])
                if facts and _once("summarizer"):
                    await _persist(record_event(settings.database_url, request_id, "summarizer", "end", payload=json.dumps({"facts": len(facts)})))
                if facts and any("verified" in f for f in facts) and _once("verifier"):
                    verified_count = sum(1 for f in facts if f.get("verified"))
                    await _persist(record_event(settings.database_url, request_id, "verifier", "end", payload=json.dumps({"verified": verified_count, "total": len(facts)})))
                if snapshot.get("synthesized_answer") and _once("synthesizer"):
                    support = snapshot.get("answer_support", {}) or {}
                    await _persist(record_event(settings.database_url, request_id, "synthesizer", "end", payload=json.dumps({
                        "support_rate": support.get("rate"),
                        "cited": support.get("cited"),
                        "supported": support.get("supported"),
                    })))
                if snapshot.get("final_report") and _once("finalize"):
                    await _persist(record_event(settings.database_url, request_id, "finalize", "end", payload=""))

            await _persist(start_research_run(
                settings.database_url,
                request_id,
                payload.query,
                complexity=str(state.get("orchestration", {}).get("complexity_level", "unknown")),
                agent_count=int(state.get("orchestration", {}).get("target_agents", 0)),
            ))

            yield event_line("progress", request_id=request_id, message="Query received")

            try:
                # Outer ceiling on total request time — individual LLM/search
                # timeouts don't bound the planner→search→summarize→critic loop.
                # `last_snapshot` is scoped to this request's coroutine — no
                # module-level state, so nothing to leak.
                last_snapshot: Dict[str, Any] = {}
                async with asyncio.timeout(settings.research_timeout_sec):
                    async for snapshot in workflow.astream(state, stream_mode="values"):
                        last_snapshot = snapshot
                        iteration = int(snapshot.get("iteration", 0))

                        await _record_node_events(snapshot)

                        if snapshot.get("sub_questions") and not emitted_plan:
                            yield event_line(
                                "plan",
                                items=plan_items_for_event(snapshot.get("sub_questions", [])),
                                orchestration=snapshot.get("orchestration", {}),
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
                            if not saved_sources:
                                saved_sources = True
                                await _persist(save_sources(
                                    settings.database_url,
                                    request_id,
                                    snapshot.get("search_results", []),
                                ))
                                await _persist(save_evidence(
                                    settings.database_url,
                                    request_id,
                                    snapshot.get("search_results", []),
                                ))

                        if iteration != last_iteration and iteration > 0:
                            critique = snapshot.get("critique", {})
                            reason = critique.get("reason", "No reason provided")
                            yield event_line("critic", iteration=iteration, reason=reason,
                                             breakdown=snapshot.get("confidence_breakdown") or {})
                            last_iteration = iteration
                            await _persist(record_event(
                                settings.database_url, request_id, "critic", "end",
                                payload=json.dumps({"iteration": iteration, "is_sufficient": critique.get("is_sufficient", False)}),
                            ))
                            # Every critic iteration is durable (3.8) — Replay
                            # shows the back-and-forth, not only the outcome.
                            await _persist(save_critic_review(
                                settings.database_url, request_id, iteration, critique,
                                breakdown=snapshot.get("confidence_breakdown") or {},
                            ))

                        facts = snapshot.get("facts", [])
                        if len(facts) > emitted_findings:
                            new_facts = facts[emitted_findings : emitted_findings + 3]
                            findings = [
                                {
                                    "claim": f.get("claim", ""),
                                    "source": f.get("source", ""),
                                    # Claim Inspector (3.6) detail fields —
                                    # harmless when absent pre-verification.
                                    "verified": f.get("verified"),
                                    "verification_score": f.get("verification_score"),
                                    "verification_reason": f.get("verification_reason"),
                                    "agent": f.get("agent", ""),
                                    "confidence": f.get("confidence"),
                                }
                                for f in new_facts
                            ]
                            yield event_line("findings", items=findings)
                            emitted_findings = len(facts)

                        if len(facts) > saved_facts and iteration > 0:
                            # Persist new claims incrementally (post-verifier snapshots only).
                            await _persist(save_claims(
                                settings.database_url,
                                request_id,
                                facts[saved_facts:],
                            ))
                            saved_facts = len(facts)

                        if budget_tracker.total_tokens != last_budget_tokens:
                            last_budget_tokens = budget_tracker.total_tokens
                            yield event_line("budget", **budget_tracker.snapshot())
                            await _persist(record_event(
                                settings.database_url, request_id, "budget", "budget_check",
                                payload=json.dumps(budget_tracker.snapshot()),
                            ))
            except TimeoutError:
                await _persist(complete_research_run(
                    settings.database_url, request_id, "timeout",
                    confidence=0.0, estimated_cost=budget_tracker.estimated_cost_usd,
                ))
                yield event_line(
                    "error",
                    message=(
                        f"Research timed out after {int(settings.research_timeout_sec)}s. "
                        "Try a narrower query or raise RESEARCH_TIMEOUT_SEC."
                    ),
                )
                return
            except Exception as exc:
                await _persist(complete_research_run(
                    settings.database_url, request_id, "failed",
                    confidence=0.0, estimated_cost=budget_tracker.estimated_cost_usd,
                ))
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

            await _persist(complete_research_run(
                settings.database_url, request_id, "completed",
                confidence=confidence,
                estimated_cost=budget_tracker.estimated_cost_usd,
            ))
            # Challenged flags land once contradictions are known (end of run).
            await _persist(mark_challenged_claims(
                settings.database_url, request_id, last_snapshot.get("contradictions") or [],
            ))

            if report:
                await save_report(
                    database_path=settings.database_url,
                    query=payload.query,
                    report=report,
                    confidence=confidence,
                )
                # Canonical per-run report row (3.8).
                await _persist(save_final_report(
                    settings.database_url, request_id, report, confidence,
                ))
                # Degradation flag: which agents fell back to deterministic
                # defaults (field on the existing event — no contract break).
                degraded = take_fallbacks()
                for agent in degraded:
                    await _persist(record_event(
                        settings.database_url, request_id, agent, "fallback",
                        payload=json.dumps({"agent": agent}),
                    ))
                support = last_snapshot.get("answer_support", {}) or {}
                yield event_line("final_report", report=report, confidence=confidence, degraded=degraded,
                                 answer_support=support.get("rate"))
            else:
                yield event_line("final_report", report="No final report generated.", confidence=confidence,
                                 degraded=take_fallbacks())

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
    budget_tracker = BudgetTracker(settings)

    def event_line(event_type: str, **data: Any) -> str:
        return json.dumps({"type": event_type, **data}, ensure_ascii=True) + "\n"

    async def resume_stream() -> AsyncGenerator[str, None]:
        bind_request_context(request_id=request_id, resumed=True)
        current_budget.set(budget_tracker)
        # A resumed run is a fresh degradation window: earlier fallbacks are
        # already recorded against this run_id from the first attempt.
        reset_fallbacks()
        # Depth controller + report builder read the tracker from state, not
        # the ContextVar — without this, budget checks are silently skipped
        # on the resume path.
        state["budget_tracker"] = budget_tracker
        last_iteration = int(state.get("iteration", 0))
        emitted_findings = 0
        saved_facts = 0
        try:
            yield event_line("progress", request_id=request_id, message=f"Resuming run {request_id[:8]}")

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

            last_snapshot: Dict[str, Any] = {}
            try:
                async with asyncio.timeout(settings.research_timeout_sec):
                    async for snapshot in resume_workflow.astream(state, stream_mode="values"):
                        last_snapshot = snapshot
                        iteration = int(snapshot.get("iteration", 0))

                        facts = snapshot.get("facts", [])
                        if len(facts) > emitted_findings:
                            emitted_findings = len(facts)

                        if iteration != last_iteration and iteration > last_iteration:
                            critique = snapshot.get("critique", {})
                            yield event_line("critic", iteration=iteration, reason=critique.get("reason", ""),
                                             breakdown=snapshot.get("confidence_breakdown") or {})
                            last_iteration = iteration
                            await _persist_record(settings.database_url, request_id, iteration, critique,
                                                  breakdown=snapshot.get("confidence_breakdown") or {})

                        if budget_tracker.total_tokens > 0:
                            yield event_line("budget", **budget_tracker.snapshot())
            except TimeoutError:
                await _persist_complete(settings.database_url, request_id, "timeout", 0.0, budget_tracker.estimated_cost_usd)
                yield event_line("error", message="Resumed run timed out. Try again or raise RESEARCH_TIMEOUT_SEC.")
                return
            except Exception as exc:
                await _persist_complete(settings.database_url, request_id, "failed", 0.0, budget_tracker.estimated_cost_usd)
                yield event_line("error", message=f"Resumed run failed: {exc}")
                return

            final_state = last_snapshot
            report = str(final_state.get("final_report", ""))
            confidence = float(final_state.get("confidence", 0.0))
            await _persist_complete(settings.database_url, request_id, "completed", confidence, budget_tracker.estimated_cost_usd)
            if report:
                await _persist_report(settings.database_url, request_id, str(state.get("query", "")), report, confidence)
                degraded = take_fallbacks()
                for agent in degraded:
                    await _persist(record_event(
                        settings.database_url, request_id, agent, "fallback",
                        payload=json.dumps({"agent": agent}),
                    ))
                support = last_snapshot.get("answer_support", {}) or {}
                yield event_line("final_report", report=report, confidence=confidence, degraded=degraded,
                                 answer_support=support.get("rate"))
            else:
                yield event_line("final_report", report="No final report generated.", confidence=confidence,
                                 degraded=take_fallbacks())
        finally:
            current_budget.set(None)
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


async def _persist_report(db: str, run_id: str, query: str, report: str, confidence: float) -> None:
    try:
        await save_report(db, query, report, confidence)  # legacy table kept in sync
        await save_final_report(db, run_id, report, confidence)  # canonical
    except Exception as exc:
        logger.warning("persistence_failed", error=str(exc), exc_info=exc)
