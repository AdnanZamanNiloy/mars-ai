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
from app.db.sqlite import (
    complete_research_run,
    record_event,
    save_agent_tasks,
    save_claims,
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


@router.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


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
        try:
            state = build_initial_state(
                payload.query,
                settings.max_iterations,
                deep_research=payload.deep_research,
                max_parallel_agents=settings.max_parallel_agents,
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
                    await _persist(record_event(settings.database_url, request_id, "synthesizer", "end", payload=""))
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

                        if iteration != last_iteration and iteration > 0:
                            critique = snapshot.get("critique", {})
                            reason = critique.get("reason", "No reason provided")
                            yield event_line("critic", iteration=iteration, reason=reason)
                            last_iteration = iteration
                            await _persist(record_event(
                                settings.database_url, request_id, "critic", "end",
                                payload=json.dumps({"iteration": iteration, "is_sufficient": critique.get("is_sufficient", False)}),
                            ))

                        facts = snapshot.get("facts", [])
                        if len(facts) > emitted_findings:
                            new_facts = facts[emitted_findings : emitted_findings + 3]
                            findings = [
                                {"claim": f.get("claim", ""), "source": f.get("source", "")}
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

            if report:
                await save_report(
                    database_path=settings.database_url,
                    query=payload.query,
                    report=report,
                    confidence=confidence,
                )
                yield event_line("final_report", report=report, confidence=confidence)
            else:
                yield event_line("final_report", report="No final report generated.", confidence=confidence)
        finally:
            unbind_request_context()

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
