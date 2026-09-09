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
from app.db.sqlite import save_report
from app.core.logging import bind_request_context, unbind_request_context
from app.graph.workflow import RUNTIME_STATE, build_initial_state


router = APIRouter()

# Applied to the expensive research stream only; other routes stay open.
limiter = Limiter(key_func=get_remote_address)


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=500)


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
            state = build_initial_state(payload.query, settings.max_iterations)
            last_iteration = -1
            emitted_plan = False
            emitted_findings = 0

            yield event_line("progress", request_id=request_id, message="Query received")

            try:
                # Outer ceiling on total request time — individual LLM/search
                # timeouts don't bound the planner→search→summarize→critic loop.
                async with asyncio.timeout(settings.research_timeout_sec):
                    async for snapshot in workflow.astream(state, stream_mode="values"):
                        RUNTIME_STATE[request_id] = snapshot
                        iteration = int(snapshot.get("iteration", 0))

                        if snapshot.get("sub_questions") and not emitted_plan:
                            yield event_line("plan", items=plan_items_for_event(snapshot.get("sub_questions", [])))
                            emitted_plan = True

                        if snapshot.get("search_results"):
                            yield event_line(
                                "search_progress",
                                snippets=len(snapshot["search_results"]),
                            )

                        if iteration != last_iteration and iteration > 0:
                            critique = snapshot.get("critique", {})
                            reason = critique.get("reason", "No reason provided")
                            yield event_line("critic", iteration=iteration, reason=reason)
                            last_iteration = iteration

                        facts = snapshot.get("facts", [])
                        if len(facts) > emitted_findings:
                            new_facts = facts[emitted_findings : emitted_findings + 3]
                            findings = [
                                {"claim": f.get("claim", ""), "source": f.get("source", "")}
                                for f in new_facts
                            ]
                            yield event_line("findings", items=findings)
                            emitted_findings = len(facts)
            except TimeoutError:
                # Don't leak the timed-out run's state snapshot.
                RUNTIME_STATE.pop(request_id, None)
                yield event_line(
                    "error",
                    message=(
                        f"Research timed out after {int(settings.research_timeout_sec)}s. "
                        "Try a narrower query or raise RESEARCH_TIMEOUT_SEC."
                    ),
                )
                return
            except Exception as exc:
                # Don't leak the failed run's state snapshot.
                RUNTIME_STATE.pop(request_id, None)
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

            # TODO(phase-1): replace RUNTIME_STATE global with local snapshot capture
            final_state: Dict[str, Any] = RUNTIME_STATE.pop(request_id, {})
            report = str(final_state.get("final_report", ""))
            confidence = float(final_state.get("confidence", 0.0))

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
