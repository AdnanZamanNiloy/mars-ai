import uuid
import json
from typing import Any, AsyncGenerator, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.db.sqlite import save_report
from app.graph.workflow import RUNTIME_STATE, build_initial_state


router = APIRouter()


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=500)


@router.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.post("/research/stream")
async def stream_research(request: Request, payload: ResearchRequest) -> StreamingResponse:
    workflow = getattr(request.app.state, "workflow", None)
    settings = getattr(request.app.state, "settings", None)

    if workflow is None or settings is None:
        raise HTTPException(status_code=500, detail="Workflow is not initialized")

    request_id = str(uuid.uuid4())

    def event_line(event_type: str, **data: Any) -> str:
        payload_data = {"type": event_type, **data}
        return json.dumps(payload_data, ensure_ascii=True) + "\n"

    async def event_stream() -> AsyncGenerator[str, None]:
        state = build_initial_state(payload.query, settings.max_iterations)
        last_iteration = -1
        emitted_plan = False
        emitted_findings = 0

        yield event_line("progress", request_id=request_id, message="Query received")

        try:
            async for snapshot in workflow.astream(state, stream_mode="values"):
                RUNTIME_STATE[request_id] = snapshot
                iteration = int(snapshot.get("iteration", 0))

                if snapshot.get("sub_questions") and not emitted_plan:
                    yield event_line("plan", items=snapshot.get("sub_questions", []))
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
        except Exception as exc:
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

        final_state: Dict[str, Any] = RUNTIME_STATE.get(request_id, {})
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

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
