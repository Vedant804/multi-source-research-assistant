import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.agents.graph import graph
from app.deps import get_session_id
from app.redis_client import run_event, run_finish, run_get, run_history, run_init
from app.schemas import ResearchRequest

router = APIRouter()
logger = logging.getLogger(__name__)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/stream")
async def stream_research(req: ResearchRequest, session_id: str = Depends(get_session_id)):
    run_id = uuid.uuid4().hex

    async def event_gen():
        await run_init(run_id, session_id, req.query)
        yield _sse({"type": "run_started", "run_id": run_id})
        final_report = ""
        try:
            initial = {"run_id": run_id, "session_id": session_id, "query": req.query}
            async for ev in graph.astream(initial, stream_mode="custom"):
                if ev.get("type") == "final":
                    final_report = ev.get("report", "")
                if ev.get("type") != "token":  # don't persist every token
                    await run_event(run_id, ev)
                yield _sse(ev)
            await run_finish(run_id, "completed", report=final_report)
            yield _sse({"type": "done", "run_id": run_id})
        except asyncio.CancelledError:
            await run_finish(run_id, "cancelled")
            raise
        except Exception as exc:
            logger.exception("Run %s failed", run_id)
            await run_finish(run_id, "failed", error=str(exc))
            yield _sse({"type": "error", "message": str(exc)[:500]})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.get("/history")
async def history(session_id: str = Depends(get_session_id)):
    return await run_history(session_id)


@router.get("/runs/{run_id}")
async def get_run(run_id: str, session_id: str = Depends(get_session_id)):
    run = await run_get(run_id)
    if not run or run.get("session_id") != session_id:
        raise HTTPException(404, "Run not found")
    return run