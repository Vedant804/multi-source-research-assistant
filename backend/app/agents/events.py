import copy
import functools
import logging
import time

from langgraph.config import get_stream_writer

logger = logging.getLogger(__name__)


def emit(**event) -> None:
    """Push a custom event to the SSE stream (requires Python 3.11+ in async nodes)."""
    event.setdefault("type", "agent")
    event["ts"] = time.time()
    get_stream_writer()(event)


def safe_agent(name: str, fallback: dict):
    """Isolate failures of non-critical agents so the pipeline can continue."""

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(state):
            try:
                return await fn(state)
            except Exception as exc:
                logger.exception("Agent %s failed", name)
                emit(agent=name, status="error", message=str(exc)[:300])
                return copy.deepcopy(fallback)

        return wrapper

    return decorator