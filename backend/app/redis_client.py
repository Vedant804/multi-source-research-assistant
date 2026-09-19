import json
import time

import redis.asyncio as redis

from app.config import settings

_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(settings.redis_url, decode_responses=True)
    return _client


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# ---------------- run / agent state tracking ----------------
def _run_key(run_id: str) -> str:
    return f"run:{run_id}"


async def run_init(run_id: str, session_id: str, query: str) -> None:
    r = get_redis()
    key = _run_key(run_id)
    await r.hset(
        key,
        mapping={"session_id": session_id, "query": query, "status": "running", "started_at": time.time()},
    )
    await r.expire(key, settings.run_ttl)
    hist = f"session:{session_id}:runs"
    await r.lpush(hist, run_id)
    await r.ltrim(hist, 0, 49)
    await r.expire(hist, settings.session_cache_ttl)


async def run_event(run_id: str, event: dict) -> None:
    r = get_redis()
    key = _run_key(run_id)
    await r.rpush(f"{key}:events", json.dumps(event))
    await r.expire(f"{key}:events", settings.run_ttl)
    if event.get("type") == "agent":
        await r.hset(key, f"agent:{event['agent']}", event.get("status", ""))


async def run_finish(run_id: str, status: str, report: str = "", error: str = "") -> None:
    r = get_redis()
    await r.hset(
        _run_key(run_id),
        mapping={"status": status, "report": report, "error": error, "finished_at": time.time()},
    )


async def run_get(run_id: str) -> dict | None:
    r = get_redis()
    meta = await r.hgetall(_run_key(run_id))
    if not meta:
        return None
    events = [json.loads(e) for e in await r.lrange(f"{_run_key(run_id)}:events", 0, -1)]
    return {**meta, "run_id": run_id, "events": events}


async def run_history(session_id: str) -> list[dict]:
    r = get_redis()
    ids = await r.lrange(f"session:{session_id}:runs", 0, 19)
    out = []
    for rid in ids:
        meta = await r.hgetall(_run_key(rid))
        if meta:
            out.append(
                {"run_id": rid, "query": meta.get("query"), "status": meta.get("status"),
                 "started_at": meta.get("started_at")}
            )
    return out