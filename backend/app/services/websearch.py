import asyncio
import hashlib
import json
import logging

from app.config import settings
from app.redis_client import get_redis

logger = logging.getLogger(__name__)


async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Returns [{title, url, content}]. Cached in Redis."""
    r = get_redis()
    key = "search:" + hashlib.sha1(f"{query}|{max_results}".encode()).hexdigest()
    cached = await r.get(key)
    if cached:
        return json.loads(cached)

    items: list[dict] = []
    try:
        if settings.tavily_api_key:
            from tavily import AsyncTavilyClient

            client = AsyncTavilyClient(api_key=settings.tavily_api_key)
            res = await client.search(query=query, max_results=max_results, search_depth="basic")
            items = [
                {"title": x.get("title", ""), "url": x.get("url", ""), "content": x.get("content", "")}
                for x in res.get("results", [])
            ]
        else:
            from duckduckgo_search import DDGS

            def _run():
                with DDGS() as d:
                    return list(d.text(query, max_results=max_results))

            raw = await asyncio.to_thread(_run)
            items = [
                {"title": x.get("title", ""), "url": x.get("href", ""), "content": x.get("body", "")}
                for x in raw
            ]
    except Exception:
        logger.exception("Web search failed for %r", query)
        return []

    if items:
        await r.set(key, json.dumps(items), ex=settings.web_search_cache_ttl)
    return items