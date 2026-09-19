import re

from fastapi import Header, HTTPException

_SESSION_RE = re.compile(r"^[A-Za-z0-9\-]{8,64}$")


async def get_session_id(x_session_id: str | None = Header(default=None)) -> str:
    if not x_session_id or not _SESSION_RE.match(x_session_id):
        raise HTTPException(status_code=400, detail="Missing or invalid X-Session-Id header")
    return x_session_id