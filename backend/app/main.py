import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import db
from app.config import settings
from app.redis_client import close_redis, get_redis
from app.routers import documents, research

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    await db.init_db()
    await get_redis().ping()
    yield
    await db.close_db()
    await close_redis()


app = FastAPI(title="Multi-Source Research Assistant", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(research.router, prefix="/api/research", tags=["research"])


@app.get("/api/health")
async def health():
    checks = {"redis": False, "postgres": False}
    try:
        checks["redis"] = bool(await get_redis().ping())
    except Exception:
        pass
    try:
        async with db.pool.connection() as conn:
            await conn.execute("SELECT 1")
        checks["postgres"] = True
    except Exception:
        pass
    return {"status": "ok" if all(checks.values()) else "degraded", **checks}