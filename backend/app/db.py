import logging

import psycopg
from psycopg_pool import AsyncConnectionPool
from pgvector.psycopg import register_vector_async

from app.config import settings

logger = logging.getLogger(__name__)

pool: AsyncConnectionPool | None = None

SCHEMA_STATEMENTS = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    """
    CREATE TABLE IF NOT EXISTS documents (
        id UUID PRIMARY KEY,
        session_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('text','data')),
        meta JSONB NOT NULL DEFAULT '{}',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_documents_session ON documents(session_id)",
    f"""
    CREATE TABLE IF NOT EXISTS chunks (
        id BIGSERIAL PRIMARY KEY,
        document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        session_id TEXT NOT NULL,
        chunk_index INT NOT NULL,
        content TEXT NOT NULL,
        embedding vector({settings.embedding_dim})
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)",
]


async def _configure(conn: psycopg.AsyncConnection) -> None:
    await register_vector_async(conn)


async def init_db() -> None:
    global pool
    # The extension must exist BEFORE pool connections register the vector type.
    async with await psycopg.AsyncConnection.connect(settings.database_url, autocommit=True) as conn:
        for stmt in SCHEMA_STATEMENTS:
            await conn.execute(stmt)

    pool = AsyncConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=10,
        configure=_configure,
        kwargs={"autocommit": True},
        open=False,
    )
    await pool.open()
    logger.info("PostgreSQL pool ready")


async def close_db() -> None:
    if pool is not None:
        await pool.close()