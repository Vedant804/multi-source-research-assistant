import numpy as np
from psycopg.rows import dict_row

from app import db


async def search_chunks(session_id: str, query_vec: list[float], k: int = 5) -> list[dict]:
    vec = np.array(query_vec)
    sql = """
        SELECT c.id, c.content, c.chunk_index, d.filename,
               1 - (c.embedding <=> %s) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.session_id = %s
        ORDER BY c.embedding <=> %s
        LIMIT %s
    """
    async with db.pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, (vec, session_id, vec, k))
            return await cur.fetchall()