import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import settings
from app.deps import get_session_id
from app.schemas import DocumentOut
from app.services.documents import delete_document, ingest_file, list_documents

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("", response_model=list[DocumentOut])
async def get_documents(session_id: str = Depends(get_session_id)):
    return await list_documents(session_id)


@router.post("", response_model=list[DocumentOut])
async def upload_documents(
    files: list[UploadFile] = File(...),
    session_id: str = Depends(get_session_id),
):
    created = []
    limit = settings.max_upload_mb * 1024 * 1024
    for f in files:
        content = await f.read()
        if len(content) > limit:
            raise HTTPException(413, f"{f.filename} exceeds {settings.max_upload_mb} MB")
        try:
            created.append(await ingest_file(session_id, f.filename or "unnamed", content))
        except ValueError as exc:
            raise HTTPException(400, f"{f.filename}: {exc}")
        except Exception as exc:
            logger.exception("Ingestion failed for %s", f.filename)
            raise HTTPException(500, f"Failed to process {f.filename}: {exc}")
    return created


@router.delete("/{doc_id}")
async def remove_document(doc_id: str, session_id: str = Depends(get_session_id)):
    ok = await delete_document(session_id, doc_id)
    if not ok:
        raise HTTPException(404, "Document not found")
    return {"deleted": doc_id}