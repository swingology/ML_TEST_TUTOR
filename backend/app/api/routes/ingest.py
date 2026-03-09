"""
/api/ingest — document ingestion endpoints.

All uploads flow through the DocumentRouter → IngestOrchestrationGateway →
appropriate processor → VectorStore → database.

Heavy processing is offloaded to Celery workers for async handling.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ...core.config import get_settings
from ...core.database import get_db
from ...models.document import SourceDocument, IngestionJob
from ...models.question import Question
from ...processors.document_router import DocumentRouter
from ...storage.object_store import ObjectStore
from ...storage.vector_store import VectorStore

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/ingest", tags=["ingestion"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class IngestionJobResponse(BaseModel):
    job_id: str
    document_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    document_id: str
    status: str
    progress: int
    questions_extracted: int
    error_message: str | None


class PendingQuestionResponse(BaseModel):
    id: str
    stem: str
    subject: str
    topics: list[str]
    difficulty: int
    source_document_id: str | None


class ApprovalResponse(BaseModel):
    question_id: str
    approved: bool
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/upload", response_model=IngestionJobResponse)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF, image, or text file")],
    hints: Annotated[str, Form()] = "",
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a document for LLM-powered question extraction.

    Accepts PDF, PNG, JPEG, WEBP, GIF, TXT, or MD files.
    Processing is done synchronously for simplicity; swap to Celery for prod.
    """
    file_bytes = await file.read()
    filename = file.filename or "upload"

    # Persist source document record
    doc = SourceDocument(
        filename=filename,
        source_type=_detect_source_type(filename),
        file_size_bytes=len(file_bytes),
        hints=hints,
        status="processing",
    )
    db.add(doc)
    await db.flush()

    # Store raw file in object storage
    object_key = f"documents/{doc.id}/{filename}"
    try:
        obj_store = ObjectStore()
        obj_store.upload(
            key=object_key,
            data=file_bytes,
            content_type=file.content_type or "application/octet-stream",
        )
        doc.object_key = object_key
    except Exception as exc:
        logger.warning("ingest.upload.object_store_error", extra={"error": str(exc)})
        # Continue without object storage (not fatal in dev mode)

    job = IngestionJob(source_document_id=doc.id, status="running", started_at=datetime.utcnow())
    db.add(job)
    await db.commit()

    # Run extraction (sync for now; Celery task in production)
    try:
        doc_router = DocumentRouter()
        result = await doc_router.route(
            file_bytes=file_bytes,
            filename=filename,
            hints=hints,
            job_id=str(job.id),
        )

        # Persist questions + embeddings
        vector_store = VectorStore(db)
        for q_data in result["questions"]:
            question = Question(
                stem=q_data.get("stem", ""),
                stimulus=q_data.get("stimulus"),
                stimulus_type=q_data.get("stimulus_type", "none"),
                answer_choices=q_data.get("answer_choices", []),
                correct_answer=q_data.get("correct_answer", ""),
                explanation=q_data.get("explanation", ""),
                subject=q_data.get("subject", "General"),
                topics=q_data.get("topics", []),
                difficulty=q_data.get("difficulty", 3),
                source_document_id=doc.id,
                source_ref=result.get("file_id"),
                source_type=result.get("source_type"),
                is_approved=False,
            )
            db.add(question)
            await db.flush()
            await vector_store.embed_and_store(question)

        doc.status = "done"
        doc.file_id = result.get("file_id")
        doc.questions_extracted = len(result["questions"])
        job.status = "done"
        job.progress = 100
        job.completed_at = datetime.utcnow()
        job.result_summary = {"questions_extracted": len(result["questions"])}

    except Exception as exc:
        logger.exception("ingest.upload.extraction_error")
        doc.status = "error"
        doc.error_message = str(exc)
        job.status = "error"
        job.error_message = str(exc)
        await db.commit()
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")

    await db.commit()

    return IngestionJobResponse(
        job_id=str(job.id),
        document_id=str(doc.id),
        status="done",
        message=f"Extracted {doc.questions_extracted} questions. Pending review.",
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, db: AsyncSession = Depends(get_db)):
    """Check the status of an ingestion job."""
    result = await db.execute(
        select(IngestionJob, SourceDocument)
        .join(SourceDocument, IngestionJob.source_document_id == SourceDocument.id)
        .where(IngestionJob.id == uuid.UUID(job_id))
    )
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    job, doc = row
    return JobStatusResponse(
        job_id=str(job.id),
        document_id=str(doc.id),
        status=job.status,
        progress=job.progress,
        questions_extracted=doc.questions_extracted,
        error_message=job.error_message,
    )


@router.get("/review", response_model=list[PendingQuestionResponse])
async def list_pending_questions(
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """List questions pending admin review (is_approved=False)."""
    result = await db.execute(
        select(Question)
        .where(Question.is_approved.is_(False), Question.is_active.is_(True))
        .limit(limit)
    )
    questions = result.scalars().all()
    return [
        PendingQuestionResponse(
            id=str(q.id),
            stem=q.stem[:200],
            subject=q.subject,
            topics=q.topics or [],
            difficulty=q.difficulty,
            source_document_id=str(q.source_document_id) if q.source_document_id else None,
        )
        for q in questions
    ]


@router.post("/review/{question_id}/approve", response_model=ApprovalResponse)
async def approve_question(question_id: str, db: AsyncSession = Depends(get_db)):
    """Approve an extracted question for use in tests/drills."""
    result = await db.execute(
        select(Question).where(Question.id == uuid.UUID(question_id))
    )
    question = result.scalar_one_or_none()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    question.is_approved = True
    await db.commit()
    return ApprovalResponse(
        question_id=question_id,
        approved=True,
        message="Question approved and available for sessions.",
    )


@router.delete("/review/{question_id}", response_model=ApprovalResponse)
async def reject_question(question_id: str, db: AsyncSession = Depends(get_db)):
    """Reject (soft-delete) an extracted question."""
    result = await db.execute(
        select(Question).where(Question.id == uuid.UUID(question_id))
    )
    question = result.scalar_one_or_none()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    question.is_active = False
    await db.commit()
    return ApprovalResponse(
        question_id=question_id,
        approved=False,
        message="Question rejected and removed from the pool.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_source_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        return "pdf"
    if ext in ("png", "jpg", "jpeg", "webp", "gif"):
        return "image"
    return "text"
