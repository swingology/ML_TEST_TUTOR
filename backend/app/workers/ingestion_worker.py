"""
Celery ingestion worker — processes document ingestion jobs asynchronously.

For heavy PDFs or batch uploads this worker runs the extraction pipeline
in the background so the API can return a job_id immediately.

Usage:
    celery -A app.workers.ingestion_worker worker --loglevel=info
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from celery import Celery

from ..core.config import get_settings
from ..core.database import AsyncSessionLocal
from ..models.document import IngestionJob, SourceDocument
from ..models.question import Question
from ..processors.document_router import DocumentRouter
from ..storage.object_store import ObjectStore
from ..storage.vector_store import VectorStore

logger = logging.getLogger(__name__)
settings = get_settings()

celery_app = Celery(
    "ml_test_tutor",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,  # Process one task at a time (IO-heavy tasks)
)


@celery_app.task(
    bind=True,
    name="ingest_document",
    max_retries=3,
    default_retry_delay=30,
)
def ingest_document_task(
    self,
    *,
    document_id: str,
    object_key: str,
    source_type: str,
    hints: str = "",
):
    """
    Celery task: download a document from object storage, run extraction,
    persist questions to the database.
    """
    asyncio.run(
        _run_ingestion(
            task=self,
            document_id=document_id,
            object_key=object_key,
            source_type=source_type,
            hints=hints,
        )
    )


async def _run_ingestion(
    *,
    task,
    document_id: str,
    object_key: str,
    source_type: str,
    hints: str,
) -> None:
    """Async inner function that performs the actual ingestion pipeline."""
    async with AsyncSessionLocal() as db:
        # Load the job record
        from sqlalchemy import select
        result = await db.execute(
            select(IngestionJob).where(
                IngestionJob.source_document_id == document_id,
                IngestionJob.status == "queued",
            )
        )
        job = result.scalar_one_or_none()
        if not job:
            logger.warning("ingestion_worker.job_not_found", extra={"document_id": document_id})
            return

        job.status = "running"
        job.started_at = datetime.utcnow()
        job.celery_task_id = task.request.id
        await db.commit()

        try:
            # Download raw file from object storage
            obj_store = ObjectStore()
            filename = object_key.split("/")[-1]
            file_bytes = obj_store.download(object_key)

            # Run extraction pipeline
            doc_router = DocumentRouter()
            extraction_result = await doc_router.route(
                file_bytes=file_bytes,
                filename=filename,
                hints=hints,
                job_id=str(job.id),
            )

            # Persist questions
            vector_store = VectorStore(db)
            for q_data in extraction_result["questions"]:
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
                    source_document_id=document_id,
                    source_ref=extraction_result.get("file_id"),
                    source_type=extraction_result.get("source_type"),
                    is_approved=False,
                )
                db.add(question)
                await db.flush()
                await vector_store.embed_and_store(question)

            # Update document and job status
            doc_result = await db.execute(
                select(SourceDocument).where(SourceDocument.id == document_id)
            )
            doc = doc_result.scalar_one_or_none()
            if doc:
                doc.status = "done"
                doc.file_id = extraction_result.get("file_id")
                doc.questions_extracted = len(extraction_result["questions"])

            job.status = "done"
            job.progress = 100
            job.completed_at = datetime.utcnow()
            job.result_summary = {
                "questions_extracted": len(extraction_result["questions"])
            }
            await db.commit()

            logger.info(
                "ingestion_worker.complete",
                extra={
                    "document_id": document_id,
                    "questions": len(extraction_result["questions"]),
                },
            )

        except Exception as exc:
            logger.exception("ingestion_worker.error")
            job.status = "error"
            job.error_message = str(exc)
            job.completed_at = datetime.utcnow()

            doc_result = await db.execute(
                select(SourceDocument).where(SourceDocument.id == document_id)
            )
            doc = doc_result.scalar_one_or_none()
            if doc:
                doc.status = "error"
                doc.error_message = str(exc)

            await db.commit()
            raise task.retry(exc=exc)
