"""SourceDocument and IngestionJob SQLAlchemy models."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from ..core.database import Base


class SourceDocument(Base):
    """Tracks every uploaded source document and its ingestion status."""

    __tablename__ = "source_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(512), nullable=False)
    source_type = Column(String(20), nullable=False)  # "pdf" | "image" | "text"
    file_id = Column(String(256), nullable=True)       # Anthropic Files API ID
    object_key = Column(String(512), nullable=True)    # MinIO / S3 key
    file_size_bytes = Column(Integer, nullable=True)
    hints = Column(Text, nullable=True)
    status = Column(
        String(32), nullable=False, default="pending"
    )  # pending | processing | done | error
    error_message = Column(Text, nullable=True)
    questions_extracted = Column(Integer, nullable=False, default=0)
    uploaded_by = Column(String(256), nullable=True)   # user_id
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    questions = relationship("Question", back_populates="source_document")
    ingestion_jobs = relationship("IngestionJob", back_populates="source_document")


class IngestionJob(Base):
    """Tracks individual Celery ingestion jobs for async processing."""

    __tablename__ = "ingestion_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    celery_task_id = Column(String(256), nullable=True)
    source_document_id = Column(
        UUID(as_uuid=True), ForeignKey("source_documents.id"), nullable=False
    )
    status = Column(
        String(32), nullable=False, default="queued"
    )  # queued | running | done | error
    progress = Column(Integer, nullable=False, default=0)  # 0-100
    result_summary = Column(JSON, nullable=True)           # {"questions": N, ...}
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    source_document = relationship("SourceDocument", back_populates="ingestion_jobs")
