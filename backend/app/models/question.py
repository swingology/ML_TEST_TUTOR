"""Question SQLAlchemy model with pgvector embedding support."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, DateTime, Float, Integer, JSON, String, Text, Boolean, ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from ..core.database import Base

EMBEDDING_DIM = 1536  # Claude / OpenAI ada-002 embedding dimension


class Question(Base):
    __tablename__ = "questions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stem = Column(Text, nullable=False)
    stimulus = Column(Text, nullable=True)
    stimulus_type = Column(String(20), nullable=False, default="none")
    stimulus_url = Column(String(512), nullable=True)

    # Answer choices stored as JSON: [{"label": "A", "text": "..."}, ...]
    answer_choices = Column(JSON, nullable=False, default=list)
    correct_answer = Column(String(64), nullable=False)
    explanation = Column(Text, nullable=True)

    # Taxonomy
    subject = Column(String(128), nullable=False, default="General")
    topics = Column(ARRAY(String), nullable=False, default=list)
    difficulty = Column(Integer, nullable=False, default=3)

    # Source tracking
    source_document_id = Column(
        UUID(as_uuid=True), ForeignKey("source_documents.id"), nullable=True
    )
    source_ref = Column(String(512), nullable=True)  # file_id or identifier
    source_type = Column(String(20), nullable=True)  # "pdf", "image", "text"

    # Review workflow
    is_approved = Column(Boolean, nullable=False, default=False)
    is_active = Column(Boolean, nullable=False, default=True)

    # pgvector embedding for semantic search
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    source_document = relationship("SourceDocument", back_populates="questions")
