"""
/api/questions — question retrieval and session generation endpoints.

Uses the RetrievalOrchestrationGateway to intelligently select questions
and the ExplanationGateway to generate per-question explanations.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ...core.database import get_db
from ...models.question import Question
from ...orchestration.gateway import RetrievalOrchestrationGateway, ExplanationGateway
from ...storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/questions", tags=["questions"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class AnswerChoice(BaseModel):
    label: str
    text: str


class QuestionOut(BaseModel):
    id: str
    stem: str
    stimulus: str | None
    stimulus_type: str
    answer_choices: list[AnswerChoice]
    correct_answer: str
    explanation: str | None
    subject: str
    topics: list[str]
    difficulty: int


class SessionRequest(BaseModel):
    user_id: str
    session_type: str = "practice_test"  # practice_test | drill | review | srs
    subject: str | None = None
    topics: list[str] | None = None
    difficulty_min: int = 1
    difficulty_max: int = 5
    count: int = 20
    exclude_ids: list[str] | None = None


class SessionResponse(BaseModel):
    question_ids: list[str]
    questions: list[QuestionOut]


class ExplanationRequest(BaseModel):
    question_id: str
    student_answer: str | None = None


class ExplanationResponse(BaseModel):
    question_id: str
    explanation: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[QuestionOut])
async def list_questions(
    subject: str | None = None,
    topics: Annotated[list[str] | None, Query()] = None,
    difficulty_min: int = 1,
    difficulty_max: int = 5,
    approved_only: bool = True,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List questions with optional filters."""
    stmt = select(Question).where(Question.is_active.is_(True))

    if approved_only:
        stmt = stmt.where(Question.is_approved.is_(True))
    if subject:
        stmt = stmt.where(Question.subject.ilike(f"%{subject}%"))

    stmt = (
        stmt.where(Question.difficulty.between(difficulty_min, difficulty_max))
        .offset(offset)
        .limit(limit)
    )

    result = await db.execute(stmt)
    questions = result.scalars().all()
    return [_to_out(q) for q in questions]


@router.get("/{question_id}", response_model=QuestionOut)
async def get_question(question_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieve a single question by ID."""
    result = await db.execute(
        select(Question).where(Question.id == uuid.UUID(question_id))
    )
    question = result.scalar_one_or_none()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    return _to_out(question)


@router.post("/session", response_model=SessionResponse)
async def create_session(
    req: SessionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate an ordered list of questions for a session using the
    RetrievalOrchestrationGateway (LLM-powered smart selection).
    """
    # Wire up the retrieval tools
    vector_store = VectorStore(db)

    async def semantic_search_fn(**kwargs):
        return await vector_store.semantic_search(**kwargs)

    async def missed_questions_fn(user_id: str, subject: str | None = None, limit: int = 10):
        # TODO: integrate with session_questions table for real miss tracking
        # For now, return empty list (no miss history yet)
        return []

    async def srs_due_fn(user_id: str, limit: int = 20):
        # TODO: integrate with SRS cards table
        return []

    gateway = RetrievalOrchestrationGateway(
        semantic_search_fn=semantic_search_fn,
        missed_questions_fn=missed_questions_fn,
        srs_due_fn=srs_due_fn,
    )

    question_ids = await gateway.select_questions(
        user_id=req.user_id,
        session_type=req.session_type,
        subject=req.subject,
        topics=req.topics,
        difficulty_min=req.difficulty_min,
        difficulty_max=req.difficulty_max,
        count=req.count,
        exclude_ids=req.exclude_ids,
    )

    # Fetch full question objects
    if question_ids:
        result = await db.execute(
            select(Question).where(
                Question.id.in_([uuid.UUID(qid) for qid in question_ids]),
                Question.is_active.is_(True),
                Question.is_approved.is_(True),
            )
        )
        questions = result.scalars().all()
    else:
        # Gateway returned no IDs — fall back to basic filtered list
        result = await db.execute(
            select(Question)
            .where(
                Question.is_active.is_(True),
                Question.is_approved.is_(True),
                Question.difficulty.between(req.difficulty_min, req.difficulty_max),
            )
            .limit(req.count)
        )
        questions = result.scalars().all()
        question_ids = [str(q.id) for q in questions]

    return SessionResponse(
        question_ids=question_ids,
        questions=[_to_out(q) for q in questions],
    )


@router.post("/explain", response_model=ExplanationResponse)
async def get_explanation(
    req: ExplanationRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate an LLM-powered explanation for a question answer using
    the ExplanationGateway.
    """
    result = await db.execute(
        select(Question).where(Question.id == uuid.UUID(req.question_id))
    )
    question = result.scalar_one_or_none()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    gateway = ExplanationGateway()
    explanation = await gateway.generate_explanation(
        question_stem=question.stem,
        answer_choices=question.answer_choices or [],
        correct_answer=question.correct_answer,
        student_answer=req.student_answer,
        subject=question.subject,
        topics=question.topics or [],
    )

    # Optionally cache the explanation back
    if not question.explanation:
        question.explanation = explanation
        await db.commit()

    return ExplanationResponse(question_id=req.question_id, explanation=explanation)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_out(q: Question) -> QuestionOut:
    choices = q.answer_choices or []
    return QuestionOut(
        id=str(q.id),
        stem=q.stem,
        stimulus=q.stimulus,
        stimulus_type=q.stimulus_type,
        answer_choices=[AnswerChoice(**c) for c in choices],
        correct_answer=q.correct_answer,
        explanation=q.explanation,
        subject=q.subject,
        topics=q.topics or [],
        difficulty=q.difficulty,
    )
