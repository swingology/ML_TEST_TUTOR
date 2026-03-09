"""
VectorStore — manages question embeddings and semantic search.

Uses pgvector for ANN (approximate nearest neighbor) search. Embeddings are
generated via the Anthropic API using claude-opus-4-6's implicit representation
(we use a small text embedding call via the messages API with a structured
prompt that produces a fixed representation).

For production, replace the embedding call with a dedicated embedding model
(e.g., text-embedding-3-small from OpenAI or a self-hosted model) as Anthropic
does not currently expose a standalone embeddings endpoint.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import get_settings
from ..models.question import Question, EMBEDDING_DIM
from ..orchestration.prompts import EMBEDDING_QUERY_TEMPLATE

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorStore:
    """
    Stores and searches question embeddings via pgvector.

    Embedding strategy: We call Claude with a compact descriptor of the question
    (subject + topics + difficulty + stem excerpt) and capture the first 1536
    "pseudo-embedding" values from a structured JSON response. This is a
    placeholder; swap in a real embedding model for production.
    """

    def __init__(self, db: AsyncSession):
        self._db = db
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def embed_and_store(self, question: Question) -> None:
        """Generate an embedding for a question and persist it."""
        descriptor = EMBEDDING_QUERY_TEMPLATE.format(
            subject=question.subject,
            topics=", ".join(question.topics or []),
            difficulty=question.difficulty,
            tags=", ".join(question.topics or []),
        ) + f"\n\nQuestion: {question.stem[:500]}"

        embedding = await self._generate_embedding(descriptor)
        if embedding:
            question.embedding = embedding
            await self._db.flush()

    async def semantic_search(
        self,
        *,
        query: str,
        subject: str | None = None,
        topics: list[str] | None = None,
        difficulty_min: int = 1,
        difficulty_max: int = 5,
        limit: int = 20,
        exclude_ids: list[str] | None = None,
    ) -> list[str]:
        """
        Find questions semantically similar to the query.

        Returns a list of question IDs ordered by similarity (closest first).
        Falls back to keyword-filtered random selection when embeddings are sparse.
        """
        query_embedding = await self._generate_embedding(query)

        stmt = select(Question.id).where(
            Question.is_active.is_(True),
            Question.is_approved.is_(True),
            Question.difficulty.between(difficulty_min, difficulty_max),
        )

        if subject:
            stmt = stmt.where(Question.subject.ilike(f"%{subject}%"))

        if topics:
            # Filter questions that contain at least one of the requested topics
            stmt = stmt.where(
                text("topics && ARRAY[:topics]::text[]").bindparams(topics=topics)
            )

        if exclude_ids:
            stmt = stmt.where(Question.id.not_in(exclude_ids))

        if query_embedding and any(v != 0 for v in query_embedding):
            # Use pgvector cosine distance ordering
            stmt = stmt.order_by(
                Question.embedding.cosine_distance(query_embedding)
            ).limit(limit)
        else:
            # Fallback: random selection
            stmt = stmt.order_by(text("RANDOM()")).limit(limit)

        result = await self._db.execute(stmt)
        return [str(row.id) for row in result.fetchall()]

    async def _generate_embedding(self, text_input: str) -> list[float] | None:
        """
        Generate a pseudo-embedding vector using Claude.

        NOTE: This is a stand-in implementation. For production, replace with
        a dedicated embedding API (e.g., text-embedding-3-small).

        We prompt Claude to summarise the semantic meaning as a 1536-dim vector
        encoded as JSON — this is intentionally lightweight for the scaffold.
        """
        try:
            response = await self._client.messages.create(
                model="claude-haiku-4-5",  # Cheapest model for embedding proxy
                max_tokens=256,
                system=(
                    "You are an embedding proxy. Given text, output ONLY a JSON array "
                    f"of exactly {EMBEDDING_DIM} float values between -1 and 1 that "
                    "represent the semantic content. No explanation, no markdown."
                ),
                messages=[
                    {
                        "role": "user",
                        "content": f"Embed this text:\n{text_input[:1000]}",
                    }
                ],
            )
            raw = response.content[0].text.strip()
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1:
                vec = json.loads(raw[start : end + 1])
                if len(vec) == EMBEDDING_DIM:
                    return vec
        except Exception as exc:
            logger.warning("vector_store.embed.error", extra={"error": str(exc)})
        return None
