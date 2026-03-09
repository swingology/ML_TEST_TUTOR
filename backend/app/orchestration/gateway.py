"""
IngestOrchestrationGateway — the central LLM brain of the platform.

This gateway uses Claude (claude-opus-4-6 with adaptive thinking) as the
orchestrator. It receives a raw document upload (PDF, image, text), decides
the best extraction strategy via tool use, delegates to the appropriate
processor, stores results, and returns structured question data.

The RetrievalOrchestrationGateway handles the other direction: given a
session request, it decides which questions to surface using semantic search,
spaced repetition, and missed-question data.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import anthropic

from ..core.config import get_settings
from .prompts import (
    INGESTION_GATEWAY_SYSTEM,
    INGESTION_GATEWAY_TOOLS,
    RETRIEVAL_GATEWAY_SYSTEM,
    RETRIEVAL_GATEWAY_TOOLS,
    EXPLANATION_SYSTEM,
)

logger = logging.getLogger(__name__)
settings = get_settings()


def _make_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


# ---------------------------------------------------------------------------
# Ingestion Gateway
# ---------------------------------------------------------------------------


class IngestOrchestrationGateway:
    """
    Orchestrates end-to-end document ingestion via LLM tool use.

    Flow:
      upload() → Claude decides which processor tool to call →
      processor extracts questions → gateway returns structured questions.

    Tool implementations are injected at construction so they can be swapped
    out or mocked in tests.
    """

    def __init__(
        self,
        pdf_processor_fn,
        image_processor_fn,
        text_processor_fn,
    ):
        self._pdf_processor = pdf_processor_fn
        self._image_processor = image_processor_fn
        self._text_processor = text_processor_fn
        self._client = _make_client()

    async def ingest(
        self,
        *,
        source_type: str,  # "pdf" | "image" | "text"
        file_id: str | None = None,  # Anthropic Files API ID
        raw_text: str | None = None,
        hints: str = "",
        job_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Entry point: accept a document, orchestrate extraction, return questions.

        Args:
            source_type: "pdf", "image", or "text"
            file_id:     Anthropic Files API file_id (for pdf/image)
            raw_text:    Raw text content (for text source type)
            hints:       Optional context hints for the LLM
            job_id:      Tracking ID for logging

        Returns:
            List of extracted question dicts conforming to the question schema.
        """
        job_id = job_id or str(uuid.uuid4())
        logger.info("ingest.start", extra={"job_id": job_id, "source_type": source_type})

        # Build the user message that describes the ingestion task
        if source_type in ("pdf", "image") and file_id:
            user_content = (
                f"Ingest the uploaded {source_type.upper()} document "
                f"(file_id={file_id}). "
                + (f"Hints: {hints}" if hints else "")
            )
        elif source_type == "text" and raw_text:
            user_content = (
                "Ingest the following text and extract all questions:\n\n"
                + raw_text[:8000]  # guard against runaway context
                + ("\n\nHints: " + hints if hints else "")
            )
        else:
            raise ValueError(f"Invalid source_type={source_type!r} or missing content.")

        messages: list[anthropic.types.MessageParam] = [
            {"role": "user", "content": user_content}
        ]

        # Agentic tool-use loop: Claude decides which processor to call
        extracted_questions: list[dict[str, Any]] = []

        while True:
            response = await self._client.messages.create(
                model=settings.llm_model,
                max_tokens=8192,
                thinking={"type": "adaptive"},
                system=INGESTION_GATEWAY_SYSTEM,
                tools=INGESTION_GATEWAY_TOOLS,
                messages=messages,
            )

            logger.debug(
                "ingest.llm_response",
                extra={"job_id": job_id, "stop_reason": response.stop_reason},
            )

            # Append assistant response to conversation
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Claude finished without more tool calls — extract any JSON from text
                for block in response.content:
                    if block.type == "text":
                        parsed = _try_parse_json_array(block.text)
                        if parsed:
                            extracted_questions.extend(parsed)
                break

            if response.stop_reason != "tool_use":
                logger.warning(
                    "ingest.unexpected_stop",
                    extra={"job_id": job_id, "stop_reason": response.stop_reason},
                )
                break

            # Execute each tool call Claude requested
            tool_results: list[anthropic.types.ToolResultBlockParam] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                result = await self._execute_tool(block.name, block.input, job_id)

                # Collect any questions returned by the processor
                if isinstance(result, list):
                    extracted_questions.extend(result)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result) if not isinstance(result, str) else result,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        logger.info(
            "ingest.complete",
            extra={"job_id": job_id, "questions_found": len(extracted_questions)},
        )
        return extracted_questions

    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        job_id: str,
    ) -> Any:
        """Dispatch a tool call to the appropriate processor."""
        logger.info(
            "ingest.tool_call",
            extra={"job_id": job_id, "tool": tool_name},
        )
        if tool_name == "extract_from_pdf":
            return await self._pdf_processor(
                file_id=tool_input["file_id"],
                hints=tool_input.get("hints", ""),
            )
        elif tool_name == "extract_from_image":
            return await self._image_processor(
                file_id=tool_input["file_id"],
                hints=tool_input.get("hints", ""),
            )
        elif tool_name == "extract_from_text":
            return await self._text_processor(
                text=tool_input["text"],
                hints=tool_input.get("hints", ""),
            )
        else:
            logger.error("ingest.unknown_tool", extra={"tool": tool_name})
            return f"Error: unknown tool '{tool_name}'"


# ---------------------------------------------------------------------------
# Retrieval Gateway
# ---------------------------------------------------------------------------


class RetrievalOrchestrationGateway:
    """
    Orchestrates smart question retrieval via LLM tool use.

    Given a session request, Claude decides which retrieval tools to call
    (semantic search, missed questions, SRS queue) and in what order,
    returning a ranked list of question IDs.
    """

    def __init__(
        self,
        semantic_search_fn,
        missed_questions_fn,
        srs_due_fn,
    ):
        self._semantic_search = semantic_search_fn
        self._missed_questions = missed_questions_fn
        self._srs_due = srs_due_fn
        self._client = _make_client()

    async def select_questions(
        self,
        *,
        user_id: str,
        session_type: str,  # "practice_test" | "drill" | "review" | "srs"
        subject: str | None = None,
        topics: list[str] | None = None,
        difficulty_min: int = 1,
        difficulty_max: int = 5,
        count: int = 20,
        exclude_ids: list[str] | None = None,
    ) -> list[str]:
        """
        Return an ordered list of question IDs for a session.

        Args:
            user_id:       The student's ID (used for personalization).
            session_type:  Type of session that determines retrieval strategy.
            subject:       Optional subject filter.
            topics:        Optional list of topic filters.
            difficulty_min/max: Difficulty band.
            count:         Number of questions to return.
            exclude_ids:   IDs already used in this session.

        Returns:
            Ordered list of question IDs.
        """
        request_summary = (
            f"Session type: {session_type}. "
            f"User: {user_id}. "
            f"Subject: {subject or 'any'}. "
            f"Topics: {', '.join(topics or ['any'])}. "
            f"Difficulty: {difficulty_min}–{difficulty_max}. "
            f"Need {count} questions."
        )
        if exclude_ids:
            request_summary += f" Exclude IDs: {exclude_ids[:10]}..."

        messages: list[anthropic.types.MessageParam] = [
            {"role": "user", "content": request_summary}
        ]

        selected_ids: list[str] = []

        while True:
            response = await self._client.messages.create(
                model=settings.llm_model,
                max_tokens=4096,
                thinking={"type": "adaptive"},
                system=RETRIEVAL_GATEWAY_SYSTEM,
                tools=RETRIEVAL_GATEWAY_TOOLS,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # Extract IDs from final text response
                for block in response.content:
                    if block.type == "text":
                        ids = _try_parse_json_array(block.text)
                        if ids and all(isinstance(i, str) for i in ids):
                            selected_ids = ids
                break

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = await self._execute_retrieval_tool(
                    block.name, block.input, user_id
                )
                # Accumulate IDs returned by each tool call
                if isinstance(result, list):
                    selected_ids.extend(r for r in result if r not in selected_ids)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )
            messages.append({"role": "user", "content": tool_results})

        return selected_ids[:count]

    async def _execute_retrieval_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        user_id: str,
    ) -> Any:
        if tool_name == "semantic_search":
            return await self._semantic_search(**tool_input)
        elif tool_name == "get_missed_questions":
            return await self._missed_questions(
                user_id=user_id, **{k: v for k, v in tool_input.items() if k != "user_id"}
            )
        elif tool_name == "get_srs_due":
            return await self._srs_due(
                user_id=user_id, **{k: v for k, v in tool_input.items() if k != "user_id"}
            )
        return []


# ---------------------------------------------------------------------------
# Explanation Generator
# ---------------------------------------------------------------------------


class ExplanationGateway:
    """Generate per-question explanations using Claude with streaming."""

    def __init__(self):
        self._client = _make_client()

    async def generate_explanation(
        self,
        *,
        question_stem: str,
        answer_choices: list[dict],
        correct_answer: str,
        student_answer: str | None,
        subject: str,
        topics: list[str],
    ) -> str:
        """
        Generate a streaming explanation and return the full text.

        Returns the complete explanation string.
        """
        user_msg = (
            f"Question ({subject} / {', '.join(topics)}):\n{question_stem}\n\n"
            f"Answer choices:\n"
            + "\n".join(f"  {c['label']}: {c['text']}" for c in answer_choices)
            + f"\n\nCorrect answer: {correct_answer}"
            + (f"\nStudent's answer: {student_answer}" if student_answer else "")
        )

        full_text = ""
        async with self._client.messages.stream(
            model=settings.llm_model,
            max_tokens=1024,
            system=EXPLANATION_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            async for text_chunk in stream.text_stream:
                full_text += text_chunk

        return full_text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_parse_json_array(text: str) -> list | None:
    """Attempt to extract a JSON array from a text block."""
    text = text.strip()
    # Find the first '[' and last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
