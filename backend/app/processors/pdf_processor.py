"""
PDF Processor — extracts educational questions from PDF files.

Uses the Anthropic Files API to upload the PDF (if not already uploaded)
and sends it to Claude as a document block. Claude reads the full PDF
including embedded images and returns structured JSON question data.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from ..core.config import get_settings
from ..orchestration.prompts import PDF_EXTRACTION_SYSTEM

logger = logging.getLogger(__name__)
settings = get_settings()


class PDFProcessor:
    """
    Extracts questions from a PDF using Claude's native PDF understanding.

    The PDF is passed as a `document` content block referencing an Anthropic
    Files API file_id, so we never re-read the raw bytes in the processor.
    """

    def __init__(self):
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def extract(
        self,
        *,
        file_id: str,
        hints: str = "",
    ) -> list[dict[str, Any]]:
        """
        Extract questions from a PDF already uploaded to the Anthropic Files API.

        Args:
            file_id: The Anthropic Files API file_id.
            hints:   Optional context (e.g., 'SAT Math, Section 3').

        Returns:
            List of question dicts matching the question schema.
        """
        logger.info("pdf_processor.extract.start", extra={"file_id": file_id})

        user_content: list[dict] = [
            {
                "type": "document",
                "source": {"type": "file", "file_id": file_id},
                "title": "Source Document",
            }
        ]
        if hints:
            user_content.append(
                {"type": "text", "text": f"Context / hints: {hints}"}
            )
        user_content.append(
            {
                "type": "text",
                "text": (
                    "Extract every question from this document. "
                    "Return ONLY a valid JSON array — no prose, no markdown fences."
                ),
            }
        )

        # Stream the response to handle large PDFs without timeouts
        full_text = ""
        async with self._client.messages.stream(
            model=settings.llm_model,
            max_tokens=16384,
            system=PDF_EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            betas=["files-api-2025-04-14"],
        ) as stream:
            async for chunk in stream.text_stream:
                full_text += chunk

        questions = _parse_questions(full_text, file_id, "pdf")
        logger.info(
            "pdf_processor.extract.complete",
            extra={"file_id": file_id, "count": len(questions)},
        )
        return questions


async def extract_from_pdf(file_id: str, hints: str = "") -> list[dict[str, Any]]:
    """Module-level convenience function used by the gateway tool dispatcher."""
    processor = PDFProcessor()
    return await processor.extract(file_id=file_id, hints=hints)


def _parse_questions(
    raw_text: str,
    source_ref: str,
    source_type: str,
) -> list[dict[str, Any]]:
    """Parse the LLM's JSON output and attach source metadata."""
    raw_text = raw_text.strip()
    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start == -1 or end == -1:
        logger.warning("pdf_processor.parse.no_json_array", extra={"preview": raw_text[:200]})
        return []

    try:
        questions = json.loads(raw_text[start : end + 1])
    except json.JSONDecodeError as exc:
        logger.error("pdf_processor.parse.json_error", extra={"error": str(exc)})
        return []

    for q in questions:
        q.setdefault("source_ref", source_ref)
        q.setdefault("source_type", source_type)
        # Normalise required fields
        q.setdefault("stimulus", None)
        q.setdefault("stimulus_type", "none")
        q.setdefault("answer_choices", [])
        q.setdefault("correct_answer", "")
        q.setdefault("explanation", "")
        q.setdefault("subject", "General")
        q.setdefault("topics", [])
        q.setdefault("difficulty", 3)

    return questions
