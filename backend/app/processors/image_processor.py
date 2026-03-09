"""
Image Processor — OCR and extract questions from image files.

Supports PNG, JPEG, WEBP, GIF. Uses Claude's vision capabilities to:
1. Perform OCR on handwritten or printed text.
2. Understand diagrams, charts, and mathematical notation embedded in images.
3. Extract structured question data.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

from ..core.config import get_settings
from ..orchestration.prompts import IMAGE_EXTRACTION_SYSTEM

logger = logging.getLogger(__name__)
settings = get_settings()

SUPPORTED_MEDIA_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}


class ImageProcessor:
    """
    Extracts questions from image files using Claude's multimodal vision.

    The image is referenced via Anthropic Files API file_id. Claude reads
    the image directly, performing OCR and understanding visual content
    (diagrams, graphs, mathematical expressions).
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
        Extract questions from an image already uploaded to the Anthropic Files API.

        Args:
            file_id: The Anthropic Files API file_id.
            hints:   Optional context (e.g., 'Chemistry multiple choice, Grade 11').

        Returns:
            List of question dicts matching the question schema.
        """
        logger.info("image_processor.extract.start", extra={"file_id": file_id})

        user_content: list[dict] = [
            {
                "type": "image",
                "source": {"type": "file", "file_id": file_id},
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
                    "Carefully OCR and extract every question from this image. "
                    "For diagrams or figures, describe them in the stimulus field. "
                    "Return ONLY a valid JSON array — no prose, no markdown fences."
                ),
            }
        )

        full_text = ""
        async with self._client.messages.stream(
            model=settings.llm_model,
            max_tokens=8192,
            system=IMAGE_EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            betas=["files-api-2025-04-14"],
        ) as stream:
            async for chunk in stream.text_stream:
                full_text += chunk

        questions = _parse_questions(full_text, file_id, "image")
        logger.info(
            "image_processor.extract.complete",
            extra={"file_id": file_id, "count": len(questions)},
        )
        return questions


class TextProcessor:
    """Extracts questions from raw pasted text or markdown."""

    def __init__(self):
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def extract(
        self,
        *,
        text: str,
        hints: str = "",
    ) -> list[dict[str, Any]]:
        """
        Parse questions from plain text / markdown input.

        Args:
            text:  Raw text containing questions.
            hints: Optional context hints.

        Returns:
            List of question dicts.
        """
        logger.info("text_processor.extract.start", extra={"length": len(text)})

        prompt = (
            "Extract every question from the text below. "
            "Return ONLY a valid JSON array — no prose, no markdown fences.\n\n"
            + (f"Hints: {hints}\n\n" if hints else "")
            + text
        )

        from ..orchestration.prompts import PDF_EXTRACTION_SYSTEM

        full_text = ""
        async with self._client.messages.stream(
            model=settings.llm_model,
            max_tokens=8192,
            system=PDF_EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for chunk in stream.text_stream:
                full_text += chunk

        questions = _parse_questions(full_text, "text_input", "text")
        logger.info(
            "text_processor.extract.complete",
            extra={"count": len(questions)},
        )
        return questions


# Module-level convenience functions used by the gateway tool dispatcher


async def extract_from_image(file_id: str, hints: str = "") -> list[dict[str, Any]]:
    processor = ImageProcessor()
    return await processor.extract(file_id=file_id, hints=hints)


async def extract_from_text(text: str, hints: str = "") -> list[dict[str, Any]]:
    processor = TextProcessor()
    return await processor.extract(text=text, hints=hints)


def _parse_questions(
    raw_text: str,
    source_ref: str,
    source_type: str,
) -> list[dict[str, Any]]:
    """Parse JSON output and attach source metadata."""
    raw_text = raw_text.strip()
    start = raw_text.find("[")
    end = raw_text.rfind("]")
    if start == -1 or end == -1:
        logger.warning(
            "image_processor.parse.no_json_array",
            extra={"preview": raw_text[:200]},
        )
        return []

    try:
        questions = json.loads(raw_text[start : end + 1])
    except json.JSONDecodeError as exc:
        logger.error("image_processor.parse.json_error", extra={"error": str(exc)})
        return []

    for q in questions:
        q.setdefault("source_ref", source_ref)
        q.setdefault("source_type", source_type)
        q.setdefault("stimulus", None)
        q.setdefault("stimulus_type", "none")
        q.setdefault("answer_choices", [])
        q.setdefault("correct_answer", "")
        q.setdefault("explanation", "")
        q.setdefault("subject", "General")
        q.setdefault("topics", [])
        q.setdefault("difficulty", 3)

    return questions
