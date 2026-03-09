"""
Document Router — accepts a raw file upload, pushes it to the Anthropic
Files API, and routes it to the correct processor via the IngestOrchestrationGateway.

This is the single entry point for all document ingestion.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import anthropic

from ..core.config import get_settings
from ..orchestration.gateway import IngestOrchestrationGateway
from .pdf_processor import extract_from_pdf
from .image_processor import extract_from_image, extract_from_text

logger = logging.getLogger(__name__)
settings = get_settings()

# Map of file extensions → MIME types accepted by the Files API
_MIME_MAP = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".txt": "text/plain",
    ".md": "text/plain",
}

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_PDF_EXTS = {".pdf"}
_TEXT_EXTS = {".txt", ".md"}


class DocumentRouter:
    """
    Routes an incoming file to the appropriate LLM-powered processor.

    Steps:
      1. Detect file type by extension / MIME.
      2. Upload file to Anthropic Files API (single upload, reusable file_id).
      3. Hand off to IngestOrchestrationGateway with detected source_type.
      4. Return extracted questions + the Files API file_id for storage.
    """

    def __init__(self):
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._gateway = IngestOrchestrationGateway(
            pdf_processor_fn=extract_from_pdf,
            image_processor_fn=extract_from_image,
            text_processor_fn=extract_from_text,
        )

    async def route(
        self,
        *,
        file_bytes: bytes,
        filename: str,
        hints: str = "",
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Full pipeline: upload → detect → extract.

        Args:
            file_bytes: Raw file content.
            filename:   Original filename (used to detect file type).
            hints:      Optional context hints passed to the gateway.
            job_id:     Optional tracking ID.

        Returns:
            {
                "file_id": "<Anthropic Files API file_id>",
                "source_type": "pdf" | "image" | "text",
                "questions": [<question dicts>],
            }
        """
        # Validate size
        max_bytes = settings.max_file_size_mb * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise ValueError(
                f"File too large: {len(file_bytes) / 1024 / 1024:.1f} MB "
                f"(max {settings.max_file_size_mb} MB)"
            )

        ext = os.path.splitext(filename)[1].lower()
        mime_type = _MIME_MAP.get(ext)
        if mime_type is None:
            raise ValueError(f"Unsupported file type: '{ext}'")

        # Determine source_type and decide whether to upload to Files API
        if ext in _TEXT_EXTS:
            source_type = "text"
            raw_text = file_bytes.decode("utf-8", errors="replace")
            questions = await self._gateway.ingest(
                source_type="text",
                raw_text=raw_text,
                hints=hints,
                job_id=job_id,
            )
            return {"file_id": None, "source_type": source_type, "questions": questions}

        # Upload PDF / image to Anthropic Files API
        source_type = "pdf" if ext in _PDF_EXTS else "image"
        logger.info(
            "document_router.upload.start",
            extra={"filename": filename, "source_type": source_type},
        )

        uploaded_file = await self._client.beta.files.upload(
            file=(filename, file_bytes, mime_type),
        )
        file_id = uploaded_file.id
        logger.info(
            "document_router.upload.complete",
            extra={"file_id": file_id},
        )

        questions = await self._gateway.ingest(
            source_type=source_type,
            file_id=file_id,
            hints=hints,
            job_id=job_id,
        )

        return {
            "file_id": file_id,
            "source_type": source_type,
            "questions": questions,
        }
