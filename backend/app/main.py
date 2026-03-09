"""
ML Test Tutor — FastAPI application entry point.

Startup:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import get_settings
from .core.database import init_db
from .api.routes.ingest import router as ingest_router
from .api.routes.questions import router as questions_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
settings = get_settings()

app = FastAPI(
    title="ML Test Tutor API",
    description=(
        "Adaptive exam-prep platform powered by Claude. "
        "Ingests PDFs, images, and text via LLM orchestration; "
        "serves smart question retrieval for practice sessions."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(ingest_router)
app.include_router(questions_router)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    logger.info("app.startup: initialising database tables...")
    await init_db()
    logger.info("app.startup: ready.")


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "service": "ml-test-tutor-api"}


@app.get("/", tags=["health"])
async def root():
    return {
        "service": "ML Test Tutor API",
        "docs": "/docs",
        "ingestion": "/api/ingest/upload",
        "questions": "/api/questions",
        "session": "/api/questions/session",
    }
