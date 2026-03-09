from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "ML Test Tutor"
    debug: bool = False

    # Anthropic / LLM
    anthropic_api_key: str
    llm_model: str = "claude-opus-4-6"

    # Database
    database_url: str = "postgresql+asyncpg://tutor:tutor@localhost:5432/tutor"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Object storage (MinIO / S3)
    storage_endpoint: str = "http://localhost:9000"
    storage_access_key: str = "minioadmin"
    storage_secret_key: str = "minioadmin"
    storage_bucket: str = "tutor-documents"
    storage_use_ssl: bool = False

    # Auth
    clerk_jwt_issuer: str = ""

    # Ingestion
    max_file_size_mb: int = 50
    ingestion_chunk_size: int = 4000  # chars per chunk for large docs

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
