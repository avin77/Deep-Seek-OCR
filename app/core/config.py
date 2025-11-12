"""Application-wide configuration settings."""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings object loaded from env vars or defaults."""

    deepseek_base_url: str = "http://localhost:8000/v1"
    deepseek_model: str = "deepseek-ai/DeepSeek-OCR"
    deepseek_api_key: str = "local-placeholder"
    request_timeout_seconds: int = 120
    max_request_retries: int = 3
    pdf_raster_dpi: int = 220
    allowed_origins: List[str] = ["*"]
    temp_dir: str = "temp_files"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance for dependency injection."""

    return Settings()
