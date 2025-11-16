from __future__ import annotations

import functools
from typing import Any, Dict, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized settings for the OCR service."""

    model_config = SettingsConfigDict(env_prefix="OCR_", case_sensitive=False)

    # FastAPI metadata
    app_name: str = "DeepSeek OCR Service"
    app_version: str = "0.1.0"

    # DeepSeek OCR endpoint configuration
    deepseek_api_base: str = "http://ubuntu-mindora.local:8000/v1"
    deepseek_api_key: str = "EMPTY"
    deepseek_model: str = "deepseek-ai/DeepSeek-OCR"
    deepseek_request_timeout: int = 3600
    deepseek_max_tokens: int = 2048
    deepseek_temperature: float = 0.0
    deepseek_skip_special_tokens: bool = False
    deepseek_vllm_xargs: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "ngram_size": 30,
            "window_size": 90,
            "whitelist_token_ids": [128821, 128822],
        }
    )

    # PaddleOCR endpoint（CPU/GPU 由服务端决定）
    paddle_endpoint: str = "http://paddle-ocr:9000"
    paddle_timeout: int = 60

    # Processing knobs
    pdf_dpi: int = 144
    max_workers: int = 4
    max_pages: int = 200
    default_prompt_markdown: str = "<image>\n<|grounding|>Convert the document to markdown."
    default_prompt_plain: str = "<image>\nFree OCR."

    # Storage / MinIO
    minio_enabled: bool = True
    minio_endpoint: str = "http://localhost:9000"
    minio_access_key: str = "deepseekadmin"
    minio_secret_key: str = "deepseeksecret"
    minio_secure: bool = False
    minio_region: Optional[str] = None
    minio_default_bucket: str = "ocr-results"

    # Result persistence
    persist_results_by_default: bool = False


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()
