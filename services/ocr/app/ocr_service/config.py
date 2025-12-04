from __future__ import annotations

import functools
import json
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized settings for the OCR service."""

    model_config = SettingsConfigDict(env_prefix="OCR_", case_sensitive=False)

    # FastAPI metadata
    app_name: str = "DeepSeek OCR Service"
    app_version: str = "0.1.0"

    # vLLM API endpoint configuration (supports multiple models)
    vllm_api_base: str = "http://deepseek-ocr:8000/v1"
    vllm_api_key: str = "EMPTY"
    vllm_model: str = "deepseek-ai/DeepSeek-OCR"
    vllm_request_timeout: int = 3600
    vllm_max_tokens: int = 2048
    vllm_temperature: float = 0.0
    vllm_skip_special_tokens: bool = False
    vllm_vllm_xargs: Optional[Dict[str, Any]] = Field(
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
    minio_endpoint: str = "http://minio:9000"
    minio_access_key: str = "deepseekadmin"
    minio_secret_key: str = "deepseeksecret"
    minio_secure: bool = False
    minio_region: Optional[str] = None
    # Bucket layout
    minio_default_bucket: str = "ocr-results"  # result bucket (backward compatibility)
    minio_raw_bucket: str = "raw-docs"
    minio_converted_pdf_bucket: str = "converted-pdf"
    minio_page_images_bucket: str = "page-images"
    minio_presign_expiry: int = 3600
    doc_converter_endpoint: str = "http://document-converter:8000"

    # Result persistence
    persist_results_by_default: bool = False

    # UI Configuration
    ui_available_models: str = Field(
        default='[{"name":"DeepSeek-OCR","value":"deepseek","endpoint":"http://ubuntu-mindora.local:8000/v1","model":"deepseek-ai/DeepSeek-OCR"},{"name":"Qwen3-VL-8B-FP8","value":"qwen3-vl","endpoint":"http://ubuntu-mindora.local:8003/v1","model":"Qwen/Qwen3-VL-8B-Instruct-FP8"},{"name":"PaddleOCR","value":"paddle","endpoint":"http://paddle-ocr:9000","model":"paddleocr"}]'
    )
    ui_prompt_presets: str = Field(
        default='[{"name":"Markdown (标准)","value":"<image>\\n<|grounding|>Convert the document to markdown."},{"name":"Markdown (高质量)","value":"<image>\\n<|grounding|>Please convert this document to high-quality markdown format, preserving all formatting, tables, and structure."},{"name":"纯文本","value":"<image>\\nFree OCR."},{"name":"表格提取","value":"<image>\\n<|grounding|>Extract all tables from this document in markdown format."},{"name":"公式识别","value":"<image>\\n<|grounding|>Extract all mathematical formulas using LaTeX notation."}]'
    )

    @field_validator("ui_available_models", "ui_prompt_presets", mode="before")
    @classmethod
    def parse_json_field(cls, v: str) -> str:
        """Validate that the field is valid JSON."""
        if isinstance(v, str):
            try:
                json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {e}")
        return v

    def get_available_models(self) -> List[Dict[str, str]]:
        """Parse and return available models as a list of dicts."""
        return json.loads(self.ui_available_models)

    def get_prompt_presets(self) -> List[Dict[str, str]]:
        """Parse and return prompt presets as a list of dicts."""
        return json.loads(self.ui_prompt_presets)


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()
