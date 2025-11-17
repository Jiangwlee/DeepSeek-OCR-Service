from functools import lru_cache
from typing import Optional

from .ocr_service.config import Settings, get_settings
from .ocr_service.converter_client import DocumentConverterClient
from .ocr_service.ocr_client import DeepSeekOCRClient
from .ocr_service.paddle_client import PaddleOCRClient
from .ocr_service.service import OCROrchestrator
from .ocr_service.storage import StorageClient, StorageConfig


def _build_storage(settings: Settings) -> Optional[StorageClient]:
    if not settings.minio_enabled:
        return None
    return StorageClient(
        StorageConfig(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
            region=settings.minio_region,
        )
    )


@lru_cache
def get_orchestrator() -> OCROrchestrator:
    settings = get_settings()
    ocr_client = DeepSeekOCRClient(settings)
    paddle_client = PaddleOCRClient(settings)
    converter_client = DocumentConverterClient(settings)
    storage = _build_storage(settings)
    return OCROrchestrator(settings, ocr_client, storage, paddle_client, converter_client)
