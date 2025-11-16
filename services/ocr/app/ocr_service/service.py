from __future__ import annotations

import asyncio
import uuid
from typing import Iterable, List, Optional

from fastapi import UploadFile

from .config import Settings
from .exceptions import (FileNotProvidedError, OCRProcessException,
                         StorageUnavailableError)
from .ocr_client import DeepSeekOCRClient
from .paddle_client import PaddleOCRClient
from .processors import FilePayload, ProcessorFactory
from .schemas import DocumentOptions, DocumentResult, PageResult
from .storage import StorageClient


class OCROrchestrator:
    def __init__(
        self,
        settings: Settings,
        ocr_client: DeepSeekOCRClient,
        storage_client: Optional[StorageClient] = None,
        paddle_client: Optional[PaddleOCRClient] = None,
    ):
        self._settings = settings
        self._ocr_client = ocr_client
        self._paddle_client = paddle_client
        self._storage = storage_client

    @property
    def storage(self) -> Optional[StorageClient]:
        return self._storage

    async def process_upload(self, file: UploadFile, options: DocumentOptions) -> DocumentResult:
        if file is None:
            raise FileNotProvidedError("file is required")
        data = await file.read()
        payload = FilePayload(filename=file.filename or "upload", content_type=file.content_type, data=data)
        return await self._process_payload(payload, options)

    async def process_bytes(self, *, filename: str, content_type: Optional[str], data: bytes, options: DocumentOptions) -> DocumentResult:
        payload = FilePayload(filename=filename, content_type=content_type, data=data)
        return await self._process_payload(payload, options)

    async def _process_payload(self, payload: FilePayload, options: DocumentOptions) -> DocumentResult:
        processor = ProcessorFactory.get_processor(payload, self._settings)
        images = await processor.to_images(payload)
        prompt = self._resolve_prompt(options)
        if options.provider == "deepseek":
            skip_special = self._should_skip_special_tokens(options)
            texts = await self._ocr_client.process_images(
                images,
                prompt,
                skip_special_tokens=skip_special,
            )
        elif options.provider == "paddle":
            if not self._paddle_client:
                raise OCRProcessException("PaddleOCR client is not configured")
            texts = await self._paddle_client.process_images(images)
        else:
            raise OCRProcessException(f"Unknown provider: {options.provider}")

        combined = "\n\n".join(texts)
        pages = [PageResult(index=i, text=text) for i, text in enumerate(texts)]

        stored_bucket = None
        stored_object = None
        if options.store_result or self._settings.persist_results_by_default:
            stored_bucket, stored_object = await self._maybe_store_result(
                combined, options, payload.filename
            )

        return DocumentResult(
            output_format=options.output_format,
            prompt_used=prompt,
            total_pages=len(pages),
            pages=pages,
            combined_text=combined,
            stored_bucket=stored_bucket,
            stored_object_name=stored_object,
        )

    def _resolve_prompt(self, options: DocumentOptions) -> str:
        if options.prompt:
            return options.prompt
        if options.output_format == "plain_text":
            return self._settings.default_prompt_plain
        return self._settings.default_prompt_markdown

    def _should_skip_special_tokens(self, options: DocumentOptions) -> bool:
        if options.output_format == "plain_text":
            return True
        return self._settings.deepseek_skip_special_tokens

    async def _maybe_store_result(
        self,
        text: str,
        options: DocumentOptions,
        original_filename: str,
    ) -> tuple[Optional[str], Optional[str]]:
        if not self._storage:
            raise StorageUnavailableError("Storage client is not configured")

        bucket = options.store_bucket or self._settings.minio_default_bucket
        object_name = options.store_object_name or self._build_object_name(original_filename, options.output_format)

        content_type = "text/markdown" if options.output_format == "markdown" else "text/plain"
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self._storage.upload_text,
            bucket,
            object_name,
            text,
            content_type,
        )
        return bucket, object_name

    def _build_object_name(self, filename: str, output_format: str) -> str:
        stem = filename.rsplit(".", 1)[0]
        suffix = "md" if output_format == "markdown" else "txt"
        return f"ocr/{stem}-{uuid.uuid4().hex[:8]}.{suffix}"
