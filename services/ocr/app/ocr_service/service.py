from __future__ import annotations

import asyncio
import uuid
from typing import Iterable, List, Optional, Tuple

from fastapi import UploadFile
import httpx
from PIL import Image
from pathlib import Path

from .config import Settings
from .exceptions import (FileNotProvidedError, OCRProcessException,
                         StorageUnavailableError)
from .converter_client import DocumentConverterClient
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
        converter_client: Optional[DocumentConverterClient] = None,
    ):
        self._settings = settings
        self._ocr_client = ocr_client
        self._paddle_client = paddle_client
        self._storage = storage_client
        self._converter = converter_client

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

    async def process_storage(self, request) -> DocumentResult:
        if not self._storage:
            raise StorageUnavailableError("Storage client not configured")
        data = self._storage.fetch(request.source.bucket, request.source.object_name)
        filename = request.source.object_name.rsplit("/", 1)[-1]
        payload = FilePayload(filename=filename, content_type=None, data=data)
        return await self._process_payload(payload, request)

    async def process_url(self, request) -> DocumentResult:
        url = str(request.source.url)
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type")
            filename = url.split("/")[-1] or f"remote-{uuid.uuid4().hex}"
            data = response.content
        payload = FilePayload(filename=filename, content_type=content_type, data=data)
        return await self._process_payload(payload, request)

    async def _process_payload(self, payload: FilePayload, options: DocumentOptions, *, task_id: Optional[str] = None) -> DocumentResult:
        task_id = task_id or uuid.uuid4().hex
        # Upload original to MinIO
        stored_orig_bucket = None
        stored_orig_object = None
        pdf_bucket = None
        pdf_object = None
        if self._storage and self._settings.minio_enabled:
            stored_orig_bucket, stored_orig_object = await self._store_original(payload, task_id)
        else:
            raise OCRProcessException("MinIO/storage is required for processing")

        # Convert doc/docx via converter to PDF first
        if payload.suffix in {"doc", "docx"}:
            if not self._converter or not self._storage or not self._settings.minio_enabled:
                raise OCRProcessException("MinIO/converter is required for doc/docx")
            stem = Path(payload.filename).stem
            pdf_bucket, pdf_object = await self._converter.convert(
                source_bucket=stored_orig_bucket or self._settings.minio_raw_bucket,
                source_object=stored_orig_object or f"raw-docs/{task_id}-{payload.filename}",
                target_bucket=self._settings.minio_converted_pdf_bucket,
                to="pdf",
                target_object=f"{task_id}-{stem}.pdf",
            )
            pdf_bytes = self._storage.fetch(pdf_bucket, pdf_object)
            payload = FilePayload(
                filename=pdf_object.rsplit("/", 1)[-1],
                content_type="application/pdf",
                data=pdf_bytes,
            )

        # Select processor after potential conversion
        processor = ProcessorFactory.get_processor(payload, self._settings)

        # Convert to images (in-memory, then store)
        images = await processor.to_images(payload)
        image_urls = None
        image_keys = None
        if self._storage and self._settings.minio_enabled:
            image_urls, image_keys = await self._store_and_presign_images(
                images,
                task_id=task_id,
                target_bucket=self._settings.minio_page_images_bucket,
            )

        prompt = self._resolve_prompt(options)
        # Paddle markdown 结构化：优先用 PDF URL
        paddled_markdown_pdf_url = None
        if options.provider == "paddle" and options.output_format == "markdown" and pdf_bucket and pdf_object and self._storage:
            paddled_markdown_pdf_url = self._storage.presign_get(
                pdf_bucket, pdf_object, expires=self._settings.minio_presign_expiry
            )

        if options.provider == "deepseek":
            skip_special = self._should_skip_special_tokens(options)
            texts = await self._ocr_client.process_images(
                image_urls if image_urls else images,
                prompt,
                skip_special_tokens=skip_special,
            )
        elif options.provider == "paddle":
            if not self._paddle_client:
                raise OCRProcessException("PaddleOCR client is not configured")
            if paddled_markdown_pdf_url:
                texts = await self._paddle_client.process_images([paddled_markdown_pdf_url], output_format=options.output_format)
            else:
                texts = await self._paddle_client.process_images(
                    image_urls if image_urls else images, output_format=options.output_format
                )
        else:
            raise OCRProcessException(f"Unknown provider: {options.provider}")

        combined = "\n\n".join(texts)
        pages = [PageResult(index=i, text=text) for i, text in enumerate(texts)]

        stored_bucket = None
        stored_object = None
        if options.store_result or self._settings.persist_results_by_default:
            stored_bucket, stored_object = await self._maybe_store_result(
                combined, options, payload.filename, task_id
            )

        return DocumentResult(
            output_format=options.output_format,
            prompt_used=prompt,
            total_pages=len(pages),
            pages=pages,
            combined_text=combined,
            stored_bucket=stored_bucket,
            stored_object_name=stored_object,
            source_bucket=stored_orig_bucket,
            source_object_name=stored_orig_object,
            converted_pdf_bucket=pdf_bucket,
            converted_pdf_object=pdf_object,
            image_bucket=self._settings.minio_page_images_bucket if image_keys else None,
            image_objects=image_keys,
        )

    def _resolve_prompt(self, options: DocumentOptions) -> str:
        if options.prompt:
            return options.prompt
        if options.output_format == "plain_text":
            return self._settings.default_prompt_plain
        return self._settings.default_prompt_markdown

    def _should_skip_special_tokens(self, options: DocumentOptions) -> bool:
        if options.output_format == "plain_text" or options.output_format == "markdown":
            return True
        return self._settings.deepseek_skip_special_tokens

    async def _maybe_store_result(
        self,
        text: str,
        options: DocumentOptions,
        original_filename: str,
        task_id: str,
    ) -> tuple[Optional[str], Optional[str]]:
        if not self._storage:
            raise StorageUnavailableError("Storage client is not configured")

        bucket = options.store_bucket or self._settings.minio_default_bucket
        object_name = options.store_object_name or self._build_object_name(original_filename, options.output_format, task_id)

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

    def _build_object_name(self, filename: str, output_format: str, task_id: str) -> str:
        stem = filename.rsplit(".", 1)[0]
        suffix = "md" if output_format == "markdown" else "txt"
        return f"ocr-results/{task_id}/{stem}.{suffix}"

    async def _store_original(self, payload: FilePayload, task_id: str) -> Tuple[Optional[str], Optional[str]]:
        if not self._storage:
            return None, None
        # Always persist raw uploads into the configured raw bucket; user-facing store_bucket
        # option only controls where OCR results are saved.
        bucket = self._settings.minio_raw_bucket
        object_name = f"{task_id}-{payload.filename}"
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self._storage.upload_bytes,
            bucket,
            object_name,
            payload.data,
            payload.content_type,
        )
        return bucket, object_name

    async def _store_and_presign_images(self, images: Iterable[Image.Image], task_id: str, target_bucket: str) -> Tuple[List[str], List[str]]:
        if not self._storage:
            return [], []
        bucket = target_bucket
        loop = asyncio.get_running_loop()
        urls: List[str] = []
        keys: List[str] = []
        for idx, img in enumerate(images):
            buf = await loop.run_in_executor(None, self._image_to_png_bytes, img)
            object_name = f"{task_id}/page-{idx+1}.png"
            await loop.run_in_executor(
                None,
                self._storage.upload_bytes,
                bucket,
                object_name,
                buf,
                "image/png",
            )
            url = self._storage.presign_get(bucket, object_name, expires=self._settings.minio_presign_expiry)
            urls.append(url)
            keys.append(object_name)
        return urls, keys

    @staticmethod
    def _image_to_png_bytes(img: Image.Image) -> bytes:
        import io

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
