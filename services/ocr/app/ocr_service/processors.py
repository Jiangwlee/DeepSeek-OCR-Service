from __future__ import annotations

import asyncio
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import fitz  # type: ignore
from PIL import Image

from .exceptions import UnsupportedFileTypeError
from .config import Settings


@dataclass
class FilePayload:
    filename: str
    content_type: str | None
    data: bytes

    @property
    def suffix(self) -> str:
        return self.filename.split(".")[-1].lower() if "." in self.filename else ""


class BaseProcessor(ABC):
    def __init__(self, settings: Settings):
        self.settings = settings

    @abstractmethod
    async def to_images(self, payload: FilePayload) -> List[Image.Image]:
        raise NotImplementedError


class PDFProcessor(BaseProcessor):
    async def to_images(self, payload: FilePayload) -> List[Image.Image]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._convert_sync, payload.data)

    def _convert_sync(self, data: bytes) -> List[Image.Image]:
        images: List[Image.Image] = []
        pdf_document = fitz.open(stream=data, filetype="pdf")
        zoom = self.settings.pdf_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        if pdf_document.page_count > self.settings.max_pages:
            raise ValueError("PDF has more pages than allowed")

        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            buffer = pixmap.tobytes("png")
            image = Image.open(io.BytesIO(buffer))
            images.append(image.convert("RGB"))

        pdf_document.close()
        return images


class ImageProcessor(BaseProcessor):
    async def to_images(self, payload: FilePayload) -> List[Image.Image]:
        loop = asyncio.get_running_loop()
        return [await loop.run_in_executor(None, self._load, payload.data)]

    def _load(self, data: bytes) -> Image.Image:
        image = Image.open(io.BytesIO(data))
        return image.convert("RGB")


class ProcessorFactory:
    IMAGE_MIME_PREFIX = "image/"
    IMAGE_SUFFIXES = {"png", "jpg", "jpeg", "webp", "bmp"}
    DOC_SUFFIXES = {"doc", "docx"}

    @staticmethod
    def get_processor(payload: FilePayload, settings: Settings) -> BaseProcessor:
        if payload.content_type and payload.content_type.startswith(ProcessorFactory.IMAGE_MIME_PREFIX):
            return ImageProcessor(settings)

        suffix = payload.suffix
        if suffix == "pdf":
            return PDFProcessor(settings)
        if suffix in ProcessorFactory.IMAGE_SUFFIXES:
            return ImageProcessor(settings)
        if suffix in ProcessorFactory.DOC_SUFFIXES:
            # 占位：doc/docx 会在 orchestrator 层转 pdf 再选 processor，这里抛错避免误用
            raise UnsupportedFileTypeError("doc/docx should be converted to pdf before processing")
        raise UnsupportedFileTypeError(f"Unsupported file type: {suffix or payload.content_type}")
