from __future__ import annotations

import asyncio
import base64
import logging
from typing import Iterable, List

import httpx
from PIL import Image

from .config import Settings
from .exceptions import OCRProcessException

logger = logging.getLogger(__name__)


class PaddleOCRClient:
    """Thin HTTP client for PaddleOCR Serving.

    Assumes an endpoint that accepts JSON {"image": <base64>} and returns
    {"text": "..."} or a list of results. Adjust parsing if your deployment differs.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._timeout = settings.paddle_timeout

    async def process_images(self, images: Iterable[Image.Image]) -> List[str]:
        tasks = [self._process_single(img, idx) for idx, img in enumerate(images)]
        return await asyncio.gather(*tasks)

    async def _process_single(self, image: Image.Image, index: int) -> str:
        img_b64 = await asyncio.get_running_loop().run_in_executor(None, self._to_base64, image)
        payload = {"image": img_b64}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._settings.paddle_endpoint.rstrip("/") + "/predict/ocr_system", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # pragma: no cover
            logger.exception("PaddleOCR request failed for page %s", index)
            raise OCRProcessException(str(exc)) from exc

        # 兼容多种返回格式
        if isinstance(data, dict) and "text" in data:
            return data["text"]
        if isinstance(data, dict) and "result" in data:
            return str(data["result"])
        if isinstance(data, list) and data:
            return str(data[0])
        raise OCRProcessException("Unexpected PaddleOCR response format")

    @staticmethod
    def _to_base64(image: Image.Image) -> str:
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
