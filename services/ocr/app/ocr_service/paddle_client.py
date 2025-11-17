from __future__ import annotations

import asyncio
import logging
from typing import Iterable, List, Union

import httpx

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

    async def process_images(self, images: Iterable[Union[str, bytes]], output_format: str = "plain_text") -> List[str]:
        # images: can be presigned URLs (str) or raw bytes (will be base64 encoded)
        tasks = [self._process_single(img, idx, output_format=output_format) for idx, img in enumerate(images)]
        return await asyncio.gather(*tasks)

    async def _process_single(self, image: Union[str, bytes], index: int, *, output_format: str) -> str:
        if isinstance(image, str):
            payload = {"image_url": image, "output_format": output_format}
        else:
            img_b64 = await asyncio.get_running_loop().run_in_executor(None, self._to_base64, image)
            payload = {"image": img_b64, "output_format": output_format}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._settings.paddle_endpoint.rstrip("/") + "/predict/ocr_system", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:  # pragma: no cover
            body = ""
            try:
                body = f" body={exc.response.text}"
            except Exception:
                body = ""
            logger.exception("PaddleOCR request failed for page %s", index)
            raise OCRProcessException(f"{exc}{body}") from exc
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
    def _to_base64(image_bytes: bytes) -> str:
        import base64

        return base64.b64encode(image_bytes).decode("utf-8")
