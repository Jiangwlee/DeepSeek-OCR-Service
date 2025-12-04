from __future__ import annotations

import asyncio
import base64
import logging
from typing import Iterable, List, Optional

from openai import AsyncOpenAI
from PIL import Image
from urllib.parse import quote

from .config import Settings
from .exceptions import OCRProcessException

logger = logging.getLogger(__name__)


class DeepSeekOCRClient:
    def __init__(self, settings: Settings, api_base: Optional[str] = None, model: Optional[str] = None):
        self._settings = settings
        self._api_base = api_base or settings.vllm_api_base
        self._model = model or settings.vllm_model
        self._client = AsyncOpenAI(
            api_key=settings.vllm_api_key,
            base_url=self._api_base,
            timeout=settings.vllm_request_timeout,
        )
        self._semaphore = asyncio.Semaphore(settings.max_workers)

    async def process_images(
        self,
        images: Iterable[Image.Image] | Iterable[str],
        prompt: str,
        *,
        skip_special_tokens: bool,
    ) -> List[str]:
        tasks = [
            self._process_single(
                image,
                prompt,
                idx,
                skip_special_tokens=skip_special_tokens,
            )
            for idx, image in enumerate(images)
        ]
        return await asyncio.gather(*tasks)

    async def _process_single(
        self,
        image: Image.Image | str,
        prompt: str,
        index: int,
        *,
        skip_special_tokens: bool,
    ) -> str:
        async with self._semaphore:
            if isinstance(image, Image.Image):
                image_b64 = await asyncio.get_running_loop().run_in_executor(
                    None, self._image_to_base64, image
                )
                image_url_payload = {"url": f"data:image/png;base64,{image_b64}"}
            else:
                # assume already a URL
                image_url_payload = {"url": image}
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": image_url_payload,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=self._settings.vllm_max_tokens,
                    temperature=self._settings.vllm_temperature,
                    extra_body={
                        "skip_special_tokens": skip_special_tokens,
                        "vllm_xargs": self._settings.vllm_vllm_xargs,
                    },
                )
            except Exception as exc:  # pragma: no cover
                logger.exception("vLLM OCR request failed for page %s (endpoint: %s, model: %s)",
                                index, self._api_base, self._model)
                raise OCRProcessException(str(exc)) from exc

            content = response.choices[0].message.content
            if not content:
                raise OCRProcessException("Empty OCR response")

            return content

    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
