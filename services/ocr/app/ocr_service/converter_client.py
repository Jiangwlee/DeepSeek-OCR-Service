from __future__ import annotations

import httpx
from typing import Optional

from .config import Settings
from .exceptions import OCRProcessException, StorageUnavailableError


class DocumentConverterClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._endpoint = settings.doc_converter_endpoint.rstrip("/")

    async def convert(
        self,
        *,
        source_bucket: str,
        source_object: str,
        target_bucket: Optional[str],
        to: str = "pdf",
        target_object: Optional[str] = None,
    ) -> tuple[str, str]:
        payload = {
            "source": {"bucket": source_bucket, "object": source_object},
            "to": to,
        }
        if target_bucket:
            payload["target_bucket"] = target_bucket
        if target_object:
            payload["target_object"] = target_object
        try:
            async with httpx.AsyncClient(timeout=600) as client:
                resp = await client.post(f"{self._endpoint}/convert", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["bucket"], data["object"]
        except Exception as exc:
            msg = ""
            if isinstance(exc, httpx.HTTPStatusError):
                try:
                    msg = f" body={exc.response.text}"
                except Exception:
                    msg = ""
            raise OCRProcessException(f"Convert failed: {exc}{msg}") from exc
