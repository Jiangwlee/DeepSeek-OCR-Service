from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional
from datetime import timedelta

from minio import Minio
from urllib.parse import urlparse

from .exceptions import StorageUnavailableError


@dataclass
class StorageConfig:
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool = False
    region: Optional[str] = None


class StorageClient:
    def __init__(self, config: StorageConfig):
        self._config = config
        parsed = urlparse(config.endpoint)
        if parsed.scheme:
            endpoint = parsed.netloc
            secure = parsed.scheme == "https"
        else:
            endpoint = config.endpoint
            secure = config.secure

        self._client = Minio(
            endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=secure,
            region=config.region,
        )

    def fetch(self, bucket: str, object_name: str) -> bytes:
        try:
            response = self._client.get_object(bucket, object_name)
            data = response.read()
            response.close()
            response.release_conn()
            return data
        except Exception as exc:  # pragma: no cover
            raise StorageUnavailableError(str(exc)) from exc

    def upload_text(self, bucket: str, object_name: str, text: str, content_type: str) -> None:
        data = text.encode("utf-8")
        self._client.put_object(
            bucket,
            object_name,
            data=io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

    def upload_bytes(self, bucket: str, object_name: str, data: bytes, content_type: Optional[str] = None) -> None:
        self._client.put_object(
            bucket,
            object_name,
            data=io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

    def presign_get(self, bucket: str, object_name: str, expires: int = 3600) -> str:
        try:
            exp = timedelta(seconds=expires) if isinstance(expires, int) else expires
            return self._client.presigned_get_object(bucket, object_name, expires=exp)
        except Exception as exc:  # pragma: no cover
            raise StorageUnavailableError(str(exc)) from exc
