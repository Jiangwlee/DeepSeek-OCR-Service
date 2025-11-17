# app/main.py
import io
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from minio import Minio
from fastapi.responses import JSONResponse

from .converter import convert_with_libreoffice, ConvertError


class MinioSource(BaseModel):
    bucket: str
    object: str


class ConvertRequest(BaseModel):
    source: MinioSource
    to: str = "pdf"
    target_bucket: Optional[str] = None
    target_object: Optional[str] = None

app = FastAPI(title="LibreOffice Doc Convert Service")

def _minio_client() -> Minio:
    endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000").replace("http://", "").replace("https://", "")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"
    access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/convert")
async def convert_document(body: ConvertRequest):
    """
    从 MinIO 下载源文件 -> 转换 -> 上传结果到 MinIO，返回对象信息。
    """
    client = _minio_client()
    target_bucket = body.target_bucket or body.source.bucket
    # 生成目标对象名
    if body.target_object:
        target_object = body.target_object
    else:
        base = body.source.object.rsplit("/", 1)[-1]
        name, _dot, _ext = base.partition(".")
        target_object = f"converted-pdf/{name}.{body.to.lstrip('.')}"

    try:
        # 下载源文件
        resp = client.get_object(body.source.bucket, body.source.object)
        content = resp.read()
        resp.close()
        resp.release_conn()

        source_name = Path(body.source.object).name

        # 转换
        output_bytes = convert_with_libreoffice(content, source_name, body.to)

        # 上传目标
        data_stream = io.BytesIO(output_bytes)
        data_stream.seek(0)
        client.put_object(
            bucket_name=target_bucket,
            object_name=target_object,
            data=data_stream,
            length=len(output_bytes),
            content_type="application/pdf" if body.to.lower() == "pdf" else "application/octet-stream",
        )

        return JSONResponse(
            {
                "bucket": target_bucket,
                "object": target_object,
                "content_type": "application/pdf" if body.to.lower() == "pdf" else "application/octet-stream",
            }
        )
    except ConvertError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:  # pragma: no cover
        # 捕获转换/MinIO 等异常，直接返回消息
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
