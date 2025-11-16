from __future__ import annotations

import io
import uuid

import httpx
from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gradio.routes import mount_gradio_app

from .dependencies import get_orchestrator
from .ocr_service.config import get_settings
from .ocr_service.exceptions import (OCRException, OCRProcessException,
                                     StorageUnavailableError)
from .ocr_service.schemas import (DocumentOptions, DocumentResult,
                                  StorageOCRRequest, URLOCRRequest)
from .ui import build_gradio_interface

app = FastAPI(title="DeepSeek OCR Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"]
)


@app.get("/healthz")
async def health_check():
    settings = get_settings()
    return {
        "status": "ok",
        "deepseek_base": settings.deepseek_api_base,
        "minio_enabled": settings.minio_enabled,
    }


@app.post("/v1/ocr/document/upload", response_model=DocumentResult)
async def ocr_from_upload(
    file: UploadFile = File(...),
    options: DocumentOptions = Depends(DocumentOptions.as_form),
    orchestrator=Depends(get_orchestrator),
):
    return await orchestrator.process_upload(file, options)


@app.post("/v1/ocr/document/from-storage", response_model=DocumentResult)
async def ocr_from_storage(
    request: StorageOCRRequest,
    orchestrator=Depends(get_orchestrator),
):
    storage = orchestrator.storage
    if not storage:
        raise StorageUnavailableError("Storage client not configured")
    data = storage.fetch(request.source.bucket, request.source.object_name)
    filename = request.source.object_name.rsplit("/", 1)[-1]
    return await orchestrator.process_bytes(
        filename=filename,
        content_type=None,
        data=data,
        options=request,
    )


@app.post("/v1/ocr/document/from-url", response_model=DocumentResult)
async def ocr_from_url(
    request: URLOCRRequest,
    orchestrator=Depends(get_orchestrator),
):
    async with httpx.AsyncClient() as client:
        response = await client.get(request.source.url)
        response.raise_for_status()
        content_type = response.headers.get("content-type")
        filename = request.source.url.split("/")[-1] or f"remote-{uuid.uuid4().hex}"
        data = response.content
    return await orchestrator.process_bytes(
        filename=filename,
        content_type=content_type,
        data=data,
        options=request,
    )


@app.exception_handler(OCRException)
async def ocr_exception_handler(_, exc: OCRException):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(Exception)
async def unknown_exception_handler(_, exc: Exception):  # pragma: no cover
    return JSONResponse(status_code=500, content={"error": "Internal Server Error"})


demo = build_gradio_interface(get_orchestrator())
app = mount_gradio_app(app, demo, path="/ui")
