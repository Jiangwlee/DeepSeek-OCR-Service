from __future__ import annotations

import base64
import io
import os
from pathlib import Path
import threading
from typing import List, Tuple

import numpy as np
import paddle
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from paddleocr import PaddleOCR, PPStructureV3
from PIL import Image
from urllib.parse import urlparse

app = FastAPI(title="PaddleOCR REST", version="0.1.0")

_OCR = None
_OCR_LOCK = threading.Lock()
_WARMED_UP = False
_WARMUP_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y0XfY4AAAAASUVORK5CYII="
)

USE_GPU = os.getenv("PADDLE_USE_GPU", "false").lower() == "true"

ocrv5_pipepine = PaddleOCR(
    use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
    use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
    use_textline_orientation=False)

structurev3_pipepine = PPStructureV3()

def do_ocr_recognition(
    image_np: np.ndarray,
    output_format: str = "plain_text",
    lang: str = "ch",
    pdf_path: str | None = None,
) -> Tuple[str, List[str]]:
    """
    plain_text: 使用 PaddleOCR det/rec。
    markdown: 使用 PP-StructureV3，推荐输入 PDF（pdf_path）。
    """
    if output_format == "markdown":
        pipeline = structurev3_pipepine
        if pdf_path:
            output = pipeline.predict(input=pdf_path)
        else:
            # 回退单页图片
            output = pipeline.predict(input=image_np)
        markdown_list = []
        markdown_images = []
        for res in output:
            if hasattr(res, "markdown"):
                md_info = res.markdown
                markdown_list.append(md_info)
                markdown_images.append(md_info.get("markdown_images", {}))
        if markdown_list:
            markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)
            return markdown_texts, markdown_texts.splitlines()

    # 回退纯文本
    pipeline = ocrv5_pipepine
    raw = pipeline.ocr(image_np)
    lines: List[str] = []
    for line in raw:
        for item in line:
            lines.append(item[1][0])
    text = "\n".join(lines)
    return text, lines

class OCRRequest(BaseModel):
    image: str | None = None  # base64 (no prefix)
    image_url: str | None = None  # optional URL (e.g., MinIO presigned)
    output_format: str = "plain_text"


@app.post("/predict/ocr_system")
def predict(req: OCRRequest):
    if not req.image and not req.image_url:
        raise HTTPException(status_code=400, detail="image or image_url is required")
    pdf_path = None
    try:
        if req.image_url:
            import requests

            resp = requests.get(req.image_url, timeout=30)
            resp.raise_for_status()
            image_bytes = resp.content
            path = urlparse(req.image_url).path.lower()
            content_type = resp.headers.get("content-type", "").lower()
            is_pdf = path.endswith(".pdf") or content_type.startswith("application/pdf")
            if is_pdf:
                import tempfile

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(image_bytes)
                tmp.flush()
                tmp.close()
                pdf_path = tmp.name
                # PDF 输入仅支持 markdown 流程，避免将 PDF 按图片解码
                if req.output_format != "markdown":
                    raise HTTPException(status_code=400, detail="PDF input requires output_format=markdown")
                text, lines = do_ocr_recognition(
                    image_np=np.zeros((1, 1, 3), dtype=np.uint8),
                    output_format=req.output_format,
                    lang=os.getenv("PADDLE_LANG", "ch"),
                    pdf_path=pdf_path,
                )
                return {"text": text, "lines": lines}
        else:
            image_bytes = base64.b64decode(req.image)  # type: ignore[arg-type]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}")

    with _OCR_LOCK:
        text, lines = do_ocr_recognition(
            image_np=image_np,
            output_format=req.output_format,
            lang=os.getenv("PADDLE_LANG", "ch"),
            pdf_path=pdf_path,
        )
    return {"text": text, "lines": lines}


@app.get("/healthz")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
