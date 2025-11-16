from __future__ import annotations

import base64
import io
import os
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from paddleocr import PaddleOCR
from PIL import Image

app = FastAPI(title="PaddleOCR REST", version="0.1.0")

_OCR = None


def get_ocr() -> PaddleOCR:
    global _OCR
    if _OCR is None:
        lang = os.getenv("PADDLE_LANG", "ch")
        _OCR = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=os.getenv("PADDLE_USE_GPU", "false").lower() == "true")
    return _OCR


class OCRRequest(BaseModel):
    image: str  # base64 (no prefix)


@app.post("/predict/ocr_system")
def predict(req: OCRRequest):
    try:
        image_bytes = base64.b64decode(req.image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}")

    ocr = get_ocr()
    result = ocr.ocr(image, cls=True)
    texts: List[str] = []
    for line in result:
        for item in line:
            texts.append(item[1][0])
    joined = "\n".join(texts)
    return {"text": joined, "lines": texts}


@app.get("/healthz")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
