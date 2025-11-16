# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from .converter import convert_with_libreoffice, ConvertError

app = FastAPI(title="LibreOffice Doc Convert Service")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/convert")
async def convert_document(
    file: UploadFile = File(...),
    to: str = Query(..., description="目标格式，例如：pdf / odt / txt 等"),
):
    """
    上传一个文件，通过 ?to=pdf 指定目标格式，返回转换后的文件。
    """
    try:
        content = await file.read()
        output_bytes = convert_with_libreoffice(content, file.filename, to)

        out_filename = f"{file.filename.rsplit('.', 1)[0]}.{to.lstrip('.')}"
        media_type = "application/pdf" if to.lower() == "pdf" else "application/octet-stream"

        return StreamingResponse(
            iter([output_bytes]),
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{out_filename}"'
            },
        )
    except ConvertError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")