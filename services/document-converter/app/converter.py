# app/converter.py
import os
import shutil
import subprocess
import uuid
from pathlib import Path

class ConvertError(Exception):
    pass

def convert_with_libreoffice(input_bytes: bytes, filename: str, target_ext: str) -> bytes:
    """
    使用 LibreOffice 将任意支持的格式转换为 target_ext（如 pdf）。
    返回目标文件的二进制内容。
    """
    # 统一扩展名格式
    target_ext = target_ext.lstrip(".").lower()

    # 工作目录（容器内）
    work_dir = Path("/tmp/convert") / str(uuid.uuid4())
    os.makedirs(work_dir, exist_ok=True)

    try:
        # 保存输入文件
        input_path = work_dir / filename
        with open(input_path, "wb") as f:
            f.write(input_bytes)

        # 调用 LibreOffice
        cmd = [
            "soffice",
            "--headless",
            "--nologo",
            "--nofirststartwizard",
            "--convert-to",
            target_ext,
            "--outdir",
            str(work_dir),
            str(input_path),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise ConvertError(f"LibreOffice failed: {result.stderr or result.stdout}")

        # 找到输出文件
        # LibreOffice 一般会生成同名但扩展名不同的文件
        output_files = list(work_dir.glob(f"*.{target_ext}"))
        if not output_files:
            raise ConvertError("No output file produced by LibreOffice")

        output_path = output_files[0]
        with open(output_path, "rb") as f:
            return f.read()

    finally:
        # 清理临时目录
        shutil.rmtree(work_dir, ignore_errors=True)