from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from .ocr_service.exceptions import OCRException
from .ocr_service.schemas import DocumentOptions, DocumentResult
from .ocr_service.service import OCROrchestrator


def build_gradio_interface(orchestrator: OCROrchestrator) -> gr.Blocks:
    async def handle_upload(file, output_format, prompt, provider):
        if file is None:
            return "请先上传文件。", "", ""
        path = Path(file.name)
        data = await asyncio.to_thread(path.read_bytes)
        options = _build_options(output_format, prompt, provider)
        result = await orchestrator.process_bytes(
            filename=path.name,
            content_type=None,
            data=data,
            options=options,
        )
        meta_md, text = _format_result(result)
        preview = f"文件：{path.name}（{path.stat().st_size} bytes）"
        return preview, meta_md, text

    async def wrapper(fn, *args):
        try:
            return await fn(*args)
        except OCRException as exc:
            return f"OCR 失败：{exc}", "", ""
        except Exception as exc:  # pragma: no cover
            return f"未知错误：{exc}", "", ""

    with gr.Blocks(title="DeepSeek OCR Playground", fill_height=True) as demo:
        gr.Markdown(
            """# DeepSeek OCR Playground
左侧：上传与配置（含文件预览）；右侧：结果预览。"""
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=420):
                gr.Markdown("## 上传与配置")
                upload_file = gr.File(
                    label="文档（doc/docx/pdf/图片）",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".doc", ".docx"],
                    interactive=True,
                    visible=True,
                )
                file_preview = gr.Markdown("未选择文件")

                gr.Markdown("### 识别配置")
                output_format = gr.Radio(["markdown", "plain_text"], label="输出格式", value="markdown")
                prompt = gr.Textbox(label="自定义 Prompt", placeholder="留空使用默认提示", lines=2)
                provider = gr.Radio(["deepseek", "paddle"], label="推理后端", value="deepseek")

                submit_button = gr.Button("开始 OCR", variant="primary")

            with gr.Column(scale=1, min_width=420):
                gr.Markdown("## 结果预览")
                meta_output = gr.Markdown(label="结果概览")
                text_output = gr.Textbox(label="识别结果", lines=18)

        upload_file.change(
            lambda f: f"文件：{Path(f.name).name}（{Path(f.name).stat().st_size} bytes）" if f else "未选择文件",
            inputs=upload_file,
            outputs=file_preview,
        )

        async def on_submit(file, output_format, prompt, provider):
            return await wrapper(handle_upload, file, output_format, prompt, provider)

        submit_button.click(
            on_submit,
            inputs=[
                upload_file,
                output_format,
                prompt,
                provider,
            ],
            outputs=[file_preview, meta_output, text_output],
            concurrency_limit=2,
        )

    return demo


def _build_options(
    output_format: str,
    prompt: Optional[str],
    provider: str,
) -> DocumentOptions:
    return DocumentOptions(
        output_format=output_format or "markdown",
        prompt=prompt or None,
        provider=provider,
    )


def _format_result(result: DocumentResult) -> Tuple[str, str]:
    meta_lines = [
        f"**文档 ID**: {result.document_id}",
        f"**页数**: {result.total_pages}",
        f"**输出格式**: {result.output_format}",
    ]
    if result.stored_bucket:
        meta_lines.append(f"**已写入对象存储**: {result.stored_bucket}/{result.stored_object_name}")
    meta_md = "\n".join(meta_lines)
    return meta_md, result.combined_text
