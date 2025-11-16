from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import httpx

from .ocr_service.exceptions import OCRException, StorageUnavailableError
from .ocr_service.schemas import DocumentOptions, DocumentResult
from .ocr_service.service import OCROrchestrator


def build_gradio_interface(orchestrator: OCROrchestrator) -> gr.Blocks:
    async def handle_upload(file, output_format, prompt, store_result, store_bucket, store_object, provider):
        if file is None:
            return "请先上传文件。", ""
        path = Path(file.name)
        data = await asyncio.to_thread(path.read_bytes)
        options = _build_options(output_format, prompt, store_result, store_bucket, store_object, provider)
        result = await orchestrator.process_bytes(
            filename=path.name,
            content_type=None,
            data=data,
            options=options,
        )
        return _format_result(result)

    async def handle_url(url, output_format, prompt, store_result, store_bucket, store_object, provider):
        if not url:
            return "请输入文件 URL。", ""
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.content
            content_type = response.headers.get("content-type")
        filename = url.split("/")[-1] or "remote-file"
        options = _build_options(output_format, prompt, store_result, store_bucket, store_object, provider)
        result = await orchestrator.process_bytes(
            filename=filename,
            content_type=content_type,
            data=data,
            options=options,
        )
        return _format_result(result)

    async def handle_storage(bucket, object_name, output_format, prompt, store_result, store_bucket, store_object, provider):
        storage = orchestrator.storage
        if not storage:
            raise StorageUnavailableError("MinIO 尚未启用，无法读取对象存储。")
        if not bucket or not object_name:
            return "请填写 bucket 和 object name。", ""
        data = await asyncio.to_thread(storage.fetch, bucket, object_name)
        filename = object_name.split("/")[-1] or object_name
        options = _build_options(output_format, prompt, store_result, store_bucket, store_object, provider)
        result = await orchestrator.process_bytes(
            filename=filename,
            content_type=None,
            data=data,
            options=options,
        )
        return _format_result(result)

    async def wrapper(fn, *args):
        try:
            return await fn(*args)
        except StorageUnavailableError as exc:
            return f"存储错误：{exc}", ""
        except OCRException as exc:
            return f"OCR 失败：{exc}", ""
        except httpx.HTTPError as exc:
            return f"下载失败：{exc}", ""
        except Exception as exc:  # pragma: no cover
            return f"未知错误：{exc}", ""

    with gr.Blocks(title="DeepSeek OCR Playground", fill_height=True) as demo:
        gr.Markdown("""# DeepSeek OCR Playground
使用底部各类入口测试 OCR API，结果实时展示。
""")

        async def upload_entry(file, *args):
            return await wrapper(handle_upload, file, *args)

        async def url_entry(url, *args):
            return await wrapper(handle_url, url, *args)

        async def storage_entry(bucket, object_name, *args):
            return await wrapper(handle_storage, bucket, object_name, *args)

        with gr.Tab("上传文件"):
            upload_file = gr.File(label="PDF / 图片", file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp"], interactive=True)
            upload_outputs = _shared_option_components()
            upload_button = gr.Button("开始 OCR", variant="primary")
            upload_button.click(
                upload_entry,
                inputs=[upload_file, *upload_outputs.inputs],
                outputs=upload_outputs.outputs,
                concurrency_limit=4,
            )

        with gr.Tab("网络链接"):
            url_input = gr.Textbox(label="文件 URL", placeholder="https://example.com/sample.pdf")
            url_outputs = _shared_option_components()
            url_button = gr.Button("抓取并 OCR")
            url_button.click(
                url_entry,
                inputs=[url_input, *url_outputs.inputs],
                outputs=url_outputs.outputs,
                concurrency_limit=4,
            )

        with gr.Tab("对象存储"):
            bucket_input = gr.Textbox(label="Bucket", value="raw-docs")
            object_input = gr.Textbox(label="Object Name", placeholder="path/to/file.pdf")
            storage_outputs = _shared_option_components()
            storage_button = gr.Button("读取并 OCR")
            storage_button.click(
                storage_entry,
                inputs=[bucket_input, object_input, *storage_outputs.inputs],
                outputs=storage_outputs.outputs,
                concurrency_limit=2,
            )

    return demo


def _build_options(
    output_format: str,
    prompt: Optional[str],
    store_result: bool,
    store_bucket: Optional[str],
    store_object: Optional[str],
    provider: str,
) -> DocumentOptions:
    return DocumentOptions(
        output_format=output_format or "markdown",
        prompt=prompt or None,
        store_result=store_result,
        store_bucket=store_bucket or None,
        store_object_name=store_object or None,
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


class _SharedComponents:
    def __init__(self):
        self.output_format = gr.Radio(["markdown", "plain_text"], label="输出格式", value="markdown")
        self.prompt = gr.Textbox(label="自定义 Prompt", placeholder="留空使用默认提示", lines=2)
        self.store_result = gr.Checkbox(label="写入 MinIO", value=False)
        self.store_bucket = gr.Textbox(label="目标 Bucket", placeholder="默认 ocr-results")
        self.store_object = gr.Textbox(label="对象名", placeholder="默认自动生成")
        self.provider = gr.Radio(["deepseek", "paddle"], label="推理后端", value="deepseek")
        self.meta_output = gr.Markdown(label="结果概览")
        self.text_output = gr.Textbox(label="识别结果", lines=15)

    @property
    def inputs(self):
        return [
            self.output_format,
            self.prompt,
            self.store_result,
            self.store_bucket,
            self.store_object,
            self.provider,
        ]

    @property
    def outputs(self):
        return [self.meta_output, self.text_output]


def _shared_option_components() -> _SharedComponents:
    return _SharedComponents()
