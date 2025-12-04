from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from .ocr_service.config import get_settings
from .ocr_service.exceptions import OCRException
from .ocr_service.schemas import DocumentOptions, DocumentResult
from .ocr_service.service import OCROrchestrator


def build_gradio_interface(orchestrator: OCROrchestrator) -> gr.Blocks:
    settings = get_settings()

    # 获取可用模型和 prompt 预设
    available_models = settings.get_available_models()
    prompt_presets = settings.get_prompt_presets()

    # 构建选项
    model_choices = [(m["name"], m["value"]) for m in available_models]
    prompt_choices = [("自定义 Prompt", "custom")] + [(p["name"], p["value"]) for p in prompt_presets]

    async def handle_upload(file, output_format, model_provider, prompt_preset, custom_prompt):
        if file is None:
            return "请先上传文件。", "", ""

        path = Path(file.name)
        data = await asyncio.to_thread(path.read_bytes)

        # 确定使用的 prompt
        final_prompt = None
        if prompt_preset == "custom":
            final_prompt = custom_prompt if custom_prompt.strip() else None
        else:
            final_prompt = prompt_preset if prompt_preset else None

        options = DocumentOptions(
            output_format=output_format or "markdown",
            prompt=final_prompt,
            provider=model_provider,
        )

        result = await orchestrator.process_bytes(
            filename=path.name,
            content_type=None,
            data=data,
            options=options,
        )

        meta_md, text = _format_result(result, model_provider)
        preview = f"文件：{path.name}（{path.stat().st_size} bytes）"
        return preview, meta_md, text

    async def wrapper(fn, *args):
        try:
            return await fn(*args)
        except OCRException as exc:
            return f"OCR 失败：{exc}", "", ""
        except Exception as exc:  # pragma: no cover
            return f"未知错误：{exc}", "", ""

    with gr.Blocks(title="Multi-Model OCR Playground", fill_height=True) as demo:
        gr.Markdown(
            """# 🚀 Multi-Model OCR Playground
**支持多模型**: DeepSeek-OCR、Qwen3-VL、PaddleOCR
左侧：上传与配置；右侧：结果预览"""
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=420):
                gr.Markdown("## 📁 文件上传")
                upload_file = gr.File(
                    label="文档（doc/docx/pdf/图片）",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".doc", ".docx"],
                    interactive=True,
                    visible=True,
                )
                file_preview = gr.Markdown("未选择文件")

                gr.Markdown("## ⚙️ 识别配置")

                # 模型选择下拉框
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    label="🤖 选择模型",
                    value=model_choices[0][1] if model_choices else "deepseek",
                    interactive=True,
                )

                # 显示当前选择的模型信息
                model_info = gr.Markdown(
                    f"**当前模型**: {available_models[0]['name']}\n"
                    f"**端点**: `{available_models[0]['endpoint']}`"
                )

                # 输出格式
                output_format = gr.Radio(
                    ["markdown", "plain_text"],
                    label="📄 输出格式",
                    value="markdown"
                )

                # Prompt 预设下拉框
                prompt_dropdown = gr.Dropdown(
                    choices=prompt_choices,
                    label="💬 Prompt 预设",
                    value="custom",
                    interactive=True,
                )

                # 自定义 Prompt 输入框
                custom_prompt_box = gr.Textbox(
                    label="✏️ 自定义 Prompt",
                    placeholder="选择上方预设或输入自定义 Prompt",
                    lines=3,
                    visible=True,
                )

                submit_button = gr.Button("🔍 开始 OCR", variant="primary", size="lg")

            with gr.Column(scale=1, min_width=420):
                gr.Markdown("## 📊 结果预览")
                meta_output = gr.Markdown(label="结果概览")
                text_output = gr.Textbox(label="识别结果", lines=20, max_lines=30)

        # 文件选择时更新预览
        upload_file.change(
            lambda f: f"📄 文件：**{Path(f.name).name}** ({Path(f.name).stat().st_size:,} bytes)" if f else "未选择文件",
            inputs=upload_file,
            outputs=file_preview,
        )

        # 模型选择时更新模型信息
        def update_model_info(selected_model):
            for model in available_models:
                if model["value"] == selected_model:
                    return (
                        f"**当前模型**: {model['name']}\n"
                        f"**端点**: `{model['endpoint']}`\n"
                        f"**模型ID**: `{model['model']}`"
                    )
            return "模型信息未找到"

        model_dropdown.change(
            update_model_info,
            inputs=model_dropdown,
            outputs=model_info,
        )

        # Prompt 预设选择时更新自定义输入框
        def update_custom_prompt(preset_value):
            if preset_value == "custom":
                return gr.update(visible=True, placeholder="输入自定义 Prompt", value="")
            else:
                return gr.update(visible=True, placeholder=f"当前使用预设: {preset_value[:50]}...", value="")

        prompt_dropdown.change(
            update_custom_prompt,
            inputs=prompt_dropdown,
            outputs=custom_prompt_box,
        )

        async def on_submit(file, output_format, model_provider, prompt_preset, custom_prompt):
            return await wrapper(handle_upload, file, output_format, model_provider, prompt_preset, custom_prompt)

        submit_button.click(
            on_submit,
            inputs=[
                upload_file,
                output_format,
                model_dropdown,
                prompt_dropdown,
                custom_prompt_box,
            ],
            outputs=[file_preview, meta_output, text_output],
            concurrency_limit=2,
        )

    return demo


def _format_result(result: DocumentResult, model_provider: str) -> Tuple[str, str]:
    """Format the OCR result for display."""
    # 获取模型名称
    settings = get_settings()
    available_models = settings.get_available_models()
    model_name = model_provider
    for model in available_models:
        if model["value"] == model_provider:
            model_name = model["name"]
            break

    meta_lines = [
        f"✅ **OCR 完成**",
        f"🤖 **使用模型**: {model_name}",
        f"📝 **文档 ID**: `{result.document_id}`",
        f"📄 **页数**: {result.total_pages}",
        f"📋 **输出格式**: {result.output_format}",
    ]
    if result.stored_bucket:
        meta_lines.append(f"💾 **已存储**: `{result.stored_bucket}/{result.stored_object_name}`")

    meta_md = "\n\n".join(meta_lines)
    return meta_md, result.combined_text
