# DeepSeek OCR PDF 客户端测试工具

## 简介

`test_pdf_client.py` 是一个纯客户端的 PDF OCR 测试脚本，通过 OpenAI 兼容的 API 调用远程 DeepSeek OCR 服务。

**主要特点：**
- ✅ 纯客户端实现，无需本地加载模型
- ✅ 不依赖 torch/vllm，轻量级部署
- ✅ 调用远程 OCR 服务
- ✅ 支持多线程并发处理
- ✅ 输出格式与原脚本完全一致

## 安装依赖

```bash
pip install openai pillow pymupdf img2pdf tqdm numpy
```

## 快速开始

### 基本用法

```bash
python test_pdf_client.py -i input.pdf -o output_dir
```

### 完整参数

```bash
python test_pdf_client.py \
  --input /path/to/input.pdf \      # 输入 PDF 文件
  --output /path/to/output_dir \    # 输出目录
  --prompt "<image>\n<|grounding|>Convert the document to markdown." \  # OCR 提示词
  --workers 4 \                      # 并发线程数
  --dpi 144 \                        # PDF 转图片 DPI
  --no-skip-repeat                   # 不跳过重复内容
```

## 配置说明

在脚本开头可修改以下配置：

```python
# API 配置
API_BASE_URL = "http://ubuntu-mindora.local:8000/v1"  # OCR 服务地址
API_KEY = "EMPTY"                                      # API 密钥
MODEL_NAME = "deepseek-ai/DeepSeek-OCR"               # 模型名称

# 并发配置
MAX_WORKERS = 4  # 并发请求数，根据服务器性能调整

# PDF 处理配置
PDF_DPI = 144  # PDF 转图片的 DPI
```

## 常用提示词

```python
# 文档转 markdown（默认）
"<image>\n<|grounding|>Convert the document to markdown."

# 纯文本 OCR
"<image>\nFree OCR."

# 表格提取
"<image>\n<|grounding|>Extract the table."

# 图片描述
"<image>\nDescribe this image in detail."
```

## 输出文件

- `<filename>_det.mmd` - 包含标注信息的 markdown
- `<filename>.mmd` - 清理后的 markdown
- `<filename>_layouts.pdf` - 标注了边界框的 PDF
- `images/` - 提取的图片文件夹

## 示例

### 处理论文 PDF

```bash
python test_pdf_client.py \
  -i paper.pdf \
  -o paper_ocr \
  --workers 8 \
  --dpi 200
```

### 纯文本提取

```bash
python test_pdf_client.py \
  -i scan.pdf \
  -o scan_ocr \
  --prompt "<image>\nFree OCR."
```

## 与原脚本对比

| 特性 | run_dpsk_ocr_pdf.py | test_pdf_client.py |
|------|---------------------|-------------------|
| 模型加载 | 本地加载 vllm | 调用远程服务 |
| 内存占用 | 高 | 低 |
| GPU 要求 | 必须 | 无需 |
| torch 依赖 | 需要 | 不需要 |
| 启动速度 | 慢 | 快 |
| 并发方式 | vllm 批处理 | 多线程请求 |

## 性能优化

1. **调整并发数**：根据服务器 GPU 性能调整 `--workers`
   - GPU 内存充足：8-16
   - GPU 内存有限：2-4

2. **调整 DPI**：根据文档质量调整
   - 高质量：144-200
   - 低质量：72-144

## 故障排查

### 连接错误
```
Error processing page 0: Connection refused
```
**解决**：检查 OCR 服务是否启动，确认 `API_BASE_URL` 配置正确

### 超时错误
```
Error processing page 0: Request timeout
```
**解决**：增加 `REQUEST_TIMEOUT`，降低 DPI，减少并发数

### GPU 内存不足
```
Error processing page 0: CUDA out of memory
```
**解决**：降低并发数，降低 DPI
