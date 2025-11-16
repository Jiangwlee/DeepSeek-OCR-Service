# OCR Service 设计文档

## 1. 概述

设计一个独立的OCR服务模块，调用本地部署的DeepSeek OCR模型，将多种文件格式（PDF、DOCX等）转换为Markdown或纯文本。

**位置**: `backend/services/ocr/`
**作用**: 公用服务模块，独立于知识库系统
**依赖**: OpenAI兼容API（本地DeepSeek OCR服务）

---

## 2. 核心设计原则

- **简化优先**：只提供API调用，不涉及模型部署
- **独立模块**：完全解耦，独立配置
- **易用性**：统一接口，自动类型判断
- **模块化**：便于扩展新的文件类型

---

## 3. API 接口设计

### 3.1 核心方法

```python
class OCRService:
    """OCR服务统一接口"""

    async def extract_text(
        self,
        file_path: str,
        output_format: str = "markdown",  # markdown | plain_text
        prompt: str = None,
        **kwargs
    ) -> str:
        """
        从文件提取文本

        Args:
            file_path: 输入文件路径（支持PDF、DOCX等）
            output_format: 输出格式
            prompt: 自定义OCR提示词
            **kwargs: 其他参数

        Returns:
            提取的文本内容

        Raises:
            FileNotFoundError: 文件不存在
            UnsupportedFileTypeError: 不支持的文件格式
            OCRProcessException: 处理失败
        """

    async def describe_image(
        self,
        image_path: str,
        description_type: str = "full"  # full | detailed | concise
    ) -> str:
        """
        描述图片内容

        Args:
            image_path: 图片路径
            description_type: 描述类型

        Returns:
            图片描述文本
        """
```

---

## 4. 架构设计

### 4.1 模块结构

```
backend/services/ocr/
├── __init__.py                 # 导出公用接口
├── config.py                   # 独立配置管理
├── client.py                   # OpenAI兼容客户端
├── processor.py                # 文件处理器（PDF、Image等）
├── service.py                  # Facade 服务
├── schemas.py                  # Pydantic数据模型
├── exceptions.py               # 自定义异常
└── utils.py                    # 工具函数
```

### 4.2 关键类设计

#### client.py - OCR客户端
```python
class OCRClient:
    """与DeepSeek OCR API交互"""

    async def process_image(
        self,
        image: Union[str, Image.Image],
        prompt: str
    ) -> str:
        """处理单张图片"""

    async def process_images_concurrent(
        self,
        images: List[Image.Image],
        prompt: str,
        max_workers: int
    ) -> List[str]:
        """并发处理多张图片"""
```

#### processor.py - 文件处理器
```python
class BaseFileProcessor(ABC):
    """文件处理器基类"""

    @abstractmethod
    async def to_images(self, file_path: str) -> List[Image.Image]:
        """将文件转换为图片列表"""

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """是否支持该文件类型"""


class PDFProcessor(BaseFileProcessor):
    """PDF处理器 - 使用PyMuPDF"""

class ImageProcessor(BaseFileProcessor):
    """图片处理器 - 直接加载"""

class DocxProcessor(BaseFileProcessor):
    """DOCX处理器 - 使用python-docx直接提取文本"""

class DocProcessor(BaseFileProcessor):
    """DOC处理器 - 使用LibreOffice转PDF，再转图片"""

class PptxProcessor(BaseFileProcessor):
    """PPTX处理器 - 使用python-pptx提取幻灯片"""

class ProcessorFactory:
    """处理器工厂"""
    @staticmethod
    def get_processor(file_path: str) -> BaseFileProcessor:
        """根据文件类型返回处理器"""
```

#### service.py - Facade服务
```python
class OCRService:
    """OCR服务对外统一接口"""

    async def extract_text(self, file_path: str, ...) -> str:
        """流程：验证 → 处理器 → 转图片 → 并发OCR → 后处理 → 缓存"""

    async def describe_image(self, image_path: str, ...) -> str:
        """图片描述"""
```

---

## 5. 配置设计

### 5.1 配置内容 (backend/config/ocr_config.py)

```python
class OCRConfig(BaseSettings):
    """OCR服务独立配置"""

    # API配置
    api_base_url: str = "http://ubuntu-mindora.local:8000/v1"
    api_key: str = "EMPTY"
    model_name: str = "deepseek-ai/DeepSeek-OCR"
    request_timeout: int = 3600
    max_tokens: int = 4000
    temperature: float = 0.0

    # 文件处理配置
    pdf_dpi: int = 144              # PDF转图片的DPI
    max_workers: int = 4            # 并发worker数
    cache_enabled: bool = True      # 是否启用缓存

    # 缓存路径
    cache_dir: str = Field(
        default_factory=lambda: settings.base_dir / "ocr_cache"
    )

    class Config:
        env_prefix = "OCR_"
```

---

## 6. 处理流程

### 6.1 文本提取流程

```
输入文件
    ↓
验证 (文件存在、类型支持)
    ↓
选择处理器
    ├─ .pdf → PDFProcessor → 转图片 → OCR
    ├─ .docx → DocxProcessor → 直接提取文本
    ├─ .doc → DocProcessor → LibreOffice转PDF → 转图片 → OCR
    ├─ .pptx → PptxProcessor → 提取幻灯片 → Markdown
    └─ .png/.jpg → ImageProcessor → 直接OCR
    ↓
[文件特定处理]
    ├─ PDF/Image: 转换为图片列表
    ├─ DOCX: 提取结构化内容
    └─ DOC: 间接转图片
    ↓
[并发OCR处理] (仅限图片类)
    ├─ 调用DeepSeek API
    ├─ base64编码
    └─ 维持页码顺序
    ↓
后处理
    ├─ 清理特殊token
    ├─ 移除标记标签
    └─ 合并多页内容
    ↓
缓存 (可选)
    ↓
返回Markdown/纯文本
```

### 6.2 默认提示词

```python
PROMPTS = {
    "markdown": "<image>\n<|grounding|>Convert the document to markdown.",
    "plain_text": "<image>\nFree OCR.",
    "image_full": "<image>\nDescribe this image in detail.",
    "image_concise": "<image>\n<|grounding|>Summarize this image.",
}
```

---

## 7. 支持的文件类型

| 类型 | 扩展名 | 处理方式 |
|------|--------|--------|
| PDF | .pdf | PyMuPDF转图片 → OCR |
| 图片 | .png, .jpg, .jpeg | 直接OCR |
| Word(新版) | .docx | python-docx提取文本 → Markdown |
| Word(旧版) | .doc | LibreOffice转PDF → 转图片 → OCR |
| PPT | .pptx | python-pptx提取幻灯片 → Markdown |

---

## 8. 集成到知识库系统

### 8.1 在IngestionService中使用

```python
# backend/agents/kb_agent/services/ingestion_service.py
from backend.services.ocr import OCRService

class IngestionService:
    def __init__(self):
        self.ocr_service = OCRService()

    async def _chunk_file(self, file_path: str):
        file_ext = Path(file_path).suffix.lower()

        # 支持的简单格式（SimpleDirectoryReader直接支持）
        simple_formats = {'.txt', '.md', '.pdf', '.rtf'}

        # 需要转换的格式
        needs_conversion = {'.doc', '.docx', '.pptx', '.png', '.jpg', '.jpeg'}

        if file_ext in needs_conversion:
            # 转换为Markdown
            markdown = await self.ocr_service.extract_text(str(file_path))
            # 写入临时文件供SimpleDirectoryReader处理
            temp_file = self._write_temp_markdown(markdown)
            file_path = temp_file
        elif file_ext not in simple_formats:
            raise UnsupportedFileTypeError(f"Unsupported file type: {file_ext}")

        # 继续原有流程...
```

### 8.2 .doc 文件的特殊处理

由于 .doc 是旧格式，处理流程如下：

```
.doc 文件上传
    ↓
OCRService.extract_text()
    ↓
DocProcessor.to_images()
    ├─ 尝试python-docx读取 → 成功则结束
    └─ 失败 → LibreOffice转PDF → PyMuPDF转图片
    ↓
OCRClient并发调用API → Markdown
    ↓
返回文本供IngestionService处理
```

---

## 9. 异常体系

```python
class OCRException(Exception):
    """基异常"""

class FileNotFoundError(OCRException):
    """文件不存在"""

class UnsupportedFileTypeError(OCRException):
    """不支持的文件格式"""

class OCRProcessException(OCRException):
    """处理失败"""

class APIException(OCRException):
    """API调用失败"""
```

---

## 11. 缓存机制

```
~/.smart_agents/ocr_cache/
├── images/
│   └── <file_hash>/
│       ├── page_0.png
│       └── ...
└── results/
    └── <file_hash>.json
```

---

## 12. 性能指标

- 单页PDF转Markdown：2-5秒
- 10页PDF（4个workers）：15-30秒
- 缓存命中：< 100ms
- API超时：3600秒（可配置）

---

## 13. 依赖

```python
# 新增依赖
- pymupdf (fitz)         # PDF处理
- python-docx           # DOCX文件处理
- python-pptx           # PPTX文件处理
- Pillow                # 图片处理（可能已有）

# 可选依赖
- libreoffice           # DOC转PDF（系统级）
- uno                   # LibreOffice Python接口（可选）

# 已有依赖
- openai                # API客户端
- pydantic              # 数据验证
```

### 处理方案说明

**DOC 文件处理**：
1. 首先尝试使用 `python-docx` 直接读取（某些新版.doc也能读）
2. 如果失败，使用 LibreOffice 转换为 PDF：
   ```bash
   libreoffice --headless --convert-to pdf input.doc
   ```
3. 将PDF转为图片后进行OCR

**DOCX 文件处理**：
- 直接使用 `python-docx` 提取文本和结构
- 保留格式信息（标题、段落、列表等）
- 转换为Markdown格式

**PPTX 文件处理**：
- 使用 `python-pptx` 提取各幻灯片
- 每张幻灯片作为一页处理
- 支持文本、表格、图片等元素

---

**版本**: 1.0
**状态**: 待实现
