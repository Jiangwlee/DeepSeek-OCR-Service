# DeepSeek OCR Service Stack

Modular dockerized stack for building a production-ready OCR pipeline powered by the DeepSeek OCR model. The stack decomposes the workflow into specialized services (MinIO storage, OCR orchestrator, document converter, and gateway) so each concern can be scaled and deployed independently.

## Service Lineup

| Service | Status | Responsibilities |
| --- | --- | --- |
| `minio` | ✅ ready | Object storage for raw uploads, converted assets, OCR outputs |
| `ocr` | 🛠 available | FastAPI facade that converts PDFs/images -> text using remote DeepSeek |
| `document-converter` | 🚧 planned | LibreOffice-powered conversion to PDF/images |
| `deepseek-ocr` | external | GPU inference node already running at `http://ubuntu-mindora.local:8000` |
| `paddle-ocr` | 🧩 optional | PaddleOCR REST 服务，CPU/GPU 取决于基础镜像（Compose profile: `paddle`） |
| `nginx` | 🚧 optional | Unified entry + TLS termination |

We are integrating each component one-by-one. The current step focuses on solid MinIO storage so files can move cleanly between services.

## Prerequisites

- Docker Engine + Docker Compose Plugin
- GPU host for the remote DeepSeek OCR instance (already available at `ubuntu-mindora.local`)

## Configuration

Every service is wired through environment variables for portability. Copy the sample file before spinning anything up:

```bash
cp .env.example .env
docker network create ocr_net
```

Key knobs (see `.env.example` for defaults):

- `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` – bootstrap credentials
- `MINIO_API_PORT` / `MINIO_CONSOLE_PORT` – host ports for API + console
- `MINIO_BUCKETS` – space-separated list of buckets created on startup
- `MINIO_EXTERNAL_*` – how other services should reference MinIO when generating presigned URLs

## Bringing Up MinIO (Step 1)

```bash
# start MinIO + one-shot bucket initializer
docker compose up -d minio minio-init
```

What happens:

1. `minio` starts with persisted data/config volumes under `.data/minio/` (ignored by git).
2. `minio-init` waits until the API is reachable, then creates all buckets defined in `MINIO_BUCKETS` and optionally marks `MINIO_PUBLIC_BUCKET` as anonymous-readable.

You can now access:

- S3-compatible endpoint: `http://localhost:${MINIO_API_PORT}`
- Admin console: `http://localhost:${MINIO_CONSOLE_PORT}`

## Bucket Layout Recommendation

| Bucket | Purpose |
| --- | --- |
| `raw-docs` | User uploads exactly as received |
| `converted-pdf` | LibreOffice outputs / normalized PDFs |
| `page-images` | Split page images (PNG/JPEG) handed to the OCR runtime |
| `ocr-results` | Final structured OCR payloads (Markdown/JSON) |

Each downstream service should read/write via presigned URLs instead of shipping large binaries over internal APIs.

## OCR Service (Step 2)

The OCR orchestrator is implemented under `services/ocr/app` and已经整合到 `docker-compose.yml`。
使用 Compose 构建 + 启动：

```bash
docker compose up -d ocr
```

启用内置 PaddleOCR（CPU 默认，可换 GPU 镜像）：

```bash
docker compose --profile paddle up -d paddle-ocr
```
然后将 `.env` 中的 `OCR_PADDLE_ENDPOINT` 设为 `http://paddle-ocr:9000`。

选择 Paddle 基础镜像（编译时变量 `PADDLE_BASE_IMAGE`）：
- CPU: `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0`
- GPU (CUDA11.8): `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6`
- GPU (CUDA12.6): `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda12.6-cudnn9.5-trt10.5`

示例：
```bash
PADDLE_BASE_IMAGE=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6 \
docker compose --profile paddle up -d --build paddle-ocr
```

### Local dev server

```bash
cd services/ocr
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# ensure `.env` is loaded or export the OCR_* variables manually
export $(grep '^OCR_' ../../.env | xargs)
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Key environment variables (see `.env.example`):

- `OCR_DEEPSEEK_API_BASE` / `OCR_DEEPSEEK_API_KEY` / `OCR_DEEPSEEK_MODEL`
- `OCR_PDF_DPI`, `OCR_MAX_WORKERS`, `OCR_DEEPSEEK_MAX_TOKENS`, `OCR_HTTP_PORT`
- `OCR_PADDLE_ENDPOINT`（可选，指向 PaddleOCR 服务）、`OCR_PADDLE_TIMEOUT`
- `OCR_MINIO_*` if you want automatic upload of OCR results back into MinIO buckets

### Web UI + API surface

- Visit `http://localhost:8001/ui` for the built-in Gradio playground (tabs cover file upload, remote URL, and MinIO bucket tests, plus optional result persistence).
- API 端点：
  - `POST /v1/ocr/document/upload` – multipart upload (`file`) + optional form fields (`output_format`, `prompt`, `store_*`).
  - `POST /v1/ocr/document/from-storage` – JSON body `{ "source": {"bucket": "raw-docs", "object_name": "..." } }`.
  - `POST /v1/ocr/document/from-url` – JSON body `{ "source": { "url": "https://..." } }`.
  - `GET /healthz` – basic readiness info (DeepSeek endpoint + MinIO flag).

Each request returns a structured payload with page-level OCR results plus the combined text. When `store_result=true`, the service writes the final text into MinIO (defaults to the `ocr-results` bucket).

### Quick API examples (curl)

Image (PNG/JPEG) → plain text, DeepSeek 后端：

```bash
curl -X POST http://localhost:8001/v1/ocr/document/upload \
  -F "file=@./samples/demo.png" \
  -F "output_format=plain_text" \
  -F "provider=deepseek"
```

DOC/DOCX → 自动转 PDF → Markdown，Paddle 后端（需开启 `document-converter` + `paddle-ocr`，依赖 MinIO）：

```bash
curl -X POST http://localhost:8001/v1/ocr/document/upload \
  -F "file=@./samples/demo.docx" \
  -F "output_format=markdown" \
  -F "provider=paddle" \
  -F "store_result=true"
```

返回字段包含 `pages`（分页面文本）、`combined_text`（合并后文本），以及存储到 MinIO 的 bucket/object 信息（若 `store_result=true`）。

## Next Steps

1. **Document converter**: wrap LibreOffice container and wire it into the orchestrator.
2. **Nginx** (optional): expose a single `/api` surface once individual services stabilize.
3. Create `docker-compose.with-deepseek-ocr.yml` once all services are production-ready so the OCR container can join the stack explicitly.

Refer to `idea.md` for the high-level architecture notes that guide the remaining work.
