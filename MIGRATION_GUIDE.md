# 环境变量迁移指南

## 概述

为了支持多种vLLM模型(不仅限于DeepSeek-OCR),我们对环境变量进行了重命名,使其更加通用。

## 需要更新的环境变量

如果你有现有的 `.env` 文件,请按照以下映射更新变量名:

### vLLM Service 配置 (原 DeepSeek OCR Service)

| 旧变量名 | 新变量名 | 说明 |
|---------|---------|------|
| `DEEPSEEK_MODEL_NAME` | `VLLM_MODEL_NAME` | 模型名称 |
| `DEEPSEEK_MODEL_TYPE` | `VLLM_MODEL_TYPE` | 模型类型 |
| `DEEPSEEK_VLLM_HOST` | `VLLM_HOST` | vLLM服务主机 |
| `DEEPSEEK_VLLM_PORT` | `VLLM_PORT` | vLLM服务端口 |
| `DEEPSEEK_GPU_MEMORY_UTILIZATION` | `VLLM_GPU_MEMORY_UTILIZATION` | GPU内存利用率 |
| `DEEPSEEK_MAX_NUM_SEQS` | `VLLM_MAX_NUM_SEQS` | 最大并发序列数 |
| `DEEPSEEK_MAX_MODEL_LEN` | `VLLM_MAX_MODEL_LEN` | 最大上下文长度 |
| `DEEPSEEK_MM_PROCESSOR_CACHE_GB` | `VLLM_MM_PROCESSOR_CACHE_GB` | 多模态处理器缓存大小 |
| `DEEPSEEK_TENSOR_PARALLEL_SIZE` | `VLLM_TENSOR_PARALLEL_SIZE` | Tensor并行大小 |
| `DEEPSEEK_LIMIT_VIDEO_INPUTS` | `VLLM_LIMIT_VIDEO_INPUTS` | 视频输入限制 |
| `DEEPSEEK_ENABLE_ASYNC_SCHEDULING` | `VLLM_ENABLE_ASYNC_SCHEDULING` | 启用异步调度 |
| `DEEPSEEK_EXTRA_VLLM_ARGS` | `VLLM_EXTRA_VLLM_ARGS` | 额外的vLLM参数 |
| `DEEPSEEK_CUDA_VISIBLE_DEVICES` | `VLLM_CUDA_VISIBLE_DEVICES` | CUDA设备可见性 |

### OCR Service 配置 (连接到vLLM的客户端配置)

| 旧变量名 | 新变量名 | 说明 |
|---------|---------|------|
| `OCR_DEEPSEEK_API_BASE` | `OCR_VLLM_API_BASE` | vLLM API基础URL |
| `OCR_DEEPSEEK_API_KEY` | `OCR_VLLM_API_KEY` | API密钥 |
| `OCR_DEEPSEEK_MODEL` | `OCR_VLLM_MODEL` | 使用的模型名称 |
| `OCR_DEEPSEEK_REQUEST_TIMEOUT` | `OCR_VLLM_REQUEST_TIMEOUT` | 请求超时时间 |
| `OCR_DEEPSEEK_MAX_TOKENS` | `OCR_VLLM_MAX_TOKENS` | 最大token数 |
| `OCR_DEEPSEEK_TEMPERATURE` | `OCR_VLLM_TEMPERATURE` | 温度参数 |
| `OCR_DEEPSEEK_SKIP_SPECIAL_TOKENS` | `OCR_VLLM_SKIP_SPECIAL_TOKENS` | 跳过特殊token |

## 迁移步骤

1. **备份现有配置**
   ```bash
   cp .env .env.backup
   ```

2. **更新 `.env` 文件**
   - 方法1: 手动编辑 `.env`,按照上表更新变量名
   - 方法2: 使用sed批量替换(Linux/Mac):
     ```bash
     # 更新vLLM服务配置
     sed -i 's/DEEPSEEK_MODEL_NAME/VLLM_MODEL_NAME/g' .env
     sed -i 's/DEEPSEEK_MODEL_TYPE/VLLM_MODEL_TYPE/g' .env
     sed -i 's/DEEPSEEK_VLLM_HOST/VLLM_HOST/g' .env
     sed -i 's/DEEPSEEK_VLLM_PORT/VLLM_PORT/g' .env
     sed -i 's/DEEPSEEK_GPU_MEMORY_UTILIZATION/VLLM_GPU_MEMORY_UTILIZATION/g' .env
     sed -i 's/DEEPSEEK_MAX_NUM_SEQS/VLLM_MAX_NUM_SEQS/g' .env
     sed -i 's/DEEPSEEK_MAX_MODEL_LEN/VLLM_MAX_MODEL_LEN/g' .env
     sed -i 's/DEEPSEEK_MM_PROCESSOR_CACHE_GB/VLLM_MM_PROCESSOR_CACHE_GB/g' .env
     sed -i 's/DEEPSEEK_TENSOR_PARALLEL_SIZE/VLLM_TENSOR_PARALLEL_SIZE/g' .env
     sed -i 's/DEEPSEEK_LIMIT_VIDEO_INPUTS/VLLM_LIMIT_VIDEO_INPUTS/g' .env
     sed -i 's/DEEPSEEK_ENABLE_ASYNC_SCHEDULING/VLLM_ENABLE_ASYNC_SCHEDULING/g' .env
     sed -i 's/DEEPSEEK_EXTRA_VLLM_ARGS/VLLM_EXTRA_VLLM_ARGS/g' .env
     sed -i 's/DEEPSEEK_CUDA_VISIBLE_DEVICES/VLLM_CUDA_VISIBLE_DEVICES/g' .env

     # 更新OCR服务配置
     sed -i 's/OCR_DEEPSEEK_API_BASE/OCR_VLLM_API_BASE/g' .env
     sed -i 's/OCR_DEEPSEEK_API_KEY/OCR_VLLM_API_KEY/g' .env
     sed -i 's/OCR_DEEPSEEK_MODEL/OCR_VLLM_MODEL/g' .env
     sed -i 's/OCR_DEEPSEEK_REQUEST_TIMEOUT/OCR_VLLM_REQUEST_TIMEOUT/g' .env
     sed -i 's/OCR_DEEPSEEK_MAX_TOKENS/OCR_VLLM_MAX_TOKENS/g' .env
     sed -i 's/OCR_DEEPSEEK_TEMPERATURE/OCR_VLLM_TEMPERATURE/g' .env
     sed -i 's/OCR_DEEPSEEK_SKIP_SPECIAL_TOKENS/OCR_VLLM_SKIP_SPECIAL_TOKENS/g' .env
     ```

3. **参考新的配置模板**
   ```bash
   # 查看新的配置示例
   cat .env.example
   ```

4. **重启服务**
   ```bash
   docker-compose down
   docker-compose up -d --build
   ```

## 新增功能

更新后,你可以轻松切换到不同的视觉语言模型:

### 使用 Qwen3-VL-8B-Instruct
```bash
# 在 .env 中设置
VLLM_MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
VLLM_MODEL_TYPE=auto  # 自动检测
OCR_VLLM_MODEL=Qwen/Qwen3-VL-8B-Instruct
```

### 使用 Qwen3-VL-8B-Instruct-FP8 (推荐,内存占用更低)
```bash
VLLM_MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct-FP8
VLLM_MODEL_TYPE=qwen3-vl
OCR_VLLM_MODEL=Qwen/Qwen3-VL-8B-Instruct-FP8
```

## 向后兼容性

- 所有配置项都有合理的默认值
- 如果不设置环境变量,系统默认使用 `deepseek-ai/DeepSeek-OCR` 模型
- 行为与之前版本完全一致

## 需要帮助?

如果在迁移过程中遇到问题,请查看:
- `.env.example` - 完整的配置示例
- `README.md` - 项目文档
- GitHub Issues - 提交问题
