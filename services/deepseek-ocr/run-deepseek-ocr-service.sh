export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vllm serve deepseek-ai/DeepSeek-OCR \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.6 \
  --max-num-seqs 16 \
  --max-model-len 4096 \
  --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0