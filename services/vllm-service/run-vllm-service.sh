#!/bin/bash
set -euo pipefail

# =============================================================================
# vLLM OCR Service Launcher
# Supports multiple vision-language models: DeepSeek-OCR, Qwen3-VL, etc.
# =============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -----------------------------------------------------------------------------
# Environment Variables with Defaults
# -----------------------------------------------------------------------------
MODEL_NAME=${MODEL_NAME:-"deepseek-ai/DeepSeek-OCR"}
MODEL_TYPE=${MODEL_TYPE:-"auto"}  # auto, deepseek-ocr, qwen3-vl, generic
VLLM_HOST=${VLLM_HOST:-"0.0.0.0"}
VLLM_PORT=${VLLM_PORT:-"8000"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-"0.6"}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-"16"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"4096"}
MM_PROCESSOR_CACHE_GB=${MM_PROCESSOR_CACHE_GB:-"0"}

# Optional advanced parameters
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-""}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-""}
EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS:-""}

# -----------------------------------------------------------------------------
# Auto-detect Model Type
# -----------------------------------------------------------------------------
if [[ "$MODEL_TYPE" == "auto" ]]; then
    echo "🔍 Auto-detecting model type from MODEL_NAME: $MODEL_NAME"

    if [[ "$MODEL_NAME" == *"DeepSeek-OCR"* ]]; then
        MODEL_TYPE="deepseek-ocr"
    elif [[ "$MODEL_NAME" == *"Qwen3-VL"* ]] || [[ "$MODEL_NAME" == *"Qwen/Qwen3"* ]]; then
        MODEL_TYPE="qwen3-vl"
    else
        MODEL_TYPE="generic"
        echo "⚠️  Unknown model, using generic configuration"
    fi

    echo "✅ Detected model type: $MODEL_TYPE"
fi

# -----------------------------------------------------------------------------
# Model-Specific Configuration
# -----------------------------------------------------------------------------
MODEL_SPECIFIC_ARGS=""

case "$MODEL_TYPE" in
    deepseek-ocr)
        echo "📝 Applying DeepSeek-OCR specific configuration..."
        MODEL_SPECIFIC_ARGS="--logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor"

        # DeepSeek-OCR requires disabling prefix caching
        if [[ -z "$ENABLE_PREFIX_CACHING" ]]; then
            MODEL_SPECIFIC_ARGS="$MODEL_SPECIFIC_ARGS --no-enable-prefix-caching"
        fi

        # Default max model length for DeepSeek-OCR
        if [[ "$MAX_MODEL_LEN" == "4096" ]]; then
            MAX_MODEL_LEN="4096"
        fi
        ;;

    qwen3-vl)
        echo "📝 Applying Qwen3-VL specific configuration..."

        # Qwen3-VL benefits from data-parallel mode for vision encoder
        MODEL_SPECIFIC_ARGS="--mm-encoder-tp-mode data"

        # Optionally limit video inputs (set to 0 for image-only workloads)
        if [[ -n "${LIMIT_VIDEO_INPUTS:-}" ]]; then
            MODEL_SPECIFIC_ARGS="$MODEL_SPECIFIC_ARGS --limit-mm-per-prompt.video ${LIMIT_VIDEO_INPUTS}"
        fi

        # Async scheduling can improve throughput
        if [[ "${ENABLE_ASYNC_SCHEDULING:-false}" == "true" ]]; then
            MODEL_SPECIFIC_ARGS="$MODEL_SPECIFIC_ARGS --async-scheduling"
            echo "  → Enabled async scheduling"
        fi
        ;;

    generic)
        echo "📝 Using generic vLLM configuration..."
        # No model-specific arguments
        ;;

    *)
        echo "❌ Unknown MODEL_TYPE: $MODEL_TYPE"
        exit 1
        ;;
esac

# -----------------------------------------------------------------------------
# Build vLLM Command
# -----------------------------------------------------------------------------
VLLM_CMD="vllm serve $MODEL_NAME"
VLLM_CMD="$VLLM_CMD --host $VLLM_HOST"
VLLM_CMD="$VLLM_CMD --port $VLLM_PORT"
VLLM_CMD="$VLLM_CMD --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
VLLM_CMD="$VLLM_CMD --max-num-seqs $MAX_NUM_SEQS"
VLLM_CMD="$VLLM_CMD --max-model-len $MAX_MODEL_LEN"
VLLM_CMD="$VLLM_CMD --mm-processor-cache-gb $MM_PROCESSOR_CACHE_GB"

# Add tensor parallel if specified
if [[ -n "$TENSOR_PARALLEL_SIZE" ]]; then
    VLLM_CMD="$VLLM_CMD --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
fi

# Add model-specific arguments
if [[ -n "$MODEL_SPECIFIC_ARGS" ]]; then
    VLLM_CMD="$VLLM_CMD $MODEL_SPECIFIC_ARGS"
fi

# Add extra user-provided arguments
if [[ -n "$EXTRA_VLLM_ARGS" ]]; then
    VLLM_CMD="$VLLM_CMD $EXTRA_VLLM_ARGS"
fi

# -----------------------------------------------------------------------------
# Display Configuration and Launch
# -----------------------------------------------------------------------------
echo ""
echo "=================================="
echo "🚀 vLLM OCR Service Configuration"
echo "=================================="
echo "Model Name:           $MODEL_NAME"
echo "Model Type:           $MODEL_TYPE"
echo "Host:                 $VLLM_HOST"
echo "Port:                 $VLLM_PORT"
echo "GPU Memory Util:      $GPU_MEMORY_UTILIZATION"
echo "Max Num Seqs:         $MAX_NUM_SEQS"
echo "Max Model Length:     $MAX_MODEL_LEN"
echo "MM Processor Cache:   $MM_PROCESSOR_CACHE_GB GB"
if [[ -n "$TENSOR_PARALLEL_SIZE" ]]; then
    echo "Tensor Parallel:      $TENSOR_PARALLEL_SIZE"
fi
echo "=================================="
echo ""
echo "📋 Full command:"
echo "$VLLM_CMD"
echo ""
echo "🔄 Starting vLLM server..."
echo ""

# Execute the command
exec $VLLM_CMD