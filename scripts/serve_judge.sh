#!/usr/bin/env bash
# Serve the single-GPU Qwen3-32B quality judge through a vLLM endpoint.

set -euo pipefail

JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:?Set JUDGE_MODEL_PATH to the Qwen3-32B model directory}"
JUDGE_GPU="${JUDGE_GPU:-0}"
JUDGE_HOST="${JUDGE_HOST:-127.0.0.1}"
JUDGE_PORT="${JUDGE_PORT:-8022}"
JUDGE_SERVED_NAME="${JUDGE_SERVED_NAME:-qwen-32b}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-16384}"
JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.92}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export CUDA_VISIBLE_DEVICES="$JUDGE_GPU"

exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$JUDGE_MODEL_PATH" \
  --served-model-name "$JUDGE_SERVED_NAME" \
  --host "$JUDGE_HOST" \
  --port "$JUDGE_PORT" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --gpu-memory-utilization "$JUDGE_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$JUDGE_MAX_MODEL_LEN"
