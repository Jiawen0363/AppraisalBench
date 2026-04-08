#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server for the evaluator model.
# Base eval should call: http://localhost:${port}/v1 with model=${served_model_name}

set -euo pipefail

export CUDA_VISIBLE_DEVICES="3"

tensor_parallel_size=1
port=8100
host="0.0.0.0"

model_base_path=/data/models  # TODO: change to your own path
model_name_or_path="$model_base_path/Qwen3-4B"
served_model_name="Qwen3-4B"

chat_template=scripts/qwen3_nonthinking.jinja

echo "Starting vLLM evaluator engine for $model_name_or_path (served as $served_model_name) ..."
python -m vllm.entrypoints.openai.api_server \
  --host "$host" \
  --port "$port" \
  --model "$model_name_or_path" \
  --served-model-name "$served_model_name" \
  --chat-template "$chat_template" \
  --tensor-parallel-size "$tensor_parallel_size" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --api-key "EMPTY"
