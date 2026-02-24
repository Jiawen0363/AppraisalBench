#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server for the evaluator model.
# Then run e.g. scripts/run_base_eval.sh with base_url=http://localhost:${PORT}

MODEL="/data/models/Qwen3-4B"
PORT="8000"
HOST="0.0.0.0"

# Reduce GPU memory use to avoid OOM on 24GB: lower gpu_memory_utilization and max_num_seqs
vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype auto \
  --chat-template scripts/qwen3_nonthinking.jinja \
  --gpu-memory-utilization 0.7
