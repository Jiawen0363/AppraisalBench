#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server for the evaluator model.
# Then run e.g. scripts/run_base_eval.sh with base_url=http://localhost:${PORT}

MODEL="/data/models/Qwen3-8B"
PORT="8000"
HOST="0.0.0.0"

# 8B uses ~15GB; use most of 24GB (0.9) so model + KV cache fit. max_num_seqs 1 reduces KV use.
vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype auto \
  --chat-template scripts/qwen3_nonthinking.jinja \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 1
