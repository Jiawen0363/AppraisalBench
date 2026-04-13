#!/usr/bin/env bash
# Task 1: dialog emotion eval. Keys in run_tasks/.env (OPENAI_API_KEY or DEEPSEEK_API_KEY).
set -euo pipefail

# openai = gpt-4o + relay | deepseek = DeepSeek official API
TASK1_PROVIDER=openai

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

if [[ -f "$ROOT/run_tasks/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/run_tasks/.env"
  set +a
fi

provider="${1:-$TASK1_PROVIDER}"
provider="${provider,,}"

if [[ "$provider" == "deepseek" ]]; then
  vllm_endpoint="${DEEPSEEK_BASE_URL:-https://api.deepseek.com/v1}"
  eval_model="${DEEPSEEK_MODEL:-deepseek-chat}"
  export OPENAI_API_KEY="${DEEPSEEK_API_KEY:-}"
  if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "Set DEEPSEEK_API_KEY in run_tasks/.env (or export it) for provider=deepseek" >&2
    exit 1
  fi
elif [[ "$provider" == "openai" ]]; then
  vllm_endpoint="${OPENAI_BASE_URL:-http://35.164.11.19:3887/v1}"
  eval_model="${OPENAI_MODEL:-gpt-4o}"
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Set OPENAI_API_KEY in run_tasks/.env for provider=openai" >&2
    exit 1
  fi
elif [[ "$provider" == "qwen" ]]; then
  # Local vLLM (OpenAI-compatible) for Qwen models.
  export QWEN_BASE_URL="${QWEN_BASE_URL:-http://127.0.0.1:8002/v1}"
  export QWEN_API_KEY="${QWEN_API_KEY:-EMPTY}"
  # IMPORTANT: must match /v1/models returned id.
  export QWEN_MODEL="${QWEN_MODEL:-/data/models/Qwen3-4B}"
  vllm_endpoint="$QWEN_BASE_URL"
  eval_model="$QWEN_MODEL"
  export OPENAI_API_KEY="$QWEN_API_KEY"
else
  echo "Unknown provider: $provider (use openai|deepseek|qwen, or: bash $0 openai|deepseek|qwen)" >&2
  exit 1
fi

eval_prompt="${EVAL_PROMPT:-task1/eval_emotion_base}"
dialog_file="${DIALOG_FILE:-output/dialog/gpt4o/dialog_advanced.jsonl}"
scenarios_file="${SCENARIOS_FILE:-output/seed2scenario/scenarios.jsonl}"
tag="${eval_model//\//_}"
output_file="${OUTPUT_FILE:-output/evaluation/task1/${tag}.jsonl}"
mkdir -p "$(dirname "$output_file")"

echo "provider=$provider model=$eval_model endpoint=$vllm_endpoint -> $output_file" >&2

python3 evaluator/run_task1_dialog_emotion_eval.py \
  --vllm_endpoint "$vllm_endpoint" \
  --eval_model "$eval_model" \
  --eval_prompt "$eval_prompt" \
  --dialog_file "$dialog_file" \
  --scenarios_file "$scenarios_file" \
  --output_file "$output_file"

echo "Done: $output_file" >&2
