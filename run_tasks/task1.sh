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
  export QWEN_BASE_URL="${QWEN_BASE_URL:-http://127.0.0.1:8001/v1}"
  export QWEN_API_KEY="${QWEN_API_KEY:-EMPTY}"
  # IMPORTANT: must match /v1/models returned id, e.g. /data/models/Qwen3-8B
  export QWEN_MODEL="${QWEN_MODEL:-/data/models/Qwen3-8B}"
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
model_leaf="${eval_model##*/}"
output_file="${OUTPUT_FILE:-output/evaluation/task1/${model_leaf}.jsonl}"
# Resume support:
# - TASK1_OFFSET: skip first N dialog rows (0-based)
# - TASK1_APPEND=1: append to output_file instead of overwrite
TASK1_OFFSET="${TASK1_OFFSET:-0}"
TASK1_APPEND="${TASK1_APPEND:-1}"
mkdir -p "$(dirname "$output_file")"

echo "provider=$provider model=$eval_model endpoint=$vllm_endpoint -> $output_file" >&2
[[ "$TASK1_OFFSET" != "0" ]] && echo "TASK1_OFFSET=$TASK1_OFFSET (resume)" >&2
[[ "$TASK1_APPEND" == "1" ]] && echo "TASK1_APPEND=1 (appending)" >&2

extra_py=()
if [[ "$TASK1_APPEND" != "1" && "$TASK1_OFFSET" != "0" ]]; then
  extra_py+=(--offset "$TASK1_OFFSET")
fi
if [[ "$TASK1_APPEND" == "1" ]]; then
  extra_py+=(--append)
  current_lines=0
  if [[ -f "$output_file" ]]; then
    current_lines=$(wc -l < "$output_file")
  fi
  resume_offset="$current_lines"
  if [[ "$TASK1_OFFSET" =~ ^[0-9]+$ ]] && (( TASK1_OFFSET > current_lines )); then
    # Optional floor: if TASK1_OFFSET is set larger than existing lines, honor it.
    resume_offset="$TASK1_OFFSET"
  fi
  extra_py+=(--offset "$resume_offset")
  echo "TASK1_AUTO_OFFSET=$resume_offset (from existing output lines)" >&2
fi

python3 evaluator/run_task1_dialog_emotion_eval.py \
  --vllm_endpoint "$vllm_endpoint" \
  --eval_model "$eval_model" \
  --eval_prompt "$eval_prompt" \
  --dialog_file "$dialog_file" \
  --scenarios_file "$scenarios_file" \
  --output_file "$output_file" \
  "${extra_py[@]}"

echo "Done: $output_file" >&2
