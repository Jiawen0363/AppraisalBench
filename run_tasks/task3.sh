#!/usr/bin/env bash
# Task 3: dialog + scenario appraisal_expansion emotion eval (gold from scenarios.jsonl).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

if [[ -f "$ROOT/run_tasks/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/run_tasks/.env"
  set +a
fi

if [[ -z "${1:-}" ]]; then
  echo "Missing provider: pass openai or deepseek or qwen as the first argument." >&2
  exit 1
fi
provider="${1}"
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
  vllm_endpoint="${QWEN_BASE_URL:-http://127.0.0.1:8003/v1}"
  eval_model="${QWEN_MODEL:-/data/models/Qwen3-4B}"
  export OPENAI_API_KEY="${QWEN_API_KEY:-EMPTY}"
else
  echo "Unknown provider: $provider (use openai|deepseek|qwen)" >&2
  exit 1
fi

eval_prompt="${EVAL_PROMPT:-task3/eval_emotion_dialog_plus_appraisal}"
dialog_file="${DIALOG_FILE:-output/dialog/gpt4o/dialog_advanced.jsonl}"
scenarios_file="${SCENARIOS_FILE:-output/seed2scenario/scenarios.jsonl}"
tag="${eval_model//\//_}"
output_file="${OUTPUT_FILE:-output/evaluation/task3/${tag}.jsonl}"
# Resume: skip first N dialogs (0-based). E.g. TASK3_OFFSET=10 starts at the 11th dialog line.
TASK3_OFFSET="${TASK3_OFFSET:-0}"
# Set TASK3_APPEND=1 to append to output_file instead of truncating.
TASK3_APPEND="${TASK3_APPEND:-0}"
mkdir -p "$(dirname "$output_file")"

echo "provider=$provider model=$eval_model endpoint=$vllm_endpoint -> $output_file" >&2
[[ "$TASK3_OFFSET" != "0" ]] && echo "TASK3_OFFSET=$TASK3_OFFSET (1-based next line is $((TASK3_OFFSET + 1)))" >&2
[[ "$TASK3_APPEND" == "1" ]] && echo "TASK3_APPEND=1 (appending)" >&2

extra_py=()
if [[ "$TASK3_OFFSET" != "0" ]]; then
  extra_py+=(--offset "$TASK3_OFFSET")
fi
if [[ "$TASK3_APPEND" == "1" ]]; then
  extra_py+=(--append)
fi
if [[ -n "${TASK3_ONLY_APPRAISAL_DIM:-}" ]]; then
  extra_py+=(--only_appraisal_dim "$TASK3_ONLY_APPRAISAL_DIM")
  echo "TASK3_ONLY_APPRAISAL_DIM=$TASK3_ONLY_APPRAISAL_DIM (single-dim ablation)" >&2
fi

python3 evaluator/run_task3_dialog_emotion_eval.py \
  --vllm_endpoint "$vllm_endpoint" \
  --eval_model "$eval_model" \
  --eval_prompt "$eval_prompt" \
  --dialog_file "$dialog_file" \
  --scenarios_file "$scenarios_file" \
  --output_file "$output_file" \
  "${extra_py[@]}"

echo "Done: $output_file" >&2
