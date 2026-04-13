#!/usr/bin/env bash
# Task 4: dialog + gold emotion + 4-option appraisal MCQA (same question JSONL as task2).
#
# OpenAI-compatible relay (see run_tasks/.env):
#   OPENAI_BASE_URL=http://35.164.11.19:3887/v1
#   OPENAI_API_KEY=sk-xxxx
#   OPENAI_MODEL=gpt-4o
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
  echo "Missing provider: pass openai or deepseek as the first argument, e.g.: bash $0 openai" >&2
  exit 1
fi
provider="${1,,}"

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
  # IMPORTANT: must match /v1/models returned id, e.g. /data/models/Qwen3-4B
  export QWEN_MODEL="${QWEN_MODEL:-/data/models/Qwen3-4B}"
  vllm_endpoint="$QWEN_BASE_URL"
  eval_model="$QWEN_MODEL"
  export OPENAI_API_KEY="$QWEN_API_KEY"
else
  echo "Unknown provider: $provider (use: bash $0 openai|deepseek|qwen)" >&2
  exit 1
fi

eval_prompt="${EVAL_PROMPT:-task4/given_dialog_emotion_infer_appraisal}"
question_dir="${QUESTION_DIR:-output/evaluation/task2_question}"
tag="${eval_model//\//_}"
output_dir="${OUTPUT_DIR:-output/evaluation/task4/${tag}}"
mkdir -p "$output_dir"

shopt -s nullglob
jsonl_paths=("$ROOT/$question_dir"/*.jsonl)
if [[ ${#jsonl_paths[@]} -eq 0 ]]; then
  echo "No *.jsonl under $ROOT/$question_dir" >&2
  exit 1
fi

echo "provider=$provider model=$eval_model endpoint=$vllm_endpoint" >&2
echo "questions=$question_dir -> $output_dir" >&2

for abs in "${jsonl_paths[@]}"; do
  base="$(basename "$abs")"
  if [[ "$base" == debug_* ]]; then
    continue
  fi
  stem="${base%.jsonl}"
  rel="${abs#$ROOT/}"
  out_file="${output_dir}/${stem}.jsonl"
  echo "  $rel -> $out_file" >&2
  python3 evaluator/run_task4_appraisal_mcqa_eval.py \
    --vllm_endpoint "$vllm_endpoint" \
    --eval_model "$eval_model" \
    --eval_prompt "$eval_prompt" \
    --question_file "$rel" \
    --output_file "${out_file#$ROOT/}"
done

echo "Done under $output_dir" >&2
