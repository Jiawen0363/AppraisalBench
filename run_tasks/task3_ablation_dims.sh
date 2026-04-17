#!/usr/bin/env bash
# Task 3 ablation: seven runs, each with exactly one appraisal_expansion dimension in the prompt.
# This script always appends to existing output files.
# Usage: bash run_tasks/task3_ablation_dims.sh openai
# Optional env: same as task3.sh (DIALOG_FILE, SCENARIOS_FILE, EVAL_PROMPT, etc.)
# Outputs: output/evaluation/task3_ablation/<model>_<dim>.jsonl
set -euo pipefail

TASK3_PROVIDER=openai

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

if [[ -f "$ROOT/run_tasks/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/run_tasks/.env"
  set +a
fi

provider="${1:-$TASK3_PROVIDER}"
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
else
  echo "Unknown provider: $provider (use openai or deepseek)" >&2
  exit 1
fi

eval_prompt="${EVAL_PROMPT:-task3/eval_emotion_dialog_plus_specific_appraisal}"
dialog_file="${DIALOG_FILE:-output/dialog/gpt4o/dialog_advanced.jsonl}"
scenarios_file="${SCENARIOS_FILE:-output/seed2scenario/scenarios.jsonl}"
tag="${eval_model//\//_}"
out_dir="${TASK3_ABLATION_OUT_DIR:-output/evaluation/task3_ablation}"
mkdir -p "$out_dir"

DIMS=(attention certainty effort pleasantness responsibility control circumstance)

echo "provider=$provider model=$eval_model -> $out_dir/${tag}_<dim>.jsonl" >&2

for dim in "${DIMS[@]}"; do
  output_file="${out_dir}/${tag}_${dim}.jsonl"
  echo "=== ablation dim=$dim -> $output_file ===" >&2
  python3 evaluator/run_task3_dialog_emotion_eval.py \
    --vllm_endpoint "$vllm_endpoint" \
    --eval_model "$eval_model" \
    --eval_prompt "$eval_prompt" \
    --dialog_file "$dialog_file" \
    --scenarios_file "$scenarios_file" \
    --output_file "$output_file" \
    --only_appraisal_dim "$dim" \
    --append
done

echo "Done. Outputs under $out_dir" >&2
