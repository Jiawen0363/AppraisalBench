#!/usr/bin/env bash
# Task 5: seed event + dimension definition + 7-option appraisal MCQA.
# By default reads scenarios.jsonl and builds each question on the fly (no task5_question JSONL required).
# Optional: export questions with python3 evaluator/run_task5_build_7way_mcq.py, then set
#   TASK5_USE_QUESTION_JSONL=1 and QUESTION_DIR=output/evaluation/task5_question to evaluate from files.
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
  echo "Missing provider: pass openai, deepseek, or qwen as the first argument, e.g.: bash $0 openai" >&2
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
  export QWEN_BASE_URL="${QWEN_BASE_URL:-http://127.0.0.1:8005/v1}"
  export QWEN_API_KEY="${QWEN_API_KEY:-EMPTY}"
  # IMPORTANT: must match /v1/models returned id, e.g. /data/models/Qwen3-8B
  export QWEN_MODEL="${QWEN_MODEL:-/data/models/Qwen3-8B}"
  vllm_endpoint="$QWEN_BASE_URL"
  eval_model="$QWEN_MODEL"
  export OPENAI_API_KEY="$QWEN_API_KEY"
else
  echo "Unknown provider: $provider (use: bash $0 openai|deepseek|qwen)" >&2
  exit 1
fi

eval_prompt="${EVAL_PROMPT:-task5/given_seed_definition_match_appraisal_expansion}"
scenarios_file="${SCENARIOS_FILE:-output/seed2scenario/scenarios.jsonl}"
shuffle_seed="${TASK5_SHUFFLE_SEED:-42}"
model_leaf="${eval_model##*/}"
output_dir="${OUTPUT_DIR:-output/evaluation/task5/${model_leaf}}"
TASK5_OFFSET="${TASK5_OFFSET:-0}"
TASK5_APPEND="${TASK5_APPEND:-1}"
mkdir -p "$output_dir"

extra_eval_args=()
if [[ -n "${LIMIT:-}" ]]; then
  extra_eval_args+=(--limit "${LIMIT}")
fi
if [[ "$TASK5_APPEND" != "1" && "$TASK5_OFFSET" != "0" ]]; then
  extra_eval_args+=(--offset "$TASK5_OFFSET")
fi
if [[ "$TASK5_APPEND" == "1" ]]; then
  extra_eval_args+=(--append)
fi

echo "provider=$provider model=$eval_model endpoint=$vllm_endpoint" >&2
[[ "$TASK5_OFFSET" != "0" ]] && echo "TASK5_OFFSET=$TASK5_OFFSET (resume per dimension file)" >&2
[[ "$TASK5_APPEND" == "1" ]] && echo "TASK5_APPEND=1 (appending)" >&2

if [[ "${TASK5_USE_QUESTION_JSONL:-0}" == "1" ]]; then
  question_dir="${QUESTION_DIR:-output/evaluation/task5_question}"
  shopt -s nullglob
  jsonl_paths=("$ROOT/$question_dir"/*.jsonl)
  if [[ ${#jsonl_paths[@]} -eq 0 ]]; then
    echo "TASK5_USE_QUESTION_JSONL=1 but no *.jsonl under $ROOT/$question_dir" >&2
    echo "Run: python3 evaluator/run_task5_build_7way_mcq.py" >&2
    exit 1
  fi
  echo "mode=prebuilt_jsonl questions=$question_dir -> $output_dir" >&2
  for abs in "${jsonl_paths[@]}"; do
    base="$(basename "$abs")"
    if [[ "$base" == debug_* ]]; then
      continue
    fi
    stem="${base%.jsonl}"
    rel="${abs#$ROOT/}"
    out_file="${output_dir}/${stem}.jsonl"
    echo "  $rel -> $out_file" >&2
    run_eval_args=("${extra_eval_args[@]}")
    if [[ "$TASK5_APPEND" == "1" ]]; then
      current_lines=0
      if [[ -f "$out_file" ]]; then
        current_lines=$(wc -l < "$out_file")
      fi
      resume_offset="$current_lines"
      if [[ "$TASK5_OFFSET" =~ ^[0-9]+$ ]] && (( TASK5_OFFSET > current_lines )); then
        # Optional floor: if TASK5_OFFSET is set larger than existing lines, honor it.
        resume_offset="$TASK5_OFFSET"
      fi
      run_eval_args+=(--offset "$resume_offset")
      echo "    resume_offset=$resume_offset (auto from existing output lines)" >&2
    fi
    python3 evaluator/run_task5_appraisal_dimension_mcqa_eval.py \
      --vllm_endpoint "$vllm_endpoint" \
      --eval_model "$eval_model" \
      --eval_prompt "$eval_prompt" \
      --question_file "$rel" \
      --output_file "${out_file#$ROOT/}" \
      "${run_eval_args[@]}"
  done
else
  if [[ ! -f "$ROOT/$scenarios_file" ]]; then
    echo "Missing scenarios file: $ROOT/$scenarios_file" >&2
    exit 1
  fi
  echo "mode=on_the_fly scenarios=$scenarios_file seed=$shuffle_seed -> $output_dir" >&2
  for dim in attention certainty effort pleasantness responsibility control circumstance; do
    out_file="${output_dir}/${dim}.jsonl"
    echo "  $scenarios_file dim=$dim -> $out_file" >&2
    run_eval_args=("${extra_eval_args[@]}")
    if [[ "$TASK5_APPEND" == "1" ]]; then
      current_lines=0
      if [[ -f "$out_file" ]]; then
        current_lines=$(wc -l < "$out_file")
      fi
      resume_offset="$current_lines"
      if [[ "$TASK5_OFFSET" =~ ^[0-9]+$ ]] && (( TASK5_OFFSET > current_lines )); then
        # Optional floor: if TASK5_OFFSET is set larger than existing lines, honor it.
        resume_offset="$TASK5_OFFSET"
      fi
      run_eval_args+=(--offset "$resume_offset")
      echo "    resume_offset=$resume_offset (auto from existing output lines)" >&2
    fi
    python3 evaluator/run_task5_appraisal_dimension_mcqa_eval.py \
      --vllm_endpoint "$vllm_endpoint" \
      --eval_model "$eval_model" \
      --eval_prompt "$eval_prompt" \
      --scenarios_file "$scenarios_file" \
      --appraisal_dimension "$dim" \
      --shuffle_seed "$shuffle_seed" \
      --output_file "${out_file#$ROOT/}" \
      "${run_eval_args[@]}"
  done
fi

echo "Done under $output_dir" >&2
