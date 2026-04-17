#!/usr/bin/env bash
# Advanced dialog mode only:
# user_advanced.txt + assistant_base.txt + scenarios.jsonl

set -euo pipefail

# Backend: vllm or gpt
backend="gpt"

# Prompt + data (fixed for advanced mode)
user_prompt="user_advanced.txt"
assistant_prompt="assistant_base.txt"
scenario_file="output/seed2scenario/scenarios.jsonl"
output_file="output/dialog/gpt4o/dialog_advanced.jsonl"

max_rounds=3
limit=""

if [[ "$backend" == "gpt" ]]; then
  user_model="gpt-4o"
  assistant_model="gpt-4o"
  # Keep defaults aligned with seed2scenario/run_seed2scenario.py.
  default_api_key="sk-8kgU8Q3tLwhvJ6wtGGVD46z0kIGIaWZElebK6Ag5NIDL18Xe"
  # Reuse your previously working GPT-compatible endpoint when provided.
  # Priority:
  # 1) OPENAI_BASE_URL
  # 2) GPT_API_BASE_URL
  # 3) same default gateway as seed2scenario/run_seed2scenario.py
  # 4) official OpenAI endpoint
  gpt_base_url="${OPENAI_BASE_URL:-${GPT_API_BASE_URL:-http://35.164.11.19:3887/v1}}"
  endpoint_user="$gpt_base_url"
  endpoint_assistant="$gpt_base_url"
  user_api_key="${OPENAI_API_KEY:-$default_api_key}"
  assistant_api_key="${ASSISTANT_OPENAI_API_KEY:-${OPENAI_API_KEY:-$default_api_key}}"
else
  user_model="Qwen3-8B"
  assistant_model="Qwen3-8B"
  endpoint_user="http://localhost:8000/v1"
  endpoint_assistant="http://localhost:8001/v1"
  user_api_key="EMPTY"
  assistant_api_key="EMPTY"
fi

cd "$(dirname "$0")/.." || exit 1

echo "[run_dialog_advanced] backend=$backend"
echo "[run_dialog_advanced] user_model=$user_model assistant_model=$assistant_model"
echo "[run_dialog_advanced] endpoint_user=$endpoint_user"

cmd=(
  python3 interaction/run_dialog.py
  --dialog_mode advanced
  --user_prompt "interaction/prompt/$user_prompt"
  --assistant_prompt "interaction/prompt/$assistant_prompt"
  --scenario_file "$scenario_file"
  --output_file "$output_file"
  --user_model "$user_model"
  --assistant_model "$assistant_model"
  --vllm_endpoint_user "$endpoint_user"
  --vllm_endpoint_assistant "$endpoint_assistant"
  --user_api_key "$user_api_key"
  --assistant_api_key "$assistant_api_key"
  --max_rounds "$max_rounds"
)

if [[ -n "$limit" ]]; then
  cmd+=(--limit "$limit")
fi

"${cmd[@]}"
