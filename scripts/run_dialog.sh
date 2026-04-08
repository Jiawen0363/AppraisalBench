#!/usr/bin/env bash
# Prompt data mode:
# - base: user_base/user_with_profile + profile_file + corpus_file
# - advanced: user_advanced + scenario_file (from seed2scenario)
dialog_mode="base"  # base or advanced

if [[ "$dialog_mode" == "advanced" ]]; then
  user_prompt="user_advanced.txt"
  profile_file="translated_user_profile.jsonl"   # unused in advanced mode
  corpus_file="emotion_appraisal_corpus.tsv"     # unused in advanced mode
  scenario_file="output/seed2scenario/scenarios.jsonl"
  output_file="output/dialog/gpt4o/dialog_advanced.jsonl"
else
  # user_prompt="user_base.txt"
  user_prompt="user_with_profile.txt"
  profile_file="translated_user_profile.jsonl"
  corpus_file="emotion_appraisal_corpus.tsv"
  scenario_file=""
  output_file="output/dialog/gpt4o/dialog_base.jsonl"
fi

assistant_prompt="assistant_base.txt"

backend="gpt"  # vllm or gpt

if [[ "$backend" == "gpt" ]]; then
  user_model="gpt-4o-mini"
  assistant_model="gpt-4o-mini"
  endpoint_user="https://api.openai.com/v1"
  endpoint_assistant="https://api.openai.com/v1"
  user_api_key="${OPENAI_API_KEY:-}"
  assistant_api_key="${ASSISTANT_OPENAI_API_KEY:-${OPENAI_API_KEY:-}}"
else
  user_model="Qwen3-8B"
  assistant_model="Qwen3-8B"
  endpoint_user="http://localhost:8000/v1"
  endpoint_assistant="http://localhost:8001/v1"
  # vLLM's OpenAI-compatible server usually accepts any API key string.
  user_api_key="EMPTY"
  assistant_api_key="EMPTY"
fi

max_rounds=4


cd "$(dirname "$0")/.." || exit 1

python3 interaction/run_dialog.py \
  --dialog_mode "$dialog_mode" \
  --user_prompt "interaction/prompt/$user_prompt" \
  --assistant_prompt "interaction/prompt/$assistant_prompt" \
  --profile_file "$profile_file" \
  --corpus_file "$corpus_file" \
  --scenario_file "$scenario_file" \
  --output_file "$output_file" \
  --user_model "$user_model" \
  --assistant_model "$assistant_model" \
  --vllm_endpoint_user "$endpoint_user" \
  --vllm_endpoint_assistant "$endpoint_assistant" \
  --user_api_key "$user_api_key" \
  --assistant_api_key "$assistant_api_key" \
  --max_rounds "$max_rounds"
