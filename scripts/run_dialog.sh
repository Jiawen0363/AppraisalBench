#!/usr/bin/env bash
# User prompt mode: user_base.txt (no persona) or user_with_profile.txt (with persona)
user_prompt="user_base.txt"
# user_prompt="user_with_profile.txt"

assistant_prompt="assistant_base.txt"

output_file="output/dialog/dialogs.jsonl"

user_model="Qwen3-4B"
assistant_model="Qwen3-4B"

vllm_endpoint_user="http://localhost:8000/v1"
vllm_endpoint_assistant="http://localhost:8001/v1"
max_rounds=4


cd "$(dirname "$0")/.." || exit 1

python3 interaction/run_dialog.py \
  --user_prompt "interaction/prompt/$user_prompt" \
  --assistant_prompt "interaction/prompt/$assistant_prompt" \
  --profile_file "translated_user_profile.jsonl" \
  --corpus_file "emotion_appraisal_corpus.tsv" \
  --output_file "$output_file" \
  --user_model "/data/models/$user_model" \
  --assistant_model "/data/models/$assistant_model" \
  --vllm_endpoint_user "$vllm_endpoint_user" \
  --vllm_endpoint_assistant "$vllm_endpoint_assistant" \
  --max_rounds "$max_rounds"
