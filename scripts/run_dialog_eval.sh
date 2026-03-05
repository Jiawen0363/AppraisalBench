#!/usr/bin/env bash
# Evaluator model: start with scripts/run_engine_evaluator.sh, then run this script.
# Set port to match the evaluator server (run_engine_evaluator.sh uses PORT=8000 by default).

eval_model="Qwen3-4B"
user_model="Qwen3-4B"
assistant_model="Qwen3-4B"
eval_prompt="eval_base"
# mode: dialog_first (first User message only), dialog_full (full conversation)
mode="dialog_full"
event_corpus="enISEAR"
dialog_file="output/dialog/dialogs.jsonl"

# Evaluator API port (must match run_engine_evaluator.sh)
port="8003"
export VLLM_BASE_URL="http://localhost:${port}/v1"

cd "$(dirname "$0")/.." || exit 1

python3 evaluator/run_dialog_eval.py \
  --eval_model "/data/models/$eval_model" \
  --eval_prompt "$eval_prompt" \
  --event_corpus "$event_corpus" \
  --mode "$mode" \
  --dialog_file "$dialog_file" \
  --output_file output/evaluation/$mode/${user_model}_${assistant_model}_${eval_model}_raw.jsonl
