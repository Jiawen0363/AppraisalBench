#!/usr/bin/env bash
# Evaluator model: start with scripts/run_engine_evaluator.sh, then run this script.
# Set port to match the evaluator server (run_engine_evaluator.sh uses PORT=8000 by default).

eval_model="Qwen3-4B"
user_model="Qwen3-4B"
assistant_model="Qwen3-4B"
# eval_prompt: only used when mode is NOT *_given_appraisal. For *_given_appraisal, eval_given_appraisal is always used.
eval_prompt="eval_given_appraisal"
# mode: dialog_first | dialog_full | dialog_first_given_appraisal | dialog_full_given_appraisal
# _given_appraisal: use gold appraisal from emotion_appraisal_corpus.tsv, evaluator only predicts emotion
mode="dialog_full_given_appraisal"
event_corpus="enISEAR"
# 与 run_dialog.sh 输出命名一致；若用 profile 可改为 output/dialog/${user_model}_${assistant_model}_with_profile.jsonl
dialog_file="output/dialog/${user_model}.jsonl"

# Evaluator API (must match run_engine_evaluator.sh: port 8004, served-model-name e.g. Qwen3-8B)
vllm_endpoint="http://localhost:8100/v1"

cd "$(dirname "$0")/.." || exit 1

python3 evaluator/run_dialog_eval.py \
  --vllm_endpoint "$vllm_endpoint" \
  --eval_model "$eval_model" \
  --eval_prompt "$eval_prompt" \
  --event_corpus "$event_corpus" \
  --mode "$mode" \
  --dialog_file "$dialog_file" \
  --output_file output/evaluation/$mode/${user_model}_${assistant_model}_${eval_model}.jsonl
