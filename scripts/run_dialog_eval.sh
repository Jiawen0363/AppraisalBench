#!/usr/bin/env bash

eval_model="Qwen3-4B"
user_model="Qwen3-4B"
assistant_model="Qwen3-4B"
eval_prompt="eval_base"
# mode: dialog_first, dialog_full
mode="dialog_first"

event_corpus="enISEAR"


python3 evaluator/run_dialog_eval.py \
    --eval_model "$eval_model" \
    --eval_prompt "$eval_prompt" \
    --event_corpus "$event_corpus" \
    --mode "$mode" \
    --output_file output/evaluation/$mode/${user_model}_${assistant_model}_${eval_model}.jsonl
