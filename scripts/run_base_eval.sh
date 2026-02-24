#!/usr/bin/env bash

eval_model="Qwen3-4B"
eval_prompt="eval_base"
# mode: base, dialog_first, dialog_full
mode="base"
event_corpus="enISEAR"

python3 evaluator/run_base_eval.py \
    --eval_model /data/models/"$eval_model" \
    --eval_prompt "$eval_prompt" \
    --event_corpus "$event_corpus" \
    --mode "$mode" \
    --output_file output/evaluation/$mode/$eval_model.jsonl

