#!/usr/bin/env bash

eval_model="Qwen3-8B"
eval_prompt="eval_base"
# mode: base
mode="base_given_appraisal"
event_corpus="enISEAR"
vllm_endpoint="http://localhost:8004/v1"

python3 evaluator/run_base_eval.py \
    --vllm_endpoint "$vllm_endpoint" \
    --eval_model "$eval_model" \
    --eval_prompt "$eval_prompt" \
    --event_corpus "$event_corpus" \
    --mode "$mode" \
    --output_file output/evaluation/$mode/$eval_model.jsonl

