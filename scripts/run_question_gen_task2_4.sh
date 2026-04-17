#!/bin/bash

cd /home/jiawen/AppraisalBench

set -a
source run_tasks/.env
set +a

PYTHONPATH=. python3 evaluator/run_task2_build_binary_qa.py \
  --scenarios output/seed2scenario/scenarios.jsonl \
  --dialogs output/dialog/gpt4o/dialog_advanced.jsonl \
  --output-dir output/evaluation/task2_question \
  --offset 635 \
  --append \
  --dimensions all