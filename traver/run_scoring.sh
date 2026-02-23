#!/bin/bash
base_dir="/home/wangjian/Coding-Tutor-Extension"
export CUDA_VISIBLE_DEVICES=7

VERIFIER_BASE_MODEL_PATH="/data/models/Mistral-7B-v0.1"
VERIFIER_MODEL_DIR="$base_dir/Verifier-7B"
ELEMENTS_FILE="$base_dir/prompt/prompt_elements_final.jsonl"
TEMPLATE_FILE="$base_dir/prompt/template/verifier.txt"

TUTOR_SETTING="vanilla"
TUTOR_MODEL="Qwen3-8B-sft-200"
ITERATION="first_iter"


STUDENT_LEVEL=("low_level" "med_level" "high_level")
for level in "${STUDENT_LEVEL[@]}"; do
    DIALOG_FILE="$base_dir/output/dialogue/$TUTOR_SETTING/$ITERATION/$TUTOR_MODEL/$level/simulated_dialogs.json"
    NAMESPACES_FILE="$base_dir/prompt/namespaces.json"
    OUTPUT_FILE="$base_dir/output/scored/$TUTOR_SETTING/$ITERATION/$TUTOR_MODEL/$level/simulated_dialogs_scoring.json"
    python3 traver/run_scoring.py \
        --dialog_file "$DIALOG_FILE" \
        --namespaces_file "$NAMESPACES_FILE" \
        --output_file "$OUTPUT_FILE" \
        --verifier_base_model_path "$VERIFIER_BASE_MODEL_PATH" \
        --verifier_model_dir "$VERIFIER_MODEL_DIR" \
        --elements_file "$ELEMENTS_FILE" \
        --template_file "$TEMPLATE_FILE"
done

bash traver/TTT/format_convert.sh "$TUTOR_MODEL"

echo ""
echo "✅ 脚本执行完成!" 