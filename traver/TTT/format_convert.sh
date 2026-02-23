#!/bin/bash

# 初始化conda
eval "$(conda shell.bash hook)"

# 设置基础目录
base_dir="/home/wangjian/Coding-Tutor-Extension"
cd "$base_dir"

# 激活conda环境
conda activate pagoda

# 检查是否提供了MODEL_NAME参数
if [ $# -eq 0 ]; then
    echo "Error: Please provide MODEL_NAME as the first argument"
    echo "Usage: $0 <MODEL_NAME>"
    exit 1
fi

STUDENT_LEVEL=("low_level" "med_level" "high_level")
TUTOR_SETTING="vanilla"
ITERATION="first_iter"
# 设置参数
PROMPT_ELEMENTS_FILE="prompt/prompt_elements_final.jsonl"
MODEL_NAME="$1"  # 使用第一个参数作为MODEL_NAME
MODEL_NAME_OR_PATH="/data/models/Qwen3-4B"

for level in "${STUDENT_LEVEL[@]}"; do
    INPUT_FILE="output/scored/$TUTOR_SETTING/$ITERATION/$MODEL_NAME/$level/simulated_dialogs_scoring.json"
    # if [ "$MODEL_NAME" == "Qwen3-4B-sft-279" ] || [ "$MODEL_NAME" == "Qwen3-4B-sft-200" ]; then
    OUTPUT_FILE="output/ppo_scored/$TUTOR_SETTING/$ITERATION/$MODEL_NAME/${MODEL_NAME}_${level}.jsonl"

    # else
    #     echo "Error: Invalid model name"
    #     exit 1
    # fi

# 确保输出目录存在
mkdir -p "$(dirname "$OUTPUT_FILE")"

# 运行格式转换
python3 traver/TTT/format_convert.py \
    --prompt_elements_file "$PROMPT_ELEMENTS_FILE" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE"

echo "✅ 格式转换完成!"
done