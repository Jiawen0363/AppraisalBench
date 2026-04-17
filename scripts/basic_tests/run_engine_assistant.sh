# !/bin/bash
export CUDA_VISIBLE_DEVICES="1"
tensor_parallel_size=1
port=8001

# model_base_path=/data/models  # TODO: change to your own path
# model_name_or_path=$model_base_path/Qwen3-4B

# chat_template=scripts/qwen3_nonthinking.jinja
# echo "Starting vllm engine for $model_name_or_path as tutor agent..."
# python -m vllm.entrypoints.openai.api_server \
#     --model $model_name_or_path \
#     --chat-template $chat_template \
#     --port $port \
#     --tensor-parallel-size $tensor_parallel_size \
#     --gpu-memory-utilization 0.8 \
#     --max-model-len 16384 \
#     --api-key "EMPTY"
# !/bin/bash

model_base_path=/data/models  # TODO: change to your own path
model_name_or_path=$model_base_path/Qwen3-8B

chat_template=scripts/qwen3_nonthinking.jinja
echo "Starting vllm engine for $model_name_or_path as assistant agent..."
python -m vllm.entrypoints.openai.api_server \
    --model $model_name_or_path \
    --chat-template $chat_template \
    --port $port \
    --tensor-parallel-size $tensor_parallel_size \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --api-key "EMPTY"


