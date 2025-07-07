#!/bin/bash

TRAINING_NPUS="0-7"
TRAINING_SCRIPT_PATH="train.py"

EVALING_NPUS="0"
VLLM_MODEL_PATH="/aistor/sjtu/hpc_stor01/home/wangzijian/workspace/NeuSym-RAG/.cache/Qwen2.5-3B-OurInstruct"
VLLM_SERVED_MODEL_NAME="qwen2.5-3b-ourinstruct"
VLLM_API_KEY="lococo"
VLLM_PORT=8001
VLLM_TP_SIZE=1
VLLM_GPU_MEM_UTIL=0.85
VLLM_LOG_FILE="/aistor/sjtu/hpc_stor01/home/wangzijian/workspace/NeuSym-RAG/log/train_vllm_server.log"


VLLM_PID=""
TRAINING_EXIT_CODE=0

cleanup() {
    if [ ! -z "${VLLM_PID}" ]; then
        if kill -0 "${VLLM_PID}" 2>/dev/null; then # Check if PID exists and process is running
            kill -SIGTERM "${VLLM_PID}"
            if ! wait "${VLLM_PID}" 2>/dev/null; then
                kill -SIGKILL "${VLLM_PID}"
                wait "${VLLM_PID}" 2>/dev/null
            fi
        fi
    fi
    exit ${TRAINING_EXIT_CODE}
}

trap cleanup SIGINT SIGTERM EXIT

start_vllm() {

    export PYTHONPATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe"

    cd vllm

    local VLLM_CMD_ARGS=(
        --served-model-name "${VLLM_SERVED_MODEL_NAME}"
        --model "${VLLM_MODEL_PATH}"
        --host "0.0.0.0"
        --port "${VLLM_PORT}"
        --dtype float16
        --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}"
        --tensor-parallel-size "${VLLM_TP_SIZE}"
    )

    if [ ! -z "${VLLM_API_KEY}" ]; then
        VLLM_CMD_ARGS+=(--api-key "${VLLM_API_KEY}")
    fi

    echo "Starting vLLM server with command:"
    # shellcheck disable=SC2086
    ASCEND_VISIBLE_DEVICES=${EVALING_NPUS} python -m vllm.entrypoints.openai.api_server "${VLLM_CMD_ARGS[@]}" > "${VLLM_LOG_FILE}" 2>&1 &
    VLLM_PID=$!

    sleep 30

    echo "vLLM has been successfully started at port ${VLLM_PORT}. Check logs at ${VLLM_LOG_FILE} for details."

    cd ..

}

start_training() {
    export TORCH_EXTENSIONS_DIR="/tmp/${USER}_torch_extensions"
    export VLLM_API_KEY="lococo"
    export VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
    export TOWHEE_HOME="./.towhee"
    export NLTK_DATA="./nltk_data"
    export TIKTOKEN_CACHE_DIR="."
    export HCCL_CONNECT_TIMEOUT=6000
    export MEMORY_FRAGMENTATION=1
    export environment variable ASCEND_LAUNCH_BLOCKING=1
    #export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:256'
    ASCEND_VISIBLE_DEVICES=${TRAINING_NPUS} accelerate launch "${TRAINING_SCRIPT_PATH}" 
    TRAINING_EXIT_CODE=$?
}

# --- Main script execution ---
#start_vllm
start_training