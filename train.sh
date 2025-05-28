#!/bin/bash

VLLM_MODEL_PATH=".cache/Qwen2.5-3B-OurInstruct"
VLLM_SERVED_MODEL_NAME="qwen2.5-3b-ourinstruct"
VLLM_API_KEY="lococo"
VLLM_PORT=8000
VLLM_GPUS="3" 
VLLM_TP_SIZE=1
VLLM_GPU_MEM_UTIL=0.85
VLLM_LOG_FILE="./log/vllm_server.log"

#TRAINING_GPUS="0,1,2,3"
TRAINING_GPUS="0,1,2"
TRAINING_SCRIPT_PATH="train.py"


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
    local VLLM_CMD_ARGS=(
        serve
        "${VLLM_MODEL_PATH}"
        --served-model-name "${VLLM_SERVED_MODEL_NAME}"
        --host "0.0.0.0"
        --port "${VLLM_PORT}"
        --dtype half
        --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}"
        --tensor-parallel-size "${VLLM_TP_SIZE}"
    )

    if [ ! -z "${VLLM_API_KEY}" ]; then
        VLLM_CMD_ARGS+=(--api-key "${VLLM_API_KEY}")
    fi

    echo "Starting vLLM server with command:"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=${VLLM_GPUS} vllm "${VLLM_CMD_ARGS[@]}" > "${VLLM_LOG_FILE}" 2>&1 &
    VLLM_PID=$!

    sleep 2

    echo "vLLM has been successfully started at port ${VLLM_PORT}. Check logs at ${VLLM_LOG_FILE} for details."
}

start_training() {
    CUDA_VISIBLE_DEVICES=${TRAINING_GPUS} accelerate launch "${TRAINING_SCRIPT_PATH}" 
    TRAINING_EXIT_CODE=$?
}

# --- Main script execution ---
start_vllm
start_training
# trap EXIT will call cleanup