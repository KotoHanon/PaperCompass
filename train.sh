#!/bin/bash

# --- Configuration - PLEASE EDIT THESE VALUES ---
VLLM_MODEL_PATH=".cache/Qwen2.5-3B-OurInstruct"
VLLM_SERVED_MODEL_NAME="qwen2.5-3b-ourinstruct"
VLLM_API_KEY="lococo"
VLLM_PORT=8000
VLLM_GPUS="4" 
VLLM_TP_SIZE=1
VLLM_GPU_MEM_UTIL=0.85
VLLM_LOG_FILE="./vllm_server.log"

TRAINING_GPUS="0,1,2,3"
TRAINING_SCRIPT_PATH="train.py"
# --- End of Configuration ---

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
    
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=${VLLM_GPUS} vllm "${VLLM_CMD_ARGS[@]}" > "${VLLM_LOG_FILE}" 2>&1 &
    VLLM_PID=$!

    if [ $? -ne 0 ] || ! kill -0 "${VLLM_PID}" 2>/dev/null ; then
        # Failed to start or PID not valid
        exit 1
    fi

    local RETRY_COUNT=0
    local MAX_RETRIES=60 
    local HEALTH_ENDPOINT="http://localhost:${VLLM_PORT}/health"

    until curl --output /dev/null --silent --head --fail "${HEALTH_ENDPOINT}"; do
        if [ ${RETRY_COUNT} -ge ${MAX_RETRIES} ]; then
            if kill -0 "${VLLM_PID}" 2>/dev/null; then
                 kill -SIGTERM "${VLLM_PID}"
                 wait "${VLLM_PID}" 2>/dev/null
            fi
            VLLM_PID="" 
            exit 1
        fi
        sleep 2
        RETRY_COUNT=$((RETRY_COUNT+1))
    done
}

start_training() {
    CUDA_VISIBLE_DEVICES=${TRAINING_GPUS} accelerate launch "${TRAINING_SCRIPT_PATH}" 
    TRAINING_EXIT_CODE=$?
}

# --- Main script execution ---
start_vllm
start_training
# trap EXIT will call cleanup