#!/bin/bash

VLLM_EVAL_MODEL_PATH=""
VLLM_EVAL_SERVED_MODEL_NAME=""
VLLM_EVAL_PORT=8000
VLLM_EVAL_NPUS="0"
VLLM_TP_SIZE=1
VLLM_GPU_MEM_UTIL=0.8
VLLM_EVAL_LOG_FILE=""


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


start_eval_vllm() {
    export PYTHONPATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe"

    cd vllm

    local VLLM_CMD_ARGS=(
        --served-model-name "${VLLM_EVAL_SERVED_MODEL_NAME}"
        --model "${VLLM_EVAL_MODEL_PATH}"
        --host "0.0.0.0"
        --port "${VLLM_EVAL_PORT}"
        --dtype float16
        --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}"
        --tensor-parallel-size "${VLLM_TP_SIZE}"
    )


    echo "Starting vLLM server with command:"
    # shellcheck disable=SC2086
    ASCEND_VISIBLE_DEVICES=${VLLM_EVAL_NPUS} python -m vllm.entrypoints.openai.api_server "${VLLM_CMD_ARGS[@]}" > "${VLLM_EVAL_LOG_FILE}" 2>&1 &
    VLLM_PID=$!

    sleep 120

    echo "vLLM has been successfully started at port ${VLLM_EVAL_PORT}. Check logs at ${VLLM_EVAL_LOG_FILE} for details."

    cd ..

}

start_evaling() {
    export VLLM_EVAL_API_KEY="your_eval_api_name"
    export VLLM_EVAL_BASE_URL="http://localhost:${VLLM_EVAL_PORT}/v1"
    python scripts/hybrid_neural_symbolic_rag.py --dataset airqa --test_data test_data_553.jsonl --database ai_research --agent_method neusym_rag --llm your_eval_model
    TRAINING_EXIT_CODE=$?
}

# --- Main script execution ---
start_eval_vllm
start_evaling
