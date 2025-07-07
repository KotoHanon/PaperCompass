#!/bin/bash

VLLM_EVAL_MODEL_PATH="/aistor/sjtu/hpc_stor01/home/wangzijian/workspace/NeuSym-RAG/.cache/Qwen2.5-3B-CKPT/checkpoint-300"
VLLM_EVAL_SERVED_MODEL_NAME="qwen2.5-3b-ckpt"
VLLM_SUP_MODEL_PATH="/aistor/sjtu/hpc_stor01/home/wangzijian/workspace/NeuSym-RAG/.cache/Qwen2.5-14B-Instruct"
VLLM_SUP_SERVED_MODEL_NAME="qwen2.5-14b-instruct"
VLLM_EVAL_API_KEY="lococo_eval"
VLLM_SUP_API_KEY="lococo_sup"
VLLM_EVAL_PORT=8000
VLLM_SUP_PORT=8001
VLLM_EVAL_GPUS="0,1"
VLLM_SUP_GPUS="2,3"
VLLM_TP_SIZE=2
VLLM_GPU_MEM_UTIL=0.9
VLLM_EVAL_LOG_FILE="./log/eval_vllm_server_1.log"
VLLM_SUP_LOG_FILE="./log/sup_vllm_server_1.log"


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



start_sup_vllm() {
    export PYTHONPATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe"

    cd vllm

    local VLLM_CMD_ARGS=(
        --served-model-name "${VLLM_SUP_SERVED_MODEL_NAME}"
        --model "${VLLM_SUP_MODEL_PATH}"
        --host "0.0.0.0"
        --port "${VLLM_SUP_PORT}"
        --dtype float16
        --gpu-memory-utilization "${VLLM_GPU_MEM_UTIL}"
        --tensor-parallel-size "${VLLM_TP_SIZE}"
    )

    echo "Starting vLLM server with command:"
    # shellcheck disable=SC2086
    ASCEND_VISIBLE_DEVICES=${VLLM_SUP_GPUS} python -m vllm.entrypoints.openai.api_server "${VLLM_CMD_ARGS[@]}" > "${VLLM_SUP_LOG_FILE}" 2>&1 &
    VLLM_PID=$!

    echo "vLLM has been successfully started at port ${VLLM_SUP_PORT}. Check logs at ${VLLM_SUP_LOG_FILE} for details."

    cd ..

}

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
    ASCEND_VISIBLE_DEVICES=${VLLM_EVAL_GPUS} python -m vllm.entrypoints.openai.api_server "${VLLM_CMD_ARGS[@]}" > "${VLLM_EVAL_LOG_FILE}" 2>&1 &
    VLLM_PID=$!

    sleep 120

    echo "vLLM has been successfully started at port ${VLLM_EVAL_PORT}. Check logs at ${VLLM_EVAL_LOG_FILE} for details."

    cd ..

}

start_evaling() {
    export VLLM_EVAL_API_KEY="lococo_eval"
    export VLLM_SUP_API_KEY="lococo_sup"
    export VLLM_EVAL_BASE_URL="http://localhost:${VLLM_EVAL_PORT}/v1"
    export VLLM_SUP_BASE_URL="http://localhost:${VLLM_SUP_PORT}/v1"
    export TOWHEE_HOME="./.towhee"
    export NLTK_DATA="./nltk_data"
    export TIKTOKEN_CACHE_DIR="."
    python scripts/hybrid_neural_symbolic_rag_g1.py --dataset airqa --test_data test_data_553.jsonl --database ai_research --agent_method neusym_rag --llm qwen2.5-3b-ckpt 
    TRAINING_EXIT_CODE=$?
}

# --- Main script execution ---
#export PYTHONPATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe"
start_sup_vllm
start_eval_vllm
start_evaling
# trap EXIT will call cleanup