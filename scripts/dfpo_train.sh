#!/bin/bash

TRAINING_NPUS="0,1,2,3,4,5,6,7"
TRAINING_SCRIPT_PATH="dfpo_train.py"
TRAINING_EXIT_CODE=0


start_training() {
    export HCCL_CONNECT_TIMEOUT=6000
    export MEMORY_FRAGMENTATION=1
    export environment variable ASCEND_LAUNCH_BLOCKING=1
    ASCEND_VISIBLE_DEVICES=${TRAINING_NPUS} accelerate launch --config_file configs/default_config.yaml "${TRAINING_SCRIPT_PATH}"
    TRAINING_EXIT_CODE=$?
}

# --- Main script execution ---
export PYTHONPATH="/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe"
#start_vllm
start_training
