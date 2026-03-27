#!/bin/bash

# 기본 설정
BASE_DIR="/home/jameskimh/gpu_project/workspace/dit/Workspace_DiT/pixart_alpha"
ACT_MODE="NVFP4"
WGT_MODE="TERNARY"
ALPHA=0.8
LOWRANK=32
BLOCK_SIZE=16
NUMERIC_DTYPE="half"

# 로그 파일 경로 조립 (가독성 버전)
LOG_NAME="pixart_b200"
LOG_NAME="${LOG_NAME}_A${ACT_MODE}"
LOG_NAME="${LOG_NAME}_W${WGT_MODE}"
LOG_NAME="${LOG_NAME}_a${ALPHA}"
LOG_NAME="${LOG_NAME}_R${LOWRANK}"
LOG_NAME="${LOG_NAME}_B${BLOCK_SIZE}"
LOG_NAME="${LOG_NAME}_D${NUMERIC_DTYPE}"

LOG_FILE="${BASE_DIR}/logs/${LOG_NAME}.log"

mkdir -p "${BASE_DIR}/logs"

echo "🚀 Starting Experiment: ${LOG_NAME}"

python ${BASE_DIR}/pixart_alpha_quant_b200.py \
    --act_mode ${ACT_MODE} \
    --wgt_mode ${WGT_MODE} \
    --alpha ${ALPHA} \
    --lowrank ${LOWRANK} \
    --block_size ${BLOCK_SIZE} \
    --numeric_dtype ${NUMERIC_DTYPE} \
    --save_dir "${BASE_DIR}/results" \
    2>&1 | tee ${LOG_FILE}