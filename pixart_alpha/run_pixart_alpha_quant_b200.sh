#!/bin/bash

# 1. GPU 및 환경 설정
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export HF_TOKEN=""

# 2. 경로 설정
BASE_DIR="$(pwd)"
REF_DIR="/data/james_dit_ref/ref_images_fp16"

# 3. 테스트 모드 스위치 (1: 테스트, 0: 본 실험)
TEST_MODE=1

# 4. 모드별 파라미터 자동 분기
if [ "$TEST_MODE" -eq 1 ]; then
    # 테스트 시에도 FID를 위해 파이썬 내부에서 최소 2개로 작동하지만, 명시적으로 1을 줍니다.
    NUM_SAMPLES=1 
    MODE_LABEL="test"
    FLAGS="--test_run"
else
    # 본 실험 시 원하는 샘플 수를 지정하세요 (예: 100, 1000 등)
    NUM_SAMPLES=100
    MODE_LABEL="prod"
    FLAGS=""
fi

ACT_MODE="NVFP4"
WGT_MODE="NVFP4"
ALPHA=0.8
LOWRANK=64
BLOCK_SIZE=16
NUMERIC="half"

# 5. 결과 저장 및 로그 경로
RESULT_DIR="${BASE_DIR}/results/${MODE_LABEL}"
LOG_FILE="${BASE_DIR}/logs/run_${MODE_LABEL}"
LOG_FILE="${LOG_FILE}_A${ACT_MODE}"
LOG_FILE="${LOG_FILE}_W${WGT_MODE}"
LOG_FILE="${LOG_FILE}_a${ALPHA}"
LOG_FILE="${LOG_FILE}_L${LOWRANK}"
LOG_FILE="${LOG_FILE}_B${BLOCK_SIZE}"
LOG_FILE="${LOG_FILE}_N${NUMERIC}.log"

mkdir -p "${BASE_DIR}/logs"
mkdir -p "${RESULT_DIR}"

echo "----------------------------------------------------"
echo "🚀 Starting B200 Experiment: ${MODE_LABEL} mode"
echo "📍 Samples to generate: ${NUM_SAMPLES} (Min 2 for FID)"
echo "📍 Results will be saved in: ${RESULT_DIR}"
echo "----------------------------------------------------"

# 6. 실행 (변경된 --num_samples 인자 포함)
accelerate launch --multi_gpu --num_processes 2 \
    "${BASE_DIR}/pixart_alpha_quant_b200.py" \
    --model_path "PixArt-alpha/PixArt-XL-2-1024-MS" \
    --ref_dir "${REF_DIR}" \
    --save_dir "${RESULT_DIR}" \
    --num_samples ${NUM_SAMPLES} \
    --act_mode "${ACT_MODE}" \
    --wgt_mode "${WGT_MODE}" \
    --alpha ${ALPHA} \
    --lowrank ${LOWRANK} \
    --block_size ${BLOCK_SIZE} \
    --numeric_dtype "${NUMERIC}" \
    ${FLAGS} \
    2>&1 | tee "${LOG_FILE}"

echo "----------------------------------------------------"
echo "✅ Done! Log: ${LOG_FILE}"