#!/bin/bash
BASE_DIR="$(pwd)/"
QUANT_METHOD="NVFP4_ALL"
SVD_DTYPE="fp8"
FP8_FORMAT="hybrid"
LOWRANK=64
BLOCK_SIZE=32

# ✅ 설정 스위치 (이것만 바꾸면 됩니다)
TEST_MODE=0        # 1: 빠른 테스트 (2샘플), 0: 본 실험 (1000샘플)
DO_DIFF_TUNING=1   # 1: Differential Tuning 켜기, 0: 끄기

# ✅ TEST_MODE에 따른 자동 설정
if [ "$TEST_MODE" -eq 1 ]; then
    NUM_SAMPLES=2
    MODE_LABEL="test"
    FLAGS="--test_run"
else
    NUM_SAMPLES=1000
    MODE_LABEL="prod"
    FLAGS=""
fi

# ✅ 추가 플래그 빌더
[ "$DO_DIFF_TUNING" -eq 1 ] && FLAGS="$FLAGS --do_diff_tuning"

LOG_FILE="${BASE_DIR}/pixart_${QUANT_METHOD}_R${LOWRANK}_S${SVD_DTYPE}_${FP8_FORMAT}_${MODE_LABEL}_T${FLAGS}_B${BLOCK_SIZE}.log"
OUTPUT_DIR="${BASE_DIR}/results/"

mkdir -p ${OUTPUT_DIR}

export PYTHONUNBUFFERED=1
export HF_TOKEN=""

# 평가 실행
CUDA_VISIBLE_DEVICES=0 python pixart_alpha_quant.py \
  --model_id "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS" \
  --quant_method "${QUANT_METHOD}" \
  --svd_dtype "${SVD_DTYPE}" \
  --fp8_format "${FP8_FORMAT}" \
  --block_size ${BLOCK_SIZE} \
  --save_dir "${OUTPUT_DIR}" \
  --num_samples ${NUM_SAMPLES} \
  --lowrank ${LOWRANK} \
  ${FLAGS} \
  2>&1 | tee "${LOG_FILE}"

이렇게 돌렸는데도 FID 변화가 없으면 다른 대안은 뭐가있어?
