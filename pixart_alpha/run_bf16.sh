#!/bin/bash
BASE_DIR="$(pwd)/"
QUANT_METHOD="BF16"
LOG_FILE="${BASE_DIR}/pixart_alpha_${QUANT_METHOD}.log"
OUTPUT_DIR="${BASE_DIR}"/results/
# 1. 환경 변수 설정 (출력을 실시간으로 확인)

mkdir -p ${OUTPUT_DIR}

export PYTHONUNBUFFERED=1

# 2. 평가 스크립트 실행
CUDA_VISIBLE_DEVICES=0 python pixart_alpha_quant.py \
  --model_id "PixArt-alpha/PixArt-XL-2-1024-MS" \
  --quant_method "${QUANT_METHOD}" \
  --save_dir "${OUTPUT_DIR}" \
  --num_samples 100 \
  2>&1 | tee "${LOG_FILE}"