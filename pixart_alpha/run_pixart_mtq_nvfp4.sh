#!/bin/bash

# 1. 환경 변수 설정 (성능 및 안정성)
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_HTTP_TIMEOUT=3600
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1

# 2. 실험 설정 변수
DATASET="MJHQ"         # MJHQ 또는 SONGHAN
NUM_SAMPLES=1        # 실제 테스트 시 100 이상 권장
MODEL_PATH="PixArt-alpha/PixArt-XL-2-1024-MS"

# 🎯 실행할 모드 선택 (manual 또는 mtq)
# 사용법: sh run_quant.sh manual  또는  sh run_quant.sh mtq
MODE="mtq"
BASE_DIR="$(pwd)"
LOG_FILE="${BASE_DIR}/logs/run_mtq_nvfp4_default.log"

# 3. Accelerate 실행
accelerate launch --num_processes 2 pixart_mtq_NVFP4_default.py \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET" \
    --num_samples $NUM_SAMPLES \
    --ref_dir "./ref_images" \
    --save_dir "./results" \
    --alpha 0.5 \
    --lowrank 32 \
    --test_run \
    2>&1 | tee "${LOG_FILE}"

echo "✅ $MODE evaluation finished for $DATASET dataset!"