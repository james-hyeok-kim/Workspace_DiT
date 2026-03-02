#!/bin/bash
# 중간 Size Model
# 사용법: ./run_dit_b_8_inference.sh [학습된_모델_경로]
MODEL_SIZE="DiT-B/8"
BASE_DIR="/app/workspace/DiT/Workspace_DiT/diffusion_from_scratch"
MODEL_PATH="${BASE_DIR}/results/CIFAR_DiT-B-8_20260227_1136/model/model_epoch_100.pt"
OUTPUT_DIR="${BASE_DIR}/output_dit_b_8/CIFAR_DiT-B-8_20260227_1136/model/model_epoch_100/"
LOG_FILE="$(pwd)/inference_dit_b_8.log"

# 모델 경로가 입력되지 않았을 경우 에러 메시지 출력
if [ -z "$MODEL_PATH" ]; then
    echo "Error: 학습된 모델(.pt) 파일 경로를 입력해주세요!"
    echo "Usage: ./run_dit_b_8_inference.sh ./results/CIFAR_DiT-B-8_.../model/model_epoch_xxx.pt"
    exit 1
fi

# 1. 환경 변수 설정 (출력을 실시간으로 확인)
export PYTHONUNBUFFERED=1

# 2. DiT 추론 실행
# --model_size를 DiT-B/8로 지정합니다.
# CUDA_VISIBLE_DEVICES=0 지정 및 2>&1 | tee 로 로그 저장
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --model_arch dit \
  --model_size "${MODEL_SIZE}" \
  --model_load_dir "${MODEL_PATH}" \
  --dataset_name CIFAR-10 \
  --image_size 32 \
  --output_dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  2>&1 | tee "${LOG_FILE}"