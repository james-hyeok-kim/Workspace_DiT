#!/bin/bash
# Small Model
# 사용법: ./run_dit_s_8_inference.sh [학습된_모델_경로]
MODEL_SIZE="DiT-S/8"
BASE_DIR="/app/workspace/DiT/Workspace_DiT/diffusion_from_scratch"
MODEL_PATH="${BASE_DIR}/results/CIFAR_DiT-S-8_20260227_0925/model/model_epoch_1900.pt"
OUTPUT_DIR="${BASE_DIR}/output_dit_s_8/CIFAR_DiT-S-8_20260227_0925/model/model_epoch_1900/"
LOG_FILE="$(pwd)/inference_dit_s_8.log"

if [ -z "$MODEL_PATH" ]; then
    echo "Error: .pt 파일 경로를 입력해주세요!"
    echo "Usage: ./run_dit_s_8_inference.sh ./results/path_to_model/model.pt"
    exit 1
fi

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --model_arch dit \
  --model_size "${MODEL_SIZE}" \
  --model_load_dir "${MODEL_PATH}" \
  --dataset_name CIFAR-10 \
  --image_size 32 \
  --output_dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  2>&1 | tee "${LOG_FILE}"