#!/bin/bash
# Large Model
# 사용법: ./run_dit_l_inference.sh DiT-L/8 [학습된_모델_경로]
MODEL_SIZE="DiT-L/8"
BASE_DIR="/app/workspace/DiT/Workspace_DiT/diffusion_from_scratch"
MODEL_PATH="${BASE_DIR}/results/CIFAR_DiT-L-8_20260227_1805/model/model_epoch_100.pt" 
OUTPUT_DIR="${BASE_DIR}/output_dit_l_8/CIFAR_DiT-L-8_20260227_1805/model/model_epoch_100/"
LOG_FILE="$(pwd)/inference_dit_l_8.log"

if [ -z "$MODEL_PATH" ]; then
    echo "Error: .pt 파일 경로를 입력해주세요!"
    echo "Example: ./run_dit_l_inference.sh DiT-L/8 ./results/CIFAR_DiT-L-8_.../model.pt"
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