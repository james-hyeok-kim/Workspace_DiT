#!/bin/bash
# 가장 빠른 Small Model
export PYTHONUNBUFFERED=1
MODEL_SIZE="DiT-S/8"
BASE_DIR="/app/workspace/DiT/Workspace_DiT/diffusion_from_scratch"
RESULT_DIR="${BASE_DIR}/results/CIFAR_DiT-S-8_$(date +%Y%m%d_%H%M)"
LOG_FILE="$(pwd)/train_dit_s_8.log"

mkdir -p ${RESULT_DIR}

CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_arch dit \
  --model_size "${MODEL_SIZE}" \
  --dataset_name CIFAR-10 \
  --image_size 32 \
  --num_workers 8 \
  --batch_size 512 \
  --max_epoch 2000 \
  --result_dir "${RESULT_DIR}" \
  --device cuda:0 \
  2>&1 | tee "${LOG_FILE}"