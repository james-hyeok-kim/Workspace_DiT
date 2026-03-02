#!/bin/bash
# 중간크기 Base Model

export PYTHONUNBUFFERED=1
MODEL_SIZE="DiT-B/8"
DATA_DIR="/data/jameskimh/DiT"
RESULT_DIR="${DATA_DIR}/results/CIFAR_DiT-B-8_$(date +%Y%m%d_%H%M)"
LOG_FILE="$(pwd)/train_dit_b_8.log"

mkdir -p ${RESULT_DIR}

CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_arch dit \
  --model_size "${MODEL_SIZE}" \
  --dataset_name CIFAR-10 \
  --image_size 32 \
  --num_workers 8 \
  --batch_size 256 \
  --max_epoch 2000 \
  --result_dir "${RESULT_DIR}" \
  --device cuda:0 \
  2>&1 | tee "${LOG_FILE}"