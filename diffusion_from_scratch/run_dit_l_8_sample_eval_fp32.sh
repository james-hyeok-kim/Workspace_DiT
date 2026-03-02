#!/bin/bash
# DiT-S/8 모델 샘플링 및 평가 (FID, IS)
# 사용법: ./run_sample_and_eval.sh [학습된_모델_경로] (생략 시 기본 경로 사용)

MODEL_SIZE="DiT-L/8"
DATA_DIR="/data/jameskimh/DiT/results"
BASE_DIR="/app/workspace/DiT/Workspace_DiT/diffusion_from_scratch"

QUANT_METHOD="FP32"

MODEL_PATH="${DATA_DIR}/CIFAR_DiT-L-8_20260228_1954/model/model_epoch_1900.pt"
OUTPUT_DIR="${BASE_DIR}/output_eval_dit_l_8_${QUANT_METHOD}/CIFAR_DiT-L-8_20260228_1954/model/model_epoch_1900/"
LOG_FILE="$(pwd)/eval_dit_s_8_${QUANT_METHOD}.log"
# 평가 설정값 (필요에 따라 조절하세요)
NUM_SAMPLES=1000  # 논문 수준의 평가가 필요하다면 50000으로 변경
BATCH_SIZE=128

# 모델 경로가 유효한지 파일 존재 여부 확인
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: 학습된 모델(.pt) 파일을 찾을 수 없습니다: $MODEL_PATH"
    echo "Usage: ./run_sample_and_eval.sh /path/to/your/model_epoch_xxx.pt"
    exit 1
fi

echo "==================================================="
echo "Starting Evaluation for Model: $MODEL_SIZE"
echo "Model Path: $MODEL_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "==================================================="

# 1. 환경 변수 설정 (출력을 실시간으로 확인)
export PYTHONUNBUFFERED=1

# 2. 평가 스크립트 실행
CUDA_VISIBLE_DEVICES=0 python sample_and_eval.py \
  --model_size "${MODEL_SIZE}" \
  --model_path "${MODEL_PATH}" \
  --dataset_name "CIFAR-10" \
  --dataset_path "./data" \
  --image_size 32 \
  --batch_size ${BATCH_SIZE} \
  --num_samples ${NUM_SAMPLES} \
  --save_dir "${OUTPUT_DIR}" \
  --device "cuda:0" \
  --quant_method "${QUANT_METHOD}" \
  2>&1 | tee "${LOG_FILE}"