#!/bin/bash

# 1. GPU 및 환경 설정
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
# HF_TOKEN
if [ -f ~/.env ]; then
  export $(grep -v '^#' ~/.env | xargs)
fi
export HF_HUB_ENABLE_HF_TRANSFER=1

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
    DATA_SET="MJHQ"
    FLAGS="--test_run"
else
    # 본 실험 시 원하는 샘플 수를 지정하세요 (예: 100, 1000 등)
    NUM_SAMPLES=100
    MODE_LABEL="prod"
    DATA_SET="MJHQ"
    FLAGS=""
fi

# 공통 고정 파라미터
ALPHA=0.5
LOWRANK=32
BLOCK_SIZE=16
NUMERIC="half"

# 🎯 5. 양자화 모드 조합 배열 정의 (원하는 조합만 남기고 지워도 됩니다)
#ACT_MODES=("NVFP4" "INT8" "INT4" "INT3" "INT2" "TERNARY")
#WGT_MODES=("NVFP4" "INT8" "INT4" "INT3" "INT2" "TERNARY")
ACT_MODES=("INT8")
WGT_MODES=("INT8")

mkdir -p "${BASE_DIR}/logs"

echo "----------------------------------------------------"
echo "🚀 Starting B200 Experiment Sweep: ${MODE_LABEL} mode"
echo "📍 Samples per config: ${NUM_SAMPLES} (Min 2 for FID if test)"
echo "----------------------------------------------------"

# 6. 모든 조합에 대한 이중 루프 실행
for act in "${ACT_MODES[@]}"; do
    for wgt in "${WGT_MODES[@]}"; do
        
        echo "===================================================="
        echo "⚡ Running Combination -> Act: ${act} | Wgt: ${wgt}"
        echo "===================================================="

        # 결과물이 섞이지 않도록 각 조합별로 별도의 폴더 생성
        CURRENT_RESULT_DIR="${BASE_DIR}/results/${MODE_LABEL}/A${act}_W${wgt}"
        mkdir -p "${CURRENT_RESULT_DIR}"

        # 로그 파일명 동적 생성
        LOG_FILE="${BASE_DIR}/logs/run_${MODE_LABEL}_A${act}_W${wgt}_a${ALPHA}_L${LOWRANK}_B${BLOCK_SIZE}_N${NUMERIC}.log"

        # 7. 실행
        accelerate launch --multi_gpu --num_processes 2 \
            "${BASE_DIR}/pixart_alpha_quant_b200.py" \
            --model_path "PixArt-alpha/PixArt-XL-2-1024-MS" \
            --ref_dir "${REF_DIR}" \
            --save_dir "${CURRENT_RESULT_DIR}" \
            --num_samples ${NUM_SAMPLES} \
            --act_mode "${act}" \
            --wgt_mode "${wgt}" \
            --alpha ${ALPHA} \
            --lowrank ${LOWRANK} \
            --block_size ${BLOCK_SIZE} \
            --numeric_dtype "${NUMERIC}" \
            --dataset_name "${DATA_SET}" \
            ${FLAGS} \
            2>&1 | tee "${LOG_FILE}"

        echo "✅ Finished Act=${act}, Wgt=${wgt}"
        echo "📄 Log saved to: ${LOG_FILE}"
        echo ""

    done
done

echo "🎉 All combination sweeps completed successfully!"