#!/bin/bash
# run_cache_lora_final.sh
# RTN / MRGPTQ / FOUROVERSIX × cache_lora rank=4 실험 (재실행)
#
# 수정 사항:
#   - calib_seed_offset=1000 (eval seeds 42+i와 겹치지 않도록)
#   - calibration time, inference time 모두 metrics.json/csv에 저장
#
# 사용법:
#   bash run_cache_lora_final.sh              # 전체 실험 (20 samples)
#   bash run_cache_lora_final.sh --test_run   # smoke test (2 samples)
#   bash run_cache_lora_final.sh --method RTN # 특정 method만

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/pixart_nvfp4_cache_compare.py"
ENV_PYTHON="/home/jameskimh/.dit/bin/python"

NUM_SAMPLES=20
TEST_RUN=false
TEST_FLAG=""
METHODS=("RTN" "MRGPTQ" "FOUROVERSIX")
RANK=4
STEPS=20
LORA_CALIB=4
CALIB_SEED_OFFSET=1000

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --test_run)    TEST_RUN=true; TEST_FLAG="--test_run" ;;
        --method)      METHODS=("$2"); shift ;;
        --num_samples) NUM_SAMPLES="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

if $TEST_RUN; then
    echo "=== SMOKE TEST MODE (2 samples) ==="
else
    echo "=== CACHE-LORA FINAL SWEEP ($NUM_SAMPLES samples) ==="
fi

echo "Methods          : ${METHODS[*]}"
echo "Rank             : $RANK"
echo "Steps            : $STEPS"
echo "Calib seed offset: $CALIB_SEED_OFFSET"
echo ""

cd "$SCRIPT_DIR"

for METHOD in "${METHODS[@]}"; do
    TAG="${METHOD}_cache_lora_r${RANK}_steps${STEPS}"
    RESULT_DIR="$SCRIPT_DIR/results/MJHQ/${TAG}"

    # 이미 완료된 결과면 skip (smoke test 모드에서는 항상 재실행)
    if [ -f "$RESULT_DIR/metrics.json" ] && ! $TEST_RUN; then
        echo "⏭️  SKIP (already exists): $TAG"
        continue
    fi

    echo ""
    echo "▶️  Running: $TAG"
    echo "   Method=$METHOD  Rank=$RANK  Steps=$STEPS  CalibSeedOffset=$CALIB_SEED_OFFSET"
    echo "-----------------------------------------------------------"

    $ENV_PYTHON "$PYTHON_SCRIPT" \
        --quant_method     "$METHOD" \
        --cache_mode       cache_lora \
        --num_steps        "$STEPS" \
        --num_samples      "$NUM_SAMPLES" \
        --lora_rank        "$RANK" \
        --lora_calib       "$LORA_CALIB" \
        --calib_seed_offset "$CALIB_SEED_OFFSET" \
        --guidance_scale   4.5 \
        $TEST_FLAG

    echo "✅ Done: $TAG"
    echo ""
done

echo "============================================================"
echo "Cache-LoRA final sweep complete."
echo "Results in: $SCRIPT_DIR/results/MJHQ/"
echo "============================================================"
