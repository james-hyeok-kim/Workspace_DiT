#!/bin/bash
# run_cache_lora_sweep.sh
# 4가지 NVFP4 양자화 방법 × Cache-LoRA rank={2,4,8} sweep
#
# 사용법:
#   bash run_cache_lora_sweep.sh              # 전체 실험 (20 samples)
#   bash run_cache_lora_sweep.sh --test_run   # smoke test (2 samples)
#   bash run_cache_lora_sweep.sh --method RTN # 특정 method만

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/pixart_nvfp4_cache_compare.py"
ENV_PYTHON="/home/jameskimh/.dit/bin/python"

NUM_SAMPLES=20
TEST_RUN=false
TEST_FLAG=""
METHODS=("RTN" "SVDQUANT" "MRGPTQ" "FOUROVERSIX")
RANKS=(2 4 8)
STEPS=20
LORA_CALIB=4

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
    echo "=== FULL CACHE-LORA SWEEP ($NUM_SAMPLES samples) ==="
fi

echo "Methods : ${METHODS[*]}"
echo "Ranks   : ${RANKS[*]}"
echo "Steps   : $STEPS"
echo ""

cd "$SCRIPT_DIR"

for METHOD in "${METHODS[@]}"; do
    for RANK in "${RANKS[@]}"; do
        TAG="${METHOD}_cache_lora_r${RANK}_steps${STEPS}"
        RESULT_DIR="$SCRIPT_DIR/results/MJHQ/${TAG}"

        if [ -f "$RESULT_DIR/metrics.json" ] && ! $TEST_RUN; then
            echo "⏭️  SKIP (already exists): $TAG"
            continue
        fi

        echo ""
        echo "▶️  Running: $TAG"
        echo "   Method=$METHOD  Rank=$RANK  Steps=$STEPS"
        echo "-----------------------------------------------------------"

        $ENV_PYTHON "$PYTHON_SCRIPT" \
            --quant_method "$METHOD" \
            --cache_mode   cache_lora \
            --num_steps    "$STEPS" \
            --num_samples  "$NUM_SAMPLES" \
            --lora_rank    "$RANK" \
            --lora_calib   "$LORA_CALIB" \
            --guidance_scale 4.5 \
            $TEST_FLAG

        echo "✅ Done: $TAG"
        echo ""
    done
done

echo "============================================================"
echo "All cache_lora sweep runs complete."
echo "Results in: $SCRIPT_DIR/results/MJHQ/"
echo "============================================================"
