#!/bin/bash
# run_cache_lora.sh
# Cache-LoRA Corrector 실험 런처
#
# 사용법:
#   bash run_cache_lora.sh                  # 전체 실험
#   bash run_cache_lora.sh --test_run       # smoke test

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/pixart_nvfp4_cache_compare.py"
ENV_PYTHON="/home/jameskimh/.dit/bin/python"

NUM_SAMPLES=20
TEST_RUN=false
TEST_FLAG=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --test_run) TEST_RUN=true; TEST_FLAG="--test_run" ;;
        --num_samples) NUM_SAMPLES="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

if $TEST_RUN; then
    echo "=== SMOKE TEST MODE (2 samples) ==="
else
    echo "=== FULL CACHE-LORA EXPERIMENT ($NUM_SAMPLES samples) ==="
fi

cd "$SCRIPT_DIR"

# Experiment matrix
#   rank=4, interval=2, steps=20
#   rank=8, interval=2, steps=20
#   rank=8, interval=3, steps=20  (더 aggressive cache)

declare -a CONFIGS=(
    "4 2 20"
    "8 2 20"
)
# interval=3 은 별도 처리 (cache_interval 파라미터 추가 필요하면)

for cfg in "${CONFIGS[@]}"; do
    read -r RANK INTERVAL STEPS <<< "$cfg"
    TAG="SVDQUANT_cache_lora_r${RANK}_steps${STEPS}"
    RESULT_DIR="/data/jameskimh/james_dit_pixart_xl_mjhq/SVDQUANT/cache_lora_r${RANK}_steps${STEPS}"

    if [ -f "$RESULT_DIR/metrics.json" ] && ! $TEST_RUN; then
        echo "⏭️  SKIP (already exists): $TAG"
        continue
    fi

    echo ""
    echo "▶️  Running: $TAG  (rank=$RANK, interval=$INTERVAL)"
    echo "-----------------------------------------------------------"

    $ENV_PYTHON "$PYTHON_SCRIPT" \
        --quant_method SVDQUANT \
        --cache_mode   cache_lora \
        --num_steps    "$STEPS" \
        --num_samples  "$NUM_SAMPLES" \
        --lora_rank    "$RANK" \
        --lora_calib   4 \
        --guidance_scale 4.5 \
        $TEST_FLAG

    echo "✅ Done: $TAG"
    echo ""
done

echo "============================================================"
echo "All cache_lora runs complete."
echo "============================================================"
