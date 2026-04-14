#!/usr/bin/env bash
# run_step_sweep.sh
# NVFP4_DEFAULT_CFG inference step sweep
# Usage: bash run_step_sweep.sh [--test_run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load HF token if available
[ -f ~/.env ] && source ~/.env

# ---- Config ----------------------------------------------------------------
MODEL_PATH="PixArt-alpha/PixArt-XL-2-1024-MS"
DATASET="MJHQ"
NUM_SAMPLES=20
LOWRANK=32
STEP_COUNTS="5,8,10,12,15,20"

REF_DIR="$SCRIPT_DIR/ref_images"
SAVE_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Parse --test_run flag
EXTRA_ARGS=""
for arg in "$@"; do
    if [ "$arg" = "--test_run" ]; then
        EXTRA_ARGS="--test_run"
        NUM_SAMPLES=2
    fi
done

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/step_sweep_${TIMESTAMP}.log"

echo "=== Step Sweep: $STEP_COUNTS steps | $NUM_SAMPLES samples ==="
echo "Log: $LOG_FILE"

accelerate launch --num_processes 1 \
    "$SCRIPT_DIR/pixart_step_sweep.py" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET" \
    --num_samples "$NUM_SAMPLES" \
    --lowrank "$LOWRANK" \
    --step_counts "$STEP_COUNTS" \
    --ref_dir "$REF_DIR" \
    --save_dir "$SAVE_DIR" \
    $EXTRA_ARGS \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Results saved to: $SAVE_DIR/$DATASET/step_sweep/summary.json"
