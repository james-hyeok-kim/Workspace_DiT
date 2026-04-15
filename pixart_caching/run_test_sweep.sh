#!/usr/bin/env bash
# run_test_sweep.sh
# DeepCache 고정 (interval=2, blocks[8,20)) + guidance/scheduler/full_steps sweep
# 모든 config를 2 samples(test_run 수준)으로 빠르게 탐색

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[ -f ~/.env ] && source ~/.env

MODEL_PATH="PixArt-alpha/PixArt-XL-2-1024-MS"
DATASET="MJHQ"
LOWRANK=32
NUM_STEPS=15   # 조합1 최적 step

REF_DIR="$SCRIPT_DIR/ref_images"
SAVE_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/test_sweep_${TIMESTAMP}.log"

echo "=== test_sweep: 10 configs × 2 samples | steps=${NUM_STEPS} ==="
echo "Log: $LOG_FILE"

accelerate launch --num_processes 1 \
    "$SCRIPT_DIR/pixart_deepcache_experiment.py" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET" \
    --lowrank "$LOWRANK" \
    --num_inference_steps "$NUM_STEPS" \
    --ref_dir "$REF_DIR" \
    --save_dir "$SAVE_DIR" \
    --test_sweep \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Results: $SAVE_DIR/$DATASET/deepcache/test_sweep/test_sweep_summary.csv"
