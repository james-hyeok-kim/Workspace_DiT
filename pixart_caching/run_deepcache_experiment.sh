#!/usr/bin/env bash
# run_deepcache_experiment.sh
# NVFP4_DEFAULT_CFG + DeepCache block caching experiment
# Usage:
#   bash run_deepcache_experiment.sh             # single config (default)
#   bash run_deepcache_experiment.sh --sweep     # ablation sweep
#   bash run_deepcache_experiment.sh --test_run  # smoke test

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
NUM_STEPS=20

# Default single-run config
CACHE_INTERVAL=2
CACHE_START=4
CACHE_END=24
FULL_STEPS="0"

REF_DIR="$SCRIPT_DIR/ref_images"
SAVE_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Parse flags
EXTRA_ARGS=""
SWEEP_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --test_run)
            EXTRA_ARGS="--test_run"
            NUM_SAMPLES=2
            ;;
        --sweep)
            SWEEP_FLAG="--sweep"
            ;;
    esac
done

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
MODE="deepcache"
[ -n "$SWEEP_FLAG" ] && MODE="deepcache_sweep"
LOG_FILE="$LOG_DIR/${MODE}_${TIMESTAMP}.log"

echo "=== DeepCache: interval=${CACHE_INTERVAL}, blocks[${CACHE_START},${CACHE_END}) | steps=${NUM_STEPS} | samples=${NUM_SAMPLES} ==="
[ -n "$SWEEP_FLAG" ] && echo "=== SWEEP MODE: iterating over predefined configs ==="
echo "Log: $LOG_FILE"

accelerate launch --num_processes 1 \
    "$SCRIPT_DIR/pixart_deepcache_experiment.py" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET" \
    --num_samples "$NUM_SAMPLES" \
    --lowrank "$LOWRANK" \
    --num_inference_steps "$NUM_STEPS" \
    --cache_interval "$CACHE_INTERVAL" \
    --cache_start "$CACHE_START" \
    --cache_end "$CACHE_END" \
    --full_steps "$FULL_STEPS" \
    --ref_dir "$REF_DIR" \
    --save_dir "$SAVE_DIR" \
    $SWEEP_FLAG \
    $EXTRA_ARGS \
    2>&1 | tee "$LOG_FILE"

echo ""
if [ -n "$SWEEP_FLAG" ]; then
    echo "Sweep results: $SAVE_DIR/$DATASET/deepcache/sweep_summary.json"
else
    echo "Results: $SAVE_DIR/$DATASET/deepcache/interval${CACHE_INTERVAL}_s${CACHE_START}_e${CACHE_END}/metrics.json"
fi
