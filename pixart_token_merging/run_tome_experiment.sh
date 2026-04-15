#!/usr/bin/env bash
# run_tome_experiment.sh
# Phase 1: Token Merging (FP16) sweep 실험
#
# 사용법:
#   bash run_tome_experiment.sh            # 전체 sweep
#   bash run_tome_experiment.sh --test_run # 2-sample smoke test
#   bash run_tome_experiment.sh --single   # 단일 config (--merge_ratio, --block_start, --block_end)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 환경 변수 ────────────────────────────────────────────────────────────────
if [ -f "$HOME/.env" ]; then
    set -a
    source "$HOME/.env"
    set +a
fi

# .dit 가상환경 우선 사용
export PATH="$HOME/.dit/bin:$PATH"

# ── 설정 ─────────────────────────────────────────────────────────────────────
MODEL_PATH="PixArt-alpha/PixArt-XL-2-1024-MS"
DATASET="MJHQ"
NUM_SAMPLES=20
NUM_STEPS=20
REF_DIR="$SCRIPT_DIR/ref_images"
SAVE_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# 단일 config 기본값
MERGE_RATIO=0.20
BLOCK_START=0
BLOCK_END=28

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
TEST_RUN_FLAG=""
SWEEP_FLAG="--sweep"   # 기본: sweep 모드
SINGLE_MODE=0

for arg in "$@"; do
    case "$arg" in
        --test_run) TEST_RUN_FLAG="--test_run"; NUM_SAMPLES=2 ;;
        --single)   SWEEP_FLAG=""; SINGLE_MODE=1 ;;
        --merge_ratio=*) MERGE_RATIO="${arg#*=}" ;;
        --block_start=*) BLOCK_START="${arg#*=}" ;;
        --block_end=*)   BLOCK_END="${arg#*=}" ;;
    esac
done

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/tome_fp16_${TIMESTAMP}.log"

echo "================================================================"
echo "  PixArt ToMe (FP16) Sweep"
echo "  Model  : $MODEL_PATH"
echo "  Dataset: $DATASET  Samples: $NUM_SAMPLES  Steps: $NUM_STEPS"
echo "  LogFile: $LOG_FILE"
echo "================================================================"

accelerate launch \
    --num_processes 1 \
    --mixed_precision no \
    "$SCRIPT_DIR/pixart_tome_experiment.py" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET" \
    --num_samples "$NUM_SAMPLES" \
    --num_inference_steps "$NUM_STEPS" \
    --ref_dir "$REF_DIR" \
    --save_dir "$SAVE_DIR" \
    --merge_ratio "$MERGE_RATIO" \
    --block_start "$BLOCK_START" \
    --block_end   "$BLOCK_END" \
    $SWEEP_FLAG \
    $TEST_RUN_FLAG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Done. Results: $SAVE_DIR/$DATASET/tome/"
echo "Log  : $LOG_FILE"
