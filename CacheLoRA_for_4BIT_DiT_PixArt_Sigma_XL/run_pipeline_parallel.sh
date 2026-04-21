#!/bin/bash
# run_pipeline_parallel.sh — 2-GPU parallel execution
#
# Dir 5/9 (병렬):
#   GPU 0 (port 29500): Dir 5 (block fixed, 3 runs) + Dir 9 (teacher-forced, 3 runs)
#   GPU 1 (port 29501): Dir 6 (nested caching, 24 runs)
#
# Usage:
#   bash run_pipeline_parallel.sh
#   bash run_pipeline_parallel.sh --test_run
#   bash run_pipeline_parallel.sh --num_samples 20

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/pixart_nvfp4_cache_compare.py"
ENV_PYTHON="/home/jameskimh/.dit/bin/accelerate"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TEST_RUN=false
TEST_FLAG=""
NUM_SAMPLES=20

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --test_run)    TEST_RUN=true; TEST_FLAG="--test_run"; NUM_SAMPLES=2 ;;
        --num_samples) NUM_SAMPLES="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# ── Helper: run one calib×rank config ────────────────────────────────────────
run_svdq_calib_rank() {
    local CALIB="$1" RANK="$2" STEP="$3" PORT="$4"
    local CALIB_TAG=""
    if [ "$CALIB" != "4" ]; then CALIB_TAG="_cal${CALIB}"; fi
    local TAG="SVDQUANT_cl_r${RANK}${CALIB_TAG}_c8-20_steps${STEP}"
    local RESULT_DIR="/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/SVDQUANT/MJHQ/${TAG}"
    if [ -f "$RESULT_DIR/metrics.json" ] && [ "$TEST_RUN" = false ]; then
        echo "  SKIP: $TAG"; return
    fi
    echo "  Running: $TAG"
    $ENV_PYTHON launch --num_processes 1 --main_process_port "$PORT" \
        "$PYTHON_SCRIPT" \
        --quant_method SVDQUANT --cache_mode cache_lora \
        --num_steps "$STEP" --num_samples "$NUM_SAMPLES" \
        --guidance_scale 4.5 --cache_start 8 --cache_end 20 \
        --deepcache_interval 2 \
        --lora_rank "$RANK" --lora_calib "$CALIB" \
        $TEST_FLAG
    echo "  Done:    $TAG"
}

echo "========================================================"
echo "  Parallel pipeline Dir5/6/9  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "  Samples: $NUM_SAMPLES  test_run=$TEST_RUN"
echo "========================================================"

# ════════════════════════════════════════════════════════════
# PHASE A (병렬):
#   GPU 0 (port 29500): Dir 5 (block fixed) + Dir 9 (teacher-forced)
#   GPU 1 (port 29501): Dir 6 (nested caching, 24 runs)
# ════════════════════════════════════════════════════════════
echo ""
echo "── PHASE A ─────────────────────────────────────────────"
echo "  GPU 0 (port 29500): Dir 5 (block) + Dir 9 (tf)"
echo "  GPU 1 (port 29501): Dir 6 (nested)"
echo "────────────────────────────────────────────────────────"

phaseA_gpu0() {
    export CUDA_VISIBLE_DEVICES=0
    echo "[GPU0]  Dir 5: cache_lora_block (fixed)"
    bash "$SCRIPT_DIR/run_svdquant_advanced.sh" \
        --axis block --gpu 0 --port 29500 \
        --num_samples "$NUM_SAMPLES" $TEST_FLAG
    echo "[GPU0]  Dir 5 done"

    echo "[GPU0]  Dir 9: cache_lora_tf"
    bash "$SCRIPT_DIR/run_svdquant_advanced.sh" \
        --axis tf --gpu 0 --port 29500 \
        --num_samples "$NUM_SAMPLES" $TEST_FLAG
    echo "[GPU0]  Dir 9 done"
}

phaseA_gpu1() {
    export CUDA_VISIBLE_DEVICES=1
    echo "[GPU1]  Dir 6: nested caching sweep"
    bash "$SCRIPT_DIR/run_svdquant_advanced.sh" \
        --axis nested --gpu 1 --port 29501 \
        --num_samples "$NUM_SAMPLES" $TEST_FLAG
    echo "[GPU1]  Dir 6 done"
}

phaseA_gpu0 >> "$LOG_DIR/gpu0_phaseA.log" 2>&1 &
PID0=$!
phaseA_gpu1 >> "$LOG_DIR/gpu1_phaseA.log" 2>&1 &
PID1=$!

echo "  GPU 0 PID=$PID0  → $LOG_DIR/gpu0_phaseA.log"
echo "  GPU 1 PID=$PID1  → $LOG_DIR/gpu1_phaseA.log"
echo ""
echo "Waiting for Phase A..."

EXIT0=0; EXIT1=0
wait $PID0 || EXIT0=$?
wait $PID1 || EXIT1=$?

echo ""
echo "════════════════════════════════════════════════════════"
if [ $EXIT0 -eq 0 ] && [ $EXIT1 -eq 0 ]; then
    echo "  ALL DONE  ✓  ($(date '+%Y-%m-%d %H:%M:%S'))"
else
    echo "  DONE WITH ERRORS: GPU0=$EXIT0 GPU1=$EXIT1"
    [ $EXIT0 -ne 0 ] && echo "  → $LOG_DIR/gpu0_phaseA.log"
    [ $EXIT1 -ne 0 ] && echo "  → $LOG_DIR/gpu1_phaseA.log"
    exit 1
fi
echo "════════════════════════════════════════════════════════"
