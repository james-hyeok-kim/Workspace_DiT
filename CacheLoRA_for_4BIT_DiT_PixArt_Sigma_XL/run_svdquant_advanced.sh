#!/bin/bash
# run_svdquant_advanced.sh
# Advanced Cache-LoRA directions for SVDQUANT
#
# Direction 1: Calibration samples × rank sweep
#   calib={8,16,32,64} × rank={4,16,32,64} × steps={10,15,20} = 48 runs
#
# Direction 2A: Phase-binned corrector
#   cache_lora_phase, rank=4, calib=4, steps={10,15,20} = 3 runs
#
# Direction 2B: Timestep-conditional corrector
#   cache_lora_ts, rank=4, calib=4, steps={10,15,20} = 3 runs
#
# Direction 3/5: Block-specific corrector (bug fixed)
#   cache_lora_block, rank=4, calib=4, steps={10,15,20} = 3 runs
#
# Direction 4: SVD-Aware corrector
#   cache_lora_svd, rank=4, calib=4, steps={10,15,20} = 3 runs
#
# Direction 6: Nested caching (deep region축소 sweep)
#   cache_start={10,12,14,16}, cache_mode={deepcache,cache_lora}, steps={10,15,20} = 24 runs
#
# Direction 9: Teacher-forced calibration
#   cache_lora_tf, rank=4, calib=4, steps={10,15,20} = 3 runs
#
# Usage:
#   bash run_svdquant_advanced.sh                     # 20 samples, all axes
#   bash run_svdquant_advanced.sh --gpu 1             # GPU 지정
#   bash run_svdquant_advanced.sh --test_run          # 2 samples smoke test
#   bash run_svdquant_advanced.sh --axis calib        # Dir 1 only
#   bash run_svdquant_advanced.sh --axis phase        # Dir 2A only
#   bash run_svdquant_advanced.sh --axis ts           # Dir 2B only
#   bash run_svdquant_advanced.sh --axis block        # Dir 3/5 (fixed) only
#   bash run_svdquant_advanced.sh --axis svd          # Dir 4 only
#   bash run_svdquant_advanced.sh --axis nested       # Dir 6 only
#   bash run_svdquant_advanced.sh --axis tf           # Dir 9 only

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/pixart_nvfp4_cache_compare.py"
ENV_PYTHON="/home/jameskimh/.dit/bin/accelerate"

# ── Defaults ──────────────────────────────────────────────────────────────────
NUM_SAMPLES=20
TEST_RUN=false
TEST_FLAG=""
GPU_ID=""
AXIS="all"
PORT=29500

STEPS=(10 15 20)
CS=8; CE=20          # range [8,20) — SVDQUANT optimal
BASE_CALIB=4
BASE_RANK=4
BASE_INTERVAL=2

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --test_run)    TEST_RUN=true; TEST_FLAG="--test_run" ;;
        --gpu)         GPU_ID="$2"; shift ;;
        --num_samples) NUM_SAMPLES="$2"; shift ;;
        --axis)        AXIS="$2"; shift ;;
        --port)        PORT="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "Using GPU: $GPU_ID"
fi

# ── Helper: run one experiment ────────────────────────────────────────────────
run_one() {
    local CACHE_MODE="$1" STEP="$2" RANK="$3" CALIB="$4"

    local CALIB_TAG=""
    if [ "$CALIB" != "$BASE_CALIB" ]; then
        CALIB_TAG="_cal${CALIB}"
    fi

    local MODE_SHORT="${CACHE_MODE/cache_lora/cl}"
    local CACHE_TAG="${MODE_SHORT}_r${RANK}${CALIB_TAG}"
    local TAG="SVDQUANT_${CACHE_TAG}_c${CS}-${CE}_steps${STEP}"
    local RESULT_DIR="/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/SVDQUANT/MJHQ/${TAG}"

    if [ -f "$RESULT_DIR/metrics.json" ] && [ "$TEST_RUN" = false ]; then
        echo "  SKIP: $TAG"
        return
    fi

    echo ""
    echo "  Running: $TAG"
    echo "  ─────────────────────────────────────────"

    $ENV_PYTHON launch \
        --num_processes 1 \
        --main_process_port "$PORT" \
        "$PYTHON_SCRIPT" \
        --quant_method   SVDQUANT \
        --cache_mode     "$CACHE_MODE" \
        --num_steps      "$STEP" \
        --num_samples    "$NUM_SAMPLES" \
        --guidance_scale 4.5 \
        --cache_start    "$CS" \
        --cache_end      "$CE" \
        --deepcache_interval "$BASE_INTERVAL" \
        --lora_rank      "$RANK" \
        --lora_calib     "$CALIB" \
        $TEST_FLAG

    echo "  Done: $TAG"
}

# ── Print header ──────────────────────────────────────────────────────────────
if $TEST_RUN; then
    echo "=== SVDQUANT ADVANCED — SMOKE TEST (2 samples) ==="
else
    echo "=== SVDQUANT ADVANCED SWEEP ($NUM_SAMPLES samples) ==="
fi
echo "Axis  : $AXIS"
echo "Steps : ${STEPS[*]}"
echo "Range : [${CS},${CE})"
echo ""

cd "$SCRIPT_DIR"

# ── Direction 1: Calibration × Rank sweep ────────────────────────────────────
if [[ "$AXIS" == "all" || "$AXIS" == "calib" ]]; then
    echo "======= Direction 1: calib={8,16,32,64} × rank={4,16,32,64} ======="
    for CALIB in 8 16 32 64; do
        for RANK in 4 16 32 64; do
            for STEP in "${STEPS[@]}"; do
                run_one "cache_lora" "$STEP" "$RANK" "$CALIB"
            done
        done
    done
fi

# ── Direction 2A: Phase-binned ───────────────────────────────────────────────
if [[ "$AXIS" == "all" || "$AXIS" == "phase" ]]; then
    echo ""
    echo "======= Direction 2A: Phase-binned corrector ======="
    for STEP in "${STEPS[@]}"; do
        run_one "cache_lora_phase" "$STEP" "$BASE_RANK" "$BASE_CALIB"
    done
fi

# ── Direction 2B: Timestep-conditional ───────────────────────────────────────
if [[ "$AXIS" == "all" || "$AXIS" == "ts" ]]; then
    echo ""
    echo "======= Direction 2B: Timestep-conditional corrector ======="
    for STEP in "${STEPS[@]}"; do
        run_one "cache_lora_ts" "$STEP" "$BASE_RANK" "$BASE_CALIB"
    done
fi

# ── Direction 3: Block-specific ───────────────────────────────────────────────
if [[ "$AXIS" == "all" || "$AXIS" == "block" ]]; then
    echo ""
    echo "======= Direction 3: Block-specific corrector ======="
    for STEP in "${STEPS[@]}"; do
        run_one "cache_lora_block" "$STEP" "$BASE_RANK" "$BASE_CALIB"
    done
fi

# ── Direction 4: SVD-Aware ───────────────────────────────────────────────────
if [[ "$AXIS" == "all" || "$AXIS" == "svd" ]]; then
    echo ""
    echo "======= Direction 4: SVD-Aware corrector ======="
    for STEP in "${STEPS[@]}"; do
        run_one "cache_lora_svd" "$STEP" "$BASE_RANK" "$BASE_CALIB"
    done
fi

# ── Direction 6: Nested caching (deep region 축소) ───────────────────────────
run_nested() {
    local CACHE_MODE="$1" STEP="$2" START="$3"
    local END=20

    local CACHE_TAG
    if [ "$CACHE_MODE" = "deepcache" ]; then
        CACHE_TAG="deepcache"
    else
        CACHE_TAG="cl_r${BASE_RANK}"
    fi
    local TAG="SVDQUANT_${CACHE_TAG}_c${START}-${END}_steps${STEP}"
    local RESULT_DIR="/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/SVDQUANT/MJHQ/${TAG}"

    if [ -f "$RESULT_DIR/metrics.json" ] && [ "$TEST_RUN" = false ]; then
        echo "  SKIP: $TAG"
        return
    fi

    echo ""
    echo "  Running: $TAG"
    echo "  ─────────────────────────────────────────"

    $ENV_PYTHON launch \
        --num_processes 1 \
        --main_process_port "$PORT" \
        "$PYTHON_SCRIPT" \
        --quant_method   SVDQUANT \
        --cache_mode     "$CACHE_MODE" \
        --num_steps      "$STEP" \
        --num_samples    "$NUM_SAMPLES" \
        --guidance_scale 4.5 \
        --cache_start    "$START" \
        --cache_end      "$END" \
        --deepcache_interval "$BASE_INTERVAL" \
        --lora_rank      "$BASE_RANK" \
        --lora_calib     "$BASE_CALIB" \
        $TEST_FLAG

    echo "  Done: $TAG"
}

if [[ "$AXIS" == "all" || "$AXIS" == "nested" ]]; then
    echo ""
    echo "======= Direction 6: Nested caching (cs={10,12,14,16}) ======="
    for START in 10 12 14 16; do
        for STEP in "${STEPS[@]}"; do
            run_nested "deepcache"   "$STEP" "$START"
            run_nested "cache_lora"  "$STEP" "$START"
        done
    done
fi

# ── Direction 9: Teacher-forced calibration ───────────────────────────────────
if [[ "$AXIS" == "all" || "$AXIS" == "tf" ]]; then
    echo ""
    echo "======= Direction 9: Teacher-forced calibration ======="
    for STEP in "${STEPS[@]}"; do
        run_one "cache_lora_tf" "$STEP" "$BASE_RANK" "$BASE_CALIB"
    done
fi

echo ""
echo "============================================================"
echo "SVDQUANT advanced sweep complete."
echo "Results in: $SCRIPT_DIR/results/MJHQ/"
echo "============================================================"
