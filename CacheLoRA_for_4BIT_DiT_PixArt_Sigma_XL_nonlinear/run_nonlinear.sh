#!/bin/bash
# run_nonlinear.sh — Nonlinear Cache-LoRA corrector experiments
#
# 4 options × steps={10,15,20} × SVDQUANT = 12 runs (n=20)
# + linear baseline for fair comparison = 3 runs
# + deepcache baseline = 3 runs
# Total: 18 runs
#
# Usage:
#   bash run_nonlinear.sh                    # 20 samples, all
#   bash run_nonlinear.sh --test_run         # 2 samples smoke test
#   bash run_nonlinear.sh --gpu 0            # GPU 지정

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/pixart_nvfp4_cache_compare.py"
ENV_PYTHON="/home/jameskimh/.dit/bin/accelerate"
DATA_ROOT="/data/jameskimh/james_dit_pixart_sigma_xl_mjhq"

NUM_SAMPLES=20
TEST_RUN=false
TEST_FLAG=""
GPU_ID=""
PORT=29500
METHOD=SVDQUANT
CS=8; CE=20; INTERVAL=2
RANK=4; MID=32; CALIB=4

STEPS=(10 15 20)

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --test_run)    TEST_RUN=true; TEST_FLAG="--test_run" ;;
        --gpu)         GPU_ID="$2"; shift ;;
        --port)        PORT="$2"; shift ;;
        --num_samples) NUM_SAMPLES="$2"; shift ;;
        --method)      METHOD="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "Using GPU: $GPU_ID"
fi

run_one() {
    local CACHE_MODE="$1" STEP="$2" EXTRA_FLAGS="$3"

    # Build tag for skip check
    local NL_TYPE="${CACHE_MODE#cache_nl_}"
    local TAG
    if [[ "$CACHE_MODE" == cache_nl_* ]]; then
        TAG="${METHOD}_nl_${NL_TYPE}_r${RANK}_m${MID}_c${CS}-${CE}_steps${STEP}"
    elif [ "$CACHE_MODE" = "cache_lora" ]; then
        TAG="${METHOD}_cl_r${RANK}_c${CS}-${CE}_steps${STEP}"
    elif [ "$CACHE_MODE" = "deepcache" ]; then
        TAG="${METHOD}_deepcache_c${CS}-${CE}_steps${STEP}"
    fi

    local RESULT_DIR="$DATA_ROOT/$METHOD/MJHQ/$TAG"
    # Baselines (deepcache/cache_lora) always skip if metrics.json exists — avoids overwriting main sweep n=100 data
    # nl_ modes skip only when not in test_run (so smoke test still exercises them)
    if [ -f "$RESULT_DIR/metrics.json" ]; then
        if [[ "$CACHE_MODE" != cache_nl_* ]] || [ "$TEST_RUN" = false ]; then
            echo "  SKIP: $TAG"
            return
        fi
    fi

    echo ""
    echo "  Running: $TAG"
    echo "  ─────────────────────────────────────────"

    $ENV_PYTHON launch \
        --num_processes 1 \
        --main_process_port "$PORT" \
        "$PYTHON_SCRIPT" \
        --quant_method   "$METHOD" \
        --cache_mode     "$CACHE_MODE" \
        --num_steps      "$STEP" \
        --num_samples    "$NUM_SAMPLES" \
        --guidance_scale 4.5 \
        --cache_start    "$CS" \
        --cache_end      "$CE" \
        --deepcache_interval "$INTERVAL" \
        --lora_rank      "$RANK" \
        --lora_calib     "$CALIB" \
        --nl_mid_dim     "$MID" \
        $EXTRA_FLAGS \
        $TEST_FLAG

    echo "  Done: $TAG"
}

echo "============================================================"
echo "  Nonlinear Cache-LoRA Experiment ($NUM_SAMPLES samples)"
echo "  Method: $METHOD  Range: [$CS,$CE)  Rank: $RANK  Mid: $MID"
echo "============================================================"

cd "$SCRIPT_DIR"

# ── Baselines ─────────────────────────────────────────────────────────
echo ""
echo "======= Baselines ======="
for STEP in "${STEPS[@]}"; do
    run_one "deepcache"  "$STEP"
    run_one "cache_lora" "$STEP"
done

# ── Option 1: GELU bottleneck ────────────────────────────────────────
echo ""
echo "======= Option 1: GELU bottleneck (rank=$RANK) ======="
for STEP in "${STEPS[@]}"; do
    run_one "cache_nl_gelu" "$STEP"
done

# ── Option 2: Bottleneck MLP (mid=$MID) ──────────────────────────────
echo ""
echo "======= Option 2: Bottleneck MLP (mid=$MID) ======="
for STEP in "${STEPS[@]}"; do
    run_one "cache_nl_mlp" "$STEP"
done

# ── Option 3: Residual MLP (mid=$MID) ────────────────────────────────
echo ""
echo "======= Option 3: Residual MLP (mid=$MID) ======="
for STEP in "${STEPS[@]}"; do
    run_one "cache_nl_res" "$STEP"
done

# ── Option 4: FiLM conditioned (mid=$MID) ────────────────────────────
echo ""
echo "======= Option 4: FiLM conditioned (mid=$MID) ======="
for STEP in "${STEPS[@]}"; do
    run_one "cache_nl_film" "$STEP"
done

echo ""
echo "============================================================"
echo "Nonlinear experiment complete."
echo "============================================================"
