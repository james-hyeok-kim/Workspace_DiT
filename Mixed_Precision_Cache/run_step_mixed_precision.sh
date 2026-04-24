#!/bin/bash
# run_step_mixed_precision.sh — Step-aware W3/W4 Mixed Precision Experiment
#
# GPU: 0, 1
# Phase 0: SVDQuant baseline (sanity)
# Phase 1: all-W4 / all-W3 upper bounds
# Phase 2: Sensitivity measurement
# Phase 3: K-sweep S1 (top-K tolerant pairs → W3)
# Phase 4: Ablation S2 (step-uniform) and S3 (layer-type-uniform)
#
# Usage:
#   bash run_step_mixed_precision.sh [phase]
#   bash run_step_mixed_precision.sh 0     # Phase 0 only
#   bash run_step_mixed_precision.sh all   # All phases sequentially

set -e
CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES

# .dit environment (has torch, diffusers, accelerate, torchmetrics, modelopt)
DIT_SITE="/home/jovyan/.dit/lib/python3.11/site-packages"
export PYTHONPATH="${DIT_SITE}:${PYTHONPATH}"
PYTHON="/opt/conda/bin/python3"

# Allow non-main rank to wait up to 2h while main computes FID/IS metrics
export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PHASE="${1:-all}"

MODEL_PATH="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
NUM_STEPS=10
NUM_SAMPLES=100
NUM_SAMPLES_SMOKE=5
GUIDANCE=4.5
DATASET="MJHQ"
N_CALIB=4
SEED_OFFSET=1000

RESULTS_DIR="results"
SENS_JSON="${RESULTS_DIR}/step_sensitivity_steps${NUM_STEPS}_cal${N_CALIB}_seed${SEED_OFFSET}.json"

ACCEL_LAUNCH="$PYTHON -m accelerate.commands.launch --num_processes 2 --mixed_precision no"

echo ""
echo "========================================"
echo "  Step-aware Mixed Precision Experiment"
echo "  GPU: ${CUDA_VISIBLE_DEVICES}"
echo "  Steps: ${NUM_STEPS}  Samples: ${NUM_SAMPLES}"
echo "========================================"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 0: SVDQuant baseline
# ─────────────────────────────────────────────────────────────────────────────
run_phase0() {
    echo "=== Phase 0: SVDQuant baseline (sanity) ==="
    $ACCEL_LAUNCH pixart_nvfp4_cache_compare.py \
        --quant_method SVDQUANT \
        --num_steps $NUM_STEPS \
        --num_samples $NUM_SAMPLES \
        --guidance_scale $GUIDANCE \
        --dataset_name $DATASET \
        --model_path "$MODEL_PATH"
    echo "[Phase 0] Done. Expected FID ≈ 121.9"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Upper / lower bounds
# ─────────────────────────────────────────────────────────────────────────────
run_phase1() {
    echo "=== Phase 1: Bounds ==="

    echo "--- Phase 1a: all-W4 (lower bound — should match SVDQuant FID ±5) ---"
    $ACCEL_LAUNCH pixart_nvfp4_cache_compare.py \
        --quant_method STEP_AWARE_MIXED \
        --num_steps $NUM_STEPS \
        --num_samples $NUM_SAMPLES \
        --guidance_scale $GUIDANCE \
        --dataset_name $DATASET \
        --model_path "$MODEL_PATH" \
        --schedule_family S1 \
        --low_bit_k 0

    echo "--- Phase 1b: all-W3 (upper bound — quality cliff) ---"
    $ACCEL_LAUNCH pixart_nvfp4_cache_compare.py \
        --quant_method STEP_AWARE_MIXED \
        --num_steps $NUM_STEPS \
        --num_samples $NUM_SAMPLES \
        --guidance_scale $GUIDANCE \
        --dataset_name $DATASET \
        --model_path "$MODEL_PATH" \
        --schedule_family S1 \
        --low_bit_k 70

    echo "[Phase 1] Done."
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Sensitivity measurement (single GPU — sequential ablation)
# ─────────────────────────────────────────────────────────────────────────────
run_phase2() {
    echo "=== Phase 2: Sensitivity measurement ==="
    echo "  Pairs: 70  n_calib: ${N_CALIB}  ETA: ~67 min"
    CUDA_VISIBLE_DEVICES=0 $PYTHON measure_step_sensitivity.py \
        --model_path "$MODEL_PATH" \
        --num_steps $NUM_STEPS \
        --n_calib $N_CALIB \
        --calib_seed_offset $SEED_OFFSET \
        --guidance_scale $GUIDANCE \
        --dataset_name $DATASET \
        --gpu 0 \
        --output_dir "$RESULTS_DIR" \
        --resume
    echo "[Phase 2] Done. Output: ${SENS_JSON}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: K-sweep S1 (top-K tolerant (layer_type, step) pairs → W3)
# ─────────────────────────────────────────────────────────────────────────────
run_phase3() {
    echo "=== Phase 3: K-sweep S1 ==="

    if [ ! -f "$SENS_JSON" ]; then
        echo "[ERROR] Sensitivity JSON not found: ${SENS_JSON}"
        echo "  Run Phase 2 first: bash run_step_mixed_precision.sh 2"
        exit 1
    fi

    for K in 10 20 30 40 50 60; do
        echo "--- K=${K} ---"
        $ACCEL_LAUNCH pixart_nvfp4_cache_compare.py \
            --quant_method STEP_AWARE_MIXED \
            --num_steps $NUM_STEPS \
            --num_samples $NUM_SAMPLES \
            --guidance_scale $GUIDANCE \
            --dataset_name $DATASET \
            --model_path "$MODEL_PATH" \
            --schedule_family S1 \
            --low_bit_k $K \
            --sensitivity_json "$SENS_JSON"
    done
    echo "[Phase 3] Done."
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Ablation S2 (step-uniform) and S3 (layer-type-uniform)
# ─────────────────────────────────────────────────────────────────────────────
run_phase4() {
    echo "=== Phase 4: Ablation S2 + S3 ==="

    echo "--- S2: step-uniform (K steps → all-W3) ---"
    for K in 2 4 6 8 10; do
        echo "  S2 K=${K}"
        $ACCEL_LAUNCH pixart_nvfp4_cache_compare.py \
            --quant_method STEP_AWARE_MIXED \
            --num_steps $NUM_STEPS \
            --num_samples $NUM_SAMPLES \
            --guidance_scale $GUIDANCE \
            --dataset_name $DATASET \
            --model_path "$MODEL_PATH" \
            --schedule_family S2 \
            --low_bit_k $K \
            --sensitivity_json "$SENS_JSON"
    done

    echo "--- S3: layer-type-uniform (K types → all-W3) ---"
    for K in 1 2 3 4 5 6 7; do
        echo "  S3 K=${K}"
        $ACCEL_LAUNCH pixart_nvfp4_cache_compare.py \
            --quant_method STEP_AWARE_MIXED \
            --num_steps $NUM_STEPS \
            --num_samples $NUM_SAMPLES \
            --guidance_scale $GUIDANCE \
            --dataset_name $DATASET \
            --model_path "$MODEL_PATH" \
            --schedule_family S3 \
            --low_bit_k $K \
            --sensitivity_json "$SENS_JSON"
    done

    echo "[Phase 4] Done."
}

# ─────────────────────────────────────────────────────────────────────────────
# Smoke test (fast sanity, --test_run)
# ─────────────────────────────────────────────────────────────────────────────
run_smoke() {
    echo "=== Smoke test ==="
    $ACCEL_LAUNCH pixart_nvfp4_cache_compare.py \
        --quant_method STEP_AWARE_MIXED \
        --num_steps $NUM_STEPS \
        --num_samples $NUM_SAMPLES_SMOKE \
        --guidance_scale $GUIDANCE \
        --dataset_name $DATASET \
        --model_path "$MODEL_PATH" \
        --schedule_family S1 \
        --low_bit_k 20 \
        --test_run
    echo "[Smoke] Done."
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────
case "$PHASE" in
    0)   run_phase0 ;;
    1)   run_phase1 ;;
    2)   run_phase2 ;;
    3)   run_phase3 ;;
    4)   run_phase4 ;;
    smoke) run_smoke ;;
    all)
        run_phase0
        run_phase1
        run_phase2
        run_phase3
        run_phase4
        ;;
    *)
        echo "Usage: bash run_step_mixed_precision.sh [0|1|2|3|4|smoke|all]"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "  Experiment complete. Phase: $PHASE"
echo "  Results CSV: ${RESULTS_DIR}/sweep_results.csv"
echo "========================================"
