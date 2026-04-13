#!/bin/bash
# =============================================================================
# run_layer_sensitivity.sh
# Layer-wise FP quantization sensitivity analysis for DiT (PixArt-Alpha/Sigma)
#
# Usage:
#   bash run_layer_sensitivity.sh [PHASE] [NUM_SAMPLES] [OPTIONS...]
#
#   PHASE       : 1 | 2 | 3 | final   (default: 1)
#   NUM_SAMPLES : integer              (default: 30 for phase 1-3, 100 for final)
#
# Examples:
#   bash run_layer_sensitivity.sh 1 30          # Phase 1 with 30 samples
#   bash run_layer_sensitivity.sh 2 30          # Phase 2 with 30 samples
#   bash run_layer_sensitivity.sh 3 30          # Phase 3 with 30 samples
#   bash run_layer_sensitivity.sh final 100     # Final evaluation (100 samples)
#   bash run_layer_sensitivity.sh 1 2 --test_run  # Smoke test
# =============================================================================

set -euo pipefail

# ── Configurable parameters ──────────────────────────────────────────────────
PHASE=${1:-1}
NUM_SAMPLES=${2:-30}
shift 2 2>/dev/null || true   # remove first two positional args; keep any extras

NUM_GPUS=2
MODEL_PATH="PixArt-alpha/PixArt-XL-2-1024-MS"
REF_DIR="ref_images"
SAVE_DIR="results/sensitivity"
DATASET="MJHQ"
T_STEPS=20

# Formats to test (space-separated)
FORMATS="NVFP4 MXFP4 MXFP6_E2M3 MXFP6_E3M2 MXFP8 NVFP8"

# Block sizes (space-separated)
BLOCK_SIZES="16 32 64"

# Scale dtypes (space-separated)
SCALE_DTYPES="FP16 BF16 FP32 NVFP8 MXFP8"

# All 28 transformer blocks (0-27); override below to probe a subset
TARGET_BLOCKS=$(seq 0 27 | tr '\n' ' ')

# Log directory
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/sensitivity_phase${PHASE}_${TIMESTAMP}.log"

# ── Derived num_samples for final phase ─────────────────────────────────────
if [[ "${PHASE}" == "final" && "${NUM_SAMPLES}" -lt 100 ]]; then
    echo "⚠️  Phase final: forcing NUM_SAMPLES=100 (was ${NUM_SAMPLES})"
    NUM_SAMPLES=100
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo "======================================================"
echo "  Layer Sensitivity Analysis"
echo "  Phase       : ${PHASE}"
echo "  Num samples : ${NUM_SAMPLES}"
echo "  GPUs        : ${NUM_GPUS}"
echo "  Model       : ${MODEL_PATH}"
echo "  Save dir    : ${SAVE_DIR}"
echo "  Log         : ${LOG_FILE}"
echo "======================================================"

# ── Load environment variables (HF token, etc.) ─────────────────────────────
if [[ -f "${HOME}/.env" ]]; then
    set -a
    source "${HOME}/.env"
    set +a
    echo "✅ Loaded ~/.env"
fi

# ── Run ──────────────────────────────────────────────────────────────────────
accelerate launch \
    --num_processes "${NUM_GPUS}" \
    pixart_layer_sensitivity.py \
    --phase          "${PHASE}" \
    --num_samples    "${NUM_SAMPLES}" \
    --model_path     "${MODEL_PATH}" \
    --ref_dir        "${REF_DIR}" \
    --save_dir       "${SAVE_DIR}" \
    --dataset_name   "${DATASET}" \
    --formats        ${FORMATS} \
    --block_sizes    ${BLOCK_SIZES} \
    --scale_dtypes   ${SCALE_DTYPES} \
    --target_blocks  ${TARGET_BLOCKS} \
    --t_steps        "${T_STEPS}" \
    "$@" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "======================================================"
echo "  Phase ${PHASE} complete.  Log → ${LOG_FILE}"
echo "======================================================"
