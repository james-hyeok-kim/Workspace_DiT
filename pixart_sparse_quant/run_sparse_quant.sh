#!/bin/bash
# ============================================================
# 2:4 Structured Sparsity + Quantization Experiment Runner
#
# Usage:
#   TEST_MODE=1 bash run_sparse_quant.sh           # smoke test (5 samples, 1 GPU)
#   TEST_MODE=0 bash run_sparse_quant.sh           # full P1 (100 samples, 2 GPU)
#   TEST_MODE=0 RUN_P2=1 bash run_sparse_quant.sh  # P1 + P2 quality experiments
#
# Config format: "sparsity_mode  wgt_mode  act_mode  svd_rank  use_semi_structured  alpha"
# ============================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1

ACCELERATE="/home/jameskimh/.dit/bin/accelerate"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="PixArt-alpha/PixArt-XL-2-1024-MS"
REF_DIR="/data/jameskimh/james_dit_ref/ref_images_fp16"
TEST_MODE="${TEST_MODE:-1}"

if [ "$TEST_MODE" -eq 1 ]; then
    NUM_SAMPLES=5
    MODE_LABEL="test"
    N_PROC=1          # single GPU: avoids torchmetrics distributed sync issue (uneven 3+2 split)
else
    NUM_SAMPLES=100
    MODE_LABEL="prod"
    N_PROC=2          # 2 GPU: 100 samples → even 50+50 split → no NCCL desync
fi

CSV_PATH="${BASE_DIR}/results/summary.csv"

# ── Experiment matrix ─────────────────────────────────────────────────────────
# Format: "sparsity_mode  wgt_mode  act_mode  svd_rank  use_semi_structured  alpha"
#
# P1: baseline vs FP8 sparsity (100 samples → reliable FID/IS + correct 2-GPU latency)
P1_CONFIGS=(
    "none      NVFP4  NVFP4  32  false  0.50"   # BASELINE_NVFP4
    "magnitude NVFP8  NVFP8  32  false  0.50"   # SP_MAG_WNVFP8_ANVFP8_R32
    "magnitude MXFP8  MXFP8  32  false  0.50"   # SP_MAG_WMXFP8_AMXFP8_R32
)

# P2: PSNR 개선 실험 — NVFP8이 가장 유망하므로 rank / alpha 탐색
P2_CONFIGS=(
    "magnitude NVFP8  NVFP8  64  false  0.50"   # R64: SVD correction 강화
    "magnitude NVFP8  NVFP8  32  false  0.75"   # A075: SmoothQuant more weight-side
)

RUN_P2="${RUN_P2:-0}"

CONFIGS=("${P1_CONFIGS[@]}")
if [ "$RUN_P2" -eq 1 ]; then
    CONFIGS+=("${P2_CONFIGS[@]}")
fi

mkdir -p "${BASE_DIR}/logs"

echo "================================================================"
echo "  2:4 Sparsity + Quantization Experiment"
echo "  Mode: ${MODE_LABEL} | Samples: ${NUM_SAMPLES} | Procs: ${N_PROC}"
echo "  Configs: ${#CONFIGS[@]}"
echo "  CSV: ${CSV_PATH}"
echo "================================================================"
echo ""

run_config() {
    local sparsity="$1"
    local wgt="$2"
    local act="$3"
    local rank="$4"
    local semi="$5"
    local alpha="$6"

    # Build config ID
    local sparsity_tag
    case "$sparsity" in
        magnitude) sparsity_tag="MAG" ;;
        sparsegpt)  sparsity_tag="GPT" ;;
        none)       sparsity_tag="NONE" ;;
        *)          sparsity_tag="${sparsity^^}" ;;
    esac

    local alpha_tag=""
    if [ "$alpha" != "0.50" ] && [ "$alpha" != "0.5" ]; then
        alpha_tag="_A$(echo "$alpha" | tr -d '.')"
    fi

    if [ "$sparsity" = "none" ]; then
        CONFIG_ID="BASELINE_${wgt}"
    else
        CONFIG_ID="SP_${sparsity_tag}_W${wgt}_A${act}_R${rank}${alpha_tag}"
        [ "$semi" = "true" ] && CONFIG_ID="${CONFIG_ID}_SS"
    fi

    local RESULT_DIR="${BASE_DIR}/results/${MODE_LABEL}/${CONFIG_ID}"
    local LOG_FILE="${BASE_DIR}/logs/${MODE_LABEL}_${CONFIG_ID}.log"
    mkdir -p "${RESULT_DIR}"

    local SEMI_FLAG=""
    [ "$semi" = "true" ] && SEMI_FLAG="--use_semi_structured"

    echo "──────────────────────────────────────────────────────────"
    echo "  Config: ${CONFIG_ID}"
    echo "  Sparsity: ${sparsity} | Wgt: ${wgt} | Act: ${act} | Rank: ${rank} | Alpha: ${alpha}"
    [ -n "$SEMI_FLAG" ] && echo "  cuSPARSELt: enabled"
    echo "  Log: ${LOG_FILE}"
    echo "──────────────────────────────────────────────────────────"

    local LAUNCH_FLAGS="--num_processes ${N_PROC}"
    [ "$N_PROC" -gt 1 ] && LAUNCH_FLAGS="${LAUNCH_FLAGS} --multi_gpu"

    "${ACCELERATE}" launch \
        ${LAUNCH_FLAGS} \
        "${BASE_DIR}/pixart_sparse_quant.py" \
        --config_id "${CONFIG_ID}" \
        --model_path "${MODEL_PATH}" \
        --ref_dir "${REF_DIR}" \
        --save_dir "${RESULT_DIR}" \
        --num_samples "${NUM_SAMPLES}" \
        --sparsity_mode "${sparsity}" \
        --wgt_mode "${wgt}" \
        --act_mode "${act}" \
        --lowrank "${rank}" \
        --alpha "${alpha}" \
        --block_size 16 \
        --numeric_dtype half \
        --dataset_name MJHQ \
        --csv_path "${CSV_PATH}" \
        ${SEMI_FLAG} \
        2>&1 | tee "${LOG_FILE}"

    echo ""
    echo "✅  ${CONFIG_ID} complete."
    echo ""
}

# ── Run all configs ───────────────────────────────────────────────────────────
TOTAL="${#CONFIGS[@]}"
CURRENT=0

for cfg in "${CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    read -r SPARSITY WGT ACT RANK SEMI ALPHA <<< "$cfg"

    echo ""
    echo "▶ [$CURRENT/$TOTAL] Running config…"
    run_config "$SPARSITY" "$WGT" "$ACT" "$RANK" "$SEMI" "$ALPHA"
done

# ── Print summary table ───────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  All experiments complete!"
echo "  Summary CSV: ${CSV_PATH}"
echo "================================================================"

if [ -f "${CSV_PATH}" ]; then
    echo ""
    echo "Results (FID↓  IS↑  PSNR↑  Latency↓):"
    CSV_PATH="${CSV_PATH}" /home/jameskimh/.dit/bin/python3 - << 'PYEOF'
import csv, sys, os
path = os.environ.get("CSV_PATH", "")
if not path or not os.path.exists(path):
    sys.exit(0)
with open(path) as f:
    rows = list(csv.DictReader(f))
cols = ["config_id","num_samples","FID","IS","PSNR","SSIM","LPIPS","CLIP","avg_latency_ms"]
fmt  = "{:<42} {:>7} {:>8} {:>6} {:>7} {:>6} {:>7} {:>6} {:>12}"
print(fmt.format(*cols))
print("-" * 105)
for r in rows:
    print(fmt.format(*[r.get(c,"") for c in cols]))
PYEOF
fi
