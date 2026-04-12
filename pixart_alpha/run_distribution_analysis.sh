#!/bin/bash
# run_distribution_analysis.sh
# PixArt-Alpha activation / weight / output 분포 분석
set -e

PYTHON="/home/jameskimh/.dit/bin/python"
PY="pixart_distribution_analysis.py"
OUTPUT_DIR="results/distribution_analysis"

# HF 토큰 로드
[ -f ~/.env ] && source ~/.env

echo "========================================"
echo "  PixArt Distribution Analysis"
echo "  Prompts: 8  |  Steps: 20"
echo "  Output: ${OUTPUT_DIR}"
echo "========================================"

"${PYTHON}" "${PY}" \
    --num_prompts 8 \
    --num_steps 20 \
    --output_dir "${OUTPUT_DIR}" \
    --dataset_name MJHQ

echo ""
echo "Done. Results in: ${OUTPUT_DIR}"
