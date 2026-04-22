#!/bin/bash
# run_nl_gelu_100.sh — nl_gelu 100-sample, step20 → step15, 2-GPU sequential
#
# Usage:
#   bash run_nl_gelu_100.sh                   # 본 실행
#   nohup bash run_nl_gelu_100.sh &> logs/full_gelu100.log &
#   disown  # 세션 끊겨도 계속 실행

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1

echo "============================================================"
echo "  nl_gelu 100-sample run  (2-GPU: step20 → step15)"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# step 20 먼저
echo ""
echo ">>> step 20 start  $(date '+%H:%M:%S')"
bash run_nonlinear.sh \
    --modes gelu \
    --num_samples 100 \
    --steps 20 \
    2>&1 | tee logs/gelu100_s20_2gpu.log
echo ">>> step 20 done   $(date '+%H:%M:%S')"

# step 15 이어서
echo ""
echo ">>> step 15 start  $(date '+%H:%M:%S')"
bash run_nonlinear.sh \
    --modes gelu \
    --num_samples 100 \
    --steps 15 \
    2>&1 | tee logs/gelu100_s15_2gpu.log
echo ">>> step 15 done   $(date '+%H:%M:%S')"

echo ""
echo "============================================================"
echo "  ALL DONE  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
