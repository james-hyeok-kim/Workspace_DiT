#!/bin/bash
# run_nl_gelu_100.sh — nl_gelu 100-sample
# GPU 0: steps20 / GPU 1: steps15 — 동시 병렬 실행 (각 1-GPU, NCCL 없음)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

echo "============================================================"
echo "  nl_gelu 100-sample run"
echo "  GPU 0: steps20  |  GPU 1: steps15  (parallel)"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# GPU 0 — steps20
CUDA_VISIBLE_DEVICES=0 bash run_nonlinear.sh \
    --modes gelu --num_samples 100 --steps 20 --gpu 0 --port 29500 \
    > logs/gelu100_s20_gpu0.log 2>&1 &
PID0=$!

# GPU 1 — steps15
CUDA_VISIBLE_DEVICES=1 bash run_nonlinear.sh \
    --modes gelu --num_samples 100 --steps 15 --gpu 1 --port 29501 \
    > logs/gelu100_s15_gpu1.log 2>&1 &
PID1=$!

echo "GPU 0 (steps20) PID=$PID0 → logs/gelu100_s20_gpu0.log"
echo "GPU 1 (steps15) PID=$PID1 → logs/gelu100_s15_gpu1.log"
echo ""
echo "Waiting..."

EXIT0=0; EXIT1=0
wait $PID0 || EXIT0=$?
wait $PID1 || EXIT1=$?

echo ""
echo "============================================================"
if [ $EXIT0 -eq 0 ] && [ $EXIT1 -eq 0 ]; then
    echo "  ALL DONE  $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "  DONE WITH ERRORS: GPU0=$EXIT0 GPU1=$EXIT1"
    [ $EXIT0 -ne 0 ] && echo "  → logs/gelu100_s20_gpu0.log"
    [ $EXIT1 -ne 0 ] && echo "  → logs/gelu100_s15_gpu1.log"
fi
echo "============================================================"
