#!/bin/bash
# run_nonlinear_parallel.sh — 2-GPU parallel nonlinear experiment
#
# GPU 0 (port 29500): nl_gelu + nl_mlp (6 runs)
# GPU 1 (port 29501): nl_res  + nl_film (6 runs)
#
# Usage:
#   bash run_nonlinear_parallel.sh               # 20 samples
#   bash run_nonlinear_parallel.sh --test_run    # 2 samples smoke test
#   bash run_nonlinear_parallel.sh --num_samples 20

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

PASSTHROUGH=""
NUM_SAMPLES=20

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --test_run)    PASSTHROUGH="$PASSTHROUGH --test_run" ;;
        --num_samples) NUM_SAMPLES="$2"; PASSTHROUGH="$PASSTHROUGH --num_samples $2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

echo "============================================================"
echo "  Nonlinear Parallel Run (GPU 0: gelu+mlp / GPU 1: res+film)"
echo "  Samples: $NUM_SAMPLES"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

bash "$SCRIPT_DIR/run_nonlinear.sh" \
    --gpu 0 --port 29500 --modes gelu,mlp $PASSTHROUGH \
    >> "$LOG_DIR/gpu0_nonlinear.log" 2>&1 &
PID0=$!

bash "$SCRIPT_DIR/run_nonlinear.sh" \
    --gpu 1 --port 29501 --modes res,film $PASSTHROUGH \
    >> "$LOG_DIR/gpu1_nonlinear.log" 2>&1 &
PID1=$!

echo "GPU 0 PID=$PID0 → $LOG_DIR/gpu0_nonlinear.log"
echo "GPU 1 PID=$PID1 → $LOG_DIR/gpu1_nonlinear.log"
echo ""
echo "Waiting for both GPUs..."

EXIT0=0; EXIT1=0
wait $PID0 || EXIT0=$?
wait $PID1 || EXIT1=$?

echo ""
echo "============================================================"
if [ $EXIT0 -eq 0 ] && [ $EXIT1 -eq 0 ]; then
    echo "  ALL DONE  ($(date '+%Y-%m-%d %H:%M:%S'))"
else
    echo "  DONE WITH ERRORS: GPU0=$EXIT0 GPU1=$EXIT1"
    [ $EXIT0 -ne 0 ] && echo "  → $LOG_DIR/gpu0_nonlinear.log"
    [ $EXIT1 -ne 0 ] && echo "  → $LOG_DIR/gpu1_nonlinear.log"
    exit 1
fi
echo "============================================================"
