#!/bin/bash
# run_drift_traj.sh — Combined Drift+Trajectory Distillation experiment
#
# GPU0: cache_nl_gelu + drift_traj, steps=20, n=100
#   → SVDQUANT_nl_gelu_drifttraj_r4_m32_c8-20_steps20
#
# Baseline ref: SVDQUANT_nl_gelu_r4_m32_c8-20_steps20  FID≈124.9

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/jovyan/.dit/bin/python3"
PY="$PYTHON $SCRIPT_DIR/pixart_nvfp4_cache_compare.py"
mkdir -p "$SCRIPT_DIR/logs"

echo "Launching drift_traj experiment on GPU0..."
echo "  Phase 2.5: load/train base drift corrector + collect drift data"
echo "  Phase 2.75: combined drift+traj fine-tuning (K=6, λ=0.1 warm-up, 200 iters)"
echo "  Phase 4: generate 100 samples, compute FID/IS/CLIP"
echo ""

CUDA_VISIBLE_DEVICES=0 nohup $PY \
    --quant_method   SVDQUANT \
    --cache_mode     cache_nl_gelu \
    --nl_loss_type   drift_traj \
    --num_steps      20 \
    --num_samples    100 \
    --guidance_scale 4.5 \
    --deepcache_interval 2 \
    --lora_rank      4 \
    --lora_calib     4 \
    --nl_mid_dim     32 \
    > "$SCRIPT_DIR/logs/drift_traj_gpu0.log" 2>&1 &
PID=$!

echo "PID: $PID"
echo "Log: tail -f $SCRIPT_DIR/logs/drift_traj_gpu0.log"
disown $PID
