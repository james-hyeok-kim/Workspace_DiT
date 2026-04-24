#!/bin/bash
# run_ablation3.sh — nl_gelu (drift) 3-way ablation
#
# Exp1 (GPU0): cache_start=12  → SVDQUANT_nl_gelu_r4_m32_c12-20_steps20
# Exp2 (GPU1): rank=8          → SVDQUANT_nl_gelu_r8_m32_c8-20_steps20
# Exp3 (GPU2): calib=8         → SVDQUANT_nl_gelu_r4_m32_cal8_c8-20_steps20
#
# Baseline (참고): SVDQUANT_nl_gelu_r4_m32_c8-20_steps20  FID=124.9

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/jovyan/.dit/bin/python3"
PY="$PYTHON $SCRIPT_DIR/pixart_nvfp4_cache_compare.py"
mkdir -p "$SCRIPT_DIR/logs"

COMMON=(
    --quant_method   SVDQUANT
    --cache_mode     cache_nl_gelu
    --nl_loss_type   drift
    --num_steps      20
    --num_samples    100
    --guidance_scale 4.5
    --deepcache_interval 2
    --nl_mid_dim     32
)

echo "Launching 3 ablation experiments in parallel..."
echo "  GPU0: cache_start=12  (구간 축소)"
echo "  GPU1: lora_rank=8     (rank 증가)"
echo "  GPU2: lora_calib=8    (calibration 증가)"
echo ""

# Exp1: cache 구간 축소 (cs=12)
CUDA_VISIBLE_DEVICES=0 nohup $PY \
    "${COMMON[@]}" \
    --cache_start 12 --cache_end 20 \
    --lora_rank  4   --lora_calib 4 \
    > "$SCRIPT_DIR/logs/ablation_cs12_gpu0.log" 2>&1 &
PID1=$!

# Exp2: rank 증가 (r=8)
CUDA_VISIBLE_DEVICES=1 nohup $PY \
    "${COMMON[@]}" \
    --cache_start 8  --cache_end 20 \
    --lora_rank  8   --lora_calib 4 \
    > "$SCRIPT_DIR/logs/ablation_r8_gpu1.log" 2>&1 &
PID2=$!

# Exp3: calibration 증가 (calib=8)
CUDA_VISIBLE_DEVICES=2 nohup $PY \
    "${COMMON[@]}" \
    --cache_start 8  --cache_end 20 \
    --lora_rank  4   --lora_calib 8 \
    > "$SCRIPT_DIR/logs/ablation_cal8_gpu2.log" 2>&1 &
PID3=$!

echo "PIDs: GPU0=$PID1  GPU1=$PID2  GPU2=$PID3"
echo "Logs:"
echo "  tail -f $SCRIPT_DIR/logs/ablation_cs12_gpu0.log"
echo "  tail -f $SCRIPT_DIR/logs/ablation_r8_gpu1.log"
echo "  tail -f $SCRIPT_DIR/logs/ablation_cal8_gpu2.log"
disown $PID1 $PID2 $PID3
