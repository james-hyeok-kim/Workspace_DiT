#!/bin/bash
# run_selective_skip.sh — block-wise sensitivity-based selective skip experiment
#
# Phase 0: measure per-block sensitivity (Metric A + C) on GPU 0 (blocking, ~6 min)
# Phase 1: K-sweep on GPU 0 (K=6,8,10) and GPU 1 (K=12,14,16) in parallel (~18 min each)
# Phase 2: print aggregated results table
#
# Baselines (steps=20, n=100):
#   SVDQUANT no-cache : FID=121.32  tpi=2.85s  1.00×
#   DeepCache c8-20   : FID=129.14  tpi=2.33s  1.22×
#   nl_gelu drift     : FID=124.94  tpi=2.53s  1.13×
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

PYTHON=/home/jovyan/.dit/bin/python3
STEPS=${STEPS:-20}
NUM=${NUM_SAMPLES:-100}
METRIC=${METRIC:-A}
INT=2
SENS_JSON=/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/SVDQUANT/MJHQ/sensitivity/SVDQUANT_steps${STEPS}_cal4_seed1000.json

echo "============================================================"
echo "  Selective Skip Experiment"
echo "  Steps=$STEPS  Samples=$NUM  Metric=$METRIC  Interval=$INT"
echo "  SensitivityJSON=$SENS_JSON"
echo "============================================================"

# ── Phase 0: sensitivity measurement (GPU 0, blocking) ───────────────────────
if [ ! -f "$SENS_JSON" ]; then
    echo ""
    echo "[$(date '+%H:%M:%S')] Phase 0: measuring block sensitivity (GPU 0)..."
    CUDA_VISIBLE_DEVICES=0 $PYTHON measure_block_sensitivity.py \
        --quant_method SVDQUANT \
        --num_steps $STEPS \
        --n_calib 4 \
        --calib_seed_offset 1000 \
        --deepcache_interval $INT \
        --metric both \
        > logs/sensitivity_steps${STEPS}.log 2>&1
    echo "[$(date '+%H:%M:%S')] Phase 0 done → $SENS_JSON"
else
    echo "[$(date '+%H:%M:%S')] Phase 0: sensitivity JSON found, skipping measurement."
fi

# ── Phase 1: K-sweep — GPU 0 (K=6,8,10) and GPU 1 (K=12,14,16) in parallel ──
run_k () {
    local GPU=$1
    local K=$2
    local TAG=SVDQUANT_selective_k${K}_m${METRIC}_steps${STEPS}
    local LOG=logs/${TAG}.log
    echo "[$(date '+%H:%M:%S')] GPU${GPU}: K=$K → $LOG"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON pixart_nvfp4_cache_compare.py \
        --quant_method SVDQUANT \
        --cache_mode selective_skip \
        --num_steps $STEPS \
        --num_samples $NUM \
        --sensitivity_json "$SENS_JSON" \
        --skip_k $K \
        --sensitivity_metric $METRIC \
        --deepcache_interval $INT \
        --guidance_scale 4.5 \
        > "$LOG" 2>&1
    echo "[$(date '+%H:%M:%S')] GPU${GPU}: K=$K done"
}

echo ""
echo "[$(date '+%H:%M:%S')] Phase 1: K-sweep (GPU0=K6,8,10 | GPU1=K12,14,16)..."
(
    for K in 6 8 10; do run_k 0 $K; done
) &
PID0=$!
(
    for K in 12 14 16; do run_k 1 $K; done
) &
PID1=$!

wait $PID0 $PID1
echo "[$(date '+%H:%M:%S')] Phase 1 complete."

# ── Phase 2: aggregate results ────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Results (steps=$STEPS, n=$NUM, metric=$METRIC)"
echo "============================================================"
$PYTHON - <<PY
import json, os
base = '/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/SVDQUANT/MJHQ'
sens_path = '$SENS_JSON'

# Print sensitivity ranking
if os.path.exists(sens_path):
    s = json.load(open(sens_path))
    if 'ranking_A' in s:
        print(f"  Ranking A (low→high): {s['ranking_A']}")
    if 'ranking_C' in s:
        print(f"  Ranking C (low→high): {s['ranking_C']}")
    if 'kendall_tau_A_vs_C' in s:
        print(f"  Kendall tau A vs C  : {s['kendall_tau_A_vs_C']:.3f}")
    print()

# Print baselines
baselines = [
    ('SVDQUANT_none_steps$STEPS',       'no-cache'),
    ('SVDQUANT_deepcache_c8-20_steps$STEPS', 'DeepCache c8-20'),
]
print(f"  {'Method':<35} {'FID':>7}  {'CLIP':>7}  {'tpi':>6}  {'spdup':>6}")
print(f"  {'-'*35} {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}")
for tag, label in baselines:
    mf = os.path.join(base, tag, 'metrics.json')
    if os.path.exists(mf):
        d = json.load(open(mf))
        print(f"  {label:<35} {d['fid']:>7.2f}  {d.get('clip', float('nan')):>7.3f}  {d['time_per_image_sec']:>6.2f}s  {d.get('speedup_est',1.0):>5.2f}x")

# Print K-sweep results
for K in (6, 8, 10, 12, 14, 16):
    tag = f'SVDQUANT_selective_k{K}_m$METRIC_steps$STEPS'
    mf  = os.path.join(base, tag, 'metrics.json')
    if os.path.exists(mf):
        d = json.load(open(mf))
        sb = sorted(d.get('skip_blocks') or [])
        label = f'selective K={K} {sb}'
        print(f"  {label:<35} {d['fid']:>7.2f}  {d.get('clip', float('nan')):>7.3f}  {d['time_per_image_sec']:>6.2f}s  {d.get('speedup_est',0):>5.2f}x")
    else:
        print(f"  selective K={K:<27} {'MISSING':>7}")
PY
echo "============================================================"
