#!/bin/bash
# run_selective_partial_skip.sh — selective + partial block skip experiment
#
# 4 runs: selective_partial_{attn,attn_mlp} × K={6,8}
# GPU0: attn K=6,8  |  GPU1: attn_mlp K=6,8  (parallel ~20 min)
#
# Reuses existing sensitivity JSON (Metric A ranking, K=6 → {5,19,21,23,24,25}).
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

if [ ! -f "$SENS_JSON" ]; then
    echo "ERROR: sensitivity JSON not found: $SENS_JSON"
    echo "Run run_selective_skip.sh (Phase 0) first."
    exit 1
fi

run_sp () {
    local GPU=$1
    local MODE=$2
    local K=$3
    local TAG=SVDQUANT_selective_partial_${MODE}_k${K}_m${METRIC}_c8-20_steps${STEPS}
    local LOG=logs/${TAG}.log
    echo "[$(date '+%H:%M:%S')] GPU${GPU}: selective_partial_${MODE} K=${K} → $LOG"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON pixart_nvfp4_cache_compare.py \
        --quant_method SVDQUANT \
        --cache_mode selective_partial_${MODE} \
        --num_steps $STEPS \
        --num_samples $NUM \
        --sensitivity_json "$SENS_JSON" \
        --skip_k $K \
        --sensitivity_metric $METRIC \
        --deepcache_interval $INT \
        --guidance_scale 4.5 \
        > "$LOG" 2>&1
    echo "[$(date '+%H:%M:%S')] GPU${GPU}: selective_partial_${MODE} K=${K} done"
}

echo "============================================================"
echo "  Selective + Partial Skip Experiment"
echo "  Steps=$STEPS  Samples=$NUM  Metric=$METRIC  Interval=$INT"
echo "============================================================"
echo ""

# GPU0: attn K=6,8  ;  GPU1: attn_mlp K=6,8  (parallel)
(
    run_sp 0 attn 6
    run_sp 0 attn 8
) &
PID0=$!
(
    run_sp 1 attn_mlp 6
    run_sp 1 attn_mlp 8
) &
PID1=$!

wait $PID0 $PID1
echo "[$(date '+%H:%M:%S')] All 4 runs complete."

# ── Aggregate results ─────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Results (steps=$STEPS, n=$NUM, metric=$METRIC)"
echo "============================================================"
$PYTHON - <<PY
import json, os
base = '/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/SVDQUANT/MJHQ'
rows = [
    ('SVDQUANT_none_steps20',                                        'no-cache (baseline)'),
    ('SVDQUANT_deepcache_c8-20_steps20',                             'DeepCache c8-20 (K=12 full)'),
    ('SVDQUANT_partial_attn_mlp_c8-20_steps20',                      'Partial attn_mlp c8-20 (K=12)'),
    ('SVDQUANT_selective_k6_mA_c8-20_steps20',                       'Selective K=6 (full block)'),
    ('SVDQUANT_selective_partial_attn_k6_mA_c8-20_steps20',          'Selective+Partial attn K=6'),
    ('SVDQUANT_selective_partial_attn_k8_mA_c8-20_steps20',          'Selective+Partial attn K=8'),
    ('SVDQUANT_selective_partial_attn_mlp_k6_mA_c8-20_steps20',      'Selective+Partial attn_mlp K=6'),
    ('SVDQUANT_selective_partial_attn_mlp_k8_mA_c8-20_steps20',      'Selective+Partial attn_mlp K=8'),
]
print(f"  {'Method':<45} {'FID':>7}  {'CLIP':>7}  {'tpi':>6}  {'spdup':>6}")
print(f"  {'-'*45} {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}")
for tag, label in rows:
    p = os.path.join(base, tag, 'metrics.json')
    if os.path.exists(p):
        d = json.load(open(p))
        print(f"  {label:<45} {d['fid']:>7.2f}  {d.get('clip', float('nan')):>7.3f}  {d['time_per_image_sec']:>6.2f}s  {d.get('speedup_est', 0):>5.2f}x")
    else:
        print(f"  {label:<45} {'MISSING':>7}")
PY
echo "============================================================"
