#!/bin/bash
# run_partial_block_skip.sh — partial-block-skip 100-sample experiment
#   partial_attn     : cache self-attn (attn1) pre-gate, run attn2+ff fresh
#   partial_mlp      : cache ff pre-gate, run attn1+attn2 fresh
#   partial_attn_mlp : cache both attn1+ff pre-gate, run attn2 fresh only
#
# Baseline comparison (steps=20, n=100, existing results):
#   SVDQUANT no-cache : FID=121.32  tpi=2.85s  1.00×
#   DeepCache c8-20   : FID=129.14  tpi=2.33s  1.22×
#   nl_gelu drift     : FID=124.94  tpi=2.53s  1.13×
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

PYTHON=/home/jovyan/.dit/bin/python3
CS=8; CE=20; INT=2; METHOD=SVDQUANT
NUM=${NUM_SAMPLES:-100}
STEPS=${STEPS:-20}
export CUDA_VISIBLE_DEVICES=${GPU:-0}

echo "============================================================"
echo "  Partial Block Skip Experiment"
echo "  Method=$METHOD  Range=[$CS,$CE)  Interval=$INT"
echo "  Steps=$STEPS  Samples=$NUM  GPU=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

for MODE in partial_attn partial_mlp partial_attn_mlp; do
    TAG=${METHOD}_${MODE}_c${CS}-${CE}_steps${STEPS}
    LOG=logs/${TAG}.log
    echo ""
    echo "[$(date '+%H:%M:%S')] Starting $MODE → $LOG"
    $PYTHON pixart_nvfp4_cache_compare.py \
        --quant_method   $METHOD \
        --cache_mode     $MODE \
        --num_steps      $STEPS \
        --num_samples    $NUM \
        --cache_start    $CS \
        --cache_end      $CE \
        --deepcache_interval $INT \
        --guidance_scale 4.5 \
        > "$LOG" 2>&1
    echo "[$(date '+%H:%M:%S')] Done: $MODE"
done

echo ""
echo "============================================================"
echo "ALL DONE. Results:"
python3 -c "
import json, os
base='/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/SVDQUANT/MJHQ'
for tag in ['SVDQUANT_partial_attn_c8-20_steps${STEPS}',
            'SVDQUANT_partial_mlp_c8-20_steps${STEPS}',
            'SVDQUANT_partial_attn_mlp_c8-20_steps${STEPS}']:
    mf = os.path.join(base, tag, 'metrics.json')
    if os.path.exists(mf):
        d = json.load(open(mf))
        print(f'{tag}: FID={d[\"fid\"]:.2f}  CLIP={d[\"clip\"]:.3f}  tpi={d[\"time_per_image_sec\"]:.2f}s  n={d[\"num_samples\"]}')
    else:
        print(f'{tag}: metrics.json not found')
" STEPS=$STEPS
echo "============================================================"
