#!/bin/bash
# Tucker Inverted Experiment
# U1, U2T = INT4 (quantized) + FP16 scale (block_size=16)
# G core  = FP16 (no quantization)
# Rank ratio: 50% / 75% / 100%
# 50 samples, compare vs BASELINE and previous Tucker results

export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
if [ -f ~/.env ]; then export $(grep -v '^#' ~/.env | xargs); fi
if ! command -v accelerate &>/dev/null; then
  source /home/jameskimh/.dit/bin/activate
fi

NUM_GPUS=2
MODEL_PATH="PixArt-alpha/PixArt-XL-2-1024-MS"
DATASET="MJHQ"
REF_DIR="/data/jameskimh/james_dit_ref/ref_images_fp16"
BASE_DIR="$(pwd)"
LOG_DIR="${BASE_DIR}/logs/tucker_inv"
RESULT_DIR="${BASE_DIR}/results/tucker_inv"
mkdir -p "${LOG_DIR}"

if [ "${TEST_MODE:-0}" = "1" ]; then
  NUM_SAMPLES=2
  echo "[TEST MODE] NUM_SAMPLES=2"
else
  NUM_SAMPLES=50
fi

RUN_LOG="${LOG_DIR}/run.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

ACCEL_CMD="accelerate launch --multi_gpu --num_processes ${NUM_GPUS}"
PY="${BASE_DIR}/pixart_tucker_experiment.py"

echo "============================================================"
echo "  Tucker Inverted: U=INT4+FP16scale, G=FP16"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Rank ratio: 50% / 75% / 100%"
echo "  NUM_SAMPLES=${NUM_SAMPLES}"
echo "============================================================"

run_tucker_inv() {
    local RANK_RATIO=$1
    local RR_TAG="${RANK_RATIO//./p}"
    local TAG="TUCKER_INV_RR${RR_TAG}_UINT4_GFP16"
    local SAVE="${RESULT_DIR}/${TAG}"
    echo ""
    echo "---- ${TAG}  ($(date '+%H:%M:%S')) ----"
    if [ -f "${SAVE}/${DATASET}/metrics.json" ]; then
        echo "  [SKIP] already exists."
        return
    fi
    ${ACCEL_CMD} "${PY}" \
        --model_path    "${MODEL_PATH}" \
        --dataset_name  "${DATASET}" \
        --ref_dir       "${REF_DIR}" \
        --save_dir      "${SAVE}" \
        --num_samples   ${NUM_SAMPLES} \
        --rank_ratio    ${RANK_RATIO} \
        --wgt_u_mode    INT4 \
        --wgt_core_mode FP16 \
        --act_mode      NVFP4 \
        --block_size    16 \
        --lora_rank     32 \
        2>&1 | tee "${LOG_DIR}/${TAG}.log"
    echo "  [$(date '+%H:%M:%S')] ${TAG} done."
}

run_tucker_inv 0.50
run_tucker_inv 0.75
run_tucker_inv 1.00

# ── 비교표 ──────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  결과 비교"
echo "============================================================"

export BASE_DIR_PY="${BASE_DIR}"
export RESULT_DIR_PY="${RESULT_DIR}"
export DATASET_PY="${DATASET}"

python3 - <<'PYEOF'
import json, os

base    = os.environ["BASE_DIR_PY"]
resdir  = os.environ["RESULT_DIR_PY"]
ds      = os.environ["DATASET_PY"]

def load(p): return json.load(open(p)) if os.path.exists(p) else None

def row(name, d, bl_fid, bl_is):
    if d is None:
        return f"  {name:<38} {'N/A':>8}"
    pm   = d.get("primary_metrics", {})
    sm   = d.get("secondary_metrics", {})
    comp = d.get("compression", {})
    fid  = pm.get("FID", 0)
    is_v = pm.get("IS",  0)
    psnr = sm.get("PSNR", 0)
    ssim = sm.get("SSIM", 0)
    cx   = comp.get("compression_vs_nvfp4_tucker_only", "--")
    cx_s = f"{cx:.2f}x" if isinstance(cx, float) else "--"
    beat = ""
    if bl_fid:
        if fid < bl_fid and is_v > bl_is: beat = "FID+IS ✓"
        elif fid < bl_fid:                beat = "FID ✓"
        elif is_v > bl_is:                beat = "IS ✓"
        else:                             beat = "✗"
    psnr_s = f"{psnr:.1f}" if psnr else "--"
    return f"  {name:<38} {fid:>8.1f} {is_v:>7.4f} {psnr_s:>7} {ssim:>6.4f}  {cx_s:>8}  {beat}"

bl = load(f"{base}/results/quip_experiment/BASELINE/{ds}/metrics.json")
bl_fid = bl["primary_metrics"]["FID"] if bl else None
bl_is  = bl["primary_metrics"]["IS"]  if bl else None

entries = [
    ("BASELINE (SVDQuant NVFP4)",
     bl),
    # Previous Tucker: FP16 U + INT3 G (for reference)
    ("Tucker 75% FP16U+INT3G+LoRA",
     load(f"{base}/results/tucker_detailed/TUCKER_RR0p75_COREINT3_UFP16_BS16/{ds}/metrics.json")),
    ("Tucker 100% FP16U+INT3G+LoRA",
     load(f"{base}/results/tucker_detailed/TUCKER_RR1p00_COREINT3_UFP16_BS16/{ds}/metrics.json")),
    # New: INT4 U + FP16 G
    ("Tucker 50% INT4U+FP16G+LoRA",
     load(f"{resdir}/TUCKER_INV_RR0p50_UINT4_GFP16/{ds}/metrics.json")),
    ("Tucker 75% INT4U+FP16G+LoRA",
     load(f"{resdir}/TUCKER_INV_RR0p75_UINT4_GFP16/{ds}/metrics.json")),
    ("Tucker 100% INT4U+FP16G+LoRA",
     load(f"{resdir}/TUCKER_INV_RR1p00_UINT4_GFP16/{ds}/metrics.json")),
]

print(f"\n  {'Config':<38} {'FID↓':>8} {'IS↑':>7} {'PSNR↑':>7} {'SSIM↑':>6}  {'vs NVFP4':>8}  beat")
print("  " + "-" * 82)
for name, d in entries:
    print(row(name, d, bl_fid, bl_is))

# bit 비교
print(f"\n  [ Bit-level 비교 (4608×1152, block_size=16) ]")
print(f"  {'Config':<30} {'U bits':>8} {'G bits':>8} {'total(MB)':>10}")
print("  " + "-" * 60)

import math
def eff(data_b, scale_b, bs): return (data_b * bs + scale_b) / bs
m, n, bs = 4608, 1152, 16
lr = 32

configs = [
    ("NVFP4+S8 (SVDQuant)",  4.5,  4.5,  None),
    ("FP16U + INT3+S16G",    16.0, eff(3,16,bs), None),
    ("INT4+S16U + FP16G",    eff(4,16,bs), 16.0, None),
]

for rank_r in [0.5, 0.75, 1.0]:
    r = int(rank_r * min(m, n))
    for cfg_name, u_b, g_b, _ in configs:
        u1_bits = m * r * u_b
        g_bits  = r * r * g_b
        u2_bits = r * n * u_b
        lora_bits = (lr*n + m*lr) * 16
        total_mb = (u1_bits + g_bits + u2_bits + lora_bits) / 8 / 1024 / 1024
        if rank_r == 0.75 or (rank_r == 1.0 and "NVFP4" not in cfg_name):
            label = f"rank={int(rank_r*100)}% {cfg_name}"
            print(f"  {label:<30} {u_b:>8.1f} {g_b:>8.1f} {total_mb:>9.2f} MB")

fp16_mb = m * n * 16 / 8 / 1024 / 1024
svd_mb = (m*n*4.5 + lr*n*16 + m*lr*16) / 8 / 1024 / 1024
print(f"\n  FP16 원본:        {fp16_mb:.2f} MB")
print(f"  SVDQuant NVFP4:   {svd_mb:.2f} MB")
print()
PYEOF

echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
