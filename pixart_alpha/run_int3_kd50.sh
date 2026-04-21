#!/bin/bash
# INT3 weight + NVFP4 activation + KD-50 실험
# RPCA IALM PTQ with wgt_mode=INT3, act_mode=NVFP4, then KD-50
# NVFP4 결과(KD-50: FID=159.5, IS=1.827)와 비교

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
LOG_DIR="${BASE_DIR}/logs/int3_kd50_experiment"
mkdir -p "${LOG_DIR}"

if [ "${TEST_MODE:-0}" = "1" ]; then
  NUM_SAMPLES=2
  echo "[TEST MODE] NUM_SAMPLES=2"
else
  NUM_SAMPLES=20
fi

RUN_LOG="${LOG_DIR}/run.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

ACCEL_CMD="accelerate launch --multi_gpu --num_processes ${NUM_GPUS}"
PY="${BASE_DIR}/pixart_kd_finetune.py"

echo "============================================================"
echo "  INT3 Weight + NVFP4 Act + KD-50 Experiment"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  NUM_SAMPLES=${NUM_SAMPLES}"
echo "  wgt=INT3  act=NVFP4  KD=50steps"
echo "============================================================"

# ---- RPCA-INT3 no-KD (baseline comparison) ----
NOKD_SAVE="${BASE_DIR}/results/int3_kd50_experiment/RPCA_INT3_NOKD"
echo ""
echo "---- RPCA-INT3 no-KD  ($(date '+%H:%M:%S')) ----"
if [ -f "${NOKD_SAVE}/${DATASET}/metrics.json" ]; then
    echo "  [SKIP] already exists."
else
    ${ACCEL_CMD} "${PY}" \
        --quant_method RPCA \
        --act_mode NVFP4 \
        --wgt_mode INT3 \
        --lowrank 32 \
        --block_size 16 \
        --model_path "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir "${REF_DIR}" \
        --save_dir "${NOKD_SAVE}" \
        --num_samples ${NUM_SAMPLES} \
        2>&1 | tee "${LOG_DIR}/int3_nokd.log"
    echo "  [$(date '+%H:%M:%S')] INT3 no-KD done."
fi

# ---- RPCA-INT3 + KD-50 ----
KD50_SAVE="${BASE_DIR}/results/int3_kd50_experiment/RPCA_INT3_KD50"
echo ""
echo "---- RPCA-INT3 + KD-50  ($(date '+%H:%M:%S')) ----"
if [ -f "${KD50_SAVE}/${DATASET}/metrics.json" ]; then
    echo "  [SKIP] already exists."
else
    ${ACCEL_CMD} "${PY}" \
        --quant_method RPCA \
        --act_mode NVFP4 \
        --wgt_mode INT3 \
        --lowrank 32 \
        --block_size 16 \
        --model_path "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir "${REF_DIR}" \
        --save_dir "${KD50_SAVE}" \
        --num_samples ${NUM_SAMPLES} \
        --do_kd \
        --kd_steps 50 \
        --kd_lr 1e-4 \
        --kd_prompts 8 \
        2>&1 | tee "${LOG_DIR}/int3_kd50.log"
    echo "  [$(date '+%H:%M:%S')] INT3 KD-50 done."
fi

echo ""
echo "============================================================"
echo "  비교 결과"
echo "============================================================"

export BASE_DIR_PY="${BASE_DIR}"
export DATASET_PY="${DATASET}"

python3 - <<'PYEOF'
import json, os

base = os.environ["BASE_DIR_PY"]
ds   = os.environ["DATASET_PY"]

def load(path):
    return json.load(open(path)) if os.path.exists(path) else None

entries = [
    ("BASELINE (NVFP4_DEFAULT_CFG)",
     load(f"{base}/results/quip_experiment/BASELINE/{ds}/metrics.json")),
    ("RPCA NVFP4w  no-KD",
     load(f"{base}/results/kd_experiment/RPCA_NVFP4_NOKD/{ds}/metrics.json")),
    ("RPCA NVFP4w + KD-50",
     load(f"{base}/results/kd_experiment/RPCA_NVFP4_KD50/{ds}/metrics.json")),
    ("RPCA INT3w   no-KD",
     load(f"{base}/results/int3_kd50_experiment/RPCA_INT3_NOKD/{ds}/metrics.json")),
    ("RPCA INT3w + KD-50",
     load(f"{base}/results/int3_kd50_experiment/RPCA_INT3_KD50/{ds}/metrics.json")),
]

bl = next((d for n, d in entries if "BASELINE" in n and d), None)
bl_fid = bl["primary_metrics"]["FID"] if bl else None
bl_is  = bl["primary_metrics"]["IS"]  if bl else None

print(f"\n{'Config':<30} {'FID↓':>8} {'IS↑':>7} {'PSNR↑':>7} {'SSIM↑':>7}  vs BASELINE")
print("-" * 75)
for name, d in entries:
    if d is None:
        print(f"  {name:<28} {'N/A':>8}")
        continue
    pm = d["primary_metrics"]
    sm = d["secondary_metrics"]
    fid  = pm.get("FID", 0)
    is_v = pm.get("IS",  0)
    psnr = sm.get("PSNR", 0)
    ssim = sm.get("SSIM", 0)
    beat = ""
    if bl_fid and "BASELINE" not in name:
        bf = fid  < bl_fid
        bi = is_v > bl_is
        if bf and bi:  beat = "FID+IS ✓"
        elif bf:       beat = "FID ✓"
        elif bi:       beat = "IS ✓"
        else:          beat = "✗"
    psnr_s = f"{psnr:.1f}" if psnr != float("inf") else "∞"
    print(f"  {name:<28} {fid:>8.1f} {is_v:>7.4f} {psnr_s:>7} {ssim:>7.4f}  {beat}")
print()
PYEOF

echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
