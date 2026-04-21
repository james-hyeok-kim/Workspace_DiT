#!/bin/bash
# ============================================================
# Timestep-Aware PTQ NVFP4 Weight + NVFP4 Activation Ablation
#   A2: TSAWARE G3 no correction
#   A3: TSAWARE G5 no correction
#   A4: TSAWARE G3 + rank-4 per-group SVD correction
#   A5: TSAWARE G5 + rank-4 per-group SVD correction
#
#  wgt_bits=NVFP4, act_mode=NVFP4  (기존 INT4a → NVFP4a)
# ============================================================

export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
if [ -f ~/.env ]; then export $(grep -v '^#' ~/.env | xargs); fi
if ! command -v accelerate &>/dev/null; then
  source /home/jameskimh/.dit/bin/activate
fi

NUM_GPUS=2
MODEL_PATH="PixArt-alpha/PixArt-XL-2-1024-MS"
DATASET="MJHQ"
BLOCK_SIZE=128
LOWRANK=32
RANK_T=4
REF_DIR="/data/jameskimh/james_dit_ref/ref_images_fp16"
BASE_DIR="$(pwd)"
LOG_DIR="${BASE_DIR}/logs/ts_aware_nvfp4a_experiment"
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
PY="${BASE_DIR}/pixart_ts_aware_experiment.py"

echo "============================================================"
echo "  Timestep-Aware PTQ NVFP4w + NVFP4a Ablation"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  NUM_SAMPLES=${NUM_SAMPLES}  LOWRANK=${LOWRANK}  RANK_T=${RANK_T}"
echo "  wgt=NVFP4  act=NVFP4"
echo "============================================================"

# ---- [A2] TSAWARE G3, no correction ----
A2_SAVE="${BASE_DIR}/results/ts_aware_nvfp4a_experiment/TSAWARE_G3_NOCORR"
echo ""
echo "---- [A2] TSAWARE G3 no-correction  ($(date '+%H:%M:%S')) ----"
if [ -f "${A2_SAVE}/${DATASET}/metrics.json" ]; then
    echo "  [SKIP] already exists."
else
    ${ACCEL_CMD} "${PY}" \
        --quant_method TSAWARE \
        --n_groups 3 \
        --lowrank ${LOWRANK} \
        --block_size ${BLOCK_SIZE} \
        --wgt_bits NVFP4 \
        --act_mode NVFP4 \
        --model_path "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir "${REF_DIR}" \
        --save_dir "${A2_SAVE}" \
        --num_samples ${NUM_SAMPLES} \
        2>&1 | tee "${LOG_DIR}/a2_g3_nocorr.log"
    echo "  [$(date '+%H:%M:%S')] A2 done."
fi

# ---- [A3] TSAWARE G5, no correction ----
A3_SAVE="${BASE_DIR}/results/ts_aware_nvfp4a_experiment/TSAWARE_G5_NOCORR"
echo ""
echo "---- [A3] TSAWARE G5 no-correction  ($(date '+%H:%M:%S')) ----"
if [ -f "${A3_SAVE}/${DATASET}/metrics.json" ]; then
    echo "  [SKIP] already exists."
else
    ${ACCEL_CMD} "${PY}" \
        --quant_method TSAWARE \
        --n_groups 5 \
        --lowrank ${LOWRANK} \
        --block_size ${BLOCK_SIZE} \
        --wgt_bits NVFP4 \
        --act_mode NVFP4 \
        --model_path "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir "${REF_DIR}" \
        --save_dir "${A3_SAVE}" \
        --num_samples ${NUM_SAMPLES} \
        2>&1 | tee "${LOG_DIR}/a3_g5_nocorr.log"
    echo "  [$(date '+%H:%M:%S')] A3 done."
fi

# ---- [A4] TSAWARE G3 + rank-4 correction ----
A4_SAVE="${BASE_DIR}/results/ts_aware_nvfp4a_experiment/TSAWARE_G3_RANK4"
echo ""
echo "---- [A4] TSAWARE G3 + rank-4 correction  ($(date '+%H:%M:%S')) ----"
if [ -f "${A4_SAVE}/${DATASET}/metrics.json" ]; then
    echo "  [SKIP] already exists."
else
    ${ACCEL_CMD} "${PY}" \
        --quant_method TSAWARE \
        --n_groups 3 \
        --use_ts_correction \
        --rank_t ${RANK_T} \
        --lowrank ${LOWRANK} \
        --block_size ${BLOCK_SIZE} \
        --wgt_bits NVFP4 \
        --act_mode NVFP4 \
        --model_path "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir "${REF_DIR}" \
        --save_dir "${A4_SAVE}" \
        --num_samples ${NUM_SAMPLES} \
        2>&1 | tee "${LOG_DIR}/a4_g3_rank4.log"
    echo "  [$(date '+%H:%M:%S')] A4 done."
fi

# ---- [A5] TSAWARE G5 + rank-4 correction ----
A5_SAVE="${BASE_DIR}/results/ts_aware_nvfp4a_experiment/TSAWARE_G5_RANK4"
echo ""
echo "---- [A5] TSAWARE G5 + rank-4 correction  ($(date '+%H:%M:%S')) ----"
if [ -f "${A5_SAVE}/${DATASET}/metrics.json" ]; then
    echo "  [SKIP] already exists."
else
    ${ACCEL_CMD} "${PY}" \
        --quant_method TSAWARE \
        --n_groups 5 \
        --use_ts_correction \
        --rank_t ${RANK_T} \
        --lowrank ${LOWRANK} \
        --block_size ${BLOCK_SIZE} \
        --wgt_bits NVFP4 \
        --act_mode NVFP4 \
        --model_path "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir "${REF_DIR}" \
        --save_dir "${A5_SAVE}" \
        --num_samples ${NUM_SAMPLES} \
        2>&1 | tee "${LOG_DIR}/a5_g5_rank4.log"
    echo "  [$(date '+%H:%M:%S')] A5 done."
fi

echo ""
echo "============================================================"
echo "  통합 비교표: NVFP4a Ablation 결과"
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
    # INT4a 기존 결과 (비교용)
    ("A2 G3 no-corr (NVFPw-INT4a)",
     load(f"{base}/results/ts_aware_experiment/TSAWARE_G3_NOCORR/{ds}/metrics.json")),
    ("A5 G5 rank4   (NVFPw-INT4a)",
     load(f"{base}/results/ts_aware_experiment/TSAWARE_G5_RANK4/{ds}/metrics.json")),
    # NVFP4a 새 결과
    ("A2 G3 no-corr (NVFPw-NVFPa)",
     load(f"{base}/results/ts_aware_nvfp4a_experiment/TSAWARE_G3_NOCORR/{ds}/metrics.json")),
    ("A3 G5 no-corr (NVFPw-NVFPa)",
     load(f"{base}/results/ts_aware_nvfp4a_experiment/TSAWARE_G5_NOCORR/{ds}/metrics.json")),
    ("A4 G3 rank4   (NVFPw-NVFPa)",
     load(f"{base}/results/ts_aware_nvfp4a_experiment/TSAWARE_G3_RANK4/{ds}/metrics.json")),
    ("A5 G5 rank4   (NVFPw-NVFPa)",
     load(f"{base}/results/ts_aware_nvfp4a_experiment/TSAWARE_G5_RANK4/{ds}/metrics.json")),
]

bl = next((d for n, d in entries if "BASELINE" in n and d), None)
bl_fid = bl["primary_metrics"]["FID"] if bl else None
bl_is  = bl["primary_metrics"]["IS"]  if bl else None

print(f"\n{'Config':<42} {'FID↓':>8} {'IS↑':>7} {'PSNR↑':>7} {'SSIM↑':>7}  vs BASELINE")
print("-" * 86)
for name, d in entries:
    if d is None:
        print(f"  {name:<40} {'N/A':>8}")
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
    print(f"  {name:<40} {fid:>8.1f} {is_v:>7.4f} {psnr_s:>7} {ssim:>7.4f}  {beat}")
print()

results_dir = f"{base}/results/ts_aware_nvfp4a_experiment"
os.makedirs(results_dir, exist_ok=True)
summary = {}
for name, d in entries:
    if d is not None:
        summary[name] = {
            "FID":  d["primary_metrics"].get("FID"),
            "IS":   d["primary_metrics"].get("IS"),
            "PSNR": d["secondary_metrics"].get("PSNR"),
            "SSIM": d["secondary_metrics"].get("SSIM"),
            "beats_baseline_FID": d["primary_metrics"].get("FID", 9999) < bl_fid if bl_fid else None,
            "beats_baseline_IS":  d["primary_metrics"].get("IS", 0) > bl_is if bl_is else None,
        }
summary_path = f"{results_dir}/results_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)
print(f"Summary saved: {summary_path}")
PYEOF

echo ""
echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
