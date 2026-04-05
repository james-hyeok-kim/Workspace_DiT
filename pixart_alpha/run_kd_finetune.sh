#!/bin/bash
# ============================================================
# Fast KD Fine-tuning Ablation
#
# Base PTQ configs:
#   RPCA-NVFP4 (IALM auto-λ)   ← proper RPCA with sparse branch
#   SVD-NVFP4                   (비교용)
#
# KD ablation:
#   no-KD : PTQ only (RPCA IALM, proper sparse branch)
#   KD-30  : 30 steps  (~1 min)
#   KD-50  : 50 steps  (~2 min)
#   KD-70  : 70 steps  (~3 min)
#   KD-100 : 100 steps (~4 min)  ← 상한선
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
BLOCK_SIZE=16
LOWRANK=32
REF_DIR="/data/james_dit_ref/ref_images_fp16"
BASE_DIR="$(pwd)"
LOG_DIR="${BASE_DIR}/logs/kd_experiment"
mkdir -p "${LOG_DIR}"

# TEST_MODE=1 → 2 samples smoke test
if [ "${TEST_MODE:-0}" = "1" ]; then
  NUM_SAMPLES=2
  KD_STEPS_LIST="10"
  echo "[TEST MODE] NUM_SAMPLES=2, KD_STEPS=10"
else
  NUM_SAMPLES=20
  KD_STEPS_LIST="30 50 70 100"
fi

RUN_LOG="${LOG_DIR}/run.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

ACCEL_CMD="accelerate launch --multi_gpu --num_processes ${NUM_GPUS}"
PY="${BASE_DIR}/pixart_kd_finetune.py"

echo "============================================================"
echo "  Fast KD Fine-tuning Ablation"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  NUM_SAMPLES=${NUM_SAMPLES}  LOWRANK=${LOWRANK}  RPCA=IALM(auto)"
echo "============================================================"

# ---- RPCA-NVFP4 no-KD (PTQ only, proper IALM sparse branch) ----
NOKD_SAVE="${BASE_DIR}/results/kd_experiment/RPCA_NVFP4_NOKD"
echo ""
echo "---- RPCA-NVFP4 no-KD  ($(date '+%H:%M:%S')) ----"
if [ -f "${NOKD_SAVE}/${DATASET}/metrics.json" ]; then
    echo "  [SKIP] already exists."
else
    ${ACCEL_CMD} "${PY}" \
        --quant_method RPCA \
        --act_mode NVFP4 \
        --wgt_mode NVFP4 \
        --lowrank ${LOWRANK} \
        --block_size ${BLOCK_SIZE} \
        --model_path "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir "${REF_DIR}" \
        --save_dir "${NOKD_SAVE}" \
        --num_samples ${NUM_SAMPLES} \
        2>&1 | tee "${LOG_DIR}/rpca_nokd.log"
    echo "  [$(date '+%H:%M:%S')] RPCA no-KD done."
fi

# ---- RPCA-NVFP4 + KD ablation ----
for KD_STEPS in ${KD_STEPS_LIST}; do
    SAVE="${BASE_DIR}/results/kd_experiment/RPCA_NVFP4_KD${KD_STEPS}"
    echo ""
    echo "---- RPCA-NVFP4 + KD-${KD_STEPS}  ($(date '+%H:%M:%S')) ----"
    if [ -f "${SAVE}/${DATASET}/metrics.json" ]; then
        echo "  [SKIP] already exists."
    else
        ${ACCEL_CMD} "${PY}" \
            --quant_method RPCA \
            --act_mode NVFP4 \
            --wgt_mode NVFP4 \
            --lowrank ${LOWRANK} \
            --block_size ${BLOCK_SIZE} \
            --model_path "${MODEL_PATH}" \
            --dataset_name "${DATASET}" \
            --ref_dir "${REF_DIR}" \
            --save_dir "${SAVE}" \
            --num_samples ${NUM_SAMPLES} \
            --do_kd \
            --kd_steps ${KD_STEPS} \
            --kd_lr 1e-4 \
            --kd_prompts 8 \
            2>&1 | tee "${LOG_DIR}/rpca_kd${KD_STEPS}.log"
        echo "  [$(date '+%H:%M:%S')] RPCA KD-${KD_STEPS} done."
    fi
done

# ---- SVD-NVFP4 + KD-50 (비교용) ----
SVD_KD_STEPS=50
SVD_SAVE="${BASE_DIR}/results/kd_experiment/SVD_NVFP4_KD${SVD_KD_STEPS}"
echo ""
echo "---- SVD-NVFP4 + KD-${SVD_KD_STEPS}  ($(date '+%H:%M:%S')) ----"
if [ -f "${SVD_SAVE}/${DATASET}/metrics.json" ]; then
    echo "  [SKIP] already exists."
else
    ${ACCEL_CMD} "${PY}" \
        --quant_method SVD \
        --act_mode NVFP4 \
        --wgt_mode NVFP4 \
        --lowrank ${LOWRANK} \
        --block_size ${BLOCK_SIZE} \
        --model_path "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir "${REF_DIR}" \
        --save_dir "${SVD_SAVE}" \
        --num_samples ${NUM_SAMPLES} \
        --do_kd \
        --kd_steps ${SVD_KD_STEPS} \
        --kd_lr 1e-4 \
        --kd_prompts 8 \
        2>&1 | tee "${LOG_DIR}/svd_kd${SVD_KD_STEPS}.log"
    echo "  [$(date '+%H:%M:%S')] SVD KD-${SVD_KD_STEPS} done."
fi

echo ""
echo "============================================================"
echo "  통합 비교표: KD Ablation 결과"
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
    ("RPCA-NVFP4 no-KD (IALM)",
     load(f"{base}/results/kd_experiment/RPCA_NVFP4_NOKD/{ds}/metrics.json")),
    ("RPCA-NVFP4 + KD-30",
     load(f"{base}/results/kd_experiment/RPCA_NVFP4_KD30/{ds}/metrics.json")),
    ("RPCA-NVFP4 + KD-50",
     load(f"{base}/results/kd_experiment/RPCA_NVFP4_KD50/{ds}/metrics.json")),
    ("RPCA-NVFP4 + KD-70",
     load(f"{base}/results/kd_experiment/RPCA_NVFP4_KD70/{ds}/metrics.json")),
    ("RPCA-NVFP4 + KD-100",
     load(f"{base}/results/kd_experiment/RPCA_NVFP4_KD100/{ds}/metrics.json")),
    ("SVD-NVFP4 + KD-50",
     load(f"{base}/results/kd_experiment/SVD_NVFP4_KD50/{ds}/metrics.json")),
]

bl = next((d for n, d in entries if "BASELINE" in n and d), None)
bl_fid = bl["primary_metrics"]["FID"] if bl else None
bl_is  = bl["primary_metrics"]["IS"]  if bl else None

print(f"\n{'Config':<38} {'FID↓':>8} {'IS↑':>7} {'PSNR↑':>7} {'SSIM↑':>7}  vs BASELINE")
print("-" * 82)
for name, d in entries:
    if d is None:
        print(f"  {name:<36} {'N/A':>8}")
        continue
    pm = d["primary_metrics"]
    sm = d["secondary_metrics"]
    fid  = pm.get("FID", 0)
    is_v = pm.get("IS",  0)
    psnr = sm.get("PSNR", 0)
    ssim = sm.get("SSIM", 0)
    kd_info = ""
    if "kd_loss" in d:
        kd_info = f"  [KD {d['kd_loss']['initial']:.4f}→{d['kd_loss']['final']:.4f}]"
    beat = ""
    if bl_fid and "BASELINE" not in name:
        bf = fid  < bl_fid
        bi = is_v > bl_is
        if bf and bi:  beat = "FID+IS ✓"
        elif bf:       beat = "FID ✓"
        elif bi:       beat = "IS ✓"
        else:          beat = "✗"
    psnr_s = f"{psnr:.1f}" if psnr != float("inf") else "∞"
    print(f"  {name:<36} {fid:>8.1f} {is_v:>7.4f} {psnr_s:>7} {ssim:>7.4f}  {beat}{kd_info}")
print()

# Save summary JSON
results_dir = f"{base}/results/kd_experiment"
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
        if "kd_loss" in d:
            summary[name]["kd_loss"] = d["kd_loss"]
summary_path = f"{results_dir}/results_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)
print(f"Summary saved: {summary_path}")
PYEOF

echo ""
echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
