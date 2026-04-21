#!/bin/bash
# ============================================================
# NVFP4w+NVFP4a: QuIP vs GPTQ 비교
# 기존 결과(BASELINE, RPCA, GPTQ-W4A4, QuIP-W4A4)와 통합 비교표 출력
# ============================================================

export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
if [ -f ~/.env ]; then export $(grep -v '^#' ~/.env | xargs); fi
if ! command -v accelerate &>/dev/null; then
  source /home/jameskimh/.dit/bin/activate
fi

NUM_GPUS=2
NUM_SAMPLES=20
MODEL_PATH="PixArt-alpha/PixArt-XL-2-1024-MS"
DATASET="MJHQ"
BLOCK_SIZE=128
LOWRANK=32
REF_DIR="/data/jameskimh/james_dit_ref/ref_images_fp16"
BASE_DIR="$(pwd)"
LOG_DIR="${BASE_DIR}/logs/nvfp4_compare"
mkdir -p "${LOG_DIR}"

RUN_LOG="${LOG_DIR}/run.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

ACCEL_CMD="accelerate launch --multi_gpu --num_processes ${NUM_GPUS}"

echo "============================================================"
echo "  NVFP4w+NVFP4a: QuIP vs GPTQ"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ---- [1/2] QuIP NVFP4w + NVFP4a ----
QUIP_NVFP4_SAVE="${BASE_DIR}/results/quip_experiment/NVFP4"
echo ""
echo "---- [1/2] QuIP-NVFP4 (NVFP4w + NVFP4a, rank=${LOWRANK})  ($(date '+%H:%M:%S')) ----"
if [ -f "${QUIP_NVFP4_SAVE}/${DATASET}/metrics.json" ]; then
    echo "  [SKIP] already exists."
else
    ${ACCEL_CMD} "${BASE_DIR}/pixart_quip_experiment.py" \
        --quant_method NVFP4 \
        --block_size ${BLOCK_SIZE} \
        --lowrank ${LOWRANK} \
        --model_path "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir "${REF_DIR}" \
        --save_dir "${QUIP_NVFP4_SAVE}" \
        --num_samples ${NUM_SAMPLES} \
        2>&1 | tee "${LOG_DIR}/quip_nvfp4.log"
    echo "  [$(date '+%H:%M:%S')] QuIP-NVFP4 done."
fi

# ---- [2/2] GPTQ NVFP4w + NVFP4a ----
GPTQ_NVFP4_SAVE="${BASE_DIR}/results/w3a4_experiment/GPTQ_NVFP4"
echo ""
echo "---- [2/2] GPTQ-NVFP4 (NVFP4w + NVFP4a, rank=${LOWRANK})  ($(date '+%H:%M:%S')) ----"
if [ -f "${GPTQ_NVFP4_SAVE}/${DATASET}/metrics.json" ]; then
    echo "  [SKIP] already exists."
else
    ${ACCEL_CMD} "${BASE_DIR}/pixart_w3a4_experiment.py" \
        --quant_method W3A4 \
        --wgt_bits NVFP4 \
        --act_mode NVFP4 \
        --block_size ${BLOCK_SIZE} \
        --lowrank ${LOWRANK} \
        --model_path "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir "${REF_DIR}" \
        --save_dir "${GPTQ_NVFP4_SAVE}" \
        --num_samples ${NUM_SAMPLES} \
        2>&1 | tee "${LOG_DIR}/gptq_nvfp4.log"
    echo "  [$(date '+%H:%M:%S')] GPTQ-NVFP4 done."
fi

echo ""
echo "============================================================"
echo "  통합 비교표: 전체 방법 비교"
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
    ("FP16",
     load(f"{base}/results/quip_experiment/FP16/{ds}/metrics.json")),
    ("NVFP4-BASELINE",
     load(f"{base}/results/quip_experiment/BASELINE/{ds}/metrics.json")),
    ("RPCA-INT8w-INT8a (OR=0.1)",
     load(f"{base}/results/rpca_sweep/RPCA_AINT8_WINT8_OR0.1/{ds}/metrics.json")),
    ("RPCA-NVFp4w-NVFp4a (OR=0.1)",
     load(f"{base}/results/rpca_sweep/RPCA_ANVFP4_WNVFP4_OR0.1/{ds}/metrics.json")),
    ("GPTQ-NVFp4w-INT4a",
     load(f"{base}/results/w3a4_experiment/W3A4/{ds}/metrics.json")),
    ("GPTQ-NVFp4w-NVFp4a [NEW]",
     load(f"{base}/results/w3a4_experiment/GPTQ_NVFP4/{ds}/metrics.json")),
    ("QuIP-INT4w-INT4a",
     load(f"{base}/results/quip_experiment/W4A4/{ds}/metrics.json")),
    ("QuIP-NVFp4w-NVFp4a [NEW]",
     load(f"{base}/results/quip_experiment/NVFP4/{ds}/metrics.json")),
]

bl = next((d for n, d in entries if "BASELINE" in n and d), None)
bl_fid = bl["primary_metrics"]["FID"] if bl else None
bl_is  = bl["primary_metrics"]["IS"]  if bl else None

print(f"\n{'Method':<38} {'FID↓':>8} {'IS↑':>7} {'PSNR↑':>7} {'SSIM↑':>7}  vs BASELINE")
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
    beat = ""
    if bl_fid and "BASELINE" not in name and "FP16" not in name:
        bf = fid  < bl_fid
        bi = is_v > bl_is
        if bf and bi:  beat = "FID+IS ✓"
        elif bf:       beat = "FID ✓"
        elif bi:       beat = "IS ✓"
        else:          beat = "✗"
    psnr_s = f"{psnr:.1f}" if psnr != float("inf") else "∞"
    print(f"  {name:<36} {fid:>8.1f} {is_v:>7.4f} {psnr_s:>7} {ssim:>7.4f}  {beat}")
print()
PYEOF

echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
