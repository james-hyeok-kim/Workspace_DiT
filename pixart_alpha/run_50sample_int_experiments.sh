#!/bin/bash
# 50-sample comparison: BASELINE + Tucker + CP + K-SVD (INT8/INT4/INT3)
# block_size=16 for all INT quantization

export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
if [ -f ~/.env ]; then export $(grep -v '^#' ~/.env | xargs); fi
if ! command -v accelerate &>/dev/null; then
  source /home/jameskimh/.dit/bin/activate
fi

NUM_GPUS=2
MODEL_PATH="PixArt-alpha/PixArt-XL-2-1024-MS"
DATASET="MJHQ"
REF_DIR="/data/james_dit_ref/ref_images_fp16"
BASE_DIR="$(pwd)"
NUM_SAMPLES=50

LOG_DIR="${BASE_DIR}/logs/int50_experiment"
mkdir -p "${LOG_DIR}"

RUN_LOG="${LOG_DIR}/run.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

ACCEL_CMD="accelerate launch --multi_gpu --num_processes ${NUM_GPUS}"

echo "============================================================"
echo "  50-Sample INT Quantization Comparison"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Tucker R64 INT8/INT4/INT3 | CP R32 INT8/INT4/INT3 | K-SVD INT8/INT4/INT3"
echo "============================================================"

# ---- 1. BASELINE (NVFP4_DEFAULT_CFG) at 50 samples ----
BASELINE_SAVE="${BASE_DIR}/results/quip_experiment/BASELINE"
echo ""
echo "==== BASELINE (NVFP4_DEFAULT_CFG, 50 samples) ===="
if [ -f "${BASELINE_SAVE}/${DATASET}/metrics.json" ]; then
    echo "  [SKIP] already exists."
else
    ${ACCEL_CMD} "${BASE_DIR}/pixart_quip_experiment.py" \
        --model_path   "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir      "${REF_DIR}" \
        --save_dir     "${BASELINE_SAVE}" \
        --num_samples  ${NUM_SAMPLES} \
        --quant_method BASELINE \
        2>&1 | tee "${LOG_DIR}/BASELINE.log"
    echo "  [$(date '+%H:%M:%S')] BASELINE done."
fi

# ---- 2. Tucker R64 INT8/INT4/INT3 ----
run_tucker() {
    local RANK=$1
    local CORE_MODE=$2
    local TAG="TUCKER_R${RANK}_CORE${CORE_MODE}"
    local SAVE="${BASE_DIR}/results/tucker_experiment/${TAG}"
    echo ""
    echo "---- Tucker ${TAG}  ($(date '+%H:%M:%S')) ----"
    if [ -f "${SAVE}/${DATASET}/metrics.json" ]; then
        echo "  [SKIP] already exists."
        return
    fi
    ${ACCEL_CMD} "${BASE_DIR}/pixart_tucker_experiment.py" \
        --model_path   "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir      "${REF_DIR}" \
        --save_dir     "${SAVE}" \
        --num_samples  ${NUM_SAMPLES} \
        --lowrank      ${RANK} \
        --lora_rank    32 \
        --act_mode     NVFP4 \
        --wgt_core_mode ${CORE_MODE} \
        --block_size   16 \
        2>&1 | tee "${LOG_DIR}/${TAG}.log"
    echo "  [$(date '+%H:%M:%S')] ${TAG} done."
}

echo ""
echo "==== Tucker R64 sweeps ===="
run_tucker 64 INT8
run_tucker 64 INT4
run_tucker 64 INT3

# ---- 3. CP R32 INT8/INT4/INT3 ----
run_cp() {
    local RANK=$1
    local WGT_MODE=$2
    local TAG="CP_R${RANK}_WGT${WGT_MODE}"
    local SAVE="${BASE_DIR}/results/cp_experiment/${TAG}"
    echo ""
    echo "---- CP ${TAG}  ($(date '+%H:%M:%S')) ----"
    if [ -f "${SAVE}/${DATASET}/metrics.json" ]; then
        echo "  [SKIP] already exists."
        return
    fi
    ${ACCEL_CMD} "${BASE_DIR}/pixart_cp_experiment.py" \
        --model_path   "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir      "${REF_DIR}" \
        --save_dir     "${SAVE}" \
        --num_samples  ${NUM_SAMPLES} \
        --lowrank      ${RANK} \
        --act_mode     NVFP4 \
        --wgt_mode     ${WGT_MODE} \
        --block_size   16 \
        2>&1 | tee "${LOG_DIR}/${TAG}.log"
    echo "  [$(date '+%H:%M:%S')] ${TAG} done."
}

echo ""
echo "==== CP R32 sweeps ===="
run_cp 32 INT8
run_cp 32 INT4
run_cp 32 INT3

# ---- 4. K-SVD K=256, s=16, dict=INT8/INT4/INT3 ----
run_ksvd() {
    local DICT_MODE=$1
    local K=$2
    local S=$3
    local TAG="KSVD_K${K}_S${S}_D${DICT_MODE}"
    local SAVE="${BASE_DIR}/results/ksvd_experiment/${TAG}"
    echo ""
    echo "---- K-SVD ${TAG}  ($(date '+%H:%M:%S')) ----"
    if [ -f "${SAVE}/${DATASET}/metrics.json" ]; then
        echo "  [SKIP] already exists."
        return
    fi
    ${ACCEL_CMD} "${BASE_DIR}/pixart_ksvd_experiment.py" \
        --model_path   "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir      "${REF_DIR}" \
        --save_dir     "${SAVE}" \
        --num_samples  ${NUM_SAMPLES} \
        --act_mode     NVFP4 \
        --dict_mode    ${DICT_MODE} \
        --wgt_reconstruct_mode NVFP4 \
        --dict_size    ${K} \
        --sparsity     ${S} \
        --ksvd_iters   10 \
        --block_size   16 \
        2>&1 | tee "${LOG_DIR}/${TAG}.log"
    echo "  [$(date '+%H:%M:%S')] ${TAG} done."
}

echo ""
echo "==== K-SVD K=256 s=16 sweeps ===="
run_ksvd INT8  256 16
run_ksvd INT4  256 16
run_ksvd INT3  256 16

# ---- Final comparison table ----
echo ""
echo "============================================================"
echo "  50-Sample Comparison: Tucker / CP / K-SVD vs BASELINE"
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
    ("BASELINE (NVFP4_DEFAULT)",
     load(f"{base}/results/quip_experiment/BASELINE/{ds}/metrics.json")),
    ("Tucker R64 core=INT8",
     load(f"{base}/results/tucker_experiment/TUCKER_R64_COREINT8/{ds}/metrics.json")),
    ("Tucker R64 core=INT4",
     load(f"{base}/results/tucker_experiment/TUCKER_R64_COREINT4/{ds}/metrics.json")),
    ("Tucker R64 core=INT3",
     load(f"{base}/results/tucker_experiment/TUCKER_R64_COREINT3/{ds}/metrics.json")),
    ("CP R32 wgt=INT8",
     load(f"{base}/results/cp_experiment/CP_R32_WGTINT8/{ds}/metrics.json")),
    ("CP R32 wgt=INT4",
     load(f"{base}/results/cp_experiment/CP_R32_WGTINT4/{ds}/metrics.json")),
    ("CP R32 wgt=INT3",
     load(f"{base}/results/cp_experiment/CP_R32_WGTINT3/{ds}/metrics.json")),
    ("K-SVD K=256 s=16 D=INT8",
     load(f"{base}/results/ksvd_experiment/KSVD_K256_S16_DINT8/{ds}/metrics.json")),
    ("K-SVD K=256 s=16 D=INT4",
     load(f"{base}/results/ksvd_experiment/KSVD_K256_S16_DINT4/{ds}/metrics.json")),
    ("K-SVD K=256 s=16 D=INT3",
     load(f"{base}/results/ksvd_experiment/KSVD_K256_S16_DINT3/{ds}/metrics.json")),
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
    pm = d.get("primary_metrics", d.get("dist_metrics", {}))
    sm = d.get("secondary_metrics", d.get("averages", {}))
    fid  = pm.get("FID", 0)
    is_v = pm.get("IS",  pm.get("inception_score", 0))
    psnr = sm.get("PSNR", sm.get("psnr", 0))
    ssim = sm.get("SSIM", sm.get("ssim", 0))
    beat = ""
    if bl_fid and "BASELINE" not in name:
        bf = fid  < bl_fid
        bi = is_v > bl_is
        if bf and bi:  beat = "FID+IS ✓"
        elif bf:       beat = "FID ✓"
        elif bi:       beat = "IS ✓"
        else:          beat = "✗"
    psnr_s = f"{psnr:.1f}" if psnr and psnr != float("inf") else "--"
    print(f"  {name:<28} {fid:>8.1f} {is_v:>7.4f} {psnr_s:>7} {ssim:>7.4f}  {beat}")
print()
PYEOF

echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
