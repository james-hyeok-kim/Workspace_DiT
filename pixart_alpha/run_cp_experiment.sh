#!/bin/bash
# CP (PARAFAC) decomposition quantization experiment
# W (m×n) → 3D reshape → ALS PARAFAC → A(m×r) B(n1×r) C(n2×r), all NVFP4, act=NVFP4
# Sweeps: rank=32/64, wgt_mode=NVFP4/INT3

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
LOG_DIR="${BASE_DIR}/logs/cp_experiment"
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
PY="${BASE_DIR}/pixart_cp_experiment.py"

echo "============================================================"
echo "  CP (PARAFAC) Decomposition Quantization Experiment"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  NUM_SAMPLES=${NUM_SAMPLES}"
echo "  Factors=NVFP4/INT3  rank=32/64"
echo "============================================================"

run_cp() {
    local RANK=$1
    local WGT_MODE=$2
    local TAG="CP_R${RANK}_WGT${WGT_MODE}"
    local SAVE="${BASE_DIR}/results/cp_experiment/${TAG}"
    echo ""
    echo "---- ${TAG}  ($(date '+%H:%M:%S')) ----"
    if [ -f "${SAVE}/${DATASET}/metrics.json" ]; then
        echo "  [SKIP] already exists."
        return
    fi
    ${ACCEL_CMD} "${PY}" \
        --model_path   "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir      "${REF_DIR}" \
        --save_dir     "${SAVE}" \
        --num_samples  ${NUM_SAMPLES} \
        --lowrank      ${RANK} \
        --act_mode     NVFP4 \
        --wgt_mode     ${WGT_MODE} \
        --als_iters    100 \
        --block_size   16 \
        2>&1 | tee "${LOG_DIR}/${TAG}.log"
    echo "  [$(date '+%H:%M:%S')] ${TAG} done."
}

run_cp 64 NVFP4
run_cp 64 INT3
run_cp 32 NVFP4

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
    ("RPCA-NVFP4 + KD-50 (best)",
     load(f"{base}/results/kd_experiment/RPCA_NVFP4_KD50/{ds}/metrics.json")),
    ("Tucker r=64, core=INT3",
     load(f"{base}/results/tucker_experiment/TUCKER_R64_COREINT3/{ds}/metrics.json")),
    ("CP r=64, wgt=NVFP4",
     load(f"{base}/results/cp_experiment/CP_R64_WGTNVFP4/{ds}/metrics.json")),
    ("CP r=64, wgt=INT3",
     load(f"{base}/results/cp_experiment/CP_R64_WGTINT3/{ds}/metrics.json")),
    ("CP r=32, wgt=NVFP4",
     load(f"{base}/results/cp_experiment/CP_R32_WGTNVFP4/{ds}/metrics.json")),
]

bl = next((d for n, d in entries if "BASELINE" in n and d), None)
bl_fid = bl["primary_metrics"]["FID"] if bl else None
bl_is  = bl["primary_metrics"]["IS"]  if bl else None

print(f"\n{'Config':<30} {'FID↓':>8} {'IS↑':>7} {'PSNR↑':>7} {'SSIM↑':>7} {'vs_FP16':>9} {'vs_NVFP4':>10}  vs BASELINE")
print("-" * 90)
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
    comp = d.get("compression", {})
    cx_fp16  = comp.get("compression_vs_fp16",  None)
    cx_nvfp4 = comp.get("compression_vs_nvfp4", None)
    beat = ""
    if bl_fid and "BASELINE" not in name:
        bf = fid  < bl_fid
        bi = is_v > bl_is
        if bf and bi:  beat = "FID+IS ✓"
        elif bf:       beat = "FID ✓"
        elif bi:       beat = "IS ✓"
        else:          beat = "✗"
    psnr_s   = f"{psnr:.1f}"   if psnr != float("inf") else "∞"
    cx_fp16_s  = f"{cx_fp16}x"  if cx_fp16  is not None else "--"
    cx_nvfp4_s = f"{cx_nvfp4}x" if cx_nvfp4 is not None else "--"
    print(f"  {name:<28} {fid:>8.1f} {is_v:>7.4f} {psnr_s:>7} {ssim:>7.4f} {cx_fp16_s:>9} {cx_nvfp4_s:>10}  {beat}")
print()
PYEOF

echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
