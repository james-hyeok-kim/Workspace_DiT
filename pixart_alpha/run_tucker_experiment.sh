#!/bin/bash
# Tucker-2 decomposition quantization experiment
# W ≈ U1 @ G @ U2ᵀ (HOSVD): U1/U2=NVFP4, G(core)=INT3, act=NVFP4
# Sweeps: rank=32/64, core_mode=INT3/INT4/NVFP4

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
LOG_DIR="${BASE_DIR}/logs/tucker_experiment"
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
PY="${BASE_DIR}/pixart_tucker_experiment.py"

echo "============================================================"
echo "  Tucker-2 Decomposition Quantization Experiment"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  NUM_SAMPLES=${NUM_SAMPLES}"
echo "  U1/U2=NVFP4  Core=INT3/INT4/NVFP4  rank=32/64"
echo "============================================================"

# ---- Tucker r=64, core=INT3 (main config) ----
run_tucker() {
    local RANK=$1
    local CORE_MODE=$2
    local TAG="TUCKER_R${RANK}_CORE${CORE_MODE}"
    local SAVE="${BASE_DIR}/results/tucker_experiment/${TAG}"
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
        --lora_rank    32 \
        --act_mode     NVFP4 \
        --wgt_core_mode ${CORE_MODE} \
        --block_size   16 \
        2>&1 | tee "${LOG_DIR}/${TAG}.log"
    echo "  [$(date '+%H:%M:%S')] ${TAG} done."
}

run_tucker 64 INT3
run_tucker 64 INT4
run_tucker 64 NVFP4
run_tucker 32 INT3

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
    ("Tucker r=64, core=INT4",
     load(f"{base}/results/tucker_experiment/TUCKER_R64_COREINT4/{ds}/metrics.json")),
    ("Tucker r=64, core=NVFP4",
     load(f"{base}/results/tucker_experiment/TUCKER_R64_CORENVFP4/{ds}/metrics.json")),
    ("Tucker r=32, core=INT3",
     load(f"{base}/results/tucker_experiment/TUCKER_R32_COREINT3/{ds}/metrics.json")),
]

bl = next((d for n, d in entries if "BASELINE" in n and d), None)
bl_fid = bl["primary_metrics"]["FID"] if bl else None
bl_is  = bl["primary_metrics"]["IS"]  if bl else None

print(f"\n{'Config':<30} {'FID↓':>8} {'IS↑':>7} {'PSNR↑':>7} {'SSIM↑':>7} {'vs_FP16':>9} {'vs_NVFP4':>10}  vs BASELINE")
print("-" * 88)
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
    cx_fp16  = comp.get("compression_vs_fp16",  "--")
    cx_nvfp4 = comp.get("compression_vs_nvfp4", "--")
    beat = ""
    if bl_fid and "BASELINE" not in name:
        bf = fid  < bl_fid
        bi = is_v > bl_is
        if bf and bi:  beat = "FID+IS ✓"
        elif bf:       beat = "FID ✓"
        elif bi:       beat = "IS ✓"
        else:          beat = "✗"
    psnr_s = f"{psnr:.1f}" if psnr != float("inf") else "∞"
    cx_fp16_s  = f"{cx_fp16}x"  if isinstance(cx_fp16,  (int, float)) else "--"
    cx_nvfp4_s = f"{cx_nvfp4}x" if isinstance(cx_nvfp4, (int, float)) else "--"
    print(f"  {name:<28} {fid:>8.1f} {is_v:>7.4f} {psnr_s:>7} {ssim:>7.4f} {cx_fp16_s:>9} {cx_nvfp4_s:>10}  {beat}")
print()
PYEOF

echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
