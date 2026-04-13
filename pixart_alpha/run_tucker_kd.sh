#!/bin/bash
# Tucker KD Experiment
#
# 4 runs:
#   1. Tucker 50%  rank, no LoRA, no KD  (U=INT4, G=FP16)
#   2. Tucker 75%  rank, no LoRA, no KD
#   3. Tucker 100% rank, no LoRA, no KD
#   4. Tucker 50%  rank, no LoRA, KD-100
#
# U1/U2 = INT4 + FP16 scale (block=16)
# G     = FP16 (no quantization)
# 50 samples, MJHQ

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
LOG_DIR="${BASE_DIR}/logs/tucker_kd"
RESULT_DIR="${BASE_DIR}/results/tucker_kd"
mkdir -p "${LOG_DIR}"

if [ "${TEST_MODE:-0}" = "1" ]; then
  NUM_SAMPLES=2
  KD_STEPS=5
  echo "[TEST MODE] NUM_SAMPLES=2, KD_STEPS=5"
else
  NUM_SAMPLES=50
  KD_STEPS=100
fi

RUN_LOG="${LOG_DIR}/run.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

ACCEL_CMD="accelerate launch --multi_gpu --num_processes ${NUM_GPUS}"
PY="${BASE_DIR}/pixart_tucker_experiment.py"

echo "============================================================"
echo "  Tucker KD Experiment"
echo "  U=INT4+FP16scale  G=FP16  act=NVFP4  block=16"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  NUM_SAMPLES=${NUM_SAMPLES}  KD_STEPS=${KD_STEPS}"
echo "============================================================"

# run_tucker RANK_RATIO U_MODE [--do_kd]
run_tucker() {
    local RANK_RATIO=$1
    local U_MODE=$2
    local DO_KD=${3:-""}
    local RR_TAG="${RANK_RATIO//./p}"

    local KD_TAG=""
    local KD_FLAGS=""
    if [ "${DO_KD}" = "--do_kd" ]; then
        KD_TAG="_KD${KD_STEPS}"
        KD_FLAGS="--kd_mode --do_kd --kd_steps ${KD_STEPS} --kd_lr 1e-4 --kd_prompts 8"
    else
        KD_FLAGS="--no_lora"
    fi

    local TAG="TUCKER_KD_RR${RR_TAG}_U${U_MODE}_GFP16${KD_TAG}"
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
        --wgt_u_mode    ${U_MODE} \
        --wgt_core_mode FP16 \
        --act_mode      NVFP4 \
        --block_size    16 \
        --lora_rank     32 \
        ${KD_FLAGS} \
        2>&1 | tee "${LOG_DIR}/${TAG}.log"
    echo "  [$(date '+%H:%M:%S')] ${TAG} done."
}

# 1: U=INT4, 50%, no KD
run_tucker 0.50 INT4
# 2: U=INT3, 75%, no KD
run_tucker 0.75 INT3
# 3: U=INT2, 100%, no KD
run_tucker 1.00 INT2
# 4: U=INT4, 50%, KD-100
run_tucker 0.50 INT4 --do_kd
# 5: U=INT2, 50%, KD-100
run_tucker 0.50 INT2 --do_kd

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

base   = os.environ["BASE_DIR_PY"]
resdir = os.environ["RESULT_DIR_PY"]
ds     = os.environ["DATASET_PY"]

def load(p): return json.load(open(p)) if os.path.exists(p) else None

bl = load(f"{base}/results/quip_experiment/BASELINE/{ds}/metrics.json")
bl_fid = bl["primary_metrics"]["FID"] if bl else None
bl_is  = bl["primary_metrics"]["IS"]  if bl else None

entries = [
    ("BASELINE (SVDQuant NVFP4)",
     bl),
    ("1. Tucker 50%  U=INT4 G=FP16 no_kd",
     load(f"{resdir}/TUCKER_KD_RR0p50_UINT4_GFP16/{ds}/metrics.json")),
    ("2. Tucker 75%  U=INT3 G=FP16 no_kd",
     load(f"{resdir}/TUCKER_KD_RR0p75_UINT3_GFP16/{ds}/metrics.json")),
    ("3. Tucker 100% U=INT2 G=FP16 no_kd",
     load(f"{resdir}/TUCKER_KD_RR1p00_UINT2_GFP16/{ds}/metrics.json")),
    ("4. Tucker 50%  U=INT4 G=FP16 KD-100",
     load(f"{resdir}/TUCKER_KD_RR0p50_UINT4_GFP16_KD100/{ds}/metrics.json")),
    ("5. Tucker 50%  U=INT2 G=FP16 KD-100",
     load(f"{resdir}/TUCKER_KD_RR0p50_UINT2_GFP16_KD100/{ds}/metrics.json")),
]

print(f"\n  {'Config':<32} {'FID↓':>8} {'IS↑':>7} {'PSNR↑':>7} {'SSIM↑':>6}  vs BASELINE")
print("  " + "-" * 72)
for name, d in entries:
    if d is None:
        print(f"  {name:<32} {'N/A':>8}")
        continue
    pm   = d.get("primary_metrics", {})
    sm   = d.get("secondary_metrics", {})
    fid  = pm.get("FID", 0)
    is_v = pm.get("IS",  0)
    psnr = sm.get("PSNR", 0)
    ssim = sm.get("SSIM", 0)
    beat = ""
    if bl_fid and "BASELINE" not in name:
        if fid < bl_fid and is_v > bl_is: beat = "FID+IS ✓"
        elif fid < bl_fid:                beat = "FID ✓"
        elif is_v > bl_is:                beat = "IS ✓"
        else:                             beat = "✗"
    psnr_s = f"{psnr:.1f}" if psnr else "--"
    kd = d.get("kd_info", {})
    kd_s = f"(loss {kd['kd_loss_init']:.4f}→{kd['kd_loss_final']:.4f})" if kd.get("do_kd") else ""
    print(f"  {name:<32} {fid:>8.1f} {is_v:>7.4f} {psnr_s:>7} {ssim:>6.4f}  {beat}  {kd_s}")

print()
PYEOF

echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
