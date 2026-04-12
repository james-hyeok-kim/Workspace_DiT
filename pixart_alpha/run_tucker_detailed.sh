#!/bin/bash
# Tucker-2 Detailed Experiment
#
# Sweep:
#   Phase 1 (main): rank_ratio=0.75, block_size=16
#     core=INT2/INT3/INT4 × U=FP16/BF16/FP32 = 9 runs
#     + no_lora variants for INT3/FP16 to measure LoRA contribution
#
#   Phase 2 (rank ablation): best core+U from Phase1, block_size=16
#     rank_ratio = 0.50 / 0.75 / 1.00
#
#   Phase 3 (block_size ablation): best core+U, rank_ratio=0.75
#     block_size = 16 / 32 / 64
#
# LoRA redesign: corrects W - U1@G_q@U2ᵀ (total quantized Tucker error)
# --no_lora: measure pure Tucker decomposition quality

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
LOG_DIR="${BASE_DIR}/logs/tucker_detailed"
RESULT_DIR="${BASE_DIR}/results/tucker_detailed"
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
echo "  Tucker-2 Detailed Experiment"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  NUM_SAMPLES=${NUM_SAMPLES}"
echo "  Core=INT2/INT3/INT4  U=FP16/BF16/FP32  rank=50/75/100%  bs=16/32/64"
echo "============================================================"

# run_tucker RANK_RATIO CORE_MODE U_MODE BLOCK_SIZE [no_lora]
run_tucker() {
    local RANK_RATIO=$1
    local CORE_MODE=$2
    local U_MODE=$3
    local BLOCK_SIZE=$4
    local NO_LORA=${5:-""}

    local RR_TAG="${RANK_RATIO//./p}"   # 0.75 → 0p75
    local LORA_TAG=""
    local LORA_FLAG=""
    if [ "${NO_LORA}" = "no_lora" ]; then
        LORA_TAG="_NOLORA"
        LORA_FLAG="--no_lora"
    fi

    local TAG="TUCKER_RR${RR_TAG}_CORE${CORE_MODE}_U${U_MODE}_BS${BLOCK_SIZE}${LORA_TAG}"
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
        --wgt_core_mode ${CORE_MODE} \
        --wgt_u_mode    ${U_MODE} \
        --act_mode      NVFP4 \
        --block_size    ${BLOCK_SIZE} \
        --lora_rank     32 \
        ${LORA_FLAG} \
        2>&1 | tee "${LOG_DIR}/${TAG}.log"
    echo "  [$(date '+%H:%M:%S')] ${TAG} done."
}

# ============================================================
# Phase 1: Main sweep - core × U mode
# rank_ratio=0.75, block_size=16
# ============================================================
echo ""
echo "==== Phase 1: Core × U mode sweep (rank=0.75, bs=16) ===="

for CORE in INT4 INT3 INT2; do
    for UMODE in FP16 BF16 FP32; do
        run_tucker 0.75 ${CORE} ${UMODE} 16
    done
done

# LoRA contribution: INT3+FP16 with and without LoRA
echo ""
echo "==== Phase 1b: LoRA contribution (INT3, FP16, rank=0.75, bs=16) ===="
run_tucker 0.75 INT3 FP16 16 no_lora
run_tucker 0.75 INT4 FP16 16 no_lora

# ============================================================
# Phase 2: Rank ablation
# Fixed: CORE=INT3, U=FP16, block_size=16
# ============================================================
echo ""
echo "==== Phase 2: Rank ablation (INT3, FP16, bs=16) ===="

run_tucker 0.50 INT3 FP16 16
# 0.75 already done in Phase 1
run_tucker 1.00 INT3 FP16 16

# ============================================================
# Phase 3: Block size ablation
# Fixed: CORE=INT3, U=FP16, rank_ratio=0.75
# ============================================================
echo ""
echo "==== Phase 3: Block size ablation (INT3, FP16, rank=0.75) ===="

# bs=16 already done in Phase 1
run_tucker 0.75 INT3 FP16 32
run_tucker 0.75 INT3 FP16 64

# ============================================================
# Final comparison table
# ============================================================
echo ""
echo "============================================================"
echo "  Tucker Detailed Results"
echo "============================================================"

export BASE_DIR_PY="${BASE_DIR}"
export RESULT_DIR_PY="${RESULT_DIR}"
export DATASET_PY="${DATASET}"

python3 - <<'PYEOF'
import json, os

base    = os.environ["BASE_DIR_PY"]
resdir  = os.environ["RESULT_DIR_PY"]
ds      = os.environ["DATASET_PY"]

def load(path):
    return json.load(open(path)) if os.path.exists(path) else None

def entry(name, tag):
    return (name, load(f"{resdir}/{tag}/{ds}/metrics.json"))

baseline = load(f"{base}/results/quip_experiment/BASELINE/{ds}/metrics.json")
bl_fid = baseline["primary_metrics"]["FID"] if baseline else None
bl_is  = baseline["primary_metrics"]["IS"]  if baseline else None

sections = [
    ("=== Phase 1: Core × U mode (rank=0.75, bs=16) ===", [
        entry("INT4 + FP16", "TUCKER_RR0p75_COREINT4_UFP16_BS16"),
        entry("INT4 + BF16", "TUCKER_RR0p75_COREINT4_UBF16_BS16"),
        entry("INT4 + FP32", "TUCKER_RR0p75_COREINT4_UFP32_BS16"),
        entry("INT3 + FP16", "TUCKER_RR0p75_COREINT3_UFP16_BS16"),
        entry("INT3 + BF16", "TUCKER_RR0p75_COREINT3_UBF16_BS16"),
        entry("INT3 + FP32", "TUCKER_RR0p75_COREINT3_UFP32_BS16"),
        entry("INT2 + FP16", "TUCKER_RR0p75_COREINT2_UFP16_BS16"),
        entry("INT2 + BF16", "TUCKER_RR0p75_COREINT2_UBF16_BS16"),
        entry("INT2 + FP32", "TUCKER_RR0p75_COREINT2_UFP32_BS16"),
    ]),
    ("=== Phase 1b: LoRA contribution (rank=0.75, bs=16) ===", [
        entry("INT3+FP16 w/ LoRA",  "TUCKER_RR0p75_COREINT3_UFP16_BS16"),
        entry("INT3+FP16 no LoRA",  "TUCKER_RR0p75_COREINT3_UFP16_BS16_NOLORA"),
        entry("INT4+FP16 w/ LoRA",  "TUCKER_RR0p75_COREINT4_UFP16_BS16"),
        entry("INT4+FP16 no LoRA",  "TUCKER_RR0p75_COREINT4_UFP16_BS16_NOLORA"),
    ]),
    ("=== Phase 2: Rank ablation (INT3, FP16, bs=16) ===", [
        entry("rank=50%",  "TUCKER_RR0p50_COREINT3_UFP16_BS16"),
        entry("rank=75%",  "TUCKER_RR0p75_COREINT3_UFP16_BS16"),
        entry("rank=100%", "TUCKER_RR1p00_COREINT3_UFP16_BS16"),
    ]),
    ("=== Phase 3: Block size ablation (INT3, FP16, rank=0.75) ===", [
        entry("block=16", "TUCKER_RR0p75_COREINT3_UFP16_BS16"),
        entry("block=32", "TUCKER_RR0p75_COREINT3_UFP16_BS32"),
        entry("block=64", "TUCKER_RR0p75_COREINT3_UFP16_BS64"),
    ]),
]

def beat_str(d):
    if d is None or bl_fid is None:
        return ""
    pm = d.get("primary_metrics", {})
    fid = pm.get("FID", 0)
    is_v = pm.get("IS", 0)
    bf = fid < bl_fid
    bi = is_v > bl_is
    if bf and bi:  return "FID+IS ✓"
    elif bf:       return "FID ✓"
    elif bi:       return "IS ✓"
    else:          return "✗"

hdr = f"  {'Config':<26} {'FID↓':>8} {'IS↑':>7} {'PSNR↑':>7} {'SSIM↑':>6}  {'vs_NVFP4':>9}  beat"
sep = "  " + "-" * 78

if baseline:
    print(f"\n  BASELINE (NVFP4_DEFAULT): FID={bl_fid:.1f}  IS={bl_is:.4f}\n")

for section_title, entries in sections:
    print(f"\n{section_title}")
    print(hdr)
    print(sep)
    for name, d in entries:
        if d is None:
            print(f"  {name:<26} {'N/A':>8}")
            continue
        pm   = d.get("primary_metrics", {})
        sm   = d.get("secondary_metrics", {})
        comp = d.get("compression", {})
        fid  = pm.get("FID", 0)
        is_v = pm.get("IS",  0)
        psnr = sm.get("PSNR", 0)
        ssim = sm.get("SSIM", 0)
        cx   = comp.get("compression_vs_nvfp4_tucker_only", "--")
        cx_s = f"{cx}x" if isinstance(cx, (int, float)) else "--"
        psnr_s = f"{psnr:.1f}" if psnr else "--"
        print(f"  {name:<26} {fid:>8.1f} {is_v:>7.4f} {psnr_s:>7} {ssim:>6.4f}  {cx_s:>9}  {beat_str(d)}")

print()
PYEOF

echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
