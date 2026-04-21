#!/bin/bash
# ============================================================
# QuIP W4A4 단독 실행 + NVFP4/RPCA/GPTQ 결과와 비교표 출력
# ============================================================

export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
if [ -f ~/.env ]; then
  export $(grep -v '^#' ~/.env | xargs)
fi
if ! command -v accelerate &>/dev/null; then
  source /home/jameskimh/.dit/bin/activate
fi

# ---- 설정 ----
NUM_GPUS=2
NUM_SAMPLES=20
MODEL_PATH="PixArt-alpha/PixArt-XL-2-1024-MS"
DATASET="MJHQ"
BLOCK_SIZE=128
LOWRANK=32
REF_DIR="/data/jameskimh/james_dit_ref/ref_images_fp16"
BASE_DIR="$(pwd)"
SAVE_DIR="${BASE_DIR}/results/quip_experiment/W4A4"
LOG_DIR="${BASE_DIR}/logs/quip_experiment"
mkdir -p "${SAVE_DIR}" "${LOG_DIR}"

RUN_LOG="${LOG_DIR}/w4a4_run.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

ACCEL_CMD="accelerate launch --multi_gpu --num_processes ${NUM_GPUS}"

echo "============================================================"
echo "  QuIP W4A4 실험  (INT4 weight + INT4 act + random Hadamard)"
echo "  rank=${LOWRANK}  block_size=${BLOCK_SIZE}  samples=${NUM_SAMPLES}"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

if [ -f "${SAVE_DIR}/${DATASET}/metrics.json" ]; then
    echo "[SKIP] W4A4 metrics already exists."
else
    ${ACCEL_CMD} "${BASE_DIR}/pixart_quip_experiment.py" \
        --quant_method W4A4 \
        --block_size ${BLOCK_SIZE} \
        --lowrank ${LOWRANK} \
        --model_path "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir "${REF_DIR}" \
        --save_dir "${SAVE_DIR}" \
        --num_samples ${NUM_SAMPLES}
fi

echo ""
echo "============================================================"
echo "  비교표: QuIP-W4A4 vs NVFP4 vs RPCA(best) vs GPTQ"
echo "============================================================"

export BASE_DIR_PY="${BASE_DIR}"
export DATASET_PY="${DATASET}"

python3 - <<'PYEOF'
import json, os

base = os.environ["BASE_DIR_PY"]
ds   = os.environ["DATASET_PY"]

def load(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))

entries = [
    ("FP16",           load(f"{base}/results/quip_experiment/FP16/{ds}/metrics.json")),
    ("NVFP4-BASELINE", load(f"{base}/results/quip_experiment/BASELINE/{ds}/metrics.json")),
    ("RPCA-best\n(INT8w INT8a OR=0.1)",
                       load(f"{base}/results/rpca_sweep/RPCA_AINT8_WINT8_OR0.1/{ds}/metrics.json")),
    ("RPCA-4bit\n(INT4w INT4a OR=0.05)",
                       load(f"{base}/results/rpca_sweep/RPCA_AINT4_WINT4_OR0.05/{ds}/metrics.json")),
    ("GPTQ-W4A4\n(NVFP4w INT4a)",
                       load(f"{base}/results/w3a4_experiment/W3A4/{ds}/metrics.json")),
    ("QuIP-W4A4\n(INT4w INT4a)",
                       load(f"{base}/results/quip_experiment/W4A4/{ds}/metrics.json")),
]

bl_fid = None
bl_is  = None
for name, d in entries:
    if "NVFP4-BASELINE" in name and d:
        bl_fid = d["primary_metrics"]["FID"]
        bl_is  = d["primary_metrics"]["IS"]

print(f"\n{'Method':<30} {'FID↓':>8} {'IS↑':>7} {'PSNR↑':>7} {'SSIM↑':>7}  Beat?")
print("-" * 75)
for name, d in entries:
    label = name.split("\n")[0]
    sub   = name.split("\n")[1] if "\n" in name else ""
    if d is None:
        print(f"  {label:<28} {'N/A':>8}")
        continue
    pm = d["primary_metrics"]
    sm = d["secondary_metrics"]
    fid  = pm.get("FID", 0)
    is_v = pm.get("IS",  0)
    psnr = sm.get("PSNR", 0)
    ssim = sm.get("SSIM", 0)
    beat = ""
    if bl_fid and "BASELINE" not in label and "FP16" not in label:
        beat_fid = fid  < bl_fid
        beat_is  = is_v > bl_is
        if beat_fid and beat_is:  beat = "FID+IS ✓"
        elif beat_fid:            beat = "FID ✓"
        elif beat_is:             beat = "IS ✓"
        else:                     beat = "✗"
    psnr_str = f"{psnr:.1f}" if psnr != float("inf") else "∞"
    print(f"  {label:<28} {fid:>8.1f} {is_v:>7.4f} {psnr_str:>7} {ssim:>7.4f}  {beat}")
    if sub:
        print(f"    {sub}")

print()
PYEOF

echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log: ${RUN_LOG}"
