#!/bin/bash
# ============================================================
# Timestep-Aware Mixed Precision Quantization Ablation
#
# Ablation configs (5개):
#   UNIFORM_FP4   : Control - 모두 동일 (NVFP4 act, rank=32)
#   MP_ACT_ONLY   : activation precision만 변화
#   MP_RANK_ONLY  : SVD rank만 변화 (high noise에서 SVD 생략)
#   MP_MODERATE   : act + rank 둘 다 적당히 변화
#   MP_AGGRESSIVE : high noise 최대 절약 + low noise INT8 보호
#
# 비교 baseline: results/quip_experiment/BASELINE (FID=161.3, IS=1.732)
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
WGT_BITS="NVFP4"
ALPHA=0.5
REF_DIR="/data/james_dit_ref/ref_images_fp16"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT_BASE="${BASE_DIR}/results"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "${LOG_DIR}"

# TEST_MODE=1 → 2 samples smoke test
if [ "${TEST_MODE:-0}" = "1" ]; then
  NUM_SAMPLES=2
  EXTRA_ARGS="--test_run"
  echo "[TEST MODE] NUM_SAMPLES=2"
else
  NUM_SAMPLES=20
  EXTRA_ARGS=""
fi

RUN_LOG="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${RUN_LOG}") 2>&1

ACCEL_CMD="accelerate launch --multi_gpu --num_processes ${NUM_GPUS}"
PY="${BASE_DIR}/pixart_ts_mixed_precision.py"

COMMON_ARGS="--model_path ${MODEL_PATH} --dataset_name ${DATASET} --ref_dir ${REF_DIR} \
  --block_size ${BLOCK_SIZE} --wgt_bits ${WGT_BITS} --alpha ${ALPHA} \
  --num_samples ${NUM_SAMPLES} ${EXTRA_ARGS}"

echo "============================================================"
echo "  Timestep-Aware Mixed Precision Ablation"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  NUM_SAMPLES=${NUM_SAMPLES}  BLOCK_SIZE=${BLOCK_SIZE}  WGT_BITS=${WGT_BITS}"
echo "============================================================"

run_config() {
    local config_name=$1
    local save_dir="${RESULT_BASE}/${config_name}"
    local log_file="${LOG_DIR}/${config_name}.log"

    echo ""
    echo "---- [${config_name}]  ($(date '+%H:%M:%S')) ----"

    if [ -f "${save_dir}/${DATASET}/metrics.json" ]; then
        echo "  [SKIP] already exists: ${save_dir}/${DATASET}/metrics.json"
        return
    fi

    ${ACCEL_CMD} "${PY}" \
        --mp_config "${config_name}" \
        --save_dir "${save_dir}" \
        ${COMMON_ARGS} \
        2>&1 | tee "${log_file}"

    echo "  [$(date '+%H:%M:%S')] ${config_name} done."
}

# ---- 5개 Ablation 실행 ----
run_config "UNIFORM_FP4"
run_config "MP_ACT_ONLY"
run_config "MP_RANK_ONLY"
run_config "MP_MODERATE"
run_config "MP_AGGRESSIVE"

echo ""
echo "============================================================"
echo "  통합 비교표: Mixed Precision Ablation 결과"
echo "============================================================"

export RESULT_BASE_PY="${RESULT_BASE}"
export DATASET_PY="${DATASET}"
export PIXART_BASE="${BASE_DIR}/../pixart_alpha"

python3 - <<'PYEOF'
import json, os

result_base = os.environ["RESULT_BASE_PY"]
ds          = os.environ["DATASET_PY"]
pixart_base = os.environ["PIXART_BASE"]

def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None

# BASELINE (NVFP4_DEFAULT_CFG) from pixart_alpha results
baseline = load(f"{pixart_base}/results/quip_experiment/BASELINE/{ds}/metrics.json")
bl_fid = baseline["primary_metrics"]["FID"] if baseline else 161.3
bl_is  = baseline["primary_metrics"]["IS"]  if baseline else 1.732

configs = ["UNIFORM_FP4", "MP_ACT_ONLY", "MP_RANK_ONLY", "MP_MODERATE", "MP_AGGRESSIVE"]
entries = []
for cfg in configs:
    d = load(f"{result_base}/{cfg}/{ds}/metrics.json")
    entries.append((cfg, d))

print(f"\n{'Config':<20} {'Act bits':>8} {'SVD save':>9} {'FID↓':>8} {'IS↑':>7} {'PSNR↑':>7} {'SSIM↑':>7}  vs BASELINE")
print("-" * 90)
print(f"  {'BASELINE(NVFP4_DEFAULT)':<18} {'N/A':>8} {'N/A':>9} {bl_fid:>8.1f} {bl_is:>7.4f} {'N/A':>7} {'N/A':>7}")
print("-" * 90)

results = {}
for name, d in entries:
    if d is None:
        print(f"  {name:<20} {'N/A':>8}")
        continue
    pm  = d["primary_metrics"]
    sm  = d["secondary_metrics"]
    eb  = d.get("effective_bitwidth", {})
    fid = pm.get("FID", 0)
    is_v= pm.get("IS", 0)
    psnr= sm.get("PSNR", 0)
    ssim= sm.get("SSIM", 0)
    avg_bits  = eb.get("avg_act_bits", 0)
    svd_save  = eb.get("svd_savings_pct", 0)
    beat = ""
    bf = fid < bl_fid
    bi = is_v > bl_is
    if bf and bi:  beat = "FID+IS BEAT"
    elif bf:       beat = "FID BEAT"
    elif bi:       beat = "IS BEAT"
    else:          beat = "-"
    psnr_s = f"{psnr:.1f}" if psnr != float("inf") else "inf"
    print(f"  {name:<20} {avg_bits:>7.2f}b {svd_save:>8.1f}% {fid:>8.1f} {is_v:>7.4f} {psnr_s:>7} {ssim:>7.4f}  {beat}")
    results[name] = {"FID": fid, "IS": is_v, "PSNR": psnr, "SSIM": ssim,
                     "avg_act_bits": avg_bits, "svd_savings_pct": svd_save,
                     "beats_baseline": beat}

print()
summary_path = f"{result_base}/results_summary.json"
os.makedirs(result_base, exist_ok=True)
results["BASELINE"] = {"FID": bl_fid, "IS": bl_is}
with open(summary_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"Summary saved: {summary_path}")
PYEOF

echo ""
echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
