#!/bin/bash
# K-SVD Dictionary Learning Sparse Coding quantization experiment
# W ≈ D @ X:  D(m×K) at various precisions, X(K×n) sparse (s nnz/col)
#
# Sweep:
#   dict_mode = FP16 / FP8 / NVFP4 / INT3
#   K=256, s=16 (main config)
#   K=128, s=16 for ablation
#
# Compression formula (repr layer m=4608, n=1152):
#   total = m*K*bits_D + s*n*(bits_val + bits_idx)
#   bits_idx = ceil(log2(K))  (e.g., K=256 → 8 bits)

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
LOG_DIR="${BASE_DIR}/logs/ksvd_experiment"
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
PY="${BASE_DIR}/pixart_ksvd_experiment.py"

echo "============================================================"
echo "  K-SVD Dictionary Learning Quantization Experiment"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  NUM_SAMPLES=${NUM_SAMPLES}"
echo "  Sweep: dict_mode=FP16/FP8/NVFP4/INT3  K=256  s=16"
echo "============================================================"

run_ksvd() {
    local DICT_MODE=$1
    local K=$2
    local S=$3
    local TAG="KSVD_K${K}_S${S}_D${DICT_MODE}"
    local SAVE="${BASE_DIR}/results/ksvd_experiment/${TAG}"
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

# Main sweep: vary dictionary precision, fixed K=256, s=16
run_ksvd FP16   256 16
run_ksvd FP8    256 16
run_ksvd NVFP4  256 16
run_ksvd INT3   256 16

# Ablation: smaller dictionary K=128
run_ksvd NVFP4  128 16

echo ""
echo "============================================================"
echo "  비교 결과 — K-SVD vs BASELINE + compression table"
echo "============================================================"

export BASE_DIR_PY="${BASE_DIR}"
export DATASET_PY="${DATASET}"

python3 - <<'PYEOF'
import json, os, math

base = os.environ["BASE_DIR_PY"]
ds   = os.environ["DATASET_PY"]

def load(path):
    return json.load(open(path)) if os.path.exists(path) else None

entries = [
    ("BASELINE (NVFP4_DEFAULT_CFG)",
     load(f"{base}/results/quip_experiment/BASELINE/{ds}/metrics.json")),
    ("RPCA-NVFP4 + KD-50",
     load(f"{base}/results/kd_experiment/RPCA_NVFP4_KD50/{ds}/metrics.json")),
    ("Tucker r=64, core=INT3",
     load(f"{base}/results/tucker_experiment/TUCKER_R64_COREINT3/{ds}/metrics.json")),
    ("K-SVD K=256 s=16 D=FP16",
     load(f"{base}/results/ksvd_experiment/KSVD_K256_S16_DFP16/{ds}/metrics.json")),
    ("K-SVD K=256 s=16 D=FP8",
     load(f"{base}/results/ksvd_experiment/KSVD_K256_S16_DFP8/{ds}/metrics.json")),
    ("K-SVD K=256 s=16 D=NVFP4",
     load(f"{base}/results/ksvd_experiment/KSVD_K256_S16_DNVFP4/{ds}/metrics.json")),
    ("K-SVD K=256 s=16 D=INT3",
     load(f"{base}/results/ksvd_experiment/KSVD_K256_S16_DINT3/{ds}/metrics.json")),
    ("K-SVD K=128 s=16 D=NVFP4",
     load(f"{base}/results/ksvd_experiment/KSVD_K128_S16_DNVFP4/{ds}/metrics.json")),
]

bl = next((d for n, d in entries if "BASELINE" in n and d), None)
bl_fid = bl["primary_metrics"]["FID"] if bl else None
bl_is  = bl["primary_metrics"]["IS"]  if bl else None

print(f"\n{'Config':<34} {'FID↓':>8} {'IS↑':>7} {'PSNR↑':>7} {'SSIM↑':>7} {'vs_FP16':>9} {'vs_NVFP4':>10}  vs BASELINE")
print("-" * 96)
for name, d in entries:
    if d is None:
        print(f"  {name:<32} {'N/A':>8}")
        continue
    pm   = d["primary_metrics"]
    sm   = d["secondary_metrics"]
    comp = d.get("compression", {})
    fid  = pm.get("FID", 0)
    is_v = pm.get("IS",  0)
    psnr = sm.get("PSNR", 0)
    ssim = sm.get("SSIM", 0)
    cx_fp16  = comp.get("compression_vs_fp16",  None)
    cx_nvfp4 = comp.get("compression_vs_nvfp4", None)
    recon_e  = comp.get("mean_reconstruction_error", None)
    beat = ""
    if bl_fid and "BASELINE" not in name:
        bf = fid  < bl_fid
        bi = is_v > bl_is
        if bf and bi:  beat = "FID+IS ✓"
        elif bf:       beat = "FID ✓"
        elif bi:       beat = "IS ✓"
        else:          beat = "✗"
    psnr_s    = f"{psnr:.1f}"   if psnr != float("inf") else "∞"
    cx_fp16_s  = f"{cx_fp16}x"  if cx_fp16  is not None else "--"
    cx_nvfp4_s = f"{cx_nvfp4}x" if cx_nvfp4 is not None else "--"
    print(f"  {name:<32} {fid:>8.1f} {is_v:>7.4f} {psnr_s:>7} {ssim:>7.4f} "
          f"{cx_fp16_s:>9} {cx_nvfp4_s:>10}  {beat}")

# --- Compression detail table ---
print(f"\n  Compression detail (representative layer 4608×1152):")
print(f"  {'Config':<28} {'D bits':>7} {'total_MB':>9} {'vs_FP16':>9} {'vs_NVFP4':>10}  recon_err")
print("  " + "-" * 70)
bits_cfg = {"FP16":16,"FP8":8,"NVFP4":4,"INT3":3}
m_r, n_r = 4608, 1152
for dm, K, s in [("FP16",256,16),("FP8",256,16),("NVFP4",256,16),("INT3",256,16),("NVFP4",128,16)]:
    bits_d   = bits_cfg[dm]
    bits_val = 4  # NVFP4 reconstruction
    bits_idx = math.ceil(math.log2(max(K,2)))
    total    = m_r*K*bits_d + s*n_r*(bits_val+bits_idx)
    fp16_b   = m_r*n_r*16
    nvfp4_b  = m_r*n_r*4
    total_mb = total / 8 / 1024 / 1024
    tag = f"K={K} s={s} D={dm}"
    path = f"{base}/results/ksvd_experiment/KSVD_K{K}_S{s}_D{dm}/{ds}/metrics.json"
    d = load(path)
    recon_e = d["compression"].get("mean_reconstruction_error", "--") if d else "--"
    print(f"  {tag:<28} {bits_d:>7}  {total_mb:>8.2f}MB "
          f"{fp16_b/total:>8.1f}x {nvfp4_b/total:>9.1f}x  {recon_e}")
print()
PYEOF

echo "Done: $(date '+%Y-%m-%d %H:%M:%S')"
