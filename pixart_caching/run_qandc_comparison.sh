#!/usr/bin/env bash
# run_qandc_comparison.sh
# Q&C (ICLR 2026) 근사 구현 vs 우리 방법 비교
#
# ⚠️  Q&C 공식 코드 미확보 — TAP/VC 근사 구현:
#   TAP (Timestep-Aware Perturbation):
#     calibration을 5-step이 아닌 full 15-step으로 실행
#     → 모든 timestep 구간의 activation 분포를 NVFP4 scaling에 반영
#   VC (Variance Calibration):
#     cached step에서 residual을 현재/저장 시점 std 비율로 스케일링
#     → distribution shift 보정
#
# 비교 구성 (4가지):
#   [1] Ours         : rank=32, DeepCache, no TAP/VC   (조합1 재현)
#   [2] Q&C approx   : rank=0,  DeepCache, TAP+VC      (Q&C 근사)
#   [3] Q&C+SVD      : rank=32, DeepCache, TAP+VC      (Q&C 기법 + SVD 추가)
#   [4] VC only      : rank=32, DeepCache, VC only      (ablation)
#
# 결과: results/MJHQ/deepcache/qandc_comparison_summary.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[ -f ~/.env ] && source ~/.env

MODEL_PATH="PixArt-alpha/PixArt-XL-2-1024-MS"
DATASET="MJHQ"
NUM_STEPS=15
CACHE_INTERVAL=2
CACHE_START=8
CACHE_END=20
NUM_SAMPLES=20

REF_DIR="$SCRIPT_DIR/ref_images"
SAVE_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

BASE_ARGS="
    --model_path $MODEL_PATH
    --dataset_name $DATASET
    --num_inference_steps $NUM_STEPS
    --cache_interval $CACHE_INTERVAL
    --cache_start $CACHE_START
    --cache_end $CACHE_END
    --num_samples $NUM_SAMPLES
    --ref_dir $REF_DIR
    --save_dir $SAVE_DIR
"

# =============================================================================
# [1] Ours: rank=32, DeepCache, standard calib (조합1 재현)
# =============================================================================
echo ""
echo "============================================================"
echo " [1/4] Ours: rank=32 + DeepCache (조합1)"
echo "============================================================"
LOG1="$LOG_DIR/qandc_ours_${TIMESTAMP}.log"
accelerate launch --num_processes 1 \
    "$SCRIPT_DIR/pixart_deepcache_experiment.py" \
    $BASE_ARGS \
    --lowrank 32 \
    2>&1 | tee "$LOG1"
echo "Done [1/4]. Log: $LOG1"

# =============================================================================
# [2] Q&C approx: rank=0, TAP+VC (SVD 없음, Q&C 핵심 기법만)
# =============================================================================
echo ""
echo "============================================================"
echo " [2/4] Q&C approx: rank=0 + TAP + VC"
echo "============================================================"
LOG2="$LOG_DIR/qandc_approx_${TIMESTAMP}.log"
accelerate launch --num_processes 1 \
    "$SCRIPT_DIR/pixart_deepcache_experiment.py" \
    $BASE_ARGS \
    --lowrank 0 \
    --qandc \
    2>&1 | tee "$LOG2"
echo "Done [2/4]. Log: $LOG2"

# =============================================================================
# [3] Q&C+SVD: rank=32, TAP+VC (Q&C 기법 + SVD branch 조합)
# =============================================================================
echo ""
echo "============================================================"
echo " [3/4] Q&C+SVD: rank=32 + TAP + VC"
echo "============================================================"
LOG3="$LOG_DIR/qandc_svd_${TIMESTAMP}.log"
accelerate launch --num_processes 1 \
    "$SCRIPT_DIR/pixart_deepcache_experiment.py" \
    $BASE_ARGS \
    --lowrank 32 \
    --qandc \
    2>&1 | tee "$LOG3"
echo "Done [3/4]. Log: $LOG3"

# =============================================================================
# [4] VC only (ablation): rank=32, VC만 적용 (TAP 없음)
# =============================================================================
echo ""
echo "============================================================"
echo " [4/4] VC only: rank=32 + VC (TAP 없음, ablation)"
echo "============================================================"
LOG4="$LOG_DIR/qandc_vc_only_${TIMESTAMP}.log"
accelerate launch --num_processes 1 \
    "$SCRIPT_DIR/pixart_deepcache_experiment.py" \
    $BASE_ARGS \
    --lowrank 32 \
    --use_vc \
    2>&1 | tee "$LOG4"
echo "Done [4/4]. Log: $LOG4"

# =============================================================================
# 결과 집계
# =============================================================================
echo ""
echo "============================================================"
echo " 결과 집계"
echo "============================================================"
python3 << 'PYEOF'
import re, sys

logs = {
    "[1] Ours (rank=32, no TAP/VC)":      "qandc_ours",
    "[2] Q&C approx (rank=0, TAP+VC)":    "qandc_approx",
    "[3] Q&C+SVD (rank=32, TAP+VC)":      "qandc_svd",
    "[4] VC only (rank=32, VC)":           "qandc_vc_only",
}

import glob, os
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
import csv

results = []
for label, prefix in logs.items():
    files = sorted(glob.glob(f"{log_dir}/{prefix}_*.log"), reverse=True)
    if not files:
        print(f"  {label}: log not found")
        continue
    content = open(files[0]).read()
    m = re.search(r'FID=([\d.]+) \| IS=([\d.]+) \| PSNR=([\d.]+) \| time=([\d.]+)s', content)
    if m:
        fid, is_, psnr, time_ = m.groups()
        print(f"  {label}: FID={fid} IS={is_} PSNR={psnr} time={time_}s")
        results.append({"method": label, "FID": fid, "IS": is_, "PSNR": psnr, "time": time_})
    else:
        print(f"  {label}: result not found in log")

if results:
    import csv
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "MJHQ", "deepcache", "qandc_comparison_summary.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "FID", "IS", "PSNR", "time"])
        w.writeheader()
        w.writerows(results)
    print(f"\nCSV saved: {out_path}")
PYEOF

echo ""
echo "All Q&C comparison experiments complete."
echo "Logs: $LOG_DIR/qandc_*_${TIMESTAMP}.log"
