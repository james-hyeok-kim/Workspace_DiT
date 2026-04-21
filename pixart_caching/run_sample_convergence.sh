#!/usr/bin/env bash
# run_sample_convergence.sh
# FID/IS 수렴 분석: 3가지 방법 × N={20,100,250,500,750,1000} 샘플
#
# 비교 방법:
#   [A] NVFP4_DEFAULT, 20-step, no cache   (공식 baseline)
#   [B] NVFP4_DEFAULT, 15-step, no cache   (step 축소 비교군)
#   [C] Ours: NVFP4+SVD+DeepCache, 15-step (우리 방식)
#
# 전략:
#   - 각 방법별 1000개 이미지 생성 (--skip_existing 으로 중복 생성 방지)
#   - 생성 완료 후 eval_convergence.py 로 N별 FID/IS 재계산
#   - FP16 ref는 step count별로 분리 저장
#
# 출력:
#   results/convergence/summary.csv       ← 최종 통합 테이블
#   results/convergence/{method}_conv.csv ← 방법별 수렴 곡선

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[ -f ~/.env ] && source ~/.env

MODEL_PATH="PixArt-alpha/PixArt-XL-2-1024-MS"
DATASET="MJHQ"
N_MAX=1000
NS="20,100,250,500,750,1000"
CACHE_START=8
CACHE_END=20

SAVE_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"
CONV_DIR="$SCRIPT_DIR/results/convergence"
mkdir -p "$LOG_DIR" "$CONV_DIR"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# ref 이미지는 /data/jameskimh/james_dit_pixart_xl_mjhq/{precision}_steps{N}/ 사용
# 없으면 먼저 bash run_generate_refs.sh 실행
REF_DIR_20="/data/jameskimh/james_dit_pixart_xl_mjhq/fp16_steps20"
REF_DIR_15="/data/jameskimh/james_dit_pixart_xl_mjhq/fp16_steps15"

for d in "$REF_DIR_20" "$REF_DIR_15"; do
    if [ ! -d "$d" ] || [ "$(ls "$d"/ref_*.png 2>/dev/null | wc -l)" -lt "$N_MAX" ]; then
        echo "⚠️  Ref images not ready: $d"
        echo "   Run first: bash run_generate_refs.sh"
        exit 1
    fi
done

# 저장 디렉토리 (config_tag 기반, 자동 생성)
SAVE_A="$SAVE_DIR/$DATASET/deepcache/interval1_s${CACHE_START}_e${CACHE_END}_gs4.5_steps20"
SAVE_B="$SAVE_DIR/$DATASET/deepcache/interval1_s${CACHE_START}_e${CACHE_END}_gs4.5_steps15"
SAVE_C="$SAVE_DIR/$DATASET/deepcache/interval2_s${CACHE_START}_e${CACHE_END}_gs4.5_steps15"

echo ""
echo "================================================================"
echo " FID/IS 수렴 분석: 3방법 × 1000 samples"
echo " N 구간: $NS"
echo "================================================================"

# =============================================================================
# [A] NVFP4 20-step, no cache (공식 baseline)
# =============================================================================
echo ""
echo "────────────────────────────────────────────────────────────────"
echo " [A/3] NVFP4 20-step, no cache  →  $SAVE_A"
echo "────────────────────────────────────────────────────────────────"
LOG_A="$LOG_DIR/conv_A_nvfp4_20step_${TIMESTAMP}.log"
accelerate launch --num_processes 1 \
    "$SCRIPT_DIR/pixart_deepcache_experiment.py" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET" \
    --lowrank 32 \
    --num_inference_steps 20 \
    --cache_interval 1 \
    --cache_start "$CACHE_START" \
    --cache_end "$CACHE_END" \
    --num_samples "$N_MAX" \
    --ref_dir "$REF_DIR_20" \
    --save_dir "$SAVE_DIR" \
    --skip_existing \
    2>&1 | tee "$LOG_A"
echo "Done [A]. Log: $LOG_A"

# timing 추출
TIME_A=$(grep -oP 'time=\K[\d.]+' "$LOG_A" | tail -1)
echo "  time/img (A): ${TIME_A}s"

# =============================================================================
# [B] NVFP4 15-step, no cache (step 축소 비교군)
# =============================================================================
echo ""
echo "────────────────────────────────────────────────────────────────"
echo " [B/3] NVFP4 15-step, no cache  →  $SAVE_B"
echo "────────────────────────────────────────────────────────────────"
LOG_B="$LOG_DIR/conv_B_nvfp4_15step_${TIMESTAMP}.log"
accelerate launch --num_processes 1 \
    "$SCRIPT_DIR/pixart_deepcache_experiment.py" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET" \
    --lowrank 32 \
    --num_inference_steps 15 \
    --cache_interval 1 \
    --cache_start "$CACHE_START" \
    --cache_end "$CACHE_END" \
    --num_samples "$N_MAX" \
    --ref_dir "$REF_DIR_15" \
    --save_dir "$SAVE_DIR" \
    --skip_existing \
    2>&1 | tee "$LOG_B"
echo "Done [B]. Log: $LOG_B"

TIME_B=$(grep -oP 'time=\K[\d.]+' "$LOG_B" | tail -1)
echo "  time/img (B): ${TIME_B}s"

# =============================================================================
# [C] Ours: NVFP4+SVD+DeepCache, 15-step
# =============================================================================
echo ""
echo "────────────────────────────────────────────────────────────────"
echo " [C/3] Ours: 15-step + DeepCache interval=2 [8,20)  →  $SAVE_C"
echo "────────────────────────────────────────────────────────────────"
LOG_C="$LOG_DIR/conv_C_ours_${TIMESTAMP}.log"
accelerate launch --num_processes 1 \
    "$SCRIPT_DIR/pixart_deepcache_experiment.py" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET" \
    --lowrank 32 \
    --num_inference_steps 15 \
    --cache_interval 2 \
    --cache_start "$CACHE_START" \
    --cache_end "$CACHE_END" \
    --num_samples "$N_MAX" \
    --ref_dir "$REF_DIR_15" \
    --save_dir "$SAVE_DIR" \
    --skip_existing \
    2>&1 | tee "$LOG_C"
echo "Done [C]. Log: $LOG_C"

TIME_C=$(grep -oP 'time=\K[\d.]+' "$LOG_C" | tail -1)
echo "  time/img (C): ${TIME_C}s"

# =============================================================================
# FID/IS 수렴 평가 (eval_convergence.py)
# =============================================================================
echo ""
echo "================================================================"
echo " FID/IS 수렴 평가 (N별 재계산)"
echo "================================================================"

python3 "$SCRIPT_DIR/eval_convergence.py" \
    --sample_dir "$SAVE_A" \
    --ref_dir    "$REF_DIR_20" \
    --method     "NVFP4 20-step (baseline)" \
    --ns         "$NS" \
    --time_per_img  "${TIME_A:-3.45}" \
    --baseline_time "${TIME_A:-3.45}" \
    --out        "$CONV_DIR/A_nvfp4_20step_conv.csv"

python3 "$SCRIPT_DIR/eval_convergence.py" \
    --sample_dir "$SAVE_B" \
    --ref_dir    "$REF_DIR_15" \
    --method     "NVFP4 15-step (no cache)" \
    --ns         "$NS" \
    --time_per_img  "${TIME_B:-2.94}" \
    --baseline_time "${TIME_A:-3.45}" \
    --out        "$CONV_DIR/B_nvfp4_15step_conv.csv"

python3 "$SCRIPT_DIR/eval_convergence.py" \
    --sample_dir "$SAVE_C" \
    --ref_dir    "$REF_DIR_15" \
    --method     "Ours (15-step + DeepCache)" \
    --ns         "$NS" \
    --time_per_img  "${TIME_C:-1.96}" \
    --baseline_time "${TIME_A:-3.45}" \
    --out        "$CONV_DIR/C_ours_conv.csv"

# =============================================================================
# 통합 테이블 출력
# =============================================================================
echo ""
echo "================================================================"
echo " 최종 통합 결과"
echo "================================================================"
python3 << 'PYEOF'
import csv, os, glob

conv_dir = os.path.join(os.path.dirname(os.path.abspath(".")),
    "workspace/Workspace_DiT/pixart_caching/results/convergence")
conv_dir = "/home/jameskimh/workspace/Workspace_DiT/pixart_caching/results/convergence"

files = sorted(glob.glob(f"{conv_dir}/*_conv.csv"))
all_rows = []
for f in files:
    with open(f) as fp:
        all_rows.extend(list(csv.DictReader(fp)))

# 통합 CSV 저장
out = os.path.join(conv_dir, "summary.csv")
if all_rows:
    with open(out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["method","N","FID","IS","time_per_img","speedup"])
        w.writeheader()
        w.writerows(all_rows)

# 테이블 출력
print(f"\n{'Method':<30} {'N':>5} {'FID':>8} {'IS':>7} {'time':>7} {'speedup':>8}")
print("-" * 72)
cur_method = None
for r in all_rows:
    if r["method"] != cur_method:
        if cur_method is not None:
            print()
        cur_method = r["method"]
    sp = f"{r['speedup']}x" if r['speedup'] else "-"
    print(f"  {r['method']:<28} {r['N']:>5} {float(r['FID']):>8.2f} "
          f"{float(r['IS']):>7.4f} {float(r['time_per_img']):>6.2f}s {sp:>8}")

print(f"\nCSV: {out}")
PYEOF

echo ""
echo "All done. Logs: $LOG_DIR/conv_*_${TIMESTAMP}.log"
echo "Results: $CONV_DIR/"
