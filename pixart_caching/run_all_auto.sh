#!/usr/bin/env bash
# run_all_auto.sh
# 자동 순차 실행: 현재 실험 완료 대기 → ref 생성 → 수렴 분석
#
# 실행: bash run_all_auto.sh &
# 로그: logs/auto_orchestration_TIMESTAMP.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[ -f ~/.env ] && source ~/.env

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/auto_orchestration_${TIMESTAMP}.log"

# tee로 로그 파일과 stdout 동시 출력
exec > >(tee -a "$MASTER_LOG") 2>&1

notify() {
    local msg="$1"
    local now
    now=$(date '+%Y-%m-%d %H:%M:%S KST')
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    printf  "║  %-62s║\n" "[$now]"
    printf  "║  %-62s║\n" "$msg"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
}

elapsed_since() {
    local start=$1
    local now
    now=$(date +%s)
    local sec=$(( now - start ))
    printf "%dm %ds" $(( sec / 60 )) $(( sec % 60 ))
}

GLOBAL_START=$(date +%s)

notify "🚀 자동 오케스트레이션 시작"
echo "마스터 로그: $MASTER_LOG"
echo ""
echo "실행 예정 단계:"
echo "  [0] 현재 실험(Deepcache_for_NVFP4) 완료 대기"
echo "  [1] fp16_steps20 ref 이미지 생성 (~55분)"
echo "  [2] 수렴 분석 Method A - NVFP4 20-step (~57분)"
echo "  [3] 수렴 분석 Method B - NVFP4 15-step (~49분)"
echo "  [4] 수렴 분석 Method C - Ours (~33분)"
echo "  [5] FID/IS N별 재계산 + 최종 테이블"
echo ""
echo "  ※ ref 이미지: fp16_steps20 단일 기준 (A/B/C 모두 동일)"
echo ""

# =============================================================================
# [0] 현재 실험 완료 대기
# =============================================================================
notify "⏳ [0/6] 현재 실험 완료 대기 중..."

WAIT_PIDS=$(pgrep -f "pixart_nvfp4_cache_compare.py" 2>/dev/null || true)

if [ -n "$WAIT_PIDS" ]; then
    echo "대기 중인 PID: $WAIT_PIDS"
    POLL_COUNT=0
    while pgrep -f "pixart_nvfp4_cache_compare.py" > /dev/null 2>&1; do
        POLL_COUNT=$(( POLL_COUNT + 1 ))
        REMAINING=$(pgrep -f "pixart_nvfp4_cache_compare.py" | wc -l)
        if [ $(( POLL_COUNT % 6 )) -eq 0 ]; then
            echo "  $(date '+%H:%M:%S')  아직 ${REMAINING}개 실험 실행 중... ($(elapsed_since $GLOBAL_START) 경과)"
        fi
        sleep 10
    done
    notify "✅ [0/6] 현재 실험 완료! ($(elapsed_since $GLOBAL_START) 경과)"
else
    echo "실행 중인 실험 없음 — 바로 진행합니다."
fi

STEP0_END=$(date +%s)

# =============================================================================
# [1] fp16_steps20 ref 생성 (이미 1000장 있으면 스킵)
# =============================================================================
REF_DIR=/data/james_dit_pixart_xl_mjhq/fp16_steps20
REF_COUNT=$(ls "$REF_DIR/MJHQ"/ref_*.png 2>/dev/null | wc -l || echo 0)

if [ "$REF_COUNT" -ge 1000 ]; then
    notify "⏭️  [1/5] fp16_steps20 ref 이미 ${REF_COUNT}장 존재 — 스킵"
else
    notify "🖼️  [1/5] fp16_steps20 reference 이미지 생성 시작 (현재 ${REF_COUNT}장, 목표 1000장)"
    STEP1_START=$(date +%s)
    accelerate launch --num_processes 1 \
        "$SCRIPT_DIR/generate_ref_images.py" \
        --precision fp16 \
        --num_inference_steps 20 \
        --num_samples 1000 \
        --out_dir /data/james_dit_pixart_xl_mjhq
    notify "✅ [1/5] fp16_steps20 완료! (소요: $(elapsed_since $STEP1_START))"
fi

# =============================================================================
# [2] 수렴 분석: Method A — NVFP4 20-step, no cache
# =============================================================================
SAVE_A="$SCRIPT_DIR/results/MJHQ/deepcache/interval1_s8_e20_gs4.5_steps20"
A_COUNT=$(ls "$SAVE_A"/sample_*.png 2>/dev/null | wc -l || echo 0)
LOG_A=$(ls "$LOG_DIR"/conv_A_nvfp4_20step_*.log 2>/dev/null | sort | tail -1 || true)

if [ "$A_COUNT" -ge 1000 ] && [ -n "$LOG_A" ] && grep -q "FID=" "$LOG_A" 2>/dev/null; then
    TIME_A=$(grep -oP 'time=\K[\d.]+' "$LOG_A" 2>/dev/null | tail -1 || echo 0)
    notify "⏭️  [2/5] Method A 이미 완료 (${A_COUNT}장, FID존재) — 스킵  time/img=${TIME_A}s"
else
    notify "📊 [2/5] 수렴 분석 Method A: NVFP4 20-step (no cache, 1000 samples, ~57분 예상)"
    STEP2_START=$(date +%s)
    LOG_A="$LOG_DIR/conv_A_nvfp4_20step_${TIMESTAMP}.log"
    accelerate launch --num_processes 1 \
        "$SCRIPT_DIR/pixart_deepcache_experiment.py" \
        --model_path "PixArt-alpha/PixArt-XL-2-1024-MS" \
        --dataset_name MJHQ \
        --lowrank 32 \
        --num_inference_steps 20 \
        --cache_interval 1 \
        --cache_start 8 \
        --cache_end 20 \
        --num_samples 1000 \
        --ref_dir "$REF_DIR" \
        --save_dir "$SCRIPT_DIR/results" \
        --skip_existing \
        2>&1 | tee "$LOG_A"
    TIME_A=$(grep -oP 'time=\K[\d.]+' "$LOG_A" 2>/dev/null | tail -1 || echo 0)
    notify "✅ [2/5] Method A 완료! time/img=${TIME_A}s  (소요: $(elapsed_since $STEP2_START))"
fi

# =============================================================================
# [3] 수렴 분석: Method B — NVFP4 15-step, no cache
# =============================================================================
SAVE_B="$SCRIPT_DIR/results/MJHQ/deepcache/interval1_s8_e20_gs4.5_steps15"
B_COUNT=$(ls "$SAVE_B"/sample_*.png 2>/dev/null | wc -l || echo 0)
LOG_B=$(ls "$LOG_DIR"/conv_B_nvfp4_15step_*.log 2>/dev/null | sort | tail -1 || true)

if [ "$B_COUNT" -ge 1000 ] && [ -n "$LOG_B" ] && grep -q "FID=" "$LOG_B" 2>/dev/null; then
    TIME_B=$(grep -oP 'time=\K[\d.]+' "$LOG_B" 2>/dev/null | tail -1 || echo 0)
    notify "⏭️  [3/5] Method B 이미 완료 (${B_COUNT}장) — 스킵  time/img=${TIME_B}s"
else
    notify "📊 [3/5] 수렴 분석 Method B: NVFP4 15-step (no cache, 1000 samples, ~49분 예상)"
    STEP3_START=$(date +%s)
    LOG_B="$LOG_DIR/conv_B_nvfp4_15step_${TIMESTAMP}.log"
    accelerate launch --num_processes 1 \
        "$SCRIPT_DIR/pixart_deepcache_experiment.py" \
        --model_path "PixArt-alpha/PixArt-XL-2-1024-MS" \
        --dataset_name MJHQ \
        --lowrank 32 \
        --num_inference_steps 15 \
        --cache_interval 1 \
        --cache_start 8 \
        --cache_end 20 \
        --num_samples 1000 \
        --ref_dir "$REF_DIR" \
        --save_dir "$SCRIPT_DIR/results" \
        --skip_existing \
        2>&1 | tee "$LOG_B"
    TIME_B=$(grep -oP 'time=\K[\d.]+' "$LOG_B" 2>/dev/null | tail -1 || echo 0)
    notify "✅ [3/5] Method B 완료! time/img=${TIME_B}s  (소요: $(elapsed_since $STEP3_START))"
fi

# =============================================================================
# [4] 수렴 분석: Method C — Ours (15-step + DeepCache)
# =============================================================================
SAVE_C="$SCRIPT_DIR/results/MJHQ/deepcache/interval2_s8_e20_gs4.5_steps15"
C_COUNT=$(ls "$SAVE_C"/sample_*.png 2>/dev/null | wc -l || echo 0)
LOG_C=$(ls "$LOG_DIR"/conv_C_ours_*.log 2>/dev/null | sort | tail -1 || true)

if [ "$C_COUNT" -ge 1000 ] && [ -n "$LOG_C" ] && grep -q "FID=" "$LOG_C" 2>/dev/null; then
    TIME_C=$(grep -oP 'time=\K[\d.]+' "$LOG_C" 2>/dev/null | tail -1 || echo 0)
    notify "⏭️  [4/5] Method C 이미 완료 (${C_COUNT}장) — 스킵  time/img=${TIME_C}s"
else
    notify "📊 [4/5] 수렴 분석 Method C: Ours 15-step+DeepCache (1000 samples, ~33분 예상)"
    STEP4_START=$(date +%s)
    LOG_C="$LOG_DIR/conv_C_ours_${TIMESTAMP}.log"
    accelerate launch --num_processes 1 \
        "$SCRIPT_DIR/pixart_deepcache_experiment.py" \
        --model_path "PixArt-alpha/PixArt-XL-2-1024-MS" \
        --dataset_name MJHQ \
        --lowrank 32 \
        --num_inference_steps 15 \
        --cache_interval 2 \
        --cache_start 8 \
        --cache_end 20 \
        --num_samples 1000 \
        --ref_dir "$REF_DIR" \
        --save_dir "$SCRIPT_DIR/results" \
        --skip_existing \
        2>&1 | tee "$LOG_C"
    TIME_C=$(grep -oP 'time=\K[\d.]+' "$LOG_C" 2>/dev/null | tail -1 || echo 0)
    notify "✅ [4/5] Method C 완료! time/img=${TIME_C}s  (소요: $(elapsed_since $STEP4_START))"
fi

# =============================================================================
# [5] FID/IS N별 재계산 (eval_convergence.py)
# =============================================================================
notify "🔢 [5/5] FID/IS 수렴 평가 시작 (N=20,100,250,500,750,1000 × 3 methods)"
STEP5_START=$(date +%s)

CONV_DIR="$SCRIPT_DIR/results/convergence"
mkdir -p "$CONV_DIR"

SAVE_A="$SCRIPT_DIR/results/MJHQ/deepcache/interval1_s8_e20_gs4.5_steps20"
SAVE_B="$SCRIPT_DIR/results/MJHQ/deepcache/interval1_s8_e20_gs4.5_steps15"
SAVE_C="$SCRIPT_DIR/results/MJHQ/deepcache/interval2_s8_e20_gs4.5_steps15"
NS="20,100,250,500,750,1000"
REF_EVAL=/data/james_dit_pixart_xl_mjhq/fp16_steps20/MJHQ

python3 "$SCRIPT_DIR/eval_convergence.py" \
    --sample_dir "$SAVE_A" \
    --ref_dir    "$REF_EVAL" \
    --method     "NVFP4 20-step (baseline)" \
    --ns         "$NS" \
    --time_per_img  "${TIME_A:-3.45}" \
    --baseline_time "${TIME_A:-3.45}" \
    --out        "$CONV_DIR/A_nvfp4_20step_conv.csv"

python3 "$SCRIPT_DIR/eval_convergence.py" \
    --sample_dir "$SAVE_B" \
    --ref_dir    "$REF_EVAL" \
    --method     "NVFP4 15-step (no cache)" \
    --ns         "$NS" \
    --time_per_img  "${TIME_B:-2.94}" \
    --baseline_time "${TIME_A:-3.45}" \
    --out        "$CONV_DIR/B_nvfp4_15step_conv.csv"

python3 "$SCRIPT_DIR/eval_convergence.py" \
    --sample_dir "$SAVE_C" \
    --ref_dir    "$REF_EVAL" \
    --method     "Ours (15-step + DeepCache)" \
    --ns         "$NS" \
    --time_per_img  "${TIME_C:-1.96}" \
    --baseline_time "${TIME_A:-3.45}" \
    --out        "$CONV_DIR/C_ours_conv.csv"

STEP5_END=$(date +%s)
notify "✅ [5/5] 수렴 평가 완료! (소요: $(elapsed_since $STEP5_START))"

# =============================================================================
# 최종 결과 테이블
# =============================================================================
python3 << 'PYEOF'
import csv, os, glob

conv_dir = "/home/jameskimh/workspace/Workspace_DiT/pixart_caching/results/convergence"
files = sorted(glob.glob(f"{conv_dir}/*_conv.csv"))
all_rows = []
for f in files:
    with open(f) as fp:
        all_rows.extend(list(csv.DictReader(fp)))

if all_rows:
    out = os.path.join(conv_dir, "summary.csv")
    with open(out, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["method","N","FID","IS","time_per_img","speedup"])
        w.writeheader()
        w.writerows(all_rows)

    print(f"\n{'Method':<30} {'N':>5} {'FID':>8} {'IS':>7} {'time':>7} {'speedup':>8}")
    print("─" * 72)
    cur_method = None
    for r in all_rows:
        if r["method"] != cur_method:
            if cur_method is not None:
                print()
            cur_method = r["method"]
        sp = f"{float(r['speedup']):.2f}x" if r.get('speedup') else "-"
        print(f"  {r['method']:<28} {int(r['N']):>5} {float(r['FID']):>8.2f} "
              f"{float(r['IS']):>7.4f} {float(r['time_per_img']):>6.2f}s {sp:>8}")
    print(f"\nCSV: {out}")
PYEOF

notify "🎉 [완료] 전체 파이프라인 완료! 총 소요: $(elapsed_since $GLOBAL_START)"
echo ""
echo "결과 위치:"
echo "  수렴 분석: $CONV_DIR/summary.csv"
echo "  마스터 로그: $MASTER_LOG"
