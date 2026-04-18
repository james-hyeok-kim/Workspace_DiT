#!/bin/bash
# run_step_range_sweep.sh
# Steps={5,10,15,20} × CacheRange={[8,20),[4,24),[2,26)} × 7 Methods × {none,deepcache,cache_lora}
# Methods: RTN, SVDQUANT, MRGPTQ, FOUROVERSIX, FP4DIT, HQDIT, SIXBIT
#
# 실행 순서 (빠른 결과 우선):
#   Phase 0: none baselines steps={5,10} (모든 method)
#   Phase 1: SVDQUANT 전체 sweep
#   Phase 2: RTN 전체 sweep
#   Phase 3: MRGPTQ 전체 sweep
#   Phase 4: FOUROVERSIX 전체 sweep
#   Phase 5: FP4DIT 전체 sweep
#   Phase 6: HQDIT 전체 sweep
#   Phase 7: SIXBIT 전체 sweep
#
# 사용법:
#   bash run_step_range_sweep.sh              # 전체
#   bash run_step_range_sweep.sh --test_run   # smoke test (2 samples)
#   bash run_step_range_sweep.sh --method FP4DIT
#   bash run_step_range_sweep.sh --method FP4DIT --cache deepcache

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/pixart_nvfp4_cache_compare.py"
ENV_PYTHON="/home/jameskimh/.dit/bin/accelerate"

# ── 기본 설정 ─────────────────────────────────────────────────────────────────
NUM_SAMPLES=20
TEST_RUN=false
TEST_FLAG=""
LORA_RANK=4
LORA_CALIB=4
CALIB_SEED_OFFSET=1000

# 실험 매트릭스
ALL_METHODS=("RTN" "SVDQUANT" "MRGPTQ" "FOUROVERSIX" "FP4DIT" "HQDIT" "SIXBIT")
STEPS=(5 10 15 20)
# cache_start cache_end pairs
RANGES=("8 20" "4 24" "2 26")
CACHES=("none" "deepcache" "cache_lora")

# 필터 (기본: 전체)
FILTER_METHODS=()
FILTER_CACHE=""

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --test_run)      TEST_RUN=true; TEST_FLAG="--test_run" ;;
        --method)        FILTER_METHODS=("$2"); shift ;;
        --cache)         FILTER_CACHE="$2"; shift ;;
        --num_samples)   NUM_SAMPLES="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

METHODS=("${FILTER_METHODS[@]:-${ALL_METHODS[@]}}")
if [ ${#FILTER_METHODS[@]} -eq 0 ]; then
    METHODS=("${ALL_METHODS[@]}")
fi

# ── 헬퍼: 단일 run ────────────────────────────────────────────────────────────
run_one() {
    local METHOD="$1" CACHE="$2" CS="$3" CE="$4" STEP="$5"

    # none 모드는 cache range 무관 → 디렉토리에 range 없음
    if [ "$CACHE" == "none" ]; then
        TAG="${METHOD}_none_steps${STEP}"
    else
        local CACHE_TAG="$CACHE"
        if [ "$CACHE" == "cache_lora" ]; then
            CACHE_TAG="cache_lora_r${LORA_RANK}"
        fi
        TAG="${METHOD}_${CACHE_TAG}_c${CS}-${CE}_steps${STEP}"
    fi

    RESULT_DIR="$SCRIPT_DIR/results/MJHQ/${TAG}"

    if [ -f "$RESULT_DIR/metrics.json" ] && ! $TEST_RUN; then
        echo "  ⏭️  SKIP: $TAG"
        return
    fi

    echo ""
    echo "  ▶️  Running: $TAG"
    echo "  Method=$METHOD  Cache=$CACHE  Range=[${CS},${CE})  Steps=$STEP"
    echo "  -----------------------------------------------------------"

    LORA_FLAGS=""
    if [ "$CACHE" == "cache_lora" ]; then
        LORA_FLAGS="--lora_rank $LORA_RANK --lora_calib $LORA_CALIB --calib_seed_offset $CALIB_SEED_OFFSET"
    fi

    RANGE_FLAGS=""
    if [ "$CACHE" != "none" ]; then
        RANGE_FLAGS="--cache_start $CS --cache_end $CE"
    fi

    $ENV_PYTHON launch \
        --num_processes 1 \
        "$PYTHON_SCRIPT" \
        --quant_method   "$METHOD" \
        --cache_mode     "$CACHE" \
        --num_steps      "$STEP" \
        --num_samples    "$NUM_SAMPLES" \
        --guidance_scale 4.5 \
        $RANGE_FLAGS \
        $LORA_FLAGS \
        $TEST_FLAG

    echo "  ✅ Done: $TAG"
}

# ── 모드 필터 적용 ────────────────────────────────────────────────────────────
should_run_cache() {
    local C="$1"
    if [ -n "$FILTER_CACHE" ]; then
        [ "$C" == "$FILTER_CACHE" ]
    else
        true
    fi
}

# ── 실행 시작 ─────────────────────────────────────────────────────────────────
if $TEST_RUN; then
    echo "=== SMOKE TEST MODE (2 samples) ==="
else
    echo "=== STEP × RANGE SWEEP ($NUM_SAMPLES samples) ==="
fi
echo "Methods : ${METHODS[*]}"
echo "Steps   : ${STEPS[*]}"
echo "Ranges  : [8,20) [4,24) [2,26)"
echo ""

cd "$SCRIPT_DIR"

# Phase 0: none baselines (cache range 무관, steps={5,10}만 새로 필요)
echo "======= Phase 0: none baselines (steps 5,10) ======="
for METHOD in "${METHODS[@]}"; do
    if should_run_cache "none"; then
        for STEP in 5 10; do
            run_one "$METHOD" "none" "8" "20" "$STEP"
        done
    fi
done

# Phase 1~4: 각 method별 deepcache + cache_lora × range × steps 전체
for METHOD in "${METHODS[@]}"; do
    echo ""
    echo "======= Method: $METHOD ======="

    # deepcache
    if should_run_cache "deepcache"; then
        echo "--- deepcache ---"
        for RANGE in "${RANGES[@]}"; do
            read -r CS CE <<< "$RANGE"
            for STEP in "${STEPS[@]}"; do
                run_one "$METHOD" "deepcache" "$CS" "$CE" "$STEP"
            done
        done
    fi

    # cache_lora
    if should_run_cache "cache_lora"; then
        echo "--- cache_lora (rank=$LORA_RANK) ---"
        for RANGE in "${RANGES[@]}"; do
            read -r CS CE <<< "$RANGE"
            for STEP in "${STEPS[@]}"; do
                run_one "$METHOD" "cache_lora" "$CS" "$CE" "$STEP"
            done
        done
    fi
done

echo ""
echo "============================================================"
echo "Step × Range sweep complete."
echo "Results in: $SCRIPT_DIR/results/MJHQ/"
echo "============================================================"
