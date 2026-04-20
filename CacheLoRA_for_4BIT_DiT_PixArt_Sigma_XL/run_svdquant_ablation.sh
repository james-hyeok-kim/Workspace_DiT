#!/bin/bash
# run_svdquant_ablation.sh
# SVDQUANT-specific ablation: interval / rank / NOLR
#
# 실험 매트릭스 (SVDQUANT, range=[8,20) 고정, steps={10,15,20}):
#   Axis 1: deepcache_interval={3,4} × cache={deepcache,cache_lora} × steps  = 12 runs
#   Axis 2: lora_rank={8,16,32}      × cache=cache_lora               × steps =  9 runs
#   Axis 3: NOLR                     × cache={none,deepcache,cl}      × steps =  9 runs
#   Total: 30 runs
#
# 사용법:
#   bash run_svdquant_ablation.sh                          # 20 samples
#   bash run_svdquant_ablation.sh --gpu 0                  # GPU 지정
#   bash run_svdquant_ablation.sh --num_samples 100        # 100 samples 확대
#   bash run_svdquant_ablation.sh --axis interval          # 특정 axis만
#   bash run_svdquant_ablation.sh --test_run               # smoke test (2 samples)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/pixart_nvfp4_cache_compare.py"
ENV_PYTHON="/home/jameskimh/.dit/bin/accelerate"

# ── 기본 설정 ─────────────────────────────────────────────────────────────────
NUM_SAMPLES=20   # quick test default
TEST_RUN=false
TEST_FLAG=""
GPU_ID=""
AXIS="all"       # interval | rank | nolr | all

# 고정 파라미터
STEPS=(10 15 20)
CS=8; CE=20      # range [8,20) — SVDQUANT 최적 range
BASE_LORA_RANK=4
BASE_INTERVAL=2

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --test_run)      TEST_RUN=true; TEST_FLAG="--test_run" ;;
        --gpu)           GPU_ID="$2"; shift ;;
        --num_samples)   NUM_SAMPLES="$2"; shift ;;
        --axis)          AXIS="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "Using GPU: $GPU_ID"
fi

# ── 헬퍼: 단일 run ────────────────────────────────────────────────────────────
run_one() {
    local METHOD="$1" CACHE="$2" STEP="$3"
    shift 3
    local EXTRA_FLAGS="$@"  # --deepcache_interval, --lora_rank, --disable_svdquant_lr 등

    # run_tag 구성 (스크립트와 동일 로직으로 skip 판단용)
    local LORA_RANK=$BASE_LORA_RANK
    local INTERVAL=$BASE_INTERVAL
    local NOLR=false

    # extra flags 파싱
    local tmp_flags=($EXTRA_FLAGS)
    local i=0
    while [ $i -lt ${#tmp_flags[@]} ]; do
        case "${tmp_flags[$i]}" in
            --lora_rank)          i=$((i+1)); LORA_RANK="${tmp_flags[$i]}" ;;
            --deepcache_interval) i=$((i+1)); INTERVAL="${tmp_flags[$i]}" ;;
            --disable_svdquant_lr) NOLR=true ;;
        esac
        i=$((i+1))
    done

    local METHOD_TAG="$METHOD"
    if [ "$NOLR" = true ]; then METHOD_TAG="${METHOD}_nolr"; fi
    local INTERVAL_TAG=""
    if [ "$INTERVAL" != "$BASE_INTERVAL" ]; then INTERVAL_TAG="_i${INTERVAL}"; fi

    local CACHE_TAG="$CACHE"
    if [ "$CACHE" = "cache_lora" ]; then CACHE_TAG="cache_lora_r${LORA_RANK}"; fi

    local TAG
    if [ "$CACHE" = "none" ]; then
        TAG="${METHOD_TAG}_${CACHE_TAG}${INTERVAL_TAG}_steps${STEP}"
    else
        TAG="${METHOD_TAG}_${CACHE_TAG}${INTERVAL_TAG}_c${CS}-${CE}_steps${STEP}"
    fi

    RESULT_DIR="$SCRIPT_DIR/results/MJHQ/${TAG}"
    if [ -f "$RESULT_DIR/metrics.json" ] && ! $TEST_RUN; then
        echo "  SKIP: $TAG"
        return
    fi

    echo ""
    echo "  Running: $TAG"
    echo "  ─────────────────────────────────────────"

    local RANGE_FLAGS=""
    local LORA_FLAGS=""
    [ "$CACHE" != "none" ] && RANGE_FLAGS="--cache_start $CS --cache_end $CE"
    [ "$CACHE" = "cache_lora" ] && LORA_FLAGS="--lora_rank $LORA_RANK --lora_calib 4"

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
        $EXTRA_FLAGS \
        $TEST_FLAG

    echo "  Done: $TAG"
}

# ── 실행 시작 ─────────────────────────────────────────────────────────────────
if $TEST_RUN; then
    echo "=== SVDQUANT ABLATION — SMOKE TEST (2 samples) ==="
else
    echo "=== SVDQUANT ABLATION SWEEP ($NUM_SAMPLES samples) ==="
fi
echo "Axis  : $AXIS"
echo "Steps : ${STEPS[*]}"
echo "Range : [${CS},${CE})"
echo ""

cd "$SCRIPT_DIR"

# ── Axis 1: Interval ──────────────────────────────────────────────────────────
if [[ "$AXIS" == "all" || "$AXIS" == "interval" ]]; then
    echo "======= Axis 1: deepcache_interval={3,4} ======="
    for INTERVAL in 3 4; do
        for CACHE in deepcache cache_lora; do
            for STEP in "${STEPS[@]}"; do
                run_one SVDQUANT "$CACHE" "$STEP" \
                    --deepcache_interval "$INTERVAL" \
                    --lora_rank "$BASE_LORA_RANK"
            done
        done
    done
fi

# ── Axis 2: Rank ──────────────────────────────────────────────────────────────
if [[ "$AXIS" == "all" || "$AXIS" == "rank" ]]; then
    echo ""
    echo "======= Axis 2: cache_lora rank={8,16,32} ======="
    for RANK in 8 16 32; do
        for STEP in "${STEPS[@]}"; do
            run_one SVDQUANT cache_lora "$STEP" \
                --lora_rank "$RANK"
        done
    done
fi

# ── Axis 3: NOLR ──────────────────────────────────────────────────────────────
if [[ "$AXIS" == "all" || "$AXIS" == "nolr" ]]; then
    echo ""
    echo "======= Axis 3: SVDQUANT_NOLR (internal LR removed) ======="
    for CACHE in none deepcache cache_lora; do
        for STEP in "${STEPS[@]}"; do
            run_one SVDQUANT "$CACHE" "$STEP" \
                --disable_svdquant_lr \
                --lora_rank "$BASE_LORA_RANK"
        done
    done
fi

echo ""
echo "============================================================"
echo "SVDQUANT ablation sweep complete."
echo "Results in: $SCRIPT_DIR/results/MJHQ/"
echo "============================================================"
