#!/bin/bash
# run_experiment.sh
# NVFP4 × DeepCache 비교 실험 런처
#
# 사용법:
#   bash run_experiment.sh --test_run               # smoke test (2 samples)
#   bash run_experiment.sh                          # 전체 실험 (20 samples)
#   bash run_experiment.sh --method RTN             # 특정 method만
#   bash run_experiment.sh --method MRGPTQ          # MR-GPTQ만
#   bash run_experiment.sh --method FOUROVERSIX     # Four Over Six만
#   bash run_experiment.sh --steps 15               # 15-step만
#   bash run_experiment.sh --method MRGPTQ --cache deepcache --steps 20
#
# 결과: results/MJHQ/{METHOD}_{CACHE}_steps{N}/

set -e

# ── 기본 설정 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/pixart_nvfp4_cache_compare.py"
ENV_PYTHON="/home/jameskimh/.dit/bin/accelerate"

NUM_SAMPLES=20
TEST_RUN=false
METHODS=("RTN" "SVDQUANT" "MRGPTQ" "FOUROVERSIX")
CACHES=("none" "deepcache")
STEPS=(20 15)

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --test_run)      TEST_RUN=true ;;
        --method)        METHODS=("$2"); shift ;;
        --cache)         CACHES=("$2"); shift ;;
        --steps)         STEPS=("$2"); shift ;;
        --num_samples)   NUM_SAMPLES="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

if $TEST_RUN; then
    echo "=== SMOKE TEST MODE (2 samples) ==="
    TEST_FLAG="--test_run"
else
    TEST_FLAG=""
    echo "=== FULL EXPERIMENT ($NUM_SAMPLES samples) ==="
fi

echo "Methods : ${METHODS[*]}"
echo "Caches  : ${CACHES[*]}"
echo "Steps   : ${STEPS[*]}"
echo ""

# ── SVDQuant 결과 존재 여부 안내 ──────────────────────────────────────────────
# SVDQuant 20-step no-cache: FID=161.30 (기존 pixart_caching 실험 결과)
# SVDQuant 20-step deepcache: FID=159.43
# SVDQuant 15-step no-cache: FID=151.77
# SVDQuant 15-step deepcache: FID=162.99
# 위 결과가 있으면 SVDQUANT run은 skip 가능

# ── 실험 실행 ─────────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"

for METHOD in "${METHODS[@]}"; do
    for CACHE in "${CACHES[@]}"; do
        for STEP in "${STEPS[@]}"; do
            TAG="${METHOD}_${CACHE}_steps${STEP}"
            RESULT_DIR="$SCRIPT_DIR/results/MJHQ/${TAG}"

            # 이미 완료된 결과면 skip
            if [ -f "$RESULT_DIR/metrics.json" ] && ! $TEST_RUN; then
                echo "⏭️  SKIP (already exists): $TAG"
                continue
            fi

            echo ""
            echo "▶️  Running: $TAG"
            echo "   Method=$METHOD  Cache=$CACHE  Steps=$STEP"
            echo "-----------------------------------------------------------"

            $ENV_PYTHON launch \
                --num_processes 1 \
                "$PYTHON_SCRIPT" \
                --quant_method "$METHOD" \
                --cache_mode   "$CACHE" \
                --num_steps    "$STEP" \
                --num_samples  "$NUM_SAMPLES" \
                --guidance_scale 4.5 \
                $TEST_FLAG

            echo "✅ Done: $TAG"
            echo ""
        done
    done
done

echo "============================================================"
echo "All runs complete."
echo "Results in: $SCRIPT_DIR/results/MJHQ/"
echo "============================================================"
