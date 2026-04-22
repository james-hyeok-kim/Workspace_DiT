#!/bin/bash
# run_nonlinear.sh — Nonlinear Cache-LoRA corrector experiments
#
# 4 options × steps={10,15,20} × SVDQUANT = 12 runs (n=20)
# + linear baseline for fair comparison = 3 runs
# + deepcache baseline = 3 runs
# Total: 18 runs
#
# Usage:
#   bash run_nonlinear.sh                         # 20 samples, all modes
#   bash run_nonlinear.sh --test_run              # 2 samples smoke test
#   bash run_nonlinear.sh --gpu 0                 # single GPU 지정
#   bash run_nonlinear.sh --modes gelu,mlp        # 특정 nl 모드만 실행
#   bash run_nonlinear.sh --modes res,film --gpu 1 --port 29501
#   bash run_nonlinear.sh --modes gelu --num_samples 100 --steps 20  # 단일 step + resume

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/pixart_nvfp4_cache_compare.py"
PYTHON="/home/jovyan/.dit/bin/python3"
ACCELERATE_CLI="/home/jovyan/.dit/lib/python3.11/site-packages/accelerate/commands/accelerate_cli.py"
ENV_PYTHON="$PYTHON $ACCELERATE_CLI"
DATA_ROOT="/data/jameskimh/james_dit_pixart_sigma_xl_mjhq"

NUM_SAMPLES=20
TEST_RUN=false
TEST_FLAG=""
GPU_ID=""
PORT=29500
METHOD=SVDQUANT
CS=8; CE=20; INTERVAL=2
RANK=4; MID=32; CALIB=4
MODES="gelu mlp res film"   # 실행할 nl 모드 (space-separated)
STEPS_OVERRIDE=""            # --steps 로 단일 step 지정 시 사용

STEPS=(10 15 20)

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --test_run)    TEST_RUN=true; TEST_FLAG="--test_run" ;;
        --gpu)         GPU_ID="$2"; shift ;;
        --port)        PORT="$2"; shift ;;
        --num_samples) NUM_SAMPLES="$2"; shift ;;
        --method)      METHOD="$2"; shift ;;
        --modes)       MODES="${2//,/ }"; shift ;;
        --steps)       STEPS_OVERRIDE="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# --steps 옵션이 있으면 해당 step만 실행
if [ -n "$STEPS_OVERRIDE" ]; then
    IFS=',' read -ra STEPS <<< "$STEPS_OVERRIDE"
fi

# GPU 설정: --gpu 로 단일 GPU 지정 시 CUDA_VISIBLE_DEVICES 설정
# --gpu 없으면 CUDA_VISIBLE_DEVICES 미설정 → accelerate 가 감지한 모든 GPU 사용
if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "Using GPU: $GPU_ID"
fi

# 2-GPU split: CUDA_VISIBLE_DEVICES=0,1 이면 num_processes=2
NUM_VISIBLE=$(echo "${CUDA_VISIBLE_DEVICES:-0,1}" | tr ',' '\n' | wc -l)
NUM_PROCS="$NUM_VISIBLE"

run_one() {
    local CACHE_MODE="$1" STEP="$2" EXTRA_FLAGS="$3"

    # Build tag for skip check
    local NL_TYPE="${CACHE_MODE#cache_nl_}"
    local TAG
    if [[ "$CACHE_MODE" == cache_nl_* ]]; then
        TAG="${METHOD}_nl_${NL_TYPE}_r${RANK}_m${MID}_c${CS}-${CE}_steps${STEP}"
    elif [ "$CACHE_MODE" = "cache_lora" ]; then
        TAG="${METHOD}_cl_r${RANK}_c${CS}-${CE}_steps${STEP}"
    elif [ "$CACHE_MODE" = "deepcache" ]; then
        TAG="${METHOD}_deepcache_c${CS}-${CE}_steps${STEP}"
    fi

    local RESULT_DIR="$DATA_ROOT/$METHOD/MJHQ/$TAG"
    # Skip only when saved num_samples >= requested NUM_SAMPLES (resumes partial runs)
    if [ -f "$RESULT_DIR/metrics.json" ]; then
        EXIST_N=$(python -c "import json; print(json.load(open('$RESULT_DIR/metrics.json')).get('num_samples',0))" 2>/dev/null || echo 0)
        if [ "$EXIST_N" -ge "$NUM_SAMPLES" ]; then
            echo "  SKIP: $TAG (already has $EXIST_N samples >= $NUM_SAMPLES)"
            return
        else
            echo "  RESUME: $TAG ($EXIST_N → $NUM_SAMPLES)"
        fi
    fi

    echo ""
    echo "  Running: $TAG  (num_processes=$NUM_PROCS)"
    echo "  ─────────────────────────────────────────"

    $ENV_PYTHON launch \
        --num_processes "$NUM_PROCS" \
        --main_process_port "$PORT" \
        "$PYTHON_SCRIPT" \
        --quant_method   "$METHOD" \
        --cache_mode     "$CACHE_MODE" \
        --num_steps      "$STEP" \
        --num_samples    "$NUM_SAMPLES" \
        --guidance_scale 4.5 \
        --cache_start    "$CS" \
        --cache_end      "$CE" \
        --deepcache_interval "$INTERVAL" \
        --lora_rank      "$RANK" \
        --lora_calib     "$CALIB" \
        --nl_mid_dim     "$MID" \
        $EXTRA_FLAGS \
        $TEST_FLAG

    echo "  Done: $TAG"
}

echo "============================================================"
echo "  Nonlinear Cache-LoRA Experiment ($NUM_SAMPLES samples)"
echo "  Method: $METHOD  Range: [$CS,$CE)  Rank: $RANK  Mid: $MID"
echo "  Steps: ${STEPS[*]}  GPUs: $NUM_PROCS"
echo "============================================================"

cd "$SCRIPT_DIR"

# ── Baselines ─────────────────────────────────────────────────────────
echo ""
echo "======= Baselines ======="
for STEP in "${STEPS[@]}"; do
    run_one "deepcache"  "$STEP"
    run_one "cache_lora" "$STEP"
done

# ── Option 1: GELU bottleneck ────────────────────────────────────────
if [[ " $MODES " == *" gelu "* ]]; then
    echo ""
    echo "======= Option 1: GELU bottleneck (rank=$RANK) ======="
    for STEP in "${STEPS[@]}"; do
        run_one "cache_nl_gelu" "$STEP"
    done
fi

# ── Option 2: Bottleneck MLP (mid=$MID) ──────────────────────────────
if [[ " $MODES " == *" mlp "* ]]; then
    echo ""
    echo "======= Option 2: Bottleneck MLP (mid=$MID) ======="
    for STEP in "${STEPS[@]}"; do
        run_one "cache_nl_mlp" "$STEP"
    done
fi

# ── Option 3: Residual MLP (mid=$MID) ────────────────────────────────
if [[ " $MODES " == *" res "* ]]; then
    echo ""
    echo "======= Option 3: Residual MLP (mid=$MID) ======="
    for STEP in "${STEPS[@]}"; do
        run_one "cache_nl_res" "$STEP"
    done
fi

# ── Option 4: FiLM conditioned (mid=$MID) ────────────────────────────
if [[ " $MODES " == *" film "* ]]; then
    echo ""
    echo "======= Option 4: FiLM conditioned (mid=$MID) ======="
    for STEP in "${STEPS[@]}"; do
        run_one "cache_nl_film" "$STEP"
    done
fi

echo ""
echo "============================================================"
echo "Nonlinear experiment complete."
echo "============================================================"
