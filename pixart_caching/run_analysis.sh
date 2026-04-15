#!/usr/bin/env bash
# run_analysis.sh
# "SVDQuant meets DeepCache" 프레임워크 핵심 실험 스크립트
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  목표: 단순 조합이 아닌 SVD × Cache 상호작용 분석 + 최적화        │
# │                                                                     │
# │  Exp D: 2×2 Ablation (rank × cache)      ← 논문 Table 핵심        │
# │  Exp A: SVD residual error profiling     ← "왜 잘 되는지" 분석    │
# │  Exp B: Block-level drift profiling      ← cache 범위 선택 근거   │
# │  Exp C: Cache-aware NVFP4 calibration   ← method 개선            │
# └─────────────────────────────────────────────────────────────────────┘
#
# Q&C / CacheQuant / QuantCache baseline 비교는 follow-up (run_baselines.sh 예정)

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

REF_DIR="$SCRIPT_DIR/ref_images"
SAVE_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# =============================================================================
# Exp D: 2×2 Ablation — rank × cache
#
#   No Cache (interval=1)  │  With Cache (interval=2)
#   ───────────────────────┼──────────────────────────
#   D-A: rank=32, no cache │  B (조합1, FID=162.99) ← 이미 있음
#   D-C: rank=0,  no cache │  A-2 (Exp A-2에서 측정)
#
# D-A와 D-C를 추가하면 4개 셀 완성:
#   SVD 기여 분리: (B - A-2) = cache가 추가한 FID 변화 (rank=32 조건)
#   Cache 기여 분리: (B - D-A) = cache 없이 rank=32만 쓸 때와 비교
#   상호작용: (B - D-A) - (A-2 - D-C) ≠ 0 이면 SVD×cache synergy 존재
# =============================================================================
echo ""
echo "=============================="
echo " Exp D-A: rank=32, NO cache (no-cache baseline)"
echo "=============================="
LOG_DA="$LOG_DIR/expD_rank32_nocache_${TIMESTAMP}.log"
accelerate launch --num_processes 1 \
    "$SCRIPT_DIR/pixart_deepcache_experiment.py" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET" \
    --lowrank 32 \
    --num_inference_steps "$NUM_STEPS" \
    --cache_interval 1 \
    --cache_start "$CACHE_START" \
    --cache_end "$CACHE_END" \
    --ref_dir "$REF_DIR" \
    --save_dir "$SAVE_DIR" \
    2>&1 | tee "$LOG_DA"

echo "Exp D-A done. Log: $LOG_DA"

echo ""
echo "=============================="
echo " Exp D-C: rank=0, NO cache (no-SVD no-cache baseline)"
echo "=============================="
LOG_DC="$LOG_DIR/expD_rank0_nocache_${TIMESTAMP}.log"
accelerate launch --num_processes 1 \
    "$SCRIPT_DIR/pixart_deepcache_experiment.py" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET" \
    --lowrank 0 \
    --num_inference_steps "$NUM_STEPS" \
    --cache_interval 1 \
    --cache_start "$CACHE_START" \
    --cache_end "$CACHE_END" \
    --ref_dir "$REF_DIR" \
    --save_dir "$SAVE_DIR" \
    2>&1 | tee "$LOG_DC"

echo "Exp D-C done. Log: $LOG_DC"

# =============================================================================
# Exp A: SVD × Cache Residual Error Profiling
#   rank=32 vs rank=0 with cache: stale residual error 비교
#   → SVD branch가 cache error를 자연 보정하는지 정량화
# =============================================================================
echo ""
echo "=============================="
echo " Exp A-1: rank=32 + cache (residual error profiling)"
echo "=============================="
LOG_A32="$LOG_DIR/expA_rank32_${TIMESTAMP}.log"
accelerate launch --num_processes 1 \
    "$SCRIPT_DIR/pixart_deepcache_experiment.py" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET" \
    --lowrank 32 \
    --num_inference_steps "$NUM_STEPS" \
    --cache_interval "$CACHE_INTERVAL" \
    --cache_start "$CACHE_START" \
    --cache_end "$CACHE_END" \
    --ref_dir "$REF_DIR" \
    --save_dir "$SAVE_DIR" \
    --profile_residual_error \
    2>&1 | tee "$LOG_A32"

echo "Exp A-1 done. Log: $LOG_A32"

echo ""
echo "=============================="
echo " Exp A-2: rank=0 + cache (no SVD, residual error profiling)"
echo "=============================="
LOG_A0="$LOG_DIR/expA_rank0_${TIMESTAMP}.log"
accelerate launch --num_processes 1 \
    "$SCRIPT_DIR/pixart_deepcache_experiment.py" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET" \
    --lowrank 0 \
    --num_inference_steps "$NUM_STEPS" \
    --cache_interval "$CACHE_INTERVAL" \
    --cache_start "$CACHE_START" \
    --cache_end "$CACHE_END" \
    --ref_dir "$REF_DIR" \
    --save_dir "$SAVE_DIR" \
    --profile_residual_error \
    2>&1 | tee "$LOG_A0"

echo "Exp A-2 done. Log: $LOG_A0"

# =============================================================================
# Exp B: Block-level Drift Profiling
#   28개 block의 timestep 간 output drift 측정
#   → cache 범위 [8,20) 선택의 정량적 근거
#   → PixArt cross-attention 구조 특성 분석
# =============================================================================
echo ""
echo "=============================="
echo " Exp B: Block drift profiling (all 28 blocks)"
echo "=============================="
LOG_B="$LOG_DIR/expB_blockdrift_${TIMESTAMP}.log"
accelerate launch --num_processes 1 \
    "$SCRIPT_DIR/pixart_deepcache_experiment.py" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET" \
    --lowrank 32 \
    --num_inference_steps "$NUM_STEPS" \
    --cache_interval "$CACHE_INTERVAL" \
    --cache_start "$CACHE_START" \
    --cache_end "$CACHE_END" \
    --ref_dir "$REF_DIR" \
    --save_dir "$SAVE_DIR" \
    --profile_blocks \
    2>&1 | tee "$LOG_B"

echo "Exp B done. Log: $LOG_B"

# =============================================================================
# Exp C: Cache-aware NVFP4 Calibration
#   DeepCache 활성화 상태로 calibration → cached step activation 분포 반영
#   → standard calibration(조합1 FID=162.99) 대비 FID/IS 비교
# =============================================================================
echo ""
echo "=============================="
echo " Exp C: Cache-aware NVFP4 calibration"
echo "=============================="
LOG_C="$LOG_DIR/expC_cache_calib_${TIMESTAMP}.log"
accelerate launch --num_processes 1 \
    "$SCRIPT_DIR/pixart_deepcache_experiment.py" \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET" \
    --lowrank 32 \
    --num_inference_steps "$NUM_STEPS" \
    --cache_interval "$CACHE_INTERVAL" \
    --cache_start "$CACHE_START" \
    --cache_end "$CACHE_END" \
    --ref_dir "$REF_DIR" \
    --save_dir "$SAVE_DIR" \
    --cache_aware_calib \
    2>&1 | tee "$LOG_C"

echo "Exp C done. Log: $LOG_C"

# =============================================================================
# 결과 요약
# =============================================================================
DEEP_DIR="$SAVE_DIR/$DATASET/deepcache"

echo ""
echo "======================================================================="
echo " 결과 파일 위치"
echo "======================================================================="
echo ""
echo "[Exp D] 2×2 Ablation (rank × cache):"
echo "  D-A rank=32 no-cache : $DEEP_DIR/interval1_s${CACHE_START}_e${CACHE_END}_gs4.5/metrics.csv"
echo "  D-C rank=0  no-cache : $DEEP_DIR/interval1_s${CACHE_START}_e${CACHE_END}_gs4.5/metrics.csv"
echo "  B   rank=32 cache    : (기존 조합1, FID=162.99)"
echo "  A-2 rank=0  cache    : $DEEP_DIR/interval${CACHE_INTERVAL}_s${CACHE_START}_e${CACHE_END}_gs4.5/metrics.csv"
echo ""
echo "  → 4개 셀로 SVD 기여 / Cache 기여 / 상호작용 분리 가능"
echo ""
echo "[Exp A] Residual error (stale vs fresh residual):"
echo "  rank=32: $DEEP_DIR/interval${CACHE_INTERVAL}_s${CACHE_START}_e${CACHE_END}_gs4.5/residual_errors_rank32.csv"
echo "  rank=0 : $DEEP_DIR/interval${CACHE_INTERVAL}_s${CACHE_START}_e${CACHE_END}_gs4.5/residual_errors_rank0.csv"
echo ""
echo "[Exp B] Block drift profile:"
echo "  $DEEP_DIR/interval${CACHE_INTERVAL}_s${CACHE_START}_e${CACHE_END}_gs4.5/block_drift_profile.csv"
echo ""
echo "[Exp C] Cache-aware calib vs standard:"
echo "  $DEEP_DIR/interval${CACHE_INTERVAL}_s${CACHE_START}_e${CACHE_END}_gs4.5_calib_cache/metrics.csv"
echo "  (baseline: FID=162.99)"
echo ""
echo "======================================================================="
echo " [Follow-up] Q&C / CacheQuant / QuantCache 비교 → run_baselines.sh 예정"
echo "======================================================================="
echo ""
echo "All experiments complete."
