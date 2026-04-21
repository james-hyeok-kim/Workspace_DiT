#!/usr/bin/env bash
# run_generate_refs.sh
# PixArt-XL MJHQ reference 이미지 생성
#
# 저장 위치: /data/jameskimh/james_dit_pixart_xl_mjhq/{precision}_steps{N}/ref_*.png
# - 이미 존재하는 이미지는 건너뜁니다 (재실행 안전)
#
# 생성 대상:
#   fp16_steps20  (공식 baseline)
#   fp16_steps15  (step 축소 비교군 + 우리 방식)
#
# 추후 추가 예시:
#   bash run_generate_refs.sh bf16 15 500

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

[ -f ~/.env ] && source ~/.env

MODEL_PATH="PixArt-alpha/PixArt-XL-2-1024-MS"
OUT_DIR="/data/jameskimh/james_dit_pixart_xl_mjhq"
NUM_SAMPLES=1000

PRECISION="${1:-}"
STEPS="${2:-}"
CUSTOM_N="${3:-}"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

run_one() {
    local prec="$1"
    local steps="$2"
    local n="${3:-$NUM_SAMPLES}"
    local tag="${prec}_steps${steps}"
    local log="$LOG_DIR/gen_ref_${tag}_${TIMESTAMP}.log"

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Generating: $tag  ($n samples)"
    echo "  → $OUT_DIR/$tag/"
    echo "════════════════════════════════════════════════════"

    accelerate launch --num_processes 1 \
        "$SCRIPT_DIR/generate_ref_images.py" \
        --model_path "$MODEL_PATH" \
        --precision  "$prec" \
        --num_inference_steps "$steps" \
        --num_samples "$n" \
        --out_dir "$OUT_DIR" \
        2>&1 | tee "$log"

    echo "  Log: $log"
}

if [ -n "$PRECISION" ] && [ -n "$STEPS" ]; then
    # 인자로 지정한 경우: bash run_generate_refs.sh fp16 15 [N]
    run_one "$PRECISION" "$STEPS" "${CUSTOM_N:-$NUM_SAMPLES}"
else
    # 기본: fp16 steps 20, 15 순으로 생성
    run_one fp16 20
    run_one fp16 15
fi

echo ""
echo "════════════════════════════════════════════════════"
echo " 저장 현황"
echo "════════════════════════════════════════════════════"
for d in "$OUT_DIR"/*/; do
    count=$(ls "$d"ref_*.png 2>/dev/null | wc -l)
    printf "  %-30s %5d images\n" "$(basename "$d")/" "$count"
done
