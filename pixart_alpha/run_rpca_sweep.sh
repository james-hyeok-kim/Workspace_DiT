#!/bin/bash
# ============================================================
# RPCA vs NVFP4_DEFAULT_CFG Quantization Sweep
# 목표: NVFP4_DEFAULT_CFG(Baseline)를 RPCA 방식으로 이기기
# ============================================================

# ---- 환경 설정 ----
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
if [ -f ~/.env ]; then
  export $(grep -v '^#' ~/.env | xargs)
fi

# ---- 실험 설정 (여기만 수정) ----
NUM_GPUS=2
NUM_SAMPLES=20
MODEL_PATH="PixArt-alpha/PixArt-XL-2-1024-MS"
DATASET="MJHQ"
LOWRANK=32
BLOCK_SIZE=16
ALPHA=0.5
NUMERIC="half"

# smoke test 여부 (1이면 2샘플로 파이프라인 통과만 확인)
TEST_MODE=0

# RPCA sweep 설정
OUTLIER_RATIOS=(0.01 0.05 0.1)
# (act_mode wgt_mode) 쌍
BIT_COMBOS=("NVFP4 NVFP4" "INT8 INT8" "INT8 INT4" "INT4 INT4")

# ---- 경로 설정 ----
BASE_DIR="$(pwd)"
REF_DIR="${BASE_DIR}/ref_images"
RESULT_BASE="${BASE_DIR}/results/rpca_sweep"
LOG_DIR="${BASE_DIR}/logs/rpca_sweep"
mkdir -p "${RESULT_BASE}" "${LOG_DIR}"

# ---- 모드별 플래그 ----
if [ "$TEST_MODE" -eq 1 ]; then
    NUM_SAMPLES=2
    MODE_LABEL="test"
    EXTRA_FLAGS="--test_run"
else
    MODE_LABEL="prod"
    EXTRA_FLAGS=""
fi

# ---- 마스터 run.log: 이 스크립트의 모든 출력을 파일과 콘솔에 동시 기록 ----
RUN_LOG="${LOG_DIR}/run.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

SWEEP_START=$(date '+%Y-%m-%d %H:%M:%S')
ACCEL_CMD="accelerate launch --multi_gpu --num_processes ${NUM_GPUS}"

echo "============================================================"
echo "  RPCA Sweep Start  [mode=${MODE_LABEL}, samples=${NUM_SAMPLES}]"
echo "  Started at : ${SWEEP_START}"
echo "  Result base: ${RESULT_BASE}"
echo "  run.log    : ${RUN_LOG}"
echo "============================================================"

# ============================================================
# Phase 1: BASELINE (NVFP4_DEFAULT_CFG)
# ============================================================
TOTAL=$(( ${#BIT_COMBOS[@]} * ${#OUTLIER_RATIOS[@]} + 1 ))
echo ""
echo "---- [1/${TOTAL}] BASELINE: NVFP4_DEFAULT_CFG  ($(date '+%H:%M:%S')) ----"

BASELINE_SAVE="${RESULT_BASE}/BASELINE"
BASELINE_LOG="${LOG_DIR}/baseline_${MODE_LABEL}.log"

${ACCEL_CMD} "${BASE_DIR}/pixart_rpca_experiment.py" \
    --quant_method BASELINE \
    --model_path "${MODEL_PATH}" \
    --dataset_name "${DATASET}" \
    --ref_dir "${REF_DIR}" \
    --save_dir "${BASELINE_SAVE}" \
    --num_samples ${NUM_SAMPLES} \
    --lowrank ${LOWRANK} \
    ${EXTRA_FLAGS} \
    2>&1 | tee "${BASELINE_LOG}"

echo "  [$(date '+%H:%M:%S')] Baseline done."

# ============================================================
# Phase 2: RPCA Sweep (outlier_ratio x bit-combo)
# ============================================================
RUN_IDX=2

for OR in "${OUTLIER_RATIOS[@]}"; do
    for COMBO in "${BIT_COMBOS[@]}"; do
        ACT=$(echo "${COMBO}" | awk '{print $1}')
        WGT=$(echo "${COMBO}" | awk '{print $2}')

        RUN_NAME="RPCA_A${ACT}_W${WGT}_OR${OR}"
        SAVE_DIR="${RESULT_BASE}/${RUN_NAME}"
        LOG_FILE="${LOG_DIR}/${RUN_NAME}_${MODE_LABEL}.log"

        echo ""
        echo "---- [${RUN_IDX}/${TOTAL}] ${RUN_NAME}  ($(date '+%H:%M:%S')) ----"

        ${ACCEL_CMD} "${BASE_DIR}/pixart_rpca_experiment.py" \
            --quant_method RPCA \
            --act_mode "${ACT}" \
            --wgt_mode "${WGT}" \
            --outlier_ratio "${OR}" \
            --alpha "${ALPHA}" \
            --lowrank ${LOWRANK} \
            --block_size ${BLOCK_SIZE} \
            --numeric_dtype "${NUMERIC}" \
            --model_path "${MODEL_PATH}" \
            --dataset_name "${DATASET}" \
            --ref_dir "${REF_DIR}" \
            --save_dir "${SAVE_DIR}" \
            --num_samples ${NUM_SAMPLES} \
            ${EXTRA_FLAGS} \
            2>&1 | tee "${LOG_FILE}"

        echo "  [$(date '+%H:%M:%S')] ${RUN_NAME} done."
        RUN_IDX=$(( RUN_IDX + 1 ))
    done
done

SWEEP_END=$(date '+%Y-%m-%d %H:%M:%S')

# ============================================================
# Phase 3: results_summary.json + handoff.md 생성
# ============================================================
echo ""
echo "---- [Phase 3] Generating results_summary.json and handoff.md ----"

python3 - <<PYEOF
import json, os, glob
from datetime import datetime

result_base = "${RESULT_BASE}"
dataset    = "${DATASET}"
sweep_start = "${SWEEP_START}"
sweep_end   = "${SWEEP_END}"
mode_label  = "${MODE_LABEL}"
num_samples = ${NUM_SAMPLES}
run_log     = "${RUN_LOG}"

# 모든 metrics.json 수집
entries = []
for path in sorted(glob.glob(os.path.join(result_base, "*", dataset, "metrics.json"))):
    run_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    with open(path) as f:
        d = json.load(f)
    entries.append({"run": run_name, **d})

# ---- results_summary.json ----
summary = {
    "sweep_start": sweep_start,
    "sweep_end": sweep_end,
    "mode": mode_label,
    "num_samples": num_samples,
    "dataset": dataset,
    "runs": entries,
}
summary_path = os.path.join(result_base, "results_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)
print(f"  results_summary.json -> {summary_path}")

# ---- handoff.md ----
# FID 기준 오름차순 정렬 (낮을수록 좋음)
sorted_entries = sorted(entries, key=lambda x: x.get("primary_metrics", {}).get("FID", 9999))
baseline = next((e for e in entries if e["run"] == "BASELINE"), None)
baseline_fid = baseline["primary_metrics"]["FID"] if baseline else None
baseline_is  = baseline["primary_metrics"]["IS"]  if baseline else None

rows = []
for e in sorted_entries:
    cfg  = e.get("config", {})
    pm   = e.get("primary_metrics", {})
    sm   = e.get("secondary_metrics", {})
    fid  = pm.get("FID", "-")
    is_v = pm.get("IS",  "-")
    beat = ""
    if baseline_fid is not None and isinstance(fid, float):
        beat = "YES" if fid < baseline_fid else "no"
    rows.append({
        "run": e["run"],
        "act": cfg.get("act_mode", "N/A"),
        "wgt": cfg.get("wgt_mode", "N/A"),
        "or":  cfg.get("outlier_ratio", "N/A"),
        "fid": f"{fid:.4f}" if isinstance(fid, float) else str(fid),
        "is":  f"{is_v:.4f}" if isinstance(is_v, float) else str(is_v),
        "psnr": f"{sm.get('PSNR', 0):.2f}",
        "ssim": f"{sm.get('SSIM', 0):.4f}",
        "lpips": f"{sm.get('LPIPS', 0):.4f}",
        "clip": f"{sm.get('CLIP', 0):.2f}",
        "beat": beat,
    })

table_lines = [
    "| Run | Act | Wgt | OR | FID↓ | IS↑ | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP↑ | Beat Baseline? |",
    "|---|---|---|---|---|---|---|---|---|---|---|",
]
for r in rows:
    table_lines.append(
        f"| {r['run']} | {r['act']} | {r['wgt']} | {r['or']} "
        f"| {r['fid']} | {r['is']} | {r['psnr']} | {r['ssim']} | {r['lpips']} | {r['clip']} | {r['beat']} |"
    )

# 베이스라인 대비 FID 개선된 조합
winners = [r for r in rows if r["beat"] == "YES"]
winner_summary = "\n".join(f"- **{w['run']}**  FID={w['fid']}  IS={w['is']}" for w in winners) if winners else "- 없음 (추가 튜닝 필요)"

md = f"""# RPCA Quantization Experiment Handoff

## 개요
NVFP4_DEFAULT_CFG(공식 baseline) 대비 RPCA 기반 양자화 방식의 우수성을 검증하는 sweep 실험.
총 {len(entries)}회 실행 / {num_samples} samples / dataset: {dataset}
시작: {sweep_start}  |  종료: {sweep_end}

---

## Change List (신규 생성 파일)

| 파일 | 설명 |
|---|---|
| `pixart_rpca_experiment.py` | RPCA / BASELINE 통합 실험 스크립트. `--outlier_ratio` argparse 정식 등록. primary(FID/IS) + secondary(PSNR/SSIM/LPIPS/CLIP) 분리 저장. |
| `run_rpca_sweep.sh` | Sweep 자동화 스크립트. 실행 시 `run.log` 통합 기록, 완료 후 `results_summary.json` · `handoff.md` 자동 생성. |
| `results/rpca_sweep/results_summary.json` | 전체 실험 결과 통합 JSON. |
| `results/rpca_sweep/*/{{dataset}}/metrics.json` | 실행별 개별 결과 JSON (primary + secondary metrics 구분). |
| `logs/rpca_sweep/run.log` | 전체 sweep 통합 로그 (타임스탬프 포함). |
| `handoff.md` | 이 파일. |

> 기존 파일(`pixart_alpha_quant_b200.py` 등)은 **일절 수정하지 않음**.

---

## 실험 매트릭스

- **Baseline**: `NVFP4_DEFAULT_CFG`  →  FID={f"{baseline_fid:.4f}" if baseline_fid else "N/A"}  IS={f"{baseline_is:.4f}" if baseline_is else "N/A"}
- **RPCA sweep**: outlier_ratio ∈ {{0.01, 0.05, 0.1}} × bit-combo ∈ {{NVFP4×NVFP4, INT8×INT8, INT8×INT4, INT4×INT4}}
- **고정값**: lowrank=32, block_size=16, alpha=0.5

---

## 실험 결과 (FID 오름차순)

{chr(10).join(table_lines)}

---

## Baseline 대비 FID 개선 조합

{winner_summary}

---

## 저장 경로

```
results/rpca_sweep/
  BASELINE/{{dataset}}/metrics.json
  RPCA_A<act>_W<wgt>_OR<or>/{{dataset}}/metrics.json
  results_summary.json          ← 전체 통합
logs/rpca_sweep/
  run.log                       ← 전체 통합 로그
  baseline_*.log
  RPCA_*.log
```

---

## 다음 단계 제안

1. 위 표에서 **FID 최저 + IS 최고** 조합을 선택해 num_samples=100으로 본 실험 재실행
2. 최적 outlier_ratio 고정 후 lowrank sweep (32 → 64) 시도
3. alpha (현재 0.5) 조정으로 추가 개선 여부 확인
"""

handoff_path = os.path.join("${BASE_DIR}", "handoff.md")
with open(handoff_path, "w") as f:
    f.write(md)
print(f"  handoff.md -> {handoff_path}")

# 콘솔 요약 출력
print()
print("  Results summary (FID ascending):")
print(f"  {'Run':<45} {'FID':>8} {'IS':>8} {'Beat?':>6}")
print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*6}")
for r in rows:
    print(f"  {r['run']:<45} {r['fid']:>8} {r['is']:>8} {r['beat']:>6}")
PYEOF

echo ""
echo "============================================================"
echo "  Sweep complete: ${SWEEP_END}"
echo "  Summary JSON : ${RESULT_BASE}/results_summary.json"
echo "  Handoff      : ${BASE_DIR}/handoff.md"
echo "  Full log     : ${RUN_LOG}"
echo "============================================================"
