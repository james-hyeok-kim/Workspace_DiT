#!/bin/bash
# ============================================================
# W3A4 vs NVFP4_DEFAULT_CFG vs FP16 실험
# 목표: Rotation + GPTQ-INT3 + SVD + INT4-Act (W3A4) 방식 검증
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
BLOCK_SIZE=128
ALPHA=0.5
WGT_BITS=NVFP4

# smoke test 여부 (1이면 2샘플로 파이프라인 통과만 확인)
TEST_MODE=0

# ---- 경로 설정 ----
BASE_DIR="$(pwd)"
REF_DIR="/data/james_dit_ref/ref_images_fp16"
RESULT_BASE="${BASE_DIR}/results/w3a4_experiment"
LOG_DIR="${BASE_DIR}/logs/w3a4_experiment"
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

# ---- 마스터 run.log: 모든 출력을 파일과 콘솔에 동시 기록 ----
RUN_LOG="${LOG_DIR}/run.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

SWEEP_START=$(date '+%Y-%m-%d %H:%M:%S')
ACCEL_CMD="accelerate launch --multi_gpu --num_processes ${NUM_GPUS}"

echo "============================================================"
echo "  W3A4 Experiment Start  [mode=${MODE_LABEL}, samples=${NUM_SAMPLES}]"
echo "  Started at : ${SWEEP_START}"
echo "  Result base: ${RESULT_BASE}"
echo "  run.log    : ${RUN_LOG}"
echo "============================================================"

# ============================================================
# Phase 1: FP16 Baseline (상한 기준)
# ============================================================
echo ""
echo "---- [1/3] FP16 Baseline  ($(date '+%H:%M:%S')) ----"
FP16_SAVE="${RESULT_BASE}/FP16"
FP16_LOG="${LOG_DIR}/fp16_${MODE_LABEL}.log"

if [ -f "${FP16_SAVE}/${DATASET}/metrics.json" ]; then
    echo "  [SKIP] metrics.json already exists: ${FP16_SAVE}/${DATASET}/metrics.json"
else
    ${ACCEL_CMD} "${BASE_DIR}/pixart_w3a4_experiment.py" \
        --quant_method FP16 \
        --model_path "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir "${REF_DIR}" \
        --save_dir "${FP16_SAVE}" \
        --num_samples ${NUM_SAMPLES} \
        ${EXTRA_FLAGS} \
        2>&1 | tee "${FP16_LOG}"
    echo "  [$(date '+%H:%M:%S')] FP16 done."
fi

# ============================================================
# Phase 2: NVFP4_DEFAULT_CFG Baseline (beat 대상)
# ============================================================
echo ""
echo "---- [2/3] BASELINE: NVFP4_DEFAULT_CFG  ($(date '+%H:%M:%S')) ----"
BASELINE_SAVE="${RESULT_BASE}/BASELINE"
BASELINE_LOG="${LOG_DIR}/baseline_${MODE_LABEL}.log"

if [ -f "${BASELINE_SAVE}/${DATASET}/metrics.json" ]; then
    echo "  [SKIP] metrics.json already exists: ${BASELINE_SAVE}/${DATASET}/metrics.json"
else
    ${ACCEL_CMD} "${BASE_DIR}/pixart_w3a4_experiment.py" \
        --quant_method BASELINE \
        --model_path "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir "${REF_DIR}" \
        --save_dir "${BASELINE_SAVE}" \
        --num_samples ${NUM_SAMPLES} \
        --baseline_lowrank ${LOWRANK} \
        ${EXTRA_FLAGS} \
        2>&1 | tee "${BASELINE_LOG}"
    echo "  [$(date '+%H:%M:%S')] BASELINE done."
fi

# ============================================================
# Phase 3: W3A4 (Rotation + GPTQ-INT3 + SVD + INT4-Act)
# ============================================================
echo ""
echo "---- [3/3] W3A4  ($(date '+%H:%M:%S')) ----"
W3A4_SAVE="${RESULT_BASE}/W3A4"
W3A4_LOG="${LOG_DIR}/w3a4_${MODE_LABEL}.log"

if [ -f "${W3A4_SAVE}/${DATASET}/metrics.json" ]; then
    echo "  [SKIP] metrics.json already exists: ${W3A4_SAVE}/${DATASET}/metrics.json"
else
    ${ACCEL_CMD} "${BASE_DIR}/pixart_w3a4_experiment.py" \
        --quant_method W3A4 \
        --wgt_bits ${WGT_BITS} \
        --lowrank ${LOWRANK} \
        --block_size ${BLOCK_SIZE} \
        --alpha ${ALPHA} \
        --model_path "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir "${REF_DIR}" \
        --save_dir "${W3A4_SAVE}" \
        --num_samples ${NUM_SAMPLES} \
        ${EXTRA_FLAGS} \
        2>&1 | tee "${W3A4_LOG}"
    echo "  [$(date '+%H:%M:%S')] W3A4 done."
fi

SWEEP_END=$(date '+%Y-%m-%d %H:%M:%S')

# ============================================================
# Phase 4: results_summary.json + handoff.md 생성
# ============================================================
echo ""
echo "---- [Phase 4] Generating results_summary.json and handoff.md ----"

python3 - <<PYEOF
import json, os, glob
from datetime import datetime

result_base  = "${RESULT_BASE}"
dataset      = "${DATASET}"
sweep_start  = "${SWEEP_START}"
sweep_end    = "${SWEEP_END}"
mode_label   = "${MODE_LABEL}"
num_samples  = ${NUM_SAMPLES}
run_log      = "${RUN_LOG}"

# 모든 metrics.json 수집
entries = []
for path in sorted(glob.glob(os.path.join(result_base, "*", dataset, "metrics.json"))):
    run_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    with open(path) as f:
        d = json.load(f)
    entries.append({"run": run_name, **d})

# FID 기준 정렬
sorted_entries = sorted(entries, key=lambda x: x.get("primary_metrics", {}).get("FID", 9999))
fp16     = next((e for e in entries if e["run"] == "FP16"),     None)
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
        "run":   e["run"],
        "method": cfg.get("quant_method", e["run"]),
        "fid":   f"{fid:.4f}" if isinstance(fid, float) else str(fid),
        "is":    f"{is_v:.4f}" if isinstance(is_v, float) else str(is_v),
        "psnr":  f"{sm.get('PSNR',  0):.2f}",
        "ssim":  f"{sm.get('SSIM',  0):.4f}",
        "lpips": f"{sm.get('LPIPS', 0):.4f}",
        "clip":  f"{sm.get('CLIP',  0):.2f}",
        "beat":  beat,
    })

table_lines = [
    "| Run | Method | FID↓ | IS↑ | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP↑ | Beat Baseline? |",
    "|---|---|---|---|---|---|---|---|---|",
]
for r in rows:
    table_lines.append(
        f"| {r['run']} | {r['method']} | {r['fid']} | {r['is']}"
        f" | {r['psnr']} | {r['ssim']} | {r['lpips']} | {r['clip']} | {r['beat']} |"
    )

# results_summary.json
summary = {
    "summary_table": rows,
    "sweep_start":   sweep_start,
    "sweep_end":     sweep_end,
    "mode":          mode_label,
    "num_samples":   num_samples,
    "dataset":       dataset,
    "runs":          entries,
}
summary_path = os.path.join(result_base, "results_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)
print(f"  results_summary.json -> {summary_path}")

# Baseline 대비 개선된 조합
winners = [r for r in rows if r["beat"] == "YES"]
winner_summary = (
    "\n".join(f"- **{w['run']}**  FID={w['fid']}  IS={w['is']}" for w in winners)
    if winners else "- 없음 (추가 튜닝 필요)"
)

fp16_fid_str = f"{fp16['primary_metrics']['FID']:.4f}" if fp16 else "N/A"
fp16_is_str  = f"{fp16['primary_metrics']['IS']:.4f}"  if fp16 else "N/A"

md = f"""# W3A4 Quantization Experiment Handoff

## 개요
Rotation(Hadamard) + GPTQ-INT3 + SVD + INT4-Act (W3A4) 방식과
FP16 / NVFP4_DEFAULT_CFG(공식 baseline) 비교 실험.
총 {len(entries)}회 실행 / {num_samples} samples / dataset: {dataset}
시작: {sweep_start}  |  종료: {sweep_end}

---

## Change List (신규 생성 파일)

| 파일 | 설명 |
|---|---|
| `pixart_w3a4_experiment.py` | W3A4 / BASELINE / FP16 통합 실험 스크립트. W3A4Linear 클래스(Hadamard+SmoothQuant+GPTQ-INT3+SVD+INT4-Act). |
| `run_w3a4_experiment.sh` | 3-run 자동화 스크립트. 실행 시 run.log 통합 기록, 완료 후 results_summary.json · handoff.md 자동 생성. |
| `results/w3a4_experiment/results_summary.json` | 전체 실험 결과 통합 JSON. |
| `results/w3a4_experiment/*/{{dataset}}/metrics.json` | 실행별 개별 결과 JSON. |
| `logs/w3a4_experiment/run.log` | 전체 실험 통합 로그. |
| `handoff_w3a4.md` | 이 파일. |

> 기존 파일은 **일절 수정하지 않음**.

---

## 실험 설정

| 항목 | 값 |
|---|---|
| W3A4 weight bits | INT3 (GPTQ-light with diagonal H) |
| W3A4 act bits | INT4 (dynamic per block) |
| Hadamard block size | 128 (PixArt 1152=9×128, 4608=36×128) |
| SVD rank | 32 |
| SmoothQuant alpha | 0.5 |
| GPTQ group size | 128 (= block_size) |
| Skip layers | x_embedder, t_embedder, proj_out |

---

## 실험 결과 (FID 오름차순)

- **FP16 baseline**: FID={fp16_fid_str}  IS={fp16_is_str}
- **NVFP4 baseline**: FID={f"{baseline_fid:.4f}" if baseline_fid else "N/A"}  IS={f"{baseline_is:.4f}" if baseline_is else "N/A"}

{chr(10).join(table_lines)}

---

## Baseline 대비 FID 개선 조합

{winner_summary}

---

## 저장 경로

```
results/w3a4_experiment/
  FP16/{{dataset}}/metrics.json
  BASELINE/{{dataset}}/metrics.json
  W3A4/{{dataset}}/metrics.json
  results_summary.json
logs/w3a4_experiment/
  run.log
  fp16_*.log
  baseline_*.log
  w3a4_*.log
```

---

## 다음 단계 제안

1. W3A4가 BASELINE을 이겼다면 → num_samples=100으로 본 실험 재실행
2. W3A4가 BASELINE을 못 이겼다면 → 다음 시도:
   - block_size 조정 (64 또는 256)
   - lowrank 증가 (32 → 64)
   - mixed-precision: cross-attention layer 보호 (FP16 유지)
3. W2A4 (INT2 weight) 탐색
"""

handoff_path = os.path.join("${BASE_DIR}", "handoff_w3a4.md")
with open(handoff_path, "w") as f:
    f.write(md)
print(f"  handoff_w3a4.md -> {handoff_path}")

# 콘솔 요약 출력
print()
print("  Results summary (FID ascending):")
print(f"  {'Run':<12} {'FID':>8} {'IS':>8} {'PSNR':>7} {'SSIM':>7} {'Beat?':>6}")
print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*6}")
for r in rows:
    print(f"  {r['run']:<12} {r['fid']:>8} {r['is']:>8} {r['psnr']:>7} {r['ssim']:>7} {r['beat']:>6}")
PYEOF

echo ""
echo "============================================================"
echo "  Experiment complete: ${SWEEP_END}"
echo "  Summary JSON  : ${RESULT_BASE}/results_summary.json"
echo "  Handoff       : ${BASE_DIR}/handoff_w3a4.md"
echo "  Full log      : ${RUN_LOG}"
echo "============================================================"
