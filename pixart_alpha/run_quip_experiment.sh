#!/bin/bash
# ============================================================
# QuIP W3A4/W2A4 vs NVFP4_DEFAULT_CFG vs FP16 실험
# 목표: Randomized Hadamard (QuIP-style) + INT3/INT2 weight + SVD + INT4-Act 검증
# ============================================================

# ---- 환경 설정 ----
export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
if [ -f ~/.env ]; then
  export $(grep -v '^#' ~/.env | xargs)
fi

# activate .dit env if accelerate not in PATH
if ! command -v accelerate &>/dev/null; then
  source /home/jameskimh/.dit/bin/activate
fi

# ---- 실험 설정 (여기만 수정) ----
NUM_GPUS=2
NUM_SAMPLES=20
MODEL_PATH="PixArt-alpha/PixArt-XL-2-1024-MS"
DATASET="MJHQ"
BLOCK_SIZE=128
LOWRANK_W3A4=64
LOWRANK_W2A4=128
BASELINE_LOWRANK=32

# smoke test 여부 (1이면 2샘플로 파이프라인 통과만 확인)
TEST_MODE=0

# ---- 경로 설정 ----
BASE_DIR="$(pwd)"
REF_DIR="/data/james_dit_ref/ref_images_fp16"
RESULT_BASE="${BASE_DIR}/results/quip_experiment"
LOG_DIR="${BASE_DIR}/logs/quip_experiment"
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
echo "  QuIP Experiment Start  [mode=${MODE_LABEL}, samples=${NUM_SAMPLES}]"
echo "  Started at : ${SWEEP_START}"
echo "  Result base: ${RESULT_BASE}"
echo "  run.log    : ${RUN_LOG}"
echo "============================================================"

# ============================================================
# Phase 1: FP16 Baseline (상한 기준)
# ============================================================
echo ""
echo "---- [1/4] FP16 Baseline  ($(date '+%H:%M:%S')) ----"
FP16_SAVE="${RESULT_BASE}/FP16"
FP16_LOG="${LOG_DIR}/fp16_${MODE_LABEL}.log"

if [ -f "${FP16_SAVE}/${DATASET}/metrics.json" ]; then
    echo "  [SKIP] metrics.json already exists: ${FP16_SAVE}/${DATASET}/metrics.json"
else
    ${ACCEL_CMD} "${BASE_DIR}/pixart_quip_experiment.py" \
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
echo "---- [2/4] BASELINE: NVFP4_DEFAULT_CFG  ($(date '+%H:%M:%S')) ----"
BASELINE_SAVE="${RESULT_BASE}/BASELINE"
BASELINE_LOG="${LOG_DIR}/baseline_${MODE_LABEL}.log"

if [ -f "${BASELINE_SAVE}/${DATASET}/metrics.json" ]; then
    echo "  [SKIP] metrics.json already exists: ${BASELINE_SAVE}/${DATASET}/metrics.json"
else
    ${ACCEL_CMD} "${BASE_DIR}/pixart_quip_experiment.py" \
        --quant_method BASELINE \
        --model_path "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir "${REF_DIR}" \
        --save_dir "${BASELINE_SAVE}" \
        --num_samples ${NUM_SAMPLES} \
        --baseline_lowrank ${BASELINE_LOWRANK} \
        ${EXTRA_FLAGS} \
        2>&1 | tee "${BASELINE_LOG}"
    echo "  [$(date '+%H:%M:%S')] BASELINE done."
fi

# ============================================================
# Phase 3: W3A4-QuIP (Randomized Hadamard + INT3 + SVD + INT4-Act)
# ============================================================
echo ""
echo "---- [3/4] W3A4-QuIP (INT3 weight, rank=${LOWRANK_W3A4})  ($(date '+%H:%M:%S')) ----"
W3A4_SAVE="${RESULT_BASE}/W3A4"
W3A4_LOG="${LOG_DIR}/w3a4_${MODE_LABEL}.log"

if [ -f "${W3A4_SAVE}/${DATASET}/metrics.json" ]; then
    echo "  [SKIP] metrics.json already exists: ${W3A4_SAVE}/${DATASET}/metrics.json"
else
    ${ACCEL_CMD} "${BASE_DIR}/pixart_quip_experiment.py" \
        --quant_method W3A4 \
        --block_size ${BLOCK_SIZE} \
        --lowrank ${LOWRANK_W3A4} \
        --model_path "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir "${REF_DIR}" \
        --save_dir "${W3A4_SAVE}" \
        --num_samples ${NUM_SAMPLES} \
        ${EXTRA_FLAGS} \
        2>&1 | tee "${W3A4_LOG}"
    echo "  [$(date '+%H:%M:%S')] W3A4 done."
fi

# ============================================================
# Phase 4: W2A4-QuIP (Randomized Hadamard + INT2 + SVD(rank=64) + INT4-Act)
# ============================================================
echo ""
echo "---- [4/4] W2A4-QuIP (INT2 weight, rank=${LOWRANK_W2A4})  ($(date '+%H:%M:%S')) ----"
W2A4_SAVE="${RESULT_BASE}/W2A4"
W2A4_LOG="${LOG_DIR}/w2a4_${MODE_LABEL}.log"

if [ -f "${W2A4_SAVE}/${DATASET}/metrics.json" ]; then
    echo "  [SKIP] metrics.json already exists: ${W2A4_SAVE}/${DATASET}/metrics.json"
else
    ${ACCEL_CMD} "${BASE_DIR}/pixart_quip_experiment.py" \
        --quant_method W2A4 \
        --block_size ${BLOCK_SIZE} \
        --lowrank ${LOWRANK_W2A4} \
        --model_path "${MODEL_PATH}" \
        --dataset_name "${DATASET}" \
        --ref_dir "${REF_DIR}" \
        --save_dir "${W2A4_SAVE}" \
        --num_samples ${NUM_SAMPLES} \
        ${EXTRA_FLAGS} \
        2>&1 | tee "${W2A4_LOG}"
    echo "  [$(date '+%H:%M:%S')] W2A4 done."
fi

SWEEP_END=$(date '+%Y-%m-%d %H:%M:%S')

# ============================================================
# Phase 5: results_summary.json + handoff_quip.md 생성
# ============================================================
echo ""
echo "---- [Phase 5] Generating results_summary.json and handoff_quip.md ----"

export RESULT_BASE_PY="${RESULT_BASE}"
export DATASET_PY="${DATASET}"
export SWEEP_START_PY="${SWEEP_START}"
export SWEEP_END_PY="${SWEEP_END}"
export MODE_LABEL_PY="${MODE_LABEL}"
export NUM_SAMPLES_PY="${NUM_SAMPLES}"
export RUN_LOG_PY="${RUN_LOG}"
export BASE_DIR_PY="${BASE_DIR}"

python3 - <<'PYEOF'
import json, os, glob
from datetime import datetime

result_base  = os.environ.get("RESULT_BASE_PY")
dataset      = os.environ.get("DATASET_PY")
sweep_start  = os.environ.get("SWEEP_START_PY")
sweep_end    = os.environ.get("SWEEP_END_PY")
mode_label   = os.environ.get("MODE_LABEL_PY")
num_samples  = int(os.environ.get("NUM_SAMPLES_PY", "20"))
run_log      = os.environ.get("RUN_LOG_PY")
base_dir     = os.environ.get("BASE_DIR_PY")

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
    beat_fid = ""
    beat_is  = ""
    if baseline_fid is not None and isinstance(fid, float):
        beat_fid = "YES" if fid < baseline_fid else "no"
    if baseline_is is not None and isinstance(is_v, float):
        beat_is  = "YES" if is_v > baseline_is  else "no"
    rows.append({
        "run":      e["run"],
        "method":   cfg.get("quant_method", e["run"]),
        "fid":      f"{fid:.4f}" if isinstance(fid, float) else str(fid),
        "is":       f"{is_v:.4f}" if isinstance(is_v, float) else str(is_v),
        "psnr":     f"{sm.get('PSNR',  0):.2f}",
        "ssim":     f"{sm.get('SSIM',  0):.4f}",
        "lpips":    f"{sm.get('LPIPS', 0):.4f}",
        "clip":     f"{sm.get('CLIP',  0):.2f}",
        "beat_fid": beat_fid,
        "beat_is":  beat_is,
    })

table_lines = [
    "| Run | Method | FID↓ | IS↑ | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP↑ | Beat FID? | Beat IS? |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for r in rows:
    table_lines.append(
        f"| {r['run']} | {r['method']} | {r['fid']} | {r['is']}"
        f" | {r['psnr']} | {r['ssim']} | {r['lpips']} | {r['clip']}"
        f" | {r['beat_fid']} | {r['beat_is']} |"
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

# Baseline 대비 FID 개선된 조합
fid_winners = [r for r in rows if r["beat_fid"] == "YES"]
is_winners  = [r for r in rows if r["beat_is"]  == "YES"]
fid_winner_summary = (
    "\n".join(f"- **{w['run']}**  FID={w['fid']}  IS={w['is']}" for w in fid_winners)
    if fid_winners else "- 없음 (추가 튜닝 필요)"
)
is_winner_summary = (
    "\n".join(f"- **{w['run']}**  IS={w['is']}  FID={w['fid']}" for w in is_winners)
    if is_winners else "- 없음"
)

fp16_fid_str = f"{fp16['primary_metrics']['FID']:.4f}" if fp16 else "N/A"
fp16_is_str  = f"{fp16['primary_metrics']['IS']:.4f}"  if fp16 else "N/A"
bl_fid_str   = f"{baseline_fid:.4f}" if baseline_fid else "N/A"
bl_is_str    = f"{baseline_is:.4f}"  if baseline_is  else "N/A"

md = f"""# QuIP W3A4/W2A4 Experiment Handoff

## 개요
Randomized Hadamard (QuIP-style) + INT3/INT2 weight + SVD + INT4-Act 방식과
FP16 / NVFP4_DEFAULT_CFG(공식 baseline) 비교 실험.
총 {len(entries)}회 실행 / {num_samples} samples / dataset: {dataset}
시작: {sweep_start}  |  종료: {sweep_end}

> **파이프라인**: Random sign S_in → Block Hadamard → per-group INT3/INT2 quantization → SVD rank 보정 → INT4 activation (bs=16)
> SmoothQuant 없음 — randomized rotation이 incoherence 이론적 보장.

---

## QuIP vs GPTQ 핵심 차이

| 항목 | GPTQ (W3A4 이전 실험) | QuIP# (이번 실험) |
|---|---|---|
| Rotation | 결정론적 block Hadamard | **랜덤 sign(S_in) + block Hadamard** |
| SmoothQuant | 필요 (outlier 처리) | **불필요** (rotation이 incoherence 보장) |
| Hessian | H^-1 Cholesky → 오차 전파 | H_diag(act_order)만 — 보상 루프 없음 |
| 2-bit 적합성 | 오차 high-rank → SVD 보정 어려움 | near-Gaussian 분포 → INT2 가능 |
| S_in | 없음 | layer별 고정 random ±1 (seed=layer_idx) |

---

## Change List (신규 생성 파일)

| 파일 | 설명 |
|---|---|
| `pixart_quip_experiment.py` | QuIP W3A4/W2A4 / BASELINE / FP16 통합 실험 스크립트. QuIPLinear 클래스. |
| `run_quip_experiment.sh` | 4-run 자동화 스크립트. run.log 통합 기록, results_summary.json · handoff_quip.md 자동 생성. |
| `results/quip_experiment/results_summary.json` | 전체 실험 결과 통합 JSON. |
| `results/quip_experiment/*/MJHQ/metrics.json` | 실행별 개별 결과 JSON. |
| `logs/quip_experiment/run.log` | 전체 실험 통합 로그. |
| `handoff_quip.md` | 이 파일. |

> 기존 파일(`pixart_w3a4_experiment.py`, `run_w3a4_experiment.sh` 등)은 **일절 수정하지 않음**.

---

## 실험 설정

| 항목 | W3A4 | W2A4 |
|---|---|---|
| Weight 양자화 | INT3 uniform (7 levels) | INT2 uniform (3 levels, ternary) |
| Activation 양자화 | INT4 per-16-element | INT4 per-16-element |
| Hadamard block size | 128 | 128 |
| SVD rank | 32 | 64 |
| SmoothQuant | 없음 | 없음 |
| Random sign S_in | layer별 고정 (seed=layer_idx) | layer별 고정 (seed=layer_idx) |
| Skip layers | x_embedder, t_embedder, proj_out | x_embedder, t_embedder, proj_out |

---

## 실험 결과 (FID 오름차순)

- **FP16 baseline**: FID={fp16_fid_str}  IS={fp16_is_str}
- **NVFP4 baseline**: FID={bl_fid_str}  IS={bl_is_str}

{chr(10).join(table_lines)}

---

## Baseline 대비 FID 개선 조합

{fid_winner_summary}

## Baseline 대비 IS 개선 조합

{is_winner_summary}

---

## W3A4-QuIP vs W3A4-GPTQ 비교

| 항목 | W3A4-GPTQ (이전) | W3A4-QuIP (이번) |
|---|---|---|
| FID | 178.4 | (실험 결과 참조) |
| IS | 1.7940 | (실험 결과 참조) |
| 핵심 차이 | Hadamard + SmoothQuant + NVFP4 weight | Randomized Hadamard + INT3 weight |

---

## 저장 경로

```
results/quip_experiment/
  FP16/MJHQ/metrics.json
  BASELINE/MJHQ/metrics.json
  W3A4/MJHQ/metrics.json
  W2A4/MJHQ/metrics.json
  results_summary.json
logs/quip_experiment/
  run.log
  fp16_*.log
  baseline_*.log
  w3a4_*.log
  w2a4_*.log
```

---

## 다음 단계 제안

1. **W3A4-QuIP가 GPTQ(FID=178, IS=1.794)를 이겼다면** → num_samples=100으로 본 실험 재실행
2. **W2A4가 BASELINE을 이겼다면** → 획기적 압축비 달성 (2-bit weight). 100샘플 재실행
3. **W2A4가 실패했다면** → rank 증가 (64→128), 또는 mixed: 첫/마지막 블록 INT3, 나머지 INT2
4. **공통 개선 방향**:
   - cross-attention Q/K/V FP16 유지 (현재 skip은 embedder/proj만)
   - FID reference를 MJHQ 실제 이미지로 교체 (현재는 FP16 충실도 측정)
   - RPCA-style outlier separation 추가 (QuIP + sparse branch)
"""

handoff_path = os.path.join(base_dir, "handoff_quip.md")
with open(handoff_path, "w") as f:
    f.write(md)
print(f"  handoff_quip.md -> {handoff_path}")

# 콘솔 요약 출력
print()
print("  Results summary (FID ascending):")
print(f"  {'Run':<12} {'FID':>8} {'IS':>8} {'PSNR':>7} {'SSIM':>7} {'Beat FID?':>10} {'Beat IS?':>9}")
print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*10} {'-'*9}")
for r in rows:
    print(f"  {r['run']:<12} {r['fid']:>8} {r['is']:>8} {r['psnr']:>7} {r['ssim']:>7} {r['beat_fid']:>10} {r['beat_is']:>9}")
PYEOF

echo ""
echo "============================================================"
echo "  Experiment complete: ${SWEEP_END}"
echo "  Summary JSON  : ${RESULT_BASE}/results_summary.json"
echo "  Handoff       : ${BASE_DIR}/handoff_quip.md"
echo "  Full log      : ${RUN_LOG}"
echo "============================================================"
