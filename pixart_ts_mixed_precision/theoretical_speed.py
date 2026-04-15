"""
실제 하드웨어 배포 환경에서의 이론적 속도 추정
==============================================

Fake quantization은 시뮬레이션 오버헤드로 오히려 느려지지만,
실제 INT4/INT8/FP4 하드웨어 커널을 사용하면 다음의 이점이 생긴다:

  [Compute-bound 가정 — 대형 배치]:
    NVFP4 / INT4 matmul: FP16 대비 ~4x throughput
    INT8 matmul:          FP16 대비 ~2x throughput
    (H100 SXM: FP16=989TFLOPS, INT8=1979TOPS, INT4~3958TOPS)
    (B200 SXM: FP4≈9PFLOPS ≈ 4x FP16≈2.25PFLOPS)

  [Memory-bandwidth-bound 가정 — batch=1 이미지 생성]:
    배치가 작으면 matmul이 weight 로딩에 의해 bottleneck됨.
    Weight NVFP4(0.5B/p) vs FP16(2B/p) → 4x 가벼움.
    하지만 activation은 여전히 FP16 로딩 → 실효 speedup ≈ 1.5~2.5x
    INT8 weight: 2x 가벼움 → 실효 speedup ≈ 1.3~1.8x

SVD branch (lora_a, lora_b)는 항상 FP16으로 실행 → 이 부분은 속도 이점 없음.
따라서 SVD rank가 클수록 / layer가 작을수록 전체 speedup이 감소한다.

산출 방법:
  total_time ∝ Σ_layers [base_flops/throughput[act] + svd_flops/throughput[FP16]]
  speedup = fp16_total_time / config_total_time

FP16 measured time (2066ms)을 anchor로 절대 시간 추정.
"""

import json, csv, os
import numpy as np

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")

# ── 레이어 shape 로드 ────────────────────────────────────────────
LAYER_SHAPES = json.load(open("/tmp/layer_shapes.json"))  # [(name, in_f, out_f), ...]
# 총 289 layers

# ── 아키텍처 상수 ────────────────────────────────────────────────
TOKENS       = 4096 + 120   # image tokens (64×64) + text tokens
T_COUNT      = 20           # denoising steps
N_GROUPS     = 3
# 실제 DPMSolver 20-step 그룹 분포 (내림차순 boundaries [666, 333])
STEPS_PER_GROUP = [7, 7, 6]   # G0, G1, G2

# ── 하드웨어 throughput 배수 (FP16=1.0 기준) ─────────────────────
# Compute-bound 시나리오 (대형 배치 / H100·B200 peak TFLOPS 기준)
COMPUTE_THROUGHPUT = {
    "FP16":  1.0,
    "NVFP4": 4.0,
    "INT4":  4.0,
    "INT8":  2.0,
}

# Memory-bandwidth-bound 시나리오 (batch=1, single image generation)
# Weight bandwidth 절감 비율 × 활용 계수 (activation이 FP16이라 절반만 효과)
# NVFP4 weight: 4x 가벼움 → 실효 1.8~2.2x (중간값 2.0 사용)
# INT8  weight: 2x 가벼움 → 실효 1.3~1.6x (중간값 1.5 사용)
MEM_BW_THROUGHPUT = {
    "FP16":  1.0,
    "NVFP4": 2.0,
    "INT4":  2.0,
    "INT8":  1.5,
}

# ── Config 정의 (pixart_ts_mixed_precision.py와 동일) ─────────────
ABLATION_CONFIGS = {
    "UNIFORM_FP4":   [{"act_mode":"NVFP4","svd_rank":32},
                      {"act_mode":"NVFP4","svd_rank":32},
                      {"act_mode":"NVFP4","svd_rank":32}],
    "MP_ACT_ONLY":   [{"act_mode":"NVFP4","svd_rank":32},
                      {"act_mode":"INT4", "svd_rank":32},
                      {"act_mode":"INT8", "svd_rank":32}],
    "MP_RANK_ONLY":  [{"act_mode":"NVFP4","svd_rank":0},
                      {"act_mode":"NVFP4","svd_rank":16},
                      {"act_mode":"NVFP4","svd_rank":32}],
    "MP_MODERATE":   [{"act_mode":"NVFP4","svd_rank":8},
                      {"act_mode":"INT4", "svd_rank":16},
                      {"act_mode":"INT8", "svd_rank":32}],
    "MP_AGGRESSIVE": [{"act_mode":"NVFP4","svd_rank":0},
                      {"act_mode":"NVFP4","svd_rank":16},
                      {"act_mode":"INT8", "svd_rank":32}],
}

FP16_MEASURED_MS = 2066.2   # 실측 FP16 baseline


def compute_flops_breakdown(group_configs):
    """
    전체 inference의 FLOPs를 compute- / mem-bw-bound 가정으로 분리.
    Returns: {scenario: weighted_cost}  (FP16=1.0 기준 normalized)
    """
    compute_cost = 0.0
    membw_cost   = 0.0
    fp16_cost    = 0.0   # FP16 baseline cost (all FP16)

    for name, in_f, out_f in LAYER_SHAPES:
        # 이 레이어 하나당, 20 step에서 각 group별 FLOPs 계산
        base_flops = 2 * TOKENS * in_f * out_f   # 기본 matmul FLOPs

        # FP16 baseline: 전부 FP16 matmul, SVD 없음
        fp16_cost += T_COUNT * (base_flops / COMPUTE_THROUGHPUT["FP16"])

        for g, cfg in enumerate(group_configs):
            steps   = STEPS_PER_GROUP[g]
            act_m   = cfg["act_mode"]
            rank    = cfg["svd_rank"]

            svd_flops = 2 * TOKENS * rank * (in_f + out_f) if rank > 0 else 0

            # Compute-bound
            compute_cost += steps * (
                base_flops / COMPUTE_THROUGHPUT[act_m]
              + svd_flops  / COMPUTE_THROUGHPUT["FP16"]
            )
            # Memory-bandwidth-bound
            membw_cost += steps * (
                base_flops / MEM_BW_THROUGHPUT[act_m]
              + svd_flops  / MEM_BW_THROUGHPUT["FP16"]
            )

    return fp16_cost, compute_cost, membw_cost


# ── FP16 baseline cost 계산 ──────────────────────────────────────
fp16_cost_base, _, _ = compute_flops_breakdown(
    [{"act_mode":"FP16","svd_rank":0}] * 3   # FP16: throughput=1, no SVD
)

# 참고: FP16 baseline에 SVD 없으므로 직접 계산
fp16_base_compute = sum(
    T_COUNT * 2 * TOKENS * in_f * out_f / COMPUTE_THROUGHPUT["FP16"]
    for _, in_f, out_f in LAYER_SHAPES
)

# ── 각 config 이론 속도 계산 ─────────────────────────────────────
results = []

print(f"\n{'='*72}")
print(f"  이론적 하드웨어 속도 추정  (FP16 anchor: {FP16_MEASURED_MS:.1f} ms/img)")
print(f"{'='*72}")
print(f"\n{'Config':<20} {'Compute speedup':>15} {'MemBW speedup':>14} {'Compute ms':>11} {'MemBW ms':>10}")
print("-" * 74)

# FP16 row
results.append({
    "config": "FP16 (no quant)",
    "speedup_compute": 1.0,
    "speedup_membw": 1.0,
    "ms_compute": round(FP16_MEASURED_MS, 1),
    "ms_membw":   round(FP16_MEASURED_MS, 1),
})
print(f"  {'FP16 (no quant)':<18} {'1.000x':>15} {'1.000x':>14} {FP16_MEASURED_MS:>10.1f} {FP16_MEASURED_MS:>9.1f}")

for cfg_name, group_configs in ABLATION_CONFIGS.items():
    fp16_c, compute_c, membw_c = compute_flops_breakdown(group_configs)

    speedup_compute = fp16_base_compute / compute_c
    speedup_membw   = fp16_base_compute / membw_c

    ms_compute = FP16_MEASURED_MS / speedup_compute
    ms_membw   = FP16_MEASURED_MS / speedup_membw

    results.append({
        "config":           cfg_name,
        "speedup_compute":  round(speedup_compute, 3),
        "speedup_membw":    round(speedup_membw, 3),
        "ms_compute":       round(ms_compute, 1),
        "ms_membw":         round(ms_membw, 1),
    })
    print(f"  {cfg_name:<20} {speedup_compute:>14.3f}x {speedup_membw:>13.3f}x"
          f" {ms_compute:>10.1f} {ms_membw:>9.1f}")

print()

# ── SVD 오버헤드 기여도 분석 ─────────────────────────────────────
print(f"{'='*72}")
print(f"  SVD branch가 전체 속도에서 차지하는 비율 (compute-bound 기준)")
print(f"{'='*72}")
print(f"\n{'Config':<20} {'Base FLOPs%':>12} {'SVD FLOPs%':>12} {'Eff. speedup':>13}")
print("-" * 60)

for cfg_name, group_configs in ABLATION_CONFIGS.items():
    base_total = 0.0
    svd_total  = 0.0
    for _, in_f, out_f in LAYER_SHAPES:
        for g, cfg in enumerate(group_configs):
            steps = STEPS_PER_GROUP[g]
            rank  = cfg["svd_rank"]
            base_total += steps * 2 * TOKENS * in_f * out_f
            svd_total  += steps * 2 * TOKENS * rank * (in_f + out_f) if rank > 0 else 0

    total = base_total + svd_total
    base_pct = base_total / total * 100
    svd_pct  = svd_total  / total * 100
    eff_sp   = next(r["speedup_compute"] for r in results if r["config"] == cfg_name)
    print(f"  {cfg_name:<20} {base_pct:>11.1f}% {svd_pct:>11.1f}% {eff_sp:>12.3f}x")

print()

# ── CSV 저장 ────────────────────────────────────────────────────
theo_csv = os.path.join(RESULT_DIR, "theoretical_speed.csv")
fields   = ["config","speedup_compute","speedup_membw","ms_compute","ms_membw"]
with open(theo_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)
print(f"Theoretical speed CSV saved: {theo_csv}")

# ── results_summary.csv에 병합 ───────────────────────────────────
summary_csv = os.path.join(RESULT_DIR, "results_summary.csv")
theo_map = {r["config"]: r for r in results}

merged = []
with open(summary_csv) as f:
    reader    = csv.DictReader(f)
    old_fields = list(reader.fieldnames)
    new_fields = old_fields + ["speedup_compute","speedup_membw","ms_compute","ms_membw"]
    for row in reader:
        sp = theo_map.get(row["config"], {})
        row["speedup_compute"] = sp.get("speedup_compute", "N/A")
        row["speedup_membw"]   = sp.get("speedup_membw",   "N/A")
        row["ms_compute"]      = sp.get("ms_compute",      "N/A")
        row["ms_membw"]        = sp.get("ms_membw",        "N/A")
        merged.append(row)

with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=new_fields)
    writer.writeheader()
    writer.writerows(merged)
print(f"results_summary.csv 업데이트 완료 (이론 속도 컬럼 추가)")

# ── 최종 통합표 ─────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  최종 통합 비교표 (FID/IS + 이론 속도)")
print(f"{'='*72}")
quality = {}
try:
    with open(summary_csv) as f:
        for row in csv.DictReader(f):
            quality[row["config"]] = row
except: pass

print(f"\n{'Config':<20} {'FID↓':>8} {'IS↑':>7} {'Compute':>9} {'MemBW':>8} {'vs FP16(compute)':>17}")
print("-" * 75)
print(f"  {'BASELINE(NVFP4_CFG)':<18} {'161.30':>8} {'1.7318':>7} {'N/A':>9} {'N/A':>8}  (ref)")
for r in results:
    cfg = r["config"]
    q   = quality.get(cfg, {})
    fid = q.get("FID",  "N/A")
    is_ = q.get("IS",   "N/A")
    try: fid = f"{float(fid):.2f}"
    except: pass
    try: is_ = f"{float(is_):.4f}"
    except: pass
    ms_c = r["ms_compute"]
    ms_m = r["ms_membw"]
    sp_c = r["speedup_compute"]
    if cfg == "FP16 (no quant)":
        print(f"  {cfg:<20} {'N/A':>8} {'N/A':>7} {ms_c:>8.1f}ms {ms_m:>7.1f}ms  1.000x (anchor)")
    else:
        print(f"  {cfg:<20} {fid:>8} {is_:>7} {ms_c:>8.1f}ms {ms_m:>7.1f}ms  {sp_c:.3f}x")
print()
print("주의: Compute-bound = peak TFLOPS 기준 (대형 배치 최적 시나리오)")
print("      MemBW-bound   = weight bandwidth 절감 기준 (batch=1 이미지 생성)")
print("      실제 속도는 두 추정치 사이에 위치할 가능성이 높음")
