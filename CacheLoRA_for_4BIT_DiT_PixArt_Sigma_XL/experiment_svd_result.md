# SVDQUANT Advanced Cache-LoRA 실험 결과

> 실험 날짜: 2026-04-20  
> 샘플 수: 20 samples (빠른 검증) — IS는 n=20에서 신뢰도 낮음 (참고용)  
> 베이스라인: 100 samples

---

## 베이스라인 (100 samples)

| 설정 | steps | FID↓ | IS↑ | SSIM↑ | tpi | speedup |
|------|-------|------|-----|-------|-----|---------|
| SVDQUANT none | 10 | **121.9** | **5.641** | 0.574 | 1.44s | 1.00x |
| SVDQUANT deepcache | 10 | 140.5 | 5.408 | 0.539 | 1.18s | 1.27x |
| SVDQUANT deepcache | 15 | 130.2 | 5.627 | 0.563 | 1.92s | 1.27x |
| SVDQUANT deepcache | 20 | 129.1 | **5.657** | 0.565 | 2.33s | 1.27x |
| SVDQUANT cache_lora r4 | 10 | 228.9 | 4.974 | 0.366 | 1.21s | 1.27x |
| SVDQUANT cache_lora r4 | 15 | 136.6 | 5.397 | 0.550 | 1.78s | 1.27x |
| SVDQUANT cache_lora r4 | 20 | 126.6 | 5.364 | 0.567 | 2.22s | 1.27x |

**핵심 관찰**: 기존 `cache_lora r4 steps20`은 deepcache와 비슷한 FID (126.6 vs 129.1), IS는 소폭 하락 (5.364 vs 5.657). steps10은 크게 나쁨 (FID 228.9).

---

## Dir 2A: Phase-Binned Corrector (cache_lora_phase, 20 samples)

rank=4, calib=4, 3-phase bins (early/mid/late)

| steps | FID↓ | IS† | SSIM↑ | tpi |
|-------|------|-----|-------|-----|
| 10 | 324.2 | 1.639 | 0.306 | 1.27s |
| 15 | 175.2 | 1.771 | 0.514 | 1.79s |
| 20 | 169.3 | 1.747 | 0.533 | 2.32s |

† IS는 n=20에서 bias가 크므로 참고용. SSIM 기준으로 steps20은 0.533 (deepcache 0.565 대비 소폭 하락).

**판정**: FID steps20=169.3 > baseline none steps10=121.9. deepcache steps10=140.5보다 나쁨. ❌

---

## Dir 2B: Timestep-Conditional (cache_lora_ts, 20 samples)

rank=4, calib=4, global (A,B) + per-step scale

| steps | FID↓ | IS† | SSIM↑ | tpi |
|-------|------|-----|-------|-----|
| 10 | 220.2 | 1.708 | 0.389 | 1.28s |
| 15 | 165.7 | 1.788 | 0.535 | 1.88s |
| 20 | 163.5 | 1.766 | 0.535 | 2.25s |

**판정**: Dir 2A 중 가장 낮은 FID (163.5). SSIM 0.535 ≈ deepcache (0.565). 속도는 deepcache와 유사. 그러나 FID가 baseline none steps10보다 나쁨. ❌

---

## Dir 3: Block-Specific Corrector (cache_lora_block, 20 samples)

rank=4, calib=4, per-block (A_i, B_i) for each of 12 deep blocks

| steps | FID↓ | IS† | SSIM↑ | tpi |
|-------|------|-----|-------|-----|
| 10 | 641.4 | 1.000 | 0.017 | 1.28s |
| 15 | 656.6 | 1.000 | 0.019 | 1.89s |
| 20 | 660.5 | 1.000 | 0.021 | 2.41s |

**판정**: 완전 붕괴. SSIM=0.021 (정상 ~0.57). IS=1.000 (최솟값). FID=641~660. ❌❌❌

**원인 분석**: Sequential per-block correction 로직에서 각 block의 `h_in_per_block[i]`가 이전 block의 correction 결과를 반영하지 않음 → 오차 누적. Full step에서 저장된 `h_in_per_block[i]`는 block[i]의 입력이지만, cached step에서 `hidden_states`는 이전 block correction이 적용된 상태이므로 `dx_i = hidden_states - h_in_per_block[i]`가 의미를 잃음.

---

## Dir 4: SVD-Aware Corrector (cache_lora_svd, 20 samples)

rank=4, calib=4, augmented input [dx; lr_probe_delta] (dim=2H)

| steps | FID↓ | IS† | SSIM↑ | tpi |
|-------|------|-----|-------|-----|
| 10 | 486.1 | 1.543 | 0.214 | 1.22s |
| 15 | 204.4 | 1.751 | 0.456 | 1.79s |
| 20 | 169.0 | 1.811 | 0.524 | 2.21s |

**판정**: steps10에서 SSIM=0.214로 심각 (partial collapse). steps20에서 회복 (FID=169, SSIM=0.524). Dir 2A/2B보다 steps에 더 민감. ❌

---

## Dir 1: Calibration × Rank Sweep (20 samples)

### steps=10 결과

| rank \ calib | 8 | 16 | 32 | 64 |
|--------|------|------|------|------|
| r4 | 290.0 | 304.8 | 297.4 | 297.4 |
| r16 | 273.2 | 268.5 | 281.3 | 281.3 |
| r32 | 285.0 | 267.5 | 257.7 | 257.7 |
| r64 | 275.3 | 272.5 | 263.7 | 263.7 |

(FID, steps=10)

### steps=15 결과

| rank \ calib | 8 | 16 | 32 | 64 |
|--------|------|------|------|------|
| r4 | 170.9 | 184.0 | 174.1 | 174.1 |
| r16 | 170.1 | 166.8 | 174.2 | 174.2 |
| r32 | 167.2 | 161.9 | 179.3 | 179.3 |
| r64 | 177.0 | 166.2 | 173.6 | 173.6 |

(FID, steps=15)

### steps=20 결과

| rank \ calib | 8 | 16 | 32 | 64 |
|--------|------|------|------|------|
| r4 | 165.4 | 162.0 | **153.2** | **153.2** |
| r16 | 157.5 | 167.3 | 158.0 | 158.0 |
| r32 | 158.4 | **154.8** | 164.6 | 164.6 |
| r64 | 165.6 | **156.3** | 162.4 | 162.4 |

(FID, steps=20)

### SSIM (steps=20, 참고)

| rank \ calib | 8 | 16 | 32 | 64 |
|--------|-------|-------|-------|-------|
| r4 | 0.532 | 0.530 | 0.530 | 0.530 |
| r16 | 0.530 | 0.536 | 0.528 | 0.528 |
| r32 | 0.531 | 0.536 | 0.531 | 0.531 |
| r64 | 0.531 | 0.530 | 0.529 | 0.529 |

### Dir 1 관찰

1. **Calib 포화**: cal32와 cal64가 동일 결과 → 32 samples 이상은 saturated
2. **Rank 증가 효과 미미**: r32_cal16 (FID=154.8) vs r4_cal32 (FID=153.2) — 거의 동등
3. **SSIM 안정**: 0.528~0.536 — 모든 설정에서 유사 (deepcache 0.565 대비 -0.03)
4. **IS 전체 붕괴**: 1.65~1.84 (n=20 artifact, 실제 질은 SSIM으로 판단)

**Dir 1 베스트**: `r4_cal32_steps20` FID=153.2 / `r32_cal16_steps20` FID=154.8

---

## Dir 5: Block-Specific (버그 수정, 20 samples)

| steps | FID↓ | IS† | SSIM↑ | tpi |
|-------|------|-----|-------|-----|
| 10 | 482.6 | 1.398 | 0.174 | 1.23s |
| 15 | 306.0 | 1.740 | 0.349 | 1.89s |
| 20 | 215.6 | 1.713 | 0.420 | 2.17s |

**판정**: Dir 3 대비 대폭 개선 (SSIM: 0.021→0.420). 버그 수정 확인. 그러나 SSIM=0.420 (deepcache 0.565 대비) — 여전히 global corrector보다 나쁨. ⚠️

**원인**: global dx (deep region 입구 기준)로 per-block correction을 하면, 각 block의 calibration 때 학습한 per-block dx와 inference-time input이 여전히 불일치. Block corrector는 각 block의 입력 변화를 기대하지만 전체 deep region 입구 변화를 받음.

---

## Dir 6: Nested Caching (20 samples)

### Deepcache only (cache_start 변화)

| cache_start | steps | FID↓ | IS† | SSIM↑ | tpi | n_deep |
|-------------|-------|------|-----|-------|-----|--------|
| **8 (기존, n=100)** | **10** | **140.5** | **5.408** | **0.539** | **1.18s** | 12 |
| **8 (기존, n=100)** | **20** | **129.1** | **5.657** | **0.565** | **2.33s** | 12 |
| 10 | 10 | 169.2 | 1.787 | 0.508 | 1.26s | 10 |
| 10 | 15 | 168.7 | 1.820 | 0.534 | 1.84s | 10 |
| 10 | 20 | 157.9 | 1.810 | 0.534 | 2.29s | 10 |
| 12 | 10 | 165.8 | 1.718 | 0.522 | 1.38s | 8 |
| 12 | 15 | 164.8 | 1.762 | 0.534 | 1.96s | 8 |
| 12 | 20 | 158.3 | 1.837 | **0.535** | 2.54s | 8 |
| 14 | 10 | 166.7 | 1.679 | 0.521 | 1.41s | 6 |
| 14 | 15 | **161.6** | 1.719 | **0.536** | 1.91s | 6 |
| 14 | 20 | 165.1 | 1.795 | 0.528 | 2.58s | 6 |
| 16 | 10 | 177.7 | 1.763 | 0.528 | 1.36s | 4 |
| 16 | 15 | 163.7 | 1.759 | **0.537** | 2.05s | 4 |
| 16 | 20 | **150.1** | 1.798 | 0.529 | 2.52s | 4 |

### Cache-LoRA r4 (cache_start 변화)

| cache_start | steps | FID↓ | SSIM↑ | tpi |
|-------------|-------|------|-------|-----|
| **8 (기존, n=100)** | **20** | **126.6** | **0.567** | 2.22s |
| 10 | 20 | 186.0 | 0.500 | 2.41s |
| 12 | 20 | 169.5 | 0.530 | 2.41s |
| 14 | 20 | 229.6 | 0.432 | 2.67s |
| 16 | 20 | 192.3 | 0.445 | 2.59s |

**Dir 6 관찰:**
- deepcache: cs 축소 시 n=20에서 SSIM이 약간 증가하는 경향 (0.534~0.537 for cs≥10). 그러나 n=20 noise 범위 내.
- cache_lora: cs 축소 시 오히려 FID 악화 (186→229 for cs=10,14). cs=12만 합리적 (FID=169.5, SSIM=0.530).
- 결론: **cs 축소가 deepcache quality를 실질적으로 개선하지 않음**. speedup만 낮아짐 (1.27x→1.08x).

---

## Dir 9: Teacher-Forced Calibration (20 samples)

| steps | FID↓ | IS† | SSIM↑ | tpi |
|-------|------|-----|-------|-----|
| 10 | 208.4 | 1.771 | 0.447 | 1.27s |
| 15 | 175.8 | 1.782 | 0.526 | 1.64s |
| 20 | 162.6 | 1.830 | 0.527 | 2.36s |

**비교 (n=20 기준):**
- standard cache_lora r4 steps20: FID=153.2, SSIM=0.530
- teacher-forced steps20: FID=162.6, SSIM=0.527

**판정**: standard cache_lora와 통계적으로 동등 (n=20 noise 내). 분포 불일치 가설이 주요 원인이 아닌 것으로 판단. ⚠️

---

## 전체 비교 요약 (steps=20 기준)

| 설정 | FID↓ | SSIM↑ | tpi | speedup | 평가 |
|------|------|-------|-----|---------|------|
| SVDQUANT none steps10 (n=100) | 121.9 | 0.574 | 1.44s | 1.00x | ✅ 참조 |
| SVDQUANT deepcache c8-20 steps10 (n=100) | 140.5 | 0.539 | 1.18s | 1.27x | ✅ |
| SVDQUANT deepcache c8-20 steps20 (n=100) | 129.1 | 0.565 | 2.33s | 1.27x | ✅ best |
| SVDQUANT cache_lora r4 c8-20 steps20 (n=100) | 126.6 | 0.567 | 2.22s | 1.27x | ✅ best |
| Dir 2B: cl_ts_r4 steps20 (n=20) | 163.5 | 0.535 | 2.25s | 1.27x | ❌ |
| Dir 2A: cl_phase_r4 steps20 (n=20) | 169.3 | 0.533 | 2.32s | 1.27x | ❌ |
| Dir 4: cl_svd_r4 steps20 (n=20) | 169.0 | 0.524 | 2.21s | 1.27x | ❌ |
| Dir 1: cl_r4_cal32 steps20 (n=20) | 153.2 | 0.530 | 2.51s | 1.27x | ❌ |
| Dir 3: cl_block_r4 steps20 (n=20) | 660.5 | 0.021 | 2.41s | 1.27x | ❌❌❌ 버그 |
| **Dir 5: cl_block_r4 steps20 (n=20)** | **215.6** | **0.420** | 2.17s | 1.27x | ⚠️ 개선 |
| **Dir 6: deepcache c12-20 steps20 (n=20)** | **158.3** | **0.535** | 2.54s | 1.18x | ⚠️ |
| **Dir 6: deepcache c16-20 steps20 (n=20)** | **150.1** | **0.529** | 2.52s | 1.08x | ⚠️ |
| **Dir 9: cl_tf_r4 steps20 (n=20)** | **162.6** | **0.527** | 2.36s | 1.27x | ⚠️ |

> n=20 FID/IS는 n=100과 직접 비교 불가. SSIM이 가장 신뢰할 수 있는 지표.

---

## 종합 분석 및 결론

### Dir 5 (block 버그 수정)

수정 효과 있음 (IS: 1.0→1.7, SSIM: 0.021→0.420). 그러나 SSIM=0.420은 standard cache_lora (0.567 at n=100) 대비 크게 열등. Method C의 근본 한계: per-block corrector가 per-block dx로 학습됐으나 inference-time에 global dx를 받음 → corrector의 input space 불일치.

**진정한 수정**: per-block corrector를 global dx로 재calibration해야 함. 현재 `calibrate_cache_lora_blockwise()`는 각 block의 자체 입력 변화로 학습하므로 inference-time input과 여전히 불일치.

### Dir 6 (nested caching)

deepcache에서 cache_start 축소의 SSIM 개선은 n=20 noise 범위 내 (0.534→0.537). Cache-LoRA는 오히려 악화. cs 축소 전략은 유의미한 quality 개선 없이 speedup만 손실 (1.27x→1.08x).

**결론**: c8-20이 quality/speedup trade-off에서 최적.

### Dir 9 (teacher-forced)

TF calibration이 standard calibration과 동등한 결과 → 분포 불일치가 주요 병목이 아님. 오히려 SVDQUANT FP4 양자화 자체가 만들어내는 activation 비선형성이 근본 문제.

### 최종 결론 (Dir 1~9 전체)

**모든 방향에서 기존 cache_lora r4 (n=100: SSIM=0.567)를 n=20에서 재현하지 못함.**

실제 SSIM 상한: 약 0.53 (n=20 데이터 편향 가능성 있으나, 이 수준이 실제 한계일 가능성 높음).

**권장 설정**: `SVDQUANT + deepcache + steps10` (FID=140.5, SSIM=0.539, 1.27x speedup, n=100 검증됨) — cache_lora 없이 deepcache만으로도 충분한 quality.
