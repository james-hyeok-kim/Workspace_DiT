# Experiment Result: 4-bit DiT Quantization × CacheLoRA — PixArt-Sigma

- **날짜**: 2026-04-19
- **모델**: PixArt-Sigma-XL-2-1024-MS (28 transformer blocks, hidden_dim=1152)
- **Dataset**: MJHQ-30K (xingjianleng/mjhq30k, test split, 100 samples)
- **환경**: Python 3.11, torch 2.11.0+cu130, diffusers 0.37.1, accelerate 1.13.0
- **NVFP4 group size (block_size)**: 16 (모든 method 공통)
- **ConvRot group size (N0)**: 64

---

## 20-step 결과

> deepcache/cache_lora는 best FID range 설정 기준

| Method | Cache | Range | FID ↓ | IS ↑ | SSIM ↑ | LPIPS ↓ | CLIP ↑ | Time/img (s) | cache_penalty |
|--------|-------|-------|--------|-------|--------|---------|--------|--------------|---------------|
| **RTN** | none | — | 211.2 | 1.000 | 0.502 | 0.429 | 35.1 | 8.04 | — |
| | cache_lora | r4 c8-20 | 130.3 | 5.829 | 0.474 | 0.499 | 34.4 | 6.37 | **-81.0** |
| **SVDQUANT** | none | — | 215.4 | 1.000 | 0.600 | 0.434 | 35.1 | 3.01 | — |
| | cache_lora | r4 c8-20 | 126.6 | 5.364 | 0.567 | 0.472 | 34.9 | 2.22 | **-88.9** |
| **FP4DIT** | none | — | 172.4 | 1.000 | 0.597 | 0.383 | 36.6 | 8.54 | — |
| | deepcache | c8-20 | 137.0 | 5.735 | 0.528 | 0.478 | 34.7 | 6.73 | **-35.4** |
| **HQDIT** | none | — | 672.1 | 1.000 | 0.170 | 0.817 | 27.4 | 8.30 | — |
| | cache_lora | r4 c2-26 | 327.5 | 4.495 | 0.209 | 0.780 | 26.3 | 4.71 | **-344.6** |
| **CONVROT** | none | — | 178.3 | 1.000 | 0.538 | 0.511 | 36.9 | 8.17 | — |
| | cache_lora | r4 c8-20 | 142.4 | 5.522 | 0.508 | 0.536 | 34.7 | 6.44 | **-35.9** |
| **SIXBIT** | none | — | 112.1 | 5.567 | 0.503 | 0.448 | 34.8 | 6.98 | — |
| | cache_lora | r4 c8-20 | 120.5 | 5.365 | 0.489 | 0.472 | 34.9 | 5.56 | **+8.4** |

## 15-step 결과

> deepcache/cache_lora는 best FID range 설정 기준

| Method | Cache | Range | FID ↓ | IS ↑ | SSIM ↑ | LPIPS ↓ | CLIP ↑ | Time/img (s) | cache_penalty |
|--------|-------|-------|--------|-------|--------|---------|--------|--------------|---------------|
| **RTN** | none | — | — | — | — | — | — | — | — |
| | deepcache | c8-20 | 137.4 | 5.378 | 0.450 | 0.531 | 34.3 | 4.89 | — |
| **SVDQUANT** | none | — | — | — | — | — | — | — | — |
| | deepcache | c8-20 | 130.2 | 5.627 | 0.563 | 0.476 | 34.8 | 1.92 | — |
| **FP4DIT** | none | — | — | — | — | — | — | — | — |
| | cache_lora | r4 c8-20 | 138.2 | 5.198 | 0.477 | 0.512 | 34.7 | 5.18 | — |
| **HQDIT** | none | — | — | — | — | — | — | — | — |
| | cache_lora | r4 c2-26 | 304.2 | 4.282 | 0.195 | 0.770 | 26.2 | 3.75 | — |
| **CONVROT** | none | — | — | — | — | — | — | — | — |
| | cache_lora | r4 c8-20 | 158.9 | 5.551 | 0.449 | 0.614 | 34.2 | 4.95 | — |
| **SIXBIT** | none | — | 116.5 | 5.613 | 0.483 | 0.469 | 34.6 | 5.28 | — |
| | deepcache | c8-20 | 125.5 | 5.504 | 0.464 | 0.506 | 34.5 | 4.28 | **+9.0** |

## 10-step 결과

> deepcache/cache_lora는 best FID range 설정 기준

| Method | Cache | Range | FID ↓ | IS ↑ | SSIM ↑ | LPIPS ↓ | CLIP ↑ | Time/img (s) | cache_penalty |
|--------|-------|-------|--------|-------|--------|---------|--------|--------------|---------------|
| **RTN** | none | — | 125.8 | 5.539 | 0.455 | 0.510 | 34.6 | 4.10 | — |
| | deepcache | c8-20 | 153.9 | 5.312 | 0.405 | 0.570 | 33.9 | 3.26 | **+28.1** |
| **SVDQUANT** | none | — | 121.9 | 5.641 | 0.574 | 0.452 | 34.8 | 1.44 | — |
| | deepcache | c8-20 | 140.5 | 5.408 | 0.539 | 0.505 | 34.8 | 1.18 | **+18.6** |
| **FP4DIT** | none | — | 130.5 | 5.511 | 0.487 | 0.511 | 34.7 | 4.34 | — |
| | deepcache | c8-20 | 158.1 | 5.516 | 0.455 | 0.539 | 34.4 | 3.44 | **+27.6** |
| **HQDIT** | none | — | 359.7 | 3.613 | 0.189 | 0.831 | 24.7 | 4.13 | — |
| | deepcache | c8-20 | 369.0 | 3.012 | 0.180 | 0.842 | 23.8 | 3.29 | **+9.3** |
| **CONVROT** | none | — | 140.3 | 5.729 | 0.513 | 0.537 | 34.5 | 4.15 | — |
| | deepcache | c8-20 | 160.5 | 5.182 | 0.480 | 0.575 | 34.1 | 3.30 | **+20.1** |
| **SIXBIT** | none | — | 113.5 | 5.449 | 0.466 | 0.486 | 34.7 | 3.57 | — |
| | deepcache | c8-20 | 147.4 | 5.155 | 0.410 | 0.556 | 33.7 | 2.86 | **+33.9** |

## 5-step 결과

> deepcache/cache_lora는 best FID range 설정 기준

| Method | Cache | Range | FID ↓ | IS ↑ | SSIM ↑ | LPIPS ↓ | CLIP ↑ | Time/img (s) | cache_penalty |
|--------|-------|-------|--------|-------|--------|---------|--------|--------------|---------------|
| **RTN** | none | — | 233.3 | 4.738 | 0.332 | 0.720 | 29.2 | 2.13 | — |
| | deepcache | c8-20 | 276.8 | 4.212 | 0.305 | 0.730 | 26.9 | 1.79 | **+43.5** |
| **SVDQUANT** | none | — | 292.3 | 4.420 | 0.270 | 0.820 | 27.0 | 0.82 | — |
| | deepcache | c4-24 | 365.2 | 3.221 | 0.279 | 0.806 | 21.1 | 0.63 | **+72.8** |
| **FP4DIT** | none | — | 287.4 | 4.386 | 0.307 | 0.761 | 26.8 | 2.25 | — |
| | deepcache | c8-20 | 336.8 | 3.865 | 0.277 | 0.797 | 24.0 | 1.89 | **+49.5** |
| **HQDIT** | none | — | 427.3 | 2.312 | 0.224 | 0.907 | 21.1 | 2.15 | — |
| | deepcache | c4-24 | 407.7 | 2.363 | 0.359 | 0.824 | 20.1 | 1.58 | **-19.6** |
| **CONVROT** | none | — | 286.5 | 4.615 | 0.326 | 0.799 | 27.6 | 2.16 | — |
| | deepcache | c8-20 | 364.9 | 3.591 | 0.204 | 0.865 | 22.2 | 1.81 | **+78.4** |
| **SIXBIT** | none | — | 215.7 | 4.863 | 0.335 | 0.706 | 29.8 | 1.86 | — |
| | deepcache | c8-20 | 264.1 | 4.440 | 0.304 | 0.720 | 27.6 | 1.58 | **+48.3** |

---

## cache_penalty 비교표

```
cache_penalty = FID(best_cache) - FID(none)   [음수 = cache가 FID 개선, 양수 = 악화]

Method        20-step  15-step  10-step   5-step  best_cache_mode (20-step)
RTN             -81.0        —    +28.1    +43.5  cache_lora r4 c8-20
SVDQUANT        -88.9        —    +18.6    +72.8  cache_lora r4 c8-20
FP4DIT          -35.4        —    +27.6    +49.5  deepcache c8-20
HQDIT          -344.6        —     +9.3    -19.6  cache_lora r4 c2-26
CONVROT         -35.9        —    +20.1    +78.4  cache_lora r4 c8-20
SIXBIT           +8.4     +9.0    +33.9    +48.3  cache_lora r4 c8-20
```

---

## FID 순위 (none, steps=20)

| 순위 | Method | FID |
|------|--------|-----|
| 1 | SIXBIT | 112.1 |
| 2 | FP4DIT | 172.4 |
| 3 | CONVROT | 178.3 |
| 4 | RTN | 211.2 |
| 5 | SVDQUANT | 215.4 |
| 6 | HQDIT | 672.1 |

## 속도 순위 (none, steps=20, cache_lora 제외)

| 순위 | Method | Time/img (s) |
|------|--------|-------------|
| 1 | SVDQUANT | 3.01 |
| 2 | SIXBIT | 6.98 |
| 3 | RTN | 8.04 |
| 4 | CONVROT | 8.17 |
| 5 | HQDIT | 8.30 |
| 6 | FP4DIT | 8.54 |

---

## SVDQUANT Ablation 실험 (추가)

> **목적**: SVDQUANT에서 Cache-LoRA 효과가 없는 원인 규명
> **설정**: range=[8,20) 고정, steps={10,15,20}, 20 samples (빠른 탐색)
> **CSV**: `results/svdquant_ablation_results.csv`

### 기준선 (baseline, 100 samples)

| Cache | Interval | Rank | Steps | FID ↓ | IS ↑ | Time (s) |
|-------|----------|------|-------|--------|------|----------|
| none | — | — | 10 | **121.9** | **5.641** | 1.44 |
| none | — | — | 20 | 215.4 | 1.000* | 3.01 |
| deepcache | 2 | — | 10 | 140.5 | 5.408 | 1.18 |
| deepcache | 2 | — | 15 | 130.2 | 5.627 | 1.92 |
| deepcache | 2 | — | 20 | 129.1 | 5.657 | 2.33 |
| cache_lora | 2 | 4 | 15 | 136.6 | 5.397 | 1.78 |
| cache_lora | 2 | 4 | 20 | 126.6 | 5.364 | 2.22 |

*IS=1.000: W4A4 activation 누적 오차로 IS 붕괴

### Axis 1: deepcache_interval 증가 (20 samples)

| Cache | Interval | Steps | FID ↓ | IS ↑ | 판정 |
|-------|----------|-------|--------|------|------|
| deepcache | **3** | 10 | 236.2 | 1.629 | ❌ IS 붕괴 |
| deepcache | **3** | 15 | 194.1 | 1.718 | ❌ |
| deepcache | **3** | 20 | 164.8 | 1.729 | ❌ |
| deepcache | **4** | 10 | 281.8 | 1.712 | ❌ |
| deepcache | **4** | 15 | 223.3 | 1.755 | ❌ |
| deepcache | **4** | 20 | 188.9 | 1.708 | ❌ |
| cache_lora | **3** | 20 | 197.8 | 1.780 | ❌ |
| cache_lora | **4** | 20 | 291.4 | 1.710 | ❌ |

**결론**: interval=2가 임계점. s20 기준 interval=2 → 10 full steps, interval=3 → 7 full steps로 줄어들면서 activation 누적 오차가 임계치를 초과 → IS 전면 붕괴. **interval 증가는 효과 없음.**

### Axis 2: Cache-LoRA rank 증가 (20 samples)

| Rank | Steps | FID ↓ | IS ↑ | 판정 |
|------|-------|--------|------|------|
| 4 (baseline) | 20 | 126.6 | 5.364 | ✅ |
| **8** | 10 | 276.7 | 1.692 | ❌ IS 붕괴 |
| **8** | 15 | 180.2 | 1.775 | ❌ |
| **8** | 20 | 158.9 | 1.799 | ❌ |
| **16** | 15 | 172.3 | 1.779 | ❌ |
| **16** | 20 | 161.0 | 1.798 | ❌ |
| **32** | 15 | 161.1 | 1.747 | ❌ |
| **32** | 20 | 165.4 | 1.789 | ❌ |

**결론**: rank 증가 시 오히려 악화. 원인은 **calib=4 samples로 rank=8~32 corrector 학습 → 심각한 overfitting**. rank=4만 4-sample calibration에서 안정적 (4 samples = 4 singular vectors로 정확히 fitting). rank를 늘리려면 calib 샘플 수도 함께 늘려야 함 (미검증).

### Axis 3: SVDQUANT_NOLR — 내부 LR 제거 (20 samples)

| Method | Cache | Steps | FID ↓ | IS ↑ | SSIM |
|--------|-------|-------|--------|------|------|
| SVDQUANT_NOLR | none | 10 | 641.4 | 1.000 | 0.017 |
| SVDQUANT_NOLR | none | 15 | 656.6 | 1.000 | 0.019 |
| SVDQUANT_NOLR | none | 20 | 660.5 | 1.000 | 0.021 |
| SVDQUANT_NOLR | deepcache | 10 | 641.4 | 1.000 | 0.017 |
| SVDQUANT_NOLR | cache_lora | 10 | 641.4 | 1.000 | 0.017 |

**결론**: 모든 캐싱 방법에서 동일하게 완전 붕괴 (SSIM=0.017 ≈ noise). SVDQUANT 양자화 시 `W_q = FP4(W − LR)` 구조로 저장되므로, LR 제거 시 inference에서 `(W − LR) @ x`가 계산되어 전 레이어에 systematic negative bias 발생. **LR correction은 구조적으로 제거 불가능.**

### 종합 결론

| 가설 | 결과 |
|------|------|
| interval 증가 시 더 공격적 캐싱 가능 | ❌ interval=2가 하한선. 3 이상에서 IS 전면 붕괴 |
| rank 증가로 internal LR(rank=32)과 매칭 | ❌ calib=4로 overfitting. calib 증가 시 재검토 필요 |
| LR 제거 후 Cache-LoRA로 간섭 해소 | ❌ NOLR 자체가 모델 파괴 — LR은 필수 구조 |

**SVDQUANT 최적 설정**: `none steps=10` (FID=121.9, IS=5.641, 1.44s/img) — 어떤 캐싱도 이보다 나은 결과 없음.

