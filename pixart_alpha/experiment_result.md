# PixArt-Alpha PTQ 실험 결과 총정리

**작성일**: 2026-04-14  
**모델**: PixArt-Alpha XL-2 1024-MS  
**데이터셋**: MJHQ-30K (20 samples)  
**목표**: NVFP4_SVDQUANT_DEFAULT_CFG (FID=161.30, IS=1.7318) 대비 FID↓ + IS↑ 동시 달성

---

## 기준값 (Reference)

| 이름 | FID ↓ | IS ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP ↑ |
|------|-------|------|--------|--------|---------|--------|
| **FP16 (no quant)** | ~0 | 1.7172 | ∞ | 1.0 | 0.0 | 35.52 |
| **BASELINE (NVFP4_DEFAULT_CFG)** | **161.30** | **1.7318** | 15.69 | 0.5902 | 0.4406 | 34.98 |

> BASELINE = NVIDIA ModelOpt `mtq.NVFP4_SVDQUANT_DEFAULT_CFG` 공식 설정. 모든 실험의 비교 기준.

---

## 실험 계보 및 주요 결과

### 1. RPCA Sweep (가장 성과 있는 실험)

**접근**: 각 Linear weight를 Sparse(outlier) + Dense(low-rank)로 분해 후 양자화.  
`outlier_ratio`로 outlier를 FP16 sparse branch에 격리, dense part만 양자화.

| Config | W | A | OR | FID ↓ | IS ↑ | PSNR ↑ | vs Baseline |
|--------|---|---|----|-------|------|--------|-------------|
| RPCA_AINT8_WINT8_OR0.1 | INT8 | INT8 | 0.10 | **18.16** | 1.6777 | 26.66 | FID↓↓ IS↓ |
| RPCA_AINT8_WINT8_OR0.05 | INT8 | INT8 | 0.05 | **22.96** | 1.7300 | 25.79 | FID↓↓ IS≈ |
| RPCA_AINT8_WINT8_OR0.01 | INT8 | INT8 | 0.01 | **27.96** | 1.7392 | 26.14 | FID↓↓ IS≈ |
| RPCA_AINT8_WINT8_OR0 | INT8 | INT8 | 0.00 | **27.09** | 1.7315 | 24.81 | FID↓↓ IS≈ |
| RPCA_AINT8_WINT4_OR0.1 | INT4 | INT8 | 0.10 | **80.34** | 1.7081 | 18.46 | FID↓ IS≈ |
| RPCA_AINT8_WINT4_OR0.05 | INT4 | INT8 | 0.05 | **97.14** | 1.7951 | 17.47 | **FID↓ IS↑** |
| RPCA_AINT4_WINT4_OR0.05 | INT4 | INT4 | 0.05 | **111.47** | 1.7218 | 16.45 | FID↓ IS≈ |
| RPCA_ANVFP4_WNVFP4_OR0.1 | NVFP4 | NVFP4 | 0.10 | **119.41** | **1.7617** | 17.08 | **FID↓ IS↑** |
| RPCA_AINT4_WINT4_OR0.1 | INT4 | INT4 | 0.10 | **120.62** | 1.7187 | 17.30 | FID↓ IS≈ |
| RPCA_AINT8_WINT4_OR0.01 | INT4 | INT8 | 0.01 | **124.72** | 1.7085 | 15.53 | FID↓ IS≈ |
| RPCA_ANVFP4_WNVFP4_OR0.05 | NVFP4 | NVFP4 | 0.05 | **136.02** | **1.7447** | 16.06 | **FID↓ IS↑** |
| RPCA_AINT8_WINT4_OR0 | INT4 | INT8 | 0.00 | **139.00** | 1.7564 | 15.77 | FID↓ IS↑ |
| RPCA_ANVFP4_WNVFP4_OR0.01 | NVFP4 | NVFP4 | 0.01 | **161.16** | **1.7612** | 15.37 | FID≈ IS↑ |
| RPCA_AINT4_WINT4_OR0.01 | INT4 | INT4 | 0.01 | 163.48 | 1.7993 | 14.25 | IS↑ |

**핵심 인사이트**:
- `outlier_ratio=0.05~0.1` 구간에서 FID가 크게 개선됨
- INT8 weight+act가 FID 기준 최고이나 실질 배포 이점은 낮음 (INT8 ≈ FP16 2x)
- **RPCA_ANVFP4_WNVFP4_OR0.1**: NVFP4 유지하면서 FID+IS 둘 다 baseline 초과 달성

---

### 2. Knowledge Distillation with RPCA

**접근**: RPCA 양자화 후 KD(Knowledge Distillation)로 FP16 teacher에 맞게 fine-tuning.  
KD step 수를 sweep하여 최적 stopping point 탐색.

| Config | KD Steps | FID ↓ | IS ↑ | PSNR ↑ | vs Baseline |
|--------|----------|-------|------|--------|-------------|
| RPCA_NVFP4_NOKD | — | 160.49 | 1.7631 | 12.90 | IS↑ |
| **RPCA_NVFP4_KD50** | 50 | **159.46** | **1.8265** | 13.11 | **FID↓ IS↑** |
| RPCA_NVFP4_KD70 | 70 | 171.47 | 1.7971 | 13.08 | IS↑ |
| RPCA_NVFP4_KD100 | 100 | 166.83 | 1.7797 | 13.31 | IS↑ |
| RPCA_NVFP4_KD300 | 300 | 184.70 | 1.7199 | 12.90 | — |
| RPCA_NVFP4_KD1000 | 1000 | 557.05 | 1.1200 | 6.08 | 붕괴 |
| SVD_NVFP4_KD50 | 50 | 164.12 | 1.7748 | 12.75 | IS↑ |
| SVD_NVFP4_KD300 | 300 | 166.98 | 1.7566 | 13.15 | IS↑ |

**핵심 인사이트**:
- KD step=50이 sweet spot. 과도한 KD(300+)는 오히려 FID 악화.
- KD1000에서 완전 붕괴 → gradient 누적으로 quantized weight 발산.
- RPCA+KD50 조합이 IS 기준 전 실험 통틀어 최고값(1.8265) 달성.

---

### 3. Cascade-3 Act/Weight 실험

**접근**: Layer별 민감도 분석(sensitivity)으로 MXFP6/MXFP8 혼합 적용.  
두 가지 variant: Hadamard rotation 후 양자화 vs SmoothQuant 사용.

| Config | 방식 | FID ↓ | IS ↑ | PSNR ↑ | vs Baseline |
|--------|------|-------|------|--------|-------------|
| **cascade_hadamard** | Hadamard + Cascade-3 | **132.10** | **1.7562** | 16.77 | **FID↓ IS↑** |
| cascade_smooth | SmoothQuant + Cascade-3 | **152.58** | **1.7636** | 14.97 | **FID↓ IS↑** |
| cascade_hadamard_varA | 변형 A | 657.03 | 1.0000 | 6.28 | 붕괴 |
| cascade_hadamard_varB | 변형 B | 314.70 | 1.6505 | 11.45 | — |
| cascade_hadamard_varC | 변형 C | 384.31 | 1.7051 | 11.40 | — |
| cascade_hadamard_smooth_varG | 혼합 G | 358.33 | 1.6406 | 12.08 | — |
| cascade_wFP3AUTO_aFP3AUTO_r64 | FP3 자동 | 502.01 | 1.4390 | 7.02 | 붕괴 |

**핵심 인사이트**:
- baseline variant(hadamard, smooth)는 FID+IS 모두 개선.
- FP3(3-bit float) 적용 시 붕괴. MXFP6 이하가 안전 하한선.
- hadamard rotation이 smooth보다 FID 면에서 더 효과적.

---

### 4. Timestep-Aware PTQ (기존 실험)

**접근**: Timestep group별 다른 SmoothQuant scale 적용 (bit-width는 동일).  
`pixart_ts_aware_experiment.py` 구현.

| Config | n_groups | correction | FID ↓ | IS ↑ | PSNR ↑ | vs Baseline |
|--------|----------|------------|-------|------|--------|-------------|
| TSAWARE_G3_NOCORR | 3 | ✗ | 174.30 | **1.8073** | 13.41 | IS↑ |
| TSAWARE_G5_NOCORR | 5 | ✗ | 177.74 | **1.7415** | 13.12 | IS↑ |
| TSAWARE_G3_RANK4 | 3 | ✓(rank=4) | 402.98 | 1.4180 | 11.92 | 붕괴 |
| TSAWARE_G5_RANK4 | 5 | ✓(rank=4) | 409.88 | 1.4303 | 11.96 | 붕괴 |

> (ts_aware_nvfp4a 변형도 유사한 패턴. IS↑ but FID↑)

**핵심 인사이트**:
- Timestep별 scale 변화는 IS를 소폭 개선하나 FID는 악화.
- per-group SVD correction(rank=4) 추가 시 오히려 붕괴 → global weight 공간 불일치.
- **버그 발견**: boundaries를 오름차순으로 검색하여 G1이 항상 0 step 배정됨. 후속 실험에서 수정.

---

### 5. Timestep-Aware Mixed Precision (이번 실험)

**접근**: Activation bit-width + SVD rank를 timestep group별로 다르게 적용.  
`pixart_ts_mixed_precision.py` 구현. (버그 수정 포함)

| Config | Act (G0/G1/G2) | SVD rank (G0/G1/G2) | FID ↓ | IS ↑ | 이론 speedup |
|--------|----------------|----------------------|-------|------|------------|
| UNIFORM_FP4 | FP4/FP4/FP4 | 32/32/32 | 177.98 | 1.7577 | 3.4x (compute) |
| MP_ACT_ONLY | FP4/INT4/INT8 | 32/32/32 | 167.65 | 1.7202 | 2.7x |
| MP_RANK_ONLY | FP4/FP4/FP4 | 0/16/32 | 200.97 | 1.7401 | **3.7x** |
| MP_MODERATE | FP4/INT4/INT8 | 8/16/32 | 172.80 | 1.6778 | 2.9x |
| MP_AGGRESSIVE | FP4/FP4/INT8 | 0/16/32 | 193.09 | **1.7327** | 2.9x |

**핵심 인사이트**:
- 모든 config이 FID 기준 baseline 미달 (161.30 초과).
- IS 기준으로 일부 config이 소폭 개선되나 유의미하지 않음.
- Activation precision 변화보다 SVD rank 변화가 속도에 더 큰 영향.
- **근본 한계**: fake quantization 방식은 실제 하드웨어 속도 이점 없음. Timestep-aware activation 변화가 품질 개선에 기여하지 못함.

---

### 6. Tucker Decomposition

**접근**: Weight matrix를 Tucker 분해(core tensor + factor matrices)로 표현 후 core만 저장.  
`reduction_ratio`로 core 크기 조절.

| Config | RR | Core | FID ↓ | IS ↑ | 비고 |
|--------|-----|------|-------|------|------|
| TUCKER_RR1p00_COREINT3_UFP16 | 1.00 | INT3 | **115.90** | **3.4542** | IS 최고값! |
| TUCKER_RR0p75_COREINT2_UFP16 | 0.75 | INT2 | 270.73 | **3.3374** | IS↑↑ FID↑ |
| TUCKER_RR0p75_COREINT3_UFP16 | 0.75 | INT3 | 270.48 | **3.3067** | IS↑↑ FID↑ |
| TUCKER_RR0p75_COREINT4_UFP16 | 0.75 | INT4 | 274.94 | **3.2720** | IS↑↑ FID↑ |
| TUCKER_INV_RR1p00_UINT4 | 1.00 | INT4 | 173.88 | **3.5759** | IS 최고 중 하나 |
| TUCKER_INV_RR0p75_UINT4 | 0.75 | INT4 | 322.33 | 3.0587 | — |
| TUCKER_R64_COREINT3/4/8 | — | — | 465-468 | 1.28-1.32 | 붕괴 |

**핵심 인사이트**:
- Tucker IS 값이 매우 높음(3.4-3.5) → 이미지 다양성/선명도 우수.
- RR=1.00(압축 없음)일 때만 FID가 baseline 근처로 내려옴. RR<1이면 FID 급상승.
- Tucker 고유한 특성: IS 높지만 FID 나쁨 → 개별 이미지는 선명하나 분포가 실제와 다름.
- KD 적용해도 개선 없음(tucker_kd 결과).

---

### 7. 기타 실험 (성과 없음)

| 실험 계열 | 최고 FID | 최고 IS | 결론 |
|-----------|----------|---------|------|
| CP (CANDECOMP) | 502.84 | 1.15 | 완전 붕괴, 접근 부적합 |
| KSVD | 456.04 | 1.38 | 붕괴 |
| INT3+KD50 | 255.69 | 1.8044 | IS↑이나 FID 너무 나쁨 |
| QuIP/GPTQ (INT4) | 188.83 | 1.7376 | baseline 미달 |
| QuIP (W2A4) | 483.08 | 1.52 | 붕괴 |

---

## 전체 Best Results 순위 (FID 기준)

| 순위 | 실험 | FID ↓ | IS ↑ | PSNR ↑ | FID+IS 모두 ↑? |
|------|------|-------|------|--------|----------------|
| 1 | RPCA_AINT8_WINT8_OR0.1 | **18.16** | 1.6777 | 26.66 | FID↓↓ IS↓ |
| 2 | RPCA_AINT8_WINT8_OR0.05 | **22.96** | 1.7300 | 25.79 | FID↓↓ IS≈ |
| 3 | RPCA_AINT8_WINT8_OR0 | **27.09** | 1.7315 | 24.81 | FID↓↓ IS≈ |
| 4 | RPCA_AINT8_WINT4_OR0.1 | **80.34** | 1.7081 | 18.46 | FID↓ IS≈ |
| 5 | RPCA_AINT8_WINT4_OR0.05 | **97.14** | **1.7951** | 17.47 | **✓ FID↓ IS↑** |
| 6 | RPCA_AINT4_WINT4_OR0.05 | **111.47** | 1.7218 | 16.45 | FID↓ IS≈ |
| 7 | **TUCKER_RR1p00_COREINT3** | **115.90** | **3.4542** | 15.25 | FID↓ IS↑↑ |
| 8 | RPCA_ANVFP4_WNVFP4_OR0.1 | **119.41** | **1.7617** | 17.08 | **✓ FID↓ IS↑** |
| 9 | RPCA_AINT4_WINT4_OR0.1 | **120.62** | 1.7187 | 17.30 | FID↓ IS≈ |
| 10 | cascade_hadamard | **132.10** | **1.7562** | 16.77 | **✓ FID↓ IS↑** |
| — | **BASELINE** | **161.30** | **1.7318** | 15.69 | 기준 |

---

## 전체 Best Results 순위 (IS 기준, baseline IS↑ 달성한 것들)

| 순위 | 실험 | FID ↓ | IS ↑ | 비고 |
|------|------|-------|------|------|
| 1 | TUCKER_INV_RR1p00_UINT4 | 173.88 | **3.5759** | FID 나쁨 |
| 2 | TUCKER_RR1p00_COREINT3 | 115.90 | **3.4542** | FID도 baseline 이하 |
| 3 | RPCA_NVFP4_KD50 | 159.46 | **1.8265** | **FID+IS 모두 ↑** |
| 4 | TSAWARE_G3_NOCORR | 174.30 | **1.8073** | FID 나쁨 |
| 5 | RPCA_AINT8_WINT4_OR0.05 | 97.14 | **1.7951** | **FID+IS 모두 ↑** |
| 6 | RPCA_AINT4_WINT4_OR0.01 | 163.48 | **1.7993** | FID≈ baseline |

---

## 핵심 결론 및 인사이트

### 성공적인 접근
1. **RPCA + 적절한 outlier_ratio (0.05~0.1)**: FID 기준으로 baseline 대비 가장 큰 개선. Outlier를 FP16 sparse branch로 격리하여 양자화 오차를 효과적으로 제어.
2. **RPCA + KD50**: IS 기준 최고. KD는 step 수 조절이 핵심 (50이 sweet spot, 300+부터 악화).
3. **Cascade-3 + Hadamard**: FID+IS 모두 baseline 초과. Layer별 민감도 기반 혼합 precision이 유효.

### 실패한 접근
- **Tucker 분해**: IS는 매우 높으나 FID 제어 불가. 이미지 분포 왜곡이 심함.
- **CP 분해, KSVD**: 모두 붕괴 수준의 품질 저하.
- **Timestep-Aware (scale만 변경, 또는 mixed precision)**: IS 소폭 개선에 그침. FID 개선 없음. Activation precision 변화가 근본적인 weight 양자화 오차를 해결하지 못함.
- **과도한 KD (300+)**: 오히려 FID 악화.

### 실험 전반의 교훈
- **FID vs IS trade-off**: 두 지표가 항상 함께 개선되지 않음. Tucker처럼 IS는 극도로 높으나 FID가 나쁜 경우 존재.
- **Weight quantization이 핵심**: Activation precision보다 weight quantization 품질이 FID에 더 큰 영향.
- **Outlier 처리**: RPCA의 성공 원인. 소수의 outlier weight가 양자화 오차의 대부분을 차지.
- **20 samples FID/IS의 한계**: 노이즈가 크므로 절대값보다 상대적 경향성으로 해석 필요.

---

## 미해결 과제 및 다음 실험 방향

- [ ] RPCA_ANVFP4_WNVFP4_OR0.1 → 100+ samples로 재검증 (FID+IS 모두 ↑ 가장 유망한 config)
- [ ] RPCA + KD50 + OR=0.05~0.1 조합 (RPCA OR 최적화 + KD 결합)
- [ ] Cascade-3 + Hadamard에 KD 적용
- [ ] Layer-wise sensitivity 기반 mixed precision (sensitivity 실험 완성)
- [ ] 실제 하드웨어 배포 (TensorRT/ModelOpt backend) 속도 측정
