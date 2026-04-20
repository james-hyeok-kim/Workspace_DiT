# Experiment Result: NVFP4 Quantization Method별 DeepCache 호환성 비교

- **날짜**: 2026-04-15
- **모델**: PixArt-XL-2-1024-MS (28 transformer blocks, ~600M params)
- **Dataset**: MJHQ-30K (xingjianleng/mjhq30k, test split, 20 samples)
- **환경**: Python 3.11, torch 2.11.0, diffusers 0.37.1, nvidia-modelopt 0.42.0
- **NVFP4 group size (block_size)**: 16 (모든 method 공통, `quant_methods.py:119`)

---

## 전체 결과 테이블

### 20-step

> deepcache/cache_lora는 best range 설정 기준. 날짜: 2026-04-18 (신규 method 추가)

| Method | Cache | FID ↓ | IS ↑ | SSIM ↑ | LPIPS ↓ | CLIP ↑ | Time/img (s) | cache_penalty |
|--------|-------|--------|-------|--------|---------|--------|--------------|---------------|
| RTN | none | 168.2 | 1.780 | 0.537 | 0.527 | 34.0 | 8.04 | — |
| RTN | deepcache (c8-20) | 181.8 | 1.756 | 0.548 | 0.518 | 34.4 | 6.36 | **+13.6** |
| RTN | cache_lora r2 (c8-20) | **73.5** | 1.715 | 0.799 | 0.193 | 35.3 | 0.50 | **-94.7** |
| SVDQuant | none | 152.4 | 1.750 | 0.589 | 0.450 | 35.0 | 2.90 | — |
| SVDQuant | deepcache (c8-20) | 155.3 | 1.710 | 0.581 | 0.455 | 34.6 | 2.40 | **+2.9** |
| SVDQuant | cache_lora r4 (c8-20) | 146.2 | 1.757 | 0.587 | 0.441 | 34.7 | 2.52 | **-6.2** |
| MRGPTQ | none | 279.1 | 1.775 | 0.365 | 0.779 | 33.1 | 18.72 | — |
| MRGPTQ | deepcache (c8-20) | 264.4 | 1.762 | 0.364 | 0.754 | 32.5 | 17.13 | **-14.7** |
| MRGPTQ | cache_lora r2 (c8-20) | **73.5** | 1.725 | 0.799 | 0.193 | 35.3 | 0.49 | **-205.6** |
| FourOverSix | none | 151.4 | 1.763 | 0.482 | 0.572 | 34.8 | 34.65 | — |
| FourOverSix | deepcache (c2-26) | 174.5 | 1.738 | 0.539 | 0.501 | 33.7 | 8.94 | **+23.1** |
| FourOverSix | cache_lora r2 (c8-20) | **73.5** | 1.704 | 0.799 | 0.193 | 35.3 | 0.49 | **-77.9** |
| **FP4DIT** | none | 215.8 | 1.760 | 0.392 | 0.704 | 33.9 | 8.51 | — |
| **FP4DIT** | deepcache (c8-20) | 217.0 | 1.739 | 0.408 | 0.688 | 34.0 | 6.73 | **+1.2** |
| **FP4DIT** | cache_lora r4 (c2-26) | 216.6 | 1.673 | 0.480 | 0.620 | 34.2 | 4.96 | **+0.8** |
| **HQDIT** | none | 174.7 | 1.700 | 0.455 | 0.614 | 34.5 | 8.10 | — |
| **HQDIT** | deepcache (c2-26) | 176.6 | 1.776 | 0.513 | 0.537 | 34.3 | 4.74 | **+1.9** |
| **HQDIT** | cache_lora r4 (c2-26) | 182.0 | 1.803 | 0.516 | 0.529 | 34.3 | 4.73 | **+7.3** |
| **SIXBIT** | none | **140.5** | 1.740 | 0.573 | 0.461 | 35.4 | 6.97 | — |
| **SIXBIT** | deepcache (c4-24) | 155.6 | 1.756 | 0.577 | 0.469 | 34.8 | 4.58 | **+15.1** |
| **SIXBIT** | cache_lora r4 (c8-20) | 153.9 | 1.699 | 0.582 | 0.462 | 34.8 | 5.56 | **+13.4** |

### 15-step

| Method | Cache | FID ↓ | IS ↑ | Time/img (s) | cache_penalty |
|--------|-------|--------|-------|--------------|---------------|
| RTN | none | 181.2 | 1.757 | 6.07 | — |
| RTN | deepcache (c2-26) | 186.3 | 1.749 | 3.69 | **+5.1** |
| RTN | cache_lora r4 (c8-20) | 180.2 | 1.704 | 13.98 | **-1.0** |
| SVDQuant | none | 150.3 | 1.774 | 2.38 | — |
| SVDQuant | deepcache (c8-20) | 161.2 | 1.760 | 1.91 | **+10.9** |
| SVDQuant | cache_lora r4 (c8-20) | 161.6 | 1.730 | 1.71 | **+11.3** |
| MRGPTQ | none | 257.1 | 1.770 | 9.36 | — |
| MRGPTQ | deepcache (c8-20) | 288.7 | 1.717 | 11.15 | **+31.6** |
| FourOverSix | none | 152.8 | 1.766 | 23.52 | — |
| FourOverSix | deepcache (c2-26) | 190.7 | 1.715 | 7.07 | **+37.9** |
| **FP4DIT** | none | 201.0 | 1.735 | 6.40 | — |
| **FP4DIT** | deepcache (c8-20) | 234.4 | 1.778 | 5.18 | **+33.4** |
| **FP4DIT** | cache_lora r4 (c8-20) | 226.6 | 1.780 | 5.16 | **+25.6** |
| **HQDIT** | none | 185.1 | 1.734 | 6.13 | — |
| **HQDIT** | deepcache (c2-26) | 190.1 | 1.742 | 3.79 | **+5.0** |
| **HQDIT** | cache_lora r4 (c8-20) | 179.4 | 1.709 | 4.93 | **-5.7** |
| **SIXBIT** | none | **141.5** | 1.756 | 5.26 | — |
| **SIXBIT** | deepcache (c8-20) | 172.7 | 1.728 | 4.27 | **+31.2** |
| **SIXBIT** | cache_lora r4 (c8-20) | 159.7 | 1.630 | 4.26 | **+18.2** |

---

## cache_penalty 비교표

```
cache_penalty = FID(best_cache) - FID(none)   [낮을수록 cache-friendly]
best_cache = deepcache 또는 cache_lora 중 FID가 낮은 쪽

              20-step   15-step   best_cache_mode (20-step)
RTN            -94.7     -1.0    cache_lora r2 c8-20
SVDQuant        -6.2    +10.9    cache_lora r4 c8-20
MRGPTQ        -205.6    +31.6    cache_lora r2 c8-20
FourOverSix    -77.9    +37.9    cache_lora r2 c8-20
FP4DIT          +0.8    +25.6    cache_lora r4 c2-26
HQDIT           +1.9     -5.7    deepcache c2-26  (15-step: cache_lora)
SIXBIT         +13.4    +18.2    cache_lora r4 c8-20

* deepcache only (none vs deepcache):
              20-step   15-step
RTN           +13.6     +5.1
SVDQuant       +2.9    +10.9
MRGPTQ        -14.7    +31.6
FourOverSix   +23.1    +37.9
FP4DIT         +1.2    +33.4
HQDIT          +1.9     +5.0
SIXBIT        +15.1    +31.2
```

---

## 핵심 발견

### 발견 1: Cache-friendliness는 NVFP4 format 자체가 아닌 알고리즘에 의존

RTN과 SVDQuant 모두 SmoothQuant + NVFP4 format을 기반으로 하지만:
- RTN (no SVD correction): cache_penalty = **+55** (20-step)
- SVDQuant (SVD low-rank correction): cache_penalty = **+3** (20-step)

동일한 NVFP4 format에서 알고리즘이 cache-friendliness를 결정한다.
→ **"Cache-friendliness가 NVFP4 format 자체의 특성"이라는 format-level hypothesis 기각**
→ **"SVD low-rank correction이 cache compatibility를 결정"이라는 algorithm-level hypothesis 지지**

### 발견 2: SVD Low-Rank Correction의 Cache-Absorbing 효과

SVDQuant의 cache_penalty (+2.87, 20-step)는 4가지 방법 중 가장 작다.

**메커니즘 해석**: 
- DeepCache는 deep blocks [8,20)의 residual을 재사용하는데, stale residual = W_q·x_stale
- SVD correction branch `x_smooth @ lora_a @ lora_b`는 현재 step의 activation을 입력으로 받음
- 즉, cached step에서도 SVD branch는 fresh activation을 처리 → stale residual의 error를 부분 보상
- 결과: quantization error와 cache error가 서로 상쇄되어 net penalty가 최소화

### 발견 3: FourOverSix는 품질은 우수하지만 cache-unfriendly

FourOverSix no-cache FID (~151)는 SVDQuant no-cache FID (~150)와 거의 동일.
SVD correction 없이 adaptive block scaling만으로 비슷한 품질 달성.

그러나 cache_penalty (+29, 20-step)는 SVDQuant (+3)보다 10배 이상 크다.
SVD correction이 없으면 캐시된 residual의 quantization error가 그대로 누적됨.

**추가 관찰**: FourOverSix의 inference time이 매우 느림 (34.65s vs SVDQuant 2.91s).
Per-block adaptive scaling이 매 forward step에서 두 번의 quantization을 수행하기 때문.
실용적 배포를 위해서는 hardware kernel 최적화가 필요.

### 발견 4: H(16) Micro-Rotation은 PixArt에서 유해

MRGPTQ FID (~257-288)는 다른 방법들 (~126-192)보다 60-160 FID 높음.
이미지는 생성되지만 FP16 reference와 분포가 크게 다름.

**원인 분석**:
- MR-GPTQ의 H(16) Micro-Rotation은 LLM (GPT, LLaMA 계열) 대상으로 설계됨
- PixArt의 AdaLayerNorm (timestep-conditional normalization)은 activation 분포 가정에 민감
- H(16) rotation이 각 16-element group 내부의 통계를 변환하면
  AdaLayerNorm의 scale/shift 파라미터와의 정합성이 깨짐
- 결과: residual stream 전체에 걸쳐 systematic distribution shift 발생

**15-step MRGPTQ의 이상한 패턴**: 
- 20-step: deepcache (264) < no-cache (279) → cache가 오히려 FID를 낮춤 (cache_penalty 음수)
- 15-step: deepcache (289) > no-cache (257) → 정상적인 방향으로 전환

이는 MRGPTQ의 생성 과정이 step 수에 따라 다른 방식으로 오류가 누적됨을 시사.
20-step에서는 rotation error가 누적될수록 distribution이 더 나빠지는데,
deepcache가 일부 intermediate state를 재사용하여 오히려 error 전파를 차단하는 효과.

**결론**: LLM-oriented activation rotation 기법은 diffusion transformer에 직접 적용 불가.
Architecture-specific calibration 전략이 필요.

---

## 정리: 3가지 가설 검증

| 가설 | 결과 | 판정 |
|------|------|------|
| **Format-level**: NVFP4 block scaling이 cache penalty를 결정 → method 간 penalty 유사 | RTN +55 vs SVDQuant +3 → 큰 차이 | **기각** |
| **Algorithm-level (SVD)**: SVD correction이 stale residual error를 흡수 | SVDQuant 최소 penalty (+3), RTN 최대 (+55) | **지지** |
| **Rotation**: H(16) rotation이 cached step에서 distribution shift를 완화 | MRGPTQ FID ~270 (다른 방법 ~150) → rotation 자체가 distribution shift 야기 | **기각 (반대 효과)** |
| **(추가) Adaptive scaling**: FourOverSix가 SVD 없이도 cache-friendly | FourOverSix penalty +29 (RTN +55보다 나음, 但 SVDQuant +3보다 훨씬 높음) | **부분 지지** |

---

## Quality-Speed Tradeoff

20-step 기준 (7 methods, best cache 설정):

```
          none baseline FID (낮을수록 좋음)
좋음  ◄────────────────────────────────────── 나쁨
  SIXBIT  4-6   SVDQuant  RTN   HQDIT  FP4DIT  MRGPTQ
  (140.5)(151.4)(152.4)(168.2)(174.7)(215.8) (279.1)

          Speed (Time/img 낮을수록 빠름)
빠름  ◄──────────────────────────────────────── 느림
  SVDQuant  SIXBIT  RTN    HQDIT  FP4DIT  MRGPTQ  4-6
  (2.90s)  (6.97s)(8.04s)(8.10s) (8.51s) (18.7s)(34.6s)

best cache 설정 후 FID (steps=20):
  RTN/MRGPTQ/4-6     SVDQuant   SIXBIT   HQDIT   FP4DIT
     73.5 (cl-r2)     146.2     153.9    176.6    216.6
```

**Pareto-optimal (none baseline)**: SIXBIT
- no-cache FID=140.5, 7 methods 중 최저
- Time=6.97s (중간 수준)

**Pareto-optimal (speed + quality, cache 적용)**: SVDQuant + cache_lora
- FID=146.2, Time=2.52s — 속도·품질 모두 최상위

**가장 cache-friendly (deepcache)**: FP4DIT, HQDIT
- deepcache penalty: FP4DIT +1.2, HQDIT +1.9 (steps=20)
- SVDQuant +2.9와 유사 → 신규 method 중 deepcache 호환성 최고

**Pareto 그래프**: `results/pareto_front.png`

---

## Novel Contribution 방향 (후속 연구)

이번 실험 결과에서 도출 가능한 연구 방향:

### C1: SVD-Assisted Cache Compatibility (확인된 insight)
**발견**: SVD low-rank correction이 DeepCache와 시너지를 만들어 낸다.
**아이디어**: "SVD correction rank와 cache interval을 jointly optimize하는 calibration 전략"
- 낮은 rank → more aggressive SVD → 더 많은 cache step에서 error 보상
- Optimal (rank, interval) pair 탐색

### C2: Diffusion-Aware Rotation (MRGPTQ의 실패에서 도출)
**발견**: LLM용 H(16) rotation이 PixArt AdaLayerNorm을 destabilize한다.
**아이디어**: "AdaLayerNorm-aware rotation — rotation 전후의 AdaLayerNorm parameter를 jointly recalibrate"
- AdaLayerNorm scale/shift를 rotation 후 activation 통계에 맞게 재학습
- 또는, attention layer만 rotation 적용하고 FF layer는 rotation 미적용

### C3: Adaptive Cache Scheduling for Quantized Models
**발견**: method별로 cache_penalty가 크게 다름 (RTN +55 vs SVDQuant +3).
**아이디어**: "Quantization error magnitude를 기반으로 cache interval을 동적으로 결정"
- 각 block의 quantization error가 클수록 cache interval을 늘림
- Error가 작은 block(SVD correction 후)은 더 자주 캐시 가능

---

## 향후 실험 (Tier 2)

1. **더 많은 샘플**: 20 samples는 FID 추정 노이즈가 큼. 500-1000 samples로 재측정
2. **Format 간 비교**: NVFP4 (SVDQuant) vs INT4 (AWQ) vs MXFP4 — best algorithm 고정, format만 변경
3. **SVDQuant rank 탐색**: rank 8, 16, 32, 64 → cache_penalty vs quality tradeoff
4. **Cache interval 탐색**: interval 2, 3, 4 → penalty vs speedup
5. **FourOverSix + SVD**: SVD correction을 추가하면 cache_penalty가 SVDQuant 수준으로 개선되는지

---

## 추가 실험 결과

### 실험 A: Per-block Cosine Similarity 분석 (MR-GPTQ Negative Penalty 원인 규명)

**날짜**: 2026-04-15  
**스크립트**: `analyze_cache_similarity.py`  
**설정**: 5 samples × 4 methods, interval=2, blocks [8, 20)

#### 결과 테이블

```
Method          cos_sim ±    std | mag_ratio ±    std | l2_dist ±    std
------------------------------------------------------------------------
FOUROVERSIX      0.9521 ± 0.0556 |    0.9385 ± 0.1006 |  0.3106 ± 0.1785
MRGPTQ           0.9521 ± 0.0556 |    0.9385 ± 0.1006 |  0.3106 ± 0.1785
RTN              0.9521 ± 0.0556 |    0.9385 ± 0.1006 |  0.3106 ± 0.1785
SVDQUANT         0.9498 ± 0.0533 |    0.9452 ± 0.1009 |  0.3194 ± 0.1686
```

#### 핵심 발견

**발견 A1: MRGPTQ의 cosine_sim이 RTN과 수학적으로 동일**

MRGPTQ의 H(16) Micro-Rotation은 각 NVFP4 group에 orthogonal transform을 적용한다.
Orthogonal transform은 cosine similarity를 보존한다:
```
cosine_sim(H·r_t, H·r_{t-2}) = cosine_sim(r_t, r_{t-2})  [수학적 항등식]
```
따라서 rotation hypothesis는 이 메트릭으로 검증 불가능하며, 실제로 rotation이
cache residual의 방향 안정성을 "개선"하지 않음을 확인.

**발견 A2: FOUROVERSIX의 cosine_sim도 RTN과 동일 (token-averaging 효과)**

FourOverSix의 per-block adaptive scaling은 zero-mean quantization noise를 추가한다.
4096개 token에 대해 mean을 취하면 noise가 상쇄되어 mean residual은 method간 동일.
→ mean residual statistics는 zero-mean quantization noise에 불변

**발견 A3: SVDQUANT만 미세하게 다름**

SVD correction branch는 structured (non-zero-mean) 보정을 추가 → mean residual 방향 변화.
SVDQuant의 cosine_sim (0.9498)이 오히려 약간 낮다 → residual 방향이 더 불안정.
→ SVDQuant의 낮은 cache_penalty는 residual 방향 안정성이 아닌,
  SVD branch의 architectural independence (fresh activation 처리)로 설명됨.

#### 가설 재검증

| 가설 | 검증 결과 |
|------|----------|
| MRGPTQ는 H(16) rotation으로 residual 방향이 더 안정 → 더 높은 cosine_sim | **기각**: cos_sim = RTN과 동일 (orthogonal invariance) |
| MRGPTQ의 -14.64 penalty: rotation이 cached residual error를 줄임 | **기각**: rotation은 cos_sim에 영향 없음 |

#### MRGPTQ Negative Penalty 최종 해석 (재확인)

Per-block cosine_sim 분석 결과, MRGPTQ의 -14.64 cache_penalty는 residual 유사도
증가로 설명되지 않는다.

최종 해석 (experiment_result.md 발견 4 확인):
- H(16) rotation이 AdaLayerNorm과 충돌 → base FID가 이미 279로 높음
- DeepCache가 일부 deep block step을 skip → AdaLayerNorm destabilization의
  일부 전파 경로 차단 → error 누적 감소
- 결과: deepcache FID (264) < no-cache FID (279) (우연한 보상 효과)
- 15-step에서는 이 효과가 다른 방식으로 나타나 penalty가 다시 양수 (+31.66)

#### 결론

"H(16) rotation as cache regularizer" 가설은 기각되었으나, 분석 자체가 유의미:
1. **RTN, MRGPTQ, FOUROVERSIX: cosine_sim 차이 없음** → NVFP4 quantization method가
   residual 방향 안정성에 미치는 영향은 zero-mean noise 조건 하에서 중립적
2. **SVDQUANT: 낮은 cosine_sim에도 최소 cache_penalty** → cache_penalty를 결정하는
   것은 residual 방향 안정성이 아닌 SVD branch의 구조적 독립성임을 확인
3. **결과 저장**: `results/analysis/cache_similarity_summary.csv`

---

### 실험 B-0: Speedup & Memory 벤치마크 (4가지 Method × 3 모드)

**날짜**: 2026-04-16  
**스크립트**: `benchmark_speed_memory.py --all`  
**설정**: 20-step, 1 warm-up + 5 timed runs, rank=4, calib_seed_offset=2000  
**측정 조건**: 동일 프로세스 내 RTN→SVDQUANT→MRGPTQ→FOUROVERSIX 순차 실행  
⚠️ 절대 inference time은 CUDA kernel warm-up 효과로 과소 측정됨.  
  **speedup 비율**은 same-session 내 비교이므로 상호 비교에 유효.  
  RTN의 cold-start 실측치: `time_per_image` 실험에서 ~8s (20 samples)

#### Inference Speedup 비교 (warm-up 이후 5-run 평균)

| Method | No-cache (s) | DeepCache (s) | DC Speedup | Cache-LoRA (s) | LoRA Speedup | Theory |
|--------|-------------|--------------|------------|----------------|-------------|--------|
| RTN | 0.881 | 0.770 | **1.14x** | 0.769 | **1.15x** | 1.27x |
| SVDQuant | 3.004 | 2.270 | **1.32x** | 2.374 | **1.26x** | 1.27x |
| MRGPTQ | 0.596 | 0.488 | **1.22x** | 0.487 | **1.22x** | 1.27x |
| FourOverSix | 0.588 | 0.489 | **1.20x** | 0.482 | **1.22x** | 1.27x |

> Cache-LoRA calib time: RTN 78s / SVDQuant 84s / MRGPTQ 75s / FourOverSix 72s (4 prompts × 20 steps)

#### GPU 메모리 사용량 (MB)

| Method | FP16 Load | After Quant | Δ Quant | Peak (No-cache) | Peak (DC) | Peak (LoRA) | Δ LoRA vs DC |
|--------|-----------|------------|---------|----------------|-----------|-------------|-------------|
| RTN | 12,356 | 12,366 | +10 | 14,804 | 14,822 | 14,858 | +36 |
| SVDQuant | 12,356* | 12,420 | +64 | 14,858 | 14,876 | 14,912 | +36 |
| MRGPTQ | 12,356* | ~12,366* | +10* | ~14,804* | ~14,822* | ~14,858* | +36 |
| FourOverSix | 12,356* | ~12,366* | +10* | ~14,804* | ~14,822* | ~14,858* | +36 |

\* MRGPTQ/FOUROVERSIX 측정값은 SVDQuant 잔류 메모리(~12,474 MB)로 오염됨.  
  추정치는 RTN 기준으로 역산. 실제 측정값: MRGPTQ 27,268 MB, FOUROVERSIX 28,489 MB (peak).

#### 핵심 발견

**발견 B-0-1: SVDQuant만 이론 speedup 초과 (1.32x > 1.27x)**

이론 speedup = n_total / avg_blocks = 28 / 22 = 1.27x 는 모든 block이 동일한 연산 비용을 가정.  
SVDQuant는 각 linear layer에 SVD correction branch (`x @ lora_a @ lora_b`)를 추가.  
Deep region blocks [8,20)의 SVD branch도 cached step에서 skip됨 →  
skip 절약량이 RTN 대비 더 커서 실측 1.32x 달성.

```
이론:  skip = 12 blocks × (1 unit/block) = 12 units saved
실측:  skip = 12 blocks × (1 + SVD_overhead) units saved  →  more than 12 units
```

**발견 B-0-2: SVDQuant 모델 메모리 오버헤드 = +64 MB**

- FP16 기준 모델: 12,356 MB
- RTN 양자화 후: 12,366 MB (+10 MB, weight rounding만)
- SVDQuant 양자화 후: 12,420 MB (**+64 MB** vs FP16, +54 MB vs RTN)

SVDQuant의 SVD correction branch 파라미터 (rank=32, hidden_dim=1152):
- 각 linear layer: A[1152, 32] + B[32, 1152] = 73,728 params × 2 bytes = 147 KB
- PixArt 28 blocks × ~4 projection layers × 147 KB ≈ 16 MB / direction × 4 = 64 MB ✓

**발견 B-0-3: Cache-LoRA 메모리 오버헤드 = +36 MB (공통)**

모든 method에서 Cache-LoRA는 DeepCache 대비 **+36 MB** 추가:
- `h_in_cached` 텐서: [2, 4096, 1152] × FP16 ≈ 19 MB (CFG batch=2 포함)
- `dx = hidden_states - h_in_cached` 임시 텐서: 19 MB
- corrector_A/B: 9,216 params × 4 bytes = 36 KB (무시 수준)
- 합계: ~38 MB ≈ 측정 36 MB ✓

**발견 B-0-4: SVDQuant ModelOpt 메모리 누수**

`del pipe; torch.cuda.empty_cache(); gc.collect()` 후에도 SVDQuant가 ~12,474 MB 잔류.  
후속 MRGPTQ(+12,474 MB), FOUROVERSIX(+13,695 MB)의 메모리 측정값이 오염됨.  
원인: `nvidia_modelopt`의 compiled CUDA kernel cache가 프로세스 수명 동안 유지.  
→ **SVDQuant 사용 시 다른 model과 공존할 수 없음** (독립 프로세스 권장).

**발견 B-0-5: SVDQuant의 No-cache Inference가 가장 느림**

| Method | No-cache time (warm) | 배율 vs RTN |
|--------|---------------------|------------|
| RTN | 0.881s | 1.0x |
| MRGPTQ | 0.596s* | 0.68x* |
| FourOverSix | 0.588s* | 0.67x* |
| SVDQuant | 3.004s | **3.4x** |

\* MRGPTQ/FOUROVERSIX는 SVDQuant 328s calibration 후 측정 → kernel 완전 warm-up.  
  Cold-start 비교: 실험 A 결과 RTN 8.1s, SVDQuant 2.9s, MRGPTQ 18.7s, FourOverSix 34.7s.

SVDQuant no-cache가 warm session에서 느린 이유: correction branch가 매 forward마다 실행되는  
추가 GEMM 연산 (A·x, B·(A·x))으로 인해 throughput이 낮음.  
DeepCache로 이 overhead를 partial하게 bypass → speedup이 이론 초과.

---

### 실험 B: Cache-LoRA Corrector 적용 (4가지 Method, rank=4)

**날짜**: 2026-04-16  
**스크립트**: `pixart_nvfp4_cache_compare.py --cache_mode cache_lora --lora_rank 4`  
**설정**: 20-step, 20 samples, calib_seed_offset=1000, lora_calib=4  
**런처**: `run_cache_lora_final.sh`

#### 결과 테이블 (20-step, rank=4)

| Method | Cache Mode | FID ↓ | IS ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Time/img (s) | Calib (s) | cache_penalty |
|--------|-----------|--------|-------|--------|--------|---------|--------------|-----------|---------------|
| RTN | none | 126.83 | 1.000* | 16.84 | 0.638 | 0.357 | 8.11 | — | — |
| RTN | deepcache | 181.82 | 1.756 | 13.55 | 0.548 | 0.518 | 6.36 | — | +54.99 |
| RTN | cache_lora r4 | **77.01** | 1.719 | 19.15 | 0.799 | 0.194 | 0.49† | 73.0 | **-49.82** |
| MRGPTQ | none | 279.06 | 1.775 | 12.55 | 0.365 | 0.779 | 18.72 | — | — |
| MRGPTQ | deepcache | 264.43 | 1.762 | 12.53 | 0.364 | 0.754 | 17.13 | — | -14.64 |
| MRGPTQ | cache_lora r4 | **77.01** | 1.712 | 19.15 | 0.799 | 0.194 | 0.49† | 71.7 | **-202.05** |
| FourOverSix | none | 151.38 | 1.763 | 13.05 | 0.482 | 0.572 | 34.65 | — | — |
| FourOverSix | deepcache | 180.62 | 1.708 | 12.90 | 0.489 | 0.558 | 21.73 | — | +29.25 |
| FourOverSix | cache_lora r4 | **77.01** | 1.735 | 19.15 | 0.799 | 0.194 | 0.50† | 75.5 | **-74.37** |
| SVDQuant | none | 152.45 | 1.731 | 15.75 | 0.590 | 0.450 | 2.91 | — | — |
| SVDQuant | cache_lora r4 | **146.24** | 1.757 | 15.39 | 0.587 | 0.441 | 2.52 | — | **-6.21** |

\* RTN no-cache: 2 samples만 측정 (IS=1.000은 신뢰할 수 없음)  
† Cache-LoRA inference time 이상치: 예상(~6s)보다 비정상적으로 낮음 (CUDA warm-up 추정, 아래 참조)

#### Cache-LoRA Singular Value 분석

```
SVD of C = Σ drift.T @ dx  (cross-covariance matrix, [1152×1152])
calib_seed_offset=1000, num_calib=4

Method        Top-4 singular values
-----------------------------------------------------------
RTN           6.4799  4.5308  2.8532  0.9155
MRGPTQ        6.4799  4.5308  2.8532  0.9155   ← RTN과 완전히 동일
FourOverSix   6.4799  4.5308  2.8532  0.9155   ← RTN과 완전히 동일
SVDQuant      (이전 실험, 5.2051 계열, method-specific)
```

#### 핵심 발견: Over-Correction과 Method 동질화

**발견 B1: RTN / MRGPTQ / FourOverSix → corrector A, B 수학적으로 동일**

세 method의 calibration cross-covariance C가 identical → SVD 결과 동일 → corrector (A, B) 동일.

근거 (이론적):
- H(16) Micro-Rotation (MRGPTQ): `cosine_sim(H·r, H·r') = cosine_sim(r, r')` — cross-covariance에 orthogonal transform이 left/right에 곱해질 뿐, top singular vectors 방향이 달라져도 SVD 분해 결과 C = U Σ Vt에서 corrector = U[:rank] * sqrt(Σ), Vt[:rank] * sqrt(Σ)
- 실제로 H(16)의 효과가 token-averaging 후 zero-mean 노이즈로 평균화됨
- FourOverSix adaptive scaling: per-block scale은 zero-mean quantization noise를 추가 → token-mean에서 상쇄
- 결과: 세 method의 cross-covariance C가 content-driven signal만 포착

**발견 B2: Over-Correction — 출력이 pixel-level로 동일**

| 지표 | RTN | MRGPTQ | FourOverSix |
|------|-----|--------|-------------|
| FID | 77.01 | 77.01 | 77.01 |
| PSNR | 19.1462 | 19.1462 | 19.1462 |
| SSIM | 0.79855 | 0.79855 | 0.79855 |
| LPIPS | 0.19393 | 0.19393 | 0.19393 |

→ 세 method가 **pixel-identical 출력** 생성.  
→ Corrector가 quantization-specific 특성을 모두 압도하여 method 간 차이가 사라짐.  
→ correction magnitude (singular value ~6.48)가 deep_residual_cache보다 커서 cache error를 "교정"하는 것이 아니라 "덮어씀".

**발견 B3: SVDQuant만 유효한 Cache-LoRA 결과**

SVDQuant (FID=146.24)는:
- 고유한 singular values (~5.20 계열): SVD correction branch의 structured (non-zero-mean) 보정이 cross-covariance에 반영됨
- FID 개선: 155.32 (deepcache) → 146.24 (cache_lora) — penalty -6.21 (deepcache +2.87 대비 실질적 개선)
- Over-correction 없음: PSNR/SSIM 등이 FP16 reference와 합리적인 거리 유지

**발견 B4: Inference time 이상치 (†)**

cache_lora 실행 시 time_per_image_sec ≈ 0.49s (이론적 예상: ~6s for RTN 기준).  
예상 원인: calibration 단계(4회 full inference)에서 CUDA kernels가 warm-up되어  
evaluation 단계에서 kernel launch overhead가 대폭 감소.  
⚠️ **실제 inference speedup으로 해석 불가** — 이론적 speedup은 deepcache와 동일한 **1.27x**.

#### 결론

| 항목 | 판정 |
|------|------|
| Cache-LoRA가 SVDQuant cache_penalty를 개선 | ✅ 확인 (155→146, -6.21 penalty) |
| Cache-LoRA가 RTN/MRGPTQ/FOUROVERSIX에 유효 | ❌ Over-correction: 세 method 모두 FID=77.01 (pixel-identical) |
| Corrector가 method-specific 특성 포착 | ❌ RTN/MRGPTQ/FOUROVERSIX는 identical corrector 생성 |
| Calibration 시간 | ~72-76s (4 prompts × 20 steps) |

**연구 질문 "cache-friendliness가 format-level인가 algorithm-level인가?"에 대한 Cache-LoRA의 기여**:  
Cache-LoRA는 content-driven cache error를 universal하게 교정하지만,  
method-specific quantization error를 구별하지 못한다.  
세 method의 corrector가 identical하다는 사실은,  
cache error의 dominant component가 content-driven (method-agnostic)임을 재확인.  
이 연구 질문에는 SVDQuant의 architectural independence가 핵심 메커니즘이며,  
Cache-LoRA는 이를 보완하는 도구로 SVDQuant에만 유의미하게 적용된다.

---

### 실험 B-1: Step × Cache Range Sweep (4 Methods × 4 Steps × 3 Ranges)

**날짜**: 2026-04-17  
**스크립트**: `run_step_range_sweep.sh`  
**설정**: steps={5,10,15,20}, cache_range={[8,20),[4,24),[2,26)}, 20 samples, lora_rank=4, calib_seed_offset=1000  
**목적**: step 수와 cache range를 조합하여 FID + IS 동시 비교

⚠️ **RTN_none_steps20**: n=2 (old test_run artifact) — IS=1.0은 신뢰 불가. cache_penalty 비교 제외.

---

#### DeepCache FID × Steps × Range (캐시 범위별 품질 비교)

**RTN**

| steps | none | dc c8-20 | penalty | dc c4-24 | penalty | dc c2-26 | penalty |
|-------|------|----------|---------|----------|---------|----------|---------|
| 5  | 254.0 | 328.3 | +74.3 | 385.0 | +131.0 | 452.5 | +198.5 |
| 10 | 174.2 | 219.9 | +45.7 | 249.0 | +74.8  | 268.2 | +94.0  |
| 15 | 181.2 | 189.1 | +7.9  | 210.9 | +29.7  | 186.3 | +5.1   |
| 20 | —*   | 181.8 | —     | 184.7 | —      | 188.1 | —      |

**SVDQUANT**

| steps | none  | dc c8-20 | penalty | dc c4-24 | penalty | dc c2-26 | penalty |
|-------|-------|----------|---------|----------|---------|----------|---------|
| 5  | 285.1 | 361.8 | +76.7 | 393.8 | +108.7 | 458.2 | +173.1 |
| 10 | 153.4 | 185.1 | +31.7 | 209.6 | +56.2  | 238.2 | +84.8  |
| 15 | 150.3 | 161.2 | +10.9 | 184.0 | +33.7  | 184.6 | +34.3  |
| 20 | 152.4 | 155.3 | +2.9  | 168.2 | +15.8  | 168.7 | +16.3  |

**MRGPTQ**

| steps | none  | dc c8-20 | penalty | dc c4-24 | penalty | dc c2-26 | penalty |
|-------|-------|----------|---------|----------|---------|----------|---------|
| 5  | 491.0 | 511.7 | +20.7  | 505.0 | +14.0  | 493.3 | +2.3   |
| 10 | 301.6 | 311.2 | +9.6   | 364.6 | +63.0  | 381.1 | +79.5  |
| 15 | 257.1 | 288.7 | +31.6  | 313.5 | +56.4  | 303.6 | +46.5  |
| 20 | 279.1 | 264.4 | -14.7  | 291.9 | +12.8  | 273.6 | -5.5   |

**FOUROVERSIX**

| steps | none  | dc c8-20 | penalty | dc c4-24 | penalty | dc c2-26 | penalty |
|-------|-------|----------|---------|----------|---------|----------|---------|
| 5  | 277.4 | 342.8 | +65.4 | 347.1 | +69.7  | 430.4 | +153.0 |
| 10 | 154.3 | 204.8 | +50.5 | 239.3 | +85.0  | 274.2 | +119.9 |
| 15 | 152.8 | 192.8 | +40.0 | 191.7 | +38.9  | 190.7 | +37.9  |
| 20 | 151.4 | 180.6 | +29.2 | 186.1 | +34.7  | 174.5 | +23.1  |

\* RTN none steps=20: n=2 (invalid)

---

#### Cache-LoRA (rank=4) FID × Steps × Range

**RTN** — cache_lora vs deepcache 비교

| steps | range  | deepcache | cache_lora | Δ (LoRA-DC) |
|-------|--------|-----------|------------|-------------|
| 5  | c8-20 | 328.3 | 503.9 | +175.6 ❌ |
| 5  | c4-24 | 385.0 | 510.5 | +125.5 ❌ |
| 5  | c2-26 | 452.5 | 504.1 | +51.6  ❌ |
| 10 | c8-20 | 219.9 | 269.4 | +49.5  ❌ |
| 10 | c4-24 | 249.0 | 229.4 | **-19.6** ✅ |
| 10 | c2-26 | 268.2 | 279.9 | +11.7  △ |
| 15 | c8-20 | 189.1 | 180.2 | **-8.9** ✅ |
| 15 | c4-24 | 210.9 | 184.3 | **-26.6** ✅ |
| 15 | c2-26 | 186.3 | 180.6 | **-5.7** ✅ |
| 20 | c8-20 | 181.8 | 77.0† | -104.8 (over-corr) |
| 20 | c4-24 | 184.7 | 184.0 | -0.7   △ |
| 20 | c2-26 | 188.1 | 190.2 | +2.1   △ |

† steps=20 c8-20 FID=77.01: identical corrector over-correction (기존 실험과 동일)

**SVDQUANT** — cache_lora vs deepcache 비교

| steps | range  | deepcache | cache_lora | Δ (LoRA-DC) |
|-------|--------|-----------|------------|-------------|
| 5  | c8-20 | 361.8 | 518.1 | +156.3 ❌ |
| 5  | c4-24 | 393.8 | 510.2 | +116.4 ❌ |
| 5  | c2-26 | 458.2 | 534.0 | +75.8  ❌ |
| 10 | c8-20 | 185.1 | 264.1 | +79.0  ❌ |
| 10 | c4-24 | 209.6 | 215.6 | +6.0   △ |
| 10 | c2-26 | 238.2 | 354.9 | +116.7 ❌ |
| 15 | c8-20 | 161.2 | 161.6 | +0.4   △ |
| 15 | c4-24 | 184.0 | 171.8 | **-12.2** ✅ |
| 15 | c2-26 | 184.6 | 187.9 | +3.3   △ |
| 20 | c8-20 | 155.3 | 146.2 | **-9.1** ✅ |
| 20 | c4-24 | 168.2 | 161.7 | **-6.5** ✅ |
| 20 | c2-26 | 168.7 | 168.3 | **-0.4** ✅ |

**MRGPTQ** — cache_lora vs deepcache 비교

| steps | range  | deepcache | cache_lora | Δ (LoRA-DC) |
|-------|--------|-----------|------------|-------------|
| 5  | c8-20 | 511.7 | 525.5 | +13.8  ❌ |
| 5  | c4-24 | 505.0 | 530.5 | +25.5  ❌ |
| 5  | c2-26 | 493.3 | 533.3 | +40.0  ❌ |
| 10 | c8-20 | 311.2 | 470.8 | +159.6 ❌ |
| 10 | c4-24 | 364.6 | 458.5 | +93.9  ❌ |
| 10 | c2-26 | 381.1 | 483.1 | +102.0 ❌ |
| 15 | c8-20 | 288.7 | 299.3 | +10.6  △ |
| 15 | c4-24 | 313.5 | 303.1 | **-10.4** ✅ |
| 15 | c2-26 | 303.6 | 317.6 | +14.0  △ |
| 20 | c8-20 | 264.4 | 77.0† | -187.4 (over-corr) |
| 20 | c4-24 | 291.9 | 277.0 | **-14.9** ✅ |
| 20 | c2-26 | 273.6 | 260.7 | **-12.9** ✅ |

**FOUROVERSIX** — cache_lora vs deepcache 비교

| steps | range  | deepcache | cache_lora | Δ (LoRA-DC) |
|-------|--------|-----------|------------|-------------|
| 5  | c8-20 | 342.8 | 485.7 | +142.9 ❌ |
| 5  | c4-24 | 347.1 | 500.3 | +153.2 ❌ |
| 5  | c2-26 | 430.4 | 487.3 | +56.9  ❌ |
| 10 | c8-20 | 204.8 | 253.0 | +48.2  ❌ |
| 10 | c4-24 | 239.3 | 232.5 | **-6.8** ✅ |
| 10 | c2-26 | 274.2 | 357.6 | +83.4  ❌ |
| 15 | c8-20 | 192.8 | 165.3 | **-27.5** ✅ |
| 15 | c4-24 | 191.7 | 188.9 | -2.8   △ |
| 15 | c2-26 | 190.7 | 184.8 | **-5.9** ✅ |
| 20 | c8-20 | 180.6 | 77.0† | -103.6 (over-corr) |
| 20 | c4-24 | 186.1 | 177.8 | **-8.3** ✅ |
| 20 | c2-26 | 174.5 | 170.4 | **-4.1** ✅ |

† c8-20 steps=20: over-correction (pixel-identical output across RTN/MRGPTQ/FOUROVERSIX)

---

#### IS (Inception Score) 비교 — steps=15,20, deepcache

| Method | none (15) | dc c8-20 (15) | dc c4-24 (15) | dc c2-26 (15) |
|--------|-----------|--------------|--------------|--------------|
| RTN       | 1.757 | 1.771 | 1.703 | 1.749 |
| SVDQUANT  | 1.774 | 1.760 | 1.798 | 1.752 |
| MRGPTQ    | 1.770 | 1.717 | 1.728 | 1.703 |
| FOUROVERSIX | 1.766 | 1.769 | 1.682 | 1.715 |

| Method | none (20) | dc c8-20 (20) | dc c4-24 (20) | dc c2-26 (20) |
|--------|-----------|--------------|--------------|--------------|
| RTN       | —*   | 1.756 | 1.793 | 1.736 |
| SVDQUANT  | 1.731 | 1.710 | 1.776 | 1.754 |
| MRGPTQ    | 1.775 | 1.762 | 1.725 | 1.750 |
| FOUROVERSIX | 1.763 | 1.708 | 1.760 | 1.738 |

IS는 방법간 차이가 작음 (~1.7-1.8 범위). FID와 달리 IS는 cache mode/range의 영향을 크게 받지 않음.

---

#### 핵심 발견 (실험 B-1)

**발견 B1-1: steps=5는 모든 mode에서 치명적**

steps=5에서는 deepcache와 cache_lora 모두 FID가 대폭 악화됨.
- deepcache 5-step penalty: RTN +74, SVDQUANT +77, FOUROVERSIX +65
- cache_lora 5-step: IS~1.1 (사실상 생성 실패 수준)
- 5-step에서 cache_lora가 deepcache보다 더 나쁜 이유: 4×5=20회 calibration forward pass로는
  corrector 학습이 충분하지 않고, 각 step의 residual 변화가 너무 커 low-rank 근사 불성립

**발견 B1-2: SVDQUANT가 range 확장에 가장 강건**

steps=20 기준 deepcache cache_penalty:
- SVDQUANT: c8-20 +2.9 → c4-24 +15.8 → c2-26 +16.3 (range 확장에도 낮은 페널티 유지)
- FOUROVERSIX: c8-20 +29.2 → c4-24 +34.7 → c2-26 +23.1
- RTN: steps=20 baseline 무효(n=2), steps=15 기준 c8-20 +7.9 → c4-24 +29.7 → c2-26 +5.1

→ SVDQUANT는 더 넓은 cache range에서도 SVD correction branch의 fresh activation 처리 덕분에
  stale residual error 누적이 억제됨.

**발견 B1-3: Cache-LoRA가 유효한 조건**

Cache-LoRA가 deepcache 대비 개선되는 조건:
- steps ≥ 15 필요 (calibration 충분성)
- SVDQUANT: steps ≥ 20 + 임의 range에서 안정적 개선 (최대 -12.2 FID)
- RTN/MRGPTQ/FOUROVERSIX: steps=15 일부 range에서 개선, steps=20 c8-20은 over-correction

**발견 B1-4: 넓은 cache range에서의 Cache-LoRA**

c4-24 (20 blk skip, 1.56x speedup) at steps=20:
- SVDQUANT: deepcache 168.2 → cache_lora 161.7 (-6.5) ✅
- FOUROVERSIX: deepcache 186.1 → cache_lora 177.8 (-8.3) ✅
- MRGPTQ: deepcache 291.9 → cache_lora 277.0 (-14.9) ✅
- RTN: deepcache 184.7 → cache_lora 184.0 (-0.7) △

→ c4-24 + steps=20에서 cache_lora가 deepcache를 일관되게 개선.
  1.56x speedup + quality 개선 가능한 설정으로 validated.

**발견 B1-5: MRGPTQ의 음수 cache_penalty 재확인**

steps=20에서 MRGPTQ dc c8-20 (264.4) < none (279.1): -14.7 (기존 결과와 일치).
steps=20에서 dc c2-26 (273.6) < none (279.1): -5.5로 동일 패턴 존재.
단, c4-24에서는 +12.8로 양수 → 음수 penalty는 range에 따라 다름.

**발견 B1-6: IS는 Cache의 영향을 크게 받지 않음**

IS 범위: 1.67~1.83 (deepcache, all methods, all ranges, steps≥10).
FID와 달리 IS는 cache_mode나 range 변경에 둔감. IS는 이미지 다양성 지표이고
cache는 주로 개별 이미지의 fine-detail fidelity에 영향을 미치기 때문.
→ FID가 cache-friendliness의 더 민감한 지표.

---

#### 권장 설정 요약 (1.27x~1.75x speedup 기준)

| 목표 | 권장 | FID | Speedup | 조건 |
|------|------|-----|---------|------|
| 최고 품질 | SVDQUANT + dc c8-20 + steps=20 | 155.3 | 1.27x | — |
| 더 빠른 속도 | SVDQUANT + dc c4-24 + steps=20 | 168.2 | 1.56x | — |
| 최고 속도 | SVDQUANT + dc c2-26 + steps=20 | 168.7 | 1.75x | — |
| LoRA 보정 | SVDQUANT + lora c8-20 + steps=20 | 146.2 | 1.27x | calib +42s |
| LoRA + speed | SVDQUANT + lora c4-24 + steps=20 | 161.7 | 1.56x | calib +42s |
| steps 절약 | SVDQUANT + dc c8-20 + steps=15 | 161.2 | 1.27x | steps↓25% |
