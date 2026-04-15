# Experiment: NVFP4 Quantization Method별 DeepCache 호환성 비교

- **날짜**: 2026-04-14
- **위치**: `/home/jameskimh/workspace/Workspace_DiT/Deepcache_for_NVFP4/`
- **모델**: PixArt-XL-2-1024-MS (28 transformer blocks, ~600M params)
- **Dataset**: MJHQ-30K (`xingjianleng/mjhq30k`, test split)
- **환경**: Python 3.11, torch 2.11.0, nvidia-modelopt 0.42.0, diffusers 0.37.1

---

## 연구 질문

**Cache-friendliness가 NVFP4 format 자체의 특성인가, 양자화 알고리즘에 의존하는가?**

기존 실험 (`pixart_caching/`)에서 SVDQuant + DeepCache 결합을 분석했다.
하지만 SVDQuant 하나만 테스트하면 "SVDQuant 구조의 우연한 부산물"이라는 reviewer 반박에 취약하다.

4가지 서로 다른 NVFP4 알고리즘 패밀리를 동일한 DeepCache 설정에서 비교함으로써:
- **cache_penalty가 method 간 유사** → NVFP4 format 자체의 inherent property (format-level insight)
- **cache_penalty가 method 간 상이** → 어떤 error compensation 전략이 cache-friendly한지 분석 가능

---

## Baseline (기존 pixart_caching 실험 결과)

| 항목 | 값 |
|------|-----|
| 방법 | SVDQuant (NVFP4_SVDQUANT_DEFAULT_CFG) |
| Scheduler | DPMSolverMultistepScheduler (DPM-Solver++ 2nd order) |
| Guidance Scale | 4.5 |
| Model | PixArt-XL-2-1024-MS |
| Lowrank | 32 |

**기존 결과 (20 samples)**:

| Config | FID ↓ | IS ↑ | PSNR ↑ | SSIM ↑ | Time/img (s) | Speedup |
|--------|--------|-------|--------|--------|--------------|---------|
| SVDQuant, no cache, 20-step | 161.30 | 1.750 | 15.69 | 0.590 | 3.45 | 1.00x |
| SVDQuant, deepcache[8,20), 20-step | 159.43 | 1.755 | 15.54 | 0.582 | 2.79 | 1.24x |
| SVDQuant, no cache, 15-step | 151.77 | 1.794 | 15.94 | 0.602 | 2.94 | 0.85x |
| SVDQuant, deepcache[8,20), 15-step | 162.99 | 1.763 | 15.41 | 0.577 | 1.93 | 1.79x |

---

## 4가지 NVFP4 양자화 방법

### M1: SmoothQuant + NVFP4 RTN (ManualRTNLinear)

```
원본 FP16 weight W
    ↓
[1] SmoothQuant: smooth_scale = w_max^(1-α) / x_max^α
    ↓
[2] W_smooth = W / diag(smooth_scale)
    ↓
[3] W_q = quantize_to_nvfp4(W_smooth)  ← RTN, no SVD correction
    ↓
Forward: x_q = quantize_to_nvfp4(x * smooth_scale)
         out = x_q @ W_q^T
```

- **Error compensation**: 없음 (가장 단순)
- **Format**: NVFP4 (weight + activation)
- **역할**: format만의 효과를 측정하는 기준선

### M2: SVDQuant (mtq.NVFP4_SVDQUANT_DEFAULT_CFG)

```
[1] AWQ-style calibration (activation max 기반)
[2] SmoothQuant + NVFP4 quantization
[3] SVD decompose error → lora_a, lora_b (rank=32)
Forward: base(x_q @ W_q^T) + svd(x_smooth @ lora_a^T @ lora_b^T)
```

- **Error compensation**: SVD low-rank correction (rank=32)
- **기존 baseline** — 기존 결과 재현 및 비교 기준

### M3: MR-GPTQ NVFP4 (MRGPTQLinear)

논문: "MR-GPTQ: Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization" (ICLR 2026)

```
핵심 insight: NVFP4 group size(16)에서 SmoothQuant은 무효
  - SmoothQuant은 layer-wide scale을 학습하지만 NVFP4는 이미 per-group(16) scale 보유
  - 대신 각 16-element group에 H(16) Hadamard 적용 → 분포 등방화

[1] act_order: H_diag(=E[x²]) 기준 채널 내림차순 정렬
    → 중요 채널이 동일 NVFP4 group(16)에 모임 → group scale 낭비 최소화
[2] H(16) Micro-Rotation: 각 16-element NVFP4 group에 Hadamard 적용 (weight & activation)
    - W_perm[:, g*16:(g+1)*16] = W_orig[:, g*16:(g+1)*16] @ H16
    - x_rot[..., g*16:(g+1)*16] = x_perm[..., g*16:(g+1)*16] @ H16
[3] Per-group NVFP4 quantization (group_size=16, SmoothQuant 없음)
[4] SVD error correction (rank=32)
Forward: x_perm = x[..., perm]
         x_rot  = H(16)(x_perm)   [group-level rotation]
         x_q    = quantize_to_nvfp4(x_rot)
         out    = base(x_q @ W_q^T) + svd(x_rot @ lora_a^T @ lora_b^T)
```

- **Error compensation**: group-level rotation (분포 등방화) + GPTQ act_order + SVD
- **검증 질문**: H(16) rotation이 cache-step에서도 activation distribution shift를 완화하는가?

### M4: Four Over Six NVFP4 (FourOverSixLinear)

논문: "Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling" (MIT Han Lab, arXiv 2512.02010)

```
핵심 insight: NVFP4 step sizes = [0.5×4, 1×2, 2×1]
  - 4→6 구간의 step size 2로 인해 near-maximal value (~5) 오류가 큼
  - max=4 선택 시: 값들이 normalized [0,4] 범위에 매핑 → step-2 구간 완전 회피

[1] SmoothQuant scale 적용 (activation scaling)
[2] Per-block adaptive scaling: 각 16-element block에서
    - scale6 = amax/6 → quantize → MSE 계산
    - scale4 = amax/4 → quantize → MSE 계산
    - 둘 중 MSE 낮은 쪽 선택
Forward: x_q = quantize_to_nvfp4_fouroversix(x * smooth_scale)
         out = x_q @ W_q^T
```

- **Error compensation**: 더 정확한 quantizer (adaptive scaling) — post-hoc correction 없음
- **검증 질문**: per-block 적응형 scaling이 cached residual과의 상호작용을 개선하는가?

---

## DeepCache 설정

기존 `pixart_caching/` 실험에서 검증된 최적 설정 고정:

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| cache_interval | 2 | 1 step 건너뛰기, quality/speed 최적 균형 |
| cache_start | 8 | 초기 shallow block (feature formation) 보존 |
| cache_end | 20 | 후기 final block (fine detail refinement) 보존 |
| full_steps | {0} | 첫 step은 항상 full (cold start) |
| guidance_scale | 4.5 | 기존 최적값 |

```
blocks [0, 8)   → shallow: 항상 연산
blocks [8, 20)  → deep:    cache 대상 (full step: 저장, cached step: residual 재사용)
blocks [20, 28) → final:   항상 연산
```

예상 speedup: **28 / ((28 + 8) / 2) ≈ 1.56x** (20-step 기준)

---

## 실험 매트릭스

**20-step 실험 (8 runs = 4 methods × 2 cache modes)**

| Method | No Cache | DeepCache int=2, [8,20) |
|--------|----------|-------------------------|
| M1: RTN | run_01 | run_02 |
| M2: SVDQuant | ~~기존 FID=161.30~~ | ~~기존 FID=159.43~~ |
| M3: MR-GPTQ | run_05 | run_06 |
| M4: FourOverSix | run_07 | run_08 |

**15-step 실험 (8 runs)**

| Method | No Cache | DeepCache int=2, [8,20) |
|--------|----------|-------------------------|
| M1: RTN | run_09 | run_10 |
| M2: SVDQuant | ~~기존 FID=151.77~~ | ~~기존 FID=162.99~~ |
| M3: MR-GPTQ | run_13 | run_14 |
| M4: FourOverSix | run_15 | run_16 |

> SVDQuant 결과는 기존 pixart_caching 실험에서 확보. 재실행 불필요.
> 단, 재현 확인이 필요하면 `--quant_method SVDQUANT` 옵션으로 실행 가능.

---

## 평가 메트릭

| 메트릭 | 방향 | 의미 |
|--------|------|------|
| FID ↓ | 낮을수록 좋음 | 생성 이미지 분포 vs FP16 ref 분포 거리 |
| IS ↑ | 높을수록 좋음 | 이미지 다양성 + 선명도 |
| PSNR ↑ | 높을수록 좋음 | FP16 ref 대비 픽셀 충실도 |
| SSIM ↑ | 높을수록 좋음 | 구조적 유사도 |
| LPIPS ↓ | 낮을수록 좋음 | 지각적 유사도 |
| CLIP ↑ | 높을수록 좋음 | 텍스트-이미지 정합성 |
| Time/img (s) | - | 1 이미지 생성 시간 (CUDA sync 기준) |
| Speedup | 높을수록 좋음 | vs SVDQuant no-cache 20-step 기준 |

**핵심 분석 지표**:
```
cache_penalty(method) = FID(with_cache) - FID(no_cache)
```

---

## 가설

1. **Format-level hypothesis**: NVFP4의 block-wise 16-element scaling 구조가
   cached residual과 quantized activation 간 error 전파 방식을 결정하므로,
   cache_penalty가 4가지 method 간 유사할 것이다.

2. **Algorithm-level hypothesis**: SVD low-rank branch가 stale residual error를
   부분 흡수한다면, SVDQuant와 RPCA (둘 다 SVD 있음)의 cache_penalty가
   RTN (SVD 없음)보다 작을 것이다.

3. **Rotation hypothesis**: MR-GPTQ의 H(16) Micro-Rotation이 각 NVFP4 group 내부
   activation 분포를 등방화하여 cached step에서도 distribution shift를 완화한다면,
   MR-GPTQ의 cache_penalty가 가장 작을 것이다.

4. **Adaptive scaling hypothesis**: Four Over Six의 per-block adaptive scaling이
   large quantization error를 줄임으로써 cache residual error 전파가 감소한다면,
   FourOverSix의 cache_penalty가 RTN보다 작을 것이다.

---

## 파일 구조

```
Deepcache_for_NVFP4/
├── deepcache_utils.py            # DeepCache monkey-patch (DeepCacheState, install_deepcache)
├── eval_utils.py                 # 평가 유틸 (get_prompts, generate_and_evaluate)
├── quant_methods.py              # 4가지 양자화 방법 구현
├── pixart_nvfp4_cache_compare.py # 메인 실험 스크립트
├── run_experiment.sh             # 전체 실험 런처
├── ref_paper/                    # 참고 논문 PDF
│   ├── MR-GPTQ - BRIDGING THE GAP...pdf
│   └── Four Over Six - More Accurate NVFP4...pdf
├── experiment.md                 # 이 파일 (실험 설계)
├── experiment_result.md          # 결과 분석 (실험 완료 후 작성)
└── results/
    └── MJHQ/
        ├── RTN_none_steps20/
        │   ├── sample_*.png
        │   ├── metrics.json
        │   └── metrics.csv
        ├── RTN_deepcache_steps20/
        ├── MRGPTQ_none_steps20/
        ├── MRGPTQ_deepcache_steps20/
        ├── FOUROVERSIX_none_steps20/
        ├── FOUROVERSIX_deepcache_steps20/
        └── ...
```

---

## 실행 방법

### 사전 준비

```bash
cd /home/jameskimh/workspace/Workspace_DiT/Deepcache_for_NVFP4
source /home/jameskimh/.dit/bin/activate
```

### Smoke test (2 samples, 특정 method 확인)

```bash
# MR-GPTQ + deepcache smoke test
/home/jameskimh/.dit/bin/accelerate launch \
    pixart_nvfp4_cache_compare.py \
    --quant_method MRGPTQ --cache_mode deepcache --num_steps 20 --test_run

# Four Over Six + no cache smoke test
/home/jameskimh/.dit/bin/accelerate launch \
    pixart_nvfp4_cache_compare.py \
    --quant_method FOUROVERSIX --cache_mode none --num_steps 20 --test_run
```

### 단일 run

```bash
/home/jameskimh/.dit/bin/accelerate launch \
    pixart_nvfp4_cache_compare.py \
    --quant_method RTN \
    --cache_mode deepcache \
    --num_steps 20 \
    --num_samples 20
```

### 전체 실험 (런처 사용)

```bash
# 전체 실험 (모든 method × cache × steps 순차 실행)
bash run_experiment.sh

# 특정 method만
bash run_experiment.sh --method RTN

# 특정 method + cache 조합만
bash run_experiment.sh --method GPTQ --cache deepcache --steps 20

# smoke test
bash run_experiment.sh --test_run
```

---

## 결과 분석 방향 (실험 후)

실험 완료 후 `experiment_result.md`에 다음을 분석:

1. **cache_penalty 비교표**: 4 methods × 2 steps → penalty 패턴 시각화
2. **Quality-Speed tradeoff**: FID vs speedup scatter plot 데이터
3. **SVD 유무 vs cache compatibility**: RTN vs SVDQuant/RPCA penalty 비교
4. **Rotation 효과**: GPTQ vs 나머지 penalty 비교
5. **15-step vs 20-step**: step 수가 cache_penalty에 미치는 영향

→ 결과 패턴에 따라 **novel contribution 아이디어** 발굴:
- Format-level 발견 → NVFP4-aware cache scheduling 연구
- Algorithm-level 발견 → cache-friendly NVFP4 calibration 전략 연구
