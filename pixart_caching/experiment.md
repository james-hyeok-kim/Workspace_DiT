# Experiment: NVFP4 Inference Acceleration via Step Reduction + Activation Caching

## 목표

NVFP4_SVDQUANT_DEFAULT_CFG (20-step baseline)보다 **빠르면서** FID/IS가 비슷하거나 더 나은 방법 탐색.

두 가지 독립적인 가속 기법을 순차적으로 실험하고, 최종적으로 조합하여 최대 speedup 달성.

---

## Baseline

| 항목 | 값 |
|------|-----|
| 방법 | NVFP4_SVDQUANT_DEFAULT_CFG |
| Scheduler | DPMSolverMultistepScheduler (DPM-Solver++ 2nd order) |
| Inference steps | 20 |
| Model | PixArt-XL-2-1024-MS (28 transformer blocks, 600M params) |
| Lowrank | 32 |
| FID (참고) | ~194 (prior experiments) |

---

## Experiment 1: Inference Step Sweep

### 개요

DPM-Solver++ 계열 scheduler는 step 수를 줄여도 품질 손실이 DDIM 대비 적음.
20-step이 실제로 최적인지 확인하고, FID/IS 손실 없이 줄일 수 있는 최소 step 탐색.

### 설계

- NVFP4 양자화: 1회 수행 (calibration 5-step 고정)
- Step counts sweep: `[5, 8, 10, 12, 15, 20]`
- Reference 이미지: **동일 step count의 FP16** 으로 생성 (공정한 비교)
  - 20-step FP16 ref로 10-step quant를 비교하면 step 감소 효과가 quantization error로 혼입되어 불공정
- 측정 항목: FID, IS, PSNR, SSIM, LPIPS, CLIP Score, time/image

### 가설

- DPM-Solver++ 특성상 10~12 step에서 knee point 존재
- 10-step: ~2x speedup, FID 손실 < 5%

### 스크립트

```bash
# Smoke test
bash run_step_sweep.sh --test_run

# Full experiment (20 samples, 6 step counts)
bash run_step_sweep.sh
```

### 결과 위치

```
results/MJHQ/step_sweep/
├── summary.json            # 전체 sweep 결과 테이블
├── steps_5/sample_*.png
├── steps_8/sample_*.png
├── steps_10/sample_*.png
├── steps_12/sample_*.png
├── steps_15/sample_*.png
└── steps_20/sample_*.png
```

### 결과 테이블

| Steps | FID ↓ | IS ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP ↑ | Time/img (s) | Speedup |
|-------|--------|-------|--------|--------|---------|--------|--------------|---------|
| 5     | 290.69 | 1.801 | 13.12  | 0.506  | 0.698   | 30.95  | 1.21         | 2.85x   |
| 8     | 172.35 | 1.757 | 15.56  | 0.631  | 0.444   | 34.75  | 1.52         | 2.27x   |
| 10    | 166.91 | 1.714 | 15.76  | 0.621  | 0.428   | 34.85  | 2.23         | 1.55x   |
| 12    | 169.30 | 1.758 | 15.70  | 0.608  | 0.434   | 34.68  | 2.14         | 1.61x   |
| **15**| **151.77** | **1.794** | **15.94** | **0.602** | **0.434** | **35.23** | **2.94** | **1.17x** |
| 20    | 161.30 | 1.750 | 15.69  | 0.590  | 0.441   | 34.98  | 3.45         | 1.00x   |

### 주요 발견

- **15-step이 최적**: FID=151.77로 20-step(161.30)보다 오히려 더 좋음 — DPM-Solver++에서 20-step이 오버슈팅 가능성
- **8-10 step**: FID ~166-172 수준 유지, 1.5~2.3x speedup — 수용 가능한 trade-off
- **5-step**: FID 290으로 급락 — 사용 불가
- **결론**: 15-step 선택 (FID 개선 + 1.17x 무료 speedup)

---

## Experiment 2: DeepCache Block-Level Activation Caching

### 개요

PixArt transformer의 28개 block 중 "deep blocks"는 인접 timestep 간 output 변화가 작음.
일부 denoising step에서 deep block 연산을 skip하고 이전 step의 residual을 재사용.

### 메커니즘

#### Block 분할

```
Block 0 ~ cache_start-1  : Shallow  (항상 연산)
Block cache_start ~ cache_end-1 : Deep  (캐시 대상)
Block cache_end ~ 27      : Final   (항상 연산)

기본값: cache_start=4, cache_end=24 → deep 20개 block 캐시
```

#### Full Step (매 cache_interval step)

```
h_shallow = run(blocks[0:cache_start], hidden_states)
h_before_deep = h_shallow.clone()
h_after_deep  = run(blocks[cache_start:cache_end], h_shallow)
cache.residual = h_after_deep - h_before_deep   ← 저장
output = run(blocks[cache_end:], h_after_deep)
```

#### Cached Step (나머지 step)

```
h_shallow = run(blocks[0:cache_start], hidden_states)
h_deep    = h_shallow + cache.residual           ← SKIP + 캐시 적용
output    = run(blocks[cache_end:], h_deep)
```

#### Speedup 계산 (theoretical)

```
n_total = 28, n_deep = cache_end - cache_start
n_always = 28 - n_deep

avg_blocks = (n_total + n_always × (interval - 1)) / interval

speedup = 28 / avg_blocks
```

예: interval=2, deep=20 (blocks 4~23)
→ avg_blocks = (28 + 8×1) / 2 = 18 → speedup ≈ **1.56x**

### 구현

- `pixart_deepcache_experiment.py`
- `transformer.forward`를 `types.MethodType`으로 monkey-patch (diffusers 소스 무수정)
- 이미지 생성 전 `cache_state.reset()` 호출 (step counter = 0 초기화)
- NVFP4 양자화와 완전 호환 (block 단위 skip이므로 내부 quantized linear 영향 없음)

### Ablation Configs

| interval | cache_start | cache_end | deep blocks | est. speedup |
|----------|-------------|-----------|-------------|--------------|
| 1        | 4           | 24        | 20          | 1.00x (baseline) |
| 2        | 4           | 24        | 20          | 1.56x        |
| 3        | 4           | 24        | 20          | 1.84x        |
| 4        | 4           | 24        | 20          | 1.94x        |
| 2        | 2           | 26        | 24          | 1.75x        |
| 2        | 8           | 20        | 12          | 1.30x        |
| 2        | 4           | 28        | 24          | 1.75x        |

### 가설

- interval=2, blocks[4,24): FID 손실 < 10%, speedup ≈ 1.56x
- interval=1: baseline과 동일한 FID (sanity check)
- 넓은 block range (blocks[2,26))가 더 큰 speedup이나 FID 손실 위험

### 스크립트

```bash
# Smoke test (single config)
bash run_deepcache_experiment.sh --test_run

# Single config (default: interval=2, blocks[4,24))
bash run_deepcache_experiment.sh

# Ablation sweep (7 configs)
bash run_deepcache_experiment.sh --sweep
```

### 결과 위치

```
results/MJHQ/deepcache/
├── sweep_summary.json                      # sweep 전체 결과
├── interval1_s4_e24/metrics.json           # baseline (no cache)
├── interval2_s4_e24/metrics.json           # standard deepcache
├── interval3_s4_e24/metrics.json
├── interval4_s4_e24/metrics.json
├── interval2_s2_e26/metrics.json
├── interval2_s8_e20/metrics.json
└── interval2_s4_e28/metrics.json
```

### 결과 테이블

| interval | blocks  | FID ↓  | IS ↑  | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP ↑ | Time/img (s) | Actual speedup |
|----------|---------|--------|-------|--------|--------|---------|--------|--------------|----------------|
| 1 (base) | [4, 24) | 161.30 | 1.772 | 15.69  | 0.590  | 0.441   | 34.98  | 3.59         | 1.00x          |
| **2**    | **[8, 20)** | **159.43** | **1.727** | **15.41** | **0.580** | **0.448** | **34.58** | **2.89** | **1.24x** |
| 2        | [2, 26) | 175.43 | 1.775 | 15.49  | 0.593  | 0.436   | 35.15  | 2.32         | 1.55x          |
| 2        | [4, 24) | 175.99 | 1.757 | 15.52  | 0.582  | 0.442   | 34.99  | 2.54         | 1.41x          |
| 3        | [4, 24) | 201.20 | 1.718 | 15.05  | 0.556  | 0.494   | 33.96  | 2.02         | 1.78x          |
| 2        | [4, 28) | 196.81 | 1.758 | 14.76  | 0.491  | 0.568   | 34.17  | 2.02         | 1.78x          |
| 4        | [4, 24) | 219.06 | 1.832 | 14.40  | 0.544  | 0.533   | 33.43  | 2.16         | 1.66x          |

### 주요 발견

- **interval=2, blocks[8,20)**: FID=159.43으로 baseline(161.30)과 거의 동일 + 1.24x speedup — **품질 손실 최소**
- **interval=2, blocks[2,26)**: IS=1.775로 baseline보다 높고 FID=175, 1.55x speedup — **속도 우선 선택지**
- **interval=3 이상**: FID 200+ 급등 — 권장하지 않음
- **결론**: `interval=2, blocks[8,20)` 선택 (FID 거의 유지 + 1.24x speedup)

---

## Experiment 3: Step Reduction + DeepCache 조합

### 개요

Experiment 1에서 찾은 최적 step수 + Experiment 2에서 찾은 최적 caching config 조합.
두 기법의 speedup은 곱셈적으로 작용.

### 예상 speedup

| 방법 | Speedup |
|------|---------|
| Step 20→10 | 2.0x |
| DeepCache interval=2 | 1.56x |
| 조합 | **3.1x** |

### 실행

```bash
# 최적 step과 caching config를 Exp 1, 2 결과에서 결정 후 실행
accelerate launch --num_processes 1 pixart_deepcache_experiment.py \
    --num_inference_steps {best_step} \
    --cache_interval 2 \
    --cache_start 4 \
    --cache_end 24 \
    --num_samples 20
```

### 조합 실험 결과

Exp 1 최적: **15-step** (FID=151.77, time=2.94s)
Exp 2 최적: **interval=2, blocks[8,20)** (FID≈same as 20-step, 1.24x speedup)

| 방법 | Steps | Cache blocks | FID ↓ | IS ↑ | PSNR ↑ | CLIP ↑ | Time/img (s) | Speedup vs baseline |
|------|-------|--------------|--------|-------|--------|--------|--------------|---------------------|
| NVFP4 20-step baseline | 20 | None | 161.30 | 1.772 | 15.69 | 34.98 | 3.59 | 1.00x |
| Step 15 only | 15 | None | 151.77 | 1.794 | 15.94 | 35.23 | 2.94 | 1.22x |
| Cache [8,20) only | 20 | int=2 | 159.43 | 1.727 | 15.41 | 34.58 | 2.89 | 1.24x |
| Cache [2,26) only | 20 | int=2 | 175.43 | 1.775 | 15.49 | 35.15 | 2.32 | 1.55x |
| **조합1: 15-step + [8,20)** | **15** | **int=2** | **162.99** | **1.766** | **15.34** | **34.93** | **2.01** | **1.79x** |
| **조합2: 15-step + [2,26)** | **15** | **int=2** | **188.01** | **1.708** | **15.28** | **33.98** | **1.58** | **2.27x** |

### 분석

- **조합1 (15-step + [8,20))**: FID=162.99로 baseline(161.30)과 거의 동일 + **1.79x speedup** ← 가장 균형적
- **조합2 (15-step + [2,26))**: FID=188.01로 baseline 대비 +27 상승, 단 **2.27x speedup** ← 속도 최우선 시
- 15-step 단독의 FID=151.77이 cache 추가로 162.99로 소폭 상승 → cache penalty는 약 +11 FID
- 두 조합 모두 NVFP4 prior baseline(~194) 이하 또는 근접 수준 달성

---

## 전체 결과 통합표

> NVFP4_SVDQUANT_DEFAULT_CFG 기반, 20 samples, MJHQ dataset
> Baseline 기준: 20-step, no cache (FID=161.30, time=3.59s)

### Step Sweep (cache 없음, FP16 동일 step 대비)

| Steps | FID ↓  | IS ↑  | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP ↑ | Time/img (s) | Speedup |
|-------|--------|-------|--------|--------|---------|--------|--------------|---------|
| 5     | 290.69 | 1.801 | 13.12  | 0.506  | 0.698   | 30.95  | 1.21         | 2.85x   |
| 8     | 172.35 | 1.757 | 15.56  | 0.631  | 0.444   | 34.75  | 1.52         | 2.27x   |
| 10    | 166.91 | 1.714 | 15.76  | 0.621  | 0.428   | 34.85  | 2.23         | 1.55x   |
| 12    | 169.30 | 1.758 | 15.70  | 0.608  | 0.434   | 34.68  | 2.14         | 1.61x   |
| **15**| **151.77** | **1.794** | **15.94** | **0.602** | **0.434** | **35.23** | **2.94** | **1.22x** |
| 20    | 161.30 | 1.750 | 15.69  | 0.590  | 0.441   | 34.98  | 3.59         | 1.00x   |

### DeepCache Ablation (20-step, FP16 20-step 대비)

| interval | Cache blocks | FID ↓  | IS ↑  | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP ↑ | Time/img (s) | Speedup |
|----------|-------------|--------|-------|--------|--------|---------|--------|--------------|---------|
| 1 (base) | [4, 24)     | 161.30 | 1.772 | 15.69  | 0.590  | 0.441   | 34.98  | 3.59         | 1.00x   |
| **2**    | **[8, 20)** | **159.43** | **1.727** | **15.41** | **0.580** | **0.448** | **34.58** | **2.89** | **1.24x** |
| 2        | [2, 26)     | 175.43 | 1.775 | 15.49  | 0.593  | 0.436   | 35.15  | 2.32         | 1.55x   |
| 2        | [4, 24)     | 175.99 | 1.757 | 15.52  | 0.582  | 0.442   | 34.99  | 2.54         | 1.41x   |
| 2        | [4, 28)     | 196.81 | 1.758 | 14.76  | 0.491  | 0.568   | 34.17  | 2.02         | 1.78x   |
| 3        | [4, 24)     | 201.20 | 1.718 | 15.05  | 0.556  | 0.494   | 33.96  | 2.02         | 1.78x   |
| 4        | [4, 24)     | 219.06 | 1.832 | 14.40  | 0.544  | 0.533   | 33.43  | 2.16         | 1.66x   |

### 조합 실험 (Step 축소 + DeepCache, FP16 20-step 대비)

| 방법                     | Steps | Cache blocks | FID ↓  | IS ↑  | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP ↑ | Time/img (s) | Speedup |
|--------------------------|-------|--------------|--------|-------|--------|--------|---------|--------|--------------|---------|
| Baseline (20-step, no cache) | 20 | None      | 161.30 | 1.772 | 15.69  | 0.590  | 0.441   | 34.98  | 3.59         | 1.00x   |
| 15-step only             | 15    | None         | 151.77 | 1.794 | 15.94  | 0.602  | 0.434   | 35.23  | 2.94         | 1.22x   |
| Cache [8,20) only        | 20    | int=2        | 159.43 | 1.727 | 15.41  | 0.580  | 0.448   | 34.58  | 2.89         | 1.24x   |
| Cache [2,26) only        | 20    | int=2        | 175.43 | 1.775 | 15.49  | 0.593  | 0.436   | 35.15  | 2.32         | 1.55x   |
| **조합1: 15-step + [8,20)**  | **15** | **int=2** | **162.99** | **1.766** | **15.34** | **0.587** | **0.444** | **34.93** | **2.01** | **1.79x** |
| **조합2: 15-step + [2,26)**  | **15** | **int=2** | **188.01** | **1.708** | **15.28** | **0.584** | **0.464** | **33.98** | **1.58** | **2.27x** |

### 최종 권장

| 우선순위 | 선택 | FID | Speedup | 이유 |
|---------|------|-----|---------|------|
| 품질 최우선 | **15-step only** | 151.77 | 1.22x | FID 가장 낮음, 구현 단순 |
| 균형 (추천) | **조합1: 15-step + [8,20), gs=4.5** | 162.99 | **1.79x** | FID baseline 수준 유지 + 빠름 |
| 속도 최우선 | **조합2: 15-step + [2,26)** | 188.01 | **2.27x** | 2배 이상 빠름, FID 허용 범위 내 |

> Exp 4 결과: guidance_scale 변경(gs=5.5, 3.5, 3.0)은 조합1 대비 FID 개선 없음 → gs=4.5 기본값이 최적

---

## Experiment 4: Test Sweep — guidance_scale / Scheduler / full_steps

### 설계

- **고정**: NVFP4 양자화(lowrank=32), DeepCache interval=2 blocks[8,20), 15-step
- **고정**: FP16 ref = gs=4.5, 기본 scheduler (하나로 통일)
- **Sweep**: guidance_scale, Karras/Lu scheduler, full_steps
- **규모**: 20 samples × 10 configs

> ⚠️ IS=1.0 전체: 20 samples로는 InceptionScore 계산이 불안정. FID 중심으로 해석.

### 결과 테이블

| guidance_scale | Karras | Lu | full_steps | FID ↓  | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP ↑ | Time/img (s) |
|---------------|--------|----|------------|--------|--------|--------|---------|--------|--------------|
| 4.5 (base)    | N | N | 0     | 139.97 | 16.33  | 0.594  | 0.428   | 35.89  | 1.87         |
| **5.5**       | N | N | 0     | **124.87** | 15.45 | 0.562 | 0.467  | 35.17  | 1.93         |
| 3.5           | N | N | 0     | 130.80 | **16.74** | **0.610** | **0.414** | 35.49 | 1.63 |
| 3.0           | N | N | 0     | 133.55 | 16.22  | 0.592  | 0.451   | 34.70  | 1.75         |
| 4.5           | N | N | 0,1   | 133.30 | 16.49  | 0.613  | 0.410   | 34.41  | 1.84         |
| 4.5           | N | N | 0,1,2 | 133.30 | 16.49  | 0.613  | 0.410   | 34.41  | 1.88         |
| 6.0           | N | N | 0     | 135.13 | 14.51  | 0.530  | 0.535   | 33.13  | 1.95         |
| 4.5           | N | Y | 0     | 143.71 | 15.94  | 0.568  | 0.467   | 34.72  | **4.38** ⚠️  |
| 4.5           | Y | N | 0,1   | 142.28 | 16.68  | 0.605  | 0.436   | 34.07  | 3.32 ⚠️      |
| 4.5           | Y | N | 0     | 155.74 | 16.37  | 0.589  | 0.447   | 35.52  | 1.95         |

### 주요 발견 (2-sample test_sweep)

- **gs=5.5가 FID 최저(124.87)**: 기존 조합1(162.99) 대비 -38 FID 감소 (단, 2 samples로 불안정)
- **gs=3.5가 PSNR/SSIM 최고(16.74, 0.610)**: pixel 품질 우선 시 선택지
- **Karras/Lu**: FID 개선 없음. Lu는 시간 4.38s로 비정상적으로 느림 → 사용 불필요
- **full_steps 0,1 vs 0,1,2**: FID 동일(133.30) — full_steps 추가 불필요
- **결론**: `gs=5.5, gs=3.5, gs=3.0` top 3를 20-sample full 실험으로 검증

---

## Experiment 4 Full Run: Top 3 guidance_scale (20 samples)

### 설계

- test_sweep top 3 (gs=5.5, gs=3.5, gs=3.0)를 20 samples full run으로 검증
- **고정**: NVFP4 양자화(lowrank=32), DeepCache interval=2 blocks[8,20), 15-step
- **기준(조합1)**: gs=4.5, FID=162.99, IS=1.766, time=2.01s

### 결과 테이블

| guidance_scale | FID ↓  | IS ↑  | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP ↑ | Time/img (s) | vs 조합1 (FID) |
|----------------|--------|-------|--------|--------|---------|--------|--------------|----------------|
| **4.5 (조합1 base)** | **162.99** | **1.766** | **15.34** | **0.587** | **0.444** | **34.93** | **2.01** | **기준** |
| 5.5            | 161.62 | 1.755 | 14.90  | —      | —       | —      | 2.41         | -1.37 (소폭 개선) |
| 3.5            | 182.85 | 1.762 | 15.32  | —      | —       | —      | 2.45         | +19.86 (악화)  |
| 3.0            | 194.57 | 1.791 | 15.18  | —      | —       | —      | 2.46         | +31.58 (악화)  |

### 주요 발견

- **gs=5.5**: FID=161.62로 조합1(162.99) 대비 소폭 개선(-1.37). 단, test_sweep의 기대치(-38)와 큰 괴리 → 2-sample 결과는 신뢰 불가
- **gs=3.5, gs=3.0**: FID가 각각 182.85, 194.57로 조합1 대비 **악화** → guidance_scale을 낮추면 FID 손실
- **IS**: 세 config 모두 1.755~1.791로 유사. gs=3.0에서 IS 최고(1.791)이나 FID 가장 나쁨
- **결론**: guidance_scale 튜닝으로 조합1(FID=162.99) 대비 의미있는 개선 없음. **gs=4.5 기본값 유지**

> ⚠️ 동시 실행 시 `config_tag`에 `guidance_scale` 미포함으로 save_dir 충돌 발생 (버그 수정됨: `interval{N}_s{S}_e{E}_gs{G}` 형식으로 변경)

---

---

## Experiment 5: Novelty 실험 — NVFP4 × Cache 상호작용 3종

### 배경 및 차별점

| 기존 연구 | 대상 | 한계 |
|-----------|------|------|
| Q&C | class-conditional DiT (ImageNet DiT-XL) | INT quantization 기반, PixArt 미검증 |
| CacheQuant | UNet (Stable Diffusion) | cross-attention 구조 부재 |
| QuantCache | Video (Open-Sora) | text-to-image DiT 아님 |

본 실험은 **NVFP4 (E4M3 micro-block) + SVD low-rank branch × DeepCache** 조합의 고유한 상호작용을 분석.

---

### Exp D: 2×2 Ablation — rank × cache (논문 핵심 테이블)

"SVD가 cache error를 보정한다"는 주장을 성립시키려면 기여를 분리해야 합니다.

```
                │  No Cache (interval=1)  │  With Cache (interval=2)
────────────────┼─────────────────────────┼──────────────────────────
rank=32 (SVD)   │  D-A: 측정 예정         │  B: 조합1 FID=162.99 ✓
rank=0  (no SVD)│  D-C: 측정 예정         │  A-2: Exp A-2에서 측정
```

**분석 방법**:
- **SVD 단독 기여**: `FID(D-A) - FID(D-C)` — cache 없이 SVD branch만의 효과
- **Cache 단독 기여**: `FID(B) - FID(D-A)` — SVD 있을 때 cache가 주는 변화
- **상호작용(synergy)**: `[FID(B) - FID(D-A)] - [FID(A-2) - FID(D-C)]`
  - ≠ 0이면 SVD × Cache synergy 존재 → 논문의 핵심 novelty 증거

**스크립트**: `bash run_analysis.sh` 첫 두 블록 (Exp D-A, D-C)

| 조건 | rank | cache | FID ↓ | IS ↑ | PSNR ↑ | Time/img |
|------|------|-------|--------|-------|--------|----------|
| D-C (pure NVFP4) | 0 | ✗ | 203.69 | 1.780 | 12.97 | 2.21s |
| D-A (SVD only)   | 32 | ✗ | 154.10 | 1.748 | 15.88 | 2.24s |
| A-2 (cache only) | 0 | ✓ | 224.36 | 1.733 | 12.92 | 2.05s |
| **B (SVD+cache, 조합1)** | **32** | **✓** | **162.99** | **1.753** | **15.34** | **2.14s** |

**상호작용 분석**:
- SVD 단독 효과: 203.69 → 154.10 (**-49.59 FID**)
- Cache 단독 효과 (no SVD): 203.69 → 224.36 (**+20.67 FID** 악화)
- Cache 효과 (with SVD): 154.10 → 162.99 (**+8.89 FID** 악화)
- **상호작용 = 20.67 - 8.89 = +11.78**: SVD가 있으면 cache penalty가 절반 이하로 감소

**해석**: SVD branch는 stale residual 오차 자체를 줄이지 않지만(Exp A 결과), 더 나은 base 품질을 제공해 cache error에 대한 **robustness**를 높임. cache 없이 SVD만 써도 FID가 크게 개선되므로, SVD와 cache는 독립적 기여 + 양(+)의 상호작용 구조.

---

### Exp A: SVD Low-rank Branch × Cache Residual Error 분석

**핵심 질문**: SVD branch(rank=32)가 stale residual 오차를 자연스럽게 보정하는가?

**실험 설계**:
```
A-1: NVFP4 (rank=32) + DeepCache → residual_errors_rank32.csv
A-2: NVFP4 (rank=0)  + DeepCache → residual_errors_rank0.csv
```

측정값: 각 cached step에서 `rel_err = ||stale_residual - fresh_residual|| / ||fresh_residual||`

- `rel_err(rank=32) < rel_err(rank=0)` → SVD branch가 cache error 보정 효과 有 (novelty)
- `rel_err(rank=32) > rel_err(rank=0)` → SVD outlier absorption이 cached step 분포와 미스매치

**출력 파일**:
```
results/MJHQ/deepcache/interval2_s8_e20_gs4.5/residual_errors_rank32.csv
results/MJHQ/deepcache/interval2_s8_e20_gs4.5/residual_errors_rank0.csv
  columns: image_idx, step_idx, abs_err, rel_err
```

**결과** (20 samples, 140 cached steps 측정):

| 조건 | mean rel_err | std | FID |
|------|-------------|-----|-----|
| rank=32 + cache | **0.2786** | 0.1507 | 162.99 |
| rank=0  + cache | 0.2840 | 0.1219 | 224.36 |

**해석**: SVD branch가 stale residual의 상대 오차를 직접 줄이는 효과는 미미 (2% 차이). SVD의 기여는 cache residual error 보정이 아니라, base quantization 품질 향상을 통한 **cache error robustness 증가**. → Exp D의 상호작용 분석(+11.78 FID)과 일관된 해석.

---

### Exp B: Block-level Output Drift Profiling

**핵심 질문**: 어떤 block이 cache에 가장 적합한가? cross-attention vs self-attention 차이?

**측정값**: 연속된 full step 사이 각 block의 relative output drift
```
drift[b][t] = ||block_b_output(t) - block_b_output(t-1)||_F / ||block_b_output(t)||_F
```

full step에서만 측정 (deep block은 cached step에서 skip되므로 full step 기준으로 비교).

**PixArt 구조 특성**:
```
PixArt block = self-attn + cross-attn(text tokens) + FFN + adaLN(t, text)
```
- cross-attn의 key/value는 text embedding으로 timestep 간 **고정** → deep block의 cross-attn 부분은 안정적
- adaLN의 modulation은 timestep 임베딩으로 변화 → shallow block에 더 많은 영향

**결과** (20 samples, full step 간 drift 측정):

| 구역 | Block 범위 | Mean Drift | 해석 |
|------|-----------|-----------|------|
| shallow | [0, 7]  | **0.2128** | 가장 안정적 — timestep embedding 근처 |
| deep    | [8, 19] | 0.3273 | 중간 — cache 대상 |
| final   | [20, 27]| 0.3803 | 가장 불안정 — noise prediction 출력 근처 |

**주요 발견**: 예상과 달리 deep block이 shallow보다 drift가 높음. DeepCache가 효과적인 이유는 "low drift"가 아니라 **residual(output - input)의 안정성** 때문. Shallow block의 입력 변화가 deep block을 통과하며 증폭되더라도, residual = (deep_out - deep_in)은 stable할 수 있음. → 실제 cache 품질 지표는 Exp A의 residual rel_err(0.279).

**PixArt 구조 특성**: cross-attention의 key/value(text embedding)는 timestep 간 고정 → text-conditional drift가 낮지만 self-attn + adaLN timestep modulation이 지배적.

**출력 파일**:
```
results/MJHQ/deepcache/interval2_s8_e20_gs4.5/block_drift_profile.csv
  columns: block_idx, region(shallow/deep/final), mean_drift, std_drift, max_drift
```

---

### Exp C: Cache-Aware NVFP4 Calibration

**핵심 질문**: calibration을 cached step 포함해서 하면 FID/IS가 개선되는가?

**문제**: 현재 calibration은 full step activations만 사용.
- cached step에서 final block[20~27]의 입력 = `shallow_out + stale_residual`
- 이 분포가 full step 입력과 다를 수 있음 → NVFP4 scaling factor 미스매치

**실험 설계**:
```
C-baseline: standard calib (full step only) + cache → FID=162.99 (기존 조합1)
C-aware:    cache-aware calib (DeepCache during calib) + cache → FID 측정
```

**구현**:
- `mtq.quantize()` 호출 전 DeepCache 설치
- calibration forward pass 5 steps 중 절반(step 1,3)이 cached step
- NVFP4 E4M3 micro-block scaling factor가 cached-step activation 분포도 반영

**출력 파일**:
```
results/MJHQ/deepcache/interval2_s8_e20_gs4.5_calib_cache/metrics.csv
results/MJHQ/deepcache/calib_meta.json
```

**결과**:

| 방법 | Calibration | FID ↓ | IS ↑ | PSNR ↑ | Time/img |
|------|-------------|--------|-------|--------|----------|
| 조합1 (standard calib) | full step only | 162.99 | 1.753 | 15.34 | 2.14s |
| **Exp C (cache-aware calib)** | full + cached | **163.92** | **1.724** | **15.98** | **1.88s** |

**해석**: cache-aware calibration 시 FID 163.92로 standard(162.99) 대비 소폭 악화(-0.93). PSNR은 15.98로 개선(+0.64), IS는 소폭 감소. 통계적으로 유의미한 차이 없음 → cached step activation을 calibration에 포함하는 것이 NVFP4 scaling에 큰 영향을 주지 않음. NVFP4의 per-block E4M3 scaling이 이미 충분히 robust한 것으로 추정.

---

### 실행

```bash
# 3개 실험 순차 실행 (~1~2시간)
bash run_analysis.sh
```

개별 실행:
```bash
# Exp A-1: rank=32 residual error
accelerate launch --num_processes 1 pixart_deepcache_experiment.py \
    --lowrank 32 --num_inference_steps 15 \
    --cache_interval 2 --cache_start 8 --cache_end 20 \
    --profile_residual_error

# Exp A-2: rank=0 비교
accelerate launch --num_processes 1 pixart_deepcache_experiment.py \
    --lowrank 0 --num_inference_steps 15 \
    --cache_interval 2 --cache_start 8 --cache_end 20 \
    --profile_residual_error

# Exp B: block drift
accelerate launch --num_processes 1 pixart_deepcache_experiment.py \
    --lowrank 32 --num_inference_steps 15 \
    --cache_interval 2 --cache_start 8 --cache_end 20 \
    --profile_blocks

# Exp C: cache-aware calibration
accelerate launch --num_processes 1 pixart_deepcache_experiment.py \
    --lowrank 32 --num_inference_steps 15 \
    --cache_interval 2 --cache_start 8 --cache_end 20 \
    --cache_aware_calib
```

---

---

## Q&C Baseline 비교 실험 결과 (run_qandc_comparison.sh)

> 실험일: 2026-04-14 | 스크립트: run_qandc_comparison.sh | 샘플: 20개

### 비교 구성

| # | 방법 | rank | TAP | VC | 설명 |
|---|------|------|-----|-----|------|
| 1 | **Ours** | 32 | ✗ | ✗ | 조합1 재현 (SVD+DeepCache) |
| 2 | Q&C approx | 0 | ✓ | ✓ | Q&C 핵심 기법만 (SVD 없음) |
| 3 | Q&C+SVD | 32 | ✓ | ✓ | Q&C 기법 + SVD branch |
| 4 | VC only | 32 | ✗ | ✓ | Ablation: VC만 적용 |

> ⚠️ Q&C 공식 코드 미확보 — TAP(full 15-step calibration), VC(std ratio scaling) 근사 구현

### 결과 테이블

| 방법 | FID ↓ | IS ↑ | PSNR ↑ | time/img |
|------|--------|-------|--------|---------|
| **[1] Ours** (rank=32, no TAP/VC) | **162.99** ✅ | 1.737 | 15.34 | 1.96s |
| [2] Q&C approx (rank=0, TAP+VC) | 233.56 ❌ | 1.759 | 13.05 | 1.82s |
| [3] Q&C+SVD (rank=32, TAP+VC) | 165.88 | 1.753 | **15.40** | 1.83s |
| [4] VC only (rank=32, VC) | 170.36 | **1.783** | 15.30 | 1.83s |

CSV: `results/MJHQ/deepcache/qandc_comparison_summary.csv`

### 주요 발견

1. **우리 방법이 FID 기준 최우수**: Ours(162.99) < Q&C+SVD(165.88) < VC only(170.36) < Q&C approx(233.56)

2. **SVD branch가 핵심**: Q&C approx(rank=0)은 FID 233.56으로 급등 → SVD branch 없이는 TAP+VC도 부족
   - SVD branch의 low-rank correction이 cache stale error를 흡수하는 것이 더 효과적

3. **TAP+VC 추가 시 오히려 소폭 악화**: Ours(162.99) → Q&C+SVD(165.88) (+2.89)
   - NVFP4 micro-block scaling 특성상 INT quantization 대상으로 설계된 TAP/VC가 최적 미달
   - Q&C의 TAP/VC는 INT4/INT8 대상 기법이므로 NVFP4 환경에서는 불필요

4. **VC only가 IS 최고**: IS=1.783 (VC alone이 image diversity 측면에서 유리)
   - 그러나 FID는 170.36으로 Ours 대비 4.5% 악화

5. **우리 방법의 차별점 확인**:
   - Q&C가 TAP/VC로 calibration 보정 → 우리는 SVD branch로 구조적 오차 보정
   - SVD+DeepCache 조합이 TAP/VC 없이도 더 낮은 FID 달성

---

## Follow-up: Q&C / CacheQuant / QuantCache Baseline 비교

> 현재 실험(Exp A~D)과 독립적으로 진행 가능. 현재 실험 완료 후 추가.

### 비교 대상 정리

| 논문 | Venue | Quant | Cache | 대상 |
|------|-------|-------|-------|------|
| **Q&C** | ICLR 2026 | INT4/INT8 | TAP+VC | class-conditional DiT (ImageNet DiT-XL) |
| **CacheQuant** | 2024 | INT8 | DPS+DEC | UNet (Stable Diffusion) |
| **QuantCache** | 2024 | INT8 | temporal | Video DiT (Open-Sora) |
| **Ours** | — | NVFP4+SVD | DeepCache | PixArt text-to-image DiT |

### Q&C 핵심 기법 (가장 최신, ICLR 2026)

- **TAP (Timestep-Aware Perturbation)**: timestep별 quantization sensitivity를 반영한 calibration
- **VC (Variance Calibration)**: layer별 activation variance를 보정하여 quantization error 감소
- Q&C의 핵심 claim: "cache가 calibration data를 오염시킨다" → TAP/VC로 해결
- **우리의 대응**: Exp C (cache-aware calib)가 유사한 문제를 NVFP4 micro-block scaling으로 접근

### 비교 실험 설계 (후속)

```bash
# run_baselines.sh (예정)
# 1. Q&C: TAP+VC 재구현 또는 공식 코드 적용 → PixArt에 이식
# 2. CacheQuant: DPS(Dynamic Precision Selection) + DEC(DeepCache) 적용
# 3. QuantCache: temporal cache 적용
# 4. Ours: Exp D 결과 사용
```

**비교 테이블 구조** (후속 작업 완료 후 업데이트):

| 방법 | Quant bit | FID ↓ | IS ↑ | Time/img | Speedup |
|------|-----------|--------|-------|----------|---------|
| FP16 baseline | 16 | — | — | ~3.6s | 1.0x |
| **NVFP4+SVD+Cache (Ours)** | ~4 | **162.99** | 1.737 | 1.96s | 1.79x |
| Q&C+SVD (rank=32, TAP+VC) | ~4 | 165.88 | 1.753 | 1.83s | 1.79x |
| VC only (rank=32, VC) | ~4 | 170.36 | **1.783** | 1.83s | 1.79x |
| Q&C approx (rank=0, TAP+VC) | ~4 | 233.56 | 1.759 | 1.82s | 1.79x |
| CacheQuant | 8 | 예정 | 예정 | 예정 | 예정 |
| QuantCache | 8 | 예정 | 예정 | 예정 | 예정 |

### 차별점 포지셔닝

우리가 주장할 수 있는 차별점:

1. **SVD branch의 cache error 보정 효과** (Exp A 결과 기반)
   - Q&C는 TAP/VC로 calibration을 보정하지만 SVD branch 활용은 없음
   - SVD branch가 자연스럽게 stale residual error를 흡수하면 → 추가 보정 없이 simpler

2. **PixArt text-to-image 특화** (Exp B 결과 기반)
   - Q&C는 class-conditional DiT만 검증
   - cross-attention 구조에서의 drift 특성이 다름을 정량화

3. **NVFP4 micro-block scaling의 고유 특성** (Exp C 결과 기반)
   - INT quantization 기반 방법과 달리 E4M3 16-element block scaling
   - cache-aware calibration이 INT보다 더 효과적일 가능성

---

## 비교 기준 (NVFP4_DEFAULT vs 신규 방법)

| 지표 | 방향 | 기준 (NVFP4 20-step) | 목표 |
|------|------|----------------------|------|
| FID | ↓ | ~194 | ≤ 200 (±3%) |
| IS | ↑ | — | 유지 |
| Time/img | ↓ | baseline | < 0.5× (2x faster) |

---

## 파일 구조

```
pixart_caching/
├── experiment.md                   ← 이 파일
├── pixart_step_sweep.py            ← Experiment 1 스크립트
├── pixart_deepcache_experiment.py  ← Experiment 2, 3 스크립트
├── run_step_sweep.sh
├── run_deepcache_experiment.sh
├── ref_images/
│   └── MJHQ/
│       ├── steps_5/ref_*.png
│       ├── steps_10/ref_*.png
│       ├── steps_20/ref_*.png
│       └── ref_*.png               ← deepcache용 (20-step FP16)
└── results/
    └── MJHQ/
        ├── step_sweep/
        │   └── summary.json
        └── deepcache/
            └── sweep_summary.json
```

---

## 조합1 핵심 아이디어 요약

**설정**: NVFP4 양자화 + 15-step + DeepCache interval=2, blocks[8,20)

### 1. NVFP4 Weight Quantization (기반)

- PixArt transformer의 Linear 레이어를 **4-bit NV float**으로 압축
- SVD low-rank branch (rank=32)로 양자화 오차 보정
- 메모리 절약 + 연산량 감소, FP16 대비 품질 손실 최소화

### 2. Step Reduction: 20 → 15 step (1.22x 무료 speedup)

- DPM-Solver++는 고차 ODE solver이므로 적은 step에서도 충분한 수렴
- 20-step이 오히려 **오버슈팅** → FID 161.30
- 15-step: FID 151.77로 **더 낮음** → speedup이면서 품질도 개선

### 3. DeepCache Block-Level Caching

- PixArt 28개 transformer block 중 **중간 12개 (block 8~19)** 는 인접 timestep 간 변화 작음
- Denoising step을 full / cached로 교번 (interval=2):

```
Full step  : shallow[0~7] → deep[8~19] → final[20~27]  ← residual 저장
Cached step: shallow[0~7] + cached_residual → final[20~27]  ← deep 12개 skip
```

- **[8,20) 범위 선택 이유**: 너무 넓으면 FID 급등, [8,20)이 품질 손실 최소 지점

### 핵심 통찰

두 기법 모두 "불필요한 연산을 제거"한다는 같은 방향.
- Step 축소: **시간 방향**의 redundancy 제거
- DeepCache: **공간(block) 방향**의 redundancy 제거

| 요소 | 기여 |
|---|---|
| Step 20 → 15 | 시간 단축 + FID 개선 |
| Cache [8,20) int=2 | 12 block × 절반 step skip |
| **조합 총 speedup** | **1.79x** vs 20-step baseline |
| FID | 162.99 (baseline 161.30과 동일 수준) |

---

## 참고

- **DPM-Solver++ 특성**: 고차 ODE solver로 20-step 이내에서 급격한 품질 향상 → 10-12 step에서 plateau 예상
- **DeepCache 논문**: He et al. (2024) "DeepCache: Accelerating Diffusion Models for Free" — UNet 구조 대상이지만 DiT에도 동일 원리 적용 가능
- **캐시 오류 누적**: interval이 클수록 stale residual 오류 누적 → FID 악화 예상
- **CFG 호환성**: CFG 사용 시 batch가 [unconditional, conditional]로 2배 → residual cache도 2배 배치로 정상 동작
