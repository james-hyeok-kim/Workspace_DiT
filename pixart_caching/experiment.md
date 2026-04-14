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

### 결과 테이블 (실험 후 기입)

| Steps | FID ↓ | IS ↑ | PSNR ↑ | SSIM ↑ | CLIP ↑ | Time/img (s) | Speedup |
|-------|--------|------|--------|--------|--------|--------------|---------|
| 20    |        |      |        |        |        |              | 1.00x   |
| 15    |        |      |        |        |        |              |         |
| 12    |        |      |        |        |        |              |         |
| 10    |        |      |        |        |        |              |         |
| 8     |        |      |        |        |        |              |         |
| 5     |        |      |        |        |        |              |         |

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

### 결과 테이블 (실험 후 기입)

| interval | blocks     | FID ↓ | IS ↑ | PSNR ↑ | Time/img (s) | Speedup est. |
|----------|------------|--------|------|--------|--------------|--------------|
| 1 (base) | [4, 24)    |        |      |        |              | 1.00x        |
| 2        | [4, 24)    |        |      |        |              | 1.56x        |
| 3        | [4, 24)    |        |      |        |              | 1.84x        |
| 4        | [4, 24)    |        |      |        |              | 1.94x        |
| 2        | [2, 26)    |        |      |        |              | 1.75x        |
| 2        | [8, 20)    |        |      |        |              | 1.30x        |
| 2        | [4, 28)    |        |      |        |              | 1.75x        |

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

### 결과 테이블 (실험 후 기입)

| 방법 | Steps | Cache | FID ↓ | IS ↑ | PSNR ↑ | Time/img (s) | Total Speedup |
|------|-------|-------|--------|------|--------|--------------|----------------|
| NVFP4 baseline | 20 | None | ~194 | | | | 1.00x |
| Step reduction only | ? | None | | | | | |
| Cache only (20-step) | 20 | int=2 | | | | | |
| Step + Cache | ? | int=2 | | | | | |

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

## 참고

- **DPM-Solver++ 특성**: 고차 ODE solver로 20-step 이내에서 급격한 품질 향상 → 10-12 step에서 plateau 예상
- **DeepCache 논문**: He et al. (2024) "DeepCache: Accelerating Diffusion Models for Free" — UNet 구조 대상이지만 DiT에도 동일 원리 적용 가능
- **캐시 오류 누적**: interval이 클수록 stale residual 오류 누적 → FID 악화 예상
- **CFG 호환성**: CFG 사용 시 batch가 [unconditional, conditional]로 2배 → residual cache도 2배 배치로 정상 동작
