# CacheLoRA-NL 실험 결과 정리

**모델**: PixArt-Sigma-XL (1024px, 28 transformer blocks)  
**양자화**: SVDQUANT (4-bit)  
**평가 데이터셋**: MJHQ (n=100, 단 n=20 표기 시 예비 결과)  
**캐시 구간**: blocks [8, 20), interval=2 (기본)  
**기준일**: 2026-04-23

---

## 1. 개요: 방법론 계층 구조

```
SVDQUANT (no cache)
  └── DeepCache            : 캐시된 스텝에서 blocks 8-20 스킵, stale residual 재사용
        └── CacheLoRA       : stale residual을 LoRA corrector로 보정 (linear)
              └── NL-GELU   : GELU bottleneck nonlinear corrector (drift loss)
                    ├── 아키텍처 변형: mlp, res, film
                    ├── 아키텍처 변형: gelu_t (timestep conditioning)
                    └── Loss 변형: fd, fd_weighted, fd_stratified, traj_distill, drift_traj
```

---

## 2. Baseline 결과 (n=100)

| 방법 | steps | FID↓ | CLIP↑ | tpi(s)↓ | speedup↑ |
|------|-------|------|-------|---------|---------|
| SVDQUANT (no cache) | 20 | 121.32 | 34.840 | 2.85 | 1.00× |
| SVDQUANT (no cache) | 15 | 150.32 | 35.226 | 2.38 | 1.00× |
| SVDQUANT (no cache) | 10 | 121.90 | 34.805 | 1.44 | 1.00× |
| DeepCache c8-20 | 20 | 129.14 | 34.742 | 2.33 | **1.22×** |
| DeepCache c8-20 | 15 | 130.20 | 34.784 | 1.87 | **1.27×** |
| DeepCache c8-20 | 10 | 140.47 | 34.760 | 1.29 | **1.12×** |
| CacheLoRA r4 c8-20 | 20 | 126.56 | 34.869 | 2.33 | **1.22×** |
| CacheLoRA r4 c8-20 | 15 | 136.63 | 34.944 | 1.84 | **1.29×** |
| CacheLoRA r4 c8-20 | 10 | 228.94 | 31.536 | 1.23 | **1.17×** |

> `tpi`: time per image (sec), `speedup` = no-cache tpi / method tpi

---

## 3. Nonlinear Corrector — 아키텍처 비교 (steps=20, n=20)

기본 설정: rank=4, mid_dim=32, loss=drift, calib=4

| 아키텍처 | FID↓ | CLIP↑ | 설명 |
|----------|------|-------|------|
| `nl_gelu` | **124.94** | 34.823 | GELU bottleneck: B(GELU(A(dx))) |
| `nl_mlp`  | 162.92 | 34.791 | 2-layer MLP, no residual |
| `nl_res`  | 161.27 | 35.176 | Residual MLP |
| `nl_film` | 161.45 | 34.799 | FiLM-conditioned MLP |

> `nl_gelu`가 모든 변형 중 최고. mlp/res/film은 DeepCache 수준도 못 미침.

---

## 4. nl_gelu (drift) — 주요 실험 결과 (n=100)

### 4.1 Steps별 성능

| steps | FID↓ | CLIP↑ | tpi(s)↓ | speedup↑ |
|-------|------|-------|---------|---------|
| 20 | **124.94** | 34.823 | 2.53 | **1.13×** |
| 15 | 131.62 | 34.899 | 1.86 | 1.28× |
| 10 | 163.89 | 34.740 | 1.20 | 1.20× |

> steps=20에서 DeepCache(129.14) 대비 FID -4.2 개선. CacheLoRA(126.56) 대비도 FID -1.6 개선.

### 4.2 Ablation: rank, calib, cache 구간 (steps=20, n=100)

| 설정 | FID↓ | CLIP↑ | 비고 |
|------|------|-------|------|
| r4, cal4, c8-20 (기본) | 124.94 | 34.823 | 기본값 |
| **r8**, cal4, c8-20 | **126.38** | 34.795 | rank 2배 → 소폭 개선 없음 |
| r4, **cal8**, c8-20 | 126.72 | **34.907** | calib 2배 → CLIP 미세 개선 |
| r4, cal4, **c12-20** | 128.48 | 34.797 | 캐시 구간 축소 → 악화 |

> rank·calib 증가가 뚜렷한 개선을 주지 않음. c8-20 기본 구간이 최적.

---

## 5. Timestep-aware Corrector: nl_gelu_t (steps=20, n=100)

`GELUBottleneckT`: FiLM-style timestep conditioning 추가.

```python
h = self.A(dx)
scale = self.scale_net(t_norm)   # t_norm ∈ [0,1]
shift = self.shift_net(t_norm)
h = scale * F.gelu(h) + shift
return self.B(h)
```

| 방법 | FID↓ | CLIP↑ | tpi(s)↓ |
|------|------|-------|---------|
| nl_gelu (drift) | 124.94 | 34.823 | 2.53 |
| **nl_gelu_t** (drift) | 128.22 | 34.800 | 2.24 |

> t-conditioning이 FID를 오히려 악화 (+3.3). 단순한 drift correction에는 timestep 정보가 noise로 작용.

---

## 6. Loss 변형 실험 (steps=20, n=100)

모든 실험 공통: SVDQUANT, cache_nl_gelu (혹은 cache_nl_gelu_t), r4, m32, c8-20

### 6.1 Feature Distillation (fd, fd_weighted)

```
drift loss  : target = (h_out_curr - h_in_curr) - (h_out_prev - h_in_prev)
fd loss     : target = h_out_curr - h_in_curr   (fresh residual 직접 예측)
fd_weighted : fd loss에 token-variance 가중치 적용
```

| 방법 | FID↓ | CLIP↑ | 결과 |
|------|------|-------|------|
| nl_gelu, drift (기준) | 124.94 | 34.823 | — |
| nl_gelu, fd_weighted, steps=20 | 347.46 | 25.732 | **붕괴** |
| nl_gelu, fd_weighted, steps=15 | 375.88 | 21.505 | **붕괴** |

> fd_weighted는 inference 시 stale_res를 더하지 않는 fd 모드와 stale_res를 더하는 일반 모드의 불일치로 추정됨 — 학습 loss는 h_in + correction 기준이지만 실제 적용 경로가 달라 발산.

### 6.2 fd_stratified (timestep-stratified feature distillation)

5-bucket timestep stratification으로 gradient 균등화.

| 방법 | steps | FID↓ | CLIP↑ | 비고 |
|------|-------|------|-------|------|
| nl_gelu_t, fd_stratified | 20 | 308.59 | 29.238 | **붕괴** |
| nl_gelu_t, fd_stratified | 15 | 313.72 | 28.643 | **붕괴** |

> DeepCache 미적용 버그(gelu_t가 `_cache_modes_with_dc` 누락) 수정 후에도 개선 없음. fd 계열 loss 자체가 현재 corrector 구조와 궁합이 나쁜 것으로 보임.

### 6.3 Trajectory Distillation (traj_distill)

K=6 step window rollout으로 teacher(FP16) trajectory를 모방.

```
total_loss = traj_MSE(student_latent[start:start+K], teacher_latent[start:start+K])
```

| 방법 | FID↓ | CLIP↑ | tpi(s)↓ |
|------|------|-------|---------|
| nl_gelu_t, drift (기준) | 128.22 | 34.800 | 2.24 |
| **nl_gelu_t, traj_distill** | **127.05** | 34.655 | 2.26 |

> FID 소폭 개선 (+1.2 vs gelu_t 기준) 하지만 CLIP 하락. 기본 drift(124.94) 대비로는 여전히 열위.

### 6.4 Combined Drift + Trajectory (drift_traj) ← **이번 세션**

Phase 2.5에서 drift corrector 로드/학습 후, Phase 2.75에서 combined fine-tuning 수행.

```
total_loss = feat_loss + λ × traj_loss
feat_loss  = MSE(corrector(dx), target_drift)   # 589k 오프라인 쌍
traj_loss  = MSE(student[start:start+K], teacher[start:start+K])
λ warm-up  : 0 → 0.1 (40 iter 선형 증가)
```

설정: K=6, λ=0.1, 200 iters, 4 calib prompts, lr=3e-4

| 방법 | FID↓ | CLIP↑ | tpi(s)↓ |
|------|------|-------|---------|
| nl_gelu, drift (기준) | **124.94** | 34.823 | 2.53 |
| **nl_gelu, drift_traj** | 128.15 | **34.863** | 2.31 |

> FID +3.2 (악화), CLIP +0.04 (미세 개선). 200 iter / 4 prompt traj rollout 신호가 너무 noisy하여 잘 학습된 drift corrector를 오히려 방해.

---

## 7. 전체 결과 순위 (steps=20, n=100)

| 순위 | 방법 | FID↓ | CLIP↑ |
|------|------|------|-------|
| 1 | SVDQUANT (no cache) | 121.32 | 34.840 |
| 2 | **nl_gelu drift** | **124.94** | 34.823 |
| 3 | CacheLoRA r4 c8-20 | 126.56 | 34.869 |
| 4 | nl_gelu r8 (ablation) | 126.38 | 34.795 |
| 5 | nl_gelu cal8 (ablation) | 126.72 | 34.907 |
| 6 | nl_gelu_t traj_distill | 127.05 | 34.655 |
| 7 | nl_gelu_t | 128.22 | 34.800 |
| 8 | nl_gelu drift_traj | 128.15 | 34.863 |
| 9 | nl_gelu c12-20 (ablation) | 128.48 | 34.797 |
| 10 | DeepCache c8-20 | 129.14 | 34.742 |

> **nl_gelu drift (기본 설정)**이 캐싱 방법 중 최고. 속도(tpi=2.53s)는 no-cache(2.85s) 대비 1.13× 빠름.

---

## 8. Cache 구간 비교 (n=100)

`CacheLoRA r4` 기준으로 여러 구간 테스트:

| 구간 | steps=20 FID | steps=15 FID | steps=10 FID |
|------|-------------|-------------|-------------|
| c2-26 | 147.42 | 140.11 | 256.99 |
| c4-24 | 132.47 | 141.69 | 287.51 |
| **c8-20** | **126.56** | **136.63** | 228.94 |

> c8-20이 최적 구간. 너무 넓게 (c2-26) 잡으면 FID 악화.

---

## 9. Corrector 저장 위치

```
/data/jameskimh/james_dit_pixart_sigma_xl_mjhq_cache_adapter/SVDQUANT/
├── nl_gelu_r4_m32_cs8_ce20_steps{N}_cal4_seed1000.pt        # nl_gelu drift
├── nl_gelu_r8_m32_cs8_ce20_steps20_cal4_seed1000.pt          # r8 ablation
├── nl_gelu_r4_m32_cs8_ce20_steps20_cal8_seed1000.pt          # cal8 ablation
├── nl_gelu_t_r4_m32_cs8_ce20_steps20_cal4_seed1000.pt        # gelu_t
├── nl_gelu_t_trajdistill_r4_m32_cs8_ce20_steps20_cal4_seed1000.pt
├── nl_gelu_t_fdstratified_r4_m32_cs8_ce20_steps{N}_cal4_seed1000.pt
└── nl_gelu_drifttraj_r4_m32_cs8_ce20_steps20_cal4_seed1000.pt  # drift_traj
```

---

## 10. 주요 버그 수정 이력

| 날짜 | 파일 | 내용 |
|------|------|------|
| 2026-04-22 | `pixart_nvfp4_cache_compare.py` | `cache_nl_gelu_t`를 `_cache_modes_with_dc`에 추가 (DeepCache 미적용 버그) |
| 2026-04-23 | `deepcache_utils.py` L385-394 | `dx.half()` → `dx.to(cdtype)` (train mode 시 dtype mismatch 크래시) |
| 2026-04-23 | `pixart_nvfp4_cache_compare.py` | `--test_run` 시 combined `.pt` 저장 금지 (cache poisoning 버그) |

---

## 11. 결론 및 향후 방향

**현재 최고 설정**: `nl_gelu drift`, rank=4, mid=32, c8-20, steps=20  
→ FID=124.94, CLIP=34.823, speedup=1.13×, no-cache 대비 FID +3.6만 손해

**시도했으나 효과 없는 방향**:
- Nonlinear 아키텍처 변형 (mlp/res/film): 단순 GELU보다 나쁨
- Timestep conditioning (gelu_t): FID 악화
- Feature distillation 계열 (fd, fd_weighted, fd_stratified): 붕괴 또는 악화
- Trajectory distillation (traj_distill, drift_traj): 미세 개선 또는 악화

**탐색 가능한 방향**:
- 더 많은 calibration 프롬프트 (현재 4개)로 trajectory loss 품질 향상
- Corrector를 steps=20에 맞게 overfitting 허용 (per-step corrector)
- nl_gelu drift를 steps=15 이하에서 개선 (현재 steps=15 FID=131.6으로 DeepCache=130.2와 유사)
- Cache 구간 동적 선택 (block별 중요도 기반)
