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

| 순위 | 방법 | FID↓ | CLIP↑ | tpi(s)↓ | speedup↑ |
|------|------|------|-------|---------|---------|
| 1 | SVDQUANT (no cache) | 121.32 | 34.840 | 2.85 | 1.00× |
| 2 | **nl_gelu drift** | **124.94** | 34.823 | 2.53 | 1.13× |
| 3 | **selective K=6** | **126.04** | 34.801 | 3.05 | 1.12× |
| 4 | CacheLoRA r4 c8-20 | 126.56 | 34.869 | 2.33 | 1.22× |
| 5 | nl_gelu r8 (ablation) | 126.38 | 34.795 | — | — |
| 6 | nl_gelu cal8 (ablation) | 126.72 | 34.907 | — | — |
| 7 | nl_gelu_t traj_distill | 127.05 | 34.655 | 2.26 | — |
| 8 | nl_gelu drift_traj | 128.15 | 34.863 | 2.31 | — |
| 9 | nl_gelu_t | 128.22 | 34.800 | 2.24 | — |
| 10 | **partial_attn_mlp** | 128.72 | 34.639 | 2.60 | 1.10× |
| 11 | nl_gelu c12-20 (ablation) | 128.48 | 34.797 | — | — |
| 12 | selective K=12 | 129.23 | 34.684 | 2.25 | **1.27×** |
| 13 | DeepCache c8-20 | 129.14 | 34.742 | 2.33 | 1.22× |
| 14 | selective K=8 | 131.01 | 34.680 | 2.88 | 1.17× |
| 15 | selective K=10 | 131.61 | 34.686 | 2.36 | 1.22× |
| 16 | **selective K=6 + nl_gelu corrector** | **125.88** | 34.718 | 2.97 | 1.12× |
| 17 | selective+partial attn_mlp K=8 | 126.86 | 34.687 | 2.70 | 1.17× |
| 18 | selective+partial attn_mlp K=6 | 127.22 | 34.708 | 2.68 | 1.12× |
| 19 | selective K=8 + nl_gelu corrector | 129.96 | 34.680 | 2.65 | 1.17× |
| 20 | selective+partial attn K=8 | 130.80 | 34.785 | 2.73 | 1.17× |
| 21 | selective+partial attn K=6 | 132.53 | 34.601 | 3.41 | 1.12× |
| 22 | partial_attn | 135.60 | 34.623 | 2.71 | 1.05× |
| 23 | selective K=14 | 143.28 | 34.555 | 2.20 | 1.33× |
| 24 | selective K=16 | 143.28 | 34.587 | 2.02 | 1.40× |
| — | partial_mlp | **361.06** | 27.302 | 2.82 | 1.01× | (붕괴) |

> **nl_gelu drift (기본 설정)**이 캐싱 방법 중 최고. 속도(tpi=2.53s)는 no-cache(2.85s) 대비 1.13× 빠름.  
> **selective K=6** (corrector 없음)이 3위: FID=126.04로 DeepCache(-3.1pt) 개선. Skip blocks: {5,19,21,23,24,25}.  
> selective K=12는 DeepCache contiguous와 FID 동일 수준이지만 speedup 1.27×로 더 빠름.  
> `partial_attn_mlp`는 corrector 없이 DeepCache(129.14)보다 FID 0.4pt 개선 (attn2 재계산 효과).  
> **selective+partial attn_mlp K=8**: FID=126.86, speedup 1.17× — selective full-block K=6(FID=126.04)와 거의 동급 FID에서 더 빠른 speedup 달성.  
> **selective K=6 + nl_gelu corrector**: FID=125.88 — corrector 추가로 -0.16pt 개선에 그침. no-cache(121.32) 대비 여전히 4.5pt 격차.  
> `partial_mlp`는 ff 입력 분포 변화로 완전 붕괴.

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

## 12. Partial Block Skip (sub-module 단위 skip) ← 이번 실험

블록 전체를 skip하는 DeepCache 대신 블록 내부의 sub-module (attn1 self-attn 또는 ff MLP)만 캐시하고 나머지는 매 step 재계산하는 방법.

### 실험 설계

| Mode | Cache 대상 | 매 step 재계산 |
|------|-----------|---------------|
| `partial_attn` | attn1 pre-gate output | adaLN, norm1/2, **attn2**, ff, gates |
| `partial_mlp` | ff pre-gate output | adaLN, norm1/2, attn1, attn2, gates |
| `partial_attn_mlp` | attn1 **AND** ff pre-gate outputs | adaLN, norm1/2, **attn2**, gates |

`attn2` (cross-attn)는 세 모드 모두에서 항상 재계산. Cache range [8,20), interval=2.

### 결과 (steps=20, n=100)

| 방법 | FID↓ | CLIP↑ | tpi(s)↓ | speedup↑ | 비고 |
|------|------|-------|---------|---------|------|
| SVDQUANT (no cache) | 121.32 | 34.840 | 2.85 | 1.00× | 기준 |
| DeepCache c8-20 (full skip) | 129.14 | 34.742 | 2.33 | 1.22× | 비교 |
| nl_gelu drift | 124.94 | 34.823 | 2.53 | 1.13× | 비교 |
| `partial_attn` | 135.60 | 34.623 | 2.71 | 1.05× | full skip보다 나쁨 |
| `partial_mlp` | 361.06 | 27.302 | 2.82 | 1.01× | **붕괴** |
| `partial_attn_mlp` | **128.72** | 34.639 | 2.60 | 1.10× | full skip 대비 미미한 개선 |

### 분석

- **partial_mlp 붕괴**: ff 입력 `norm2(hidden_states)`가 self-attn/cross-attn 누적 후 크게 변함. Stale pre-gate ff output을 재사용하면 현재 step 분포와 완전 불일치 → FID=361로 발산.
- **partial_attn 악화**: Full block skip(129.14)보다 FID가 오히려 나쁨(135.6). Stale attn1 + fresh attn2+ff의 혼합이 full-block stale reuse보다 내부 일관성이 낮음.
- **partial_attn_mlp 미미한 개선**: attn2 재계산으로 FID 0.4pt 개선에 그침 (129.14 → 128.72). Cross-attn은 block FLOPs의 ~3%에 불과.

### 결론

Pre-gate submodule caching 전략은 효과적이지 않음. DeepCache의 강점은 전체 block residual을 하나의 단위로 재사용하는 내부 일관성에 있으며, 이를 sub-module 수준으로 분해하면 일관성이 깨져 품질이 오히려 나빠짐.

### 결과 위치

```
/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/SVDQUANT/MJHQ/
├── SVDQUANT_partial_attn_c8-20_steps20/
├── SVDQUANT_partial_mlp_c8-20_steps20/
└── SVDQUANT_partial_attn_mlp_c8-20_steps20/
```

---

## 11. 결론 및 향후 방향

**현재 최고 설정**: `nl_gelu drift`, rank=4, mid=32, c8-20, steps=20  
→ FID=124.94, CLIP=34.823, speedup=1.13×, no-cache 대비 FID +3.6만 손해

**시도했으나 효과 없는 방향**:
- Nonlinear 아키텍처 변형 (mlp/res/film): 단순 GELU보다 나쁨
- Timestep conditioning (gelu_t): FID 악화
- Feature distillation 계열 (fd, fd_weighted, fd_stratified): 붕괴 또는 악화
- Trajectory distillation (traj_distill, drift_traj): 미세 개선 또는 악화
- **Sub-module 단위 partial skip** (partial_attn/mlp/attn_mlp): 모두 full block skip 대비 개선 미미하거나 오히려 악화. DeepCache의 내부 일관성이 핵심임을 확인.

**탐색 가능한 방향**:
- 더 많은 calibration 프롬프트 (현재 4개)로 trajectory loss 품질 향상
- Corrector를 steps=20에 맞게 overfitting 허용 (per-step corrector)
- nl_gelu drift를 steps=15 이하에서 개선 (현재 steps=15 FID=131.6으로 DeepCache=130.2와 유사)
- Cache 구간 동적 선택 (block별 중요도 기반) ← **selective skip으로 완료**
- partial_attn_mlp (FID=128.72)에 nl_gelu drift corrector 조합 시도 (attn2 재계산 + corrector)
- selective K=6 + nl_gelu drift corrector 조합 (corrector-free K=6 FID=126.04 대비 개선 기대)

---

## 13. Sensitivity-based Selective Skip ← 이번 실험

Sensitivity ranking으로 skip-safe 하위 K개 블록만 선택적으로 skip. Cache range [8,20)이 아닌 전체 [0,28) 중 선택.

### Block Sensitivity Ranking (Metric A, low → high)

```
[19, 5, 25, 21, 24, 23, 14, 13, 10, 18, 22, 12, 26, 9, 11, 3, 16, 17, 0, 15, 6, 8, 4, 20, 7, 2, 1, 27]
```

Skip-safe 블록: 주로 네트워크 후반부 (18-26)와 중반부 일부 (5, 10, 13, 14)  
가장 민감한 블록: 27 (MSE=0.41, 이상값), 1, 2, 7, 20

### K-sweep 결과 (steps=20, n=100)

| K | FID↓ | CLIP↑ | tpi(s)↓ | speedup↑ | skip blocks |
|---|------|-------|---------|---------|------------|
| 6 | **126.04** | 34.801 | 3.05 | 1.12× | [5,19,21,23,24,25] |
| 8 | 131.01 | 34.680 | 2.88 | 1.17× | [5,13,14,19,21,23,24,25] |
| 10 | 131.61 | 34.686 | 2.36 | 1.22× | [5,10,13,14,18,19,21,23,24,25] |
| 12 | 129.23 | 34.684 | 2.25 | **1.27×** | [5,10,12,13,14,18,19,21,22,23,24,25] |
| 14 | 143.28 | 34.555 | 2.20 | 1.33× | [5,9,10,12,13,14,18,19,21,22,23,24,25,26] |
| 16 | 143.28 | 34.587 | 2.02 | 1.40× | [3,5,9,10,11,12,13,14,18,19,21,22,23,24,25,26] |
| (비교) DeepCache c8-20 | 129.14 | 34.742 | 2.33 | 1.22× | [8..19] |

### 분석

- **K=6 sweet spot**: corrector 없이 FID=126.04 (DeepCache -3.1pt). 선택된 블록이 후반부에 집중.
- **K=12 selective ≈ DeepCache contiguous**: FID 차이 0.09pt (noise 범위), speedup 1.27× vs 1.22×. 어느 블록을 skip하든 12개 skip 시 품질 차이 미미.
- **K=8, 10 비단조 FID**: FID noise (n=100에서 ±3-5pt)로 K=8>K=10>K=12 순서가 역전. 통계적으로 무의미한 차이.
- **K=14+ 급락**: 민감한 블록들이 포함되기 시작 → FID 143으로 급락.

### 결론

핵심 발견: **어떤 블록을 skip하느냐보다 몇 개를 skip하느냐**가 FID 결정에 더 중요. K=6 selective skip은 corrector 없이 우수한 FID-speedup trade-off를 보이며, 향후 corrector 조합 시 추가 개선 가능.

### 결과 위치

```
/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/SVDQUANT/MJHQ/
├── sensitivity/SVDQUANT_steps20_cal4_seed1000.json
├── SVDQUANT_selective_k6_mA_c8-20_steps20/
├── SVDQUANT_selective_k8_mA_c8-20_steps20/
├── SVDQUANT_selective_k10_mA_c8-20_steps20/
├── SVDQUANT_selective_k12_mA_c8-20_steps20/
├── SVDQUANT_selective_k14_mA_c8-20_steps20/
└── SVDQUANT_selective_k16_mA_c8-20_steps20/
```

---

## 14. Selective + Partial Block Skip (steps=20, n=100)

sensitivity ranking 하위 K개 블록에 partial skip(attn2 항상 fresh) 적용. selective full-block skip과 달리 블록 내부에서 attn2/adaLN은 매 step fresh.

**방법론**: `install_selective_partial_skip` — Metric A ranking 기반 K개 블록에만 `_make_partial_block_forward` 적용.

### K-sweep 결과

| mode | K | FID↓ | CLIP↑ | tpi(s)↓ | speedup↑ | skip blocks |
|------|---|------|-------|---------|---------|------------|
| (비교) selective full-block | 6 | 126.04 | 34.801 | 3.05 | 1.12× | [5,19,21,23,24,25] |
| **selective_partial_attn_mlp** | **8** | **126.86** | 34.687 | **2.70** | **1.17×** | [5,13,14,19,21,23,24,25] |
| selective_partial_attn_mlp | 6 | 127.22 | 34.708 | 2.68 | 1.12× | [5,19,21,23,24,25] |
| selective_partial_attn | 8 | 130.80 | 34.785 | 2.73 | 1.17× | [5,13,14,19,21,23,24,25] |
| selective_partial_attn | 6 | 132.53 | 34.601 | 3.41 | 1.12× | [5,19,21,23,24,25] |
| (비교) no-cache | — | 121.32 | 34.840 | 2.85 | 1.00× | — |

### 분석

- **attn_mlp K=8 best**: FID=126.86, speedup=1.17× — selective full-block K=6(FID=126.04)와 FID 거의 동등(+0.82pt)하면서 speedup은 1.17×으로 더 높음. K=8 블록에 attn1+ff를 캐시하되 attn2는 fresh 유지.
- **selective_partial_attn 악화**: attn만 캐시하면 full-block skip보다 오히려 FID 나쁨. stale attn1 + fresh attn2 혼재가 내부 일관성을 떨어뜨림. 이전 contiguous partial_attn 결과(FID=135.6)와 동일한 패턴.
- **attn_mlp > attn**: ff도 함께 캐시(attn_mlp)하면 attn1 stale에 의한 ff 입력 불일치가 줄어들어 일관성 개선.
- **speedup 한계**: partial 모드는 블록 전체가 아닌 sub-module만 skip하므로 실제 tpi 개선이 full-block 대비 작음. attn_mlp K=6 tpi=2.68s < K=6 full-block tpi=3.05s (오버헤드 감소).

### 결론

selective+partial 조합으로 SVDQUANT no-cache(FID=121.32) 달성은 어려움. **attn_mlp K=8이 FID-speedup 트레이드오프에서 selective full-block K=6와 동등한 FID에 더 높은 speedup** 제공. corrector 없이 caching quality ceiling은 FID≈126-127 수준.

### 결과 위치

```
/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/SVDQUANT/MJHQ/
├── SVDQUANT_selective_partial_attn_k6_mA_c8-20_steps20/
├── SVDQUANT_selective_partial_attn_k8_mA_c8-20_steps20/
├── SVDQUANT_selective_partial_attn_mlp_k6_mA_c8-20_steps20/
└── SVDQUANT_selective_partial_attn_mlp_k8_mA_c8-20_steps20/
```

---

## 15. Selective Skip + nl_gelu Corrector (steps=20, n=100)

sensitivity ranking 하위 K개 블록 skip + nl_gelu drift corrector 조합. corrector는 skip 블록 span [min_skip, max_skip+1) 기준으로 재calibrate.

**corrector calibration 방식**:
- K=6 skip_blocks={5,19,21,23,24,25} → 훈련 구간 [5, 26)
- K=8 skip_blocks={5,13,14,19,21,23,24,25} → 훈련 구간 [5, 26)
- 4 calib prompts, drift loss, rank=4, mid=32, ~53초
- `dx = h_in(t, block_5) - h_in(t-2, block_5)` → corrector → block_25 출력에 correction 적용

**적용 방식**: 첫 skip block(5)에서 fresh step마다 h_in 저장, cached step에서 dx 계산 → 마지막 skip block(25) 이후 correction 적용.

### 결과 (steps=20, n=100)

| 방법 | FID↓ | CLIP↑ | tpi(s)↓ | speedup↑ |
|------|------|-------|---------|---------|
| SVDQUANT no-cache | **121.32** | 34.840 | 2.85 | 1.00× |
| DeepCache + nl_gelu (기준) | 124.94 | 34.823 | 2.53 | **1.27×** |
| **Selective K=6 + nl_gelu** | **125.88** | 34.718 | 2.97 | 1.12× |
| Selective K=6 (no corrector) | 126.04 | 34.801 | 3.05 | 1.12× |
| Selective K=8 + nl_gelu | 129.96 | 34.680 | 2.65 | 1.17× |
| Selective K=8 (no corrector) | 131.01 | 34.680 | 2.88 | 1.17× |

### 분석

- **K=6 corrector 효과 미미**: FID 126.04 → 125.88 (-0.16pt). no-cache(121.32) 대비 여전히 4.5pt 격차. DeepCache + nl_gelu(124.94)에도 못 미침.
- **K=8 corrector 악화**: corrector 없는 K=8(131.01)보다 오히려 나쁜 129.96. 더 많은 블록의 비연속 drift를 작은 MLP(18K params)로 보정하기엔 한계.
- **DeepCache + nl_gelu가 여전히 최선**: contiguous [8,20) 구간의 단순하고 일관된 drift 패턴이 corrector 학습에 유리. selective skip의 비연속 drift는 패턴이 복잡해 corrector 일반화 어려움.
- **caching quality ceiling 확인**: corrector 추가와 무관하게 ~124-126 수준이 skip-based 방법의 실질적 한계. no-cache FID 달성은 더 강력한 corrector(예: 블록별 corrector, timestep-aware 등) 필요.

### Corrector 가중치 위치

```
/data/jameskimh/james_dit_pixart_sigma_xl_mjhq_cache_adapter/SVDQUANT/
├── nl_gelu_selective_k6_mA_cs5_ce26_steps20_cal4_seed1000.pt
└── nl_gelu_selective_k8_mA_cs5_ce26_steps20_cal4_seed1000.pt
```

### 결과 위치

```
/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/SVDQUANT/MJHQ/
├── SVDQUANT_selective_nl_gelu_k6_mA_c8-20_steps20/
└── SVDQUANT_selective_nl_gelu_k8_mA_c8-20_steps20/
```
