# SVDQUANT Advanced Cache-LoRA 실험

## 실험 목적

이전 ablation에서 SVDQUANT + cache_lora (rank sweep, interval sweep)가 기대치를 충족하지 못함을 확인:
- rank 증가 (8→32): calib=4 고정 시 overfitting → IS 붕괴
- interval 증가 (3, 4): IS 붕괴
- NOLR (LoRA 제거된 SVDQUANT): 모델 완파

**근본 한계**: 단일 static corrector `(A, B)`가 모든 timestep/block에 동일하게 적용됨
→ SVDQUANT의 복잡한 activation 패턴 (내부 rank-32 SVD LR correction) 에 부적합

이를 해결하기 위해 4가지 방향의 개선된 corrector를 설계하고 검증:

| Direction | 방법 | 핵심 아이디어 |
|-----------|------|-------------|
| Dir 1 | Calibration 샘플 수 증가 | rank overfitting 완화 (calib × rank sweep) |
| Dir 2A | Phase-binned corrector | timestep 구간별 별도 (A, B) 학습 |
| Dir 2B | Timestep-conditional | 공유 (A, B) + per-step learned scale |
| Dir 3 | Block-specific corrector | 각 블록별 독립 corrector |
| Dir 4 | SVD-Aware corrector | SVDQUANT LR branch output을 추가 signal로 활용 |

---

## 모델 및 설정

- **모델**: PixArt-Sigma XL (28 transformer blocks, hidden_dim=1152)
- **양자화**: SVDQUANT (FP4 + internal rank-32 LR, internal lowrank=32)
- **Cache region**: blocks [8, 20), interval=2
- **Guidance scale**: 4.5
- **평가 데이터**: MJHQ-30K subset
- **샘플 수**: 20 samples (빠른 검증)

---

## Direction 1: Calibration 샘플 수 증가

### 동기

기존 `cache_lora`는 `lora_calib=4` (4개의 calibration 프롬프트). rank 증가 시 underdetermined system → 단일 calib sample에 overfitting.

### 방법

`--lora_calib` 증가: {8, 16, 32, 64} × rank={4, 16, 32, 64} × steps={10, 15, 20}

### 구현 변경

`pixart_nvfp4_cache_compare.py`:
```python
calib_tag = f"_cal{args.lora_calib}" if args.lora_calib != 4 else ""
cache_tag = f"cl_r{args.lora_rank}{calib_tag}"
# 예: SVDQUANT_cl_r16_cal32_c8-20_steps20
```

기존 `calibrate_cache_lora()` 함수 재사용, `num_calib` 인자만 변경.

### 실험 매트릭스

calib={8, 16, 32, 64} × rank={4, 16, 32, 64} × steps={10, 15, 20} = **48 runs**

---

## Direction 2A: Phase-Binned Corrector

### 동기

denoising trajectory에서 early/mid/late step은 서로 다른 정보를 처리함 (low→high frequency). 하나의 corrector가 전 구간에 적용되면 비효율적.

### 방법

steps를 3개 bin으로 분할, 각 bin마다 별도의 `(A, B)` 학습:

```
phase = min(step_idx * n_phases // t_count, n_phases - 1)
```

### 구현: `calibrate_cache_lora_phased()` (`deepcache_utils.py`)

- `C_phases = [zeros(H,H)] * 3` — phase별 cross-covariance 분리 축적
- 각 phase별 SVD → 3개 `(A_i, B_i)` 쌍 반환
- cached step에서 현재 `step_idx`에 해당하는 phase의 corrector 선택

### cache_mode: `cache_lora_phase`

### 실험 매트릭스

rank=4, calib=4 × steps={10, 15, 20} = **3 runs**

---

## Direction 2B: Timestep-Conditional (Per-Step Scale)

### 동기

Phase-binned보다 세밀한 temporal 적응: 공유 (A,B)를 학습한 뒤, 각 step마다 최적 scale을 least-squares로 결정.

### 방법

```
scale_s = <drift_s, B@A@dx_s> / ||B@A@dx_s||²
```

### 구현: `calibrate_cache_lora_timestep()` (`deepcache_utils.py`)

1단계: 기존과 동일하게 global `(A, B)` 학습
2단계: calibration buffer에서 per-step scale 계산 → `step_scales: [t_count]` tensor 반환

cached step에서:
```python
scale = state.step_scales[step_idx]
lora_correction = scale * F.linear(F.linear(dx, A), B)
```

scale은 `clamp(0.1, 10.0)` 적용.

### cache_mode: `cache_lora_ts`

### 실험 매트릭스

rank=4, calib=4 × steps={10, 15, 20} = **3 runs**

---

## Direction 3: Block-Specific Corrector

### 동기

기존: cache region [8, 20) 전체를 single corrector로 근사. 각 블록은 다른 구조/역할 → block-level granularity 필요.

### 방법

각 block `i ∈ [8, 20)` 에 독립적인 `(A_i, B_i)` 학습. 12개 block, 각 block 별 cross-covariance `C_i`.

### 구현: `calibrate_cache_lora_blockwise()` (`deepcache_utils.py`)

**메모리 효율화**: 12개 block을 동시 calibration하면 ~8.6 GB 필요. 대신 sequential per-block:

```python
for bi in range(n_blocks):
    block = transformer.transformer_blocks[cache_start + bi]
    # bi번 block에만 hook 등록, calibration 실행
    # 완료 후 hook 제거, 다음 block으로
```

**Full step**: per-block `h_in` 과 `residual` 저장:
```python
state.h_in_per_block = []
state.residual_per_block = []
for i, block in enumerate(self.transformer_blocks[cs:ce]):
    h_before = hidden_states.detach().clone()
    hidden_states = block(hidden_states, **block_kwargs)
    state.h_in_per_block.append(h_before)
    state.residual_per_block.append((hidden_states - h_before).detach())
```

**Cached step**: sequential per-block correction:
```python
for i, (A_i, B_i) in enumerate(state.block_correctors):
    dx_i = hidden_states - state.h_in_per_block[i]
    hidden_states = hidden_states + state.residual_per_block[i] + F.linear(F.linear(dx_i, A_i), B_i)
```

**메모리 추가 비용**: 12 × 2 × [2, 4096, 1152] @ fp16 ≈ 460 MB

### cache_mode: `cache_lora_block`

### 실험 매트릭스

rank=4, calib=4 × steps={10, 15, 20} = **3 runs**

---

## Direction 4: SVD-Aware Corrector

### 동기

SVDQUANT 내부 LR branch (`lora_b @ lora_a @ pre_scale(x)`) 는 FP4 양자화 오차를 직접 보정하는 rank-32 FP16 신호. 이를 corrector의 추가 input으로 활용하면 더 정확한 correction 가능.

### SVDQUANT 내부 구조

```
output = FP4(W_residual) @ quant(x)  +  lora_b @ lora_a @ pre_scale(x)
          └── 4-bit 주 경로               └── rank-32 FP16 LR branch
```

### Probe 설계

block[cache_start].attn1.to_out[0] (SVDQuantLinear, input/output 모두 1152)의 LR output을 probe로 사용:

```python
def _collect_block_lr_probe(block, hidden_states):
    to_out = block.attn1.to_out[0]
    if isinstance(to_out, SVDQuantLinear):
        x_scaled = to_out._apply_pre_quant_scale(hidden_states)
        return F.linear(F.linear(x_scaled, lora_a), lora_b)  # [B, T, H]
    return torch.zeros_like(hidden_states)
```

### 구현: `calibrate_cache_lora_svdaware()` (`deepcache_utils.py`)

Augmented input `[dx; lr_probe_delta]` (shape: [N, 2H]) 로 cross-covariance 계산:

```python
C = zeros(H, 2*H)  # [H, 2H]
augmented = cat([dx, lr_delta], dim=-1)
C += drift.T @ augmented
```

SVD → `A: [rank, 2H]`, `B: [H, rank]`

Cached step:
```python
lr_probe = _collect_block_lr_probe(self.transformer_blocks[cs], hidden_states)
lr_delta = lr_probe - state.lr_probe_cached
augmented = cat([dx, lr_delta], dim=-1)
correction = F.linear(F.linear(augmented, A), B)
```

### cache_mode: `cache_lora_svd`

### 실험 매트릭스

rank=4, calib=4 × steps={10, 15, 20} = **3 runs**

---

## 코드 구조 변경

### `deepcache_utils.py`

**`DeepCacheState` 추가 필드:**
```python
self.phase_correctors = None      # list[(A,B)] (Dir 2A)
self.phase_t_count = None
self.step_scales = None           # [t_count] (Dir 2B)
self.block_correctors = None      # list[(A_i,B_i)] (Dir 3/5)
self.residual_per_block = None    # reset per image (h_in_per_block 제거됨)
self.svd_corrector_A = None       # [rank, 2H] (Dir 4)
self.svd_corrector_B = None
self.lr_probe_cached = None       # reset per image
```

**새 함수 5개**: `calibrate_cache_lora_phased`, `calibrate_cache_lora_timestep`, `calibrate_cache_lora_blockwise`, `calibrate_cache_lora_svdaware`, `calibrate_cache_lora_teacherforced`

**`_make_cached_forward` full step 확장**: block_correctors 활성화 시 per-block residual 수집 (`h_in_per_block` 불필요)

**`_make_cached_forward` cached step dispatch**: tf → block → svd → phase → ts → standard 순서

### `pixart_nvfp4_cache_compare.py`

- `--cache_mode` choices: `["none", "deepcache", "cache_lora", "cache_lora_phase", "cache_lora_ts", "cache_lora_block", "cache_lora_svd", "cache_lora_tf"]`
- Phase 2.5: 6개 calibration 함수 dispatch
- Phase 3: state에 corrector 부착
- run_tag: calib 수 반영 (`_cal{N}` suffix)

### `run_svdquant_advanced.sh`

```bash
bash run_svdquant_advanced.sh --axis {phase|ts|block|svd|calib|nested|tf} \
    --gpu {0|1} --port {29500|29501} --num_samples 20
```

### `run_pipeline_parallel.sh`

2-GPU 병렬 파이프라인:
- Phase A: GPU0 (Dir5 + Dir9) ‖ GPU1 (Dir6 nested sweep)

---

## 전체 실험 매트릭스

| Direction | 조합 | runs |
|-----------|------|------|
| Dir 1: calib×rank sweep | calib={8,16,32,64} × rank={4,16,32,64} × steps={10,15,20} | 48 |
| Dir 2A: phase-binned | rank=4, calib=4 × steps={10,15,20} | 3 |
| Dir 2B: timestep-conditional | rank=4, calib=4 × steps={10,15,20} | 3 |
| Dir 3: block-specific (버그 존재) | rank=4, calib=4 × steps={10,15,20} | 3 |
| Dir 4: SVD-aware | rank=4, calib=4 × steps={10,15,20} | 3 |
| Dir 5: block-specific (버그 수정) | rank=4, calib=4 × steps={10,15,20} | 3 |
| Dir 6: nested caching | cs={10,12,14,16}, mode={deepcache,cache_lora} × steps={10,15,20} | 24 |
| Dir 9: teacher-forced calib | rank=4, calib=4 × steps={10,15,20} | 3 |
| **합계** | | **90 runs** |

모든 실험: SVDQUANT, interval=2, guidance=4.5, **20 samples**

---

## Direction 5: Block-Specific 버그 수정

### 버그 원인

Dir 3의 cached step에서 `dx_i = hidden_states - h_in_per_block[i]` 사용:
- `h_in_per_block[i]`는 full step 당시 block[i]의 입력
- cached step에서 `hidden_states`는 이전 block correction이 누적된 상태
- → `dx_i` 기준 불일치, 오차 폭발 (SSIM=0.021)

### 수정 (Method C)

```python
# cached step: global dx 한 번만 계산
dx = hidden_states - state.h_in_cached  # deep region 입구 기준
for i, (A_i, B_i) in enumerate(state.block_correctors):
    corr_i = F.linear(F.linear(dx, A_i), B_i)
    hidden_states = hidden_states + state.residual_per_block[i] + corr_i
```

- `h_in_per_block` 제거 → `residual_per_block`만 저장 (메모리 230MB → 절반)
- `needs_h_in` 조건에 `block_correctors` 추가 → `h_in_cached` 저장 보장

---

## Direction 6: Nested Caching (Deep Region 축소)

### 동기

SVDQUANT의 activation error는 block을 거치면서 누적. 전반부 deep blocks ([8,14))를 fresh 계산하고 후반부만 cache하면 오차 감소 가능.

### 방법

코드 변경 없음. `--cache_start` 조정:

| cache_start | cached blocks | n_deep | speedup |
|-------------|--------------|--------|---------|
| 8 (기존) | [8,20) | 12 | ~1.27x |
| 10 | [10,20) | 10 | ~1.22x |
| 12 | [12,20) | 8 | ~1.18x |
| 14 | [14,20) | 6 | ~1.13x |
| 16 | [16,20) | 4 | ~1.08x |

### 실험 매트릭스

cs={10,12,14,16} × mode={deepcache, cache_lora} × steps={10,15,20} = **24 runs**

---

## Direction 9: Teacher-Forced Calibration

### 동기

기존 calibration: fresh (no cache) inference에서 dx/drift 수집 → corrector 학습. Inference 시 cached step의 `hidden_states`는 stale trajectory → **분포 불일치**.

Teacher-forced: DeepCache ON 상태에서 calibration → 실제 inference-time 분포 학습.

### 알고리즘

```python
# 1. 임시 DeepCache 설치 (no LoRA)
tf_state = install_deepcache(transformer, ...)
tf_state._tf_calibration_mode = True

# 2. 각 cached step에서:
stale_dx = hidden_states - h_in_cached      # 실제 inference-time input
# shadow fresh forward:
h_fresh = deep_blocks(hidden_states)
fresh_residual = h_fresh - hidden_states
drift = fresh_residual - deep_residual_cache  # correction 목표

# 3. Cross-covariance: C = Σ drift.T @ stale_dx (fresh dx가 아닌 stale dx!)
# 4. SVD → (A, B)
# 5. DeepCache 해제, orig forward 복원
```

### cache_mode: `cache_lora_tf`

### 새 함수: `calibrate_cache_lora_teacherforced()` (`deepcache_utils.py`)
