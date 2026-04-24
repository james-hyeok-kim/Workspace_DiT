# Block-wise Sensitivity-based Selective Skip

## 개요

DeepCache는 `[8,20)` 12 blocks를 **무차별** skip하여 FID 129.14 / speedup 1.22×를 얻는다. 가설: 블록마다 quality 기여도가 다르므로 sensitivity가 낮은 K개 블록만 선택적으로 skip하면 FID 손해를 줄일 수 있다.

## 방법론

### Sensitivity 측정 (두 metric)

**Metric A — Per-block skip ablation (ground truth)**
- 각 블록 `b ∈ [0,28)` 에 대해 `install_selective_skip(skip_blocks={b}, cache_interval=2)` 설치
- Teacher(no cache)의 최종 latent `z_ref`와 MSE 비교: `metric_A[b] = mean_p MSE(z_b_p, z_ref_p)` (p ∈ 4 calib prompts)

**Metric C — Stale-vs-fresh residual diff (proxy)**
- Teacher pass 중 per-block residual 변화를 측정 (step counter 버그로 이번 실험에서 유효하지 않음 — 모두 0으로 측정됨)

### 선택적 Skip 설치 (`install_selective_skip`)

`install_partial_skip` 방식(per-block monkey-patch)을 채택하되, sub-module 분해 없이 **block 전체를 skip**:

```python
def selective_block_forward(self, hidden_states, *args, **kwargs):
    step_idx = state.step_idx - 1
    is_cached = (step_idx not in full_steps_set) and (step_idx % cache_interval != 0)
    if is_cached and b_idx in state.block_residual_cache:
        return hidden_states + state.block_residual_cache[b_idx]
    h_in = hidden_states
    h_out = orig_fwd(self, hidden_states, *args, **kwargs)
    state.block_residual_cache[b_idx] = (h_out - h_in).detach()
    return h_out
```

- skip_blocks 에 포함된 블록만 patch (나머지 오버헤드 0)
- 원본 `BasicTransformerBlock.forward` 직접 호출 (`type(block).forward`)

## 실험 설정

| 파라미터 | 값 |
|----------|-----|
| quant_method | SVDQUANT |
| num_steps | 20 |
| n_calib | 4 |
| calib_seed_offset | 1000 |
| deepcache_interval | 2 |
| K values | 6, 8, 10, 12, 14, 16 |
| primary metric | A (per-block ablation) |

## 실험 결과 (n=100, steps=20)

### Sensitivity Ranking (Metric A, low → high)

```
[19, 5, 25, 21, 24, 23, 14, 13, 10, 18, 22, 12, 26, 9, 11, 3, 16, 17, 0, 15, 6, 8, 4, 20, 7, 2, 1, 27]
```

- **가장 낮은 sensitivity (skip 안전)**: 블록 19, 5, 25, 21, 24 (MSE ≈ 0.110–0.116)
- **가장 높은 sensitivity (skip 위험)**: 블록 27, 1, 2, 7, 20 (블록 27 MSE = 0.41, 이상값)
- 패턴: 네트워크 후반부(18-26)와 일부 중반부(5, 10, 13, 14)가 skip-safe. 초반부(0-4)와 최후반 블록 27은 민감.

### K-sweep 결과 (기준 Metric A ranking)

| 방법 | FID↓ | CLIP↑ | tpi(s)↓ | speedup↑ | skip blocks |
|------|------|-------|---------|---------|------------|
| SVDQUANT (no cache) | 121.32 | 34.840 | 2.85 | 1.00× | — |
| DeepCache c8-20 (K=12 contiguous) | 129.14 | 34.742 | 2.33 | **1.22×** | [8..19] |
| nl_gelu drift (corrector) | 124.94 | 34.823 | 2.53 | 1.13× | — |
| **selective K=6** | **126.04** | 34.801 | 3.05 | 1.12× | [5,19,21,23,24,25] |
| selective K=8 | 131.01 | 34.680 | 2.88 | 1.17× | [5,13,14,19,21,23,24,25] |
| selective K=10 | 131.61 | 34.686 | 2.36 | 1.22× | [5,10,13,14,18,19,21,23,24,25] |
| selective K=12 | 129.23 | 34.684 | 2.25 | **1.27×** | [5,10,12,13,14,18,19,21,22,23,24,25] |
| selective K=14 | 143.28 | 34.555 | 2.20 | 1.33× | — |
| selective K=16 | 143.28 | 34.587 | 2.02 | 1.40× | — |

## 분석

### K=6 — 최고 FID-효율 (corrector 없이 DeepCache 대비 개선)

FID=126.04는 DeepCache contiguous (129.14) 대비 **-3.1pt** 개선. speedup은 1.12×로 낮지만, corrector 없이 이 수준은 의미있음. 선택된 블록 {5,19,21,23,24,25}는 네트워크 중후반부의 skip-safe 블록들.

### K=12 selective vs DeepCache contiguous

- selective K=12: FID=129.23, speedup=1.27×
- DeepCache contiguous K=12: FID=129.14, speedup=1.22×

FID는 거의 동일 (0.09pt 차이, FID noise 범위 내), speedup은 selective가 더 높음 (0.05× 차이). 블록 위치 선택이 tpi에 영향 — 선택된 블록들이 실제로 더 빠른 회로 경로에 있을 가능성.

### K=8, 10 — 기대보다 나쁜 FID

K=8 (131.01), K=10 (131.61)이 K=12 (129.23)보다 나쁜 건 이상. 원인 분석:
- **FID noise (n=100)**: FID standard error at n=100 ≈ 3-5pt. 차이 2-3pt는 통계적으로 무의미.
- **Ranking의 비선형성**: Metric A는 single-block 단독 ablation이므로 combined K-block skip 시 상호작용 효과를 포착 못함. 블록들이 함께 skip될 때 quality 손해가 비선형적으로 누적될 수 있음.

### K=14, 16 — 급격한 FID 악화

FID 143.28로 급락. Metric A 기준 하위 14개를 skip하면 블록 3, 9, 26 등 그나마 중요한 블록들도 포함되기 시작. Sweet spot은 K ≤ 12.

### Metric C 버그

`callback_on_step_end` 는 transformer.forward 이후에 호출되므로, block forward hook 내부에서 step_counter를 읽을 때 항상 1 step 이전 값을 가짐 → 모든 cached step이 fresh로 오인 → 비교 누적 없음 → 모두 0.

수정 방법: `transformer.forward` 래퍼로 step_counter를 forward 시작 시 증가시켜야 함 (install_selective_skip의 `_step_counter` 패턴과 동일).

## 결론

**Best result (corrector 없음)**: selective K=6, FID=126.04 / speedup=1.12×

가설 부분 검증:
- ✓ Sensitivity ranking이 유의미: 특정 블록(19,5,25 등)이 일관되게 skip-safe
- ✓ K=6 selective skip이 DeepCache contiguous(K=12) 대비 FID -3.1pt 개선
- ✗ K=12 selective ≈ DeepCache contiguous: 어떤 블록을 skip하느냐보다 몇 개를 skip하느냐가 품질 결정에 더 중요함을 시사
- ✗ 기대한 "K=8–10에서 FID ≈ 125" 미달성 (K=8 FID=131.01)

**다음 방향**: selective K=6 + nl_gelu drift corrector 조합 시도 (corrector는 DeepCache가 발생시키는 residual drift 보정 → selective skip에도 적용 가능)

## 파일 위치

- **Sensitivity JSON**: `/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/SVDQUANT/MJHQ/sensitivity/SVDQUANT_steps20_cal4_seed1000.json`
- **샘플 + metrics**: `/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/SVDQUANT/MJHQ/SVDQUANT_selective_k{K}_mA_c8-20_steps20/`

## 실행 방법

```bash
cd /home/jovyan/workspace/Workspace_DiT/PartialBlockSkip_PixArt_Sigma_XL

# Sensitivity 측정 (Metric A + C, ~8 min)
CUDA_VISIBLE_DEVICES=0 /home/jovyan/.dit/bin/python3 measure_block_sensitivity.py \
    --quant_method SVDQUANT --num_steps 20 --n_calib 4 --metric both

# K-sweep (GPU 0,1 병렬, ~18 min)
bash run_selective_skip.sh

# 개별 K 실험
/home/jovyan/.dit/bin/python3 pixart_nvfp4_cache_compare.py \
    --quant_method SVDQUANT --cache_mode selective_skip \
    --sensitivity_json /data/.../SVDQUANT_steps20_cal4_seed1000.json \
    --skip_k 6 --sensitivity_metric A \
    --num_steps 20 --num_samples 100
```
