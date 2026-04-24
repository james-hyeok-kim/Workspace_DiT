# Partial Block Skip

## 개요

DeepCache는 `[cache_start, cache_end)` 구간의 **block 전체**를 cached step에서 skip하여 1.22× speedup을 얻지만 FID 129.14로 no-cache(121.32) 대비 품질 손해가 크다. 이를 개선하기 위해 block 내부 sub-module 단위로 선택적 skip을 실험함.

가설: attn1(self-attention) 또는 ff(MLP)만 skip하면 speedup은 줄지만 FID 손해가 훨씬 작을 것.

## 방법론

### PixArt-Sigma-XL Block 구조 (ada_norm_single)

```
timestep → adaLN → shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

norm1(h) * (1+scale_msa) + shift_msa
  → [attn1]  self-attention
  → gate_msa * attn1_out  (gate는 timestep 의존 — 매 step 재계산 필수)
  → h += attn1_gated

h  (no norm)
  → [attn2]  cross-attention (encoder_hidden_states)
  → h += attn2_out

norm2(h) * (1+scale_mlp) + shift_mlp
  → [ff]  MLP
  → gate_mlp * ff_out
  → h += ff_gated
```

- `attn1`, `attn2`, `ff`, `norm1`, `norm2` 는 각각 독립된 `nn.Module` 속성
- `gate_*` 는 timestep 의존 → 캐싱 불가. 대신 submodule raw output(pre-gate)만 캐싱 후 fresh gate 재적용

### 3가지 캐시 모드

| Mode | Cache 대상 | 매 step 재계산 |
|------|-----------|---------------|
| `partial_attn` | attn1 pre-gate output | adaLN, norm1/norm2, **attn2**, ff, gates, residuals |
| `partial_mlp`  | ff pre-gate output | adaLN, norm1/norm2, attn1, attn2, gates, residuals |
| `partial_attn_mlp` | attn1 AND ff pre-gate outputs | adaLN, norm1/norm2, **attn2**, gates, residuals |

**attn2(cross-attn)는 모든 모드에서 항상 재계산**: hidden_states(매 step 변화)와 text embedding을 함께 입력받아 변동이 크고, block FLOPs의 ~3%에 불과하여 캐시 이득 미미.

### 구현

기존 `install_deepcache()`와 독립적으로 `install_partial_skip()` 구현:

- `transformer.forward` 원본 유지 + step_idx 증분용 얇은 래퍼
- `transformer_blocks[i].forward` (i ∈ [cs, ce))를 `BasicTransformerBlock.forward` (ada_norm_single) 를 mirror한 custom forward로 교체
- skip 조건: `step_idx % cache_interval != 0` AND `step_idx ∉ full_steps_set` AND cache miss가 아닌 경우

```python
# per-block custom forward 핵심 (partial_attn 기준)
skip_attn = state.partial_mode in ("attn", "attn_mlp")

if skip_attn and is_cached and b_idx in state.attn1_pre_gate_cache:
    attn1_out = state.attn1_pre_gate_cache[b_idx]          # 캐시 히트
else:
    attn1_out = self.attn1(norm_h, ...)
    if skip_attn:
        state.attn1_pre_gate_cache[b_idx] = attn1_out.detach()

hidden_states = gate_msa * attn1_out + hidden_states        # fresh gate 재적용
```

## 실험 설정

| 파라미터 | 값 |
|----------|-----|
| cache_mode | `partial_attn` / `partial_mlp` / `partial_attn_mlp` |
| cache_start / end | 8 / 20 |
| deepcache_interval | 2 |
| num_steps | 20 |
| num_samples | 100 |
| quant_method | SVDQUANT |

## 결과 (n=100, steps=20)

| 방법 | FID↓ | CLIP↑ | tpi(s)↓ | speedup↑ | 비고 |
|------|------|-------|---------|---------|------|
| SVDQUANT (no cache) | 121.32 | 34.840 | 2.85 | 1.00× | 기준 |
| DeepCache c8-20 (full) | 129.14 | 34.742 | 2.33 | 1.22× | 기준 |
| nl_gelu drift | 124.94 | 34.823 | 2.53 | 1.13× | 기준 |
| `partial_attn` | 135.60 | 34.623 | 2.71 | 1.05× | full skip보다 나쁨 |
| `partial_mlp` | 361.06 | 27.302 | 2.82 | 1.01× | **붕괴** |
| `partial_attn_mlp` | 128.72 | 34.639 | 2.60 | 1.10× | full skip 대비 미미한 개선 |

## 분석

### partial_mlp 붕괴 (FID=361)

ff의 입력 `norm2(hidden_states)` 는 self-attn과 cross-attn 이후의 누적된 hidden_states를 기반으로 하며, 이는 매 diffusion step마다 크게 변한다. Pre-gate ff output을 그대로 재사용하면 현재 step의 hidden_states 분포와 전혀 맞지 않아 결과가 발산함.

반면 attn1의 입력 `norm1(hidden_states)` 는 block 입력에 직접 가깝고, 인접 step 간 hidden_states 변화가 상대적으로 작아 partial_attn이 생존 가능했던 것으로 추정.

### partial_attn (FID=135.6) — 예상보다 나쁨

기존 DeepCache(129.14)보다 FID가 오히려 나쁨. 원인 분석:
- Self-attention output은 단순히 "작은 변화"가 아닐 수 있음. Block [8,20) 에서 attn1이 전체 diffusion trajectory 정보를 크게 갱신하는 역할을 함.
- Full block skip에서는 stale residual 전체가 하나의 단위로 재사용되어 내부적으로 일관성이 있지만, partial skip은 attn1은 stale + attn2와 ff는 fresh → 서로 다른 시점의 정보가 혼합됨.

### partial_attn_mlp (FID=128.72) — 미미한 개선

attn2 재계산 효과는 FID 기준 0.4pt 개선에 그침. Cross-attention은 block FLOPs의 ~3%이므로 speedup 이득도 거의 없음 (1.10× vs full skip 1.22×).

## 결론

**Pre-gate submodule caching 전략은 효과적이지 않음**. 특히:
- MLP(ff) pre-gate caching은 입력 분포 변화로 인해 완전 붕괴
- Self-attn(attn1) caching도 full block skip보다 FID가 나쁨
- Full block skip + nl_gelu drift corrector (FID=124.94)가 여전히 최고

본 실험이 시사하는 바: DeepCache의 강점은 "전체 block residual을 하나의 단위로 재사용"하는 내부 일관성에 있으며, 이를 submodule 수준으로 쪼개면 일관성이 깨져 오히려 품질이 나빠짐.

## 결과 위치

- **샘플 + metrics**: `/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/SVDQUANT/MJHQ/SVDQUANT_{partial_attn,partial_mlp,partial_attn_mlp}_c8-20_steps20/`

## 실행 방법

```bash
cd /home/jovyan/workspace/Workspace_DiT/PartialBlockSkip_PixArt_Sigma_XL

# 단일 모드
python pixart_nvfp4_cache_compare.py \
    --quant_method SVDQUANT --cache_mode partial_attn \
    --num_steps 20 --num_samples 100 \
    --cache_start 8 --cache_end 20 --deepcache_interval 2 \
    --guidance_scale 4.5

# 전체 3종 순차 (단일 GPU)
bash run_partial_block_skip.sh

# 2-GPU 병렬 (GPU0: partial_attn+partial_mlp, GPU1: partial_attn_mlp)
CUDA_VISIBLE_DEVICES=0 bash -c "for m in partial_attn partial_mlp; do python ... --cache_mode \$m; done" &
CUDA_VISIBLE_DEVICES=1 python ... --cache_mode partial_attn_mlp &
```
