# Level 2: Timestep-aware Corrector + Timestep-stratified Loss

## 개요

DeepCache로 스킵된 transformer block의 stale residual을 보정하는 corrector를 timestep-aware하게 설계하고, 학습 loss를 timestep-stratified feature distillation으로 구성하는 방법.

## Architecture: GELUBottleneckT

기존 GELUBottleneck에 FiLM-style timestep conditioning 추가.

```python
class GELUBottleneckT(nn.Module):
    def __init__(self, hidden_dim: int, rank: int):
        super().__init__()
        self.A = nn.Linear(hidden_dim, rank, bias=False)
        self.B = nn.Linear(rank, hidden_dim, bias=False)
        self.scale_net = nn.Linear(1, rank)
        self.shift_net  = nn.Linear(1, rank)

    def forward(self, dx, t_norm=None):
        h = self.A(dx)
        if t_norm is not None:
            scale = self.scale_net(t_norm)
            shift = self.shift_net(t_norm)
            h = scale * F.gelu(h) + shift
        else:
            h = F.gelu(h)
        return self.B(h)
```

- `dx`: stale residual (h_out_prev - h_in_prev)
- `t_norm`: normalized timestep ∈ [0, 1], shape (B, 1)
- FiLM 조건화: `scale * GELU(A(dx)) + shift`
- 파라미터 수: `2 × hidden_dim × rank + 2 × rank` (rank=4, hidden_dim=1152 → ~9.2K params/block)

## Loss: fd_stratified (Feature Distillation + Timestep Stratification)

### Training target

Drift loss (기존) 대신 feature distillation target 사용:

```
target = h_out_curr - h_in_curr   # 현재 step의 block output delta (fresh residual)
```

corrector가 직접 fresh residual을 예측 → inference 시 `h = h_in + correction` (stale_res 불필요).

### Stratified loss

timestep을 5개 bucket으로 균등 분할하여 각 bucket의 loss를 평균:

```python
per_sample_loss = (pred - target).pow(2).mean(dim=-1)
bucket = (t_norm * n_buckets).long().clamp(0, n_buckets - 1)
bucket_losses = [per_sample_loss[bucket==b].mean() for b in range(n_buckets) if (bucket==b).any()]
loss = torch.stack(bucket_losses).mean()
```

- 목적: late timestep (t→0, 디테일 생성 단계)이 gradient를 독점하는 현상 방지
- n_buckets=5: early / early-mid / mid / mid-late / late

## 실험 설정

| 파라미터 | 값 |
|----------|-----|
| cache_mode | `cache_nl_gelu_t` |
| nl_loss_type | `fd_stratified` |
| lora_rank | 4 |
| nl_mid_dim | 32 |
| cache_start / end | 8 / 20 |
| deepcache_interval | 2 |
| lora_calib | 4 |
| calib_seed_offset | 1000 |
| n_epochs | 300 |
| batch_size | min(8192, N) |

학습 데이터는 전부 GPU에 preload (`dx_gpu`, `target_gpu`, `t_gpu`) → 매 epoch GPU-side randperm indexing으로 shuffle.

## 실행 방법

```bash
# run_nonlinear.sh Option 7
bash run_nonlinear.sh --modes gelu_ts --num_samples 100 --steps 20
bash run_nonlinear.sh --modes gelu_ts --num_samples 100 --steps 15

# 직접 실행
python pixart_nvfp4_cache_compare.py \
    --quant_method SVDQUANT \
    --cache_mode cache_nl_gelu_t \
    --nl_loss_type fd_stratified \
    --num_steps 20 \
    --num_samples 100 \
    --lora_rank 4 \
    --nl_mid_dim 32 \
    --cache_start 8 --cache_end 20 \
    --deepcache_interval 2 \
    --lora_calib 4
```

## 결과 위치

- **Corrector .pt**: `/data/jameskimh/james_dit_pixart_sigma_xl_mjhq_cache_adapter/SVDQUANT/nl_gelu_t_fdstratified_r4_m32_cs8_ce20_steps{N}_cal4_seed1000.pt`
- **샘플 + metrics**: `/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/SVDQUANT/MJHQ/SVDQUANT_nl_gelu_t_fdstratified_r4_m32_c8-20_steps{N}/`

## 결과 (n=100, 버그 수정 전 — DeepCache 미적용 상태)

> **주의**: `cache_nl_gelu_t`가 `_cache_modes_with_dc` 리스트에 누락되어 DeepCache 없이 실행됨. 아래 수치는 plain SVDQUANT와 동일한 조건.

| steps | FID↓ | IS↑ | CLIP↑ | speedup |
|-------|------|-----|-------|---------|
| 20 | 121.3 | 5.40 | 34.84 | 1.0× (버그) |
| 15 | 122.7 | 5.86 | 34.92 | 1.0× (버그) |

버그 수정 후 (`cache_nl_gelu_t`를 `_cache_modes_with_dc`에 추가) 재실험 필요.

## 비교 baseline (n=100)

| 방법 | steps | FID↓ | IS↑ | CLIP↑ | speedup |
|------|-------|------|-----|-------|---------|
| SVDQUANT (no cache) | 20 | 121.3 | 5.61 | 34.84 | 1.0× |
| nl_gelu (drift) | 20 | 124.9 | 5.51 | 34.82 | 1.27× |
| nl_gelu_t_fdstratified (**재실험 필요**) | 20 | TBD | TBD | TBD | ~1.27× |
| SVDQUANT (no cache) | 15 | ~150 | — | — | 1.0× |
| nl_gelu (drift) | 15 | 131.6 | 5.32 | 34.90 | 1.27× |
| nl_gelu_t_fdstratified (**재실험 필요**) | 15 | TBD | TBD | TBD | ~1.27× |

## 버그 수정 내역

`pixart_nvfp4_cache_compare.py` L381–385:

```python
# 수정 전
_cache_modes_with_dc = (..., "cache_nl_film")

# 수정 후
_cache_modes_with_dc = (..., "cache_nl_film", "cache_nl_gelu_t")
```
