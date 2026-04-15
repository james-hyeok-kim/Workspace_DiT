# Timestep-Aware Mixed Precision Quantization

**날짜**: 2026-04-14  
**실험 위치**: `/home/jameskimh/workspace/Workspace_DiT/pixart_ts_mixed_precision/`  
**모델**: PixArt-Alpha XL-2 1024-MS  
**데이터셋**: MJHQ-30K

---

## 1. 목표

NVFP4_SVDQUANT_DEFAULT_CFG (baseline) 대비:
- **FID ↓ + IS ↑**: 품질 동등 이상
- **속도 ↑**: high-noise step에서 SVD 연산을 줄여 평균 bit-width 및 연산량 절약

**Baseline** (NVFP4_DEFAULT_CFG, 20 samples MJHQ):

| FID ↓ | IS ↑ | PSNR ↑ | SSIM ↑ |
|-------|------|--------|--------|
| 161.30 | 1.7318 | 15.69 | 0.5902 |

---

## 2. 핵심 아이디어

### 기존 접근의 한계
기존 실험(`pixart_ts_aware_experiment.py`, A2~A5)은 timestep group별로 **SmoothQuant scale만 다르게** 적용했지만, quantization **bit-width는 동일** (NVFP4 weight + INT4 activation 고정).

| Config | FID | IS | 비고 |
|--------|-----|-----|------|
| BASELINE | 161.30 | 1.7318 | — |
| A2 (G3, no-corr) | 174.30 | 1.8073 | IS↑ but FID↑ |
| A3 (G5, no-corr) | 177.74 | 1.7415 | IS↑ but FID↑ |
| A4 (G3, rank-4) | 402.98 | 1.4180 | 붕괴 |
| A5 (G5, rank-4) | 409.88 | 1.4303 | 붕괴 |

> A4/A5의 붕괴 원인: per-group SVD correction이 global quantized weight와 공간이 맞지 않아 발생.

### 새로운 접근: Mixed Precision (bit-width 자체를 timestep별 변경)

DiT denoising의 물리적 특성을 활용:

| 구간 | Timestep 범위 | Step 수 (20-step DPMSolver) | 특성 |
|------|-------------|---------------------------|------|
| **G0** (high noise) | t > 666 | ~7 steps | 노이즈 dominant, fine detail 없음 → 공격적 quant 허용 |
| **G1** (mid) | 333 < t ≤ 666 | ~7 steps | 구조 형성 중 → 중간 precision |
| **G2** (low noise) | t ≤ 333 | ~6 steps | detail-sensitive → higher precision 필요 |

**핵심 전략**:
- **Weight**: 단일 NVFP4 양자화 (모든 timestep 공유, 재양자화 없음)
- **Activation**: group별 다른 bit-width (NVFP4=4bit / INT4=4bit / INT8=8bit)
- **SVD Rank**: group별 다른 rank → high-noise에서 SVD 생략으로 연산 절약
  - 단일 SVD (max_rank)를 한 번 계산 후 runtime에 rank slicing (`lora_a[:r]`, `lora_b[:, :r]`)

```
Step t=999 (G0):   act=NVFP4(4bit), SVD rank=0  → 최대 속도, 최소 품질 투자
Step t=500 (G1):   act=INT4 (4bit), SVD rank=16 → 중간
Step t=100 (G2):   act=INT8 (8bit), SVD rank=32 → 최대 품질 보호
```

---

## 3. 구현

### 핵심 클래스: `MixedPrecisionTSLinear`

```
Forward(x, group=g):
  x_rot     ← block_hadamard(x)                    # offline 회전 (선택)
  x_smooth  ← x_rot * smooth_scales[g]             # per-group SmoothQuant scale
  x_q       ← quantize(x_smooth, act_mode[g])      # per-group activation precision
  x_q_sc    ← x_q * scale_correction[g]            # global weight space로 rescale
  x_global  ← x_rot * smooth_scale_global          # SVD 보정용 (global space)
  out       ← F.linear(x_q_sc, w_quantized)
            + F.linear(F.linear(x_global, lora_a[:r]), lora_b[:, :r])   # r = svd_rank[g]
```

### 핵심 수정 사항 (기존 `pixart_ts_aware_experiment.py` 대비)

1. **Boundary 버그 수정**: 기존 코드는 boundaries를 오름차순(`[333, 666]`)으로 검색하여 G1이 항상 0 step → 내림차순(`[666, 333]`)으로 수정
2. **SVD Rank Slicing 추가**: 단일 global SVD에서 group별 다른 rank를 runtime에 슬라이싱
3. **group_configs 시스템**: `[{act_mode, svd_rank}, ...]` 딕셔너리로 group별 설정 관리
4. **Effective Bitwidth 계산**: 실제 DPMSolver step 분포 기반 가중 평균 bit-width + SVD savings 계산

---

## 4. Ablation 설계 (5개 config)

각 config은 어떤 요소(activation precision / SVD rank)가 품질·속도에 기여하는지 분리 분석.

```
                        G0 (high noise)    G1 (mid)          G2 (low noise)
Config                  act  / svd_rank    act  / svd_rank   act  / svd_rank
─────────────────────────────────────────────────────────────────────────────
UNIFORM_FP4             NVFP4 / r32        NVFP4 / r32       NVFP4 / r32
  → Control: mixed precision 없음 (기존 global calibration과 동일 구조)

MP_ACT_ONLY             NVFP4 / r32        INT4  / r32       INT8  / r32
  → activation precision만 변화, SVD rank 고정
  → "low-noise step에서 8bit act이 품질에 얼마나 기여하나?"

MP_RANK_ONLY            NVFP4 / r0         NVFP4 / r16       NVFP4 / r32
  → activation 고정, SVD rank만 변화
  → "high-noise SVD 생략이 품질에 영향 없이 속도를 높이나?"

MP_MODERATE             NVFP4 / r8         INT4  / r16       INT8  / r32
  → act + rank 둘 다 점진적 변화 (밸런스)

MP_AGGRESSIVE           NVFP4 / r0         NVFP4 / r16       INT8  / r32
  → high-noise: 4bit act + SVD 완전 생략 / low-noise: 8bit + full SVD
  → 속도 최대화 + low-noise 품질 보호
```

### Effective Bitwidth (20-step DPMSolver, steps per group: [7, 7, 6])

| Config | Avg Act Bits | SVD Savings |
|--------|-------------|-------------|
| UNIFORM_FP4 | 4.00 bit | 0.0% |
| MP_ACT_ONLY | 5.20 bit | 0.0% |
| MP_RANK_ONLY | 4.00 bit | 43.8% |
| MP_MODERATE | 5.20 bit | 43.8% |
| MP_AGGRESSIVE | 5.20 bit | 68.8% |

> SVD savings = G0/G1에서 skip한 rank 비율. 전체 Linear 연산의 상당 부분(SVD branch = 2개 matmul)을 절약.

---

## 5. 하이퍼파라미터

| 파라미터 | 값 | 비고 |
|---------|------|------|
| `wgt_bits` | NVFP4 | Weight 양자화 포맷 |
| `block_size` | 128 | SmoothQuant + 양자화 block 크기 |
| `alpha` | 0.5 | SmoothQuant balance (weight:act = 0.5:0.5) |
| `max_rank` | 32 | Global SVD 최대 rank |
| `n_groups` | 3 | Timestep group 수 |
| `t_count` | 20 | DPMSolver denoising steps |
| `num_samples` | 20 | 평가 이미지 수 |
| `p_count` | min(64, num_samples) | Calibration prompt 수 |

---

## 6. 실험 파일 구조

```
pixart_ts_mixed_precision/
├── pixart_ts_mixed_precision.py   # 메인 실험 스크립트
├── run_ts_mixed_precision.sh      # Ablation sweep 실행 스크립트
├── experiment.md                  # 이 파일
├── logs/                          # 실행 로그
└── results/
    ├── UNIFORM_FP4/MJHQ/metrics.json
    ├── MP_ACT_ONLY/MJHQ/metrics.json
    ├── MP_RANK_ONLY/MJHQ/metrics.json
    ├── MP_MODERATE/MJHQ/metrics.json
    ├── MP_AGGRESSIVE/MJHQ/metrics.json
    └── results_summary.json       # 통합 비교표
```

---

## 7. 실험 결과

> 실행 중 (2026-04-14). 완료 후 아래 표를 채울 것.

| Config | Avg Act Bits | SVD Savings | FID ↓ | IS ↑ | PSNR ↑ | SSIM ↑ | vs Baseline |
|--------|-------------|-------------|-------|------|--------|--------|-------------|
| **BASELINE (NVFP4_DEFAULT_CFG)** | 4.00 | 0.0% | 161.30 | 1.7318 | 15.69 | 0.5902 | reference |
| **UNIFORM_FP4** | 4.00 | 0.0% | 177.98 | 1.7577 | 14.39 | 0.5777 | IS BEAT |
| **MP_ACT_ONLY** | 5.20 | 0.0% | 167.65 | 1.7202 | 14.36 | 0.5843 | - |
| **MP_RANK_ONLY** | 4.00 | 52.5% | 200.97 | 1.7401 | 13.09 | 0.5435 | IS BEAT |
| **MP_MODERATE** | 5.20 | 43.8% | 172.80 | 1.6778 | 14.00 | 0.5644 | - |
| **MP_AGGRESSIVE** | 5.20 | 52.5% | 193.09 | 1.7327 | 13.07 | 0.5457 | IS BEAT |

---

## 8. 분석 방향

결과 수신 후 아래 질문에 답한다:

1. **MP_RANK_ONLY vs UNIFORM_FP4**: SVD rank gating만으로 FID/IS가 유지되는가?
   - YES → high-noise step SVD는 품질에 기여가 적다 (속도 공짜 향상 가능)
   - NO → SVD가 전 구간에서 필요하다

2. **MP_ACT_ONLY vs UNIFORM_FP4**: low-noise 8bit activation이 효과적인가?
   - YES → 품질 향상이 확인되면 MP_MODERATE/AGGRESSIVE의 FID 개선 기대 가능

3. **MP_AGGRESSIVE vs BASELINE**: 속도·품질 trade-off가 acceptable한가?
   - 목표: FID < 161.3, IS > 1.732, SVD savings ~70%

4. **Failure mode 분석**: A4/A5의 FID~400 붕괴가 재현되는가?
   - 이번 구현은 per-group SVD가 아닌 global SVD + rank slicing이므로 다른 결과 예상

---

## 9. 비고

- **기존 버그 (boundary 방향)**: `pixart_ts_aware_experiment.py`의 A2/A3도 G1이 항상 0 step이었음 (`boundaries=[333,666]`을 오름차순 검색). 이번 실험에서 수정 (내림차순 `[666,333]`).
- **IS with 20 samples**: 매우 노이즈 큼. 경향성 참고용으로만 사용. 신뢰도 높은 비교는 100+ samples 필요.
- **FID with 20 samples**: 마찬가지로 노이즈 있으나 FID의 경우 20 samples에서도 상대적 비교 가능.
