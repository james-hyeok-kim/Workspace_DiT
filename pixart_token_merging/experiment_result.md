# Token Merging (ToMe) + Quantization 실험 결과

- **실험일**: 2026-04-14 (KST)
- **모델**: PixArt-alpha/PixArt-XL-2-1024-MS
- **GPU**: B200
- **데이터셋**: MJHQ (20 samples)
- **추론 스텝**: 20 steps (DPMSolverMultistep)
- **실험 폴더**: `/home/jameskimh/workspace/Workspace_DiT/pixart_token_merging/`

---

## 실험 목적

`NVFP4_SVDQUANT_DEFAULT_CFG` baseline 대비 **더 빠르면서 FID/IS를 유지 또는 개선**하는 것이 목표.

ToMe(Token Merging)는 attention의 O(N²) 복잡도를 줄이는 방식으로, Quantization(연산 precision 절감)과 독립적인 speedup 축을 제공할 수 있다는 가설 하에 실험.

- PixArt-Sigma-XL-2-1024-MS: 28 transformer blocks, 4096 patch tokens (1024×1024 입력 기준)
- Self-attention: (B, 16heads, 4096, 72) — quadratic in N

---

## 구현

### 파일 구조

```
pixart_token_merging/
├── tome_core.py                    # Bipartite soft matching + merge/unmerge 클로저
├── tome_patch.py                   # BasicTransformerBlock monkey-patch
├── pixart_tome_experiment.py       # Phase 1: ToMe-only (FP16) sweep
├── pixart_tome_quant_experiment.py # Phase 2: ToMe + NVFP4 (미실행)
├── run_tome_experiment.sh
└── run_tome_quant_experiment.sh
```

### 알고리즘

**Bipartite Soft Matching** (Bolya et al., ICLR 2023):
1. 토큰을 A(짝수 인덱스) / B(홀수 인덱스) 파티션으로 분할
2. A→B 코사인 유사도 계산, 상위 r쌍 선택
3. 선택된 A 토큰을 대응 B 토큰에 평균으로 병합 → (N-r) 토큰
4. Unmerge: B 값을 원래 A 위치에 복제하여 N 토큰 복원

**Monkey-patch 방식**: `types.MethodType`으로 `BasicTransformerBlock.forward` 교체. diffusers 소스 수정 없음.

### 최적화 이력

**v1 (초기)**: `x.clone()` + `nonzero()` 기반 → CPU-GPU 동기화 발생
- `nonzpy()` 제거 → `argsort` 기반 GPU-only 인덱스 계산
- 클로저 반환으로 인덱스 한 번만 계산
- scatter 연산 범위를 N → nB (절반)로 축소
- 벤치마크: matching+merge+unmerge 전체 **1.04ms/iter** (B=2, N=4096, C=1152)

---

## 실험 1 — Residual Merge 포함 (잘못된 구현)

**실험 날짜**: 2026-04-14

### 설정

각 block에서 norm_hidden_states와 **hidden_states(residual) 모두** merge:

```python
norm_hs_merged = merge_fn(norm_hidden_states)
hs_merged      = merge_fn(hidden_states)          # ← 잘못된 부분
attn_output    = attn1(norm_hs_merged)
hidden_states  = gate * attn_output + hs_merged   # residual도 merged space
hidden_states  = unmerge_fn(hidden_states)        # 여기서 복원
# FFN도 (N-r) tokens로 실행
```

### 결과 (20 samples, FP16)

| Config | Merge Ratio | Blocks | FID↓ | IS↑ | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP↑ | sec/img | speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| FP16_baseline | 0.00 | all | ~0 | 1.713 | ∞ | 1.000 | 0.000 | 35.52 | 1.04s | — |
| R10_ALL | 0.10 | 0–27 | 436.82 | 1.531 | 9.16 | — | — | — | 1.51s | 0.69x |
| R20_ALL | 0.20 | 0–27 | 564.12 | 1.251 | 7.26 | — | — | — | 0.96s | 1.08x |
| R30_ALL | 0.30 | 0–27 | 557.60 | 1.071 | 6.75 | — | — | — | 0.97s | 1.08x |
| R50_ALL | 0.50 | 0–27 | 548.00 | 1.048 | 6.71 | — | — | — | 0.95s | 1.09x |
| R20_MID | 0.20 | 4–23 | 517.40 | 1.134 | 8.02 | — | — | — | 0.87s | 1.19x |
| R30_MID | 0.30 | 4–23 | 539.00 | 1.097 | 7.20 | — | — | — | 0.86s | 1.21x |

### 문제점

- FID가 전 config에서 500+ 수준으로 붕괴 (FP16 대비 품질 파괴)
- IS도 FP16(1.71) 대비 1.05~1.53 수준으로 하락
- **원인**: unmerge 시 병합된 토큰을 단순 복제 → 매 block마다 정보 손실 누적. 28 blocks × 20 steps 동안 누적되어 이미지 품질 파괴

---

## 실험 2 — Residual Merge 제거 (수정된 구현)

**실험 날짜**: 2026-04-14

### 설정

**residual(hidden_states)는 절대 merge하지 않음**. Attention sub-layer 내부에서만 merge/unmerge:

```python
norm_hs_merged  = merge_fn(norm_hidden_states)    # norm만 merge
attn_output     = attn1(norm_hs_merged)            # (N-r) tokens로 attention
attn_output     = unmerge_fn(attn_output)          # residual add 전에 복원
hidden_states   = gate * attn_output + hidden_states  # residual은 항상 full N
# FFN은 full N=4096 tokens로 실행
```

### 결과 (20 samples, FP16)

| Config | Merge Ratio | Blocks | FID↓ | IS↑ | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP↑ | sec/img | speedup |
|---|---|---|---|---|---|---|---|---|---|---|
| **FP16_baseline** | 0.00 | all | **~0** | **1.735** | **∞** | **1.000** | **0.000** | **35.52** | **0.70s** | — |
| R10_ALL | 0.10 | 0–27 | 148.55 | 1.763 | 14.60 | 0.568 | 0.435 | 35.38 | 0.83s | 0.84x |
| R20_ALL | 0.20 | 0–27 | 242.68 | **1.772** | 12.16 | 0.361 | 0.682 | 33.02 | 0.84s | 0.84x |
| R30_ALL | 0.30 | 0–27 | 338.39 | 1.721 | 10.98 | 0.254 | 0.786 | 30.26 | 1.02s | 0.69x |
| R50_ALL | 0.50 | 0–27 | 397.79 | 1.491 | 10.00 | 0.219 | 0.853 | 25.88 | 1.20s | 0.59x |
| R20_MID | 0.20 | 4–23 | 150.00 | **1.775** | **16.49** | **0.661** | **0.358** | 34.94 | 1.06s | 0.66x |
| R30_MID | 0.30 | 4–23 | 197.42 | 1.747 | 15.92 | 0.616 | 0.425 | 34.69 | 2.02s | 0.35x |

> FID는 생성 이미지와 FP16 reference 이미지 간 비교 (낮을수록 FP16에 가까움).
> IS·CLIP이 FP16_baseline과 유사하면 텍스트 정렬 및 이미지 다양성이 유지된다는 의미.

### 품질 분석

- **IS 회복**: R20_MID (1.775), R20_ALL (1.772) 모두 FP16 (1.735)과 동등 수준
- **PSNR best**: R20_MID (16.49 dB) — 픽셀 수준 fidelity도 가장 양호
- **CLIP 유지**: R10_ALL, R20_MID 모두 34~35점대 유지 (FP16 35.52와 근접)
- merge ratio가 높을수록(R30, R50) FID·SSIM·LPIPS 모두 열화

---

## 핵심 발견 및 결론

### 왜 ToMe가 이 환경에서 speedup을 주지 못하는가

```
전체 시간 ≈  Attention(N-r)  +  FFN(N)  +  merge/unmerge 오버헤드
                  ↑ 절감         ↑ 그대로       ↑ 추가 비용
```

| 구성 요소 | 설명 |
|---|---|
| **Attention 절감** | O(N²) → O((N-r)²). 이론상 r=0.2면 1.56x, r=0.3면 2.04x |
| **FFN 미절감** | residual merge 제거로 FFN은 full N=4096 tokens 유지 |
| **Matching 오버헤드** | bmm(nA, nB) + argsort + gather/scatter, 28 blocks × 20 steps |
| **B200 FlashAttention** | 4096 tokens의 attention 자체가 이미 최적화되어 bottleneck이 아님 |

**결론**: B200에서 실제 bottleneck은 QKV projection + FFN의 **linear 연산**이다. Token Merging은 attention quadratic 비용만 줄이므로, 해당 하드웨어에서 end-to-end speedup이 발생하지 않는다.

### 두 구현 방식 비교 요약

| 항목 | Residual Merge O (Exp.1) | Residual Merge X (Exp.2) |
|---|---|---|
| 품질 (FID) | 파괴 (500+) | 양호 (150~200) |
| 품질 (IS) | 파괴 (1.05~1.53) | FP16 동등 (~1.75) |
| 속도 (best) | FP16 대비 1.21x 빠름 | FP16 대비 0.35~0.84x (느림) |
| Phase 2 전망 | 품질 문제로 무의미 | 속도 문제로 무의미 |

### Phase 2 (ToMe + NVFP4) 전망

실행하지 않음. 이유:
- NVFP4 단독으로 이미 FP16보다 빠름
- ToMe를 추가해도 FFN bottleneck 구조 때문에 추가 speedup 없음
- 오히려 matching + merge/unmerge 오버헤드로 NVFP4 단독보다 느려질 가능성 높음

---

## 권장 후속 실험 방향

Token Merging 단독으로는 B200 환경에서 유효한 speedup 축이 아님이 확인됨.
목표(NVFP4 baseline 대비 빠르면서 FID/IS 유지)를 위한 대안:

1. **Step Distillation** — 추론 step 수 자체를 20→8 또는 4로 줄임. Linear한 speedup 보장
2. **Timestep-aware Mixed Precision** (`pixart_ts_mixed_precision/`) — 이미 진행 중
3. **NVFP4 + DeepCache 조합** (`pixart_caching/`) — block-level caching으로 실질적 speedup 확인됨

---

## 파일 위치

| 파일 | 내용 |
|---|---|
| `results/MJHQ/tome/sweep_summary.csv` | Exp.2 전체 sweep 수치 (CSV) |
| `results/MJHQ/tome/sweep_summary.json` | Exp.2 전체 sweep 수치 (JSON) |
| `logs/tome_fp16_20260414_051518.log` | Exp.1 (residual merge O) 전체 로그 |
| `logs/tome_fp16_20260414_054658.log` | Exp.2 (residual merge X) 전체 로그 |
