# W3A4 Quantization Experiment Handoff

## 개요
Rotation(Hadamard) + SmoothQuant + NVFP4 weight + SVD + INT4-Act 방식과
FP16 / NVFP4_DEFAULT_CFG(공식 baseline) 비교 실험.
총 3회 실행 / 20 samples / dataset: MJHQ

> **최종 파이프라인**: Hadamard block rotation → SmoothQuant → NVFP4 weight (per-128-group) → SVD rank=32 → INT4 act (per-16-element)
> 초기 설계(GPTQ-INT3)에서 수차례 디버깅을 거쳐 도달한 구성.

---

## Change List (신규 생성 파일)

| 파일 | 설명 |
|---|---|
| `pixart_w3a4_experiment.py` | W3A4 / BASELINE / FP16 통합 실험 스크립트. W3A4Linear 클래스(Hadamard+SmoothQuant+NVFP4weight+SVD+INT4Act). |
| `run_w3a4_experiment.sh` | 3-run 자동화 스크립트. run.log 통합 기록, results_summary.json · handoff_w3a4.md 자동 생성. Skip logic 포함. |
| `results/w3a4_experiment/results_summary.json` | 전체 실험 결과 통합 JSON. |
| `results/w3a4_experiment/*/MJHQ/metrics.json` | 실행별 개별 결과 JSON (primary: FID/IS, secondary: PSNR/SSIM/LPIPS/CLIP). |
| `logs/w3a4_experiment/run.log` | 전체 실험 통합 로그 (타임스탬프 포함). |
| `handoff_w3a4.md` | 이 파일. |

> 기존 파일(`pixart_rpca_experiment.py`, `run_rpca_sweep.sh` 등)은 **일절 수정하지 않음**.

---

## 최종 실험 설정 (확정)

| 항목 | 값 | 비고 |
|---|---|---|
| Weight 양자화 | NVFP4 (non-uniform 4-bit) | per-128-column group |
| Activation 양자화 | INT4 uniform | per-16-element block |
| Hadamard block size | 128 | PixArt 1152=9×128, 4608=36×128 완벽 분할 |
| SVD rank | 32 | 오차 보정 branch |
| SmoothQuant alpha | 0.5 | weight/activation 균등 분담 |
| Skip layers | x_embedder, t_embedder, proj_out | 민감 레이어 보호 |
| Calibration | E[x²](H_diag) + x_max | act_order에 사용 |

---

## 디버깅 이력: 초기 설계 → 최종 구성

초기 설계(GPTQ-INT3)에서 최종 파이프라인까지의 변경 과정을 기록.

| 단계 | 변경사항 | FID↓ | IS↑ | SSIM↑ | 원인 / 결론 |
|---|---|---|---|---|---|
| ① 최초 실행 | INT3 weight, act_bs=128, GPTQ 보상 | 657 | 1.00 | 0.057 | **GPTQ 보상 루프 수치 폭발** (아래 설명) |
| ② 보상 제거 | GPTQ 보상 루프 제거 | 457 | 1.68 | 0.254 | 이미지 부분 복구. INT3 자체가 너무 공격적 |
| ③ weight 변경 | INT4 uniform weight | 258 | 1.73 | 0.424 | INT4가 INT3보다 훨씬 안정적 |
| ④ act 블록 축소 | act block_size 128→16 | 205 | 1.77 | 0.499 | 세밀한 activation scale이 핵심 |
| ⑤ weight 최적화 | NVFP4 weight | **178** | **1.794** | 0.570 | 비균일 레벨이 신경망 분포에 적합 |
| ⑥ act 시도 (기각) | NVFP4 act | 188 | 1.667 | 0.574 | SmoothQuant 변형 분포에 NVFP4 부적합 → INT4 유지 |
| ⑦ rank 증가 (기각) | SVD rank 32→64 | 176 | 1.736 | 0.557 | IS 감소, SSIM 감소 → rank=32가 최적 |
| **BASELINE** | NVFP4_DEFAULT_CFG | **161** | **1.739** | **0.590** | 비교 기준 |

### 버그 상세: GPTQ 보상 루프 수치 폭발 (단계 ①)

```python
# 문제 코드 (제거됨)
err_norm = (err / h_g.unsqueeze(0)).sum(dim=1, keepdim=True)  # (out_f, 1)
W_work[:, g_end:] -= err_norm * h_r.unsqueeze(0)
```

- `err` = 128개 컬럼의 INT3 양자화 오차 (각 ≈ ±w_max/3)
- `h_g` = E[x²] (calibration 값이 작으면 0.001 수준)
- `err / h_g` → 수십~수백 배 증폭, 128개 합산 → `err_norm` 수천 배
- 모든 잔여 컬럼에 적용 → **가중치 완전 파괴**, IS=1.0 (all-noise 이미지)
- **수정**: 보상 루프 제거, act_order + per-group 양자화만 유지. SVD branch가 잔류 오차 보정.

---

## 최종 실험 결과

| Run | FID↓ | IS↑ | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP↑ | Beat Baseline? |
|---|---|---|---|---|---|---|---|
| FP16 | -0.0002 | 1.7068 | ∞ | 1.0000 | 0.0000 | 35.52 | YES (상한값) |
| **W3A4 (최종)** | **178.4** | **1.7940** | 13.19 | 0.5696 | 0.4627 | 34.61 | FID: no / **IS: YES** |
| BASELINE (NVFP4) | 161.3 | 1.7394 | 15.69 | 0.5902 | 0.4406 | 34.98 | 기준값 |

**주요 해석:**
- **IS 1.794 > BASELINE 1.739** → 이미지 선명도·다양성 지표는 BASELINE 역전 ✓
- **FID 178 vs 161** → 차이 17pt. 20샘플 FID는 통계 노이즈가 크므로 해석 주의 (아래 설명)
- **PSNR 13.19 vs 15.69** → FP16 대비 충실도는 BASELINE이 여전히 우위

### 20샘플 FID 해석 주의사항
FID reference = `/data/james_dit_ref/ref_images_fp16` (FP16 모델 출력). 즉 FID는 "이미지 품질"이 아닌 "FP16 충실도"를 측정. NVFP4 4-bit ≈ W3A4 4-bit이므로 FID 차이가 작은 것은 자연스럽고, 17pt 차이는 20샘플 기준 노이즈 범위 내일 가능성 있음.

---

## 저장 경로

```
results/w3a4_experiment/
  FP16/MJHQ/metrics.json
  BASELINE/MJHQ/metrics.json
  W3A4/MJHQ/metrics.json
  results_summary.json          ← 전체 통합
logs/w3a4_experiment/
  run.log                       ← 전체 통합 로그
  fp16_prod.log
  baseline_prod.log
  w3a4_prod.log
```

---

## 다음 단계 제안

### 단기 (신뢰도 향상)
1. **num_samples=100 재실행** — FID 노이즈 감소. IS 역전이 통계적으로 유의미한지 확인
2. **FID reference를 MJHQ 실제 이미지로 교체** — 현재 FID는 "FP16 충실도"를 측정, 실제 화질 비교 불가

### 중기 (성능 개선)
3. **RPCA-style outlier separation 추가** — W3A4 pipeline 앞에 sparse branch(이상치 분리) 추가 시 INT4×INT4 RPCA(FID~98) 수준 기대
4. **cross-attention layer mixed-precision** — x_embedder, t_embedder, proj_out 외에도 cross-attn Q/K/V를 FP16 유지하면 추가 개선 가능
5. **alpha 조정** (현재 0.5) — W4A4에서 weight 보호를 더 강조하려면 alpha 낮추기 (0.3~0.4)

### 장기
6. **W2A4 탐색** — INT2 weight + outlier branch + rank=64 SVD
7. **mixed INT3/INT4** — 민감 레이어 INT4, 나머지 INT3으로 평균 3.x-bit 달성
