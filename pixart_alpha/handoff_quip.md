# QuIP W3A4/W2A4 Experiment Handoff

## 개요
Randomized Hadamard (QuIP-style) + INT3/INT2 weight + SVD + INT4-Act 방식과
FP16 / NVFP4_DEFAULT_CFG(공식 baseline) 비교 실험.
총 4회 실행 / 20 samples / dataset: MJHQ
시작: 2026-04-03 02:12:13  |  종료: 2026-04-03 02:15:42

> **파이프라인**: Random sign S_in → Block Hadamard → per-group INT3/INT2 quantization → SVD rank 보정 → INT4 activation (bs=16)
> SmoothQuant 없음 — randomized rotation이 incoherence 이론적 보장.

---

## QuIP vs GPTQ 핵심 차이

| 항목 | GPTQ (W3A4 이전 실험) | QuIP# (이번 실험) |
|---|---|---|
| Rotation | 결정론적 block Hadamard | **랜덤 sign(S_in) + block Hadamard** |
| SmoothQuant | 필요 (outlier 처리) | **불필요** (rotation이 incoherence 보장) |
| Hessian | H^-1 Cholesky → 오차 전파 | H_diag(act_order)만 — 보상 루프 없음 |
| 2-bit 적합성 | 오차 high-rank → SVD 보정 어려움 | near-Gaussian 분포 → INT2 가능 |
| S_in | 없음 | layer별 고정 random ±1 (seed=layer_idx) |

---

## Change List (신규 생성 파일)

| 파일 | 설명 |
|---|---|
| `pixart_quip_experiment.py` | QuIP W3A4/W2A4 / BASELINE / FP16 통합 실험 스크립트. QuIPLinear 클래스. |
| `run_quip_experiment.sh` | 4-run 자동화 스크립트. run.log 통합 기록, results_summary.json · handoff_quip.md 자동 생성. |
| `results/quip_experiment/results_summary.json` | 전체 실험 결과 통합 JSON. |
| `results/quip_experiment/*/MJHQ/metrics.json` | 실행별 개별 결과 JSON. |
| `logs/quip_experiment/run.log` | 전체 실험 통합 로그. |
| `handoff_quip.md` | 이 파일. |

> 기존 파일(`pixart_w3a4_experiment.py`, `run_w3a4_experiment.sh` 등)은 **일절 수정하지 않음**.

---

## 실험 설정

| 항목 | W3A4 | W2A4 |
|---|---|---|
| Weight 양자화 | INT3 uniform (7 levels) | INT2 uniform (3 levels, ternary) |
| Activation 양자화 | INT4 per-16-element | INT4 per-16-element |
| Hadamard block size | 128 | 128 |
| SVD rank | 32 | 64 |
| SmoothQuant | 없음 | 없음 |
| Random sign S_in | layer별 고정 (seed=layer_idx) | layer별 고정 (seed=layer_idx) |
| Skip layers | x_embedder, t_embedder, proj_out | x_embedder, t_embedder, proj_out |

---

## 실험 결과 (FID 오름차순)

- **FP16 baseline**: FID=-0.0002  IS=1.7172
- **NVFP4 baseline**: FID=161.3019  IS=1.7318

| Run | Method | FID↓ | IS↑ | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP↑ | Beat FID? | Beat IS? |
|---|---|---|---|---|---|---|---|---|---|
| FP16 | FP16 | -0.0002 | 1.7172 | inf | 1.0000 | 0.0000 | 35.52 | YES | no |
| BASELINE | BASELINE | 161.3019 | 1.7318 | 15.69 | 0.5902 | 0.4406 | 34.98 | no | no |
| W3A4 | W3A4 | 298.4180 | 1.7811 | 11.77 | 0.4426 | 0.6465 | 31.68 | no | YES |
| W2A4 | W2A4 | 483.0751 | 1.5204 | 7.06 | 0.1669 | 0.9504 | 18.71 | no | no |

---

## Baseline 대비 FID 개선 조합

- **FP16**  FID=-0.0002  IS=1.7172

## Baseline 대비 IS 개선 조합

- **W3A4**  IS=1.7811  FID=298.4180

---

## W3A4-QuIP vs W3A4-GPTQ 비교

| 항목 | W3A4-GPTQ (이전) | W3A4-QuIP (이번) |
|---|---|---|
| FID | 178.4 | (실험 결과 참조) |
| IS | 1.7940 | (실험 결과 참조) |
| 핵심 차이 | Hadamard + SmoothQuant + NVFP4 weight | Randomized Hadamard + INT3 weight |

---

## 저장 경로

```
results/quip_experiment/
  FP16/MJHQ/metrics.json
  BASELINE/MJHQ/metrics.json
  W3A4/MJHQ/metrics.json
  W2A4/MJHQ/metrics.json
  results_summary.json
logs/quip_experiment/
  run.log
  fp16_*.log
  baseline_*.log
  w3a4_*.log
  w2a4_*.log
```

---

## 다음 단계 제안

1. **W3A4-QuIP가 GPTQ(FID=178, IS=1.794)를 이겼다면** → num_samples=100으로 본 실험 재실행
2. **W2A4가 BASELINE을 이겼다면** → 획기적 압축비 달성 (2-bit weight). 100샘플 재실행
3. **W2A4가 실패했다면** → rank 증가 (64→128), 또는 mixed: 첫/마지막 블록 INT3, 나머지 INT2
4. **공통 개선 방향**:
   - cross-attention Q/K/V FP16 유지 (현재 skip은 embedder/proj만)
   - FID reference를 MJHQ 실제 이미지로 교체 (현재는 FP16 충실도 측정)
   - RPCA-style outlier separation 추가 (QuIP + sparse branch)
