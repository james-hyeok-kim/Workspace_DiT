# 2:4 Structured Sparsity + Quantization for DiT

## 실험 목표

NVIDIA B200 GPU의 하드웨어 네이티브 2:4 structured sparsity를 quantization과 결합하여,
**NVFP4_SVDQUANT_DEFAULT_CFG baseline**보다 빠르면서 FID/IS가 비슷하거나 좋아지는 방법을 탐색한다.

**Baseline (50 samples, MJHQ)**: FID=126.5, IS=3.332

---

## 핵심 아이디어

FP4 quantization은 weight 크기를 줄이지만, inference 시 memory bandwidth가 bottleneck이 될 수 있다.
2:4 structured sparsity를 추가하면:
- Weight 50%를 zero로 만들어 **effective compute 2x 감소**
- cuSPARSELt의 하드웨어 가속으로 **실제 ~2x throughput 향상** (Ampere 이상)
- FP8 + 2:4 sparsity 조합이 FP4 단독보다 faster + better quality 가능

## 기법 설명

### ManualSparseQuantLinear 클래스

기존 `ManualSVDLinear`를 확장하여 2:4 sparsity를 추가:

```
원본 FP16 weight W
    ↓
[1] 2:4 Sparsity Mask 생성 (Magnitude 또는 SparseGPT)
    - Magnitude: 각 4개 연속 값 중 절댓값 큰 2개 선택 (빠름)
    - SparseGPT: Hessian 기반 최적 2개 선택 + 나머지 보상 업데이트 (느리지만 고품질)
    ↓
[2] W_sparse = W * mask  (50% 0으로 설정)
    ↓
[3] SmoothQuant: smooth_scale = w_max^(1-α) / x_max^α
    ↓
[4] W_smoothed = W_sparse / diag(smooth_scale)
    ↓
[5] W_q = Quantize(W_smoothed)  (INT8 or NVFP4)
    ↓
[6] W_error = W_orig_smoothed - W_q  ← sparsity + quantization 오차 통합
    ↓
[7] SVD(W_error) → lora_a, lora_b  (rank 32)
    ↓
[8] (optional) to_sparse_semi_structured(W_q) → cuSPARSELt 가속 포맷
```

**Forward pass:**
```
x_smoothed = x * smooth_scale          # activation smoothing
x_q = quantize(x_smoothed)             # activation quantization
base_out = sparse_matmul(x_q, W_q^T)  # 2:4 sparse + quantized (cuSPARSELt)
svd_out = x_smoothed @ lora_a^T @ lora_b^T  # error correction (dense, low-rank)
output = base_out + svd_out + bias
```

---

## 실험 매트릭스

### Priority 1: 기본 검증 (반드시 실행)

| Config ID | Sparsity | Weight | Act | Rank | Semi-Struct | 목적 |
|---|---|---|---|---|---|---|
| BASELINE_NVFP4 | none | NVFP4 | NVFP4 | 32 | No | Baseline 재현 확인 |
| SP_MAG_WINT8_AINT8 | magnitude | INT8 | INT8 | 32 | No | 품질 확인 |
| SP_MAG_WINT8_AINT8_SS | magnitude | INT8 | INT8 | 32 | **Yes** | cuSPARSELt 속도 측정 |
| SP_MAG_WFP4_AINT8 | magnitude | NVFP4 | INT8 | 32 | No | 최대 압축 |

### Priority 2: SparseGPT (P1 결과 유망할 경우)

| Config ID | Sparsity | Weight | Act | Rank | 목적 |
|---|---|---|---|---|---|
| SP_GPT_WINT8_AINT8 | sparsegpt | INT8 | INT8 | 32 | Hessian 최적 mask |
| SP_GPT_WFP4_AINT8 | sparsegpt | NVFP4 | INT8 | 32 | 최적 mask + FP4 압축 |
| SP_MAG_WINT4_AINT8 | magnitude | INT4 | INT8 | 32 | 중간 압축 레벨 |
| SP_MAG_WINT8_AINT8_R64 | magnitude | INT8 | INT8 | 64 | 높은 rank SVD |

### Priority 3: Hybrid (P2 이후 결정)

| Config ID | 설명 |
|---|---|
| HYBRID_SENS | Block 0-2, 26-27: INT8+2:4 / Block 3-25: NVFP4+2:4 |

---

## 평가 지표 및 출력

모든 실험은 다음 지표를 계산하고 `results/summary.csv`에 누적 저장:

| 지표 | 방향 | 비고 |
|---|---|---|
| **FID** | **낮을수록** | 실제 이미지 분포 대비 품질 (Primary) |
| **IS** | **높을수록** | 선명도 + 다양성 (Primary) |
| PSNR | 높을수록 | FP16 teacher 대비 픽셀 충실도 |
| SSIM | 높을수록 | 구조 유사도 |
| LPIPS | 낮을수록 | 지각적 유사도 |
| CLIP Score | 높을수록 | 텍스트-이미지 정렬 |
| **avg_latency_ms** | **낮을수록** | 이미지당 평균 생성 시간 |

CSV 컬럼:
```
config_id, sparsity_mode, wgt_mode, act_mode, svd_rank, use_semi_structured,
num_samples, FID, IS, PSNR, SSIM, LPIPS, CLIP, avg_latency_ms, timestamp
```

---

## 가설 및 기대 결과

| 가설 | 근거 |
|---|---|
| INT8+2:4 ≈ NVFP4 품질 | INT8은 8비트, NVFP4는 4비트: 2:4 sparsity로 50% weight 제거해도 남은 weight가 더 정밀 |
| SparseGPT > Magnitude | Hessian 가중 최적화로 더 중요한 weight 보존 |
| Semi-structured = ~2x latency | cuSPARSELt의 하드웨어 2:4 sparse matmul 가속 |
| SVD correction이 key | Sparsity + quantization 합산 오차를 low-rank로 보정 → FID 개선 |

---

## 위험 요소

| 위험 | 대응 |
|---|---|
| SVD branch가 dense이므로 속도 이득 상쇄 가능 | rank 줄이기 (32→16) 또는 SVD branch를 FP8로 |
| `to_sparse_semi_structured` 실패 | try-except로 dense fallback |
| SparseGPT Hessian 수집 메모리 | 280 layers × 5.3MB ≈ 1.5GB, B200 183GB로 충분 |
| 양자화 후 zero 패턴 변경 | 코드에서 assertion으로 검증 |

---

## 파일 구조

```
pixart_sparse_quant/
├── experiment.md              ← 이 파일
├── pixart_sparse_quant.py     ← 메인 실험 스크립트
├── run_sparse_quant.sh        ← 실행 스크립트
├── results/
│   ├── summary.csv            ← 모든 config 결과 누적 CSV
│   ├── BASELINE_NVFP4/        ← config별 결과 폴더
│   ├── SP_MAG_WINT8_AINT8/
│   └── ...
└── logs/
    ├── sparse_quant_BASELINE_NVFP4.log
    └── ...
```

---

## 실행 방법

```bash
# Smoke test (2 samples, 빠른 검증)
TEST_MODE=1 bash run_sparse_quant.sh

# 전체 실험 (100 samples)
TEST_MODE=0 bash run_sparse_quant.sh

# 단일 config 실행 예시
accelerate launch --multi_gpu --num_processes 2 pixart_sparse_quant.py \
    --sparsity_mode magnitude \
    --wgt_mode INT8 \
    --act_mode INT8 \
    --lowrank 32 \
    --num_samples 100 \
    --config_id SP_MAG_WINT8_AINT8
```

---

## 관련 연구

- **SparseGPT** (Frantar & Alistarh, 2023): Hessian 기반 최적 2:4 pruning
- **ASP** (NVIDIA, 2020): Automatic SParsity, magnitude 기반 2:4 pruning
- **SmoothQuant** (Xiao et al., 2022): 활성화-가중치 분포 균형화
- **SVDQuant** (Li et al., 2024): SVD로 quantization 오차 보정
- **cuSPARSELt** (NVIDIA): 하드웨어 가속 2:4 sparse matmul
