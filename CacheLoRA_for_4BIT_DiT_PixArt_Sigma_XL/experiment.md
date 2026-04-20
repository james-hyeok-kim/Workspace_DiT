# Experiment: 4-bit DiT Quantization × CacheLoRA — PixArt-Sigma

## Overview

PixArt-Sigma-XL-2-1024-MS 모델에 대해 6가지 NVFP4 양자화 방법과 DeepCache / Cache-LoRA 조합을 전면 비교한다.
이전 실험(Deepcache_for_NVFP4, PixArt-Alpha, 20 samples)의 후속으로, 모델을 Sigma로 교체하고 샘플 수를 100개로 늘려 통계적 신뢰도를 높인다.
또한 ConvRot(Regular Hadamard 기반 그룹 회전 양자화)을 새로 추가하고, LLM-계열 방법(MRGPTQ, FourOverSix)은 제외한다.

---

## Setup

| 항목 | 내용 |
|------|------|
| 모델 | `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` |
| 아키텍처 | DiT, 28 transformer blocks, hidden_dim=1152, FF=4608 |
| 파이프라인 | `PixArtSigmaPipeline` |
| 스케줄러 | DPMSolverMultistepScheduler |
| 해상도 | 1024×1024 |
| guidance_scale | 4.5 |
| Dataset | MJHQ-30K (xingjianleng/mjhq30k, test split) |
| num_samples | 100 |
| 환경 | Python 3.11, torch 2.11.0+cu130, diffusers 0.37.1, accelerate 1.13.0 |
| NVFP4 group size (block_size) | 16 (모든 method 공통) |
| ConvRot group size (N0) | 64 (1152 % 64 = 0) |
| GPU | NVIDIA B200 × 2 (병렬 실행) |

---

## 실험 매트릭스

### 양자화 방법 (6개)

| Method | 논문 / 기법 | 핵심 아이디어 |
|--------|------------|--------------|
| **RTN** | SmoothQuant + NVFP4 RTN | 가장 단순한 기준선. per-channel smooth scale → RTN 양자화 |
| **SVDQUANT** | SVDQuant (NeurIPS 2024) | nvidia-modelopt `mtq.NVFP4_SVDQUANT_DEFAULT_CFG`. SVD low-rank error correction (rank=32) |
| **FP4DIT** | FP4DiT (arXiv 2504) | per-channel FP format selection (E2M1/E1M2/E3M0) + token-wise activation FP4 |
| **HQDIT** | HQ-DiT (arXiv 2504) | Random Hadamard H(64) block-diagonal rotation + per-channel format selection (W4A4) |
| **CONVROT** | ConvRot (arXiv) | Regular Hadamard H4⊗H4⊗H4=H64, group-wise RHT (N0=64) + SmoothQuant + NVFP4 W4A4 |
| **SIXBIT** | 6Bit-Diffusion (arXiv 2504) | per-layer weight sensitivity → 상위 33% → INT8, 나머지 → NVFP4 (평균 ~5.3 bits) |

### 캐싱 방법 (3가지)

| Cache | 설명 |
|-------|------|
| **none** | 캐싱 없음. 기준선 |
| **deepcache** | 블록 [cache_start, cache_end) residual 캐싱, interval=2 |
| **cache_lora** | DeepCache + low-rank corrector (SVD of cross-covariance). rank=4 |

### 실험 파라미터

| 파라미터 | 값 |
|---------|-----|
| num_steps | 5, 10, 15, 20 |
| cache_range | [8, 20), [4, 24), [2, 26) |
| lora_rank | 4 |
| lora_calib | 4 samples |
| calib prompts | min(64, num_samples) |

**총 실행 수**: 6 methods × 28 configs (4 none + 12 deepcache + 12 cache_lora) = **168 runs**

---

## 평가 지표

| 지표 | 방향 | 설명 |
|------|------|------|
| FID | ↓ | Fréchet Inception Distance (vs FP16 ref, 같은 seed) |
| IS | ↑ | Inception Score |
| PSNR (dB) | ↑ | Peak Signal-to-Noise Ratio |
| SSIM | ↑ | Structural Similarity |
| LPIPS | ↓ | Learned Perceptual Image Patch Similarity |
| CLIP | ↑ | CLIP image-text cosine similarity (×100) |
| Time/img (s) | ↓ | 평균 이미지 생성 시간 (캘리브레이션 제외) |

FP16 reference: Sigma 모델로 동일 seed/prompt/steps/guidance로 생성.
ref 경로: `/data/james_dit_pixart_sigma_xl_mjhq/fp16_steps{N}/MJHQ/`

---

## 주요 연구 질문

1. **Cache-friendliness**: DeepCache/Cache-LoRA 적용 시 FID 변화(cache_penalty)가 method마다 다른가?
2. **ConvRot vs RTN**: Regular Hadamard rotation이 RTN 대비 FID를 개선하는가?
3. **Speed-quality tradeoff**: Pareto front에서 어느 method+cache 조합이 optimal한가?
4. **SIXBIT mixed-precision**: ~5.3-bit 평균이 순수 4-bit 대비 실제 품질 이득을 주는가?

---

## 파일 구조

```
CacheLoRA_for_4BIT_DiT_PixArt_Sigma_XL/
├── pixart_nvfp4_cache_compare.py   # 메인 실험 스크립트
├── quant_methods.py                # 6가지 양자화 구현 (ConvRot 포함)
├── deepcache_utils.py              # DeepCache + Cache-LoRA calibration
├── eval_utils.py                   # FID/IS/PSNR/SSIM/LPIPS/CLIP 계산
├── run_step_range_sweep.sh         # 전체 sweep 실행 스크립트 (GPU 분리)
├── update_sweep_csv.py             # metrics.json → sweep_all_results.csv
├── add_best_tags.py                # CSV에 best 태그 추가
├── plot_pareto.py                  # Method별 Pareto front 시각화
├── results/
│   ├── MJHQ/                       # 실험 결과 (168 디렉토리)
│   │   └── {METHOD}_{CACHE}_steps{N}/
│   │       ├── metrics.json
│   │       └── sample_*.png
│   └── sweep_all_results.csv       # 집계 결과
└── experiment.md                   # 이 파일
```

---

## 실행 방법

```bash
# GPU 2개로 병렬 실행
bash run_step_range_sweep.sh --methods RTN,SVDQUANT,FP4DIT --gpu 0 > gpu0.log 2>&1 &
bash run_step_range_sweep.sh --methods HQDIT,CONVROT,SIXBIT --gpu 1 > gpu1.log 2>&1 &

# 결과 집계
python update_sweep_csv.py
python add_best_tags.py
python plot_pareto.py
```

## 이전 실험과의 차이점

| | Deepcache_for_NVFP4 (이전) | CacheLoRA_for_4BIT_DiT_PixArt_Sigma_XL (현재) |
|--|--------------------------|-----------------------------------------------|
| 모델 | PixArt-Alpha | **PixArt-Sigma** |
| Pipeline | PixArtAlphaPipeline | **PixArtSigmaPipeline** |
| Methods | RTN/SVDQUANT/MRGPTQ/FOUROVERSIX/FP4DIT/HQDIT/SIXBIT (7개) | RTN/SVDQUANT/FP4DIT/HQDIT/**CONVROT**/SIXBIT (6개) |
| Samples | 20 | **100** |
| ref 이미지 | `/data/james_dit_pixart_xl_mjhq/` | `/data/james_dit_pixart_sigma_xl_mjhq/` |
