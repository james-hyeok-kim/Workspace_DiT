# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repo researches post-training quantization of **DiT-based diffusion models** (PixArt-Alpha/Sigma). The goal is to find quantization methods that outperform the `NVFP4_SVDQUANT_DEFAULT_CFG` baseline from NVIDIA ModelOpt.

## Research Goal

**Objective:** Beat `NVFP4_SVDQUANT_DEFAULT_CFG` on image quality metrics for DiT quantization.

**Evaluation metrics (priority order):**

| Priority | Metric | Direction | Notes |
|---|---|---|---|
| Primary | FID | lower is better | Distribution-level quality vs. real images |
| Primary | IS | higher is better | Sharpness + diversity of generated images |
| Secondary | PSNR | higher is better | Pixel-level fidelity vs. FP16 teacher |
| Secondary | SSIM | higher is better | Structural similarity vs. FP16 teacher |
| Secondary | LPIPS | lower is better | Perceptual similarity vs. FP16 teacher |
| Secondary | CLIP Score | higher is better | Text-image alignment |

PSNR/SSIM/LPIPS compare quantized output to the FP16 teacher (not ground truth); FID uses real dataset images as the reference distribution.

> **Known gap:** `pixart_alpha_quant.py` (single-GPU legacy script) does not compute IS. `pixart_alpha_quant_b200.py` is the current main script and computes all metrics.

## Setup

```bash
bash run_install_requirements.sh   # or: pip install -r requirements.txt
```

To freeze the current environment:
```bash
bash run_freeze_requirements.sh
```

HuggingFace token: set `HF_TOKEN` in `~/.env` (loaded automatically by B200 scripts) or export it inline.

## Running Experiments

### Single-GPU (primary script: `pixart_alpha_quant.py`)

```bash
# BF16 baseline
bash run_bf16.sh

# NVFP4 weight + FP8 SVD branches, no fine-tuning
bash run_nvfp4_svdfp8_hybrid_notune.sh

# NVFP4 weight + FP8 SVD branches + differential fine-tuning
bash run_nvfp4_svdfp8_hybrid_tune.sh
```

Direct invocation with all options:
```bash
python pixart_alpha_quant.py \
  --model_id "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS" \
  --quant_method NVFP4_ALL \
  --svd_dtype fp8 \
  --fp8_format hybrid \
  --block_size 32 \
  --lowrank 64 \
  --num_samples 100 \
  --save_dir results/ \
  [--do_diff_tuning] \
  [--test_run]     # 2-sample smoke test
```

### Multi-GPU via Accelerate (`pixart_mtq_NVFP4_default.py`)

```bash
bash run_pixart_mtq_nvfp4.sh     # 2-GPU, official NVFP4 SVDQuant config
```

### B200 Act/Weight Sweep (`pixart_alpha_quant_b200.py`)

```bash
bash run_pixart_alpha_quant_b200.sh   # accelerate multi-GPU, loops over ACT_MODES × WGT_MODES
```

Edit `ACT_MODES` / `WGT_MODES` arrays and `TEST_MODE` / `QUANT_METHOD` flags at the top of the script.

## Architecture

### `pixart_alpha_quant_b200.py` — **main script** (multi-GPU, all metrics)

4-stage pipeline using `accelerate`:

1. **Reference generation** — FP16 teacher generates `ref_<i>.png` into `ref_dir/<dataset>/`. Skips existing files so re-runs are fast.
2. **Calibration** — forward hooks collect per-channel activation max across `p_count` prompts (distributed across GPUs, then reduced via mean). Skipped layers: `x_embedder`, `t_embedder`, `proj_out`.
3. **Layer replacement** — each targeted `nn.Linear` in the transformer is replaced with either:
   - `ManualSVDLinear` (`--quant_method SVD`): SmoothQuant scale → quantize weight → SVD on quantization error → low-rank correction branch (lora_a/lora_b).
   - `ManualRPCALinear` (`--quant_method RPCA`): RPCA decomposition first separates outliers into a sparse branch (S), then applies SVD+SmoothQuant on the dense residual (L). Forward = base_out + svd_out + sparse_out.
4. **Evaluation** — distributed image generation, per-GPU metric accumulation (PSNR, SSIM, LPIPS, IS, FID), then main process computes CLIP Score. Metrics output format:

```json
{
  "dist_metrics": {"FID": ..., "IS": ...},
  "averages": {"psnr": ..., "ssim": ..., "lpips": ..., "clip": ...}
}
```

**Act/Weight quantization modes** (`--act_mode` / `--wgt_mode`): `NVFP4`, `INT8`, `INT4`, `INT3`, `INT2`, `TERNARY`

---

### `pixart_alpha_quant.py` — legacy single-GPU script

3-stage pipeline (no IS metric). Uses NVIDIA ModelOpt `mtq.quantize()` directly instead of manual layer replacement. Supports:
- `apply_hybrid_blockwise_quant()` — blockwise FP8 fake-quantization with outlier-aware scaling (top 5% per block excluded from scale calculation).
- `run_diff_tuning()` — master-weight distillation: BF16 master weights updated via MSE loss between teacher and quantized-student outputs, final weights re-quantized to FP8.

**Quantization methods** (`--quant_method`):

| Value | Description |
|---|---|
| `BF16` / `FP16` | Baseline, no quantization |
| `NVFP4_ALL` | Act 4-bit dynamic + Weight 4-bit static |
| `TERNARY_NVFP4` | Act 4-bit + Weight 2-bit (ternary) |
| `NVFP4_SVDQUANT_DEFAULT_CFG` | Official ModelOpt config — the current baseline to beat |
| Any `mtq.*_CFG` name | Passed directly via `getattr(mtq, ...)` |

SVD branch protection: `*high_rank*` and `*low_rank*` weights bypass MTQ at 16-bit, then optionally replaced with blockwise FP8 via `apply_hybrid_blockwise_quant()`.

---

### `pixart_mtq_NVFP4_default.py` — official MTQ baseline (multi-GPU)

Uses `mtq.NVFP4_SVDQUANT_DEFAULT_CFG` unmodified (except injected `lowrank`). Reference images saved to `ref_images/<dataset>/`, results to `results/<dataset>/mtq/`.

## Key Parameters

| Param | Default | Notes |
|---|---|---|
| `--lowrank` | 64 | SVD low-rank branch dimension |
| `--block_size` | 128 | FP8 blockwise quantization block size (16/32/64/128/256) |
| `--svd_dtype` | `fp16` | `fp8` enables hybrid blockwise quant |
| `--fp8_format` | `e4m3` | `e4m3`, `e5m2`, or `hybrid` (auto-selects per-layer) |
| `--num_samples` | 100 | Images to generate; use `--test_run` for 2-sample smoke test |

## Outputs

Results saved to `--save_dir` (default `results/`):
- `results_rank_<R>_<METHOD>_<dtype>_<format>_<block>_<static|diff_tuned>.json` — metrics
- `layer_quant_log_<format>_rank<R>.json` — per-layer FP8 format decisions
- `model_param_names.txt` — full parameter name dump (debug aid)
- `logs/` — shell script tee'd logs
