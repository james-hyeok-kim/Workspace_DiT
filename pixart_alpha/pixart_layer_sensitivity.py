#!/usr/bin/env python3
"""
pixart_layer_sensitivity.py
Layer-wise FP quantization sensitivity analysis for DiT transformer blocks.

Strategy (3-phase phased search):
  Phase 1 — Format search   : each block tested with 6 formats (block_size=32 fixed, scale=FP16 fixed)
  Phase 2 — Block size tune  : best format per block swept over [16, 32, 64]
  Phase 3 — Scale dtype tune : best format+bs per block swept over 5 scale dtypes
  Phase final               : apply optimal per-layer config, full evaluation

Formats  : NVFP4, MXFP4, MXFP6_E2M3, MXFP6_E3M2, MXFP8, NVFP8
Block sizes: 16, 32, 64
Scale dtypes: FP16, BF16, FP32, NVFP8, MXFP8

Usage:
  accelerate launch --num_processes 2 pixart_layer_sensitivity.py --phase 1 --num_samples 30
  accelerate launch --num_processes 2 pixart_layer_sensitivity.py --phase 2 --num_samples 30
  accelerate launch --num_processes 2 pixart_layer_sensitivity.py --phase 3 --num_samples 30
  accelerate launch --num_processes 2 pixart_layer_sensitivity.py --phase final --num_samples 100
"""

import os, csv, gc, json, copy, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from accelerate import Accelerator
from torchvision.transforms import ToTensor
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from datasets import load_dataset


# =============================================================================
# 1. Scale-dtype simulation
# =============================================================================

def apply_scale_dtype(scale: torch.Tensor, scale_dtype: str) -> torch.Tensor:
    """Simulate scale storage in the requested precision, then back to float32."""
    if scale_dtype == "FP32":
        return scale.float()
    elif scale_dtype == "FP16":
        return scale.half().float()
    elif scale_dtype == "BF16":
        return scale.bfloat16().float()
    elif scale_dtype == "NVFP8":
        # FP8 E4M3 scale (NVIDIA style)
        return scale.to(torch.float8_e4m3fn).float()
    elif scale_dtype == "MXFP8":
        # E8M0: exponent-only, round to nearest power of 2
        safe = scale.clamp(min=2 ** -127)
        return (2.0 ** torch.floor(torch.log2(safe)))
    else:
        raise ValueError(f"Unknown scale_dtype: {scale_dtype}")


# =============================================================================
# 2. MXFP grid generation and efficient nearest-neighbor quantization
# =============================================================================

_GRID_CACHE: dict = {}


def build_mxfp_grid(exp_bits: int, man_bits: int) -> torch.Tensor:
    """
    Generate sorted tensor of all positive representable values for
    FP(exp_bits, man_bits).  Handles edge cases:
      - exp_bits=0 : pure linear (fixed-point), values = m / 2^man_bits
      - man_bits=0 : power-of-2 quantization, values = 2^(e - bias)
      - General     : OCP MX spec, all exponent patterns valid, no Inf/NaN
    """
    key = (exp_bits, man_bits)
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]

    if exp_bits == 0:
        # Pure linear/fixed-point: uniform spacing in [0, 1 - 1/2^man_bits]
        vals = {m / (1 << man_bits) for m in range(1 << man_bits)}
        grid = torch.tensor(sorted(vals), dtype=torch.float32)
        _GRID_CACHE[key] = grid
        return grid

    exp_bias = (1 << (exp_bits - 1)) - 1
    max_exp  = (1 << exp_bits) - 1   # all-ones exponent valid (OCP MX)

    vals = {0.0}
    if man_bits > 0:
        # Subnormals: stored exponent = 0, true exponent = 1 - exp_bias
        for m in range(1, 1 << man_bits):
            vals.add((m / (1 << man_bits)) * (2.0 ** (1 - exp_bias)))
    # Normals: stored exponent >= 1
    for e in range(1, max_exp + 1):
        if man_bits == 0:
            vals.add(2.0 ** (e - exp_bias))          # only 1.0 * 2^exp
        else:
            for m in range(1 << man_bits):
                vals.add((1.0 + m / (1 << man_bits)) * (2.0 ** (e - exp_bias)))

    grid = torch.tensor(sorted(vals), dtype=torch.float32)
    _GRID_CACHE[key] = grid
    return grid


def snap_to_grid(abs_flat: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    Nearest-neighbor snap of |x| to grid values using binary search.
    abs_flat : 1-D float32 tensor of absolute values
    grid     : 1-D sorted float32 grid (on same device)
    Returns  : snapped absolute values (same shape as abs_flat)
    """
    pos = torch.searchsorted(grid, abs_flat)          # insertion point
    lo  = (pos - 1).clamp(min=0)
    hi  = pos.clamp(max=grid.numel() - 1)
    v_lo = grid[lo]
    v_hi = grid[hi]
    use_hi = (abs_flat - v_lo) > (v_hi - abs_flat)
    return torch.where(use_hi, v_hi, v_lo)


def _quant_mxfp(w: torch.Tensor, block_size: int, scale_dtype: str,
                exp_bits: int, man_bits: int) -> torch.Tensor:
    """Generic MXFP block quantization."""
    grid = build_mxfp_grid(exp_bits, man_bits).to(w.device)
    grid_max = grid[-1].item()

    orig = w.shape
    wf = w.reshape(-1, block_size).float()           # (N_blocks, block_size)

    amax  = wf.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale = apply_scale_dtype(amax / grid_max, scale_dtype)  # (N_blocks, 1)

    w_norm = wf / scale.clamp(min=1e-12)             # normalised to grid range
    sign   = w_norm.sign()
    abs_n  = w_norm.abs()

    snapped = snap_to_grid(abs_n.reshape(-1), grid).reshape(abs_n.shape)
    w_q = (sign * snapped * scale).reshape(orig)
    return w_q.to(w.dtype)


# =============================================================================
# 3. Per-format quantization functions
# =============================================================================

_NVFP4_LEVELS: torch.Tensor  # initialised lazily on first call


def quantize_nvfp4(w: torch.Tensor, block_size: int, scale_dtype: str) -> torch.Tensor:
    """NVFP4: non-uniform 4-bit {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}, FP-scale."""
    global _NVFP4_LEVELS
    if not hasattr(quantize_nvfp4, "_levels_cache"):
        quantize_nvfp4._levels_cache = {}
    if w.device not in quantize_nvfp4._levels_cache:
        quantize_nvfp4._levels_cache[w.device] = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=w.device, dtype=torch.float32
        )
    levels = quantize_nvfp4._levels_cache[w.device]

    orig = w.shape
    wf   = w.reshape(-1, block_size).float()
    amax  = wf.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale = apply_scale_dtype(amax / 6.0, scale_dtype)

    w_norm = wf.abs() / scale.clamp(min=1e-12)
    # Nearest-neighbor via searchsorted (levels is sorted)
    snapped = snap_to_grid(w_norm.reshape(-1), levels).reshape(w_norm.shape)
    w_q = (wf.sign() * snapped * scale).reshape(orig)
    return w_q.to(w.dtype)


def quantize_mxfp4(w: torch.Tensor, block_size: int, scale_dtype: str) -> torch.Tensor:
    """MXFP4 E2M1: same value grid as NVFP4, E8M0 scale semantics."""
    # E2M1 all-exponent-valid grid == NVFP4 grid  ({0,.5,1,1.5,2,3,4,6})
    return _quant_mxfp(w, block_size, scale_dtype, exp_bits=2, man_bits=1)


def quantize_mxfp6_e2m3(w: torch.Tensor, block_size: int, scale_dtype: str) -> torch.Tensor:
    """MXFP6 E2M3: 6-bit, exp=2 man=3, max=7.5, E8M0 scale."""
    return _quant_mxfp(w, block_size, scale_dtype, exp_bits=2, man_bits=3)


def quantize_mxfp6_e3m2(w: torch.Tensor, block_size: int, scale_dtype: str) -> torch.Tensor:
    """MXFP6 E3M2: 6-bit, exp=3 man=2, max=28.0, E8M0 scale."""
    return _quant_mxfp(w, block_size, scale_dtype, exp_bits=3, man_bits=2)


def quantize_mxfp8(w: torch.Tensor, block_size: int, scale_dtype: str) -> torch.Tensor:
    """MXFP8 E4M3: 8-bit block-wise, E8M0 scale (OCP MX spec)."""
    orig = w.shape
    wf   = w.reshape(-1, block_size).float()
    q_max = 448.0  # max of float8_e4m3fn
    amax  = wf.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale = apply_scale_dtype(amax / q_max, scale_dtype)
    w_fp8 = (wf / scale.clamp(min=1e-12)).to(torch.float8_e4m3fn).float()
    return (w_fp8 * scale).reshape(orig).to(w.dtype)


def quantize_nvfp8(w: torch.Tensor, block_size: int, scale_dtype: str) -> torch.Tensor:
    """NVFP8 E4M3: same data format as MXFP8 but flexible (non-E8M0) scale."""
    # Structurally same as MXFP8; the distinction is that NVFP8 defaults to
    # FP16/FP32 scale while MXFP8 defaults to E8M0.  Both are swept here.
    orig = w.shape
    wf   = w.reshape(-1, block_size).float()
    q_max = 448.0
    amax  = wf.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale = apply_scale_dtype(amax / q_max, scale_dtype)
    w_fp8 = (wf / scale.clamp(min=1e-12)).to(torch.float8_e4m3fn).float()
    return (w_fp8 * scale).reshape(orig).to(w.dtype)


# ── Custom FP3 formats (E1M1 / E2M0 / E0M2) ──────────────────────────────────
# Grid sizes: 4 positive values each (2^2 sign-excluded patterns)
# E1M1 : {0, 1.0, 2.0, 3.0}  exp_bias=0  → good for moderate-range weights
# E2M0 : {0, 1.0, 2.0, 4.0}  exp_bias=1  → power-of-2 only, very low precision
# E0M2 : {0, 0.25, 0.5, 0.75}            → pure linear, capped range

def quantize_fp3_e1m1(w: torch.Tensor, block_size: int, scale_dtype: str) -> torch.Tensor:
    """FP3 E1M1: 3-bit, exp=1 man=1, grid={0, 1.0, 2.0, 3.0}."""
    return _quant_mxfp(w, block_size, scale_dtype, exp_bits=1, man_bits=1)


def quantize_fp3_e2m0(w: torch.Tensor, block_size: int, scale_dtype: str) -> torch.Tensor:
    """FP3 E2M0: 3-bit, exp=2 man=0, grid={0, 1.0, 2.0, 4.0} (power-of-2)."""
    return _quant_mxfp(w, block_size, scale_dtype, exp_bits=2, man_bits=0)


def quantize_fp3_e0m2(w: torch.Tensor, block_size: int, scale_dtype: str) -> torch.Tensor:
    """FP3 E0M2: 3-bit, exp=0 man=2, grid={0, 0.25, 0.5, 0.75} (linear)."""
    return _quant_mxfp(w, block_size, scale_dtype, exp_bits=0, man_bits=2)


QUANT_FNS = {
    "NVFP4":      quantize_nvfp4,
    "MXFP4":      quantize_mxfp4,
    "MXFP6_E2M3": quantize_mxfp6_e2m3,
    "MXFP6_E3M2": quantize_mxfp6_e3m2,
    "MXFP8":      quantize_mxfp8,
    "NVFP8":      quantize_nvfp8,
    # Custom FP3
    "FP3_E1M1":   quantize_fp3_e1m1,
    "FP3_E2M0":   quantize_fp3_e2m0,
    "FP3_E0M2":   quantize_fp3_e0m2,
}

ALL_FORMATS      = list(QUANT_FNS.keys())
ALL_BLOCK_SIZES  = [16, 32, 64]
ALL_SCALE_DTYPES = ["FP16", "BF16", "FP32", "NVFP8", "MXFP8"]


# =============================================================================
# 4. Weight Statistics & Format Prediction (--phase stats)
# =============================================================================

# bit-width family for accuracy comparison
_FAMILY = {
    "FP3_E1M1": "3bit", "FP3_E2M0": "3bit", "FP3_E0M2": "3bit",
    "NVFP4":    "4bit", "MXFP4":    "4bit",
    "MXFP6_E2M3": "6bit", "MXFP6_E3M2": "6bit",
    "MXFP8":    "8bit", "NVFP8":    "8bit",
}

# Q-SNR threshold (dB): gap = SNR(MXFP8) - SNR(lower_bit)
# smaller gap → lower bit is nearly as good as 8-bit → use it
_T3 = 0.5   # very conservative — no Phase 1 FP3 data yet
_T4 = 1.0   # Phase 1: 4-bit always 16-51 FID worse → almost never fires
_T6 = 0.216 # Phase 1 calibrated: T6 sweep optimal (20/28=71.4% acc, mean Δ FID=0.65)
             # All block gap_6bit ∈ [0.148, 0.727] dB; split at 0.216 maximises accuracy

# Per-channel imbalance thresholds for block_size selection
_IMBALANCE_HI = 1.5   # > 1.5 → bs=16
_IMBALANCE_LO = 0.8   # < 0.8 → bs=64


def analyze_layer_stats(layer_name: str, W: torch.Tensor) -> dict:
    """Compute Q-SNR (per format) and per-channel range imbalance for one layer.

    Measures all sub-formats so that sub-format selection in predict_quant_config_v3
    can pick the best one per bit-width by direct SNR comparison.

    All measurements use block_size=32, scale_dtype=FP16.

    3-bit sub-formats (value grids, all with E8M0 block scale):
      E1M1: {0, 1.0, 2.0, 3.0}           — moderate range + precision
      E2M0: {0, 1.0, 2.0, 4.0}           — power-of-2, wide range, low precision
      E0M2: {0, 0.25, 0.50, 0.75}        — linear/fixed-point, narrow range

    4-bit sub-formats:
      MXFP4 (E2M1, E8M0 scale): {0,.5,1,1.5,2,3,4,6} — OCP MX spec
      NVFP4 (E2M1, FP16 scale): same grid, finer scale → SNR >= MXFP4 always

    6-bit sub-formats:
      MXFP6_E2M3: max=7.5  — better precision, always wins for DiT weights
      MXFP6_E3M2: max=28.0 — wider range but coarser, always inferior here

    gap_Xbit = SNR(MXFP8) - best_SNR(X-bit family)
      → smaller gap = lower bit-width is nearly as good as 8-bit → use it
    """
    W_f32 = W.detach().float()

    # ── Per-channel Range Imbalance ──────────────────────────────────────────
    ch_amax   = W_f32.abs().max(dim=1)[0]          # [out_features]
    imbalance = (ch_amax.std() / ch_amax.mean().clamp(min=1e-8)).item()

    # ── Q-SNR: all sub-formats at bs=32, FP16 scale ──────────────────────────
    _MEASURE_FMTS = [
        "FP3_E1M1", "FP3_E2M0", "FP3_E0M2",   # 3-bit family
        "NVFP4", "MXFP4",                        # 4-bit family
        "MXFP6_E2M3", "MXFP6_E3M2",             # 6-bit family
        "MXFP8",                                 # 8-bit reference
    ]
    sig_norm = W_f32.norm().item()
    snr = {}
    for fmt in _MEASURE_FMTS:
        W_q        = QUANT_FNS[fmt](W_f32, block_size=32, scale_dtype="FP16")
        noise_norm = (W_f32 - W_q).norm().clamp(min=1e-12).item()
        snr[fmt]   = 20.0 * math.log10(max(sig_norm, 1e-12) / noise_norm)

    best_snr_3bit = max(snr["FP3_E1M1"], snr["FP3_E2M0"], snr["FP3_E0M2"])
    best_snr_4bit = max(snr["NVFP4"],    snr["MXFP4"])
    best_snr_6bit = max(snr["MXFP6_E2M3"], snr["MXFP6_E3M2"])

    return {
        "layer":            layer_name,
        "shape":            list(W.shape),
        "imbalance":        round(imbalance, 5),
        # Per-format SNR
        "snr_FP3_E1M1":    round(snr["FP3_E1M1"],    3),
        "snr_FP3_E2M0":    round(snr["FP3_E2M0"],    3),
        "snr_FP3_E0M2":    round(snr["FP3_E0M2"],    3),
        "snr_NVFP4":       round(snr["NVFP4"],        3),
        "snr_MXFP4":       round(snr["MXFP4"],        3),
        "snr_MXFP6_E2M3":  round(snr["MXFP6_E2M3"],  3),
        "snr_MXFP6_E3M2":  round(snr["MXFP6_E3M2"],  3),
        "snr_MXFP8":       round(snr["MXFP8"],        3),
        # Backward-compat aliases
        "snr_FP3":         round(snr["FP3_E1M1"],    3),
        "snr_MXFP4":       round(snr["MXFP4"],        3),
        "snr_MXFP6":       round(snr["MXFP6_E2M3"],  3),
        # gap = SNR(MXFP8) - best_SNR(family): smaller → lower bit acceptable
        "gap_3bit":        round(snr["MXFP8"] - best_snr_3bit, 3),
        "gap_4bit":        round(snr["MXFP8"] - best_snr_4bit, 3),
        "gap_6bit":        round(snr["MXFP8"] - best_snr_6bit, 3),
        # Best sub-format per family (for sub-format selection in v3)
        "best_3bit_fmt":   max({"FP3_E1M1": snr["FP3_E1M1"],
                                "FP3_E2M0": snr["FP3_E2M0"],
                                "FP3_E0M2": snr["FP3_E0M2"]}, key=lambda k: snr[k]),
        "best_4bit_fmt":   "NVFP4" if snr["NVFP4"] >= snr["MXFP4"] else "MXFP4",
        "best_6bit_fmt":   "MXFP6_E2M3" if snr["MXFP6_E2M3"] >= snr["MXFP6_E3M2"]
                           else "MXFP6_E3M2",
    }


def predict_quant_config(block_stats: list, block_idx: int) -> dict:
    """Predict optimal (format, block_size) for one transformer block.

    Step 1 — block_size via per-channel range imbalance (max across 10 layers).
    Step 2 — format via Q-SNR cascade (mean gap across 10 layers).
    """
    # ── Step 1: Block Size ────────────────────────────────────────────────────
    max_imbalance = max(s["imbalance"] for s in block_stats)
    if   max_imbalance > _IMBALANCE_HI: block_size = 16
    elif max_imbalance > _IMBALANCE_LO: block_size = 32
    else:                               block_size = 64

    # ── Step 2: Format (Q-SNR cascade, mean over 10 layers) ──────────────────
    mean_gap_3bit = sum(s["gap_3bit"] for s in block_stats) / len(block_stats)
    mean_gap_4bit = sum(s["gap_4bit"] for s in block_stats) / len(block_stats)
    mean_gap_6bit = sum(s["gap_6bit"] for s in block_stats) / len(block_stats)

    if   mean_gap_3bit < _T3: fmt, path = "FP3_E1M1",     "3bit_ok"
    elif mean_gap_4bit < _T4: fmt, path = "MXFP4",        "4bit_ok"
    elif mean_gap_6bit < _T6: fmt, path = "MXFP6_E2M3",   "6bit_ok"
    else:                     fmt, path = "MXFP8",         "8bit_required"

    return {
        "block_idx":      block_idx,
        "format":         fmt,
        "block_size":     block_size,
        "scale_dtype":    "FP16",
        "max_imbalance":  round(max_imbalance, 5),
        "mean_gap_3bit":  round(mean_gap_3bit, 3),
        "mean_gap_4bit":  round(mean_gap_4bit, 3),
        "mean_gap_6bit":  round(mean_gap_6bit, 3),
        "decision_path":  path,
    }


def calibrate_activations(pipe, transformer: nn.Module, prompts: list,
                          n_calib: int = 8, t_steps: int = 20,
                          device=None) -> dict:
    """Short calibration pass: collect per-layer activation Q-SNR gap & imbalance.

    Runs n_calib prompts through the FP16 pipeline with forward hooks on all
    linear layers.  Each hook computes two scalars per step (no tensor storage):
      - act_gap_6bit : SNR(MXFP8) − SNR(MXFP6_E2M3) of activation tensor  [dB]
      - act_imbalance: std(ch_amax) / mean(ch_amax) of activation tensor

    Returns {layer_name: {"act_gap_6bit": float, "act_imbalance": float}}
    """
    stats: dict[str, list] = defaultdict(list)

    def hook_fn(name):
        def forward_hook(module, inp, out):
            x = inp[0].detach().float()
            flat = x.reshape(-1, x.shape[-1])          # [tokens, hidden]

            # ── Activation Q-SNR gap (same formula as weight Q-SNR) ──
            # Use FP32 scale to avoid FP16 overflow (activations can exceed 65504)
            x_t  = flat.T.contiguous()                 # [hidden, tokens]
            x_q6 = QUANT_FNS["MXFP6_E2M3"](x_t, block_size=32,
                                             scale_dtype="FP32").T.contiguous()
            x_q8 = QUANT_FNS["MXFP8"](x_t, block_size=32,
                                        scale_dtype="FP32").T.contiguous()
            sig   = flat.norm().item()
            if sig < 1e-12:   # all-zero activation → no meaningful gap
                a_gap = 0.0
            else:
                n6  = (flat - x_q6).norm().clamp(min=1e-12).item()
                n8  = (flat - x_q8).norm().clamp(min=1e-12).item()
                snr6 = 20.0 * math.log10(sig / n6)
                snr8 = 20.0 * math.log10(sig / n8)
                a_gap = snr8 - snr6
                if not math.isfinite(a_gap):
                    a_gap = 0.0

            # ── Activation Imbalance (same formula as weight imbalance) ──
            ch_amax = flat.abs().max(dim=0)[0]         # [hidden]
            a_imb   = (ch_amax.std() / ch_amax.mean().clamp(min=1e-8)).item()

            stats[name].append({"act_gap_6bit": a_gap,
                                "act_imbalance": a_imb})
        return forward_hook

    # Register hooks on all 280 linear layers
    hooks = []
    for block_idx in range(28):
        for layer_name in block_linear_names(transformer, block_idx):
            mod = get_mod(transformer, layer_name)
            hooks.append(mod.register_forward_hook(hook_fn(layer_name)))

    calib_prompts = prompts[:n_calib]
    print(f"  Calibrating with {len(calib_prompts)} prompts × {t_steps} steps …",
          flush=True)
    with torch.no_grad():
        for idx, p in enumerate(calib_prompts):
            gen = torch.Generator(device=device).manual_seed(42 + idx)
            pipe(p, num_inference_steps=t_steps, generator=gen)
            print(f"    prompt {idx+1}/{len(calib_prompts)} done", flush=True)

    for h in hooks:
        h.remove()

    # Aggregate: mean across all timesteps × prompts
    result = {}
    for name, step_list in stats.items():
        result[name] = {
            "act_gap_6bit":  round(
                sum(s["act_gap_6bit"]  for s in step_list) / len(step_list), 5),
            "act_imbalance": round(
                sum(s["act_imbalance"] for s in step_list) / len(step_list), 5),
        }
    return result


def predict_quant_config_v2(block_w_stats: list, block_a_stats: dict,
                            block_idx: int,
                            T_w6: float, T_a6: float,
                            T_imb_hi: float = _IMBALANCE_HI,
                            T_imb_lo: float = _IMBALANCE_LO) -> dict:
    """Predict (format, block_size) using 4 criteria: weight + activation.

    Decision logic:
      block_size: max(weight_imbalance, act_imbalance) vs thresholds
      format:     weight_gap AND act_gap must BOTH say 6-bit is OK
    """
    n = len(block_w_stats)

    # Weight metrics
    w_gap3    = sum(s["gap_3bit"]  for s in block_w_stats) / n
    w_gap4    = sum(s["gap_4bit"]  for s in block_w_stats) / n
    w_gap6    = sum(s["gap_6bit"]  for s in block_w_stats) / n
    w_max_imb = max(s["imbalance"] for s in block_w_stats)

    # Activation metrics
    a_layers   = list(block_a_stats.values())
    a_gap6     = sum(s["act_gap_6bit"]  for s in a_layers) / len(a_layers)
    a_max_imb  = max(s["act_imbalance"] for s in a_layers)

    # ── Block size: combined imbalance ────────────────────────────────────────
    combined_imb = max(w_max_imb, a_max_imb)
    if   combined_imb > T_imb_hi: block_size = 16
    elif combined_imb > T_imb_lo: block_size = 32
    else:                         block_size = 64

    # ── Format: cascade + activation gate ─────────────────────────────────────
    max_kurt = None   # not used in v2 symmetric
    if   w_gap3 < _T3: fmt, path = "FP3_E1M1", "3bit_ok"
    elif w_gap4 < _T4: fmt, path = "MXFP4",    "4bit_ok"
    else:
        weight_ok = (w_gap6 < T_w6)
        act_ok    = (a_gap6 < T_a6)
        if weight_ok and act_ok:
            fmt, path = "MXFP6_E2M3", "6bit_ok"
        elif not weight_ok:
            fmt, path = "MXFP8", "8bit_by_weight"
        else:
            fmt, path = "MXFP8", "8bit_by_activation"

    return {
        "block_idx":          block_idx,
        "format":             fmt,
        "block_size":         block_size,
        "scale_dtype":        "FP16",
        "w_gap_6bit":         round(w_gap6, 3),
        "a_gap_6bit":         round(a_gap6, 3),
        "w_max_imbalance":    round(w_max_imb, 5),
        "a_max_imbalance":    round(a_max_imb, 5),
        "combined_imbalance": round(combined_imb, 5),
        "decision_path":      path,
    }


def run_phase_stats(transformer: nn.Module, base_dir: str,
                    accelerator, device, args) -> None:
    """Phase stats: weight statistics analysis + format prediction.

    No image generation — analyzes static model weights only.
    Outputs:
      weight_stats_analysis.csv   — 280 rows (28 blocks × 10 layers)
      predicted_config.json       — predicted optimal config per block
      prediction_vs_phase1.json   — comparison with Phase 1 empirical results
    """
    import csv
    from scipy.stats import pearsonr

    if accelerator.is_main_process:
        print(f"\n{'='*60}", flush=True)
        print("Phase stats — Weight statistics & format prediction", flush=True)
        print(f"  Thresholds: T3={_T3}dB  T4={_T4}dB  T6={_T6}dB", flush=True)
        print(f"  Imbalance:  hi={_IMBALANCE_HI}  lo={_IMBALANCE_LO}", flush=True)
        print(f"{'='*60}", flush=True)

    # ── Compute stats for all blocks × layers (main process only) ────────────
    all_layer_rows   = []   # for CSV
    predicted_configs = {}  # block_idx → predict_quant_config result

    if accelerator.is_main_process:
        for block_idx in args.target_blocks:
            layer_names = block_linear_names(transformer, block_idx)
            block_stats = []
            for name in layer_names:
                mod   = get_mod(transformer, name)
                stats = analyze_layer_stats(name, mod.weight.data)
                stats["block_idx"] = block_idx
                all_layer_rows.append(stats)
                block_stats.append(stats)

            cfg = predict_quant_config(block_stats, block_idx)
            predicted_configs[block_idx] = cfg
            print(f"  block {block_idx:02d}: {cfg['format']:<14} "
                  f"bs={cfg['block_size']}  imbalance={cfg['max_imbalance']:.3f}  "
                  f"gap_6bit={cfg['mean_gap_6bit']:.2f}dB  "
                  f"({cfg['decision_path']})", flush=True)

        # ── Save weight_stats_analysis.csv ───────────────────────────────────
        csv_path = os.path.join(base_dir, "weight_stats_analysis.csv")
        csv_cols = ["block_idx", "layer", "shape", "imbalance",
                    "snr_FP3_E1M1", "snr_FP3_E2M0", "snr_FP3_E0M2",
                    "snr_NVFP4", "snr_MXFP4",
                    "snr_MXFP6_E2M3", "snr_MXFP6_E3M2", "snr_MXFP8",
                    "gap_3bit", "gap_4bit", "gap_6bit",
                    "best_3bit_fmt", "best_4bit_fmt", "best_6bit_fmt"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_layer_rows)
        print(f"\n✅ Saved: {csv_path}  ({len(all_layer_rows)} rows)", flush=True)

        # ── Save predicted_config.json ────────────────────────────────────────
        pred_out = {}
        for blk, cfg in predicted_configs.items():
            pred_out[f"block_{blk}"] = {
                "format":         cfg["format"],
                "block_size":     cfg["block_size"],
                "scale_dtype":    cfg["scale_dtype"],
                "_max_imbalance": cfg["max_imbalance"],
                "_gap_3bit":      cfg["mean_gap_3bit"],
                "_gap_4bit":      cfg["mean_gap_4bit"],
                "_gap_6bit":      cfg["mean_gap_6bit"],
                "_decision":      cfg["decision_path"],
            }
        pred_path = os.path.join(base_dir, "predicted_config.json")
        with open(pred_path, "w") as f:
            json.dump(pred_out, f, indent=2)
        print(f"✅ Saved: {pred_path}", flush=True)

        # ── Compare with Phase 1 empirical results ────────────────────────────
        p1_best_path = os.path.join(base_dir, "phase1_best_per_block.json")
        p1_raw_dir   = os.path.join(base_dir, "phase1")

        if not os.path.exists(p1_best_path):
            print("\n⚠️  phase1_best_per_block.json not found — skipping comparison.",
                  flush=True)
            return

        with open(p1_best_path) as f:
            p1_best = json.load(f)

        # Load all Phase 1 FID values per block per format
        p1_fid = {}   # p1_fid[block_idx][fmt] = FID
        if os.path.exists(p1_raw_dir):
            for fname in os.listdir(p1_raw_dir):
                if not fname.endswith(".json"):
                    continue
                with open(os.path.join(p1_raw_dir, fname)) as f:
                    d = json.load(f)
                blk = d.get("block_idx")
                fmt = d.get("quant_format")
                fid = d.get("FID")
                if blk is not None and fmt is not None and fid is not None:
                    p1_fid.setdefault(blk, {})[fmt] = fid

        # Per-block comparison
        per_block    = []
        exact_hits   = 0
        family_hits  = 0
        fid_deltas   = []
        gap6_list    = []
        fid_gap_list = []

        for blk in sorted(predicted_configs.keys()):
            key  = f"block_{blk}"
            pred_fmt = predicted_configs[blk]["format"]
            emp_fmt  = p1_best.get(key, {}).get("format", "N/A")
            emp_fid  = p1_best.get(key, {}).get("FID", None)

            exact  = (pred_fmt == emp_fmt)
            family = (_FAMILY.get(pred_fmt) == _FAMILY.get(emp_fmt))
            if exact:  exact_hits  += 1
            if family: family_hits += 1

            # FID delta: predicted format's actual FID - empirical best FID
            pred_fid = p1_fid.get(blk, {}).get(pred_fmt, None)
            delta    = round(pred_fid - emp_fid, 3) if (pred_fid is not None
                                                         and emp_fid is not None) else None
            if delta is not None:
                fid_deltas.append(delta)

            # Collect for SNR-FID correlation
            gap6 = predicted_configs[blk]["mean_gap_6bit"]
            if blk in p1_fid and "MXFP8" in p1_fid[blk] and "MXFP6_E2M3" in p1_fid[blk]:
                fid_gap = p1_fid[blk]["MXFP8"] - p1_fid[blk]["MXFP6_E2M3"]
                gap6_list.append(gap6)
                fid_gap_list.append(fid_gap)

            per_block.append({
                "block":        blk,
                "predicted":    pred_fmt,
                "empirical":    emp_fmt,
                "exact_match":  exact,
                "family_match": family,
                "fid_delta":    delta,
                "gap_6bit_dB":  round(gap6, 3),
                "emp_fid":      round(emp_fid, 3) if emp_fid else None,
                "pred_fid":     round(pred_fid, 3) if pred_fid else None,
            })

        n = len(predicted_configs)
        exact_acc  = exact_hits  / n
        family_acc = family_hits / n
        mean_delta = round(sum(fid_deltas) / len(fid_deltas), 3) if fid_deltas else None
        max_delta  = round(max(fid_deltas), 3)  if fid_deltas else None

        snr_corr = None
        if len(gap6_list) >= 3:
            corr, pval = pearsonr(gap6_list, fid_gap_list)
            snr_corr = round(corr, 4)

        comparison = {
            "thresholds":         {"T3": _T3, "T4": _T4, "T6": _T6,
                                   "imbalance_hi": _IMBALANCE_HI,
                                   "imbalance_lo": _IMBALANCE_LO},
            "exact_accuracy":     round(exact_acc,  4),
            "family_accuracy":    round(family_acc, 4),
            "mean_fid_delta":     mean_delta,
            "max_fid_delta":      max_delta,
            "snr_fid_correlation": snr_corr,
            "n_blocks":           n,
            "per_block":          per_block,
        }

        cmp_path = os.path.join(base_dir, "prediction_vs_phase1.json")
        with open(cmp_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"✅ Saved: {cmp_path}", flush=True)

        # ── Print summary table ───────────────────────────────────────────────
        print(f"\n{'─'*75}", flush=True)
        print(f"{'Blk':>4}  {'Predicted':>14}  {'Empirical':>14}  "
              f"{'Match':>6}  {'FID_delta':>10}  {'gap_6bit':>9}", flush=True)
        print(f"{'─'*75}", flush=True)
        for pb in per_block:
            match_str = "✅ exact" if pb["exact_match"] else (
                        "〜 family" if pb["family_match"] else "❌ miss")
            delta_str = f"{pb['fid_delta']:+.2f}" if pb["fid_delta"] is not None else "  N/A"
            print(f"  {pb['block']:>2}  {pb['predicted']:>14}  {pb['empirical']:>14}  "
                  f"{match_str:>9}  {delta_str:>10}  {pb['gap_6bit_dB']:>7.2f}dB",
                  flush=True)
        print(f"{'─'*75}", flush=True)
        print(f"\n  Exact  accuracy : {exact_acc*100:.1f}%  ({exact_hits}/{n})", flush=True)
        print(f"  Family accuracy : {family_acc*100:.1f}%  ({family_hits}/{n})", flush=True)
        print(f"  Mean FID delta  : {mean_delta if mean_delta is not None else 'N/A'}", flush=True)
        print(f"  Max  FID delta  : {max_delta  if max_delta  is not None else 'N/A'}", flush=True)
        print(f"  SNR-FID corr    : {snr_corr   if snr_corr   is not None else 'N/A'}", flush=True)
        print(f"\n  Target: family_acc≥0.70 | mean_fid_delta≤5.0 | snr_corr≥0.50",
              flush=True)


# =============================================================================
# 4b. Phase criteria — weight + activation 4-criteria algorithm
# =============================================================================

def _loo_cv_criteria(w_gap6_per_block: list, a_gap6_per_block: list,
                     labels_bin: list, T_w6: float, T_a6: float) -> int:
    """Count correct predictions for a given (T_w6, T_a6) pair.

    labels_bin: 1 = MXFP8 optimal, 0 = MXFP6 optimal.
    Prediction: MXFP6 if (w_gap < T_w6 AND a_gap < T_a6), else MXFP8.
    Returns number of correct predictions.
    """
    correct = 0
    for w, a, lbl in zip(w_gap6_per_block, a_gap6_per_block, labels_bin):
        pred = 0 if (w < T_w6 and a < T_a6) else 1  # 0=MXFP6, 1=MXFP8
        if pred == lbl:
            correct += 1
    return correct


def _loo_cv_sweep(w_gap6_list: list, a_gap6_list: list,
                  labels_bin: list) -> tuple:
    """LOO-CV sweep over (T_w6, T_a6) thresholds.

    Returns (best_loo_acc, best_T_w6, best_T_a6, best_in_sample_acc).
    """
    n = len(labels_bin)

    # Candidate thresholds: midpoints between sorted unique values + boundaries
    def cands(vals):
        sv = sorted(set(vals))
        cs = [sv[0] - 0.01]
        for i in range(len(sv) - 1):
            cs.append((sv[i] + sv[i + 1]) / 2.0)
        cs.append(sv[-1] + 0.01)
        return cs

    w_cands = cands(w_gap6_list)
    a_cands = cands(a_gap6_list)

    # In-sample sweep first (fast)
    best_is = (-1, 0, 0)
    for tw in w_cands:
        for ta in a_cands:
            c = _loo_cv_criteria(w_gap6_list, a_gap6_list, labels_bin, tw, ta)
            if c > best_is[0]:
                best_is = (c, tw, ta)

    # LOO-CV for top in-sample thresholds (check top ~20)
    # Collect all threshold pairs with in-sample acc >= best_is - 2
    top_pairs = []
    for tw in w_cands:
        for ta in a_cands:
            c = _loo_cv_criteria(w_gap6_list, a_gap6_list, labels_bin, tw, ta)
            if c >= best_is[0] - 2:
                top_pairs.append((c, tw, ta))

    best_loo = (-1, 0, 0, 0)  # (loo_correct, tw, ta, in_sample)
    for is_c, tw, ta in top_pairs:
        loo_correct = 0
        for i in range(n):
            # Train on n-1
            tw_train = [w_gap6_list[j] for j in range(n) if j != i]
            ta_train = [a_gap6_list[j] for j in range(n) if j != i]
            lb_train = [labels_bin[j]  for j in range(n) if j != i]
            # Find best threshold on train set (reuse the same tw, ta as proxy
            # — full inner sweep would be O(n^3), use outer threshold as warm start)
            c_train = _loo_cv_criteria(tw_train, ta_train, lb_train, tw, ta)
            # Predict held-out
            pred = 0 if (w_gap6_list[i] < tw and a_gap6_list[i] < ta) else 1
            if pred == labels_bin[i]:
                loo_correct += 1
        if (loo_correct, is_c) > (best_loo[0], best_loo[3]):
            best_loo = (loo_correct, tw, ta, is_c)

    return (best_loo[0] / n, best_loo[1], best_loo[2], best_loo[3] / n)


def run_phase_criteria(transformer: nn.Module, pipe, prompts: list,
                       ref_dir: str, base_dir: str,
                       accelerator, device, args) -> None:
    """Phase criteria: Cascade-3 format prediction + block_size tuning.

    Flow:
      1. Weight statistics (analyze_layer_stats per layer)
      2. Phase B: predict_quant_config_v3 (cascade-3) per block
      3. Save cascade_predictions.csv
      4. Compare with Phase 1 empirical results → cascade_vs_empirical.csv
      5. Phase C: block_size tuning → block_size_tuning.csv, block_size_optimal.csv
      6. Save final_config.csv (format + optimal_bs per block)
    """
    import csv as csv_mod

    if accelerator.is_main_process:
        print(f"\n{'='*60}", flush=True)
        print("Phase criteria — Cascade-3 + Block Size Tuning", flush=True)
        print(f"  Cascade-3 thresholds:", flush=True)
        print(f"    T_attn2out={_CASCADE3_T_ATTN2OUT}  "
              f"T_attn1_mean={_CASCADE3_T_ATTN1_MEAN}  "
              f"T_attn2q_imb={_CASCADE3_T_ATTN2Q_IMB}", flush=True)
        print(f"{'='*60}", flush=True)

    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return

    # ── Step 1: Weight statistics ─────────────────────────────────────────────
    print("\n  Step 1/6: Computing weight statistics …", flush=True)
    all_w_stats: dict[int, list] = {}
    all_w_rows = []
    for block_idx in args.target_blocks:
        layer_names = block_linear_names(transformer, block_idx)
        block_stats = []
        for name in layer_names:
            mod   = get_mod(transformer, name)
            stats = analyze_layer_stats(name, mod.weight.data)
            stats["block_idx"] = block_idx
            all_w_rows.append(stats)
            block_stats.append(stats)
        all_w_stats[block_idx] = block_stats
    print(f"    {len(all_w_rows)} layers analysed.", flush=True)

    # Save weight_stats_analysis.csv (reuse from stats phase)
    w_csv_path = os.path.join(base_dir, "weight_stats_analysis.csv")
    if not os.path.exists(w_csv_path):
        w_csv_cols = ["block_idx", "layer", "shape", "imbalance",
                      "snr_FP3_E1M1", "snr_FP3_E2M0", "snr_FP3_E0M2",
                      "snr_NVFP4", "snr_MXFP4",
                      "snr_MXFP6_E2M3", "snr_MXFP6_E3M2", "snr_MXFP8",
                      "gap_3bit", "gap_4bit", "gap_6bit",
                      "best_3bit_fmt", "best_4bit_fmt", "best_6bit_fmt"]
        with open(w_csv_path, "w", newline="") as f:
            w = csv_mod.DictWriter(f, fieldnames=w_csv_cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_w_rows)
        print(f"  ✅ Saved: {w_csv_path}", flush=True)

    # ── Step 2: Phase B — Cascade-3 predictions ──────────────────────────────
    print("\n  Step 2/6: Phase B — Cascade-3 format selection …", flush=True)
    phase_b_results = []
    for block_idx in sorted(all_w_stats.keys()):
        cfg = predict_quant_config_v3(
            all_w_stats[block_idx], block_idx, block_size=32)
        phase_b_results.append(cfg)
        print(f"    block {block_idx:02d}: {cfg['format']:<14}  "
              f"stage={cfg['stage']:<6}  "
              f"attn2out={cfg['w_attn2out_gap6']:.3f}  "
              f"attn1={cfg['w_attn1_mean_gap6']:.3f}  "
              f"attn2q_imb={cfg['w_attn2q_imb']:.3f}", flush=True)

    # ── Step 3: Save cascade_predictions.csv ─────────────────────────────────
    pred_csv_path = os.path.join(base_dir, "cascade_predictions.csv")
    pred_cols = ["block_idx", "format", "block_size", "decision_path", "stage",
                 "w_attn2out_gap6", "w_attn1_mean_gap6", "w_attn2q_imb",
                 "mean_gap_3bit", "mean_gap_4bit",
                 "T_attn2out", "T_attn1_mean", "T_attn2q_imb"]
    with open(pred_csv_path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=pred_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(phase_b_results)
    print(f"\n  ✅ Saved: {pred_csv_path}  ({len(phase_b_results)} rows)", flush=True)

    # Save criteria_predicted_config.json (backward compat)
    pred_out = {}
    for cfg in phase_b_results:
        blk = cfg["block_idx"]
        pred_out[f"block_{blk}"] = {
            "format":           cfg["format"],
            "block_size":       cfg["block_size"],
            "scale_dtype":      cfg["scale_dtype"],
            "_decision":        cfg["decision_path"],
            "_stage":           cfg["stage"],
            "_w_attn2out_gap6": cfg["w_attn2out_gap6"],
        }
    pred_json_path = os.path.join(base_dir, "criteria_predicted_config.json")
    with open(pred_json_path, "w") as f:
        json.dump(pred_out, f, indent=2)
    print(f"  ✅ Saved: {pred_json_path}", flush=True)

    # ── Step 4: Compare with Phase 1 empirical → cascade_vs_empirical.csv ────
    print("\n  Step 4/6: Comparing with Phase 1 empirical results …", flush=True)

    p1_best_path = os.path.join(base_dir, "phase1_best_per_block.json")
    if not os.path.exists(p1_best_path):
        print("  ⚠️  phase1_best_per_block.json not found — skipping comparison.",
              flush=True)
        p1_best = {}
        p1_fid  = {}
    else:
        with open(p1_best_path) as f:
            p1_best = json.load(f)

        p1_raw_dir = os.path.join(base_dir, "phase1")
        p1_fid: dict[int, dict] = {}
        if os.path.exists(p1_raw_dir):
            for fname in os.listdir(p1_raw_dir):
                if not fname.endswith(".json"):
                    continue
                with open(os.path.join(p1_raw_dir, fname)) as f:
                    d = json.load(f)
                blk = d.get("block_idx")
                fmt = d.get("quant_format")
                fid = d.get("FID")
                if blk is not None and fmt is not None and fid is not None:
                    p1_fid.setdefault(blk, {})[fmt] = fid

    vs_rows    = []
    exact_hits = family_hits = 0
    fid_deltas = []
    fid_margins = []

    for cfg in phase_b_results:
        blk      = cfg["block_idx"]
        pred_fmt = cfg["format"]
        key      = f"block_{blk}"
        emp_fmt  = p1_best.get(key, {}).get("format", "N/A") if p1_best else "N/A"
        emp_fid  = p1_best.get(key, {}).get("FID", None)     if p1_best else None

        exact  = (pred_fmt == emp_fmt)
        family = (_FAMILY.get(pred_fmt) == _FAMILY.get(emp_fmt))
        if exact:  exact_hits  += 1
        if family: family_hits += 1

        pred_fid = p1_fid.get(blk, {}).get(pred_fmt, None)
        delta    = round(pred_fid - emp_fid, 3) if (pred_fid and emp_fid) else None
        if delta is not None:
            fid_deltas.append(delta)

        # FID margin: |FID(MXFP8) - FID(MXFP6_E2M3)| for this block
        fid_mxfp8 = p1_fid.get(blk, {}).get("MXFP8", None)
        fid_mxfp6 = p1_fid.get(blk, {}).get("MXFP6_E2M3", None)
        margin    = round(abs(fid_mxfp8 - fid_mxfp6), 3) if (fid_mxfp8 and fid_mxfp6) else None
        if margin is not None:
            fid_margins.append(margin)

        # Borderline: distance from each cascade threshold
        d_attn2out   = abs(cfg["w_attn2out_gap6"]   - _CASCADE3_T_ATTN2OUT)
        d_attn1_mean = abs(cfg["w_attn1_mean_gap6"] - _CASCADE3_T_ATTN1_MEAN)
        d_attn2q_imb = abs(cfg["w_attn2q_imb"]      - _CASCADE3_T_ATTN2Q_IMB)
        min_dist     = round(min(d_attn2out, d_attn1_mean, d_attn2q_imb), 4)
        borderline   = (min_dist < 0.02)   # within 2% of any threshold

        vs_rows.append({
            "block":             blk,
            "predicted":         pred_fmt,
            "empirical":         emp_fmt,
            "exact_match":       exact,
            "family_match":      family,
            "stage":             cfg["stage"],
            "fid_delta":         delta,
            "fid_margin_6v8":    margin,
            "borderline":        borderline,
            "min_threshold_dist": min_dist,
            "w_attn2out_gap6":   cfg["w_attn2out_gap6"],
            "w_attn1_mean_gap6": cfg["w_attn1_mean_gap6"],
            "w_attn2q_imb":      cfg["w_attn2q_imb"],
            "emp_fid":           round(emp_fid, 3) if emp_fid else None,
            "pred_fid":          round(pred_fid, 3) if pred_fid else None,
        })

    n = len(phase_b_results)
    exact_acc  = exact_hits  / n
    family_acc = family_hits / n
    mean_delta = round(sum(fid_deltas) / len(fid_deltas), 3) if fid_deltas else None
    max_delta  = round(max(fid_deltas), 3)  if fid_deltas else None

    # FID-weighted accuracy
    if fid_margins:
        fid_w_acc = round(
            sum(r["fid_margin_6v8"] for r in vs_rows
                if r["exact_match"] and r["fid_margin_6v8"] is not None) /
            sum(m for m in fid_margins), 4)
    else:
        fid_w_acc = None

    # Save cascade_vs_empirical.csv
    vs_csv_path = os.path.join(base_dir, "cascade_vs_empirical.csv")
    vs_cols = ["block", "predicted", "empirical", "exact_match", "family_match",
               "stage", "fid_delta", "fid_margin_6v8", "borderline",
               "min_threshold_dist", "w_attn2out_gap6", "w_attn1_mean_gap6",
               "w_attn2q_imb", "emp_fid", "pred_fid"]
    with open(vs_csv_path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=vs_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(vs_rows)
    print(f"  ✅ Saved: {vs_csv_path}  ({len(vs_rows)} rows)", flush=True)

    # Also save as JSON for compatibility
    comparison_json = {
        "thresholds": {
            "T_attn2out":   _CASCADE3_T_ATTN2OUT,
            "T_attn1_mean": _CASCADE3_T_ATTN1_MEAN,
            "T_attn2q_imb": _CASCADE3_T_ATTN2Q_IMB,
        },
        "exact_accuracy":   round(exact_acc,  4),
        "family_accuracy":  round(family_acc, 4),
        "fid_weighted_acc": fid_w_acc,
        "mean_fid_delta":   mean_delta,
        "max_fid_delta":    max_delta,
        "n_blocks":         n,
        "per_block":        vs_rows,
    }
    cmp_json_path = os.path.join(base_dir, "criteria_vs_phase1.json")
    with open(cmp_json_path, "w") as f:
        json.dump(comparison_json, f, indent=2)
    print(f"  ✅ Saved: {cmp_json_path}", flush=True)

    # Print comparison table
    print(f"\n{'─'*90}", flush=True)
    print(f"{'Blk':>4}  {'Predicted':>14}  {'Empirical':>14}  "
          f"{'Match':>9}  {'Stage':>6}  {'FID_Δ':>7}  {'Margin':>7}  "
          f"{'Border':>6}", flush=True)
    print(f"{'─'*90}", flush=True)
    for r in vs_rows:
        match_str  = "✅ exact" if r["exact_match"] else (
                     "〜 family" if r["family_match"] else "❌ miss")
        delta_str  = f"{r['fid_delta']:+.2f}" if r["fid_delta"] is not None else "   N/A"
        margin_str = f"{r['fid_margin_6v8']:.3f}" if r["fid_margin_6v8"] is not None else "  N/A"
        border_str = "⚠️ " if r["borderline"] else "   "
        print(f"  {r['block']:>2}  {r['predicted']:>14}  {r['empirical']:>14}  "
              f"{match_str:>9}  {r['stage']:>6}  "
              f"{delta_str:>7}  {margin_str:>7}  {border_str}", flush=True)
    print(f"{'─'*90}", flush=True)
    print(f"\n  Exact  accuracy      : {exact_acc*100:.1f}%  ({exact_hits}/{n})",
          flush=True)
    print(f"  Family accuracy      : {family_acc*100:.1f}%  ({family_hits}/{n})",
          flush=True)
    if fid_w_acc:
        print(f"  FID-weighted acc     : {fid_w_acc*100:.1f}%", flush=True)
    print(f"  Mean FID delta       : {mean_delta}", flush=True)
    print(f"  Max  FID delta       : {max_delta}", flush=True)
    borderline_blocks = [r["block"] for r in vs_rows if r["borderline"]]
    if borderline_blocks:
        print(f"  Borderline blocks    : {borderline_blocks}", flush=True)

    # ── Step 5: Phase C — Block Size Tuning ──────────────────────────────────
    print("\n  Step 5/6: Phase C — Block Size Tuning …", flush=True)
    bs_candidates = getattr(args, "bs_tune_candidates", [16, 32, 64, 128, 256])
    bs_tolerance  = getattr(args, "bs_snr_tolerance",  0.5)
    opt_rows = run_block_size_tuning(
        transformer, phase_b_results, base_dir,
        candidates=bs_candidates, snr_tolerance=bs_tolerance)

    # ── Step 6: Save final_config.csv ─────────────────────────────────────────
    print("\n  Step 6/6: Saving final_config.csv …", flush=True)
    final_rows = []
    opt_map = {r["block"]: r for r in opt_rows}
    for cfg in phase_b_results:
        blk     = cfg["block_idx"]
        opt_bs  = opt_map[blk]["optimal_bs"]   if blk in opt_map else 32
        opt_snr = opt_map[blk]["snr_db"]       if blk in opt_map else None
        final_rows.append({
            "block":         blk,
            "format":        cfg["format"],
            "block_size":    opt_bs,
            "scale_dtype":   "FP16",
            "snr_db":        opt_snr,
            "decision_path": cfg["decision_path"],
            "stage":         cfg["stage"],
        })

    final_csv_path = os.path.join(base_dir, "final_config.csv")
    final_cols = ["block", "format", "block_size", "scale_dtype",
                  "snr_db", "decision_path", "stage"]
    with open(final_csv_path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=final_cols)
        w.writeheader()
        w.writerows(final_rows)
    print(f"  ✅ Saved: {final_csv_path}  ({len(final_rows)} rows)", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("  Done. Output files:", flush=True)
    print(f"    cascade_predictions.csv     — Phase B cascade-3 decisions", flush=True)
    print(f"    cascade_vs_empirical.csv    — vs Phase 1 comparison", flush=True)
    print(f"    block_size_tuning.csv       — Phase C SNR-memory sweep", flush=True)
    print(f"    block_size_optimal.csv      — optimal bs per block", flush=True)
    print(f"    final_config.csv            — final (format, bs) per block", flush=True)
    print(f"{'='*60}", flush=True)


# =============================================================================
# 4c. Cascade-3 predictor (predict_quant_config_v3) + Block Size Tuning
# =============================================================================

# Cascade-3 thresholds discovered via LOO-CV ablation (LOO=85.7%, 24/28)
# Adjusted +0.001 from raw ablation values to avoid boundary tie-breaking
# (blocks 1, 18 landed exactly on the raw threshold → wrongly fired MXFP8)
_CASCADE3_T_ATTN2OUT   = 0.100   # w_attn2out_gap6  >= T → MXFP8 (stage 1)
_CASCADE3_T_ATTN1_MEAN = 0.346   # w_attn1_mean_gap6 >= T → MXFP8 (stage 2)
_CASCADE3_T_ATTN2Q_IMB = 0.112   # w_attn2q_imb      <= T → MXFP8 (stage 3)


def classify_layer_type(layer_name: str) -> str:
    """Map a layer name to its functional group.

    Groups:
      attn1_qkv  – self-attn to_q/to_k/to_v
      attn1_out  – self-attn to_out.0
      attn2_q    – cross-attn to_q  (varies per block)
      attn2_kv   – cross-attn to_k / to_v  (constant: text encoder)
      attn2_out  – cross-attn to_out.0
      ff         – feedforward net.0.proj / net.2
    """
    if "attn1" in layer_name:
        return "attn1_out" if "to_out" in layer_name else "attn1_qkv"
    if "attn2" in layer_name:
        if "to_out" in layer_name:
            return "attn2_out"
        if "to_k" in layer_name or "to_v" in layer_name:
            return "attn2_kv"
        return "attn2_q"
    return "ff"


def classify_block_layers(block_w_stats: list) -> dict:
    """Group per-layer stats by functional type.

    Returns {type_str: [stats_dict, ...]} where type_str ∈
    {'attn1_qkv', 'attn1_out', 'attn2_q', 'attn2_kv', 'attn2_out', 'ff'}.
    """
    groups: dict = defaultdict(list)
    for s in block_w_stats:
        groups[classify_layer_type(s["layer"])].append(s)
    return dict(groups)


def _mean_field(items: list, field: str) -> float:
    """Mean of items[i][field], returns 0.0 if items is empty."""
    if not items:
        return 0.0
    return sum(s[field] for s in items) / len(items)


def _best_3bit_format(block_w_stats: list) -> str:
    """Pick FP3 sub-format with highest mean SNR across the block's layers.

    Selection logic:
      - E1M1 {0,1,2,3}:    moderate range + precision  (general case)
      - E2M0 {0,1,2,4}:    wide range, power-of-2 only  (heavy-tail / high kurtosis)
      - E0M2 {0,.25,.5,.75}: linear, narrow range         (uniform, low dynamic range)
    The format with the highest mean per-layer SNR is chosen.
    """
    fmt_snrs = {
        "FP3_E1M1": _mean_field(block_w_stats, "snr_FP3_E1M1"),
        "FP3_E2M0": _mean_field(block_w_stats, "snr_FP3_E2M0"),
        "FP3_E0M2": _mean_field(block_w_stats, "snr_FP3_E0M2"),
    }
    return max(fmt_snrs, key=fmt_snrs.get)


def _best_4bit_format(block_w_stats: list) -> str:
    """Pick 4-bit sub-format with highest mean SNR across the block's layers.

    Selection logic:
      - NVFP4 (E2M1, FP16 scale): finer scale precision → SNR >= MXFP4 always
      - MXFP4 (E2M1, E8M0 scale): OCP MX spec, scale rounded to power-of-2
    Both share the same value grid {0,.5,1,1.5,2,3,4,6}; the only difference is
    scale precision, so NVFP4 is always ≥ MXFP4 in SNR.
    Kept as explicit comparison for generality (future E1M2 support, etc.).
    """
    fmt_snrs = {
        "NVFP4": _mean_field(block_w_stats, "snr_NVFP4"),
        "MXFP4": _mean_field(block_w_stats, "snr_MXFP4"),
    }
    return max(fmt_snrs, key=fmt_snrs.get)


def _best_6bit_format(block_w_stats: list) -> str:
    """Pick 6-bit sub-format with highest mean SNR.

    E2M3 (max=7.5)  wins for DiT weights (concentrated near 0).
    E3M2 (max=28.0) only wins when weights have extreme outliers.
    """
    fmt_snrs = {
        "MXFP6_E2M3": _mean_field(block_w_stats, "snr_MXFP6_E2M3"),
        "MXFP6_E3M2": _mean_field(block_w_stats, "snr_MXFP6_E3M2"),
    }
    return max(fmt_snrs, key=fmt_snrs.get)


def predict_quant_config_v3(block_w_stats: list, block_idx: int,
                             T3: float = _T3,
                             T4: float = _T4,
                             T_attn2out:   float = _CASCADE3_T_ATTN2OUT,
                             T_attn1_mean: float = _CASCADE3_T_ATTN1_MEAN,
                             T_attn2q_imb: float = _CASCADE3_T_ATTN2Q_IMB,
                             block_size: int = 32) -> dict:
    """Predict optimal (format, block_size) via 3-level cascade.

    Level 1 — Bit-width filter (3→4→6→8)
      Uses mean Q-SNR gap across all layers in the block.
      PixArt-Alpha: gap_3bit~20dB, gap_4bit~12.6dB → never fires for 3/4-bit.

    Level 2 — Cascade-3 (6-bit vs 8-bit)
      Uses per-layer-TYPE aggregated features:
        stage 1: w_attn2out_gap6 >= T_attn2out     → MXFP8
        stage 2: w_attn1_mean_gap6 >= T_attn1_mean → MXFP8
        stage 3: w_attn2q_imb <= T_attn2q_imb      → MXFP8
        else:                                       → MXFP6_E2M3

    Level 3 — Sub-format within bit-width (SNR-based)
      3-bit: best of E1M1/E2M0/E0M2 (smallest gap_3bit as proxy)
      4-bit: MXFP4 (always better SNR than NVFP4 for weights)
      6-bit: E2M3 (E3M2 always inferior)
      8-bit: MXFP8 (NVFP8 identical)

    block_size is passed through — Phase C block_size_tuning adjusts it separately.
    """
    # ── Level 1: Bit-width filter ─────────────────────────────────────────────
    mean_gap_3bit = _mean_field(block_w_stats, "gap_3bit")
    mean_gap_4bit = _mean_field(block_w_stats, "gap_4bit")

    if mean_gap_3bit < T3:
        fmt  = _best_3bit_format(block_w_stats)
        path = "3bit_ok"
        return {"block_idx": block_idx, "format": fmt, "block_size": block_size,
                "scale_dtype": "FP16", "decision_path": path,
                "stage": "L1_3bit",
                "w_attn2out_gap6": 0.0, "w_attn1_mean_gap6": 0.0, "w_attn2q_imb": 0.0,
                "mean_gap_3bit": round(mean_gap_3bit, 3),
                "mean_gap_4bit": round(mean_gap_4bit, 3),
                "T_attn2out": T_attn2out, "T_attn1_mean": T_attn1_mean,
                "T_attn2q_imb": T_attn2q_imb}

    if mean_gap_4bit < T4:
        fmt  = _best_4bit_format(block_w_stats)
        path = "4bit_ok"
        return {"block_idx": block_idx, "format": fmt, "block_size": block_size,
                "scale_dtype": "FP16", "decision_path": path,
                "stage": "L1_4bit",
                "w_attn2out_gap6": 0.0, "w_attn1_mean_gap6": 0.0, "w_attn2q_imb": 0.0,
                "mean_gap_3bit": round(mean_gap_3bit, 3),
                "mean_gap_4bit": round(mean_gap_4bit, 3),
                "T_attn2out": T_attn2out, "T_attn1_mean": T_attn1_mean,
                "T_attn2q_imb": T_attn2q_imb}

    # ── Level 2: Cascade-3 (6-bit vs 8-bit) ──────────────────────────────────
    typed = classify_block_layers(block_w_stats)

    attn2out_layers  = typed.get("attn2_out", [])
    attn1_layers     = typed.get("attn1_qkv", []) + typed.get("attn1_out", [])
    attn2q_layers    = typed.get("attn2_q",   [])

    w_attn2out_gap6   = _mean_field(attn2out_layers, "gap_6bit")
    w_attn1_mean_gap6 = _mean_field(attn1_layers,    "gap_6bit")
    w_attn2q_imb      = _mean_field(attn2q_layers,   "imbalance")

    if w_attn2out_gap6 >= T_attn2out:
        fmt, path, stage = "MXFP8",                    "8bit_s1_attn2out", "s1"
    elif w_attn1_mean_gap6 >= T_attn1_mean:
        fmt, path, stage = "MXFP8",                    "8bit_s2_attn1",    "s2"
    elif w_attn2q_imb <= T_attn2q_imb:
        fmt, path, stage = "MXFP8",                    "8bit_s3_attn2q",   "s3"
    else:
        fmt  = _best_6bit_format(block_w_stats)
        path = "6bit_ok"
        stage = "6bit"

    return {
        "block_idx":          block_idx,
        "format":             fmt,
        "block_size":         block_size,
        "scale_dtype":        "FP16",
        "decision_path":      path,
        "stage":              stage,
        "w_attn2out_gap6":    round(w_attn2out_gap6,   4),
        "w_attn1_mean_gap6":  round(w_attn1_mean_gap6, 4),
        "w_attn2q_imb":       round(w_attn2q_imb,      4),
        "mean_gap_3bit":      round(mean_gap_3bit,      3),
        "mean_gap_4bit":      round(mean_gap_4bit,      3),
        "T_attn2out":         T_attn2out,
        "T_attn1_mean":       T_attn1_mean,
        "T_attn2q_imb":       T_attn2q_imb,
    }


# ── Block-size tuning helpers ─────────────────────────────────────────────────

def compute_block_snr_at_bs(transformer: nn.Module, block_idx: int,
                              fmt: str, block_size: int) -> float:
    """Mean Q-SNR (dB) across all linear layers in block at given block_size."""
    snrs = []
    for name in block_linear_names(transformer, block_idx):
        W = get_mod(transformer, name).weight.data.detach().float()
        W_q = QUANT_FNS[fmt](W, block_size=block_size, scale_dtype="FP16")
        sig  = W.norm().item()
        if sig < 1e-12:
            continue
        noise = (W - W_q).norm().clamp(min=1e-12).item()
        snrs.append(20.0 * math.log10(sig / noise))
    return round(sum(snrs) / len(snrs), 4) if snrs else 0.0


def estimate_scale_memory(transformer: nn.Module, block_idx: int,
                           fmt: str, block_size: int) -> int:
    """Estimate scale overhead in bytes for one block at given block_size.

    Each group of block_size weight elements shares one FP16 scale (2 bytes).
    For NVFP4: scale is stored per block_size elements (OCP MX spec).
    """
    total_bytes = 0
    for name in block_linear_names(transformer, block_idx):
        W = get_mod(transformer, name).weight.data
        n_elements = W.numel()
        n_scales   = math.ceil(n_elements / block_size)
        # FP16 scale = 2 bytes each
        total_bytes += n_scales * 2
    return total_bytes


def run_block_size_tuning(transformer: nn.Module, phase_b_results: list,
                           base_dir: str, candidates=None,
                           snr_tolerance: float = 0.5) -> list:
    """Phase C: For each block, sweep block_size candidates and pick optimal.

    Optimal = largest block_size whose SNR >= SNR(bs=32) - snr_tolerance.

    Args:
        transformer:      FP16 model
        phase_b_results:  list of predict_quant_config_v3 dicts (28 items)
        base_dir:         output directory
        candidates:       block sizes to sweep (default [16, 32, 64, 128, 256])
        snr_tolerance:    max allowed SNR drop vs bs=32 (dB)

    Outputs:
        block_size_tuning.csv  — all (block, format, bs, snr, memory_bytes)
        block_size_optimal.csv — chosen (block, format, optimal_bs, snr, memory_bytes)
    """
    if candidates is None:
        candidates = [16, 32, 64, 128, 256]

    import csv as csv_mod
    print(f"\n{'='*60}", flush=True)
    print("Phase C — Block Size Tuning", flush=True)
    print(f"  Candidates: {candidates}  |  SNR tolerance: {snr_tolerance} dB",
          flush=True)
    print(f"{'='*60}", flush=True)

    all_rows  = []
    opt_rows  = []

    for cfg in phase_b_results:
        blk = cfg["block_idx"]
        fmt = cfg["format"]

        # Reference SNR at bs=32
        snr_ref = compute_block_snr_at_bs(transformer, blk, fmt, 32)

        best_bs  = 32
        best_snr = snr_ref
        best_mem = estimate_scale_memory(transformer, blk, fmt, 32)

        for bs in candidates:
            snr = compute_block_snr_at_bs(transformer, blk, fmt, bs)
            mem = estimate_scale_memory(transformer, blk, fmt, bs)
            all_rows.append({
                "block":        blk,
                "format":       fmt,
                "block_size":   bs,
                "snr_db":       snr,
                "memory_bytes": mem,
                "snr_vs_ref":   round(snr - snr_ref, 4),
            })
            # Pick largest bs with acceptable SNR loss
            if snr >= snr_ref - snr_tolerance and bs > best_bs:
                best_bs  = bs
                best_snr = snr
                best_mem = mem

        opt_rows.append({
            "block":        blk,
            "format":       fmt,
            "optimal_bs":   best_bs,
            "snr_db":       best_snr,
            "memory_bytes": best_mem,
            "snr_ref_bs32": snr_ref,
            "snr_delta":    round(best_snr - snr_ref, 4),
        })
        print(f"  block {blk:02d}: {fmt:<14}  optimal_bs={best_bs:3d}  "
              f"snr={best_snr:.2f}dB  Δ={best_snr-snr_ref:+.2f}dB  "
              f"mem={best_mem//1024}KB", flush=True)

    # Save block_size_tuning.csv
    tune_path = os.path.join(base_dir, "block_size_tuning.csv")
    tune_cols = ["block", "format", "block_size", "snr_db", "memory_bytes", "snr_vs_ref"]
    with open(tune_path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=tune_cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n✅ Saved: {tune_path}  ({len(all_rows)} rows)", flush=True)

    # Save block_size_optimal.csv
    opt_path = os.path.join(base_dir, "block_size_optimal.csv")
    opt_cols = ["block", "format", "optimal_bs", "snr_db", "memory_bytes",
                "snr_ref_bs32", "snr_delta"]
    with open(opt_path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=opt_cols)
        w.writeheader()
        w.writerows(opt_rows)
    print(f"✅ Saved: {opt_path}  ({len(opt_rows)} rows)", flush=True)

    return opt_rows


def run_block_size_ablation(transformer: nn.Module, pipe, prompts: list,
                             ref_dir: str, base_dir: str,
                             phase_b_results: list,
                             accelerator, device, args,
                             candidates=None) -> None:
    """Phase C ablation: measure FID/IS for each block_size candidate.

    Applies Phase B format config to all blocks, sweeps block_size uniformly,
    measures full-model FID + IS for each candidate.

    Output: block_size_fid_ablation.csv
      block_size, snr_mean, memory_total_mb, fid, is_score
    """
    if candidates is None:
        candidates = getattr(args, "ablation_block_sizes", [16, 32, 64, 128])

    import csv as csv_mod
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return

    print(f"\n{'='*60}", flush=True)
    print("Phase C Ablation — Block Size FID/IS Sweep", flush=True)
    print(f"  Block sizes: {candidates}", flush=True)
    print(f"  Format config: Phase B cascade-3 decisions", flush=True)
    print(f"{'='*60}", flush=True)

    # Load saved Phase 1 FIDs for baseline reference
    p1_raw_dir = os.path.join(base_dir, "phase1")
    p1_fid = {}
    if os.path.exists(p1_raw_dir):
        for fname in os.listdir(p1_raw_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(p1_raw_dir, fname)) as f:
                d = json.load(f)
            blk = d.get("block_idx")
            fmt = d.get("quant_format")
            fid = d.get("FID")
            if blk is not None and fmt is not None and fid is not None:
                p1_fid.setdefault(blk, {})[fmt] = fid

    abl_rows = []

    for bs in candidates:
        print(f"\n  ── block_size={bs} ──", flush=True)

        # Apply Phase B format + this block_size to all blocks
        from copy import deepcopy
        transformer_copy = deepcopy(transformer)
        for cfg in phase_b_results:
            apply_quant(transformer_copy, cfg["block_idx"],
                        cfg["format"], bs, "FP16", device)

        # Compute mean SNR across all blocks at this bs
        snr_vals = []
        mem_total = 0
        for cfg in phase_b_results:
            snr = compute_block_snr_at_bs(transformer, cfg["block_idx"],
                                           cfg["format"], bs)
            mem = estimate_scale_memory(transformer, cfg["block_idx"],
                                         cfg["format"], bs)
            snr_vals.append(snr)
            mem_total += mem
        mean_snr  = round(sum(snr_vals) / len(snr_vals), 4)
        mem_total_mb = round(mem_total / (1024 * 1024), 3)

        # Evaluate: swap transformer in pipe temporarily
        orig_transformer = pipe.transformer
        pipe.transformer = transformer_copy

        is_val, fid_val = evaluate(pipe, prompts, ref_dir,
                                    os.path.join(base_dir, f"bs_ablation_bs{bs}"),
                                    accelerator, device, args.t_steps,
                                    save_images=False)

        pipe.transformer = orig_transformer
        del transformer_copy
        torch.cuda.empty_cache()

        print(f"    SNR_mean={mean_snr:.2f}dB  mem={mem_total_mb}MB  "
              f"FID={fid_val:.2f}  IS={is_val:.3f}", flush=True)

        abl_rows.append({
            "block_size":    bs,
            "snr_mean_db":   mean_snr,
            "memory_total_mb": mem_total_mb,
            "fid":           round(fid_val, 3),
            "is_score":      round(is_val,  4),
        })

    # Save block_size_fid_ablation.csv
    abl_path = os.path.join(base_dir, "block_size_fid_ablation.csv")
    abl_cols = ["block_size", "snr_mean_db", "memory_total_mb", "fid", "is_score"]
    with open(abl_path, "w", newline="") as f:
        w = csv_mod.DictWriter(f, fieldnames=abl_cols)
        w.writeheader()
        w.writerows(abl_rows)
    print(f"\n✅ Saved: {abl_path}", flush=True)

    # Print summary table
    print(f"\n{'─'*65}", flush=True)
    print(f"{'bs':>5}  {'SNR_mean':>10}  {'Mem(MB)':>9}  {'FID':>8}  {'IS':>8}",
          flush=True)
    print(f"{'─'*65}", flush=True)
    for r in abl_rows:
        print(f"  {r['block_size']:>3}  {r['snr_mean_db']:>9.2f}dB  "
              f"{r['memory_total_mb']:>8.1f}MB  "
              f"{r['fid']:>8.3f}  {r['is_score']:>8.4f}", flush=True)
    print(f"{'─'*65}", flush=True)


# =============================================================================
# 5. DirectQuantLinear — weight-only, no SVD correction
# =============================================================================

class DirectQuantLinear(nn.Module):
    """Drop-in replacement for nn.Linear with static weight quantization."""

    def __init__(self, orig: nn.Linear, fmt: str, block_size: int, scale_dtype: str):
        super().__init__()
        fn = QUANT_FNS[fmt]
        w_q = fn(orig.weight.data, block_size, scale_dtype)
        self.weight     = nn.Parameter(w_q, requires_grad=False)
        self.bias       = nn.Parameter(orig.bias.data.clone(), requires_grad=False) \
                          if orig.bias is not None else None
        self.in_features  = orig.in_features
        self.out_features = orig.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


# =============================================================================
# 5. Module utilities
# =============================================================================

SKIP_KEYWORDS = ["x_embedder", "t_embedder", "proj_out"]


def get_mod(model: nn.Module, name: str) -> nn.Module:
    for p in name.split("."):
        model = getattr(model, p)
    return model


def set_mod(model: nn.Module, name: str, new: nn.Module):
    parts  = name.split(".")
    parent = get_mod(model, ".".join(parts[:-1])) if len(parts) > 1 else model
    setattr(parent, parts[-1], new)


def block_linear_names(transformer: nn.Module, block_idx: int) -> list:
    prefix = f"transformer_blocks.{block_idx}."
    return [
        n for n, m in transformer.named_modules()
        if isinstance(m, (nn.Linear, DirectQuantLinear))   # BUG FIX: include already-quantized layers
        and n.startswith(prefix)
        and not any(sk in n for sk in SKIP_KEYWORDS)
    ]


def save_weights(transformer: nn.Module, block_idx: int) -> dict:
    """Save FP16 weight copies before quantization.
    Works for both nn.Linear and already-quantized DirectQuantLinear modules."""
    saved = {}
    for n in block_linear_names(transformer, block_idx):
        m = get_mod(transformer, n)
        saved[n] = {
            "weight": m.weight.data.clone(),
            "bias":   m.bias.data.clone() if m.bias is not None else None,
        }
    return saved


def apply_quant(transformer: nn.Module, block_idx: int,
                fmt: str, block_size: int, scale_dtype: str, device) -> None:
    for n in block_linear_names(transformer, block_idx):
        orig = get_mod(transformer, n)
        set_mod(transformer, n,
                DirectQuantLinear(orig, fmt, block_size, scale_dtype).to(device))


def restore_fp16(transformer: nn.Module, block_idx: int,
                 saved: dict, device) -> None:
    for n, tensors in saved.items():
        m = get_mod(transformer, n)
        m.weight.data.copy_(tensors["weight"].to(device))
        if m.bias is not None and tensors["bias"] is not None:
            m.bias.data.copy_(tensors["bias"].to(device))


# =============================================================================
# 6. Prompts  (same source as pixart_alpha_quant_b200.py)
# =============================================================================

def get_prompts(num_samples: int, dataset_name: str) -> list:
    fallback = [
        "A professional high-quality photo of a futuristic city with neon lights",
        "A beautiful landscape of mountains during sunset, cinematic lighting",
        "A cute robot holding a flower in a field, highly detailed digital art",
        "A gourmet burger with melting cheese and fresh vegetables on a wooden table",
        "An astronaut walking on a purple planet's surface under a starry sky",
        "A majestic eagle soaring over snow-capped mountain peaks at dawn",
        "A detailed portrait of an elderly woman with wise eyes, natural lighting",
        "An underwater coral reef teeming with colorful tropical fish",
        "A steampunk clockwork city under a red stormy sky, concept art",
        "A serene Japanese garden with cherry blossoms and koi pond",
    ]
    if dataset_name == "MJHQ":
        path, split, key = "xingjianleng/mjhq30k", "test", "text"
        try:
            ds = load_dataset(path, split=split, streaming=True)
            prompts = []
            for i, entry in enumerate(ds):
                if i >= num_samples:
                    break
                prompts.append(entry[key])
            if len(prompts) >= num_samples:
                return prompts[:num_samples]
        except Exception as e:
            print(f"⚠️ Dataset load failed ({e}). Using fallback prompts.", flush=True)
    extended = (fallback * (num_samples // len(fallback) + 1))[:num_samples]
    return extended


# =============================================================================
# 7. Evaluation: IS + FID  (distributed, same pattern as quant_b200.py)
# =============================================================================

def evaluate(pipe, prompts: list, ref_dir: str, save_dir: str,
             accelerator: Accelerator, device, t_steps: int = 20,
             save_images: bool = True) -> tuple:
    """
    Generate images with `pipe`, compute IS and FID.
    Returns (IS_mean, FID).
    """
    os.makedirs(save_dir, exist_ok=True)
    s_count = len(prompts)

    is_m  = InceptionScore().to(device)
    fid_m = FrechetInceptionDistance(feature=2048).to(device)

    indices = list(range(s_count))
    with accelerator.split_between_processes(indices) as local_idx:
        for i in local_idx:
            gen   = torch.Generator(device=device).manual_seed(42 + i)
            q_img = pipe(prompts[i], num_inference_steps=t_steps,
                         generator=gen).images[0]
            if save_images:
                q_img.save(os.path.join(save_dir, f"sample_{i}.png"))

            ref_path = os.path.join(ref_dir, f"ref_{i}.png")
            r_img    = Image.open(ref_path).convert("RGB")

            q_t = (ToTensor()(q_img).unsqueeze(0) * 255).to(torch.uint8).to(device)
            r_t = (ToTensor()(r_img).unsqueeze(0) * 255).to(torch.uint8).to(device)

            is_m.update(q_t)
            fid_m.update(r_t, real=True)
            fid_m.update(q_t, real=False)

    accelerator.wait_for_everyone()
    is_val, _  = is_m.compute()
    fid_val    = fid_m.compute()
    return float(is_val), float(fid_val)


# =============================================================================
# 8. Result I/O helpers
# =============================================================================

def result_path(base_dir: str, phase: int, block_idx: int,
                fmt: str, bs: int, sd: str) -> str:
    d = os.path.join(base_dir, f"phase{phase}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"b{block_idx:02d}_{fmt}_bs{bs}_{sd}.json")


def save_result(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_phase_results(base_dir: str, phase: int) -> list:
    d = os.path.join(base_dir, f"phase{phase}")
    results = []
    if not os.path.isdir(d):
        return results
    for fn in os.listdir(d):
        if fn.endswith(".json"):
            with open(os.path.join(d, fn)) as f:
                results.append(json.load(f))
    return results


def best_per_block(results: list, metric: str = "IS",
                   higher_is_better: bool = True) -> dict:
    by_block = defaultdict(list)
    for r in results:
        by_block[r["block_idx"]].append(r)
    best = {}
    for blk, runs in by_block.items():
        key_fn = (lambda x: x.get(metric, -1e9)) if higher_is_better \
                 else (lambda x: -x.get(metric, 1e9))
        best[blk] = max(runs, key=key_fn)
    return best


def write_csv(base_dir: str, phase: int, results: list) -> None:
    if not results:
        return
    path = os.path.join(base_dir, f"phase{phase}_summary.csv")
    fields = sorted(results[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"📊 CSV saved: {path}", flush=True)


# =============================================================================
# 9. Phase runners
# =============================================================================

def _run_experiment(transformer, pipe, prompts, ref_dir, base_dir,
                    accelerator, device, args,
                    phase: int, block_idx: int,
                    fmt: str, bs: int, sd: str) -> dict | None:
    """
    Run one sensitivity experiment: quantize block_idx with (fmt, bs, sd),
    evaluate, restore FP16.  Returns result dict (main process) or None.
    Skips if result file already exists (checkpoint resume).
    """
    rpath = result_path(base_dir, phase, block_idx, fmt, bs, sd)

    # All processes check for checkpoint (shared FS)
    if os.path.exists(rpath):
        if accelerator.is_main_process:
            print(f"  ⏭  Cached: block={block_idx:02d} {fmt} bs={bs} scale={sd}", flush=True)
        return None  # signal "skipped"

    if accelerator.is_main_process:
        print(f"  🔬 block={block_idx:02d} {fmt} bs={bs} scale={sd}", flush=True)

    # Save FP16 weights, apply quant (all processes)
    saved = save_weights(transformer, block_idx)
    apply_quant(transformer, block_idx, fmt, bs, sd, device)

    img_dir = os.path.join(base_dir, f"phase{phase}",
                           f"imgs_b{block_idx:02d}_{fmt}_bs{bs}_{sd}")
    is_val, fid_val = evaluate(pipe, prompts, ref_dir, img_dir,
                               accelerator, device, args.t_steps,
                               save_images=args.save_images)

    restore_fp16(transformer, block_idx, saved, device)

    result = {
        "phase": phase, "block_idx": block_idx,
        "quant_format": fmt, "block_size": bs, "scale_dtype": sd,
        "IS": is_val, "FID": fid_val,
    }

    if accelerator.is_main_process:
        save_result(rpath, result)
        print(f"      IS={is_val:.3f}  FID={fid_val:.2f}", flush=True)

    return result


def run_phase1(transformer, pipe, prompts, ref_dir, base_dir,
               accelerator, device, args) -> None:
    """Phase 1: format search (block_size=32, scale=FP16 fixed)."""
    fixed_bs, fixed_sd = 32, "FP16"
    all_res = load_phase_results(base_dir, 1)  # pre-existing

    if accelerator.is_main_process:
        print(f"\n{'='*60}", flush=True)
        print(f"Phase 1 — Format search  (bs={fixed_bs}, scale={fixed_sd})", flush=True)
        print(f"  Blocks : {args.target_blocks}", flush=True)
        print(f"  Formats: {args.formats}", flush=True)
        total = len(args.target_blocks) * len(args.formats)
        print(f"  Total  : {total} experiments", flush=True)
        print(f"{'='*60}", flush=True)

    for block_idx in args.target_blocks:
        for fmt in args.formats:
            r = _run_experiment(transformer, pipe, prompts, ref_dir, base_dir,
                                accelerator, device, args,
                                1, block_idx, fmt, fixed_bs, fixed_sd)
            if r is not None:
                all_res.append(r)

    if accelerator.is_main_process:
        write_csv(base_dir, 1, all_res)
        best = best_per_block(all_res, "FID", higher_is_better=False)
        report = {
            f"block_{k}": {"format": v["quant_format"],
                           "IS": v["IS"], "FID": v["FID"]}
            for k, v in best.items()
        }
        out = os.path.join(base_dir, "phase1_best_per_block.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Phase 1 done.  Best-per-block → {out}", flush=True)
        for blk in sorted(report):
            b = report[blk]
            print(f"  {blk}: {b['format']}  IS={b['IS']:.3f}  FID={b['FID']:.2f}",
                  flush=True)


def run_phase2(transformer, pipe, prompts, ref_dir, base_dir,
               accelerator, device, args) -> None:
    """Phase 2: block_size search using best format from phase 1."""
    best_path = os.path.join(base_dir, "phase1_best_per_block.json")
    if not os.path.exists(best_path):
        raise FileNotFoundError(
            f"Phase 1 results not found ({best_path}). Run phase 1 first.")
    with open(best_path) as f:
        p1_best = json.load(f)

    fixed_sd = "FP16"
    all_res  = load_phase_results(base_dir, 2)

    if accelerator.is_main_process:
        print(f"\n{'='*60}", flush=True)
        print(f"Phase 2 — Block size search  (scale={fixed_sd} fixed)", flush=True)
        print(f"  Block sizes: {args.block_sizes}", flush=True)
        print(f"{'='*60}", flush=True)

    for block_idx in args.target_blocks:
        key = f"block_{block_idx}"
        if key not in p1_best:
            continue
        best_fmt = p1_best[key]["format"]
        for bs in args.block_sizes:
            r = _run_experiment(transformer, pipe, prompts, ref_dir, base_dir,
                                accelerator, device, args,
                                2, block_idx, best_fmt, bs, fixed_sd)
            if r is not None:
                all_res.append(r)

    if accelerator.is_main_process:
        write_csv(base_dir, 2, all_res)
        best = best_per_block(all_res, "FID", higher_is_better=False)
        report = {
            f"block_{k}": {"format": v["quant_format"], "block_size": v["block_size"],
                           "IS": v["IS"], "FID": v["FID"]}
            for k, v in best.items()
        }
        out = os.path.join(base_dir, "phase2_best_per_block.json")
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Phase 2 done.  Best-per-block → {out}", flush=True)
        for blk in sorted(report):
            b = report[blk]
            print(f"  {blk}: {b['format']} bs={b['block_size']}  "
                  f"IS={b['IS']:.3f}  FID={b['FID']:.2f}", flush=True)


def run_phase3(transformer, pipe, prompts, ref_dir, base_dir,
               accelerator, device, args) -> None:
    """Phase 3: scale dtype search using best format+block_size from phase 2."""
    best_path = os.path.join(base_dir, "phase2_best_per_block.json")
    if not os.path.exists(best_path):
        raise FileNotFoundError(
            f"Phase 2 results not found ({best_path}). Run phase 2 first.")
    with open(best_path) as f:
        p2_best = json.load(f)

    all_res = load_phase_results(base_dir, 3)

    if accelerator.is_main_process:
        print(f"\n{'='*60}", flush=True)
        print(f"Phase 3 — Scale dtype search", flush=True)
        print(f"  Scale dtypes: {args.scale_dtypes}", flush=True)
        print(f"{'='*60}", flush=True)

    for block_idx in args.target_blocks:
        key = f"block_{block_idx}"
        if key not in p2_best:
            continue
        best_fmt = p2_best[key]["format"]
        best_bs  = p2_best[key]["block_size"]
        for sd in args.scale_dtypes:
            r = _run_experiment(transformer, pipe, prompts, ref_dir, base_dir,
                                accelerator, device, args,
                                3, block_idx, best_fmt, best_bs, sd)
            if r is not None:
                all_res.append(r)

    if accelerator.is_main_process:
        write_csv(base_dir, 3, all_res)
        best = best_per_block(all_res, "FID", higher_is_better=False)
        optimal = {
            f"block_{k}": {
                "format":     v["quant_format"],
                "block_size": v["block_size"],
                "scale_dtype": v["scale_dtype"],
                "IS":  v["IS"],
                "FID": v["FID"],
            }
            for k, v in best.items()
        }
        out = os.path.join(base_dir, "optimal_per_layer_config.json")
        with open(out, "w") as f:
            json.dump(optimal, f, indent=2)
        print(f"\n✅ Phase 3 done.  Optimal config → {out}", flush=True)
        for blk in sorted(optimal):
            b = optimal[blk]
            print(f"  {blk}: {b['format']} bs={b['block_size']} scale={b['scale_dtype']}  "
                  f"IS={b['IS']:.3f}  FID={b['FID']:.2f}", flush=True)


def run_final(transformer, pipe, prompts, ref_dir, base_dir,
              accelerator, device, args) -> None:
    """Final: apply optimal per-layer config and run full evaluation."""
    opt_path = os.path.join(base_dir, "optimal_per_layer_config.json")
    if not os.path.exists(opt_path):
        raise FileNotFoundError(
            f"Optimal config not found ({opt_path}). Run phases 1-3 first.")
    with open(opt_path) as f:
        optimal = json.load(f)

    if accelerator.is_main_process:
        print(f"\n{'='*60}", flush=True)
        print("Phase final — Applying optimal per-layer config", flush=True)
        print(f"{'='*60}", flush=True)

    for block_idx in range(28):
        key = f"block_{block_idx}"
        if key in optimal:
            cfg = optimal[key]
            apply_quant(transformer, block_idx,
                        cfg["format"], cfg["block_size"], cfg["scale_dtype"], device)
            if accelerator.is_main_process:
                print(f"  block {block_idx:02d}: {cfg['format']}  "
                      f"bs={cfg['block_size']}  scale={cfg['scale_dtype']}", flush=True)

    final_dir = os.path.join(
        os.path.dirname(base_dir), "final_mixed_precision",
        os.path.basename(base_dir))
    os.makedirs(final_dir, exist_ok=True)

    is_val, fid_val = evaluate(pipe, prompts, ref_dir, final_dir,
                               accelerator, device, args.t_steps,
                               save_images=True)

    if accelerator.is_main_process:
        result = {
            "optimal_config": optimal,
            "IS":  is_val,
            "FID": fid_val,
            "num_samples": len(prompts),
        }
        out = os.path.join(final_dir, "metrics.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n✅ Final:  IS={is_val:.3f}  FID={fid_val:.2f}", flush=True)
        print(f"   Results → {out}", flush=True)


# =============================================================================
# 10. CLI & main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Layer-wise FP quantization sensitivity analysis for DiT")
    p.add_argument("--phase", choices=["1", "2", "3", "final", "stats", "criteria", "bs_ablation"], default="1")
    p.add_argument("--num_samples", type=int, default=30,
                   help="Images per experiment (use 100 for --phase final)")
    p.add_argument("--model_path", type=str,
                   default="PixArt-alpha/PixArt-XL-2-1024-MS")
    p.add_argument("--ref_dir",  type=str, default="ref_images")
    p.add_argument("--save_dir", type=str, default="results/sensitivity")
    p.add_argument("--dataset_name", type=str, default="MJHQ",
                   choices=["MJHQ"])
    p.add_argument("--target_blocks", type=int, nargs="+",
                   default=list(range(28)),
                   help="Which transformer blocks to probe (default: 0-27)")
    p.add_argument("--formats", type=str, nargs="+",
                   default=["NVFP4", "MXFP4", "MXFP6_E2M3", "MXFP6_E3M2", "MXFP8", "NVFP8",
                             "FP3_E1M1", "FP3_E2M0", "FP3_E0M2"],
                   choices=ALL_FORMATS)
    p.add_argument("--block_sizes", type=int, nargs="+",
                   default=ALL_BLOCK_SIZES, choices=ALL_BLOCK_SIZES)
    p.add_argument("--scale_dtypes", type=str, nargs="+",
                   default=ALL_SCALE_DTYPES, choices=ALL_SCALE_DTYPES)
    p.add_argument("--t_steps", type=int, default=20,
                   help="Diffusion sampling steps")
    p.add_argument("--save_images", action="store_true", default=True,
                   help="Save generated images to disk")
    p.add_argument("--no_save_images", dest="save_images", action="store_false",
                   help="Skip saving images (saves disk space)")
    p.add_argument("--test_run", action="store_true",
                   help="Smoke test: 2 samples, block 0 only, all formats")
    p.add_argument("--calib_prompts", type=int, default=8,
                   help="Number of calibration prompts for --phase criteria")
    p.add_argument("--calib_steps", type=int, default=20,
                   help="Diffusion steps per calibration prompt for --phase criteria")
    # Phase C block-size tuning options
    p.add_argument("--bs_tune_candidates", type=int, nargs="+",
                   default=[16, 32, 64, 128, 256],
                   help="Block sizes to evaluate in Phase C tuning")
    p.add_argument("--bs_snr_tolerance", type=float, default=0.5,
                   help="Max SNR drop (dB) vs bs=32 to accept larger block_size")
    # Phase C ablation options
    p.add_argument("--ablation_block_sizes", type=int, nargs="+",
                   default=[16, 32, 64, 128],
                   help="Block sizes to sweep for FID/IS ablation (--phase bs_ablation)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.test_run:
        args.num_samples   = 2
        args.target_blocks = [0]
        print("🧪 test_run mode: 2 samples, block 0 only", flush=True)

    accelerator = Accelerator()
    device      = accelerator.device

    ref_dir  = os.path.join(args.ref_dir,  args.dataset_name)
    base_dir = os.path.join(args.save_dir, args.dataset_name)

    if accelerator.is_main_process:
        os.makedirs(ref_dir,  exist_ok=True)
        os.makedirs(base_dir, exist_ok=True)

    prompts = get_prompts(args.num_samples, args.dataset_name)
    s_count = len(prompts)

    # ── Generate FP16 reference images (skip for weight-only phases) ─────────
    _NO_REF_PHASES = ("stats", "criteria")   # bs_ablation needs refs
    if args.phase in _NO_REF_PHASES and accelerator.is_main_process:
        print(f"ℹ️  Phase {args.phase}: skipping ref image generation.", flush=True)

    if args.phase not in _NO_REF_PHASES and accelerator.is_main_process:
        missing = [i for i in range(s_count)
                   if not os.path.exists(os.path.join(ref_dir, f"ref_{i}.png"))]
        if missing:
            print(f"🌟 Generating {len(missing)} FP16 reference images …", flush=True)
            pipe_ref = PixArtAlphaPipeline.from_pretrained(
                args.model_path, torch_dtype=torch.float16).to(device)
            for i in tqdm(missing, desc="FP16 ref"):
                gen = torch.Generator(device=device).manual_seed(42 + i)
                img = pipe_ref(prompts[i], num_inference_steps=args.t_steps,
                               generator=gen).images[0]
                img.save(os.path.join(ref_dir, f"ref_{i}.png"))
            del pipe_ref
            torch.cuda.empty_cache()
            gc.collect()
        else:
            print(f"✅ All {s_count} reference images already cached.", flush=True)

    accelerator.wait_for_everyone()

    # ── Load model ────────────────────────────────────────────────────────────
    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config)
    transformer = pipe.transformer

    if   args.phase == "1":     run_phase1(transformer, pipe, prompts, ref_dir, base_dir, accelerator, device, args)
    elif args.phase == "2":     run_phase2(transformer, pipe, prompts, ref_dir, base_dir, accelerator, device, args)
    elif args.phase == "3":     run_phase3(transformer, pipe, prompts, ref_dir, base_dir, accelerator, device, args)
    elif args.phase == "final": run_final (transformer, pipe, prompts, ref_dir, base_dir, accelerator, device, args)
    elif args.phase == "stats":    run_phase_stats(transformer, base_dir, accelerator, device, args)
    elif args.phase == "criteria": run_phase_criteria(transformer, pipe, prompts, ref_dir, base_dir, accelerator, device, args)
    elif args.phase == "bs_ablation":
        # Load Phase B results from saved CSV (requires --phase criteria to have run first)
        import csv as _csv
        pred_csv = os.path.join(base_dir, "cascade_predictions.csv")
        if not os.path.exists(pred_csv) and accelerator.is_main_process:
            print(f"⚠️  {pred_csv} not found — run --phase criteria first.", flush=True)
        else:
            phase_b = []
            with open(pred_csv) as f:
                for row in _csv.DictReader(f):
                    phase_b.append({
                        "block_idx":   int(row["block_idx"]),
                        "format":      row["format"],
                        "block_size":  int(row["block_size"]),
                        "scale_dtype": "FP16",
                    })
            run_block_size_ablation(transformer, pipe, prompts, ref_dir, base_dir,
                                    phase_b, accelerator, device, args)

    del pipe
    torch.cuda.empty_cache()
    gc.collect()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("🚀 Done.", flush=True)


if __name__ == "__main__":
    main()
