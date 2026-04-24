"""
Cascade-3 + Activation Quantization Experiments
=================================================
두 기법의 개별 기여도 비교 (통합 설계 없이):

Method A (cascade_hadamard):
  - Weight: Cascade-3 MXFP6/MXFP8 (per-block, final_config.csv 기반)
    → offline Hadamard rotation 후 QUANT_FNS[fmt] 적용
  - Activation: online Hadamard rotation + NVFP4/INT4/INT8 quantization
  - SmoothQuant 없음
  - SVD error correction (rank=32)

Method B (cascade_smooth):
  - Weight: SmoothQuant scale 적용 후 Cascade-3 MXFP6/MXFP8 quantization
  - Activation: SmoothQuant scale 곱 + NVFP4/INT4/INT8 quantization
  - Hadamard rotation 없음
  - SVD error correction (rank=32)

pipeline: calibrate → replace layers → generate images → compute metrics
"""

import os
import csv
import gc
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToTensor
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler
from accelerate import Accelerator
from transformers import CLIPModel, CLIPProcessor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

from datasets import load_dataset

# Cascade-3 quantization functions (MXFP6_E2M3, MXFP8, etc.)
import sys
sys.path.insert(0, os.path.dirname(__file__))
from pixart_layer_sensitivity import QUANT_FNS, SKIP_KEYWORDS, build_mxfp_grid, snap_to_grid


# =============================================================================
# 0. 유틸리티
# =============================================================================

def get_module_by_name(model, name):
    for part in name.split("."):
        model = getattr(model, part)
    return model


def set_module_by_name(model, name, new_module):
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def block_idx_from_name(layer_name: str) -> int:
    """'transformer_blocks.N.xxx' → N"""
    parts = layer_name.split(".")
    for i, p in enumerate(parts):
        if p == "transformer_blocks" and i + 1 < len(parts):
            return int(parts[i + 1])
    raise ValueError(f"Cannot extract block index from: {layer_name}")


def load_cascade_config(
    csv_path: str,
    override_format: str = None,
    override_scale_dtype: str = None,
    downgrade_config: bool = False,
) -> dict:
    """final_config.csv → {block_idx: (format, block_size, scale_dtype)}

    Override / downgrade options:
      override_format     : 모든 블록을 지정 format으로 강제 (Variant B)
      override_scale_dtype: 모든 블록의 scale_dtype 강제 (Variant A,B,C)
      downgrade_config    : MXFP8→MXFP6_E2M3, MXFP6_E2M3→MXFP4 (Variant C)
    """
    _DOWNGRADE = {
        "MXFP8":      "MXFP6_E2M3",
        "MXFP6_E2M3": "MXFP4",
        "MXFP6_E3M2": "MXFP4",
    }
    cfg = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            b   = int(row["block"])
            fmt = row["format"]
            bs  = int(row["block_size"])
            sd  = row["scale_dtype"]

            if override_format is not None:
                fmt = override_format
            elif downgrade_config:
                fmt = _DOWNGRADE.get(fmt, fmt)

            if override_scale_dtype is not None:
                sd = override_scale_dtype

            cfg[b] = (fmt, bs, sd)
    return cfg


def get_prompts(num_samples: int, dataset_name: str) -> list:
    fallback = [
        "A professional high-quality photo of a futuristic city with neon lights",
        "A beautiful landscape of mountains during sunset, cinematic lighting",
        "A cute robot holding a flower in a field, highly detailed digital art",
        "A gourmet burger with melting cheese and fresh vegetables on a wooden table",
        "An astronaut walking on a purple planet surface under a starry sky",
        "A majestic eagle soaring over snow-capped mountain peaks at dawn",
        "A detailed portrait of an elderly woman with wise eyes, natural lighting",
        "An underwater coral reef teeming with colorful tropical fish",
        "A steampunk clockwork city under a red stormy sky, concept art",
        "A serene Japanese garden with cherry blossoms and koi pond",
    ]
    if dataset_name == "MJHQ":
        try:
            ds = load_dataset("xingjianleng/mjhq30k", split="test", streaming=True)
            prompts = []
            for i, entry in enumerate(ds):
                if i >= num_samples:
                    break
                prompts.append(entry["text"])
            if len(prompts) >= num_samples:
                return prompts[:num_samples]
        except Exception as e:
            print(f"Warning: MJHQ load failed ({e}). Using fallback.", flush=True)
    elif dataset_name == "sDCI":
        try:
            from huggingface_hub import hf_hub_download
            import yaml
            yaml_path = hf_hub_download(
                repo_id="mit-han-lab/svdquant-datasets",
                filename="sDCI.yaml",
                repo_type="dataset",
            )
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            all_prompts = [item["prompt"] for item in data if "prompt" in item]
            return (all_prompts * (num_samples // len(all_prompts) + 1))[:num_samples]
        except Exception as e:
            print(f"Warning: sDCI load failed ({e}). Using fallback.", flush=True)
    return (fallback * (num_samples // len(fallback) + 1))[:num_samples]


# =============================================================================
# 1. Activation quantization helpers (with padding, for dynamic quant)
# =============================================================================

def quantize_to_nvfp4_act(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """NVFP4 dynamic activation quantization with padding support."""
    orig_shape = x.shape
    num_el = x.numel()
    pad = (block_size - num_el % block_size) % block_size
    flat = x.flatten()
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=x.device, dtype=x.dtype)])
    x_flat = flat.view(-1, block_size)
    levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device, dtype=x.dtype
    )
    amax = x_flat.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale = amax / 6.0
    x_norm = x_flat.abs() / scale
    closest_idx = torch.argmin(torch.abs(x_norm.unsqueeze(-1) - levels), dim=-1)
    x_q = torch.sign(x_flat) * levels[closest_idx] * scale
    return x_q.view(-1)[:num_el].view(orig_shape)


def quantize_uniform_act(x: torch.Tensor, block_size: int = 16,
                         mode: str = "INT4") -> torch.Tensor:
    """Uniform symmetric activation quantization with padding support."""
    orig_shape = x.shape
    num_el = x.numel()
    pad = (block_size - num_el % block_size) % block_size
    flat = x.flatten()
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=x.device, dtype=x.dtype)])
    x_flat = flat.view(-1, block_size)
    bits = int(mode.replace("INT", ""))
    q_max = float(2 ** (bits - 1) - 1)
    amax = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / q_max
    x_q = torch.clamp(torch.round(x_flat / scale), -q_max, q_max) * scale
    return x_q.view(-1)[:num_el].view(orig_shape)


# FP3 format configs: name → (exp_bits, man_bits)
_FP3_ACT_CONFIGS = {
    "FP3_E1M1": (1, 1),
    "FP3_E2M0": (2, 0),
    "FP3_E0M2": (0, 2),
}

# 3-bit weight format candidates for per-layer auto selection
_3BIT_WGT_FORMATS = ["FP3_E1M1", "FP3_E2M0", "FP3_E0M2", "INT3"]

# 3-bit act format candidates for per-layer auto selection
_3BIT_ACT_FORMATS = ["FP3_E1M1", "FP3_E2M0", "FP3_E0M2", "INT3"]


def quantize_fp3_act(x: torch.Tensor, block_size: int,
                     exp_bits: int, man_bits: int) -> torch.Tensor:
    """FP3 dynamic activation quantization (per-block scale, FP32 precision)."""
    grid = build_mxfp_grid(exp_bits, man_bits).to(x.device)
    grid_max = grid[-1].item()
    orig_shape = x.shape
    num_el = x.numel()
    pad = (block_size - num_el % block_size) % block_size
    flat = x.flatten()
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=x.device, dtype=x.dtype)])
    x_flat = flat.view(-1, block_size).float()
    amax = x_flat.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale = amax / grid_max
    x_norm = x_flat / scale
    sign = x_norm.sign()
    abs_n = x_norm.abs()
    snapped = snap_to_grid(abs_n.reshape(-1), grid).reshape(abs_n.shape)
    x_q = (sign * snapped * scale).to(x.dtype)
    return x_q.view(-1)[:num_el].view(orig_shape)


def select_best_3bit_act_format(x_sample: torch.Tensor, block_size: int = 16) -> str:
    """Activation sample에 각 3-bit format 적용 → SNR 비교 → best format 반환."""
    x_f = x_sample.float()
    norm_x = x_f.norm().clamp(min=1e-12)
    best_fmt, best_snr = _3BIT_ACT_FORMATS[0], -float("inf")
    for fmt in _3BIT_ACT_FORMATS:
        if fmt in _FP3_ACT_CONFIGS:
            eb, mb = _FP3_ACT_CONFIGS[fmt]
            x_q = quantize_fp3_act(x_f, block_size, eb, mb)
        else:  # INT3
            x_q = quantize_uniform_act(x_f, block_size, "INT3")
        snr = 20.0 * torch.log10(norm_x / (x_f - x_q).norm().clamp(min=1e-12))
        if snr.item() > best_snr:
            best_snr = snr.item()
            best_fmt = fmt
    return best_fmt


def select_best_3bit_wgt_format(W: torch.Tensor, block_size: int,
                                 scale_dtype: str) -> str:
    """Weight에 각 3-bit format 적용 → SNR 비교 → best format 반환."""
    W_f = W.float()
    norm_w = W_f.norm().clamp(min=1e-12)
    best_fmt, best_snr = _3BIT_WGT_FORMATS[0], -float("inf")
    for fmt in _3BIT_WGT_FORMATS:
        W_q = QUANT_FNS[fmt](W_f, block_size, scale_dtype)
        snr = 20.0 * torch.log10(norm_w / (W_f - W_q).norm().clamp(min=1e-12))
        if snr.item() > best_snr:
            best_snr = snr.item()
            best_fmt = fmt
    return best_fmt


def quantize_act(x: torch.Tensor, mode: str = "NVFP4",
                 block_size: int = 16) -> torch.Tensor:
    if mode == "NVFP4":
        return quantize_to_nvfp4_act(x, block_size)
    elif mode in _FP3_ACT_CONFIGS:
        eb, mb = _FP3_ACT_CONFIGS[mode]
        return quantize_fp3_act(x, block_size, eb, mb)
    else:  # INT8, INT4, INT3
        return quantize_uniform_act(x, block_size, mode)


# =============================================================================
# 2. CascadeHadamardLinear
#    Weight: Hadamard rotation → (act_order) → QUANT_FNS[fmt] → SVD error
#    Activation: online Hadamard rotation → NVFP4/INT4 quantization
#    NO SmoothQuant
# =============================================================================

class CascadeHadamardLinear(nn.Module):
    """
    Cascade-3 weight quant + Hadamard rotation + activation quant + SVD

    calibrate 흐름:
      1. W_rot = block_hadamard(W * S_in)           [offline, 한 번만]
      2. H_diag_rot = block-average(H_diag)          [Hadamard 후 등분산]
      3. act_order: sort columns by H_diag_rot
      4. W_q = QUANT_FNS[fmt](W_rot_reordered, cascade_bs, scale_dtype)
      5. W_q_inv = inverse_reorder(W_q)
      6. SVD(W_rot - W_q_inv) → lora_a, lora_b

    forward 흐름:
      x_rot = block_hadamard(x * S_in)
      x_q   = quantize_act(x_rot, NVFP4)            [no SmoothQuant]
      out   = F.linear(x_q, w_quantized) + SVD(x_rot) + bias
    """

    def __init__(self, orig_linear: nn.Linear,
                 fmt: str, cascade_block_size: int, scale_dtype: str,
                 rank: int = 32, had_block_size: int = 128,
                 act_mode: str = "NVFP4", act_block_size: int = 16,
                 dtype: torch.dtype = torch.float16, seed: int = 42):
        super().__init__()
        self.fmt = fmt
        self.cascade_block_size = cascade_block_size
        self.scale_dtype = scale_dtype
        self.rank = rank
        self.had_block_size = had_block_size
        self.act_mode = act_mode
        self.act_block_size = act_block_size
        self.target_dtype = dtype
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.use_rotation = (self.in_features % had_block_size == 0)

        # Original FP16 weight (참조용)
        self.register_buffer("weight", orig_linear.weight.data.clone().to(dtype))
        self.bias = (
            nn.Parameter(orig_linear.bias.data.clone().to(dtype))
            if orig_linear.bias is not None else None
        )

        # Hadamard rotation buffers
        if self.use_rotation:
            gen = torch.Generator()
            gen.manual_seed(seed)
            s = (torch.randint(0, 2, (self.in_features,), generator=gen) * 2 - 1).to(dtype)
            self.register_buffer("S_in", s)
            from scipy.linalg import hadamard as scipy_hadamard
            H_np = scipy_hadamard(had_block_size) / (had_block_size ** 0.5)
            self.register_buffer("H_block", torch.from_numpy(H_np).to(dtype))
        else:
            self.S_in = None
            self.H_block = None

        # Quantized weight + SVD branches (rank=0 → SVD 없음)
        self.register_buffer("w_quantized", orig_linear.weight.data.clone().to(dtype))
        if rank > 0:
            self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features, dtype=dtype))
            self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank, dtype=dtype))
        else:
            self.lora_a = None
            self.lora_b = None
        self.is_calibrated = False

    def _hadamard_rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Last dimension에 random sign + block Hadamard 적용 (weight/activation 공용)"""
        if not self.use_rotation:
            return x
        shape = x.shape
        x_s = x * self.S_in.to(x.dtype)
        n_blocks = shape[-1] // self.had_block_size
        x_3d = x_s.reshape(*shape[:-1], n_blocks, self.had_block_size)
        return (x_3d @ self.H_block.to(x.dtype)).reshape(shape)

    @torch.no_grad()
    def calibrate(self, H_diag: torch.Tensor, x_max: torch.Tensor):
        """
        H_diag: (in_features,) — E[x²] per channel (act_order용)
        x_max:  (in_features,) — per-channel max (진단용, SmoothQuant에 미사용)
        """
        device = self.weight.device
        W = self.weight.float()
        in_f = self.in_features

        # Step 1: Offline Hadamard weight rotation
        if self.use_rotation:
            W_rot = self._hadamard_rotate(W)
            # Rotation 후 H_diag: block 내 평균 (Hadamard 후 채널 간 등분산)
            H_diag = H_diag.float().to(device).clamp(min=1e-8)
            H_rot = H_diag.clone()
            n_blocks = in_f // self.had_block_size
            for b in range(n_blocks):
                s, e = b * self.had_block_size, (b + 1) * self.had_block_size
                H_rot[s:e] = H_diag[s:e].mean()
        else:
            W_rot = W
            H_rot = H_diag.float().to(device).clamp(min=1e-8)

        # Step 2: act_order — 중요 채널 순서로 column 정렬
        order = torch.argsort(H_rot, descending=True)
        inv_order = torch.argsort(order)
        W_rot_reordered = W_rot[:, order].contiguous()

        # Step 3: Cascade-3 MXFP weight quantization (QUANT_FNS 사용)
        W_q_reordered = QUANT_FNS[self.fmt](
            W_rot_reordered, self.cascade_block_size, self.scale_dtype
        )
        # Inverse reorder → original (rotated) column order
        W_q = W_q_reordered[:, inv_order].contiguous()
        self.w_quantized.copy_(W_q.to(self.target_dtype))

        # Step 4: SVD error correction (rank=0이면 skip)
        if self.rank > 0:
            W_err = W_rot - W_q  # (out_f, in_f)
            U, S, Vh = torch.linalg.svd(W_err.float(), full_matrices=False)
            r = min(self.rank, S.shape[0])
            sqrt_S = S[:r].sqrt()

            la = torch.zeros(self.rank, in_f, dtype=self.target_dtype, device=device)
            la[:r] = (Vh[:r] * sqrt_S.unsqueeze(1)).to(self.target_dtype)
            lb = torch.zeros(self.out_features, self.rank, dtype=self.target_dtype, device=device)
            lb[:, :r] = (U[:, :r] * sqrt_S.unsqueeze(0)).to(self.target_dtype)
            self.lora_a.data = la
            self.lora_b.data = lb

        self.is_calibrated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if not self.is_calibrated:
            bias = self.bias.to(input_dtype) if self.bias is not None else None
            return F.linear(x, self.weight.to(input_dtype), bias)

        x_t = x.to(self.target_dtype)

        # Online Hadamard rotation
        x_rot = self._hadamard_rotate(x_t)

        # Activation quantization (no SmoothQuant)
        x_q = quantize_act(x_rot, self.act_mode, self.act_block_size)

        # Base (MXFP weight × quantized act) + SVD correction if rank > 0
        out = F.linear(x_q, self.w_quantized)
        if self.rank > 0:
            out = out + F.linear(F.linear(x_rot, self.lora_a), self.lora_b)
        if self.bias is not None:
            out = out + self.bias
        return out.to(input_dtype)


# =============================================================================
# 3. CascadeSmoothLinear
#    Weight: SmoothQuant scale → QUANT_FNS[fmt] → SVD error
#    Activation: SmoothQuant scale → NVFP4/INT4 quantization
#    NO Hadamard
# =============================================================================

class CascadeSmoothLinear(nn.Module):
    """
    Cascade-3 weight quant + SmoothQuant + activation quant + SVD

    calibrate 흐름:
      1. smooth = w_max^(1-α) / x_max^α  (α=0.5)
      2. W_smooth = W / smooth
      3. W_q = QUANT_FNS[fmt](W_smooth, cascade_bs, scale_dtype)
      4. SVD(W_smooth - W_q) → lora_a, lora_b

    forward 흐름:
      x_smooth = x * smooth_scale
      x_q      = quantize_act(x_smooth, NVFP4)
      out      = F.linear(x_q, w_quantized) + SVD(x_smooth) + bias
    """

    def __init__(self, orig_linear: nn.Linear,
                 fmt: str, cascade_block_size: int, scale_dtype: str,
                 rank: int = 32, alpha: float = 0.5,
                 act_mode: str = "NVFP4", act_block_size: int = 16,
                 dtype: torch.dtype = torch.float16):
        super().__init__()
        self.fmt = fmt
        self.cascade_block_size = cascade_block_size
        self.scale_dtype = scale_dtype
        self.rank = rank
        self.alpha = alpha
        self.act_mode = act_mode
        self.act_block_size = act_block_size
        self.target_dtype = dtype
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features

        # Original FP16 weight (참조용)
        self.register_buffer("weight", orig_linear.weight.data.clone().to(dtype))
        self.bias = (
            nn.Parameter(orig_linear.bias.data.clone().to(dtype))
            if orig_linear.bias is not None else None
        )

        # SmoothQuant scale + quantized weight + SVD branches
        self.register_buffer("smooth_scale", torch.ones(self.in_features, dtype=dtype))
        self.register_buffer("w_quantized", orig_linear.weight.data.clone().to(dtype))
        self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank, dtype=dtype))
        self.is_calibrated = False

    @torch.no_grad()
    def calibrate(self, x_max: torch.Tensor):
        """
        x_max: (in_features,) — per-channel activation max
        """
        device = self.weight.device
        W = self.weight.float()
        x_max = x_max.float().to(device).clamp(min=1e-8)

        # Step 1: SmoothQuant scale (α=0.5 equal split)
        w_max = W.abs().max(dim=0).values.clamp(min=1e-8)
        smooth = (w_max.pow(1 - self.alpha) / x_max.pow(self.alpha)).clamp(1e-4, 1e4)
        self.smooth_scale.data = smooth.to(self.target_dtype)

        # Step 2: Smooth weight
        W_smooth = W / smooth.unsqueeze(0)  # (out_f, in_f)

        # Step 3: Cascade-3 MXFP weight quantization (QUANT_FNS 사용)
        W_q = QUANT_FNS[self.fmt](
            W_smooth, self.cascade_block_size, self.scale_dtype
        )
        self.w_quantized.copy_(W_q.to(self.target_dtype))

        # Step 4: SVD error correction (smoothed space)
        W_err = W_smooth - W_q  # (out_f, in_f)
        U, S, Vh = torch.linalg.svd(W_err.float(), full_matrices=False)
        r = min(self.rank, S.shape[0])
        sqrt_S = S[:r].sqrt()

        la = torch.zeros(self.rank, self.in_features, dtype=self.target_dtype, device=device)
        la[:r] = (Vh[:r] * sqrt_S.unsqueeze(1)).to(self.target_dtype)
        lb = torch.zeros(self.out_features, self.rank, dtype=self.target_dtype, device=device)
        lb[:, :r] = (U[:, :r] * sqrt_S.unsqueeze(0)).to(self.target_dtype)
        self.lora_a.data = la
        self.lora_b.data = lb

        self.is_calibrated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if not self.is_calibrated:
            bias = self.bias.to(input_dtype) if self.bias is not None else None
            return F.linear(x, self.weight.to(input_dtype), bias)

        x_t = x.to(self.target_dtype)

        # SmoothQuant: activation에 smooth_scale 곱하기 (weight에는 나눠져 있음)
        x_smooth = x_t * self.smooth_scale.to(x_t.dtype)

        # Activation quantization
        x_q = quantize_act(x_smooth, self.act_mode, self.act_block_size)

        # Base (MXFP weight × quantized act) + SVD correction (smoothed x)
        out = (
            F.linear(x_q, self.w_quantized)
            + F.linear(F.linear(x_smooth, self.lora_a), self.lora_b)
        )
        if self.bias is not None:
            out = out + self.bias
        return out.to(input_dtype)


# =============================================================================
# 4. CascadeHadamardSmoothLinear
#    Weight: SmoothQuant scale → Hadamard rotation → (act_order) → QUANT_FNS[fmt]
#    Activation: SmoothQuant scale → Hadamard rotation → NVFP4 quantization
#    NO SVD (rank=0 기본)
# =============================================================================

class CascadeHadamardSmoothLinear(nn.Module):
    """
    Cascade-3 weight quant + SmoothQuant + Hadamard rotation + activation quant

    calibrate 흐름:
      1. smooth = w_max^(1-α) / x_max^α     [SmoothQuant scale]
      2. W_smooth = W / smooth
      3. W_rot = block_hadamard(W_smooth)    [offline Hadamard on smoothed weight]
      4. H_rot = block-avg(H_diag)           [Hadamard 후 등분산]
      5. act_order: sort columns by H_rot
      6. W_q = QUANT_FNS[fmt](W_rot_reordered, cascade_bs, scale_dtype)
      7. rank=0 → SVD skip

    forward 흐름:
      x_smooth = x * smooth_scale            [SmoothQuant]
      x_rot    = block_hadamard(x_smooth)    [Hadamard]
      x_q      = quantize_act(x_rot, NVFP4)
      out      = F.linear(x_q, w_quantized) + bias
    """

    def __init__(self, orig_linear: nn.Linear,
                 fmt: str, cascade_block_size: int, scale_dtype: str,
                 rank: int = 0, alpha: float = 0.5,
                 had_block_size: int = 128,
                 act_mode: str = "NVFP4", act_block_size: int = 16,
                 dtype: torch.dtype = torch.float16, seed: int = 42):
        super().__init__()
        self.fmt = fmt
        self.cascade_block_size = cascade_block_size
        self.scale_dtype = scale_dtype
        self.rank = rank
        self.alpha = alpha
        self.had_block_size = had_block_size
        self.act_mode = act_mode
        self.act_block_size = act_block_size
        self.target_dtype = dtype
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.use_rotation = (self.in_features % had_block_size == 0)

        # Original FP16 weight (참조용)
        self.register_buffer("weight", orig_linear.weight.data.clone().to(dtype))
        self.bias = (
            nn.Parameter(orig_linear.bias.data.clone().to(dtype))
            if orig_linear.bias is not None else None
        )

        # SmoothQuant scale
        self.register_buffer("smooth_scale", torch.ones(self.in_features, dtype=dtype))

        # Hadamard rotation buffers
        if self.use_rotation:
            gen = torch.Generator()
            gen.manual_seed(seed)
            s = (torch.randint(0, 2, (self.in_features,), generator=gen) * 2 - 1).to(dtype)
            self.register_buffer("S_in", s)
            from scipy.linalg import hadamard as scipy_hadamard
            H_np = scipy_hadamard(had_block_size) / (had_block_size ** 0.5)
            self.register_buffer("H_block", torch.from_numpy(H_np).to(dtype))
        else:
            self.S_in = None
            self.H_block = None

        # Quantized weight + optional SVD
        self.register_buffer("w_quantized", orig_linear.weight.data.clone().to(dtype))
        if rank > 0:
            self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features, dtype=dtype))
            self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank, dtype=dtype))
        else:
            self.lora_a = None
            self.lora_b = None
        self.is_calibrated = False

    def _hadamard_rotate(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_rotation:
            return x
        shape = x.shape
        x_s = x * self.S_in.to(x.dtype)
        n_blocks = shape[-1] // self.had_block_size
        x_3d = x_s.reshape(*shape[:-1], n_blocks, self.had_block_size)
        return (x_3d @ self.H_block.to(x.dtype)).reshape(shape)

    @torch.no_grad()
    def calibrate(self, H_diag: torch.Tensor, x_max: torch.Tensor):
        """
        H_diag: (in_features,) — E[x²] per channel (act_order용)
        x_max:  (in_features,) — per-channel max (SmoothQuant용)
        """
        device = self.weight.device
        W = self.weight.float()
        in_f = self.in_features
        x_max = x_max.float().to(device).clamp(min=1e-8)

        # Step 1: SmoothQuant scale
        w_max = W.abs().max(dim=0).values.clamp(min=1e-8)
        smooth = (w_max.pow(1 - self.alpha) / x_max.pow(self.alpha)).clamp(1e-4, 1e4)
        self.smooth_scale.data = smooth.to(self.target_dtype)

        # Step 2: Smooth weight
        W_smooth = W / smooth.unsqueeze(0)

        # Step 3: Offline Hadamard rotation on smoothed weight
        if self.use_rotation:
            W_rot = self._hadamard_rotate(W_smooth)
            H_diag = H_diag.float().to(device).clamp(min=1e-8)
            H_rot = H_diag.clone()
            n_blocks = in_f // self.had_block_size
            for b in range(n_blocks):
                s, e = b * self.had_block_size, (b + 1) * self.had_block_size
                H_rot[s:e] = H_diag[s:e].mean()
        else:
            W_rot = W_smooth
            H_rot = H_diag.float().to(device).clamp(min=1e-8)

        # Step 4: act_order — 중요 채널 순서로 column 정렬
        order = torch.argsort(H_rot, descending=True)
        inv_order = torch.argsort(order)
        W_rot_reordered = W_rot[:, order].contiguous()

        # Step 5: Cascade-3 MXFP weight quantization
        W_q_reordered = QUANT_FNS[self.fmt](
            W_rot_reordered, self.cascade_block_size, self.scale_dtype
        )
        W_q = W_q_reordered[:, inv_order].contiguous()
        self.w_quantized.copy_(W_q.to(self.target_dtype))

        # Step 6: SVD (rank > 0일 때만)
        if self.rank > 0:
            W_err = W_rot - W_q
            U, S, Vh = torch.linalg.svd(W_err.float(), full_matrices=False)
            r = min(self.rank, S.shape[0])
            sqrt_S = S[:r].sqrt()
            la = torch.zeros(self.rank, in_f, dtype=self.target_dtype, device=device)
            la[:r] = (Vh[:r] * sqrt_S.unsqueeze(1)).to(self.target_dtype)
            lb = torch.zeros(self.out_features, self.rank, dtype=self.target_dtype, device=device)
            lb[:, :r] = (U[:, :r] * sqrt_S.unsqueeze(0)).to(self.target_dtype)
            self.lora_a.data = la
            self.lora_b.data = lb

        self.is_calibrated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if not self.is_calibrated:
            bias = self.bias.to(input_dtype) if self.bias is not None else None
            return F.linear(x, self.weight.to(input_dtype), bias)

        x_t = x.to(self.target_dtype)

        # SmoothQuant → Hadamard → act quant
        x_smooth = x_t * self.smooth_scale.to(x_t.dtype)
        x_rot = self._hadamard_rotate(x_smooth)
        x_q = quantize_act(x_rot, self.act_mode, self.act_block_size)

        out = F.linear(x_q, self.w_quantized)
        if self.rank > 0:
            out = out + F.linear(F.linear(x_rot, self.lora_a), self.lora_b)
        if self.bias is not None:
            out = out + self.bias
        return out.to(input_dtype)


# =============================================================================
# 5. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cascade-3 + Activation Quant: Hadamard vs SmoothQuant"
    )
    parser.add_argument("--method",         type=str, required=True,
                        choices=["cascade_hadamard", "cascade_smooth", "cascade_hadamard_smooth"],
                        help="cascade_hadamard | cascade_smooth | cascade_hadamard_smooth")
    parser.add_argument("--config_csv",     type=str,
                        default="results/sensitivity/MJHQ/final_config.csv",
                        help="Cascade-3 per-block config CSV (final_config.csv)")
    parser.add_argument("--act_mode",       type=str, default="NVFP4",
                        choices=["NVFP4", "INT8", "INT4", "INT3",
                                 "FP3_E1M1", "FP3_E2M0", "FP3_E0M2", "FP3_AUTO"],
                        help="Activation quantization format (FP3_AUTO: per-layer SNR 선택)")
    parser.add_argument("--act_block_size", type=int, default=16,
                        help="Activation quantization block size")
    parser.add_argument("--lowrank",        type=int, default=32,
                        help="SVD rank for error correction")
    parser.add_argument("--alpha",          type=float, default=0.5,
                        help="SmoothQuant alpha (cascade_smooth only)")
    parser.add_argument("--had_block_size", type=int, default=128,
                        help="Hadamard block size (cascade_hadamard only, must divide in_features)")
    parser.add_argument("--num_samples",    type=int, default=20)
    parser.add_argument("--test_run",       action="store_true",
                        help="2-sample smoke test")
    parser.add_argument("--model_path",     type=str,
                        default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--dataset_name",   type=str, default="MJHQ",
                        choices=["MJHQ", "sDCI"])
    parser.add_argument("--img_base_dir",   type=str,
                        default="/data/jameskimh/james_dit_pixart_xl_mjhq")
    parser.add_argument("--ref_dir",        type=str,
                        default="/data/jameskimh/james_dit_ref/ref_images_fp16",
                        help="FP16 reference image directory")
    parser.add_argument("--save_dir",       type=str,
                        default="results/cascade_act")
    parser.add_argument("--p_count",        type=int, default=64,
                        help="Calibration prompt count (default 64)")
    # Variant overrides (cascade_hadamard only)
    parser.add_argument("--override_format",     type=str, default=None,
                        help="모든 블록 weight format 강제 (e.g. MXFP6_E2M3). Variant B")
    parser.add_argument("--override_scale_dtype", type=str, default=None,
                        help="모든 블록 scale dtype 강제 (e.g. MXFP8=E8M0). Variant A,B,C")
    parser.add_argument("--downgrade_config",    action="store_true",
                        help="MXFP8→MXFP6_E2M3, MXFP6→MXFP4 1단계 하향. Variant C")
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    s_count = 2 if args.test_run else args.num_samples
    p_count = 2 if args.test_run else min(args.p_count, s_count)
    t_count = 20

    # Variant suffix: A (E8M0 only), B (override_format), C (downgrade), G (had+smooth)
    variant_suffix = ""
    if args.method == "cascade_hadamard_smooth":
        wgt_tag = (args.override_format or "casc3").replace("_", "")
        act_tag = args.act_mode.replace("_", "")
        variant_suffix = f"_w{wgt_tag}_a{act_tag}_r{args.lowrank}"
    elif args.method == "cascade_hadamard":
        if args.downgrade_config:
            variant_suffix = "_varC"
        elif args.override_format is not None:
            variant_suffix = "_varB"
        elif args.override_scale_dtype is not None:
            variant_suffix = "_varA"

    dataset_ref_dir  = os.path.join(args.ref_dir,  args.dataset_name)
    _img_rel = args.save_dir
    if "/results/" in _img_rel:
        _img_rel = _img_rel.split("/results/", 1)[1]
    elif _img_rel.startswith("./results/"):
        _img_rel = _img_rel[len("./results/"):]
    elif _img_rel.startswith("results/"):
        _img_rel = _img_rel[len("results/"):]
    dataset_save_dir = os.path.join(args.img_base_dir, _img_rel, f"{args.method}{variant_suffix}")

    if accelerator.is_main_process:
        os.makedirs(dataset_ref_dir,  exist_ok=True)
        os.makedirs(dataset_save_dir, exist_ok=True)

    prompts = get_prompts(s_count, args.dataset_name)
    s_count = len(prompts)

    if accelerator.is_main_process:
        accelerator.print(f"\n{'='*64}")
        accelerator.print(f"  Method      : {args.method}{variant_suffix}")
        accelerator.print(f"  Weight quant: Cascade-3 MXFP (per-block from {args.config_csv})")
        if args.override_format:
            accelerator.print(f"  Override fmt: {args.override_format} (all blocks)")
        if args.override_scale_dtype:
            accelerator.print(f"  Override sdt: {args.override_scale_dtype} (all blocks, MXFP8=E8M0)")
        if args.downgrade_config:
            accelerator.print(f"  Downgrade   : MXFP8→MXFP6_E2M3, MXFP6_E2M3→MXFP4")
        accelerator.print(f"  Act quant   : {args.act_mode}")
        if args.method == "cascade_hadamard":
            accelerator.print(f"  Rotation    : Hadamard block_size={args.had_block_size}")
            accelerator.print(f"  SmoothQuant : No")
        else:
            accelerator.print(f"  SmoothQuant : Yes (alpha={args.alpha})")
            accelerator.print(f"  Rotation    : No")
        accelerator.print(f"  SVD rank    : {args.lowrank}")
        accelerator.print(f"  Samples     : {s_count}  |  Dataset: {args.dataset_name}")
        accelerator.print(f"  Save dir    : {dataset_save_dir}")
        accelerator.print(f"{'='*64}\n")

    # ------------------------------------------------------------------
    # Phase 1: FP16 reference 이미지 (없으면 생성, 있으면 skip)
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        missing = [
            i for i in range(s_count)
            if not os.path.exists(os.path.join(dataset_ref_dir, f"ref_{i}.png"))
        ]
        if missing:
            accelerator.print(f"[Phase 1] Generating {len(missing)} reference images...")
            pipe_ref = PixArtAlphaPipeline.from_pretrained(
                args.model_path, torch_dtype=torch.float16
            ).to(device)
            for i in tqdm(missing, desc="FP16 refs"):
                gen = torch.Generator(device=device).manual_seed(42 + i)
                img = pipe_ref(
                    prompts[i], num_inference_steps=t_count, generator=gen
                ).images[0]
                img.save(os.path.join(dataset_ref_dir, f"ref_{i}.png"))
            del pipe_ref
            torch.cuda.empty_cache()
            gc.collect()
        else:
            accelerator.print("[Phase 1] All reference images exist, skipping.")
    accelerator.wait_for_everyone()

    # ------------------------------------------------------------------
    # Phase 2: 모델 로드
    # ------------------------------------------------------------------
    accelerator.print("[Phase 2] Loading model...")
    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    transformer = pipe.transformer

    # Target layers: transformer_blocks.N.* 만, skip_keywords 제외
    target_names = [
        n for n, m in transformer.named_modules()
        if isinstance(m, nn.Linear)
        and "transformer_blocks." in n
        and not any(kw in n for kw in SKIP_KEYWORDS)
    ]
    if accelerator.is_main_process:
        accelerator.print(f"  Targeted {len(target_names)} Linear layers.")

    # ------------------------------------------------------------------
    # Phase 3: Calibration
    # ------------------------------------------------------------------
    accelerator.print(f"[Phase 3] Calibrating with {p_count} prompts...")
    calib_data: dict = {}

    # Method별 hook: Hadamard는 hdiag + xmax, SmoothQuant는 xmax만
    need_act_sample = args.act_mode == "FP3_AUTO"

    def hook_fn(name):
        def forward_hook(m, inputs, output):
            x = inputs[0].detach().view(-1, inputs[0].shape[-1]).float()
            step_xmax = x.abs().max(dim=0).values.cpu()
            if name not in calib_data:
                calib_data[name] = {"xmax": []}
            calib_data[name]["xmax"].append(step_xmax)
            if args.method in ("cascade_hadamard", "cascade_hadamard_smooth"):
                step_hdiag = x.pow(2).mean(dim=0).cpu()
                if "hdiag" not in calib_data[name]:
                    calib_data[name]["hdiag"] = []
                calib_data[name]["hdiag"].append(step_hdiag)
            # FP3_AUTO: save one activation sample for per-layer format selection
            if need_act_sample and "act_sample" not in calib_data[name]:
                calib_data[name]["act_sample"] = x[:256].cpu()
        return forward_hook

    hooks = [
        get_module_by_name(transformer, n).register_forward_hook(hook_fn(n))
        for n in target_names
    ]
    with torch.no_grad():
        with accelerator.split_between_processes(prompts[:p_count]) as local_prompts:
            for prompt in local_prompts:
                pipe(prompt, num_inference_steps=t_count,
                     generator=torch.Generator(device=device).manual_seed(42))
    for h in hooks:
        h.remove()

    # Distributed reduce (mean)
    for name in calib_data:
        xm_mean = torch.stack(calib_data[name]["xmax"]).mean(dim=0).to(device)
        calib_data[name]["xmax"] = accelerator.reduce(xm_mean, reduction="mean")
        if args.method in ("cascade_hadamard", "cascade_hadamard_smooth"):
            hd_mean = torch.stack(calib_data[name]["hdiag"]).mean(dim=0).to(device)
            calib_data[name]["hdiag"] = accelerator.reduce(hd_mean, reduction="mean")
    accelerator.wait_for_everyone()
    accelerator.print("  Calibration done.")

    # ------------------------------------------------------------------
    # Phase 4: Layer replacement
    # ------------------------------------------------------------------
    accelerator.print("[Phase 4] Loading cascade config and replacing layers...")
    _is_hadamard_method = args.method in ("cascade_hadamard", "cascade_hadamard_smooth")
    cascade_cfg = load_cascade_config(
        args.config_csv,
        override_format=args.override_format if _is_hadamard_method else None,
        override_scale_dtype=args.override_scale_dtype if _is_hadamard_method else None,
        downgrade_config=args.downgrade_config if _is_hadamard_method else False,
    )

    for idx, name in enumerate(tqdm(target_names, desc="Replacing",
                                    disable=not accelerator.is_main_process)):
        orig_m = get_module_by_name(transformer, name)
        if next(orig_m.parameters()).device != device:
            continue

        blk = block_idx_from_name(name)
        fmt, bs, sd = cascade_cfg[blk]

        # Per-layer weight format selection (FP3_AUTO)
        if fmt == "FP3_AUTO":
            fmt = select_best_3bit_wgt_format(
                get_module_by_name(transformer, name).weight.data.float(), bs, sd
            )
            if accelerator.is_main_process and idx < 10:
                accelerator.print(f"    wgt [{name}] → {fmt}")

        # Per-layer act format selection (FP3_AUTO)
        act_mode = args.act_mode
        if act_mode == "FP3_AUTO" and name in calib_data and "act_sample" in calib_data[name]:
            act_mode = select_best_3bit_act_format(
                calib_data[name]["act_sample"].to(device), args.act_block_size
            )
            if accelerator.is_main_process and idx < 10:
                accelerator.print(f"    act [{name}] → {act_mode}")

        if args.method == "cascade_hadamard":
            new_layer = CascadeHadamardLinear(
                orig_m, fmt=fmt, cascade_block_size=bs, scale_dtype=sd,
                rank=args.lowrank, had_block_size=args.had_block_size,
                act_mode=act_mode, act_block_size=args.act_block_size,
                dtype=torch.float16, seed=idx,
            ).to(device)
            if name in calib_data:
                new_layer.calibrate(
                    calib_data[name]["hdiag"],
                    calib_data[name]["xmax"],
                )
        elif args.method == "cascade_hadamard_smooth":
            new_layer = CascadeHadamardSmoothLinear(
                orig_m, fmt=fmt, cascade_block_size=bs, scale_dtype=sd,
                rank=args.lowrank, alpha=args.alpha,
                had_block_size=args.had_block_size,
                act_mode=act_mode, act_block_size=args.act_block_size,
                dtype=torch.float16, seed=idx,
            ).to(device)
            if name in calib_data:
                new_layer.calibrate(
                    calib_data[name]["hdiag"],
                    calib_data[name]["xmax"],
                )
        else:  # cascade_smooth
            new_layer = CascadeSmoothLinear(
                orig_m, fmt=fmt, cascade_block_size=bs, scale_dtype=sd,
                rank=args.lowrank, alpha=args.alpha,
                act_mode=act_mode, act_block_size=args.act_block_size,
                dtype=torch.float16,
            ).to(device)
            if name in calib_data:
                new_layer.calibrate(calib_data[name]["xmax"])

        set_module_by_name(transformer, name, new_layer)

    del calib_data
    accelerator.wait_for_everyone()
    accelerator.print("  Layer replacement done.")

    # ------------------------------------------------------------------
    # Phase 5: 이미지 생성 및 메트릭
    # ------------------------------------------------------------------
    accelerator.print("[Phase 5] Generating images and computing metrics...")
    psnr_m  = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    is_m    = InceptionScore().to(device)
    fid_m   = FrechetInceptionDistance(feature=2048).to(device)

    if accelerator.is_main_process:
        clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    with accelerator.split_between_processes(list(range(s_count))) as local_indices:
        for i in local_indices:
            gen   = torch.Generator(device=device).manual_seed(42 + i)
            q_img = pipe(
                prompts[i], num_inference_steps=t_count, generator=gen
            ).images[0]
            q_img.save(os.path.join(dataset_save_dir, f"sample_{i}.png"))

            r_path = os.path.join(dataset_ref_dir, f"ref_{i}.png")
            r_img  = Image.open(r_path).convert("RGB")
            q_ten  = ToTensor()(q_img).unsqueeze(0).to(device)
            r_ten  = ToTensor()(r_img).unsqueeze(0).to(device)

            psnr_m.update(q_ten, r_ten)
            ssim_m.update(q_ten, r_ten)
            lpips_m.update(q_ten * 2 - 1, r_ten * 2 - 1)
            img_u8 = (q_ten * 255).to(torch.uint8)
            ref_u8 = (r_ten * 255).to(torch.uint8)
            is_m.update(img_u8)
            fid_m.update(ref_u8, real=True)
            fid_m.update(img_u8, real=False)
            print(f"  GPU {accelerator.process_index} -> sample_{i}.png", flush=True)

    accelerator.wait_for_everyone()

    # ------------------------------------------------------------------
    # Phase 6: 집계 및 저장
    # ------------------------------------------------------------------
    res_psnr  = psnr_m.compute()
    res_ssim  = ssim_m.compute()
    res_lpips = lpips_m.compute()
    res_is, _ = is_m.compute()
    res_fid   = fid_m.compute()

    if accelerator.is_main_process:
        clip_scores = []
        for i in range(s_count):
            path = os.path.join(dataset_save_dir, f"sample_{i}.png")
            if not os.path.exists(path):
                continue
            img    = Image.open(path).convert("RGB")
            inputs = clip_processor(
                text=[prompts[i]], images=img, return_tensors="pt",
                padding=True, truncation=True, max_length=77
            ).to(device)
            with torch.no_grad():
                clip_scores.append(float(clip_model(**inputs).logits_per_image.item()))

        # Config info
        config_info = {
            "method":             args.method,
            "weight_quant":       f"Cascade-3 MXFP ({args.config_csv})",
            "act_quant":          args.act_mode,
            "act_block_size":     args.act_block_size,
            "svd_rank":           args.lowrank,
            "dataset":            args.dataset_name,
            "num_samples":        s_count,
        }
        if args.method == "cascade_hadamard":
            config_info["hadamard_block_size"] = args.had_block_size
            config_info["smoothquant"]         = "no"
        else:
            config_info["alpha"]               = args.alpha
            config_info["hadamard"]            = "no"

        final_res = {
            "config": config_info,
            "primary_metrics": {
                "FID": float(res_fid),
                "IS":  float(res_is),
            },
            "secondary_metrics": {
                "PSNR":  float(res_psnr),
                "SSIM":  float(res_ssim),
                "LPIPS": float(res_lpips),
                "CLIP":  float(np.mean(clip_scores)) if clip_scores else 0.0,
            },
        }

        metrics_path = os.path.join(dataset_save_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(final_res, f, indent=4)

        print(f"\n{'='*64}")
        print(f"  Results saved : {metrics_path}")
        print(f"  [PRIMARY]   FID:  {res_fid:.4f}  |  IS:  {res_is:.4f}")
        print(f"  [SECONDARY] PSNR: {res_psnr:.2f}  SSIM: {res_ssim:.4f}"
              f"  LPIPS: {res_lpips:.4f}  CLIP: {np.mean(clip_scores):.2f}")
        print(f"{'='*64}\n")

    accelerator.wait_for_everyone()
    del pipe
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
