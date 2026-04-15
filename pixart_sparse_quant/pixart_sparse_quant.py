"""
2:4 Structured Sparsity + Quantization for PixArt DiT.

Pipeline:
  1. Reference generation (FP16 teacher)
  2. Calibration (per-channel activation max + Hessian for SparseGPT)
  3. Layer replacement (ManualSparseQuantLinear)
  4. Evaluation (FID, IS, PSNR, SSIM, LPIPS, CLIP, latency)
  5. Save results → JSON + CSV
"""

import os
import csv
import json
import math
import gc
import time
import argparse
from datetime import datetime, timezone, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler
from torchvision.transforms import ToTensor
from accelerate import Accelerator
from datasets import load_dataset

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import CLIPModel, CLIPProcessor

# ── ModelOpt sparsity utilities (magnitude + SparseGPT) ─────────────────────
# Uses the installed nvidia-modelopt package (pip-installed in the .dit venv).
from modelopt.torch.sparsity.weight_sparsity.magnitude import m4n2_1d
from modelopt.torch.sparsity.weight_sparsity.sparsegpt import prepare, get_nmprune_info

# ── PyTorch 2:4 semi-structured (cuSPARSELt) ────────────────────────────────
from torch.sparse.semi_structured import to_sparse_semi_structured

KST = timezone(timedelta(hours=9))


# ============================================================
# 1. Quantization helpers (reused from pixart_alpha_quant_b200)
# ============================================================

def quantize_uniform(x, block_size=16, mode="INT8"):
    orig_shape = x.shape
    x_flat = x.reshape(-1, block_size)

    if mode == "TERNARY":
        q_max = 1.0
    elif mode.startswith("INT"):
        bits = int(mode.replace("INT", ""))
        q_max = (2 ** (bits - 1)) - 1.0
    else:
        raise ValueError(f"Unsupported quantization mode: {mode}")

    amax = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / q_max
    x_dq = torch.clamp(torch.round(x_flat / scale), -q_max, q_max) * scale
    return x_dq.view(orig_shape)


def quantize_to_nvfp4(x, block_size=16):
    orig_shape = x.shape
    x_flat = x.reshape(-1, block_size)
    nvfp4_levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device, dtype=x.dtype
    )
    amax = x_flat.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale = amax / 6.0
    x_norm = x_flat.abs() / scale
    closest_idx = torch.argmin(torch.abs(x_norm.unsqueeze(-1) - nvfp4_levels), dim=-1)
    x_q = torch.sign(x_flat) * nvfp4_levels[closest_idx] * scale
    return x_q.view(orig_shape)


# FP8 E4M3 max representable value
_FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0


def quantize_to_nvfp8(x, block_size=16):
    """NVIDIA FP8 E4M3 — block-wise continuous scale, block_size=16.

    Equivalent to the FP8 forward-pass format used on H100/B200.
    Scale = amax(block) / 448, stored as float32 (not power-of-2 constrained).
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    x_flat = x.float().reshape(-1, block_size)

    amax = x_flat.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale = amax / _FP8_E4M3_MAX

    x_scaled = (x_flat / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    # Cast through FP8 to get the quantized grid, then dequantize
    x_q = x_scaled.to(torch.float8_e4m3fn).float() * scale
    return x_q.view(orig_shape).to(orig_dtype)


def quantize_to_mxfp8(x, block_size=32):
    """OCP MX FP8 E4M3 — block-wise power-of-2 scale (E8M0), block_size=32.

    The OCP MX spec mandates:
      - 32 elements per block
      - Shared scale in E8M0 format (= 2^e, integer exponent only)
      - Per-element values in FP8 E4M3
    """
    orig_shape = x.shape
    orig_dtype = x.dtype
    x_flat = x.float().reshape(-1, block_size)

    amax = x_flat.abs().max(dim=-1, keepdim=True)[0].clamp(min=2 ** -127)
    scale_fp32 = amax / _FP8_E4M3_MAX
    # Round to nearest power of 2 → E8M0 format
    scale_e8m0 = 2.0 ** torch.floor(torch.log2(scale_fp32))

    x_scaled = (x_flat / scale_e8m0).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    x_q = x_scaled.to(torch.float8_e4m3fn).float() * scale_e8m0
    return x_q.view(orig_shape).to(orig_dtype)


def _do_quantize(x, mode, block_size):
    if mode == "NVFP4":
        return quantize_to_nvfp4(x, block_size)
    if mode == "NVFP8":
        return quantize_to_nvfp8(x, block_size)
    if mode == "MXFP8":
        return quantize_to_mxfp8(x, block_size=32)   # MX spec 고정 block_size=32
    return quantize_uniform(x, block_size, mode=mode)


# ============================================================
# 2. 2:4 Sparsity helpers
# ============================================================

def get_magnitude_mask_2_4(weight: torch.Tensor) -> torch.BoolTensor:
    """Magnitude-based 2:4 mask: keep 2 largest per group of 4 per row."""
    return m4n2_1d(weight.float()).to(dtype=torch.bool, device=weight.device)


def sparsegpt_prune_compensate(
    tensor: torch.Tensor,
    hessian: torch.Tensor,
    hessian_damp: float = 0.1,
    col_block_size: int = 128,
) -> tuple[torch.Tensor, torch.BoolTensor]:
    """SparseGPT 2:4 pruning. Returns (compensated_weight, mask).

    Unlike ModelOpt's create_sgpt_mask(), this also returns the weight
    updated by the SparseGPT column-compensation pass.
    """
    weight, hessian_inv = prepare(tensor, hessian.to(tensor.device), hessian_damp)
    rows, cols = weight.size()
    hessian_inv_diag = torch.diagonal(hessian_inv, dim1=0, dim2=1)

    is_nm_prune, n, m = get_nmprune_info("2:4 sparsity")

    for r1 in range(0, rows, rows):          # row_block_size = -1 (all rows)
        r2 = min(r1 + rows, rows)
        w_rows = weight[r1:r2].float()

        for i1 in range(0, cols, col_block_size):
            i2 = min(i1 + col_block_size, cols)
            w_blk = w_rows[:, i1:i2].clone()
            q_blk = torch.zeros_like(w_blk)
            delta_blk = torch.zeros_like(w_blk)
            hinv_blk = hessian_inv[i1:i2, i1:i2]
            hinv_diag_blk = hessian_inv_diag[i1:i2]

            errors_blk = (w_blk ** 2) / (hinv_diag_blk ** 2 + 1e-9)
            mask_blk = torch.zeros_like(w_blk, dtype=torch.bool)

            for j in range(i2 - i1):
                w_col = w_blk[:, j]
                d = hinv_diag_blk[j]
                if is_nm_prune and j % m == 0:
                    errors_blk = (w_blk[:, j:j + m] ** 2) / (
                        hinv_diag_blk[j:j + m] ** 2 + 1e-9
                    )
                    mask_blk.scatter_(
                        1,
                        j + torch.topk(errors_blk, n, dim=1, largest=False)[1],
                        True,
                    )
                q = w_col.clone()
                q[mask_blk[:, j]] = 0
                q_blk[:, j] = q
                err = (w_col - q) / d
                w_blk[:, j:] -= err.unsqueeze(1).matmul(hinv_blk[j, j:].unsqueeze(0))
                delta_blk[:, j] = err

            w_rows[:, i1:i2] = q_blk
            if i2 < cols:
                w_rows[:, i2:] -= delta_blk.matmul(hessian_inv[i1:i2, i2:])

        weight[r1:r2] = w_rows

    mask = weight != 0
    return weight.to(tensor.dtype), mask.view(tensor.shape)


# ============================================================
# 3. ManualSparseQuantLinear
# ============================================================

class ManualSparseQuantLinear(nn.Module):
    """2:4 Sparsity + SmoothQuant + Quantization + SVD error correction.

    Forward:
        x_smoothed = x * smooth_scale
        x_q        = quantize(x_smoothed)
        base_out   = sparse_matmul(x_q, W_q^T)       # cuSPARSELt or dense
        svd_out    = x_smoothed @ lora_a^T @ lora_b^T  # low-rank correction
        out        = base_out + svd_out + bias
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        act_mode: str,
        wgt_mode: str,
        alpha: float = 0.5,
        rank: int = 32,
        block_size: int = 16,
        dtype=torch.float16,
        sparsity_mode: str = "magnitude",   # "magnitude" | "sparsegpt" | "none"
        use_semi_structured: bool = False,
    ):
        super().__init__()
        self.act_mode = act_mode
        self.wgt_mode = wgt_mode
        self.alpha = alpha
        self.rank = rank
        self.block_size = block_size
        self.target_dtype = dtype
        self.sparsity_mode = sparsity_mode
        self.use_semi_structured = use_semi_structured
        self.use_semi_structured_active = False

        out_f, in_f = original_linear.weight.shape
        self.register_buffer("weight", original_linear.weight.data.clone().to(dtype))
        self.bias = (
            nn.Parameter(original_linear.bias.data.clone().to(dtype))
            if original_linear.bias is not None
            else None
        )
        self.register_buffer("w_quantized", original_linear.weight.data.clone().to(dtype))
        self.register_buffer("smooth_scale", torch.ones(in_f, dtype=dtype))
        self.lora_a = nn.Parameter(torch.zeros(rank, in_f, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(out_f, rank, dtype=dtype))

        # Semi-structured handle (set during calibration)
        self.w_semi = None
        self.is_calibrated = False

    @torch.no_grad()
    def manual_calibrate_sparse_and_svd(
        self, x_max: torch.Tensor, hessian: torch.Tensor | None = None
    ):
        x_max = x_max.clamp(min=1e-5).float()
        w = self.weight.float()
        dev = w.device

        # ── Step 1: Generate 2:4 sparsity mask ──────────────────────────────
        if self.sparsity_mode == "magnitude":
            mask = get_magnitude_mask_2_4(w)                 # BoolTensor same shape
            w_sparse = w * mask

        elif self.sparsity_mode == "sparsegpt":
            assert hessian is not None, "Hessian required for sparsegpt mode"
            w_sparse, mask = sparsegpt_prune_compensate(w, hessian.to(dev))

        else:  # "none" — no sparsity, behave like ManualSVDLinear
            mask = torch.ones_like(w, dtype=torch.bool)
            w_sparse = w

        # ── Step 2: SmoothQuant scale (from sparsified weight) ───────────────
        w_max = w_sparse.abs().max(dim=0)[0].clamp(min=1e-5)
        smooth_scale = (w_max.pow(1 - self.alpha) / x_max.pow(self.alpha)).clamp(1e-4, 1e4)
        self.smooth_scale.data = smooth_scale.to(self.target_dtype)

        # ── Step 3: Smooth the sparsified weight ─────────────────────────────
        w_smoothed_sparse = w_sparse / smooth_scale.view(1, -1)

        # ── Step 4: Quantize ─────────────────────────────────────────────────
        w_q = _do_quantize(w_smoothed_sparse, self.wgt_mode, self.block_size)
        self.w_quantized.copy_(w_q.to(self.target_dtype))

        # ── Step 5: SVD error correction on TOTAL error ──────────────────────
        # Error = (original smoothed) − (quantized sparse)
        # Captures both sparsity loss and quantization loss.
        w_orig_smoothed = w / smooth_scale.view(1, -1)
        w_error = w_orig_smoothed - w_q                      # float32

        U, S, Vh = torch.linalg.svd(w_error, full_matrices=False)
        actual_rank = min(self.rank, S.shape[0])
        sqrt_S = torch.sqrt(S[:actual_rank])

        lora_a_data = torch.zeros(self.rank, w.shape[1], dtype=self.target_dtype, device=dev)
        lora_a_data[:actual_rank] = (Vh[:actual_rank] * sqrt_S.unsqueeze(1)).to(self.target_dtype)
        self.lora_a.data = lora_a_data

        lora_b_data = torch.zeros(w.shape[0], self.rank, dtype=self.target_dtype, device=dev)
        lora_b_data[:, :actual_rank] = (U[:, :actual_rank] * sqrt_S.unsqueeze(0)).to(self.target_dtype)
        self.lora_b.data = lora_b_data

        # ── Step 6: Convert to semi-structured format (optional) ─────────────
        if self.use_semi_structured and self.sparsity_mode != "none":
            try:
                w_for_sparse = self.w_quantized.contiguous().half()
                self.w_semi = to_sparse_semi_structured(w_for_sparse)
                self.use_semi_structured_active = True
            except Exception as e:
                print(f"⚠️  to_sparse_semi_structured failed ({e}). Using dense fallback.")
                self.use_semi_structured_active = False

        self.is_calibrated = True

    def forward(self, x):
        input_dtype = x.dtype
        if not self.is_calibrated:
            return F.linear(
                x,
                self.weight.to(input_dtype),
                self.bias.to(input_dtype) if self.bias is not None else None,
            )

        # 1. Activation smoothing
        x_smoothed = x.to(self.target_dtype) * self.smooth_scale

        # 2. Activation quantization
        x_q = _do_quantize(x_smoothed, self.act_mode, self.block_size)

        # 3. Base branch
        # When w_semi is a SparseSemiStructuredTensor, F.linear automatically
        # dispatches to the cuSPARSELt kernel via __torch_function__.
        if self.use_semi_structured_active and self.w_semi is not None:
            base_out = F.linear(x_q.half(), self.w_semi)
        else:
            base_out = F.linear(x_q, self.w_quantized)

        # 4. SVD correction branch (always dense, low-rank)
        la = self.lora_a.to(x_smoothed.dtype)
        lb = self.lora_b.to(x_smoothed.dtype)
        svd_out = F.linear(F.linear(x_smoothed, la), lb)

        # 5. Sum + bias
        out = base_out.to(x_smoothed.dtype) + svd_out
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out.to(input_dtype)


# ============================================================
# 4. Utility: module navigation
# ============================================================

def get_module_by_name(model, name):
    for part in name.split("."):
        model = getattr(model, part)
    return model


def set_module_by_name(model, name, new_module):
    parts = name.split(".")
    parent = get_module_by_name(model, ".".join(parts[:-1])) if len(parts) > 1 else model
    setattr(parent, parts[-1], new_module)


# ============================================================
# 5. Dataset helpers
# ============================================================

def get_prompts(num_samples, dataset_name="MJHQ"):
    if dataset_name == "MJHQ":
        path, split, key = "xingjianleng/mjhq30k", "test", "text"
    try:
        dataset = load_dataset(path, split=split, streaming=True)
        prompts = []
        for i, entry in enumerate(dataset):
            if i >= num_samples:
                break
            prompts.append(entry[key])
        return prompts
    except Exception as e:
        print(f"⚠️  Dataset loading failed ({e}). Using fallback prompts.")
        fallback = [
            "A professional high-quality photo of a futuristic city with neon lights",
            "A beautiful landscape of mountains during sunset, cinematic lighting",
            "A cute robot holding a flower in a field, highly detailed digital art",
            "A gourmet burger with melting cheese and fresh vegetables on a wooden table",
            "An astronaut walking on a purple planet's surface under a starry sky",
        ]
        return (fallback * (num_samples // len(fallback) + 1))[:num_samples]


# ============================================================
# 6. CSV helper
# ============================================================

CSV_COLUMNS = [
    "config_id", "sparsity_mode", "wgt_mode", "act_mode", "svd_rank",
    "use_semi_structured", "num_samples",
    "FID", "IS", "PSNR", "SSIM", "LPIPS", "CLIP",
    "avg_latency_ms", "timestamp",
]


def append_csv(csv_path: str, row: dict):
    """Append one result row to the shared summary CSV."""
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--test_run", action="store_true")
    parser.add_argument("--ref_dir", type=str, default="./ref_images")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--model_path", type=str, default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lowrank", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--numeric_dtype", type=str, default="half", choices=["half", "full"])
    parser.add_argument("--dataset_name", type=str, default="MJHQ", choices=["MJHQ"])
    _modes = ["NVFP4", "NVFP8", "MXFP8", "INT8", "INT4", "INT3", "INT2", "TERNARY"]
    parser.add_argument("--act_mode", type=str, default="INT8", choices=_modes)
    parser.add_argument("--wgt_mode", type=str, default="INT8", choices=_modes)
    parser.add_argument(
        "--sparsity_mode", type=str, default="magnitude",
        choices=["magnitude", "sparsegpt", "none"],
    )
    parser.add_argument("--use_semi_structured", action="store_true",
                        help="Enable cuSPARSELt for 2:4 sparse matmul acceleration")
    parser.add_argument("--hessian_damp", type=float, default=0.1)
    parser.add_argument("--csv_path", type=str, default="",
                        help="Shared summary CSV path (default: <save_dir>/../summary.csv)")
    args = parser.parse_args()

    target_dtype = torch.float16 if args.numeric_dtype == "half" else torch.float32
    s_target = 2 if args.test_run else args.num_samples
    t_count = 20  # denoising steps

    # Auto-generate config_id if not provided
    if not args.config_id:
        sparsity_tag = args.sparsity_mode.upper()[:3]
        ss_tag = "_SS" if args.use_semi_structured else ""
        args.config_id = f"SP_{sparsity_tag}_W{args.wgt_mode}_A{args.act_mode}_R{args.lowrank}{ss_tag}"

    accelerator = Accelerator()
    device = accelerator.device

    dataset_ref_dir = os.path.join(args.ref_dir, args.dataset_name)
    dataset_save_dir = os.path.join(args.save_dir, args.dataset_name)

    if accelerator.is_main_process:
        os.makedirs(dataset_ref_dir, exist_ok=True)
        os.makedirs(dataset_save_dir, exist_ok=True)

    prompts = get_prompts(s_target, args.dataset_name)
    actual_count = len(prompts)
    if actual_count < s_target:
        print(f"⚠️  Requested {s_target} but only {actual_count} available. Adjusting.")
    s_count = actual_count
    p_count = 2 if args.test_run else min(64, s_count)

    accelerator.print(f"\n{'='*60}")
    accelerator.print(f"Config: {args.config_id}")
    accelerator.print(f"Sparsity: {args.sparsity_mode} | Wgt: {args.wgt_mode} | Act: {args.act_mode}")
    accelerator.print(f"Rank: {args.lowrank} | Semi-structured: {args.use_semi_structured}")
    accelerator.print(f"Samples: {s_count} | Calibration: {p_count}")
    accelerator.print(f"{'='*60}\n")

    # ── Stage 1: Reference image generation ─────────────────────────────────
    if accelerator.is_main_process:
        accelerator.print(f"🌟 [Stage 1] Generating FP16 reference images → {dataset_ref_dir}")
        pipe_ref = PixArtAlphaPipeline.from_pretrained(
            args.model_path, torch_dtype=torch.float16
        ).to(device)
        for i in range(s_count):
            ref_path = os.path.join(dataset_ref_dir, f"ref_{i}.png")
            if not os.path.exists(ref_path):
                gen = torch.Generator(device=device).manual_seed(42 + i)
                img = pipe_ref(prompts[i], num_inference_steps=t_count, generator=gen).images[0]
                img.save(ref_path)
        del pipe_ref
        torch.cuda.empty_cache()
        gc.collect()
    accelerator.wait_for_everyone()

    # ── Stage 2: Load model ──────────────────────────────────────────────────
    accelerator.print("⚙️  [Stage 2] Loading PixArt pipeline…")
    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    transformer = pipe.transformer

    # ── Stage 3: Calibration ────────────────────────────────────────────────
    # adaln_single: timestep/resolution embedding layers (linear_1/2) + output linear
    # proj_out: final patch-to-pixel projection
    # x_embedder, t_embedder: kept for compatibility with older model variants
    skip_keywords = ["x_embedder", "t_embedder", "adaln_single", "proj_out"]
    target_linear_names = [
        n for n, m in transformer.named_modules()
        if isinstance(m, nn.Linear) and not any(kw in n for kw in skip_keywords)
    ]
    accelerator.print(f"🎯 Targeted {len(target_linear_names)} layers for sparsity + quantization.")

    all_x_max = {}        # name → per-channel activation max
    hessian_acc = {}      # name → accumulated X^T X  (only for sparsegpt)
    hessian_count = {}    # name → number of samples seen

    def make_hook(name):
        def forward_hook(mod, inp, out):
            act = inp[0].detach().float()
            flat = act.view(-1, act.shape[-1])

            # Per-channel max (for SmoothQuant)
            step_max = flat.abs().max(dim=0)[0].cpu()
            if name not in all_x_max:
                all_x_max[name] = []
            all_x_max[name].append(step_max)

            # Hessian accumulation (for SparseGPT): H += X^T @ X
            if args.sparsity_mode == "sparsegpt":
                x_t = flat.t()  # (in_features, tokens)
                n_new = x_t.shape[1]
                if name not in hessian_acc:
                    hessian_acc[name] = torch.zeros(
                        x_t.shape[0], x_t.shape[0], dtype=torch.float32, device="cpu"
                    )
                    hessian_count[name] = 0
                hessian_acc[name] += (x_t.cpu().float() @ x_t.cpu().float().t())
                hessian_count[name] += n_new

        return forward_hook

    hooks = [
        get_module_by_name(transformer, n).register_forward_hook(make_hook(n))
        for n in target_linear_names
    ]

    accelerator.print(f"⚙️  Running calibration ({p_count} prompts)…")
    with accelerator.split_between_processes(prompts[:p_count]) as local_prompts:
        for prompt in local_prompts:
            pipe(
                prompt,
                num_inference_steps=t_count,
                generator=torch.Generator(device=device).manual_seed(42),
            )

    for h in hooks:
        h.remove()

    # Reduce per-channel max across GPUs
    for name in all_x_max:
        local_mean = torch.stack(all_x_max[name]).mean(dim=0).to(device)
        all_x_max[name] = accelerator.reduce(local_mean, reduction="mean")

    # Finalize Hessian: H = (2 / n_samples) * Σ(X^T X)
    if args.sparsity_mode == "sparsegpt":
        for name in hessian_acc:
            n = max(hessian_count[name], 1)
            hessian_acc[name] = (2.0 / n) * hessian_acc[name]
        accelerator.print("✅  Hessian collection complete.")

    accelerator.wait_for_everyone()

    # ── Stage 4: Layer replacement ───────────────────────────────────────────
    accelerator.print("🔧 [Stage 4] Replacing layers with ManualSparseQuantLinear…")

    for name in tqdm(
        target_linear_names,
        desc="Layer replacement",
        disable=not accelerator.is_main_process,
    ):
        orig_m = get_module_by_name(transformer, name)
        if next(orig_m.parameters()).device != device:
            continue

        new_l = ManualSparseQuantLinear(
            orig_m,
            act_mode=args.act_mode,
            wgt_mode=args.wgt_mode,
            alpha=args.alpha,
            rank=args.lowrank,
            block_size=args.block_size,
            dtype=target_dtype,
            sparsity_mode=args.sparsity_mode,
            use_semi_structured=args.use_semi_structured,
        ).to(device)

        if name in all_x_max:
            hess = hessian_acc.get(name, None)
            new_l.manual_calibrate_sparse_and_svd(all_x_max[name], hessian=hess)

        set_module_by_name(transformer, name, new_l)

    semi_active_count = sum(
        1 for n in target_linear_names
        if isinstance(get_module_by_name(transformer, n), ManualSparseQuantLinear)
        and get_module_by_name(transformer, n).use_semi_structured_active
    )
    accelerator.print(
        f"✅  Layer replacement done. Semi-structured active: {semi_active_count}/{len(target_linear_names)}"
    )
    accelerator.wait_for_everyone()

    # ── Stage 5: Evaluation ──────────────────────────────────────────────────
    accelerator.print("📊 [Stage 5] Evaluating…")

    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    is_m = InceptionScore().to(device)
    fid_m = FrechetInceptionDistance(feature=2048).to(device)

    if accelerator.is_main_process:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    latencies_ms = []

    # Warmup: 1 run for test_run (JIT 컴파일 제거), 3 runs for production
    n_warmup = 1 if args.test_run else 3
    accelerator.print(f"🔥 Warming up ({n_warmup} run(s))…")
    for _ in range(n_warmup):
        pipe(
            prompts[0],
            num_inference_steps=t_count,
            generator=torch.Generator(device=device).manual_seed(0),
        )
    torch.cuda.synchronize(device)

    indices = list(range(s_count))
    with accelerator.split_between_processes(indices) as local_indices:
        for i in local_indices:
            gen = torch.Generator(device=device).manual_seed(42 + i)

            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            q_img = pipe(prompts[i], num_inference_steps=t_count, generator=gen).images[0]
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000)

            save_path = os.path.join(dataset_save_dir, f"sample_{i}.png")
            q_img.save(save_path)

            ref_path = os.path.join(dataset_ref_dir, f"ref_{i}.png")
            r_img = Image.open(ref_path).convert("RGB")
            q_ten = ToTensor()(q_img).unsqueeze(0).to(device)
            r_ten = ToTensor()(r_img).unsqueeze(0).to(device)

            psnr_m.update(q_ten, r_ten)
            ssim_m.update(q_ten, r_ten)
            lpips_m.update(q_ten * 2 - 1, r_ten * 2 - 1)
            img_uint8 = (q_ten * 255).to(torch.uint8)
            ref_uint8 = (r_ten * 255).to(torch.uint8)
            is_m.update(img_uint8)
            fid_m.update(ref_uint8, real=True)
            fid_m.update(img_uint8, real=False)

    accelerator.wait_for_everyone()

    # Compute metrics (all ranks simultaneously — no desync since single-GPU in test mode,
    # and 2-GPU prod runs ensure even split so torchmetrics gathers work correctly)
    accelerator.print(f"📊 GPU {accelerator.process_index} computing metrics…")
    res_psnr = float(psnr_m.compute())
    res_ssim = float(ssim_m.compute())
    res_lpips = float(lpips_m.compute())
    res_is = float(is_m.compute()[0])
    res_fid = float(fid_m.compute())

    # Gather latency across GPUs
    sum_lat = torch.tensor(sum(latencies_ms) if latencies_ms else 0.0, device=device)
    cnt_lat = torch.tensor(float(len(latencies_ms)), device=device)
    sum_lat = accelerator.reduce(sum_lat, reduction="sum")
    cnt_lat = accelerator.reduce(cnt_lat, reduction="sum")
    avg_latency_ms = float(sum_lat / cnt_lat) if float(cnt_lat) > 0 else 0.0

    # ── Save results ─────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        all_clip_scores = []
        for i in range(s_count):
            q_img_path = os.path.join(dataset_save_dir, f"sample_{i}.png")
            if not os.path.exists(q_img_path):
                continue
            q_img = Image.open(q_img_path).convert("RGB")
            inputs = clip_processor(
                text=[prompts[i]], images=q_img, return_tensors="pt",
                padding=True, truncation=True, max_length=77
            ).to(device)
            clip_out = clip_model(**inputs)
            all_clip_scores.append(float(clip_out.logits_per_image.item()))

        res_clip = float(np.mean(all_clip_scores)) if all_clip_scores else 0.0
        ts_kst = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

        final_res = {
            "config_id": args.config_id,
            "config": vars(args),
            "averages": {
                "psnr": res_psnr,
                "ssim": res_ssim,
                "lpips": res_lpips,
                "clip": res_clip,
            },
            "dist_metrics": {"FID": res_fid, "IS": res_is},
            "latency": {"avg_latency_ms": avg_latency_ms},
            "semi_structured_layers": semi_active_count,
            "timestamp_kst": ts_kst,
        }

        # JSON
        json_path = os.path.join(dataset_save_dir, "metrics.json")
        with open(json_path, "w") as f:
            json.dump(final_res, f, indent=4)

        # CSV
        csv_path = args.csv_path or os.path.join(args.save_dir, "..", "summary.csv")
        csv_path = os.path.abspath(csv_path)
        csv_row = {
            "config_id": args.config_id,
            "sparsity_mode": args.sparsity_mode,
            "wgt_mode": args.wgt_mode,
            "act_mode": args.act_mode,
            "svd_rank": args.lowrank,
            "use_semi_structured": args.use_semi_structured,
            "num_samples": s_count,
            "FID": f"{res_fid:.4f}",
            "IS": f"{res_is:.4f}",
            "PSNR": f"{res_psnr:.4f}",
            "SSIM": f"{res_ssim:.4f}",
            "LPIPS": f"{res_lpips:.4f}",
            "CLIP": f"{res_clip:.4f}",
            "avg_latency_ms": f"{avg_latency_ms:.1f}",
            "timestamp": ts_kst,
        }
        append_csv(csv_path, csv_row)

        print(f"\n✅  Done! Results saved to {dataset_save_dir}", flush=True)
        print(
            f"   FID:      {res_fid:.2f}  (baseline 126.5)\n"
            f"   IS:       {res_is:.3f}  (baseline 3.332)\n"
            f"   PSNR:     {res_psnr:.2f}\n"
            f"   SSIM:     {res_ssim:.4f}\n"
            f"   LPIPS:    {res_lpips:.4f}\n"
            f"   CLIP:     {res_clip:.2f}\n"
            f"   Latency:  {avg_latency_ms:.1f} ms/image\n"
            f"   CSV:      {csv_path}",
            flush=True,
        )

    accelerator.wait_for_everyone()

    del pipe
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
