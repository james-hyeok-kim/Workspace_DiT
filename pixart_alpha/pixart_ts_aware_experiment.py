"""
Timestep-Aware PTQ W4A4 Ablation Study
========================================
핵심 아이디어: DiT activation 분포는 timestep마다 다름.
  → Timestep 구간별 다른 SmoothQuant scale (Direction 1)
  → + 구간별 per-group SVD correction (Direction 2)

목표: NVFP4_DEFAULT_CFG (FID=161.3, IS=1.732) FID+IS 동시 격파.

Ablation:
  BASELINE  → NVFP4_DEFAULT_CFG (MTQ, 기존 재사용)
  A1        → GPTQ global scale (기존 w3a4_experiment 재사용)
  A2 (G3)   → 3-group timestep scale, no correction
  A3 (G5)   → 5-group timestep scale, no correction
  A4 (G3+C) → 3-group scale + rank-4 per-group SVD correction
  A5 (G5+C) → 5-group scale + rank-4 per-group SVD correction
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import numpy as np
import copy
import gc
from PIL import Image
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm
from accelerate import Accelerator
from torchvision.transforms import ToTensor

import modelopt.torch.quantization as mtq

from datasets import load_dataset
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from transformers import CLIPModel, CLIPProcessor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance


# ==========================================
# 0. 전역 Timestep 상태
# ==========================================

_TS_STATE: dict = {"group": 0}


def patch_transformer_for_ts(transformer, n_groups: int, t_max: int = 1000):
    """
    transformer.forward를 monkey-patch하여 매 denoising step에서
    _TS_STATE["group"]을 현재 timestep 기반으로 업데이트.

    경계 정의 (n_groups=3, t_max=1000):
      boundaries = [333, 666]
      t > 666  → group 0 (early/high-noise)
      333 < t ≤ 666 → group 1 (mid)
      t ≤ 333  → group 2 (late/low-noise, refinement)

    실제 DPMSolver 20-step 범위: t ∈ [50, 999]
    """
    orig_fwd = transformer.forward
    boundaries = [t_max * (i + 1) // n_groups for i in range(n_groups - 1)]

    def patched_forward(*args, **kwargs):
        t = kwargs.get("timestep", None)
        if t is not None:
            t_val = float(t.float().mean().item())
            g = n_groups - 1  # default: late timestep (lowest group)
            for idx, b in enumerate(boundaries):
                if t_val > b:
                    g = idx
                    break
            _TS_STATE["group"] = g
        return orig_fwd(*args, **kwargs)

    transformer.forward = patched_forward
    return transformer


# ==========================================
# 1. 유틸리티
# ==========================================

def get_prompts(num_samples, args):
    if args.dataset_name == "MJHQ":
        path, split, key = "xingjianleng/mjhq30k", "test", "text"
    else:
        path, split, key = "mit-han-lab/svdquant-datasets", "train", "prompt"
    try:
        dataset = load_dataset(path, split=split, streaming=True)
        prompts = []
        for i, entry in enumerate(dataset):
            if i >= num_samples:
                break
            prompts.append(entry[key])
        return prompts
    except Exception as e:
        print(f"Warning: Dataset loading failed ({e}). Using fallback prompts.")
        fallback = [
            "A professional high-quality photo of a futuristic city with neon lights",
            "A beautiful landscape of mountains during sunset, cinematic lighting",
            "A cute robot holding a flower in a field, highly detailed digital art",
            "A gourmet burger with melting cheese and fresh vegetables on a wooden table",
            "An astronaut walking on a purple planet surface under a starry sky",
        ]
        return (fallback * (num_samples // len(fallback) + 1))[:num_samples]


def quantize_uniform(x, block_size=16, mode="INT4", clip_ratio=1.0):
    orig_shape = x.shape
    num_el = x.numel()
    pad = (block_size - num_el % block_size) % block_size
    flat = x.flatten()
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=x.device, dtype=x.dtype)])
    x_flat = flat.view(-1, block_size)
    if mode == "TERNARY":
        q_max = 1.0
    elif mode.startswith("INT"):
        bits = int(mode.replace("INT", ""))
        q_max = (2 ** (bits - 1)) - 1.0
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    if clip_ratio < 1.0:
        k = max(1, int(block_size * clip_ratio))
        sorted_abs = x_flat.abs().sort(dim=-1).values
        amax = sorted_abs[:, k - 1].unsqueeze(-1).clamp(min=1e-12)
    else:
        amax = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / q_max
    x_q_flat = torch.clamp(torch.round(x_flat / scale), -q_max, q_max)
    return (x_q_flat * scale).view(-1)[:num_el].view(orig_shape)


def quantize_to_nvfp4(x, block_size=16):
    orig_shape = x.shape
    num_el = x.numel()
    pad = (block_size - num_el % block_size) % block_size
    flat = x.flatten()
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, device=x.device, dtype=x.dtype)])
    x_flat = flat.view(-1, block_size)
    nvfp4_levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device, dtype=x.dtype
    )
    amax = x_flat.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale = amax / 6.0
    x_norm = x_flat.abs() / scale
    distances = torch.abs(x_norm.unsqueeze(-1) - nvfp4_levels)
    closest_idx = torch.argmin(distances, dim=-1)
    x_q = torch.sign(x_flat) * nvfp4_levels[closest_idx] * scale
    return x_q.view(-1)[:num_el].view(orig_shape)


def get_module_by_name(model, name):
    for part in name.split("."):
        model = getattr(model, part)
    return model


def set_module_by_name(model, name, new_module):
    parts = name.split(".")
    parent = get_module_by_name(model, ".".join(parts[:-1])) if len(parts) > 1 else model
    setattr(parent, parts[-1], new_module)


def _block_avg(v: torch.Tensor, block_size: int) -> torch.Tensor:
    """(in_f,) 텐서를 block_size 단위로 블록 내 평균화."""
    out = v.clone()
    n_blocks = v.shape[0] // block_size
    for b in range(n_blocks):
        s, e = b * block_size, (b + 1) * block_size
        out[s:e] = v[s:e].mean()
    return out


# ==========================================
# 2. TimestepAwarePTQLinear
# ==========================================

class TimestepAwarePTQLinear(nn.Module):
    """
    Timestep-Aware PTQ 양자화 레이어.

    Direction 1: Timestep 구간별 다른 SmoothQuant scale (smooth_scales[g])
    Direction 2: 구간별 rank-4 SVD correction (delta_a[g], delta_b[g])

    forward 흐름:
      g        = _TS_STATE["group"]
      x_rot    = block_hadamard(x)
      x_smooth = x_rot * smooth_scales[g]      ← 그룹별 scale
      x_q      = quantize(x_smooth, INT4/NVFP4)
      out      = F.linear(x_q, w_quantized)
               + F.linear(F.linear(x_smooth, lora_a), lora_b)   # global SVD
               [+ F.linear(F.linear(x_smooth, delta_a[g]), delta_b[g])]  # per-group
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        n_groups: int = 3,
        rank: int = 32,
        rank_t: int = 4,
        block_size: int = 128,
        alpha: float = 0.5,
        use_ts_correction: bool = False,
        wgt_bits: str = "NVFP4",
        act_mode: str = "INT4",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.rank = rank
        self.rank_t = rank_t
        self.block_size = block_size
        self.alpha = alpha
        self.use_ts_correction = use_ts_correction
        self.wgt_bits = wgt_bits
        self.act_mode = act_mode
        self.target_dtype = dtype
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.use_rotation = (self.in_features % block_size == 0)

        # 원본 weight/bias 보존
        self.register_buffer("weight", original_linear.weight.data.clone().to(dtype))
        self.bias = nn.Parameter(
            original_linear.bias.data.clone().to(dtype)
        ) if original_linear.bias is not None else None

        # Block Hadamard matrix
        if self.use_rotation:
            from scipy.linalg import hadamard as scipy_hadamard
            H_np = scipy_hadamard(block_size) / (block_size ** 0.5)
            self.register_buffer("H_block", torch.from_numpy(H_np).to(dtype))

        # 양자화된 weight placeholder
        self.register_buffer("w_quantized", original_linear.weight.data.clone().to(dtype))

        # Smooth scales: global + per-group
        in_f = self.in_features
        self.register_buffer("smooth_scale_global", torch.ones(in_f, dtype=dtype))
        self.register_buffer("smooth_scales", torch.ones(n_groups, in_f, dtype=dtype))

        # Per-group scale correction: v_g = smooth_global / smooth_g  (n_groups, in_f)
        # Forward: F.linear(x_q * v_g, w_quantized) ≈ F.linear(x*smooth_global, w_quantized)
        # → no high-rank lora needed, perfect consistency with w_quantized
        self.register_buffer("scale_correction", torch.ones(n_groups, in_f, dtype=dtype))

        # Global SVD lora (same as GPTQ) — lora ≈ SVD(W/smooth_global - W_q)
        self.lora_a = nn.Parameter(torch.zeros(rank, in_f, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank, dtype=dtype))

        # Per-group rank-t additional correction (Direction 2)
        # delta[g] ≈ SVD(remaining residual per timestep group, rank_t)
        # Applied with x_smooth_global for consistency
        if use_ts_correction:
            self.delta_a = nn.Parameter(
                torch.zeros(n_groups, rank_t, in_f, dtype=dtype))
            self.delta_b = nn.Parameter(
                torch.zeros(n_groups, self.out_features, rank_t, dtype=dtype))
        else:
            self.delta_a = None
            self.delta_b = None

        self.is_calibrated = False

    def _block_hadamard(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_rotation:
            return x
        shape = x.shape
        in_f = shape[-1]
        n_blocks = in_f // self.block_size
        x_3d = x.reshape(*shape[:-1], n_blocks, self.block_size)
        return (x_3d @ self.H_block.to(x.dtype)).reshape(shape)

    @torch.no_grad()
    def calibrate(self, grouped_data: dict):
        """
        grouped_data[g] = {"xmax": Tensor(in_f,) or None, "hdiag": Tensor(in_f,) or None}
        """
        device = self.weight.device
        in_f   = self.in_features
        out_f  = self.out_features
        W      = self.weight.float()

        # ---- Step 1: Global statistics (유효한 그룹 평균) ----
        valid_xmax  = [grouped_data[g]["xmax"]  for g in range(self.n_groups)
                       if grouped_data[g]["xmax"] is not None]
        valid_hdiag = [grouped_data[g]["hdiag"] for g in range(self.n_groups)
                       if grouped_data[g]["hdiag"] is not None]

        global_xmax  = torch.stack(valid_xmax).mean(0).float().to(device).clamp(min=1e-5)
        global_hdiag = torch.stack(valid_hdiag).mean(0).float().to(device).clamp(min=1e-8)

        # ---- Step 2: Offline Hadamard rotation ----
        if self.use_rotation:
            W_rot  = self._block_hadamard(W)
            H_rot  = _block_avg(global_hdiag, self.block_size)
            xm_rot = _block_avg(global_xmax,  self.block_size)
        else:
            W_rot  = W
            H_rot  = global_hdiag
            xm_rot = global_xmax

        # ---- Step 3: Global SmoothQuant scale ----
        w_max_rot     = W_rot.abs().max(dim=0)[0].clamp(min=1e-5)
        smooth_global = (w_max_rot.pow(1 - self.alpha) / xm_rot.pow(self.alpha)
                         ).clamp(1e-4, 1e4)
        self.smooth_scale_global.copy_(smooth_global.to(self.target_dtype))
        W_smooth = W_rot / smooth_global.unsqueeze(0)  # (out_f, in_f)

        # ---- Step 4: act_order + per-group weight quantization ----
        order     = torch.argsort(H_rot, descending=True)
        inv_order = torch.argsort(order)
        W_reord   = W_smooth[:, order]
        W_q_reord = torch.zeros_like(W_reord)

        gs = self.block_size
        for g_start in range(0, in_f, gs):
            g_end  = min(g_start + gs, in_f)
            group_w = W_reord[:, g_start:g_end]
            if self.wgt_bits == "NVFP4":
                W_q_reord[:, g_start:g_end] = quantize_to_nvfp4(
                    group_w, block_size=g_end - g_start)
            else:
                W_q_reord[:, g_start:g_end] = quantize_uniform(
                    group_w, block_size=g_end - g_start, mode=self.wgt_bits)

        W_q = W_q_reord[:, inv_order]
        self.w_quantized.copy_(W_q.to(self.target_dtype))

        # ---- Step 5: Global SVD lora (same as GPTQ) ----
        # lora ≈ SVD(W/smooth_global - W_q)  ← only quantization noise, low-rank
        W_err_global = W_smooth - W_q
        U0, S0, Vh0 = torch.linalg.svd(W_err_global.float(), full_matrices=False)
        r0 = min(self.rank, S0.shape[0])
        sqrt_S0 = S0[:r0].sqrt()
        la = torch.zeros(self.rank, in_f,  dtype=self.target_dtype, device=device)
        lb = torch.zeros(out_f, self.rank, dtype=self.target_dtype, device=device)
        la[:r0] = (Vh0[:r0] * sqrt_S0.unsqueeze(1)).to(self.target_dtype)
        lb[:, :r0] = (U0[:, :r0] * sqrt_S0.unsqueeze(0)).to(self.target_dtype)
        self.lora_a.data = la
        self.lora_b.data = lb

        # ---- Step 6: Per-group smooth scales + scale_correction ----
        # scale_correction[g] = smooth_global / smooth_g  (per-channel ratio)
        # Forward: x_q_g * scale_correction[g] ≈ x * smooth_g * (smooth_global/smooth_g)
        #        = x * smooth_global  → consistent with w_quantized (no high-rank error!)
        global_lora_reconstruct = lb.float() @ la.float()  # (out_f, in_f)

        for g in range(self.n_groups):
            xmax_g = grouped_data[g]["xmax"]
            if xmax_g is None:
                # Fallback: same as global
                self.smooth_scales[g].copy_(smooth_global.to(self.target_dtype))
                self.scale_correction[g].fill_(1.0)
                continue

            xmax_g = xmax_g.float().to(device).clamp(min=1e-5)
            if self.use_rotation:
                xm_g = _block_avg(xmax_g, self.block_size)
            else:
                xm_g = xmax_g
            smooth_g = (w_max_rot.pow(1 - self.alpha) / xm_g.pow(self.alpha)
                        ).clamp(1e-4, 1e4)
            self.smooth_scales[g].copy_(smooth_g.to(self.target_dtype))

            # scale_correction = smooth_global / smooth_g (per-channel)
            v_g = (smooth_global / smooth_g).clamp(1e-4, 1e4)
            self.scale_correction[g].copy_(v_g.to(self.target_dtype))

        # ---- Step 7: Per-group rank-t additional correction (Direction 2) ----
        # Residual per group AFTER global lora: W/smooth_g - W_q - lora_reconstruct
        # Applied with x_smooth_global in forward → no high-rank issue since x uses global scale
        if self.use_ts_correction:
            for g in range(self.n_groups):
                smooth_g  = self.smooth_scales[g].float().to(device)
                W_smooth_g = W_rot / smooth_g.unsqueeze(0)
                residual_g = W_smooth_g - W_q - global_lora_reconstruct

                Ug2, Sg2, Vhg2 = torch.linalg.svd(residual_g.float(), full_matrices=False)
                r_t     = min(self.rank_t, Sg2.shape[0])
                sqrt_Sg2 = Sg2[:r_t].sqrt()

                da = torch.zeros(self.rank_t, in_f,  dtype=self.target_dtype, device=device)
                db = torch.zeros(out_f, self.rank_t, dtype=self.target_dtype, device=device)
                da[:r_t] = (Vhg2[:r_t] * sqrt_Sg2.unsqueeze(1)).to(self.target_dtype)
                db[:, :r_t] = (Ug2[:, :r_t] * sqrt_Sg2.unsqueeze(0)).to(self.target_dtype)
                self.delta_a.data[g] = da
                self.delta_b.data[g] = db

        self.is_calibrated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if not self.is_calibrated:
            bias = self.bias.to(input_dtype) if self.bias is not None else None
            return F.linear(x, self.weight.to(input_dtype), bias)

        x_t = x.to(self.target_dtype)

        # Online block Hadamard rotation
        x_rot = self._block_hadamard(x_t)

        g = max(0, min(_TS_STATE["group"], self.n_groups - 1))

        # Activation quantization in per-group smooth space (better calibration per group)
        x_smooth_g = x_rot * self.smooth_scales[g]
        if self.act_mode == "NVFP4":
            x_q = quantize_to_nvfp4(x_smooth_g, block_size=16)
        else:
            x_q = quantize_uniform(x_smooth_g, block_size=16, mode="INT4")

        # Rescale x_q to global smooth space via scale_correction[g] = smooth_global/smooth_g
        # x_q * v_g ≈ x*smooth_g * (smooth_global/smooth_g) = x*smooth_global  ← matches w_quantized!
        x_q_rescaled = x_q * self.scale_correction[g]

        # Global smooth x for lora (consistent with calibration of W/smooth_global - W_q)
        x_smooth_global = x_rot * self.smooth_scale_global

        # Base + global lora: x*smooth_global @ W_q + x*smooth_global @ (W/smooth_global - W_q) = x @ W
        out = F.linear(x_q_rescaled, self.w_quantized)
        out = out + F.linear(F.linear(x_smooth_global, self.lora_a), self.lora_b)

        # Per-group rank-t additional correction (Direction 2)
        if self.use_ts_correction and self.delta_a is not None:
            out = out + F.linear(F.linear(x_smooth_global, self.delta_a[g]), self.delta_b[g])

        if self.bias is not None:
            out = out + self.bias
        return out.to(input_dtype)


# ==========================================
# 3. 메인
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Timestep-Aware PTQ W4A4 Ablation (A2~A5)"
    )
    parser.add_argument("--num_samples",      type=int,   default=20)
    parser.add_argument("--test_run",         action="store_true")
    parser.add_argument("--ref_dir",          type=str,   default="/data/james_dit_ref/ref_images_fp16")
    parser.add_argument("--save_dir",         type=str,   default="./results/ts_aware_experiment/TSAWARE_G3_NOCORR")
    parser.add_argument("--model_path",       type=str,   default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--dataset_name",     type=str,   default="MJHQ", choices=["MJHQ", "sDCI"])
    parser.add_argument("--quant_method",     type=str,   default="TSAWARE",
                        choices=["FP16", "BASELINE", "TSAWARE"])
    # TSAWARE 하이퍼파라미터
    parser.add_argument("--n_groups",         type=int,   default=3,
                        help="Timestep 그룹 수 (1=global, 3, 5)")
    parser.add_argument("--use_ts_correction",action="store_true",
                        help="Direction 2: per-group SVD correction 활성화")
    parser.add_argument("--rank_t",           type=int,   default=4,
                        help="Per-group SVD correction rank")
    parser.add_argument("--lowrank",          type=int,   default=32,
                        help="Global SVD rank")
    parser.add_argument("--block_size",       type=int,   default=128)
    parser.add_argument("--wgt_bits",         type=str,   default="NVFP4",
                        choices=["NVFP4", "INT4"])
    parser.add_argument("--act_mode",         type=str,   default="INT4",
                        choices=["INT4", "NVFP4"])
    parser.add_argument("--alpha",            type=float, default=0.5)
    parser.add_argument("--baseline_lowrank", type=int,   default=32)
    args = parser.parse_args()

    accelerator = Accelerator()
    device      = accelerator.device

    s_target = 2 if args.test_run else args.num_samples
    dataset_ref_dir  = os.path.join(args.ref_dir,  args.dataset_name)
    dataset_save_dir = os.path.join(args.save_dir, args.dataset_name)

    if accelerator.is_main_process:
        os.makedirs(dataset_ref_dir,  exist_ok=True)
        os.makedirs(dataset_save_dir, exist_ok=True)

    prompts = get_prompts(s_target, args)
    s_count = len(prompts)
    p_count = 2 if args.test_run else min(64, s_count)
    t_count = 20

    skip_keywords = ["x_embedder", "t_embedder", "proj_out"]

    if accelerator.is_main_process:
        accelerator.print(f"\n{'='*60}")
        accelerator.print(f"  Method    : {args.quant_method}")
        if args.quant_method == "TSAWARE":
            accelerator.print(f"  n_groups  : {args.n_groups}  |  ts_correction: {args.use_ts_correction}")
            if args.use_ts_correction:
                accelerator.print(f"  rank_t    : {args.rank_t}")
            accelerator.print(f"  rank(SVD) : {args.lowrank}  |  BlockSize: {args.block_size}")
            accelerator.print(f"  wgt_bits  : {args.wgt_bits}  |  act_mode: {args.act_mode}")
        accelerator.print(f"  Samples   : {s_count}  |  Dataset: {args.dataset_name}")
        accelerator.print(f"  Save dir  : {dataset_save_dir}")
        accelerator.print(f"{'='*60}\n")

    # ------------------------------------------
    # Phase 1: Reference FP16 이미지 (skip if exists)
    # ------------------------------------------
    if accelerator.is_main_process:
        missing = [i for i in range(s_count)
                   if not os.path.exists(os.path.join(dataset_ref_dir, f"ref_{i}.png"))]
        if missing:
            accelerator.print(f"[Phase 1] Generating {len(missing)} reference images...")
            pipe_ref = PixArtAlphaPipeline.from_pretrained(
                args.model_path, torch_dtype=torch.float16
            ).to(device)
            for i in missing:
                gen = torch.Generator(device=device).manual_seed(42 + i)
                img = pipe_ref(prompts[i], num_inference_steps=t_count, generator=gen).images[0]
                img.save(os.path.join(dataset_ref_dir, f"ref_{i}.png"))
            del pipe_ref
            torch.cuda.empty_cache()
            gc.collect()
        else:
            accelerator.print("[Phase 1] All reference images exist, skipping.")
    accelerator.wait_for_everyone()

    # ------------------------------------------
    # Phase 2: 모델 준비 + 양자화 적용
    # ------------------------------------------
    accelerator.print(f"[Phase 2] Loading model ({args.quant_method})...")
    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    transformer = pipe.transformer

    if args.quant_method == "FP16":
        accelerator.print("  [FP16] No quantization.")

    elif args.quant_method == "BASELINE":
        accelerator.print("  [BASELINE] Applying mtq.NVFP4_SVDQUANT_DEFAULT_CFG...")
        quant_config = copy.deepcopy(mtq.NVFP4_SVDQUANT_DEFAULT_CFG)
        quant_config["algorithm"]["lowrank"] = args.baseline_lowrank

        def forward_loop(model):
            for prompt in prompts[:p_count]:
                pipe(prompt, num_inference_steps=5,
                     generator=torch.Generator(device=device).manual_seed(42))

        with torch.no_grad():
            pipe.transformer = mtq.quantize(pipe.transformer, quant_config, forward_loop=forward_loop)

    else:  # TSAWARE
        n_groups = args.n_groups
        accelerator.print(f"  [TSAWARE] n_groups={n_groups}, ts_correction={args.use_ts_correction}")

        target_names = [
            n for n, m in transformer.named_modules()
            if isinstance(m, nn.Linear) and not any(kw in n for kw in skip_keywords)
        ]
        if accelerator.is_main_process:
            print(f"  Targeted {len(target_names)} Linear layers.")

        # ---- patch transformer for timestep tracking ----
        patch_transformer_for_ts(transformer, n_groups=n_groups)

        # ---- Calibration: grouped xmax + hdiag ----
        calib_data = {
            name: {g: {"xmax": [], "hdiag": []} for g in range(n_groups)}
            for name in target_names
        }

        def hook_fn(name):
            def forward_hook(m, inputs, output):
                x = inputs[0].detach().view(-1, inputs[0].shape[-1]).float()
                g = _TS_STATE["group"]
                calib_data[name][g]["xmax"].append(x.abs().max(dim=0)[0].cpu())
                calib_data[name][g]["hdiag"].append(x.pow(2).mean(dim=0).cpu())
            return forward_hook

        hooks = [
            get_module_by_name(transformer, n).register_forward_hook(hook_fn(n))
            for n in target_names
        ]

        accelerator.print(f"  Calibrating with {p_count} prompts (timestep-grouped)...")
        with accelerator.split_between_processes(prompts[:p_count]) as local_prompts:
            for prompt in local_prompts:
                pipe(prompt, num_inference_steps=t_count,
                     generator=torch.Generator(device=device).manual_seed(42))
        for h in hooks:
            h.remove()

        # ---- 분산 reduce (그룹별) ----
        for name in calib_data:
            for g in range(n_groups):
                xmax_list  = calib_data[name][g]["xmax"]
                hdiag_list = calib_data[name][g]["hdiag"]
                if xmax_list:
                    xm = torch.stack(xmax_list).mean(0).to(device)
                    hd = torch.stack(hdiag_list).mean(0).to(device)
                    calib_data[name][g] = {
                        "xmax":  accelerator.reduce(xm, reduction="mean"),
                        "hdiag": accelerator.reduce(hd, reduction="mean"),
                    }
                else:
                    calib_data[name][g] = {"xmax": None, "hdiag": None}
        accelerator.wait_for_everyone()

        # ---- Layer replacement + calibrate ----
        for idx, name in enumerate(tqdm(target_names, desc="Replacing layers",
                                        disable=not accelerator.is_main_process)):
            orig_m = get_module_by_name(transformer, name)
            if next(orig_m.parameters()).device == device:
                new_layer = TimestepAwarePTQLinear(
                    orig_m,
                    n_groups=n_groups,
                    rank=args.lowrank,
                    rank_t=args.rank_t,
                    block_size=args.block_size,
                    alpha=args.alpha,
                    use_ts_correction=args.use_ts_correction,
                    wgt_bits=args.wgt_bits,
                    act_mode=args.act_mode,
                    dtype=torch.float16,
                ).to(device)
                if name in calib_data:
                    new_layer.calibrate(calib_data[name])
                set_module_by_name(transformer, name, new_layer)

        del calib_data
        accelerator.wait_for_everyone()

    # ------------------------------------------
    # Phase 3: 이미지 생성 및 메트릭
    # ------------------------------------------
    accelerator.print("[Phase 3] Generating images and computing metrics...")
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
            q_img = pipe(prompts[i], num_inference_steps=t_count, generator=gen).images[0]
            q_img.save(os.path.join(dataset_save_dir, f"sample_{i}.png"))

            r_img = Image.open(os.path.join(dataset_ref_dir, f"ref_{i}.png")).convert("RGB")
            q_ten = ToTensor()(q_img).unsqueeze(0).to(device)
            r_ten = ToTensor()(r_img).unsqueeze(0).to(device)

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

    # ------------------------------------------
    # Phase 4: 집계 및 저장
    # ------------------------------------------
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
                text=[prompts[i]], images=img, return_tensors="pt", padding=True
            ).to(device)
            clip_scores.append(float(clip_model(**inputs).logits_per_image.item()))

        config_info = {
            "quant_method":     args.quant_method,
            "n_groups":         args.n_groups         if args.quant_method == "TSAWARE" else "N/A",
            "use_ts_correction":args.use_ts_correction if args.quant_method == "TSAWARE" else "N/A",
            "rank_t":           args.rank_t            if args.use_ts_correction         else "N/A",
            "lowrank":          args.lowrank           if args.quant_method == "TSAWARE" else "N/A",
            "block_size":       args.block_size        if args.quant_method == "TSAWARE" else "N/A",
            "wgt_bits":         args.wgt_bits          if args.quant_method == "TSAWARE" else "N/A",
            "act_mode":         args.act_mode          if args.quant_method == "TSAWARE" else "N/A",
            "alpha":            args.alpha             if args.quant_method == "TSAWARE" else "N/A",
            "baseline_lowrank": args.baseline_lowrank  if args.quant_method == "BASELINE" else "N/A",
            "num_samples":      s_count,
            "dataset":          args.dataset_name,
        }
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

        print(f"\n{'='*60}")
        print(f"  Results saved : {metrics_path}")
        print(f"  [PRIMARY]   FID: {res_fid:.4f}  |  IS: {res_is:.4f}")
        print(f"  [SECONDARY] PSNR: {res_psnr:.2f}  SSIM: {res_ssim:.4f}"
              f"  LPIPS: {res_lpips:.4f}  CLIP: {np.mean(clip_scores):.2f}")
        print(f"{'='*60}\n")

    accelerator.wait_for_everyone()
    del pipe
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
