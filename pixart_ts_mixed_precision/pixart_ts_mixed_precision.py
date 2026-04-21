"""
Timestep-Aware Mixed Precision Quantization for PixArt-Alpha DiT
=================================================================
핵심 아이디어:
  - Weight: 단일 NVFP4 양자화 (모든 timestep 공유)
  - Activation: timestep group별 다른 precision (NVFP4 / INT4 / INT8)
  - SVD Rank: timestep group별 다른 rank (0~32) → high noise step에서 SVD 생략으로 속도 향상

Ablation configs (5개):
  UNIFORM_FP4   → Control: 모두 동일 (NVFP4 act, rank=32)
  MP_ACT_ONLY   → activation precision만 변화 (NVFP4/INT4/INT8), rank 고정
  MP_RANK_ONLY  → activation 고정, SVD rank만 변화 (0/16/32)
  MP_MODERATE   → act + rank 둘 다 적당히 변화
  MP_AGGRESSIVE → high noise에서 SVD 완전 생략 + low noise INT8 보호

목표: NVFP4_DEFAULT_CFG (FID=161.3, IS=1.732) 대비 FID/IS 동등 이상 + 속도 향상
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import numpy as np
import gc
from PIL import Image
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm
from accelerate import Accelerator
from torchvision.transforms import ToTensor
from datasets import load_dataset
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from transformers import CLIPModel, CLIPProcessor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance


# ==========================================
# 0. Ablation Config 정의
# ==========================================

# group_configs: 각 timestep group별 {act_mode, svd_rank}
# G0=high noise(t>666), G1=mid(333<t≤666), G2=low noise(t≤333)
# act_mode: "NVFP4" | "INT4" | "INT8" | "FP16"
# svd_rank: 0 → SVD 생략, 1~max_rank → truncated SVD

ABLATION_CONFIGS = {
    "UNIFORM_FP4": [
        # Control: 모두 동일한 precision (mixed precision 없음)
        {"act_mode": "NVFP4", "svd_rank": 32},   # G0: high noise
        {"act_mode": "NVFP4", "svd_rank": 32},   # G1: mid
        {"act_mode": "NVFP4", "svd_rank": 32},   # G2: low noise
    ],
    "MP_ACT_ONLY": [
        # Activation precision만 변화, SVD rank 고정
        {"act_mode": "NVFP4", "svd_rank": 32},   # G0: 4bit
        {"act_mode": "INT4",  "svd_rank": 32},   # G1: 4bit (INT4)
        {"act_mode": "INT8",  "svd_rank": 32},   # G2: 8bit 품질 보호
    ],
    "MP_RANK_ONLY": [
        # Activation 고정, SVD rank만 변화
        {"act_mode": "NVFP4", "svd_rank": 0},    # G0: SVD 생략
        {"act_mode": "NVFP4", "svd_rank": 16},   # G1: 절반
        {"act_mode": "NVFP4", "svd_rank": 32},   # G2: 풀 보정
    ],
    "MP_MODERATE": [
        # Act + Rank 둘 다 적당히 변화 (밸런스)
        {"act_mode": "NVFP4", "svd_rank": 8},    # G0
        {"act_mode": "INT4",  "svd_rank": 16},   # G1
        {"act_mode": "INT8",  "svd_rank": 32},   # G2
    ],
    "MP_AGGRESSIVE": [
        # High noise에서 최대 절약, Low noise에서 최대 품질
        {"act_mode": "NVFP4", "svd_rank": 0},    # G0: 4bit + SVD 완전 생략
        {"act_mode": "NVFP4", "svd_rank": 16},   # G1: 4bit + 절반 SVD
        {"act_mode": "INT8",  "svd_rank": 32},   # G2: 8bit + 풀 SVD
    ],
}

ACT_BITS = {"NVFP4": 4, "INT4": 4, "INT8": 8, "FP16": 16}


# ==========================================
# 1. 전역 Timestep 상태
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
    """
    orig_fwd = transformer.forward
    # 내림차순 정렬: [666, 333] → t > 666이면 G0, 333 < t ≤ 666이면 G1, t ≤ 333이면 G2
    boundaries = sorted(
        [t_max * (i + 1) // n_groups for i in range(n_groups - 1)],
        reverse=True
    )

    def patched_forward(*args, **kwargs):
        t = kwargs.get("timestep", None)
        if t is not None:
            t_val = float(t.float().mean().item())
            g = n_groups - 1  # default: 마지막 group (low-noise)
            for idx, b in enumerate(boundaries):
                if t_val > b:
                    g = idx
                    break
            _TS_STATE["group"] = g
        return orig_fwd(*args, **kwargs)

    transformer.forward = patched_forward
    return transformer


def compute_effective_bitwidth(group_configs, n_groups=3, t_max=1000, t_count=20):
    """
    DPMSolver 20-step의 실제 timestep 분포를 기반으로
    각 group에 몇 step이 배정되는지 계산 후 가중 평균 bit-width 반환.
    """
    # 내림차순 정렬 (patch_transformer_for_ts와 동일한 로직)
    boundaries = sorted(
        [t_max * (i + 1) // n_groups for i in range(n_groups - 1)],
        reverse=True
    )

    # DPMSolver linspace timestep 근사
    ts = np.linspace(t_max - 1, 0, t_count + 1)[:-1]
    steps_per_group = [0] * n_groups
    for t_val in ts:
        g = n_groups - 1
        for idx, b in enumerate(boundaries):
            if t_val > b:
                g = idx
                break
        steps_per_group[g] += 1

    total = sum(steps_per_group)
    avg_act_bits = sum(
        ACT_BITS.get(cfg["act_mode"], 4) * steps_per_group[g]
        for g, cfg in enumerate(group_configs)
    ) / total

    # SVD compute fraction (rank/max_rank로 비례)
    max_rank = max(cfg["svd_rank"] for cfg in group_configs)
    if max_rank > 0:
        avg_svd_fraction = sum(
            (cfg["svd_rank"] / max_rank) * steps_per_group[g]
            for g, cfg in enumerate(group_configs)
        ) / total
    else:
        avg_svd_fraction = 0.0

    return {
        "steps_per_group": steps_per_group,
        "avg_act_bits": avg_act_bits,
        "avg_svd_fraction": avg_svd_fraction,
        "svd_savings_pct": (1.0 - avg_svd_fraction) * 100,
    }


# ==========================================
# 2. 유틸리티
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
        amax = x_flat.abs().sort(dim=-1).values[:, k - 1].unsqueeze(-1).clamp(min=1e-12)
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
    out = v.clone()
    n_blocks = v.shape[0] // block_size
    for b in range(n_blocks):
        s, e = b * block_size, (b + 1) * block_size
        out[s:e] = v[s:e].mean()
    return out


# ==========================================
# 3. MixedPrecisionTSLinear
# ==========================================

class MixedPrecisionTSLinear(nn.Module):
    """
    Timestep-Aware Mixed Precision Linear Layer.

    핵심 특징:
      - Weight: 단일 NVFP4 양자화 (모든 timestep 공유)
      - Activation: group별 다른 precision (NVFP4/INT4/INT8)
      - SVD: 단일 SVD (max_rank)를 한 번 계산, runtime에 rank slicing으로 동적 전환

    Forward 흐름 (group g에서):
      x_rot     = block_hadamard(x)               ← 선택적 Hadamard rotation
      x_smooth  = x_rot * smooth_scales[g]        ← per-group SmoothQuant scale
      x_q       = quantize(x_smooth, act_mode[g]) ← per-group activation precision
      x_q_sc    = x_q * scale_correction[g]       ← global weight space로 rescale
      x_global  = x_rot * smooth_scale_global     ← SVD 보정용
      out       = F.linear(x_q_sc, w_quantized)
                + F.linear(F.linear(x_global, lora_a[:r]), lora_b[:, :r])  # r = svd_rank[g]
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        group_configs: list,    # [{"act_mode": str, "svd_rank": int}, ...]
        n_groups: int = 3,
        block_size: int = 128,
        alpha: float = 0.5,
        wgt_bits: str = "NVFP4",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.group_configs = group_configs
        self.block_size = block_size
        self.alpha = alpha
        self.wgt_bits = wgt_bits
        self.target_dtype = dtype
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        # max_rank: 모든 group 중 최대 SVD rank (단일 SVD 계산 기준)
        self.max_rank = max(cfg["svd_rank"] for cfg in group_configs)

        # Hadamard rotation 적용 여부 (in_features가 block_size의 배수일 때)
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

        # 양자화된 weight (단일 NVFP4)
        self.register_buffer("w_quantized", original_linear.weight.data.clone().to(dtype))

        # Global SmoothQuant scale
        in_f = self.in_features
        self.register_buffer("smooth_scale_global", torch.ones(in_f, dtype=dtype))

        # Per-group smooth scales + scale correction
        self.register_buffer("smooth_scales", torch.ones(n_groups, in_f, dtype=dtype))
        self.register_buffer("scale_correction", torch.ones(n_groups, in_f, dtype=dtype))

        # Global SVD lora (max_rank)
        if self.max_rank > 0:
            self.lora_a = nn.Parameter(torch.zeros(self.max_rank, in_f, dtype=dtype))
            self.lora_b = nn.Parameter(torch.zeros(self.out_features, self.max_rank, dtype=dtype))
        else:
            self.lora_a = None
            self.lora_b = None

        self.is_calibrated = False

    def _block_hadamard(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_rotation:
            return x
        shape = x.shape
        in_f = shape[-1]
        n_blocks = in_f // self.block_size
        x_3d = x.reshape(*shape[:-1], n_blocks, self.block_size)
        return (x_3d @ self.H_block.to(x.dtype)).reshape(shape)

    def _quantize_activation(self, x: torch.Tensor, act_mode: str) -> torch.Tensor:
        if act_mode == "NVFP4":
            return quantize_to_nvfp4(x, block_size=16)
        elif act_mode in ("INT4", "INT8"):
            return quantize_uniform(x, block_size=16, mode=act_mode)
        elif act_mode == "FP16":
            return x  # pass-through
        else:
            raise ValueError(f"Unknown act_mode: {act_mode}")

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

        if not valid_xmax:
            self.is_calibrated = True
            return

        global_xmax  = torch.stack(valid_xmax).mean(0).float().to(device).clamp(min=1e-5)
        global_hdiag = torch.stack(valid_hdiag).mean(0).float().to(device).clamp(min=1e-8)

        # ---- Step 2: Hadamard rotation ----
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
        smooth_global = (w_max_rot.pow(1 - self.alpha) / xm_rot.pow(self.alpha)).clamp(1e-4, 1e4)
        self.smooth_scale_global.copy_(smooth_global.to(self.target_dtype))
        W_smooth = W_rot / smooth_global.unsqueeze(0)

        # ---- Step 4: act_order + weight quantization (단일 NVFP4) ----
        order     = torch.argsort(H_rot, descending=True)
        inv_order = torch.argsort(order)
        W_reord   = W_smooth[:, order]
        W_q_reord = torch.zeros_like(W_reord)

        gs = self.block_size
        for g_start in range(0, in_f, gs):
            g_end  = min(g_start + gs, in_f)
            group_w = W_reord[:, g_start:g_end]
            if self.wgt_bits == "NVFP4":
                W_q_reord[:, g_start:g_end] = quantize_to_nvfp4(group_w, block_size=g_end - g_start)
            else:
                W_q_reord[:, g_start:g_end] = quantize_uniform(group_w, block_size=g_end - g_start, mode=self.wgt_bits)

        W_q = W_q_reord[:, inv_order]
        self.w_quantized.copy_(W_q.to(self.target_dtype))

        # ---- Step 5: Global SVD lora (max_rank) ----
        if self.max_rank > 0:
            W_err = W_smooth - W_q
            U0, S0, Vh0 = torch.linalg.svd(W_err.float(), full_matrices=False)
            r0 = min(self.max_rank, S0.shape[0])
            sqrt_S0 = S0[:r0].sqrt()

            la = torch.zeros(self.max_rank, in_f,   dtype=self.target_dtype, device=device)
            lb = torch.zeros(out_f, self.max_rank,  dtype=self.target_dtype, device=device)
            la[:r0] = (Vh0[:r0] * sqrt_S0.unsqueeze(1)).to(self.target_dtype)
            lb[:, :r0] = (U0[:, :r0] * sqrt_S0.unsqueeze(0)).to(self.target_dtype)
            self.lora_a.data = la
            self.lora_b.data = lb

        # ---- Step 6: Per-group smooth scales + scale_correction ----
        for g in range(self.n_groups):
            xmax_g = grouped_data[g]["xmax"]
            if xmax_g is None:
                self.smooth_scales[g].copy_(smooth_global.to(self.target_dtype))
                self.scale_correction[g].fill_(1.0)
                continue

            xmax_g = xmax_g.float().to(device).clamp(min=1e-5)
            xm_g = _block_avg(xmax_g, self.block_size) if self.use_rotation else xmax_g
            smooth_g = (w_max_rot.pow(1 - self.alpha) / xm_g.pow(self.alpha)).clamp(1e-4, 1e4)
            self.smooth_scales[g].copy_(smooth_g.to(self.target_dtype))

            # scale_correction[g] = smooth_global / smooth_g
            # 역할: x_q (per-group space) → global weight space로 rescale
            v_g = (smooth_global / smooth_g).clamp(1e-4, 1e4)
            self.scale_correction[g].copy_(v_g.to(self.target_dtype))

        self.is_calibrated = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if not self.is_calibrated:
            bias = self.bias.to(input_dtype) if self.bias is not None else None
            return F.linear(x, self.weight.to(input_dtype), bias)

        x_t = x.to(self.target_dtype)
        g   = max(0, min(_TS_STATE["group"], self.n_groups - 1))
        cfg = self.group_configs[g]

        # Online Hadamard rotation
        x_rot = self._block_hadamard(x_t)

        # Per-group activation quantization
        x_smooth_g = x_rot * self.smooth_scales[g]
        x_q        = self._quantize_activation(x_smooth_g, cfg["act_mode"])
        x_q_sc     = x_q * self.scale_correction[g]  # global weight space로 rescale

        # Base branch: quantized matmul
        out = F.linear(x_q_sc, self.w_quantized)

        # SVD branch: rank slicing으로 per-group effective rank 적용
        svd_rank = cfg["svd_rank"]
        if svd_rank > 0 and self.lora_a is not None:
            x_smooth_global = x_rot * self.smooth_scale_global
            r = min(svd_rank, self.max_rank)
            out = out + F.linear(
                F.linear(x_smooth_global, self.lora_a[:r]),
                self.lora_b[:, :r]
            )

        if self.bias is not None:
            out = out + self.bias
        return out.to(input_dtype)


# ==========================================
# 4. 메인
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Timestep-Aware Mixed Precision Quantization for DiT")
    parser.add_argument("--num_samples",   type=int,   default=20)
    parser.add_argument("--test_run",      action="store_true")
    parser.add_argument("--ref_dir",       type=str,   default="/data/jameskimh/james_dit_ref/ref_images_fp16")
    parser.add_argument("--save_dir",      type=str,   default="./results")
    parser.add_argument("--model_path",    type=str,   default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--dataset_name",  type=str,   default="MJHQ", choices=["MJHQ", "sDCI"])
    parser.add_argument("--mp_config",     type=str,   default="MP_MODERATE",
                        choices=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--n_groups",      type=int,   default=3)
    parser.add_argument("--wgt_bits",      type=str,   default="NVFP4", choices=["NVFP4", "INT4"])
    parser.add_argument("--block_size",    type=int,   default=128)
    parser.add_argument("--alpha",         type=float, default=0.5)
    parser.add_argument("--t_count",       type=int,   default=20)
    args = parser.parse_args()

    accelerator = Accelerator()
    device      = accelerator.device

    s_target = 2 if args.test_run else args.num_samples
    dataset_ref_dir  = os.path.join(args.ref_dir,  args.dataset_name)
    dataset_save_dir = os.path.join(args.save_dir, args.dataset_name)

    if accelerator.is_main_process:
        os.makedirs(dataset_ref_dir,  exist_ok=True)
        os.makedirs(dataset_save_dir, exist_ok=True)

    group_configs = ABLATION_CONFIGS[args.mp_config]
    max_rank      = max(cfg["svd_rank"] for cfg in group_configs)

    if accelerator.is_main_process:
        eb = compute_effective_bitwidth(group_configs, args.n_groups, t_count=args.t_count)
        print(f"\n{'='*60}")
        print(f"  mp_config  : {args.mp_config}")
        print(f"  n_groups   : {args.n_groups}  |  wgt_bits: {args.wgt_bits}  |  block_size: {args.block_size}")
        print(f"  max_rank   : {max_rank}  |  alpha: {args.alpha}")
        print(f"  Steps/group: {eb['steps_per_group']}")
        print(f"  Avg act bits: {eb['avg_act_bits']:.2f}  |  SVD savings: {eb['svd_savings_pct']:.1f}%")
        print(f"  Group configs:")
        for i, cfg in enumerate(group_configs):
            print(f"    G{i}: act={cfg['act_mode']:<6}  svd_rank={cfg['svd_rank']}")
        print(f"  Samples    : {s_target}  |  Dataset: {args.dataset_name}")
        print(f"  Save dir   : {dataset_save_dir}")
        print(f"{'='*60}\n")

    prompts = get_prompts(s_target, args)
    s_count = len(prompts)
    p_count = 2 if args.test_run else min(64, s_count)
    t_count = args.t_count

    skip_keywords = ["x_embedder", "t_embedder", "proj_out"]

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
    # Phase 2: 모델 로드 + Timestep 패치 + Calibration + Layer 교체
    # ------------------------------------------
    accelerator.print(f"[Phase 2] Loading model (mp_config={args.mp_config})...")
    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    transformer = pipe.transformer

    target_names = [
        n for n, m in transformer.named_modules()
        if isinstance(m, nn.Linear) and not any(kw in n for kw in skip_keywords)
    ]
    if accelerator.is_main_process:
        print(f"  Targeted {len(target_names)} Linear layers.")

    # Timestep group 추적을 위해 transformer.forward를 패치
    patch_transformer_for_ts(transformer, n_groups=args.n_groups)

    # ---- Calibration: per-group xmax + hdiag ----
    calib_data = {
        name: {g: {"xmax": [], "hdiag": []} for g in range(args.n_groups)}
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

    accelerator.print(f"  Calibrating with {p_count} prompts (per-group, {args.n_groups} groups)...")
    with accelerator.split_between_processes(prompts[:p_count]) as local_prompts:
        for prompt in local_prompts:
            pipe(prompt, num_inference_steps=t_count,
                 generator=torch.Generator(device=device).manual_seed(42))
    for h in hooks:
        h.remove()

    # 분산 reduce
    for name in calib_data:
        for g in range(args.n_groups):
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

    # ---- Layer 교체 ----
    for name in tqdm(target_names, desc="Replacing layers", disable=not accelerator.is_main_process):
        orig_m = get_module_by_name(transformer, name)
        if next(orig_m.parameters()).device == device:
            new_layer = MixedPrecisionTSLinear(
                orig_m,
                group_configs=group_configs,
                n_groups=args.n_groups,
                block_size=args.block_size,
                alpha=args.alpha,
                wgt_bits=args.wgt_bits,
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

        eb = compute_effective_bitwidth(group_configs, args.n_groups, t_count=t_count)

        final_res = {
            "config": {
                "mp_config":       args.mp_config,
                "n_groups":        args.n_groups,
                "wgt_bits":        args.wgt_bits,
                "block_size":      args.block_size,
                "alpha":           args.alpha,
                "max_rank":        max_rank,
                "group_configs":   group_configs,
                "num_samples":     s_count,
                "dataset":         args.dataset_name,
            },
            "effective_bitwidth": eb,
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

        baseline_fid = 161.3
        baseline_is  = 1.732
        beat_fid = res_fid < baseline_fid
        beat_is  = res_is  > baseline_is

        print(f"\n{'='*60}")
        print(f"  Results saved: {metrics_path}")
        print(f"  Effective avg act bits: {eb['avg_act_bits']:.2f}  |  SVD savings: {eb['svd_savings_pct']:.1f}%")
        print(f"  [PRIMARY]   FID: {res_fid:.4f} ({'BEAT' if beat_fid else 'MISS'} baseline {baseline_fid})")
        print(f"              IS : {res_is:.4f}  ({'BEAT' if beat_is  else 'MISS'} baseline {baseline_is})")
        print(f"  [SECONDARY] PSNR: {res_psnr:.2f}  SSIM: {res_ssim:.4f}  LPIPS: {res_lpips:.4f}")
        if clip_scores:
            print(f"              CLIP: {np.mean(clip_scores):.2f}")
        print(f"{'='*60}\n")

    accelerator.wait_for_everyone()
    del pipe
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
