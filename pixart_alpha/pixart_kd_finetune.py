"""
pixart_kd_finetune.py
=====================
PTQ (RPCA or SVD) + Fast Knowledge Distillation fine-tuning for PixArt-XL-2.

Pipeline:
  Phase 1: Reference FP16 image generation (skip if exist)
  Phase 2: PTQ quantization (RPCA or SVD)
  Phase 2.5: KD fine-tuning (lora_a/lora_b only, noise prediction distillation)
  Phase 3: Evaluation (FID, IS, PSNR, SSIM, LPIPS, CLIP)

Usage:
  accelerate launch --multi_gpu --num_processes 2 pixart_kd_finetune.py \\
      --quant_method RPCA --lowrank 32 \\
      --do_kd --kd_steps 300 --kd_lr 1e-4 \\
      --save_dir results/kd_experiment/RPCA_NVFP4_KD300
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
import time
import math
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
# 0. 유틸리티
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


def quantize_uniform(x, block_size=16, mode="INT4"):
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
        raise ValueError(f"Unsupported quantization mode: {mode}")
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


def rpca_ialm(W, lam=None, max_iter=20, tol=1e-5):
    """
    W = L + S via Inexact ALM (Candes et al. 2009).
    L: low-rank dense component  →  quantize
    S: sparse component (outliers)  →  keep in FP16

    lam: sparsity regularization. Default = 1/sqrt(max(m,n)).
    """
    m, n = W.shape
    if lam is None:
        lam = 1.0 / math.sqrt(max(m, n))

    L = torch.zeros_like(W)
    S = torch.zeros_like(W)
    Y = W.clone()

    norm_fro = W.norm(p='fro').clamp(min=1e-10)
    mu = float(m * n) / (4.0 * float(W.abs().sum().clamp(min=1e-10)))
    mu_bar = mu * 1e7
    rho = 1.5

    for _ in range(max_iter):
        # Update L: soft-threshold singular values (SVT)
        T = W - S + Y / mu
        U, sv, Vh = torch.linalg.svd(T, full_matrices=False)
        sv_t = torch.clamp(sv - 1.0 / mu, min=0)
        L = (U * sv_t.unsqueeze(0)) @ Vh

        # Update S: element-wise soft threshold
        R = W - L + Y / mu
        S = torch.sign(R) * torch.clamp(R.abs() - lam / mu, min=0)

        # Update Lagrange multiplier
        residual = W - L - S
        Y = Y + mu * residual
        mu = min(mu * rho, mu_bar)

        if residual.norm(p='fro') / norm_fro < tol:
            break

    return L, S


def get_module_by_name(model, name):
    for part in name.split("."):
        model = getattr(model, part)
    return model


def set_module_by_name(model, name, new_module):
    parts = name.split(".")
    parent = get_module_by_name(model, ".".join(parts[:-1])) if len(parts) > 1 else model
    setattr(parent, parts[-1], new_module)


# ==========================================
# 1. 양자화 레이어 클래스
# ==========================================

class ManualSVDLinear(nn.Module):
    """SmoothQuant + SVD 오차 보정 레이어"""

    def __init__(self, original_linear, act_mode, wgt_mode, alpha=0.5, rank=32, block_size=16, dtype=torch.float32):
        super().__init__()
        self.target_dtype = dtype
        self.act_mode = act_mode
        self.wgt_mode = wgt_mode
        self.alpha = alpha
        self.rank = rank
        self.block_size = block_size
        self.register_buffer("weight", original_linear.weight.data.clone().to(dtype))
        self.bias = nn.Parameter(original_linear.bias.data.clone().to(dtype)) if original_linear.bias is not None else None
        self.register_buffer("w_quantized", original_linear.weight.data.clone().to(dtype))
        self.register_buffer("smooth_scale", torch.ones(self.weight.shape[1]).to(dtype))
        self.lora_a = nn.Parameter(torch.zeros(rank, self.weight.shape[1]).to(dtype))
        self.lora_b = nn.Parameter(torch.zeros(self.weight.shape[0], rank).to(dtype))
        self.is_calibrated = False

    @torch.no_grad()
    def manual_calibrate_and_svd(self, x_max):
        x_max = x_max.clamp(min=1e-5).float()
        w_max = self.weight.abs().max(dim=0)[0].clamp(min=1e-5).float()
        self.smooth_scale.data = (w_max.pow(1 - self.alpha) / x_max.pow(self.alpha)).to(self.target_dtype)
        self.smooth_scale.data = self.smooth_scale.data.clamp(min=1e-4, max=1e4)
        w_smoothed = self.weight.float() / self.smooth_scale.float().view(1, -1)
        w_q = quantize_to_nvfp4(w_smoothed, self.block_size) if self.wgt_mode == "NVFP4" else quantize_uniform(w_smoothed, self.block_size, mode=self.wgt_mode)
        self.w_quantized.copy_(w_q.to(self.target_dtype))
        w_error = w_smoothed - w_q
        U, S, Vh = torch.linalg.svd(w_error.float(), full_matrices=False)
        actual_rank = min(self.rank, S.shape[0])
        sqrt_S = torch.sqrt(S[:actual_rank])
        lora_a_data = torch.zeros(self.rank, self.weight.shape[1], dtype=self.target_dtype, device=x_max.device)
        lora_a_data[:actual_rank, :] = (Vh[:actual_rank, :] * sqrt_S.unsqueeze(1)).to(self.target_dtype)
        self.lora_a.data = lora_a_data
        lora_b_data = torch.zeros(self.weight.shape[0], self.rank, dtype=self.target_dtype, device=x_max.device)
        lora_b_data[:, :actual_rank] = (U[:, :actual_rank] * sqrt_S.unsqueeze(0)).to(self.target_dtype)
        self.lora_b.data = lora_b_data
        self.is_calibrated = True

    def forward(self, x):
        input_dtype = x.dtype
        if not self.is_calibrated:
            return F.linear(x, self.weight.to(input_dtype), self.bias.to(input_dtype) if self.bias is not None else None)
        x_smoothed = x.to(self.target_dtype) * self.smooth_scale
        x_q = quantize_to_nvfp4(x_smoothed, self.block_size) if self.act_mode == "NVFP4" else quantize_uniform(x_smoothed, self.block_size, mode=self.act_mode)
        # cast lora to target_dtype to support float32 lora during KD fine-tuning
        out = F.linear(x_q, self.w_quantized) + F.linear(F.linear(x_smoothed, self.lora_a.to(self.target_dtype)), self.lora_b.to(self.target_dtype))
        if self.bias is not None:
            out += self.bias
        return out.to(input_dtype)


class ManualRPCALinear(nn.Module):
    """
    RPCA 분해: W = L + S (Inexact ALM)
      L (low-rank dense) → SmoothQuant → quantize → SVD error correction (lora)
      S (sparse outliers) → FP16 고정밀도 branch

    SVD와 구조적 차이:
      SVD:  2 branches — quantized(W) + lora
      RPCA: 3 branches — quantized(L) + lora(L_err) + sparse(S)
    """

    def __init__(self, original_linear, act_mode, wgt_mode, alpha=0.5, rank=32,
                 block_size=16, rpca_lam=None, dtype=torch.float32):
        super().__init__()
        self.target_dtype = dtype
        self.act_mode = act_mode
        self.wgt_mode = wgt_mode
        self.alpha = alpha
        self.rank = rank
        self.block_size = block_size
        self.rpca_lam = rpca_lam  # None → auto = 1/sqrt(max(m,n))
        self.register_buffer("weight", original_linear.weight.data.clone().to(dtype))
        self.bias = nn.Parameter(original_linear.bias.data.clone().to(dtype)) if original_linear.bias is not None else None
        self.register_buffer("w_sparse", torch.zeros_like(original_linear.weight.data).to(dtype))
        self.register_buffer("w_quantized", original_linear.weight.data.clone().to(dtype))
        self.register_buffer("smooth_scale", torch.ones(self.weight.shape[1]).to(dtype))
        self.lora_a = nn.Parameter(torch.zeros(rank, self.weight.shape[1]).to(dtype))
        self.lora_b = nn.Parameter(torch.zeros(self.weight.shape[0], rank).to(dtype))
        self.is_calibrated = False

    @torch.no_grad()
    def manual_calibrate_and_rpca(self, x_max):
        x_max = x_max.clamp(min=1e-5).float()

        # Step 1: RPCA ALM decomposition W = L + S
        W_float = self.weight.float()
        L, S = rpca_ialm(W_float, lam=self.rpca_lam)
        self.w_sparse.copy_(S.to(self.target_dtype))

        # Step 2: SmoothQuant on L (dense low-rank component)
        w_max = L.abs().max(dim=0)[0].clamp(min=1e-5)
        self.smooth_scale.data = (w_max.pow(1 - self.alpha) / x_max.pow(self.alpha)).to(self.target_dtype)
        self.smooth_scale.data = self.smooth_scale.data.clamp(min=1e-4, max=1e4)
        w_smoothed = L / self.smooth_scale.float().view(1, -1)

        # Step 3: Quantize smoothed L
        w_q = quantize_to_nvfp4(w_smoothed, self.block_size) if self.wgt_mode == "NVFP4" else quantize_uniform(w_smoothed, self.block_size, mode=self.wgt_mode)
        self.w_quantized.copy_(w_q.to(self.target_dtype))

        # Step 4: SVD on quantization error of L
        w_error = w_smoothed - w_q
        U, Sv, Vh = torch.linalg.svd(w_error.float(), full_matrices=False)
        actual_rank = min(self.rank, Sv.shape[0])
        sqrt_S = torch.sqrt(Sv[:actual_rank])
        lora_a_data = torch.zeros(self.rank, self.weight.shape[1], dtype=self.target_dtype, device=x_max.device)
        lora_a_data[:actual_rank, :] = (Vh[:actual_rank, :] * sqrt_S.unsqueeze(1)).to(self.target_dtype)
        self.lora_a.data = lora_a_data
        lora_b_data = torch.zeros(self.weight.shape[0], self.rank, dtype=self.target_dtype, device=x_max.device)
        lora_b_data[:, :actual_rank] = (U[:, :actual_rank] * sqrt_S.unsqueeze(0)).to(self.target_dtype)
        self.lora_b.data = lora_b_data
        self.is_calibrated = True

    def forward(self, x):
        input_dtype = x.dtype
        if not self.is_calibrated:
            return F.linear(x, self.weight.to(input_dtype), self.bias.to(input_dtype) if self.bias is not None else None)
        x_smoothed = x.to(self.target_dtype) * self.smooth_scale
        x_q = quantize_to_nvfp4(x_smoothed, self.block_size) if self.act_mode == "NVFP4" else quantize_uniform(x_smoothed, self.block_size, mode=self.act_mode)
        out = F.linear(x_q, self.w_quantized) + F.linear(F.linear(x_smoothed, self.lora_a.to(self.target_dtype)), self.lora_b.to(self.target_dtype)) + F.linear(x.to(self.target_dtype), self.w_sparse)
        if self.bias is not None:
            out += self.bias
        return out.to(input_dtype)


# ==========================================
# 2. KD 파인튜닝
# ==========================================

def run_kd_finetuning(pipe, teacher_transformer, calib_prompts, args, accelerator):
    """
    PTQ 후 lora_a/lora_b만 fine-tune via noise prediction distillation.
    Teacher: frozen FP16 transformer
    Student: quantized transformer (ManualRPCALinear / ManualSVDLinear)

    Multi-GPU: DDP wrapper를 사용하지 않고 torch.distributed.all_reduce로 직접 gradient sync.
    이를 통해 float32 lora 파라미터와 float16 evaluation 사이의 dtype 충돌을 방지.
    """
    import torch.distributed as dist

    device = accelerator.device
    n_kd_steps = args.kd_steps
    lr = args.kd_lr
    kd_prompts = calib_prompts[:args.kd_prompts]

    # Upcast lora_a/lora_b to float32 for stable AdamW optimization
    for name, p in pipe.transformer.named_parameters():
        if "lora_a" in name or "lora_b" in name:
            p.data = p.data.float()

    # Trainable params: lora_a, lora_b only
    trainable = [
        p for name, p in pipe.transformer.named_parameters()
        if ("lora_a" in name or "lora_b" in name) and p.requires_grad
    ]
    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in trainable)
        accelerator.print(f"  KD trainable params: {len(trainable)} tensors, {n_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_kd_steps, eta_min=lr * 0.01)

    B = len(kd_prompts)
    latent_channels = pipe.transformer.config.in_channels
    latent_h = latent_w = 128  # 1024px / 8 (VAE downscale)

    # Pre-encode prompts
    accelerator.print("  Pre-encoding KD prompts...")
    with torch.no_grad():
        prompt_embeds_list = []
        prompt_masks_list = []
        for prompt in kd_prompts:
            enc = pipe.encode_prompt(
                prompt,
                do_classifier_free_guidance=False,
                num_images_per_prompt=1,
                device=device,
            )
            if isinstance(enc, (tuple, list)):
                pe = enc[0]
                pm = enc[1] if len(enc) > 1 else None
            else:
                pe, pm = enc, None
            prompt_embeds_list.append(pe)
            if pm is not None:
                prompt_masks_list.append(pm)

    prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
    prompt_masks = torch.cat(prompt_masks_list, dim=0) if prompt_masks_list else None

    # Fixed latents (same seed on all ranks for consistent gradients)
    generator = torch.Generator(device=device).manual_seed(42)
    latents = torch.randn(B, latent_channels, latent_h, latent_w,
                          device=device, dtype=torch.float16, generator=generator)

    # KD loop — NO accelerator.prepare, use torch.distributed.all_reduce directly
    pipe.transformer.train()
    teacher_transformer.eval()
    teacher_transformer.requires_grad_(False)

    num_processes = accelerator.num_processes
    accelerator.print(f"  Starting KD: {n_kd_steps} steps, lr={lr}")
    loss_log = []

    for step in range(n_kd_steps):
        # Sync random seed across ranks so t and noise are identical → identical gradients
        torch.manual_seed(step + 1000)
        t = torch.randint(0, 1000, (B,), device=device)
        noise = torch.randn(B, latent_channels, latent_h, latent_w,
                            device=device, dtype=torch.float16)

        if hasattr(pipe.scheduler, "add_noise"):
            noisy_latents = pipe.scheduler.add_noise(latents, noise, t)
        else:
            alpha_t = pipe.scheduler.alphas_cumprod[t].to(device).view(-1, 1, 1, 1).sqrt()
            sigma_t = (1 - pipe.scheduler.alphas_cumprod[t]).to(device).view(-1, 1, 1, 1).sqrt()
            noisy_latents = alpha_t * latents + sigma_t * noise

        resolution = torch.tensor([[1024, 1024]], dtype=noisy_latents.dtype, device=device).repeat(B, 1)
        aspect_ratio = torch.tensor([[1.0]], dtype=noisy_latents.dtype, device=device).repeat(B, 1)
        added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        # Teacher prediction (no grad)
        with torch.no_grad():
            teacher_kwargs = dict(
                hidden_states=noisy_latents,
                encoder_hidden_states=prompt_embeds,
                timestep=t,
                added_cond_kwargs=added_cond_kwargs,
            )
            if prompt_masks is not None:
                teacher_kwargs["encoder_attention_mask"] = prompt_masks
            teacher_out = teacher_transformer(**teacher_kwargs)
            teacher_pred = teacher_out.sample if hasattr(teacher_out, "sample") else teacher_out[0]

        # Student prediction
        student_kwargs = dict(
            hidden_states=noisy_latents,
            encoder_hidden_states=prompt_embeds,
            timestep=t,
            added_cond_kwargs=added_cond_kwargs,
        )
        if prompt_masks is not None:
            student_kwargs["encoder_attention_mask"] = prompt_masks
        student_out = pipe.transformer(**student_kwargs)
        student_pred = student_out.sample if hasattr(student_out, "sample") else student_out[0]

        loss = F.mse_loss(student_pred.float(), teacher_pred.float())

        optimizer.zero_grad()
        loss.backward()

        # Manual gradient allreduce (since same inputs on all ranks, grads are identical,
        # but allreduce is cheap insurance and keeps models in sync)
        if num_processes > 1 and dist.is_initialized():
            for p in trainable:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        loss_val = loss.item()
        loss_log.append(loss_val)

        if step % 50 == 0 and accelerator.is_main_process:
            accelerator.print(f"  [KD] step {step:4d}/{n_kd_steps}  loss={loss_val:.5f}  lr={lr_scheduler.get_last_lr()[0]:.2e}")

    # Revert lora back to float16 for inference — keeps all params consistent
    for name, p in pipe.transformer.named_parameters():
        if "lora_a" in name or "lora_b" in name:
            p.data = p.data.half()
    pipe.transformer.eval()

    accelerator.print(f"  KD complete. Initial loss={loss_log[0]:.5f} → Final loss={loss_log[-1]:.5f}")
    return loss_log


# ==========================================
# 3. 메인
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="PTQ + Fast KD Fine-tuning for PixArt-XL-2")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--test_run", action="store_true", help="2샘플로 파이프라인 통과 확인")
    parser.add_argument("--ref_dir", type=str, default="./ref_images")
    parser.add_argument("--save_dir", type=str, default="./results/kd_experiment/RPCA_NVFP4_KD300")
    parser.add_argument("--model_path", type=str, default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--dataset_name", type=str, default="MJHQ", choices=["MJHQ", "sDCI"])
    parser.add_argument("--img_base_dir", type=str, default="/data/jameskimh/james_dit_pixart_xl_mjhq")
    # 양자화 방법
    parser.add_argument("--quant_method", type=str, default="RPCA", choices=["BASELINE", "RPCA", "SVD"],
                        help="BASELINE: mtq / RPCA: ManualRPCALinear(W=L+S ALM) / SVD: ManualSVDLinear")
    # 공통 하이퍼파라미터
    quant_modes = ["NVFP4", "INT8", "INT4", "INT3", "INT2", "TERNARY"]
    parser.add_argument("--act_mode", type=str, default="NVFP4", choices=quant_modes)
    parser.add_argument("--wgt_mode", type=str, default="NVFP4", choices=quant_modes)
    parser.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant alpha")
    parser.add_argument("--lowrank", type=int, default=32, help="SVD low-rank 차원")
    parser.add_argument("--block_size", type=int, default=16, help="블록 양자화 크기")
    parser.add_argument("--numeric_dtype", type=str, default="half", choices=["half", "float"])
    # KD 하이퍼파라미터
    parser.add_argument("--do_kd", action="store_true", help="KD fine-tuning 활성화")
    parser.add_argument("--kd_steps", type=int, default=300, help="KD 학습 스텝 수")
    parser.add_argument("--kd_lr", type=float, default=1e-4, help="KD AdamW 학습률")
    parser.add_argument("--kd_prompts", type=int, default=8, help="KD에 사용할 프롬프트 수")
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    s_target = 2 if args.test_run else args.num_samples
    dataset_ref_dir = os.path.join(args.ref_dir, args.dataset_name)
    _img_rel = args.save_dir
    if "/results/" in _img_rel:
        _img_rel = _img_rel.split("/results/", 1)[1]
    elif _img_rel.startswith("./results/"):
        _img_rel = _img_rel[len("./results/"):]
    elif _img_rel.startswith("results/"):
        _img_rel = _img_rel[len("results/"):]
    dataset_save_dir = os.path.join(args.img_base_dir, _img_rel)

    if accelerator.is_main_process:
        os.makedirs(dataset_ref_dir, exist_ok=True)
        os.makedirs(dataset_save_dir, exist_ok=True)

    prompts = get_prompts(s_target, args)
    s_count = len(prompts)
    p_count = 2 if args.test_run else min(64, s_count)
    t_count = 20

    if accelerator.is_main_process:
        kd_tag = f"KD{args.kd_steps}" if args.do_kd else "no-KD"
        accelerator.print(f"\n{'='*60}")
        accelerator.print(f"  Method : {args.quant_method} + {kd_tag}")
        if args.quant_method in ("RPCA", "SVD"):
            accelerator.print(f"  Act    : {args.act_mode}  |  Wgt: {args.wgt_mode}")
            accelerator.print(f"  Rank   : {args.lowrank}   |  Block: {args.block_size}")
        if args.quant_method == "RPCA":
            accelerator.print(f"  RPCA λ: auto (1/sqrt(max(m,n)))")
        if args.do_kd:
            accelerator.print(f"  KD steps: {args.kd_steps}  lr: {args.kd_lr}  prompts: {args.kd_prompts}")
        accelerator.print(f"  Samples: {s_count}  |  Dataset: {args.dataset_name}")
        accelerator.print(f"  Save   : {dataset_save_dir}")
        accelerator.print(f"{'='*60}\n")

    t_total_start = time.time()

    # ------------------------------------------
    # Phase 1: Reference FP16 이미지 생성 (존재하면 skip)
    # ------------------------------------------
    if accelerator.is_main_process:
        missing = [i for i in range(s_count) if not os.path.exists(os.path.join(dataset_ref_dir, f"ref_{i}.png"))]
        if missing:
            accelerator.print(f"[Phase 1] Generating {len(missing)} reference images...")
            pipe_ref = PixArtAlphaPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
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
    # Phase 2: 양자화 모델 준비
    # ------------------------------------------
    t_ptq_start = time.time()
    accelerator.print(f"[Phase 2] Loading and quantizing model ({args.quant_method})...")
    pipe = PixArtAlphaPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    transformer = pipe.transformer

    # KD를 위해 FP16 teacher 저장 (양자화 전 deepcopy, CPU에 보관)
    teacher_transformer = None
    if args.do_kd and args.quant_method != "BASELINE":
        accelerator.print("  [KD] Deepcopying FP16 teacher to CPU...")
        teacher_transformer = copy.deepcopy(transformer).cpu()
        teacher_transformer.eval().requires_grad_(False)

    if args.quant_method == "BASELINE":
        quant_config = copy.deepcopy(mtq.NVFP4_SVDQUANT_DEFAULT_CFG)
        quant_config["algorithm"]["lowrank"] = args.lowrank

        def forward_loop(model):
            for prompt in prompts[:p_count]:
                pipe(prompt, num_inference_steps=5, generator=torch.Generator(device=device).manual_seed(42))

        with torch.no_grad():
            pipe.transformer = mtq.quantize(pipe.transformer, quant_config, forward_loop=forward_loop)
        transformer = pipe.transformer

    elif args.quant_method in ("RPCA", "SVD"):
        target_dtype = torch.float16 if args.numeric_dtype == "half" else torch.float32

        skip_keywords = ["x_embedder", "t_embedder", "proj_out"]
        target_names = [
            n for n, m in transformer.named_modules()
            if isinstance(m, nn.Linear) and not any(kw in n for kw in skip_keywords)
        ]
        if accelerator.is_main_process:
            print(f"  Targeted {len(target_names)} Linear layers for {args.quant_method} quantization.")

        # Calibration
        all_samples = {}
        def hook_fn(name):
            def forward_hook(m, i, o):
                flat = i[0].detach().view(-1, i[0].shape[-1]).abs().float()
                step_max = flat.max(dim=0)[0].cpu()
                all_samples.setdefault(name, []).append(step_max)
            return forward_hook

        hooks = [get_module_by_name(transformer, n).register_forward_hook(hook_fn(n)) for n in target_names]
        print(f"  Calibrating with {p_count} prompts...", flush=True)
        with accelerator.split_between_processes(prompts[:p_count]) as local_prompts:
            for prompt in local_prompts:
                pipe(prompt, num_inference_steps=t_count, generator=torch.Generator(device=device).manual_seed(42))
        for h in hooks:
            h.remove()

        # 분산 reduce
        for name in all_samples:
            local_mean = torch.stack(all_samples[name]).mean(dim=0).to(device)
            all_samples[name] = accelerator.reduce(local_mean, reduction="mean")
        accelerator.wait_for_everyone()

        # 레이어 교체
        for name in tqdm(target_names, desc=f"Replacing layers ({args.quant_method})", disable=not accelerator.is_main_process):
            orig_m = get_module_by_name(transformer, name)
            if next(orig_m.parameters()).device == device:
                if args.quant_method == "RPCA":
                    new_layer = ManualRPCALinear(
                        orig_m, args.act_mode, args.wgt_mode,
                        args.alpha, args.lowrank, args.block_size,
                        rpca_lam=None, dtype=target_dtype
                    ).to(device)
                    if name in all_samples:
                        new_layer.manual_calibrate_and_rpca(all_samples[name])
                else:  # SVD
                    new_layer = ManualSVDLinear(
                        orig_m, args.act_mode, args.wgt_mode,
                        args.alpha, args.lowrank, args.block_size,
                        target_dtype
                    ).to(device)
                    if name in all_samples:
                        new_layer.manual_calibrate_and_svd(all_samples[name])
                set_module_by_name(transformer, name, new_layer)
        accelerator.wait_for_everyone()

    t_ptq_end = time.time()
    ptq_time = t_ptq_end - t_ptq_start

    # ------------------------------------------
    # Phase 2.5: KD Fine-tuning (optional)
    # ------------------------------------------
    kd_loss_log = None
    kd_time = 0.0
    t_kd_start = time.time()
    if args.do_kd and args.quant_method != "BASELINE":
        accelerator.print(f"\n[Phase 2.5] KD fine-tuning ({args.kd_steps} steps)...")

        # Teacher를 GPU로 올림 (FP16, frozen)
        teacher_transformer = teacher_transformer.to(device, dtype=torch.float16)
        torch.cuda.empty_cache()

        kd_calib_prompts = get_prompts(max(args.kd_prompts, 8), args)[:args.kd_prompts]
        kd_loss_log = run_kd_finetuning(pipe, teacher_transformer, kd_calib_prompts, args, accelerator)

        del teacher_transformer
        torch.cuda.empty_cache()
        gc.collect()
        accelerator.wait_for_everyone()
    elif args.do_kd and args.quant_method == "BASELINE":
        accelerator.print("  [KD] Skipped: BASELINE (mtq) uses quantized params without accessible lora.")
    kd_time = time.time() - t_kd_start

    # ------------------------------------------
    # Phase 3: 이미지 생성 및 메트릭 업데이트
    # ------------------------------------------
    t_eval_start = time.time()
    accelerator.print("[Phase 3] Generating images and computing metrics...")

    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    is_m = InceptionScore().to(device)
    fid_m = FrechetInceptionDistance(feature=2048).to(device)

    if accelerator.is_main_process:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    with accelerator.split_between_processes(list(range(s_count))) as local_indices:
        for i in local_indices:
            gen = torch.Generator(device=device).manual_seed(42 + i)
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
    eval_time = time.time() - t_eval_start
    total_time = time.time() - t_total_start

    # ------------------------------------------
    # Phase 4: 메트릭 집계 및 저장
    # ------------------------------------------
    res_psnr = psnr_m.compute()
    res_ssim = ssim_m.compute()
    res_lpips = lpips_m.compute()
    res_is, _ = is_m.compute()
    res_fid = fid_m.compute()

    if accelerator.is_main_process:
        clip_scores = []
        for i in range(s_count):
            path = os.path.join(dataset_save_dir, f"sample_{i}.png")
            if not os.path.exists(path):
                continue
            img = Image.open(path).convert("RGB")
            inputs = clip_processor(text=[prompts[i]], images=img, return_tensors="pt", padding=True).to(device)
            clip_scores.append(float(clip_model(**inputs).logits_per_image.item()))

        config_info = {
            "quant_method": args.quant_method,
            "act_mode": args.act_mode,
            "wgt_mode": args.wgt_mode,
            "rpca_decomp": "IALM(auto_lam)" if args.quant_method == "RPCA" else "N/A",
            "lowrank": args.lowrank,
            "block_size": args.block_size,
            "alpha": args.alpha,
            "do_kd": args.do_kd,
            "kd_steps": args.kd_steps if args.do_kd else 0,
            "kd_lr": args.kd_lr if args.do_kd else "N/A",
            "kd_prompts": args.kd_prompts if args.do_kd else 0,
            "num_samples": s_count,
            "dataset": args.dataset_name,
        }
        final_res = {
            "config": config_info,
            "primary_metrics": {
                "FID": float(res_fid),
                "IS": float(res_is),
            },
            "secondary_metrics": {
                "PSNR": float(res_psnr),
                "SSIM": float(res_ssim),
                "LPIPS": float(res_lpips),
                "CLIP": float(np.mean(clip_scores)) if clip_scores else 0.0,
            },
            "timing_sec": {
                "ptq":   round(ptq_time, 1),
                "kd":    round(kd_time, 1),
                "eval":  round(eval_time, 1),
                "total": round(total_time, 1),
            },
        }
        if kd_loss_log:
            final_res["kd_loss"] = {
                "initial": kd_loss_log[0],
                "final": kd_loss_log[-1],
                "curve_sampled": kd_loss_log[::50],
            }

        metrics_path = os.path.join(dataset_save_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(final_res, f, indent=4)

        def fmt_time(s):
            m, sec = divmod(int(s), 60)
            return f"{m}m{sec:02d}s"

        print(f"\n{'='*60}")
        print(f"  Results saved: {metrics_path}")
        print(f"  [PRIMARY]   FID: {res_fid:.4f}  |  IS: {res_is:.4f}")
        print(f"  [SECONDARY] PSNR: {res_psnr:.2f}  SSIM: {res_ssim:.4f}  LPIPS: {res_lpips:.4f}  CLIP: {np.mean(clip_scores):.2f}")
        print(f"  [TIMING]    PTQ: {fmt_time(ptq_time)}  KD: {fmt_time(kd_time)}  Eval: {fmt_time(eval_time)}  Total: {fmt_time(total_time)}")
        if kd_loss_log:
            print(f"  [KD]        loss: {kd_loss_log[0]:.5f} → {kd_loss_log[-1]:.5f}")
        print(f"{'='*60}\n")

    accelerator.wait_for_everyone()
    del pipe
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
