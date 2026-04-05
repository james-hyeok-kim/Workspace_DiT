"""
pixart_cp_experiment.py
=======================
CP (PARAFAC) decomposition quantization for PixArt-XL-2.

Weight matrix W (m×n) is reshaped to W_3d (m×n1×n2) where n = n1*n2,
then decomposed via PARAFAC/ALS:
  W_3d ≈ Σ_r  a_r ⊗ b_r ⊗ c_r
       = A @ khatri_rao(C, B)ᵀ
where:
  A ∈ R^(m×r),  B ∈ R^(n1×r),  C ∈ R^(n2×r)

All factors quantized to NVFP4. Activation NVFP4.

Forward (efficient, no KR materialization):
  x (batch×n) → reshape to (batch×n1×n2)
  → contract with C:  h1 = einsum('bpq,qr→bpr', x_3d, C)  (batch×n1×r)
  → contract with B:  h2 = (h1 * B).sum(-2)               (batch×r)
  → contract with A:  out = h2 @ Aᵀ                        (batch×m)

Compression (m=4608, n=1152, n1=32, n2=36, r=64):
  factors: (m*r + n1*r + n2*r) × 4 bits = (4608+32+36)*64*4
  vs NVFP4: m*n*4 = 4608*1152*4
  → ~6% of NVFP4 ≈ 17x compression

Usage:
  accelerate launch --multi_gpu --num_processes 2 pixart_cp_experiment.py \\
      --lowrank 64 --save_dir results/cp_experiment/CP_R64
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import math
import numpy as np
import gc
import time
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

from pixart_kd_finetune import (
    get_prompts,
    quantize_to_nvfp4,
    quantize_uniform,
    get_module_by_name,
    set_module_by_name,
)


# ==========================================
# Utilities
# ==========================================

def _factor_split(n):
    """Split n into (n1, n2) with n1*n2 == n and n1 ≤ n2, as balanced as possible."""
    best = (1, n)
    for f in range(2, int(math.isqrt(n)) + 1):
        if n % f == 0:
            best = (f, n // f)
    return best  # (smaller, larger)


def khatri_rao(A, B):
    """Khatri-Rao product (column-wise Kronecker).
    A: (p, r), B: (q, r) → output: (p*q, r)
    """
    p, r = A.shape
    q, _ = B.shape
    return (A.unsqueeze(1) * B.unsqueeze(0)).reshape(p * q, r)


def als_parafac(W_3d, r, max_iter=100, tol=1e-5, device=None):
    """
    PARAFAC decomposition via Alternating Least Squares (ALS).
    W_3d: (m, n1, n2) tensor (float32)
    Returns: A (m×r), B (n1×r), C (n2×r)
    """
    if device is None:
        device = W_3d.device
    W_3d = W_3d.float().to(device)
    m, n1, n2 = W_3d.shape

    # Mode unfoldings
    W_1 = W_3d.reshape(m, n1 * n2)                    # (m,  n1*n2)
    W_2 = W_3d.permute(1, 0, 2).reshape(n1, m * n2)   # (n1, m*n2)
    W_3 = W_3d.permute(2, 0, 1).reshape(n2, m * n1)   # (n2, m*n1)

    # SVD-based initialization for A (first r left singular vectors)
    U, _, _ = torch.linalg.svd(W_1, full_matrices=False)
    A = U[:, :r].contiguous()
    # Random init for B, C (normalized)
    B = torch.randn(n1, r, device=device)
    C = torch.randn(n2, r, device=device)
    B = F.normalize(B, dim=0)
    C = F.normalize(C, dim=0)

    prev_err = float("inf")
    for it in range(max_iter):
        # Update A: A = W_(1) @ (C ⊙ B) @ pinv((CᵀC * BᵀB))
        KB  = khatri_rao(C, B)                         # (n1*n2, r)
        V   = (C.T @ C) * (B.T @ B)                   # (r, r)
        A   = (W_1 @ KB) @ torch.linalg.pinv(V)        # (m,  r)

        # Update B: B = W_(2) @ (C ⊙ A) @ pinv((CᵀC * AᵀA))
        KCA = khatri_rao(C, A)                         # (m*n2, r)
        V   = (C.T @ C) * (A.T @ A)                   # (r, r)
        B   = (W_2 @ KCA) @ torch.linalg.pinv(V)       # (n1, r)

        # Update C: C = W_(3) @ (B ⊙ A) @ pinv((BᵀB * AᵀA))
        KBA = khatri_rao(B, A)                         # (m*n1, r)
        V   = (B.T @ B) * (A.T @ A)                   # (r, r)
        C   = (W_3 @ KBA) @ torch.linalg.pinv(V)       # (n2, r)

        # Convergence check (reconstruction error)
        if (it + 1) % 10 == 0:
            W_rec = A @ khatri_rao(C, B).T             # (m, n1*n2)
            err   = (W_1 - W_rec).norm().item() / W_1.norm().clamp(min=1e-10).item()
            if abs(prev_err - err) < tol:
                break
            prev_err = err

    return A, B, C


# ==========================================
# CP quantized linear layer
# ==========================================

class ManualCPLinear(nn.Module):
    """
    CP (PARAFAC) decomposed + quantized linear layer.

    W (m×n) → reshape to W_3d (m×n1×n2) → ALS → A (m×r), B (n1×r), C (n2×r)
    All factors quantized to NVFP4 (or specified wgt_mode).

    Forward:
      x (batch×n) → smooth → quant → reshape (batch×n1×n2)
      → contract C:  h1 (batch×n1×r) = einsum(x_3d, C)
      → contract B:  h2 (batch×r)    = (h1 * B).sum(-2)
      → contract A:  out (batch×m)   = F.linear(h2, A)  [= h2 @ Aᵀ]
    """

    def __init__(self, original_linear, act_mode="NVFP4", wgt_mode="NVFP4",
                 alpha=0.5, rank=64, block_size=16,
                 als_iters=100, dtype=torch.float16):
        super().__init__()
        self.target_dtype = dtype
        self.act_mode  = act_mode
        self.wgt_mode  = wgt_mode
        self.alpha     = alpha
        self.rank      = rank
        self.block_size = block_size
        self.als_iters  = als_iters

        m, n = original_linear.weight.data.shape
        self.m, self.n = m, n
        n1, n2 = _factor_split(n)
        self.n1, self.n2 = n1, n2

        r = rank
        self.register_buffer("weight", original_linear.weight.data.clone().to(dtype))
        self.bias = (nn.Parameter(original_linear.bias.data.clone().to(dtype))
                     if original_linear.bias is not None else None)
        self.register_buffer("smooth_scale", torch.ones(n, dtype=dtype))

        # CP factors (stored for efficient forward)
        # a_q: (m, r)  — F.linear(h2, a_q) = h2 @ Aᵀ
        self.register_buffer("a_q",  torch.zeros(m,  r, dtype=dtype))
        # b_q: (n1, r) — elementwise multiply with h1 (batch×n1×r)
        self.register_buffer("b_q",  torch.zeros(n1, r, dtype=dtype))
        # c_q: (n2, r) — contracts with x_3d last dim:  x_3d @ C = (batch×n1×r)
        #   stored as (r, n2) for F.linear convention
        self.register_buffer("c_q_t", torch.zeros(r, n2, dtype=dtype))
        self.is_calibrated = False

    @torch.no_grad()
    def calibrate(self, x_max):
        """ALS PARAFAC decomposition + quantize factors."""
        x_max = x_max.clamp(min=1e-5).float()
        W = self.weight.float()  # (m, n)

        # SmoothQuant scale
        w_max = W.abs().max(dim=0)[0].clamp(min=1e-5)
        smooth = (w_max.pow(1 - self.alpha) / x_max.pow(self.alpha)).clamp(1e-4, 1e4)
        self.smooth_scale.data = smooth.to(self.target_dtype)
        W_smooth = W / smooth.unsqueeze(0)  # (m, n)

        # Reshape W_smooth to 3D and run ALS
        W_3d = W_smooth.reshape(self.m, self.n1, self.n2)  # (m, n1, n2)
        A, B, C = als_parafac(W_3d, self.rank, self.als_iters,
                               device=W.device)

        # Quantize factors
        def _q(t):
            if self.wgt_mode == "NVFP4":
                return quantize_to_nvfp4(t, self.block_size)
            return quantize_uniform(t, self.block_size, mode=self.wgt_mode)

        self.a_q.data   = _q(A).to(self.target_dtype)             # (m,  r)
        self.b_q.data   = _q(B).to(self.target_dtype)             # (n1, r)
        self.c_q_t.data = _q(C.T).to(self.target_dtype)           # (r,  n2)
        self.is_calibrated = True

    def compression_stats(self):
        m, n, r, n1, n2 = self.m, self.n, self.rank, self.n1, self.n2
        bits_per_elem = 4  # NVFP4
        total_cp   = (m + n1 + n2) * r * bits_per_elem
        fp16_bits  = m * n * 16
        nvfp4_bits = m * n * 4
        return {
            "factor_bits":  bits_per_elem,
            "n1": n1, "n2": n2, "rank": r,
            "total_cp_bits":        total_cp,
            "compression_vs_fp16":  round(fp16_bits  / total_cp, 2),
            "compression_vs_nvfp4": round(nvfp4_bits / total_cp, 2),
            "pct_of_nvfp4":         round(100.0 * total_cp / nvfp4_bits, 2),
        }

    def forward(self, x):
        input_dtype = x.dtype
        if not self.is_calibrated:
            return F.linear(x, self.weight.to(input_dtype),
                            self.bias.to(input_dtype) if self.bias is not None else None)

        x_smooth = x.to(self.target_dtype) * self.smooth_scale
        if self.act_mode == "NVFP4":
            x_q = quantize_to_nvfp4(x_smooth, self.block_size)
        else:
            x_q = quantize_uniform(x_smooth, self.block_size, mode=self.act_mode)

        # Reshape activation to 3D: (..., n) → (..., n1, n2)
        orig_shape = x_q.shape
        n1, n2, r = self.n1, self.n2, self.rank
        x_3d = x_q.view(*orig_shape[:-1], n1, n2)  # (..., n1, n2)

        # Contract with C: h1[..., n1, r] = x_3d[..., n1, n2] @ C[n2, r]
        # c_q_t has shape (r, n2) → F.linear(x_3d, c_q_t) = x_3d @ c_q_t.T = x_3d @ C
        h1 = F.linear(x_3d, self.c_q_t)   # (..., n1, r)

        # Contract with B: h2[..., r] = (h1[..., n1, r] * B[n1, r]).sum(-2)
        h2 = (h1 * self.b_q).sum(-2)       # (..., r)

        # Contract with A: out[..., m] = h2 @ Aᵀ
        out = F.linear(h2, self.a_q)        # (..., m)

        if self.bias is not None:
            out = out + self.bias
        return out.to(input_dtype)


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="CP (PARAFAC) PTQ for PixArt-XL-2")
    parser.add_argument("--num_samples",  type=int,   default=20)
    parser.add_argument("--test_run",     action="store_true")
    parser.add_argument("--ref_dir",      type=str,   default="./ref_images")
    parser.add_argument("--save_dir",     type=str,   default="./results/cp_experiment/CP_R64")
    parser.add_argument("--model_path",   type=str,   default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--dataset_name",type=str,   default="MJHQ", choices=["MJHQ", "sDCI"])
    parser.add_argument("--act_mode",     type=str,   default="NVFP4")
    parser.add_argument("--wgt_mode",     type=str,   default="NVFP4",
                        help="Quantization mode for CP factors")
    parser.add_argument("--alpha",        type=float, default=0.5)
    parser.add_argument("--lowrank",      type=int,   default=64)
    parser.add_argument("--block_size",   type=int,   default=16)
    parser.add_argument("--als_iters",    type=int,   default=100,
                        help="Max ALS iterations for PARAFAC")
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    s_target = 2 if args.test_run else args.num_samples
    dataset_ref_dir  = os.path.join(args.ref_dir,  args.dataset_name)
    dataset_save_dir = os.path.join(args.save_dir, args.dataset_name)

    if accelerator.is_main_process:
        os.makedirs(dataset_ref_dir,  exist_ok=True)
        os.makedirs(dataset_save_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  CP (PARAFAC) Quantization Experiment")
        print(f"  Act={args.act_mode}  Wgt={args.wgt_mode}  r={args.lowrank}")
        print(f"  ALS_iters={args.als_iters}")
        print(f"  Samples={s_target}  Dataset={args.dataset_name}")
        print(f"  Save → {dataset_save_dir}")
        print(f"{'='*60}\n")

    prompts = get_prompts(s_target, args)
    s_count = len(prompts)
    p_count = 2 if args.test_run else min(64, s_count)
    t_count = 20
    t_total_start = time.time()

    # ----------------------------------------------------------
    # Phase 1: Reference FP16 images
    # ----------------------------------------------------------
    if accelerator.is_main_process:
        missing = [i for i in range(s_count)
                   if not os.path.exists(os.path.join(dataset_ref_dir, f"ref_{i}.png"))]
        if missing:
            accelerator.print(f"[Phase 1] Generating {len(missing)} reference images...")
            pipe_ref = PixArtAlphaPipeline.from_pretrained(
                args.model_path, torch_dtype=torch.float16).to(device)
            for i in missing:
                gen = torch.Generator(device=device).manual_seed(42 + i)
                img = pipe_ref(prompts[i], num_inference_steps=t_count, generator=gen).images[0]
                img.save(os.path.join(dataset_ref_dir, f"ref_{i}.png"))
            del pipe_ref
            torch.cuda.empty_cache()
            gc.collect()
        else:
            accelerator.print("[Phase 1] Reference images exist, skipping.")
    accelerator.wait_for_everyone()

    # ----------------------------------------------------------
    # Phase 2: CP PTQ
    # ----------------------------------------------------------
    t_ptq_start = time.time()
    accelerator.print("[Phase 2] Loading model and applying CP (PARAFAC) PTQ...")

    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    transformer = pipe.transformer

    skip_kw = ["x_embedder", "t_embedder", "proj_out"]
    target_names = [
        n for n, m_ in transformer.named_modules()
        if isinstance(m_, nn.Linear) and not any(kw in n for kw in skip_kw)
    ]
    accelerator.print(f"  Targeted {len(target_names)} Linear layers for CP.")

    # Calibration
    all_samples = {}
    def hook_fn(name):
        def fwd(m_, inp, out):
            flat = inp[0].detach().view(-1, inp[0].shape[-1]).abs().float()
            all_samples.setdefault(name, []).append(flat.max(dim=0)[0].cpu())
        return fwd

    hooks = [get_module_by_name(transformer, n).register_forward_hook(hook_fn(n))
             for n in target_names]
    accelerator.print(f"  Calibrating with {p_count} prompts...")
    with accelerator.split_between_processes(prompts[:p_count]) as local_p:
        for prompt in local_p:
            pipe(prompt, num_inference_steps=t_count,
                 generator=torch.Generator(device=device).manual_seed(42))
    for h in hooks:
        h.remove()

    for name in all_samples:
        local_mean = torch.stack(all_samples[name]).mean(dim=0).to(device)
        all_samples[name] = accelerator.reduce(local_mean, reduction="mean")
    accelerator.wait_for_everyone()

    # Layer replacement
    for name in tqdm(target_names, desc="CP ALS decompose + replace",
                     disable=not accelerator.is_main_process):
        orig_m = get_module_by_name(transformer, name)
        if next(orig_m.parameters()).device != device:
            continue
        new_layer = ManualCPLinear(
            orig_m, args.act_mode, args.wgt_mode,
            args.alpha, args.lowrank, args.block_size,
            als_iters=args.als_iters, dtype=torch.float16
        ).to(device)
        if name in all_samples:
            new_layer.calibrate(all_samples[name])
        set_module_by_name(transformer, name, new_layer)
    accelerator.wait_for_everyone()

    t_ptq_end = time.time()
    ptq_time  = t_ptq_end - t_ptq_start
    accelerator.print(f"[Phase 2] CP PTQ done: {ptq_time/60:.1f}m ({ptq_time:.0f}s)")

    # ----------------------------------------------------------
    # Phase 3: Evaluation
    # ----------------------------------------------------------
    t_eval_start = time.time()
    accelerator.print("[Phase 3] Generating images and computing metrics...")

    psnr_m  = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    is_m    = InceptionScore().to(device)
    fid_m   = FrechetInceptionDistance(feature=2048).to(device)

    if accelerator.is_main_process:
        clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    with accelerator.split_between_processes(list(range(s_count))) as local_idx:
        for i in local_idx:
            gen   = torch.Generator(device=device).manual_seed(42 + i)
            q_img = pipe(prompts[i], num_inference_steps=t_count, generator=gen).images[0]
            q_img.save(os.path.join(dataset_save_dir, f"sample_{i}.png"))

            r_img = Image.open(os.path.join(dataset_ref_dir, f"ref_{i}.png")).convert("RGB")
            q_ten = ToTensor()(q_img).unsqueeze(0).to(device)
            r_ten = ToTensor()(r_img).unsqueeze(0).to(device)

            psnr_m.update(q_ten, r_ten)
            ssim_m.update(q_ten, r_ten)
            lpips_m.update(q_ten * 2 - 1, r_ten * 2 - 1)
            q_u8 = (q_ten * 255).to(torch.uint8)
            r_u8 = (r_ten * 255).to(torch.uint8)
            is_m.update(q_u8)
            fid_m.update(r_u8, real=True)
            fid_m.update(q_u8, real=False)
            print(f"  GPU {accelerator.process_index} → sample_{i}.png", flush=True)

    accelerator.wait_for_everyone()
    eval_time  = time.time() - t_eval_start
    total_time = time.time() - t_total_start

    # ----------------------------------------------------------
    # Phase 4: Aggregate and save metrics
    # ----------------------------------------------------------
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
            inputs = clip_processor(text=[prompts[i]], images=img,
                                    return_tensors="pt", padding=True).to(device)
            clip_scores.append(float(clip_model(**inputs).logits_per_image.item()))

        # Compression for representative layer
        m_rep, n_rep, r = 4608, 1152, args.lowrank
        n1, n2 = _factor_split(n_rep)
        total_cp   = (m_rep + n1 + n2) * r * 4    # NVFP4
        fp16_bits  = m_rep * n_rep * 16
        nvfp4_bits = m_rep * n_rep * 4
        comp_info = {
            "wgt_mode": args.wgt_mode,
            "rank": r, "n1": n1, "n2": n2,
            "representative_layer": f"{m_rep}×{n_rep}",
            "compression_vs_fp16":  round(fp16_bits  / total_cp, 2),
            "compression_vs_nvfp4": round(nvfp4_bits / total_cp, 2),
            "pct_of_fp16":   round(100.0 * total_cp / fp16_bits,  2),
            "pct_of_nvfp4":  round(100.0 * total_cp / nvfp4_bits, 2),
        }

        def fmt_time(s):
            m_, sec = divmod(int(s), 60)
            return f"{m_}m{sec:02d}s"

        result = {
            "config": {
                "method": "CP_PARAFAC_ALS",
                "act_mode": args.act_mode,
                "wgt_mode": args.wgt_mode,
                "rank": args.lowrank,
                "n1": n1, "n2": n2,
                "als_iters": args.als_iters,
                "block_size": args.block_size,
                "alpha": args.alpha,
                "num_samples": s_count,
                "dataset": args.dataset_name,
            },
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
            "compression": comp_info,
            "timing_sec": {
                "ptq":   round(ptq_time,   1),
                "eval":  round(eval_time,  1),
                "total": round(total_time, 1),
            },
        }

        metrics_path = os.path.join(dataset_save_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(result, f, indent=4)

        print(f"\n{'='*60}")
        print(f"  CP (PARAFAC) Results  →  {metrics_path}")
        print(f"  [PRIMARY]   FID: {res_fid:.4f}  |  IS: {res_is:.4f}")
        print(f"  [SECONDARY] PSNR: {res_psnr:.2f}  SSIM: {res_ssim:.4f}  LPIPS: {res_lpips:.4f}")
        print(f"  [COMPRESS]  vs FP16: {comp_info['compression_vs_fp16']}x  "
              f"vs NVFP4: {comp_info['compression_vs_nvfp4']}x  "
              f"({comp_info['pct_of_nvfp4']:.1f}% of NVFP4)")
        print(f"  [TIMING]    PTQ: {fmt_time(ptq_time)}  Eval: {fmt_time(eval_time)}  "
              f"Total: {fmt_time(total_time)}")
        print(f"{'='*60}\n")

    accelerator.wait_for_everyone()
    del pipe
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
