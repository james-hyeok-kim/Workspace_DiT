"""
pixart_ksvd_experiment.py
=========================
K-SVD / Dictionary Learning sparse coding quantization for PixArt-XL-2.

W ≈ D @ X   where:
  D ∈ R^(m × K)  — learned dictionary  (quantized at various precisions)
  X ∈ R^(K × n)  — sparse code matrix  (s nonzeros per column, stored as sparse)

Compression analysis (m=4608, n=1152, K=256, s=16):
  Dictionary bits: m * K * bits_D
  Sparse code bits: s * n * (bits_val + bits_idx)
    - bits_val  = precision of nonzero values (FP16/NVFP4/INT3)
    - bits_idx  = ceil(log2(K)) = 8 bits for K=256
  vs FP16 baseline: m * n * 16

  Example (D=NVFP4, val=NVFP4, K=256, s=16):
    D:   4608 * 256 * 4    = 4,718,592 bits
    X:   16   * 1152 * (4+8) = 221,184 bits
    Total: 4,939,776 bits  vs FP16: 84,934,656 bits → ~17x vs FP16, ~4.3x vs NVFP4

Algorithm: Approximate K-SVD via batch OMP (Orthogonal Matching Pursuit)
  1. Initialize D with K random normalized columns from W columns
  2. For each iteration:
     a. Sparse coding: OMP on each column of W with sparsity s
     b. Dictionary update: for each atom d_k, update via SVD on residuals
        that use d_k
  3. Quantize D at target precision

Forward:
  x (batch×n) → smooth → quant_act → sparse matmul X → D → out (batch×m)
  But sparse matmul of x@X is expensive; instead reconstruct W_approx = D@X
  and use it as a dense quantized weight (efficient at inference time).
  At inference: store W_approx quantized → standard F.linear

Usage:
  accelerate launch --multi_gpu --num_processes 2 pixart_ksvd_experiment.py \\
      --dict_size 256 --sparsity 16 \\
      --dict_mode FP16 --save_dir results/ksvd_experiment/KSVD_K256_S16_DFP16

  # Sweep all dict modes:
  bash run_ksvd_experiment.sh
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
# OMP and K-SVD utilities
# ==========================================

def omp_batch(D, Y, sparsity):
    """
    Orthogonal Matching Pursuit — batch version.
    D: (m, K)  normalized dictionary atoms (columns)
    Y: (m, n)  signal matrix (columns are signals to represent)
    sparsity: s  number of nonzeros per column of X

    Returns X: (K, n) sparse code matrix (dense float, zeros where inactive)
    """
    m, K = D.shape
    _, n = Y.shape
    device = D.device

    X = torch.zeros(K, n, device=device, dtype=D.dtype)

    # Precompute Gram matrix for efficient updates
    # Process all columns simultaneously using batched operations
    residual = Y.clone()          # (m, n)
    selected = [[] for _ in range(n)]  # active sets per signal

    # Precompute D.T @ D for Cholesky updates (optional, skip for clarity)
    Dt = D.T                      # (K, m)

    for step in range(sparsity):
        # Correlations: (K, n) = (K, m) @ (m, n)
        corr = Dt @ residual       # (K, n)
        best = corr.abs().argmax(dim=0)  # (n,) — best atom per signal

        # For each unique atom selected, update the corresponding signals
        for j in range(n):
            k = best[j].item()
            if k not in selected[j]:
                selected[j].append(k)
            # Least-squares update for signal j over its active set
            active = selected[j]
            D_a = D[:, active]     # (m, |active|)
            # Solve min ||Y[:,j] - D_a @ x_a||
            x_a, _, _, _ = torch.linalg.lstsq(D_a, Y[:, j:j+1])
            X[active, j] = x_a[:, 0]
            residual[:, j] = Y[:, j] - D_a @ x_a[:, 0]

    return X


def omp_batch_fast(D, Y, sparsity):
    """
    Faster OMP using vectorized operations (approximate: greedy atom selection
    without per-signal LS refinement at each step — Matching Pursuit variant).
    Sufficient for dictionary learning initialization.

    D: (m, K), Y: (m, n)
    Returns X: (K, n)
    """
    m, K = D.shape
    _, n = Y.shape
    device = D.device

    X        = torch.zeros(K, n, device=device, dtype=torch.float32)
    residual = Y.float().clone()
    D_f      = D.float()
    Dt       = D_f.T                       # (K, m)
    active   = torch.full((n, sparsity), -1, dtype=torch.long, device=device)

    for step in range(sparsity):
        corr = (Dt @ residual).abs()       # (K, n)
        best = corr.argmax(dim=0)          # (n,) — greedy atom index
        active[:, step] = best

        # Coefficient update: project residual onto selected atom
        d_best   = D_f[:, best]            # (m, n)  one atom per signal
        coeff    = (d_best * residual).sum(0) / (d_best * d_best).sum(0).clamp(min=1e-12)
        X[best, torch.arange(n, device=device)] += coeff
        residual -= d_best * coeff.unsqueeze(0)

    return X


def ksvd(W, K, sparsity, max_iter=10, device=None):
    """
    K-SVD algorithm: learn dictionary D and sparse codes X such that W ≈ D @ X.

    W: (m, n)  weight matrix
    K: int     dictionary size (number of atoms)
    sparsity: int  max nonzeros per column of X (s)
    max_iter: int  K-SVD iterations

    Returns: D (m, K), X (K, n)
    """
    if device is None:
        device = W.device
    W = W.float().to(device)
    m, n = W.shape

    # Initialize D: random normalized columns from W
    idx  = torch.randperm(n, device=device)[:K]
    D    = W[:, idx].clone()
    # Normalize each atom
    norms = D.norm(dim=0, keepdim=True).clamp(min=1e-12)
    D = D / norms                          # (m, K)

    for iteration in range(max_iter):
        # --- Sparse coding stage ---
        X = omp_batch_fast(D, W, sparsity)  # (K, n)

        # --- Dictionary update stage ---
        # Update each atom d_k using SVD on the error matrix
        for k in range(K):
            # Find signals that use atom k
            use_k = X[k].abs() > 1e-10     # (n,) boolean
            if use_k.sum() == 0:
                # Dead atom: re-initialize to worst-represented signal
                resid = (W - D @ X).norm(dim=0)
                worst = resid.argmax().item()
                D[:, k] = W[:, worst] / W[:, worst].norm().clamp(min=1e-12)
                continue

            # Restricted error: E_k = W_k - Σ_{j≠k} d_j x_j
            X_k_row = X[k:k+1, :]          # (1, n)
            E_k = W[:, use_k] - (D @ X)[:, use_k] + D[:, k:k+1] @ X_k_row[:, use_k]
            # E_k: (m, |use_k|)

            # Update via rank-1 SVD
            try:
                U, sv, Vh = torch.linalg.svd(E_k, full_matrices=False)
                D[:, k]        = U[:, 0]
                X[k, use_k]    = sv[0] * Vh[0, :]
            except Exception:
                pass  # keep old atom if SVD fails

    return D, X


def quantize_dict(D, mode, block_size=16):
    """Quantize dictionary D according to mode."""
    if mode == "FP16":
        return D.half()
    elif mode == "FP8":
        # Emulate FP8 E4M3 range (~448 max, ~1/16384 min)
        D_f = D.float()
        orig_shape = D_f.shape
        flat = D_f.flatten()
        pad  = (block_size - flat.numel() % block_size) % block_size
        if pad > 0:
            flat = torch.cat([flat, torch.zeros(pad, device=D.device)])
        x_bl = flat.view(-1, block_size)
        amax  = x_bl.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = amax / 448.0
        levels = 256  # 8-bit
        x_q = torch.clamp((x_bl / scale).round(), -127, 127)
        return (x_q * scale).view(-1)[:D_f.numel()].view(orig_shape).half()
    elif mode == "NVFP4":
        return quantize_to_nvfp4(D.float(), block_size).half()
    elif mode in ("INT3", "INT4", "INT8"):
        return quantize_uniform(D.float(), block_size, mode=mode).half()
    else:
        raise ValueError(f"Unsupported dict_mode: {mode}")


def dict_bits(mode):
    """Return bits per element for the given quantization mode."""
    return {"FP16": 16, "FP8": 8, "NVFP4": 4, "INT4": 4, "INT3": 3, "INT8": 8}.get(mode, 16)


# ==========================================
# K-SVD quantized linear layer
# ==========================================

class ManualKSVDLinear(nn.Module):
    """
    K-SVD dictionary-learning quantized linear layer.

    At PTQ time:
      1. Learn D (m×K) and X (K×n) via K-SVD so W ≈ D @ X
      2. Quantize D at dict_mode precision
      3. Reconstruct W_approx = D_q @ X  (dense, quantized at wgt_reconstruct_mode)
      4. Store W_approx as the inference weight (standard quantized linear)

    Forward: SmoothQuant → quantize activation → F.linear(x_q, W_approx_q)
    This is equivalent to quantizing the K-SVD reconstruction rather than
    the original weight — the quality of reconstruction determines image quality.

    The compression stats show what would be needed if you stored D+X sparsely
    instead of the dense reconstruction.
    """

    def __init__(self, original_linear, act_mode="NVFP4", dict_mode="FP16",
                 wgt_reconstruct_mode="NVFP4", alpha=0.5,
                 dict_size=256, sparsity=16, ksvd_iters=10,
                 block_size=16, dtype=torch.float16):
        super().__init__()
        self.target_dtype   = dtype
        self.act_mode       = act_mode
        self.dict_mode      = dict_mode
        self.wgt_reconstruct_mode = wgt_reconstruct_mode
        self.alpha          = alpha
        self.dict_size      = dict_size
        self.sparsity       = sparsity
        self.ksvd_iters     = ksvd_iters
        self.block_size     = block_size

        m, n = original_linear.weight.data.shape
        self.m, self.n = m, n

        self.register_buffer("weight", original_linear.weight.data.clone().to(dtype))
        self.bias = (nn.Parameter(original_linear.bias.data.clone().to(dtype))
                     if original_linear.bias is not None else None)
        self.register_buffer("smooth_scale", torch.ones(n, dtype=dtype))
        self.register_buffer("w_approx_q",   original_linear.weight.data.clone().to(dtype))
        self.is_calibrated = False

        # Saved for compression reporting
        self._ksvd_nnz = 0       # actual nonzeros in X after K-SVD
        self._ksvd_err = 0.0     # relative reconstruction error ||W - DX|| / ||W||

    @torch.no_grad()
    def calibrate(self, x_max):
        """K-SVD decompose + quantize reconstruction."""
        x_max = x_max.clamp(min=1e-5).float()
        W = self.weight.float()   # (m, n)

        # SmoothQuant scale
        w_max = W.abs().max(dim=0)[0].clamp(min=1e-5)
        smooth = (w_max.pow(1 - self.alpha) / x_max.pow(self.alpha)).clamp(1e-4, 1e4)
        self.smooth_scale.data = smooth.to(self.target_dtype)
        W_smooth = W / smooth.unsqueeze(0)  # (m, n)

        # K-SVD
        K = min(self.dict_size, self.n)
        D, X = ksvd(W_smooth, K, self.sparsity, self.ksvd_iters, device=W.device)

        # Record stats
        self._ksvd_nnz = int((X.abs() > 1e-10).sum().item())
        W_rec = D @ X
        self._ksvd_err = float(
            (W_smooth - W_rec).norm() / W_smooth.norm().clamp(min=1e-12))

        # Quantize dictionary D
        D_q = quantize_dict(D, self.dict_mode, self.block_size)

        # Reconstruct with quantized D
        W_approx = D_q.float() @ X.float()   # (m, n)

        # Quantize reconstruction at inference precision
        if self.wgt_reconstruct_mode == "NVFP4":
            w_q = quantize_to_nvfp4(W_approx, self.block_size)
        else:
            w_q = quantize_uniform(W_approx, self.block_size, mode=self.wgt_reconstruct_mode)

        self.w_approx_q.data = w_q.to(self.target_dtype)
        self.is_calibrated = True

    def compression_stats(self):
        m, n, K, s = self.m, self.n, self.dict_size, self.sparsity
        bits_d  = dict_bits(self.dict_mode)
        bits_val = dict_bits(self.wgt_reconstruct_mode)
        bits_idx = math.ceil(math.log2(max(K, 2)))   # index bits for K atoms

        total_dict    = m * K * bits_d
        total_sparse  = s * n * (bits_val + bits_idx)
        total_ksvd    = total_dict + total_sparse
        fp16_bits     = m * n * 16
        nvfp4_bits    = m * n * 4

        return {
            "dict_mode":   self.dict_mode,
            "K":           K,
            "s":           s,
            "bits_dict":   bits_d,
            "bits_val":    bits_val,
            "bits_idx":    bits_idx,
            "total_dict_bits":   total_dict,
            "total_sparse_bits": total_sparse,
            "total_ksvd_bits":   total_ksvd,
            "compression_vs_fp16":  round(fp16_bits  / total_ksvd, 2),
            "compression_vs_nvfp4": round(nvfp4_bits / total_ksvd, 2),
            "pct_of_fp16":   round(100.0 * total_ksvd / fp16_bits,  2),
            "pct_of_nvfp4":  round(100.0 * total_ksvd / nvfp4_bits, 2),
            "ksvd_reconstruction_error": round(self._ksvd_err, 6),
            "ksvd_actual_nnz":           self._ksvd_nnz,
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

        out = F.linear(x_q, self.w_approx_q)
        if self.bias is not None:
            out = out + self.bias
        return out.to(input_dtype)


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="K-SVD Dictionary Learning PTQ for PixArt-XL-2")
    parser.add_argument("--num_samples",   type=int,   default=20)
    parser.add_argument("--test_run",      action="store_true")
    parser.add_argument("--ref_dir",       type=str,   default="./ref_images")
    parser.add_argument("--save_dir",      type=str,   default="./results/ksvd_experiment/KSVD_K256_S16_DFP16")
    parser.add_argument("--model_path",    type=str,   default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--dataset_name", type=str,   default="MJHQ", choices=["MJHQ", "sDCI"])
    parser.add_argument("--act_mode",      type=str,   default="NVFP4")
    parser.add_argument("--dict_mode",     type=str,   default="FP16",
                        choices=["FP16", "FP8", "NVFP4", "INT4", "INT3"],
                        help="Dictionary D quantization precision")
    parser.add_argument("--wgt_reconstruct_mode", type=str, default="NVFP4",
                        choices=["NVFP4", "INT4", "INT3", "INT8"],
                        help="Quantization of reconstructed W=DX at inference")
    parser.add_argument("--dict_size",     type=int,   default=256,
                        help="K: number of dictionary atoms")
    parser.add_argument("--sparsity",      type=int,   default=16,
                        help="s: nonzeros per column of X")
    parser.add_argument("--ksvd_iters",   type=int,   default=10,
                        help="K-SVD iterations")
    parser.add_argument("--alpha",         type=float, default=0.5)
    parser.add_argument("--block_size",    type=int,   default=16)
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    s_target = 2 if args.test_run else args.num_samples
    dataset_ref_dir  = os.path.join(args.ref_dir,  args.dataset_name)
    dataset_save_dir = os.path.join(args.save_dir, args.dataset_name)

    if accelerator.is_main_process:
        os.makedirs(dataset_ref_dir,  exist_ok=True)
        os.makedirs(dataset_save_dir, exist_ok=True)

        # Pre-compute compression for representative layer
        m_rep, n_rep = 4608, 1152
        K, s = args.dict_size, args.sparsity
        bits_d   = dict_bits(args.dict_mode)
        bits_val = dict_bits(args.wgt_reconstruct_mode)
        bits_idx = math.ceil(math.log2(max(K, 2)))
        total_ksvd = m_rep * K * bits_d + s * n_rep * (bits_val + bits_idx)
        fp16_bits  = m_rep * n_rep * 16
        nvfp4_bits = m_rep * n_rep * 4

        print(f"\n{'='*65}")
        print(f"  K-SVD Dictionary Learning Quantization")
        print(f"  Act={args.act_mode}  Dict={args.dict_mode}  Recon={args.wgt_reconstruct_mode}")
        print(f"  K={K}  s={s}  ksvd_iters={args.ksvd_iters}")
        print(f"  Compression (repr {m_rep}×{n_rep}):")
        print(f"    vs FP16:  {fp16_bits/total_ksvd:.1f}x  ({100*total_ksvd/fp16_bits:.1f}% of FP16)")
        print(f"    vs NVFP4: {nvfp4_bits/total_ksvd:.1f}x  ({100*total_ksvd/nvfp4_bits:.1f}% of NVFP4)")
        print(f"  Samples={s_target}  Dataset={args.dataset_name}")
        print(f"  Save → {dataset_save_dir}")
        print(f"{'='*65}\n")

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
    # Phase 2: K-SVD PTQ
    # ----------------------------------------------------------
    t_ptq_start = time.time()
    accelerator.print(f"[Phase 2] Loading model and applying K-SVD PTQ "
                      f"(K={args.dict_size}, s={args.sparsity}, dict={args.dict_mode})...")

    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    transformer = pipe.transformer

    skip_kw = ["x_embedder", "t_embedder", "proj_out"]
    target_names = [
        n for n, m_ in transformer.named_modules()
        if isinstance(m_, nn.Linear) and not any(kw in n for kw in skip_kw)
    ]
    accelerator.print(f"  Targeted {len(target_names)} Linear layers for K-SVD.")

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
    recon_errors = []
    for name in tqdm(target_names, desc=f"K-SVD (K={args.dict_size} s={args.sparsity})",
                     disable=not accelerator.is_main_process):
        orig_m = get_module_by_name(transformer, name)
        if next(orig_m.parameters()).device != device:
            continue
        new_layer = ManualKSVDLinear(
            orig_m,
            act_mode=args.act_mode,
            dict_mode=args.dict_mode,
            wgt_reconstruct_mode=args.wgt_reconstruct_mode,
            alpha=args.alpha,
            dict_size=args.dict_size,
            sparsity=args.sparsity,
            ksvd_iters=args.ksvd_iters,
            block_size=args.block_size,
            dtype=torch.float16,
        ).to(device)
        if name in all_samples:
            new_layer.calibrate(all_samples[name])
            recon_errors.append(new_layer._ksvd_err)
        set_module_by_name(transformer, name, new_layer)
    accelerator.wait_for_everyone()

    t_ptq_end = time.time()
    ptq_time  = t_ptq_end - t_ptq_start
    mean_err  = float(np.mean(recon_errors)) if recon_errors else 0.0
    accelerator.print(f"[Phase 2] K-SVD PTQ done: {ptq_time/60:.1f}m ({ptq_time:.0f}s)  "
                      f"mean_recon_err={mean_err:.4f}")

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
    # Phase 4: Aggregate and save
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

        # Compression stats for representative layer
        m_rep, n_rep = 4608, 1152
        K, s = args.dict_size, args.sparsity
        bits_d   = dict_bits(args.dict_mode)
        bits_val = dict_bits(args.wgt_reconstruct_mode)
        bits_idx = math.ceil(math.log2(max(K, 2)))
        total_ksvd = m_rep * K * bits_d + s * n_rep * (bits_val + bits_idx)
        fp16_bits  = m_rep * n_rep * 16
        nvfp4_bits = m_rep * n_rep * 4
        comp_info = {
            "dict_mode":    args.dict_mode,
            "recon_mode":   args.wgt_reconstruct_mode,
            "K":            K,
            "s":            s,
            "bits_dict":    bits_d,
            "bits_val":     bits_val,
            "bits_idx":     bits_idx,
            "representative_layer": f"{m_rep}×{n_rep}",
            "total_ksvd_bits":      total_ksvd,
            "compression_vs_fp16":  round(fp16_bits  / total_ksvd, 2),
            "compression_vs_nvfp4": round(nvfp4_bits / total_ksvd, 2),
            "pct_of_fp16":   round(100.0 * total_ksvd / fp16_bits,  2),
            "pct_of_nvfp4":  round(100.0 * total_ksvd / nvfp4_bits, 2),
            "mean_reconstruction_error": round(mean_err, 6),
        }

        def fmt_time(s_):
            m_, sec = divmod(int(s_), 60)
            return f"{m_}m{sec:02d}s"

        result = {
            "config": {
                "method": "K-SVD_DictionaryLearning",
                "act_mode": args.act_mode,
                "dict_mode": args.dict_mode,
                "wgt_reconstruct_mode": args.wgt_reconstruct_mode,
                "dict_size": K,
                "sparsity":  s,
                "ksvd_iters": args.ksvd_iters,
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

        print(f"\n{'='*65}")
        print(f"  K-SVD Results  →  {metrics_path}")
        print(f"  [PRIMARY]    FID: {res_fid:.4f}  |  IS: {res_is:.4f}")
        print(f"  [SECONDARY]  PSNR: {res_psnr:.2f}  SSIM: {res_ssim:.4f}  LPIPS: {res_lpips:.4f}")
        print(f"  [COMPRESS]   D={args.dict_mode} K={K} s={s}")
        print(f"               vs FP16:  {comp_info['compression_vs_fp16']}x  "
              f"({comp_info['pct_of_fp16']:.1f}% of FP16)")
        print(f"               vs NVFP4: {comp_info['compression_vs_nvfp4']}x  "
              f"({comp_info['pct_of_nvfp4']:.1f}% of NVFP4)")
        print(f"  [RECON ERR]  mean ||W-DX||/||W|| = {mean_err:.4f}")
        print(f"  [TIMING]     PTQ: {fmt_time(ptq_time)}  Eval: {fmt_time(eval_time)}  "
              f"Total: {fmt_time(total_time)}")
        print(f"{'='*65}\n")

    accelerator.wait_for_everyone()
    del pipe
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
