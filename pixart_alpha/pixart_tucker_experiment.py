"""
pixart_tucker_experiment.py
===========================
Tucker-2 decomposition quantization for PixArt-XL-2.

W ≈ U1 @ G @ U2ᵀ  (HOSVD)
  U1 (m×r): NVFP4 quantized (left factor)
  G  (r×r): INT3  quantized (dense core - aggressively quantized since small)
  U2 (n×r): NVFP4 quantized (right factor)
  act: NVFP4

Compression vs NVFP4 baseline (m=4608, n=1152, r=64):
  Tucker: (m*r*4 + r*r*3 + n*r*4) / (m*n*4) ≈ 7% → ~14x compression

Usage:
  accelerate launch --multi_gpu --num_processes 2 pixart_tucker_experiment.py \\
      --lowrank 64 --save_dir results/tucker_experiment/TUCKER_R64
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
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
# Tucker-2 quantized linear layer
# ==========================================

class ManualTuckerLinear(nn.Module):
    """
    Tucker-2 + LoRA error correction.

    W ≈ Tucker(W) + LoRA(W - Tucker(W))

    Tucker part (quantized):
      u2_q (r, n): NVFP4  — compressed right factor
      g_q  (r, r): INT3   — compressed core
      u1_q (m, r): NVFP4  — compressed left factor

    LoRA correction (FP16, corrects truncation error):
      lora_a (lora_r, n): SVD on (W_smooth - Tucker_approx)
      lora_b (m, lora_r): SVD on (W_smooth - Tucker_approx)

    Forward:
      tucker_out = x_q → U2 → Gᵀ → U1ᵀ        (quantized path)
      lora_out   = x_smooth → lora_a → lora_b   (error correction, FP16)
      out = tucker_out + lora_out

    Why LoRA is needed: Tucker rank=64 retains only 5.6% of 1152 singular values
    (for 4608×1152 layers). Without correction, PSNR ≈ 5-6 dB (unusable).
    LoRA on truncation residual restores quality to competitive level.
    """

    def __init__(self, original_linear, act_mode="NVFP4", wgt_core_mode="INT3",
                 alpha=0.5, rank=64, lora_rank=32, block_size=16, dtype=torch.float16):
        super().__init__()
        self.target_dtype = dtype
        self.act_mode = act_mode
        self.wgt_core_mode = wgt_core_mode
        self.alpha = alpha
        self.rank = rank
        self.lora_rank = lora_rank
        self.block_size = block_size
        m, n = original_linear.weight.data.shape
        self.m, self.n = m, n

        self.register_buffer("weight", original_linear.weight.data.clone().to(dtype))
        self.bias = (nn.Parameter(original_linear.bias.data.clone().to(dtype))
                     if original_linear.bias is not None else None)
        self.register_buffer("smooth_scale", torch.ones(n, dtype=dtype))

        r = rank
        # Tucker factors (F.linear convention: weight shape = out×in)
        self.register_buffer("u2_q", torch.zeros(r, n, dtype=dtype))   # U2ᵀ: (r, n)
        self.register_buffer("g_q",  torch.zeros(r, r, dtype=dtype))   # G:   (r, r)
        self.register_buffer("u1_q", torch.zeros(m, r, dtype=dtype))   # U1:  (m, r)

        # LoRA correction for truncation error (FP16)
        self.register_buffer("lora_a", torch.zeros(lora_rank, n, dtype=dtype))  # (lr, n)
        self.register_buffer("lora_b", torch.zeros(m, lora_rank, dtype=dtype))  # (m, lr)
        self.is_calibrated = False

    @torch.no_grad()
    def calibrate(self, x_max):
        """HOSVD decomposition + quantize factors + SVD on truncation error."""
        x_max = x_max.clamp(min=1e-5).float()
        W = self.weight.float()  # (m, n)

        # SmoothQuant scale
        w_max = W.abs().max(dim=0)[0].clamp(min=1e-5)
        smooth = (w_max.pow(1 - self.alpha) / x_max.pow(self.alpha)).clamp(1e-4, 1e4)
        self.smooth_scale.data = smooth.to(self.target_dtype)
        W_smooth = W / smooth.unsqueeze(0)  # (m, n)

        # HOSVD: W_smooth = U Σ Vᵀ  (full SVD reused for both Tucker and LoRA)
        U, S, Vh = torch.linalg.svd(W_smooth, full_matrices=False)
        r = min(self.rank, S.shape[0])
        U1  = U[:, :r]       # (m, r)
        U2t = Vh[:r, :]      # (r, n)  = U2ᵀ
        U2  = U2t.T          # (n, r)

        # Core tensor G = U1ᵀ @ W_smooth @ U2
        G = U1.T @ W_smooth @ U2   # (r, r)

        # Quantize Tucker factors
        self.u1_q.data  = quantize_to_nvfp4(U1, self.block_size).to(self.target_dtype)
        self.g_q.data   = quantize_uniform(G, self.block_size,
                                           mode=self.wgt_core_mode).to(self.target_dtype)
        self.u2_q.data  = quantize_to_nvfp4(U2t, self.block_size).to(self.target_dtype)

        # Truncation error: W_smooth - Tucker_approx
        # Tucker_approx = U1 @ G @ U2ᵀ = U[:,:r] @ diag(S[:r]) @ Vh[:r,:] (same as truncated SVD)
        Tucker_approx = U1 @ G @ U2t      # (m, n)
        trunc_error   = W_smooth - Tucker_approx

        # SVD on truncation error → LoRA correction
        lr = min(self.lora_rank, trunc_error.shape[0], trunc_error.shape[1])
        Ue, Se, Vhe = torch.linalg.svd(trunc_error.float(), full_matrices=False)
        sqrt_Se = torch.sqrt(Se[:lr])
        lora_a_data = torch.zeros(self.lora_rank, self.n,
                                  dtype=self.target_dtype, device=x_max.device)
        lora_a_data[:lr] = (Vhe[:lr] * sqrt_Se.unsqueeze(1)).to(self.target_dtype)
        self.lora_a.data = lora_a_data

        lora_b_data = torch.zeros(self.m, self.lora_rank,
                                  dtype=self.target_dtype, device=x_max.device)
        lora_b_data[:, :lr] = (Ue[:, :lr] * sqrt_Se.unsqueeze(0)).to(self.target_dtype)
        self.lora_b.data = lora_b_data

        self.is_calibrated = True

    def compression_stats(self):
        """Return compression ratios (Tucker factors only; LoRA is overhead)."""
        m, n, r, lr = self.m, self.n, self.rank, self.lora_rank
        bits_u1    = m * r  * 4   # NVFP4
        bits_g     = r * r  * 3   # INT3 core
        bits_u2    = r * n  * 4   # NVFP4
        bits_lora  = (lr * n + m * lr) * 16   # FP16 LoRA correction
        total      = bits_u1 + bits_g + bits_u2 + bits_lora
        total_tucker_only = bits_u1 + bits_g + bits_u2
        fp16_bits   = m * n * 16
        nvfp4_bits  = m * n * 4
        return {
            "bits_tucker_only": total_tucker_only,
            "bits_lora_fp16":   bits_lora,
            "total_bits":       total,
            "compression_vs_fp16_tucker_only":  round(fp16_bits  / total_tucker_only, 2),
            "compression_vs_nvfp4_tucker_only": round(nvfp4_bits / total_tucker_only, 2),
            "compression_vs_fp16_with_lora":    round(fp16_bits  / total, 2),
            "compression_vs_nvfp4_with_lora":   round(nvfp4_bits / total, 2),
            "pct_of_nvfp4_with_lora":           round(100.0 * total / nvfp4_bits, 2),
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

        # Tucker path: x_q → U2 → Gᵀ → U1ᵀ
        h1  = F.linear(x_q, self.u2_q)    # (batch, r)
        h2  = F.linear(h1,  self.g_q)     # (batch, r)
        out = F.linear(h2,  self.u1_q)    # (batch, m)

        # LoRA correction: x_smooth → lora_a → lora_b (truncation error compensation)
        out = out + F.linear(F.linear(x_smooth, self.lora_a), self.lora_b)

        if self.bias is not None:
            out = out + self.bias
        return out.to(input_dtype)


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Tucker-2 PTQ for PixArt-XL-2")
    parser.add_argument("--num_samples",  type=int,   default=20)
    parser.add_argument("--test_run",     action="store_true")
    parser.add_argument("--ref_dir",      type=str,   default="./ref_images")
    parser.add_argument("--save_dir",     type=str,   default="./results/tucker_experiment/TUCKER_R64")
    parser.add_argument("--model_path",   type=str,   default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--dataset_name",type=str,   default="MJHQ", choices=["MJHQ", "sDCI"])
    parser.add_argument("--act_mode",     type=str,   default="NVFP4")
    parser.add_argument("--wgt_core_mode",type=str,   default="INT3",
                        help="Quantization mode for Tucker core G (INT3/INT4/NVFP4)")
    parser.add_argument("--alpha",        type=float, default=0.5)
    parser.add_argument("--lowrank",      type=int,   default=64,
                        help="Tucker rank r (number of singular values kept)")
    parser.add_argument("--lora_rank",    type=int,   default=32,
                        help="LoRA rank for truncation error correction")
    parser.add_argument("--block_size",   type=int,   default=16)
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
        print(f"  Tucker-2 + LoRA Quantization Experiment")
        print(f"  Act={args.act_mode}  Core={args.wgt_core_mode}  tucker_r={args.lowrank}  lora_r={args.lora_rank}")
        print(f"  Samples={s_target}  Dataset={args.dataset_name}")
        print(f"  Save → {dataset_save_dir}")
        print(f"{'='*60}\n")

    prompts = get_prompts(s_target, args)
    s_count = len(prompts)
    p_count = 2 if args.test_run else min(64, s_count)
    t_count = 20
    t_total_start = time.time()

    # ----------------------------------------------------------
    # Phase 1: Reference FP16 images (skip if exist)
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
    # Phase 2: Tucker PTQ
    # ----------------------------------------------------------
    t_ptq_start = time.time()
    accelerator.print("[Phase 2] Loading model and applying Tucker-2 PTQ...")

    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    transformer = pipe.transformer

    skip_kw = ["x_embedder", "t_embedder", "proj_out"]
    target_names = [
        n for n, m_ in transformer.named_modules()
        if isinstance(m_, nn.Linear) and not any(kw in n for kw in skip_kw)
    ]
    accelerator.print(f"  Targeted {len(target_names)} Linear layers for Tucker-2.")

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
    comp_stats_list = []
    for name in tqdm(target_names, desc="Tucker-2 decompose + replace",
                     disable=not accelerator.is_main_process):
        orig_m = get_module_by_name(transformer, name)
        if next(orig_m.parameters()).device != device:
            continue
        new_layer = ManualTuckerLinear(
            orig_m, args.act_mode, args.wgt_core_mode,
            args.alpha, args.lowrank, args.lora_rank, args.block_size,
            dtype=torch.float16
        ).to(device)
        if name in all_samples:
            new_layer.calibrate(all_samples[name])
            if accelerator.is_main_process and len(comp_stats_list) == 0:
                comp_stats_list.append(new_layer.compression_stats())
        set_module_by_name(transformer, name, new_layer)
    accelerator.wait_for_everyone()

    t_ptq_end = time.time()
    ptq_time = t_ptq_end - t_ptq_start
    accelerator.print(f"[Phase 2] Tucker-2 PTQ done: {ptq_time/60:.1f}m ({ptq_time:.0f}s)")

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

        # Compression stats for representative layer
        m_rep, n_rep, r, lr = 4608, 1152, args.lowrank, args.lora_rank
        bits_u1    = m_rep * r  * 4    # NVFP4
        bits_g     = r     * r  * 3    # INT3 core
        bits_u2    = r     * n_rep * 4 # NVFP4
        bits_lora  = (lr * n_rep + m_rep * lr) * 16  # FP16 LoRA
        total_tucker_only = bits_u1 + bits_g + bits_u2
        total_with_lora   = total_tucker_only + bits_lora
        fp16_bits  = m_rep * n_rep * 16
        nvfp4_bits = m_rep * n_rep * 4
        comp_info = {
            "u1_mode": "NVFP4", "g_mode": args.wgt_core_mode, "u2_mode": "NVFP4",
            "lora_mode": "FP16",
            "tucker_rank": r, "lora_rank": lr,
            "representative_layer": f"{m_rep}×{n_rep}",
            "compression_vs_fp16_tucker_only":  round(fp16_bits  / total_tucker_only, 2),
            "compression_vs_nvfp4_tucker_only": round(nvfp4_bits / total_tucker_only, 2),
            "compression_vs_fp16":  round(fp16_bits  / total_with_lora, 2),
            "compression_vs_nvfp4": round(nvfp4_bits / total_with_lora, 2),
            "pct_of_nvfp4": round(100.0 * total_with_lora / nvfp4_bits, 2),
        }

        def fmt_time(s):
            m_, sec = divmod(int(s), 60)
            return f"{m_}m{sec:02d}s"

        result = {
            "config": {
                "method": "Tucker-2_HOSVD+LoRA",
                "act_mode": args.act_mode,
                "wgt_u_mode": "NVFP4",
                "wgt_core_mode": args.wgt_core_mode,
                "tucker_rank": args.lowrank,
                "lora_rank": args.lora_rank,
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
        print(f"  Tucker-2+LoRA Results  →  {metrics_path}")
        print(f"  [PRIMARY]   FID: {res_fid:.4f}  |  IS: {res_is:.4f}")
        print(f"  [SECONDARY] PSNR: {res_psnr:.2f}  SSIM: {res_ssim:.4f}  LPIPS: {res_lpips:.4f}")
        print(f"  [COMPRESS]  Tucker-only: {comp_info['compression_vs_nvfp4_tucker_only']}x vs NVFP4")
        print(f"              With LoRA:   {comp_info['compression_vs_nvfp4']}x vs NVFP4  "
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
