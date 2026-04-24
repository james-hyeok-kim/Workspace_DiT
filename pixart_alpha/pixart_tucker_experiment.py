"""
pixart_tucker_experiment.py
===========================
Tucker-2 decomposition quantization for PixArt-XL-2.

W ≈ U1 @ G @ U2ᵀ  (HOSVD)
  U1 (m×r): configurable precision (NVFP4 / FP16 / BF16 / FP32)
  G  (r×r): low precision core (INT2 / INT3 / INT4 / NVFP4)
  U2 (n×r): same as U1

LoRA correction:
  Corrects W_smooth - U1 @ G_q @ U2ᵀ  (truncation + G quantization error)
  Disable with --no_lora to measure LoRA contribution.

Rank:
  --lowrank N       : absolute rank (default 64)
  --rank_ratio R    : fraction of min(m,n), e.g. 0.5 / 0.75 / 1.0

Usage:
  accelerate launch --multi_gpu --num_processes 2 pixart_tucker_experiment.py \\
      --rank_ratio 0.75 --wgt_core_mode INT3 --wgt_u_mode FP16 --block_size 16
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
    Tucker-2 + optional LoRA error correction.

    W ≈ Tucker(W) + LoRA(W - Tucker_quantized(W))

    Tucker part:
      u2_q (r, n): U2ᵀ at wgt_u_mode precision
      g_q  (r, r): Core G at wgt_core_mode precision
      u1_q (m, r): U1  at wgt_u_mode precision

    LoRA correction (FP16):
      Corrects W_smooth - U1 @ G_q @ U2ᵀ  (total quantized Tucker error)
      This includes both truncation error AND G quantization error.
      Disabled if no_lora=True.

    Rank:
      If rank_ratio is given: r = int(rank_ratio * min(m, n)) per layer.
      Otherwise: r = rank (absolute).
    """

    def __init__(self, original_linear, act_mode="NVFP4",
                 wgt_core_mode="INT3", wgt_u_mode="NVFP4",
                 alpha=0.5, rank=64, rank_ratio=None,
                 lora_rank=32, no_lora=False, kd_mode=False,
                 block_size=16, dtype=torch.float16):
        super().__init__()
        self.target_dtype  = dtype
        self.act_mode      = act_mode
        self.wgt_core_mode = wgt_core_mode
        self.wgt_u_mode    = wgt_u_mode
        self.alpha         = alpha
        self.rank          = rank
        self.rank_ratio    = rank_ratio
        self.lora_rank     = lora_rank
        self.no_lora       = no_lora
        self.kd_mode       = kd_mode
        self.block_size    = block_size

        m, n = original_linear.weight.data.shape
        self.m, self.n = m, n

        # Compute actual rank (per-layer)
        if rank_ratio is not None:
            r = max(1, int(rank_ratio * min(m, n)))
        else:
            r = min(rank, min(m, n))
        self.actual_rank = r

        self.register_buffer("weight", original_linear.weight.data.clone().to(dtype))
        self.bias = (nn.Parameter(original_linear.bias.data.clone().to(dtype))
                     if original_linear.bias is not None else None)
        self.register_buffer("smooth_scale", torch.ones(n, dtype=dtype))

        # Tucker factors (stored as target_dtype for inference efficiency)
        self.register_buffer("u2_q", torch.zeros(r, n, dtype=dtype))   # U2ᵀ: (r, n)
        self.register_buffer("g_q",  torch.zeros(r, r, dtype=dtype))   # G:   (r, r)
        self.register_buffer("u1_q", torch.zeros(m, r, dtype=dtype))   # U1:  (m, r)

        # LoRA correction
        # - kd_mode=False: buffers filled by SVD at calibrate() time
        # - kd_mode=True:  nn.Parameter (zero-init), trained by KD
        # - no_lora=True:  disabled entirely
        if not no_lora:
            if kd_mode:
                self.lora_a = nn.Parameter(torch.zeros(lora_rank, n, dtype=dtype))
                self.lora_b = nn.Parameter(torch.zeros(m, lora_rank, dtype=dtype))
            else:
                self.register_buffer("lora_a", torch.zeros(lora_rank, n, dtype=dtype))
                self.register_buffer("lora_b", torch.zeros(m, lora_rank, dtype=dtype))
        else:
            self.lora_a = None
            self.lora_b = None

        self.is_calibrated = False

    @torch.no_grad()
    def calibrate(self, x_max):
        """HOSVD decomposition + quantize factors + LoRA on quantized Tucker error."""
        x_max = x_max.clamp(min=1e-5).float()
        W = self.weight.float()  # (m, n)

        # SmoothQuant scale
        w_max  = W.abs().max(dim=0)[0].clamp(min=1e-5)
        smooth = (w_max.pow(1 - self.alpha) / x_max.pow(self.alpha)).clamp(1e-4, 1e4)
        self.smooth_scale.data = smooth.to(self.target_dtype)
        W_smooth = W / smooth.unsqueeze(0)  # (m, n)

        # HOSVD: full SVD, keep top-r components
        U, S, Vh = torch.linalg.svd(W_smooth, full_matrices=False)
        r   = min(self.actual_rank, S.shape[0])
        U1  = U[:, :r]    # (m, r)
        U2t = Vh[:r, :]   # (r, n) = U2ᵀ
        U2  = U2t.T       # (n, r)

        # Core tensor G = U1ᵀ @ W_smooth @ U2
        G = U1.T @ W_smooth @ U2  # (r, r)

        # --- Quantize U factors ---
        def _qu(t):
            if self.wgt_u_mode == "NVFP4":
                return quantize_to_nvfp4(t, self.block_size)
            elif self.wgt_u_mode == "BF16":
                return t.to(torch.bfloat16).to(self.target_dtype)
            elif self.wgt_u_mode in ("INT2", "INT3", "INT4", "INT8"):
                return quantize_uniform(t, self.block_size, mode=self.wgt_u_mode)
            else:  # FP16, FP32: cast only
                return t.to(self.target_dtype)

        # --- Quantize core G ---
        def _qcore(t):
            if self.wgt_core_mode == "NVFP4":
                return quantize_to_nvfp4(t, self.block_size)
            elif self.wgt_core_mode == "FP16":
                return t.to(self.target_dtype)  # no quantization
            return quantize_uniform(t, self.block_size, mode=self.wgt_core_mode)

        U1_q  = _qu(U1)
        G_q   = _qcore(G)
        U2t_q = _qu(U2t)

        self.u1_q.data  = U1_q.to(self.target_dtype)
        self.g_q.data   = G_q.to(self.target_dtype)
        self.u2_q.data  = U2t_q.to(self.target_dtype)

        # --- LoRA on total quantized Tucker error (SVD mode only) ---
        # kd_mode: lora_a/lora_b are nn.Parameter(zeros), trained later by KD
        if not self.no_lora and not self.kd_mode:
            Tucker_q_approx = U1_q.float() @ G_q.float() @ U2t_q.float()  # (m, n)
            total_error     = W_smooth - Tucker_q_approx

            lr = min(self.lora_rank, total_error.shape[0], total_error.shape[1])
            Ue, Se, Vhe = torch.linalg.svd(total_error.float(), full_matrices=False)
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
        """Return compression ratios based on actual bit-widths (data + scale overhead)."""
        bs = self.block_size
        # Effective bits/elem = (data_bits * bs + scale_bits) / bs
        u_bits_map    = {
            "NVFP4": 4,   "FP16": 16,  "BF16": 16,  "FP32": 32,
            "INT8":  (8*bs+16)/bs,   # INT8 + FP16 scale
            "INT4":  (4*bs+16)/bs,   # INT4 + FP16 scale
            "INT3":  (3*bs+16)/bs,   # INT3 + FP16 scale
            "INT2":  (2*bs+16)/bs,   # INT2 + FP16 scale
        }
        core_bits_map = {
            "NVFP4": 4,   "FP16": 16,
            "INT8":  (8*bs+16)/bs,
            "INT4":  (4*bs+16)/bs,
            "INT3":  (3*bs+16)/bs,
            "INT2":  (2*bs+16)/bs,
        }
        bits_u = u_bits_map.get(self.wgt_u_mode, 16)
        bits_c = core_bits_map.get(self.wgt_core_mode, 4)
        m, n, r, lr = self.m, self.n, self.actual_rank, self.lora_rank
        bits_u1   = m * r * bits_u
        bits_g    = r * r * bits_c
        bits_u2   = r * n * bits_u
        bits_lora = (lr * n + m * lr) * 16 if not self.no_lora else 0
        total_tucker_only = bits_u1 + bits_g + bits_u2
        total             = total_tucker_only + bits_lora
        fp16_bits  = m * n * 16
        nvfp4_bits = m * n * 4
        return {
            "total_tucker_only_bits": total_tucker_only,
            "total_bits": total,
            "compression_vs_fp16_tucker_only":  round(fp16_bits  / total_tucker_only, 2),
            "compression_vs_nvfp4_tucker_only": round(nvfp4_bits / total_tucker_only, 2),
            "compression_vs_fp16":  round(fp16_bits  / total, 2) if total > 0 else 0,
            "compression_vs_nvfp4": round(nvfp4_bits / total, 2) if total > 0 else 0,
            "pct_of_nvfp4": round(100.0 * total / nvfp4_bits, 2),
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

        # Tucker path: x_q → U2 → G → U1
        h1  = F.linear(x_q, self.u2_q)   # (batch, r)
        h2  = F.linear(h1,  self.g_q)    # (batch, r)
        out = F.linear(h2,  self.u1_q)   # (batch, m)

        # LoRA correction: x_smooth → lora_a → lora_b
        if not self.no_lora and self.lora_a is not None:
            la = self.lora_a.to(x_smooth.dtype)
            lb = self.lora_b.to(x_smooth.dtype)
            out = out + F.linear(F.linear(x_smooth, la), lb)

        if self.bias is not None:
            out = out + self.bias
        return out.to(input_dtype)


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Tucker-2 PTQ for PixArt-XL-2")
    parser.add_argument("--num_samples",   type=int,   default=20)
    parser.add_argument("--test_run",      action="store_true")
    parser.add_argument("--ref_dir",       type=str,   default="./ref_images")
    parser.add_argument("--save_dir",      type=str,   default="./results/tucker_experiment/TUCKER_R64")
    parser.add_argument("--model_path",    type=str,   default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--dataset_name",  type=str,   default="MJHQ", choices=["MJHQ", "sDCI"])
    parser.add_argument("--img_base_dir",  type=str,   default="/data/jameskimh/james_dit_pixart_xl_mjhq")
    parser.add_argument("--act_mode",      type=str,   default="NVFP4")
    parser.add_argument("--wgt_core_mode", type=str,   default="INT3",
                        choices=["FP16", "INT2", "INT3", "INT4", "INT8", "NVFP4"],
                        help="Core G precision: FP16(unquantized)/INT2/INT3/INT4/NVFP4")
    parser.add_argument("--wgt_u_mode",    type=str,   default="NVFP4",
                        choices=["NVFP4", "FP16", "BF16", "FP32", "INT2", "INT3", "INT4", "INT8"],
                        help="U1/U2 precision: NVFP4 (quantized) or FP16/BF16/FP32 (high precision)")
    parser.add_argument("--alpha",         type=float, default=0.5)
    parser.add_argument("--lowrank",       type=int,   default=64,
                        help="Tucker rank r (absolute). Ignored if --rank_ratio is set.")
    parser.add_argument("--rank_ratio",    type=float, default=None,
                        help="Tucker rank as fraction of min(m,n) per layer (e.g. 0.5/0.75/1.0)")
    parser.add_argument("--lora_rank",     type=int,   default=32,
                        help="LoRA rank for error correction")
    parser.add_argument("--no_lora",       action="store_true",
                        help="Disable LoRA correction (measure Tucker-only quality)")
    parser.add_argument("--kd_mode",       action="store_true",
                        help="KD mode: lora_a/b are trainable nn.Parameters (zero-init), skip SVD")
    parser.add_argument("--do_kd",         action="store_true",
                        help="Run KD fine-tuning after PTQ (requires --kd_mode)")
    parser.add_argument("--kd_steps",      type=int,   default=100)
    parser.add_argument("--kd_lr",         type=float, default=1e-4)
    parser.add_argument("--kd_prompts",    type=int,   default=8)
    parser.add_argument("--block_size",    type=int,   default=16)
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    s_target = 2 if args.test_run else args.num_samples
    dataset_ref_dir  = os.path.join(args.ref_dir,  args.dataset_name)
    _img_rel = args.save_dir
    if "/results/" in _img_rel:
        _img_rel = _img_rel.split("/results/", 1)[1]
    elif _img_rel.startswith("./results/"):
        _img_rel = _img_rel[len("./results/"):]
    elif _img_rel.startswith("results/"):
        _img_rel = _img_rel[len("results/"):]
    dataset_save_dir = os.path.join(args.img_base_dir, _img_rel)

    rank_desc = f"ratio={args.rank_ratio}" if args.rank_ratio is not None else f"abs={args.lowrank}"
    lora_desc = "no_lora" if args.no_lora else f"lora_r={args.lora_rank}"

    if accelerator.is_main_process:
        os.makedirs(dataset_ref_dir,  exist_ok=True)
        os.makedirs(dataset_save_dir, exist_ok=True)
        print(f"\n{'='*65}")
        print(f"  Tucker-2 Quantization Experiment")
        print(f"  Core={args.wgt_core_mode}  U={args.wgt_u_mode}  rank={rank_desc}  {lora_desc}")
        print(f"  Act={args.act_mode}  block_size={args.block_size}  alpha={args.alpha}")
        print(f"  Samples={s_target}  Dataset={args.dataset_name}")
        print(f"  Save → {dataset_save_dir}")
        print(f"{'='*65}\n")

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

    # Calibration: collect per-channel activation max
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
            orig_m,
            act_mode      = args.act_mode,
            wgt_core_mode = args.wgt_core_mode,
            wgt_u_mode    = args.wgt_u_mode,
            alpha         = args.alpha,
            rank          = args.lowrank,
            rank_ratio    = args.rank_ratio,
            lora_rank     = args.lora_rank,
            no_lora       = args.no_lora,
            kd_mode       = args.kd_mode,
            block_size    = args.block_size,
            dtype         = torch.float16,
        ).to(device)
        if name in all_samples:
            new_layer.calibrate(all_samples[name])
            if accelerator.is_main_process and len(comp_stats_list) == 0:
                comp_stats_list.append(new_layer.compression_stats())
                accelerator.print(
                    f"  [First layer] rank={new_layer.actual_rank}  "
                    f"U={args.wgt_u_mode}  G={args.wgt_core_mode}  "
                    f"compress_vs_nvfp4={comp_stats_list[0]['compression_vs_nvfp4_tucker_only']}x (tucker-only)"
                )
        set_module_by_name(transformer, name, new_layer)
    accelerator.wait_for_everyone()

    t_ptq_end = time.time()
    ptq_time = t_ptq_end - t_ptq_start
    accelerator.print(f"[Phase 2] Tucker-2 PTQ done: {ptq_time/60:.1f}m ({ptq_time:.0f}s)")

    # ----------------------------------------------------------
    # Phase 2.5: KD fine-tuning (optional)
    # ----------------------------------------------------------
    kd_time = 0.0
    kd_loss_log = []
    if args.do_kd and args.kd_mode:
        from pixart_kd_finetune import run_kd_finetuning
        accelerator.print("[Phase 2.5] KD fine-tuning...")
        # Load frozen FP16 teacher
        teacher_transformer = PixArtAlphaPipeline.from_pretrained(
            args.model_path, torch_dtype=torch.float16
        ).to(device).transformer
        teacher_transformer.eval()
        teacher_transformer.requires_grad_(False)

        t_kd_start = time.time()
        kd_loss_log = run_kd_finetuning(pipe, teacher_transformer, prompts[:args.kd_prompts], args, accelerator)
        kd_time = time.time() - t_kd_start
        del teacher_transformer
        torch.cuda.empty_cache()
        accelerator.print(f"[Phase 2.5] KD done: {kd_time/60:.1f}m")

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
                                    return_tensors="pt", padding=True,
                                    truncation=True, max_length=77).to(device)
            clip_scores.append(float(clip_model(**inputs).logits_per_image.item()))

        # Compression stats for representative layer (4608×1152)
        u_bits_map    = {"NVFP4": 4, "FP16": 16, "BF16": 16, "FP32": 32}
        core_bits_map = {"NVFP4": 4, "INT4": 4, "INT3": 3, "INT2": 2, "INT8": 8}
        bits_u = u_bits_map.get(args.wgt_u_mode, 16)
        bits_c = core_bits_map.get(args.wgt_core_mode, 3)
        m_rep, n_rep = 4608, 1152
        if args.rank_ratio is not None:
            r_rep = max(1, int(args.rank_ratio * min(m_rep, n_rep)))
        else:
            r_rep = min(args.lowrank, min(m_rep, n_rep))
        lr = args.lora_rank
        bits_u1   = m_rep * r_rep * bits_u
        bits_g    = r_rep * r_rep * bits_c
        bits_u2   = r_rep * n_rep * bits_u
        bits_lora = (lr * n_rep + m_rep * lr) * 16 if not args.no_lora else 0
        total_tucker_only = bits_u1 + bits_g + bits_u2
        total_with_lora   = total_tucker_only + bits_lora
        fp16_bits  = m_rep * n_rep * 16
        nvfp4_bits = m_rep * n_rep * 4

        comp_info = {
            "u_mode":   args.wgt_u_mode,
            "g_mode":   args.wgt_core_mode,
            "lora_mode": "none" if args.no_lora else "FP16",
            "tucker_rank_ratio": args.rank_ratio,
            "tucker_rank_abs":   args.lowrank,
            "tucker_rank_rep":   r_rep,
            "lora_rank":   lr,
            "representative_layer": f"{m_rep}×{n_rep}",
            "compression_vs_fp16_tucker_only":  round(fp16_bits  / total_tucker_only, 2),
            "compression_vs_nvfp4_tucker_only": round(nvfp4_bits / total_tucker_only, 2),
            "compression_vs_fp16":  round(fp16_bits  / total_with_lora, 2) if total_with_lora > 0 else 0,
            "compression_vs_nvfp4": round(nvfp4_bits / total_with_lora, 2) if total_with_lora > 0 else 0,
            "pct_of_nvfp4": round(100.0 * total_with_lora / nvfp4_bits, 2),
        }

        def fmt_time(s):
            m_, sec = divmod(int(s), 60)
            return f"{m_}m{sec:02d}s"

        result = {
            "config": {
                "method":        "Tucker-2_HOSVD+LoRA",
                "act_mode":      args.act_mode,
                "wgt_u_mode":    args.wgt_u_mode,
                "wgt_core_mode": args.wgt_core_mode,
                "rank_ratio":    args.rank_ratio,
                "lowrank":       args.lowrank,
                "lora_rank":     args.lora_rank,
                "no_lora":       args.no_lora,
                "block_size":    args.block_size,
                "alpha":         args.alpha,
                "num_samples":   s_count,
                "dataset":       args.dataset_name,
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
                "kd":    round(kd_time,    1),
                "eval":  round(eval_time,  1),
                "total": round(total_time, 1),
            },
            "kd_info": {
                "do_kd":    args.do_kd,
                "kd_steps": args.kd_steps if args.do_kd else 0,
                "kd_loss_init":  round(kd_loss_log[0],  5) if kd_loss_log else None,
                "kd_loss_final": round(kd_loss_log[-1], 5) if kd_loss_log else None,
            },
        }

        metrics_path = os.path.join(dataset_save_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(result, f, indent=4)

        print(f"\n{'='*65}")
        print(f"  Tucker-2 Results  →  {metrics_path}")
        print(f"  Core={args.wgt_core_mode}  U={args.wgt_u_mode}  rank_ratio={args.rank_ratio}  {lora_desc}")
        print(f"  [PRIMARY]   FID: {res_fid:.4f}  |  IS: {res_is:.4f}")
        print(f"  [SECONDARY] PSNR: {res_psnr:.2f}  SSIM: {res_ssim:.4f}  LPIPS: {res_lpips:.4f}")
        print(f"  [COMPRESS]  Tucker-only: {comp_info['compression_vs_nvfp4_tucker_only']}x vs NVFP4")
        print(f"              With LoRA:   {comp_info['compression_vs_nvfp4']}x vs NVFP4  "
              f"({comp_info['pct_of_nvfp4']:.1f}% of NVFP4)")
        print(f"  [TIMING]    PTQ: {fmt_time(ptq_time)}  Eval: {fmt_time(eval_time)}  "
              f"Total: {fmt_time(total_time)}")
        print(f"{'='*65}\n")

    accelerator.wait_for_everyone()
    del pipe
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
