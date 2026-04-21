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


def quantize_uniform(x, block_size=128, mode="INT4"):
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


def get_module_by_name(model, name):
    for part in name.split("."):
        model = getattr(model, part)
    return model


def set_module_by_name(model, name, new_module):
    parts = name.split(".")
    parent = get_module_by_name(model, ".".join(parts[:-1])) if len(parts) > 1 else model
    setattr(parent, parts[-1], new_module)


# ==========================================
# 1. W3A4Linear 클래스
# ==========================================

class W3A4Linear(nn.Module):
    """
    Weight 3-bit + Activation 4-bit 양자화 레이어.

    파이프라인 (calibrate 시점):
      1. Block Hadamard rotation (offline): W_rot = block_hadamard(W)
         - PixArt hidden_size=1152=9×128, MLP dim=4608=36×128 → block_size=128 완벽 적합
      2. SmoothQuant: smooth_scale = w_max^(1-α) / x_max^α (per input channel)
         - W_smooth = W_rot / smooth_scale
      3. GPTQ-light INT3 with act_order:
         - act_order: H_diag(=E[x²]) 기준 내림차순 정렬
         - 각 group(=block_size 컬럼)을 순서대로 INT3 양자화 + diagonal H 기반 잔여 컬럼 보정
      4. SVD error correction (rank=lowrank):
         - W_error = W_smooth - W_q_int3 → SVD → lora_a, lora_b

    forward:
      x_rot = block_hadamard(x)   (online, per block_size 채널 묶음)
      x_smooth = x_rot * smooth_scale
      x_q_int4 = quantize_uniform(x_smooth, mode="INT4")
      out = x_q_int4 @ W_q_int3.T + (x_smooth @ lora_a.T) @ lora_b.T
    """

    def __init__(self, original_linear, rank=32, block_size=128, alpha=0.5, dtype=torch.float16, wgt_bits="INT4", act_mode="INT4"):
        super().__init__()
        self.rank = rank
        self.block_size = block_size
        self.alpha = alpha
        self.wgt_bits = wgt_bits
        self.act_mode = act_mode
        self.target_dtype = dtype
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.use_rotation = (self.in_features % block_size == 0)

        self.register_buffer("weight", original_linear.weight.data.clone().to(dtype))
        self.bias = nn.Parameter(
            original_linear.bias.data.clone().to(dtype)
        ) if original_linear.bias is not None else None

        # Block Hadamard matrix (one block, reused for all blocks)
        if self.use_rotation:
            from scipy.linalg import hadamard as scipy_hadamard
            H_np = scipy_hadamard(block_size) / (block_size ** 0.5)
            self.register_buffer("H_block", torch.from_numpy(H_np).to(dtype))
        else:
            self.H_block = None

        self.register_buffer("smooth_scale", torch.ones(self.in_features, dtype=dtype))
        self.register_buffer("w_quantized", original_linear.weight.data.clone().to(dtype))
        self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank, dtype=dtype))
        self.is_calibrated = False

    def _block_hadamard(self, x):
        """Block Hadamard transform: 마지막 차원의 block_size 채널씩 H_block 적용"""
        shape = x.shape
        in_f = shape[-1]
        n_blocks = in_f // self.block_size
        x_3d = x.reshape(*shape[:-1], n_blocks, self.block_size)
        x_rot = x_3d @ self.H_block.to(x.dtype)
        return x_rot.reshape(shape)

    @torch.no_grad()
    def calibrate(self, x_max, H_diag):
        """
        x_max : (in_features,) 텐서 — 캘리브레이션 중 채널별 activation max
        H_diag: (in_features,) 텐서 — E[x_i²] (GPTQ Hessian 대각선 근사)
        """
        device = self.weight.device
        x_max  = x_max.float().to(device).clamp(min=1e-5)
        H_diag = H_diag.float().to(device).clamp(min=1e-8)
        W = self.weight.float()  # (out_f, in_f)
        in_f = self.in_features

        # ---- Step 1: Offline Hadamard rotation ----
        if self.use_rotation:
            W_rot = self._block_hadamard(W)  # (out_f, in_f)
            # 회전된 채널별 H_diag, x_max: 블록 내 평균 (Hadamard 후 채널 간 등분산)
            H_rot  = H_diag.clone()
            xm_rot = x_max.clone()
            n_blocks = in_f // self.block_size
            for b in range(n_blocks):
                s, e = b * self.block_size, (b + 1) * self.block_size
                H_rot[s:e]  = H_diag[s:e].mean()
                xm_rot[s:e] = x_max[s:e].mean()
        else:
            W_rot  = W
            H_rot  = H_diag
            xm_rot = x_max

        # ---- Step 2: SmoothQuant scale ----
        w_max = W_rot.abs().max(dim=0)[0].clamp(min=1e-5)
        smooth = (w_max.pow(1.0 - self.alpha) / xm_rot.pow(self.alpha)).clamp(1e-4, 1e4)
        self.smooth_scale.copy_(smooth.to(self.target_dtype))
        W_smooth = W_rot / smooth.unsqueeze(0)  # (out_f, in_f)

        # ---- Step 3: act_order + per-group INT3 ----
        # act_order: H_diag 내림차순 정렬 → 중요한 채널끼리 같은 그룹으로 묶어 scale 낭비 최소화
        # 보상 루프 없음 — 그룹 128개 오차 합산 보상은 h_g가 작을 때 수치 폭발 유발
        # (SVD rank=32 branch가 체계적 양자화 오차를 보정)
        order     = torch.argsort(H_rot, descending=True)
        inv_order = torch.argsort(order)
        W_reordered   = W_smooth[:, order]              # (out_f, in_f)
        W_q_reordered = torch.zeros_like(W_reordered)

        gs = self.block_size  # group_size = block_size
        for g_start in range(0, in_f, gs):
            g_end = min(g_start + gs, in_f)
            group_w = W_reordered[:, g_start:g_end]    # (out_f, gs)
            # wgt_bits — per-output-channel scale
            if self.wgt_bits == "NVFP4":
                W_q_reordered[:, g_start:g_end] = quantize_to_nvfp4(
                    group_w, block_size=g_end - g_start
                )
            else:
                W_q_reordered[:, g_start:g_end] = quantize_uniform(
                    group_w, block_size=g_end - g_start, mode=self.wgt_bits
                )

        W_q = W_q_reordered[:, inv_order]  # 원래 채널 순서로 복원
        self.w_quantized.copy_(W_q.to(self.target_dtype))

        # ---- Step 4: SVD error correction ----
        W_err = W_smooth - W_q  # (out_f, in_f)
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

    def forward(self, x):
        input_dtype = x.dtype
        if not self.is_calibrated:
            bias = self.bias.to(input_dtype) if self.bias is not None else None
            return F.linear(x, self.weight.to(input_dtype), bias)

        x_t = x.to(self.target_dtype)

        # Online block Hadamard rotation
        if self.use_rotation:
            x_rot = self._block_hadamard(x_t)
        else:
            x_rot = x_t

        # SmoothQuant: activation × smooth_scale (broadcasting over last dim)
        x_smooth = x_rot * self.smooth_scale

        # Activation quantization (act_mode: INT4 or NVFP4)
        if self.act_mode == "NVFP4":
            x_q = quantize_to_nvfp4(x_smooth, block_size=16)
        else:
            x_q = quantize_uniform(x_smooth, block_size=16, mode="INT4")

        # base(INT3 weight × INT4 act) + SVD 보정(FP16)
        out = F.linear(x_q, self.w_quantized) + F.linear(F.linear(x_smooth, self.lora_a), self.lora_b)

        if self.bias is not None:
            out = out + self.bias
        return out.to(input_dtype)


# ==========================================
# 2. 메인
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="W3A4 vs FP16 / NVFP4_DEFAULT_CFG Quantization Experiment")
    parser.add_argument("--num_samples",   type=int,   default=20)
    parser.add_argument("--test_run",      action="store_true", help="2샘플로 파이프라인 통과 확인")
    parser.add_argument("--ref_dir",       type=str,   default="/data/jameskimh/james_dit_ref/ref_images_fp16")
    parser.add_argument("--save_dir",      type=str,   default="./results/w3a4_experiment/W3A4")
    parser.add_argument("--model_path",    type=str,   default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--dataset_name",  type=str,   default="MJHQ", choices=["MJHQ", "sDCI"])
    parser.add_argument("--quant_method",  type=str,   default="W3A4",
                        choices=["FP16", "BASELINE", "W3A4"],
                        help="FP16: no quant | BASELINE: NVFP4_DEFAULT_CFG | W3A4: Rotation+SmoothQuant+SVD+A4")
    parser.add_argument("--wgt_bits",      type=str,   default="NVFP4",
                        choices=["INT3", "INT4", "NVFP4"],
                        help="INT3: 3-bit / INT4: uniform 4-bit / NVFP4: non-uniform 4-bit (default)")
    parser.add_argument("--act_mode",      type=str,   default="INT4",
                        choices=["INT4", "NVFP4"],
                        help="INT4: uniform 4-bit act (default) / NVFP4: non-uniform 4-bit act")
    # W3A4 하이퍼파라미터
    parser.add_argument("--lowrank",       type=int,   default=32,  help="SVD low-rank 차원")
    parser.add_argument("--block_size",    type=int,   default=128, help="Hadamard block 크기 = GPTQ group size")
    parser.add_argument("--alpha",         type=float, default=0.5, help="SmoothQuant alpha")
    # BASELINE용
    parser.add_argument("--baseline_lowrank", type=int, default=32, help="BASELINE mtq lowrank")
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

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

    if accelerator.is_main_process:
        accelerator.print(f"\n{'='*60}")
        accelerator.print(f"  Method    : {args.quant_method}")
        if args.quant_method == "W3A4":
            accelerator.print(f"  Rank      : {args.lowrank}  |  BlockSize: {args.block_size}  |  alpha: {args.alpha}")
        accelerator.print(f"  Samples   : {s_count}  |  Dataset: {args.dataset_name}")
        accelerator.print(f"  Save dir  : {dataset_save_dir}")
        accelerator.print(f"{'='*60}\n")

    # ------------------------------------------
    # Phase 1: Reference FP16 이미지 생성 (존재하면 skip)
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
    # Phase 2: 모델 준비 (양자화 적용)
    # ------------------------------------------
    accelerator.print(f"[Phase 2] Loading model ({args.quant_method})...")
    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    transformer = pipe.transformer

    skip_keywords = ["x_embedder", "t_embedder", "proj_out"]

    if args.quant_method == "FP16":
        # 양자화 없이 FP16 그대로
        accelerator.print("  [FP16] No quantization applied.")

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
        transformer = pipe.transformer

    elif args.quant_method == "W3A4":
        accelerator.print("  [W3A4] Rotation + GPTQ-INT3 + SVD + INT4 Act")
        target_dtype = torch.float16

        target_names = [
            n for n, m in transformer.named_modules()
            if isinstance(m, nn.Linear) and not any(kw in n for kw in skip_keywords)
        ]
        if accelerator.is_main_process:
            print(f"  Targeted {len(target_names)} Linear layers for W3A4 quantization.")

        # ---- Calibration: x_max + H_diag(=E[x²]) 수집 ----
        calib_data = {}

        def hook_fn(name):
            def forward_hook(m, inputs, output):
                x = inputs[0].detach().view(-1, inputs[0].shape[-1]).float()
                step_xmax  = x.abs().max(dim=0)[0].cpu()
                step_hdiag = x.pow(2).mean(dim=0).cpu()
                if name not in calib_data:
                    calib_data[name] = {"x_max": [], "h_diag": []}
                calib_data[name]["x_max"].append(step_xmax)
                calib_data[name]["h_diag"].append(step_hdiag)
            return forward_hook

        hooks = [
            get_module_by_name(transformer, n).register_forward_hook(hook_fn(n))
            for n in target_names
        ]
        print(f"  Calibrating with {p_count} prompts...", flush=True)
        with accelerator.split_between_processes(prompts[:p_count]) as local_prompts:
            for prompt in local_prompts:
                pipe(prompt, num_inference_steps=t_count,
                     generator=torch.Generator(device=device).manual_seed(42))
        for h in hooks:
            h.remove()

        # 분산 reduce (mean)
        for name in calib_data:
            x_max_mean  = torch.stack(calib_data[name]["x_max"]).mean(dim=0).to(device)
            h_diag_mean = torch.stack(calib_data[name]["h_diag"]).mean(dim=0).to(device)
            calib_data[name] = {
                "x_max":  accelerator.reduce(x_max_mean,  reduction="mean"),
                "h_diag": accelerator.reduce(h_diag_mean, reduction="mean"),
            }
        accelerator.wait_for_everyone()

        # ---- Layer replacement ----
        for name in tqdm(target_names, desc="Replacing layers",
                         disable=not accelerator.is_main_process):
            orig_m = get_module_by_name(transformer, name)
            if next(orig_m.parameters()).device == device:
                new_layer = W3A4Linear(
                    orig_m,
                    rank=args.lowrank,
                    block_size=args.block_size,
                    alpha=args.alpha,
                    dtype=target_dtype,
                    wgt_bits=args.wgt_bits,
                    act_mode=args.act_mode,
                ).to(device)
                if name in calib_data:
                    new_layer.calibrate(
                        calib_data[name]["x_max"],
                        calib_data[name]["h_diag"],
                    )
                set_module_by_name(transformer, name, new_layer)

        del calib_data
        accelerator.wait_for_everyone()

    # ------------------------------------------
    # Phase 3: 이미지 생성 및 메트릭 업데이트
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
    # Phase 4: 메트릭 집계 및 저장
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
            "quant_method":   args.quant_method,
            "wgt_bits":       args.wgt_bits        if args.quant_method == "W3A4" else "N/A",
            "lowrank":        args.lowrank         if args.quant_method == "W3A4" else "N/A",
            "block_size":     args.block_size      if args.quant_method == "W3A4" else "N/A",
            "alpha":          args.alpha           if args.quant_method == "W3A4" else "N/A",
            "baseline_lowrank": args.baseline_lowrank if args.quant_method == "BASELINE" else "N/A",
            "num_samples":    s_count,
            "dataset":        args.dataset_name,
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
