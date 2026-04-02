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


def quantize_uniform(x, block_size=16, mode="INT4"):
    orig_shape = x.shape
    x_flat = x.view(-1, block_size)
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
    return (x_q_flat * scale).view(orig_shape)


def quantize_to_nvfp4(x, block_size=16):
    orig_shape = x.shape
    x_flat = x.view(-1, block_size)
    nvfp4_levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device, dtype=x.dtype
    )
    amax = x_flat.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale = amax / 6.0
    x_norm = x_flat.abs() / scale
    distances = torch.abs(x_norm.unsqueeze(-1) - nvfp4_levels)
    closest_idx = torch.argmin(distances, dim=-1)
    x_q = torch.sign(x_flat) * nvfp4_levels[closest_idx] * scale
    return x_q.view(orig_shape)


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
    """SmoothQuant + SVD 오차 보정 레이어 (비교용으로 포함)"""

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
        out = F.linear(x_q, self.w_quantized) + F.linear(F.linear(x_smoothed, self.lora_a), self.lora_b)
        if self.bias is not None:
            out += self.bias
        return out.to(input_dtype)


class ManualRPCALinear(nn.Module):
    """RPCA 분해: Sparse(이상치) + Dense(SVD+SmoothQuant) 세 브랜치 구조"""

    def __init__(self, original_linear, act_mode, wgt_mode, alpha=0.5, rank=32, block_size=16, outlier_ratio=0.01, dtype=torch.float32):
        super().__init__()
        self.target_dtype = dtype
        self.act_mode = act_mode
        self.wgt_mode = wgt_mode
        self.alpha = alpha
        self.rank = rank
        self.block_size = block_size
        self.outlier_ratio = outlier_ratio
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

        # Step 1: Sparse / Dense 분리
        w_abs = self.weight.abs()
        threshold = torch.quantile(w_abs.view(-1).float(), 1.0 - self.outlier_ratio)
        sparse_mask = w_abs >= threshold
        self.w_sparse.copy_((self.weight * sparse_mask).to(self.target_dtype))
        w_dense = self.weight * (~sparse_mask)

        # Step 2: Outlier 제거 후 안정화된 Smooth Scale
        w_max = w_dense.abs().max(dim=0)[0].clamp(min=1e-5).float()
        self.smooth_scale.data = (w_max.pow(1 - self.alpha) / x_max.pow(self.alpha)).to(self.target_dtype)
        self.smooth_scale.data = self.smooth_scale.data.clamp(min=1e-4, max=1e4)

        # Step 3: Dense 행렬 Smoothing + 양자화
        w_smoothed = w_dense.float() / self.smooth_scale.float().view(1, -1)
        w_q = quantize_to_nvfp4(w_smoothed, self.block_size) if self.wgt_mode == "NVFP4" else quantize_uniform(w_smoothed, self.block_size, mode=self.wgt_mode)
        self.w_quantized.copy_(w_q.to(self.target_dtype))

        # Step 4: 양자화 오차 SVD
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
        # base(양자화된 Dense) + svd(오차 보정) + sparse(이상치 고정밀도)
        out = F.linear(x_q, self.w_quantized) + F.linear(F.linear(x_smoothed, self.lora_a), self.lora_b) + F.linear(x.to(self.target_dtype), self.w_sparse)
        if self.bias is not None:
            out += self.bias
        return out.to(input_dtype)


# ==========================================
# 2. 메인
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="RPCA vs NVFP4_DEFAULT_CFG Quantization Experiment")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--test_run", action="store_true", help="2샘플로 파이프라인 통과 확인")
    parser.add_argument("--ref_dir", type=str, default="./ref_images")
    parser.add_argument("--save_dir", type=str, default="./results/rpca_sweep/RPCA_ANVFP4_WNVFP4_OR0.01")
    parser.add_argument("--model_path", type=str, default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--dataset_name", type=str, default="MJHQ", choices=["MJHQ", "sDCI"])
    # 양자화 방법
    parser.add_argument("--quant_method", type=str, default="RPCA", choices=["BASELINE", "RPCA"],
                        help="BASELINE: mtq.NVFP4_SVDQUANT_DEFAULT_CFG / RPCA: ManualRPCALinear")
    parser.add_argument("--outlier_ratio", type=float, default=0.01,
                        help="RPCA: 가중치 상위 outlier_ratio 비율을 Sparse branch로 분리 (0.01=1%%)")
    # RPCA 하이퍼파라미터
    quant_modes = ["NVFP4", "INT8", "INT4", "INT3", "INT2", "TERNARY"]
    parser.add_argument("--act_mode", type=str, default="NVFP4", choices=quant_modes)
    parser.add_argument("--wgt_mode", type=str, default="NVFP4", choices=quant_modes)
    parser.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant alpha")
    parser.add_argument("--lowrank", type=int, default=32, help="SVD low-rank 차원")
    parser.add_argument("--block_size", type=int, default=16, help="블록 양자화 크기")
    parser.add_argument("--numeric_dtype", type=str, default="half", choices=["half", "float"])
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    s_target = 2 if args.test_run else args.num_samples
    dataset_ref_dir = os.path.join(args.ref_dir, args.dataset_name)
    dataset_save_dir = os.path.join(args.save_dir, args.dataset_name)

    if accelerator.is_main_process:
        os.makedirs(dataset_ref_dir, exist_ok=True)
        os.makedirs(dataset_save_dir, exist_ok=True)

    prompts = get_prompts(s_target, args)
    s_count = len(prompts)
    p_count = 2 if args.test_run else min(64, s_count)
    t_count = 20

    if accelerator.is_main_process:
        accelerator.print(f"\n{'='*60}")
        accelerator.print(f"  Method : {args.quant_method}")
        if args.quant_method == "RPCA":
            accelerator.print(f"  Act    : {args.act_mode}  |  Wgt: {args.wgt_mode}")
            accelerator.print(f"  Outlier ratio: {args.outlier_ratio}  |  Rank: {args.lowrank}")
        accelerator.print(f"  Samples: {s_count}  |  Dataset: {args.dataset_name}")
        accelerator.print(f"  Save   : {dataset_save_dir}")
        accelerator.print(f"{'='*60}\n")

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
    accelerator.print(f"[Phase 2] Loading and quantizing model ({args.quant_method})...")
    pipe = PixArtAlphaPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    transformer = pipe.transformer

    if args.quant_method == "BASELINE":
        # mtq 공식 NVFP4_SVDQUANT_DEFAULT_CFG 그대로 적용
        quant_config = copy.deepcopy(mtq.NVFP4_SVDQUANT_DEFAULT_CFG)
        quant_config["algorithm"]["lowrank"] = args.lowrank

        def forward_loop(model):
            for prompt in prompts[:p_count]:
                pipe(prompt, num_inference_steps=5, generator=torch.Generator(device=device).manual_seed(42))

        with torch.no_grad():
            pipe.transformer = mtq.quantize(pipe.transformer, quant_config, forward_loop=forward_loop)
        transformer = pipe.transformer

    elif args.quant_method == "RPCA":
        target_dtype = torch.float16 if args.numeric_dtype == "half" else torch.float32

        skip_keywords = ["x_embedder", "t_embedder", "proj_out"]
        target_names = [
            n for n, m in transformer.named_modules()
            if isinstance(m, nn.Linear) and not any(kw in n for kw in skip_keywords)
        ]
        if accelerator.is_main_process:
            print(f"  Targeted {len(target_names)} Linear layers for RPCA quantization.")

        # Calibration: 채널별 activation max 수집
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
        for name in tqdm(target_names, desc="Replacing layers", disable=not accelerator.is_main_process):
            orig_m = get_module_by_name(transformer, name)
            if next(orig_m.parameters()).device == device:
                new_layer = ManualRPCALinear(
                    orig_m, args.act_mode, args.wgt_mode,
                    args.alpha, args.lowrank, args.block_size,
                    args.outlier_ratio, target_dtype
                ).to(device)
                if name in all_samples:
                    new_layer.manual_calibrate_and_rpca(all_samples[name])
                set_module_by_name(transformer, name, new_layer)
        accelerator.wait_for_everyone()

    # ------------------------------------------
    # Phase 3: 이미지 생성 및 메트릭 업데이트
    # ------------------------------------------
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
            "act_mode": args.act_mode if args.quant_method == "RPCA" else "N/A",
            "wgt_mode": args.wgt_mode if args.quant_method == "RPCA" else "N/A",
            "outlier_ratio": args.outlier_ratio if args.quant_method == "RPCA" else "N/A",
            "lowrank": args.lowrank,
            "block_size": args.block_size if args.quant_method == "RPCA" else "N/A",
            "alpha": args.alpha if args.quant_method == "RPCA" else "N/A",
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
        }

        metrics_path = os.path.join(dataset_save_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(final_res, f, indent=4)

        print(f"\n{'='*60}")
        print(f"  Results saved: {metrics_path}")
        print(f"  [PRIMARY]   FID: {res_fid:.4f}  |  IS: {res_is:.4f}")
        print(f"  [SECONDARY] PSNR: {res_psnr:.2f}  SSIM: {res_ssim:.4f}  LPIPS: {res_lpips:.4f}  CLIP: {np.mean(clip_scores):.2f}")
        print(f"{'='*60}\n")

    accelerator.wait_for_everyone()
    del pipe
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
