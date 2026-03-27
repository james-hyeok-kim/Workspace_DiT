import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import numpy as np
from PIL import Image
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm
from accelerate import Accelerator
from torchvision.transforms import ToTensor

# 평가 지표 라이브러리
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from transformers import CLIPModel, CLIPProcessor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance # 🎯 더 안정적인 FID로 변경

from datasets import load_dataset

# ==========================================
# 0. MJHQ Data set
# ==========================================

def get_prompts(num_samples, args):
    # 경로 및 키 결정
    if args.dataset_name == "MJHQ":
        path, split, key = "xingjianleng/mjhq30k", "test", "text"
    else:
        path, split, key = "mit-han-lab/svdquant-datasets", "train", "prompt"

    try:
        # streaming 모드로 로드 및 셔플
        dataset = load_dataset(path, split=split, streaming=True)
        prompts = []
        for i, entry in enumerate(dataset):
            if i >= num_samples:
                break
            
            # MJHQ('text') 또는 SVDQuant('prompt') 키를 사용하여 프롬프트 추출
            prompt = entry[key]
            prompts.append(prompt)
            
        return prompts

    except Exception as e:
        print(f"❌ Error while fetching prompts: {e}")
        return ["A professional high-quality photo"] * num_samples


# ==========================================
# 1. 핵심 양자화 함수 및 클래스
# ==========================================

def manual_quant_act(x, block_size=16, mode="NVFP4"):
    curr_dtype = x.dtype
    orig_shape = x.shape
    x_flat = x.to(torch.float32).view(-1, block_size)
    
    q_max = 6.0 if mode == "NVFP4" else 448.0 if mode == "FP8" else 7.0
    raw_scale = x_flat.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12) / q_max
    scale_clamped = torch.clamp(raw_scale, max=448.0) 
    x_q = torch.clamp(torch.round(x_flat / scale_clamped), -q_max, q_max) * scale_clamped
    return x_q.view(orig_shape).to(curr_dtype)

class ManualSVDLinear(torch.nn.Module):
    def __init__(self, original_linear, act_mode, weight_mode, alpha=0.8, rank=64, block_size=128, dtype=torch.float32):
        super().__init__()
        self.target_dtype = dtype
        self.act_mode = act_mode
        self.weight_mode = weight_mode
        self.alpha = alpha
        self.rank = rank
        self.block_size = block_size
        
        # 🎯 [Noise Fix] 초기 가중치를 0이 아닌 원본으로 설정
        self.register_buffer('weight', original_linear.weight.data.clone().to(dtype))
        self.register_buffer('w_ternary', original_linear.weight.data.clone().to(dtype))
        self.register_buffer('smooth_scale', torch.ones(self.weight.shape[1]).to(dtype))
        
        self.bias = nn.Parameter(original_linear.bias.data.clone().to(dtype)) if original_linear.bias is not None else None
        self.lora_a = nn.Parameter(torch.zeros(rank, self.weight.shape[1]).to(dtype))
        self.lora_b = nn.Parameter(torch.zeros(self.weight.shape[0], rank).to(dtype))
        self.is_calibrated = False

    @torch.no_grad()
    def manual_calibrate_and_svd(self, x_samples):
        x_flat = x_samples.view(-1, x_samples.shape[-1]).float()
        x_max = x_flat.abs().max(dim=0)[0].clamp(min=1e-12)
        w_max = self.weight.abs().max(dim=0)[0].clamp(min=1e-12)

        self.smooth_scale.data = ((x_max ** self.alpha) / (w_max ** (1 - self.alpha)) / 
                                  torch.sqrt(x_max / w_max)).to(self.target_dtype)
        
        w_smoothed = self.weight.float() / self.smooth_scale.float().view(1, -1)
        w_reshaped = w_smoothed.view(w_smoothed.shape[0], -1, self.block_size)
        t_scale = w_reshaped.abs().amax(dim=2, keepdim=True).clamp(min=1e-12)

        q_max = 6.0 if self.weight_mode in ["NVFP4", "MXFP4"] else 1.0
        w_quant_reshaped = torch.clamp(torch.round(w_reshaped / t_scale * q_max), -q_max, q_max)
        w_quant = (w_quant_reshaped * t_scale / q_max).view(w_smoothed.shape)
        
        self.w_ternary.copy_(w_quant.to(self.target_dtype))
        w_error = w_smoothed - w_quant
        U, S, Vh = torch.linalg.svd(w_error.float(), full_matrices=False)
        sqrt_S = torch.sqrt(S[:self.rank])
        self.lora_b.data = (U[:, :self.rank] * sqrt_S.unsqueeze(0)).to(self.target_dtype)
        self.lora_a.data = (Vh[:self.rank, :] * sqrt_S.unsqueeze(1)).to(self.target_dtype)
        self.is_calibrated = True

    def forward(self, x):
        input_dtype = x.dtype
        if not self.is_calibrated: return F.linear(x, self.weight.to(input_dtype), self.bias.to(input_dtype) if self.bias is not None else None)
        curr_x = x.to(self.target_dtype) * self.smooth_scale
        x_q = manual_quant_act(curr_x, self.block_size, mode=self.act_mode)
        base_out = F.linear(x_q, self.w_ternary)
        svd_out = F.linear(F.linear(x_q, self.lora_a), self.lora_b)
        return (base_out + svd_out + (self.bias if self.bias is not None else 0)).to(input_dtype)

# ==========================================
# 2. 메인 실행 로직 (Multi-GPU Sync 강화)
# ==========================================

def get_module_by_name(model, name):
    for part in name.split('.'): model = getattr(model, part)
    return model

def set_module_by_name(model, name, new_module):
    parts = name.split('.')
    parent = get_module_by_name(model, ".".join(parts[:-1])) if len(parts) > 1 else model
    setattr(parent, parts[-1], new_module)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--test_run", action="store_true")
    parser.add_argument("--ref_dir", type=str, default="./ref_images")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--model_path", type=str, default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--lowrank", type=int, default=32)
    parser.add_argument("--act_mode", type=str, default="NVFP4")
    parser.add_argument("--wgt_mode", type=str, default="TERNARY")
    parser.add_argument("--numeric_dtype", type=str, default="half")
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--dataset_name", type=str, default="MJHQ", choices=["MJHQ", "sDCI"])
    args = parser.parse_args()

# 🎯 [Step 0] 카운트 및 프롬프트 설정
    # s_count: 최종 평가할 이미지 수, p_count: 캘리브레이션에 쓸 데이터 수
    s_target = 2 if args.test_run else args.num_samples
    
    accelerator = Accelerator()
    device = accelerator.device
    os.makedirs(args.ref_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # 데이터셋에서 프롬프트 로드
    prompts = get_prompts(s_target, args)
    actual_count = len(prompts)

    if actual_count < s_target:
        print(f"⚠️ Requested {s_target} but only {actual_count} available. Adjusting...")
    
    s_count = actual_count
    p_count = 2 if args.test_run else min(64, s_count)
    t_count = 2 if args.test_run else 20

    # 1. Reference FP16 생성
    if accelerator.is_main_process:
        print(f"🌟 Generating {s_count} Reference images using COCO captions...")
        pipe_ref = PixArtAlphaPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
        for i in range(s_count):
            ref_path = os.path.join(args.ref_dir, f"ref_{i}.png")
            if not os.path.exists(ref_path): # 이미 있으면 스킵
                gen = torch.Generator(device=device).manual_seed(42 + i)
                img = pipe_ref(prompts[i], num_inference_steps=t_count, generator=gen).images[0]
                img.save(ref_path)
        del pipe_ref; torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    # 2. 양자화 모델 로드
    pipe = PixArtAlphaPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="balanced")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    transformer = pipe.transformer

    # 3. Calibration (모든 GPU가 각자 담당 레이어 데이터 수집)
    all_samples = {}
    def hook_fn(name):
        def forward_hook(m, i, o):
            if name not in all_samples: all_samples[name] = []
            if len(all_samples[name]) < p_count: all_samples[name].append(i[0].detach().cpu())
        return forward_hook
    
    hooks = [m.register_forward_hook(hook_fn(n)) for n, m in transformer.named_modules() if isinstance(m, nn.Linear)]
    
    print(f"⚙️ Calibrating with {p_count} different prompts...")
    for i in range(p_count):
        pipe(prompts[i], num_inference_steps=t_count)
        
    for h in hooks: h.remove()
    for name in all_samples: 
        all_samples[name] = torch.cat(all_samples[name], dim=0).mean(dim=0, keepdim=True)

    # 4. 레이어 교체 (🎯 모든 GPU가 자기 구역 레이어를 직접 캘리브레이션 - 노이즈 방지)
    linear_names = [n for n, m in transformer.named_modules() if isinstance(m, nn.Linear)]
    for name in tqdm(linear_names, desc="Quantizing Layers", disable=not accelerator.is_main_process):
        orig_m = get_module_by_name(transformer, name)
        curr_dev = next(orig_m.parameters()).device
        
        if curr_dev == device:
            new_l = ManualSVDLinear(orig_m, args.act_mode, args.wgt_mode, args.alpha, args.lowrank, args.block_size, 
                                     torch.float16 if args.numeric_dtype == "half" else torch.float32).to(curr_dev)
            if name in all_samples:
                new_l.manual_calibrate_and_svd(all_samples[name].to(curr_dev))
            set_module_by_name(transformer, name, new_l)

    accelerator.wait_for_everyone()

    # 5. 최종 이미지 생성 및 지표 계산
    if accelerator.is_main_process:
        print(f"📊 Calculating Metrics for {s_count} samples...")
        psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        lpips_m = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
        is_m = InceptionScore().to(device)
        fid_m = FrechetInceptionDistance(feature=2048).to(device)
        
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        metrics = {"psnr": [], "ssim": [], "lpips": [], "clip": []}

        for i in range(s_count):
            gen = torch.Generator(device=device).manual_seed(42 + i)
            # 🎯 양자화된 모델로 생성
            q_img = pipe(prompts[i], num_inference_steps=t_count, generator=gen).images[0]
            r_img = Image.open(os.path.join(args.ref_dir, f"ref_{i}.png")).convert("RGB")
            q_img.save(os.path.join(args.save_dir, f"sample_{i}.png"))

            q_ten = ToTensor()(q_img).unsqueeze(0).to(device)
            r_ten = ToTensor()(r_img).unsqueeze(0).to(device)
            
            # 지표 업데이트
            metrics["psnr"].append(float(psnr_m(q_ten, r_ten)))
            metrics["ssim"].append(float(ssim_m(q_ten, r_ten)))
            metrics["lpips"].append(float(lpips_m(q_ten * 2 - 1, r_ten * 2 - 1)))
            
            img_uint8 = (q_ten * 255).to(torch.uint8)
            ref_uint8 = (r_ten * 255).to(torch.uint8)
            is_m.update(img_uint8)
            fid_m.update(ref_uint8, real=True)
            fid_m.update(img_uint8, real=False)

            # CLIP Score: 각 이미지와 그에 맞는 원본 캡션(prompts[i]) 비교
            inputs = clip_processor(text=[prompts[i]], images=q_img, return_tensors="pt", padding=True).to(device)
            clip_out = clip_model(**inputs)
            metrics["clip"].append(float(clip_out.logits_per_image.item()))

        res_is, _ = is_m.compute()
        res_fid = fid_m.compute()

        # 결과 저장
        final_res = {
            "config": vars(args), 
            "averages": {k: np.mean(v) for k, v in metrics.items()}, 
            "dist_metrics": {"FID": float(res_fid), "IS": float(res_is)}
        }
        
        with open(f"{args.save_dir}/metrics.json", "w") as f: 
            json.dump(final_res, f, indent=4)
            
        print(f"\n✅ All Done! Results saved to {args.save_dir}")
        print(f"FID: {res_fid:.2f} | CLIP: {np.mean(metrics['clip']):.2f} | PSNR: {np.mean(metrics['psnr']):.2f}")

if __name__ == "__main__": 
    main()
