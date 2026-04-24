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
import gc

# ==========================================
# 0. MJHQ Data set
# ==========================================

def get_prompts(num_samples, args):
    if args.dataset_name == "MJHQ":
        path, split, key = "xingjianleng/mjhq30k", "test", "text"

    try:
        # 🎯 trust_remote_code를 제거하고 시도합니다. 
        # 최신 버전에서는 로딩 스크립트가 없는 데이터셋은 이게 없어야 돌아갑니다.
        dataset = load_dataset(path, split=split, streaming=True)
        prompts = []
        for i, entry in enumerate(dataset):
            if i >= num_samples: break
            prompts.append(entry[key])
        return prompts

    except Exception as e:
        print(f"⚠️ Dataset loading failed ({e}). Using fallback prompts...")
        # 🎯 SONGHAN 데이터셋 로드가 실패할 경우를 대비한 퀄리티 높은 기본 프롬프트들
        fallback_prompts = [
            "A professional high-quality photo of a futuristic city with neon lights",
            "A beautiful landscape of mountains during sunset, cinematic lighting",
            "A cute robot holding a flower in a field, highly detailed digital art",
            "A gourmet burger with melting cheese and fresh vegetables on a wooden table",
            "An astronaut walking on a purple planet's surface under a starry sky"
        ]
        # 요청한 수만큼 프롬프트 생성 (부족하면 반복)
        return (fallback_prompts * (num_samples // len(fallback_prompts) + 1))[:num_samples]


# 🎯 [신규 추가] INT8~INT2 및 TERNARY 범용 양자화 함수
def quantize_uniform(x, block_size=16, mode="INT4"):
    orig_shape = x.shape
    x_flat = x.view(-1, block_size)
    
    # 모드에 따른 최대 표현 가능 정수 (q_max) 계산
    if mode == "TERNARY":
        q_max = 1.0  # -1, 0, 1
    elif mode.startswith("INT"):
        bits = int(mode.replace("INT", ""))
        q_max = (2 ** (bits - 1)) - 1.0  # 예: INT4 -> 2^3 - 1 = 7.0
    else:
        raise ValueError(f"Unsupported quantization mode: {mode}")

    # Block 단위 스케일 계산 (대칭 양자화)
    amax = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / q_max
    
    # 정수 반올림 후 클리핑, 다시 원래 스케일로 복원 (Dequantize)
    x_q_flat = torch.clamp(torch.round(x_flat / scale), -q_max, q_max)
    x_dq = x_q_flat * scale
    
    return x_dq.view(orig_shape)

def quantize_to_nvfp4(x, block_size=16):
    orig_shape = x.shape
    x_flat = x.view(-1, block_size)
    
    nvfp4_levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device, dtype=x.dtype)

    amax = x_flat.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    
    scale = amax / 6.0

    x_norm = x_flat.abs() / scale
    x_norm_expanded = x_norm.unsqueeze(-1)
    distances = torch.abs(x_norm_expanded - nvfp4_levels)
    closest_idx = torch.argmin(distances, dim=-1)

    x_q = torch.sign(x_flat) * nvfp4_levels[closest_idx] * scale
    
    return x_q.view(orig_shape)


class ManualSVDLinear(torch.nn.Module):
    def __init__(self, original_linear, act_mode, wgt_mode, alpha=0.5, rank=32, block_size=16, dtype=torch.float32):
        super().__init__()
        self.target_dtype = dtype
        self.act_mode = act_mode
        self.wgt_mode = wgt_mode
        self.alpha = alpha
        self.rank = rank
        self.block_size = block_size
        
        # 원본 파라미터 백업
        self.register_buffer('weight', original_linear.weight.data.clone().to(dtype))
        self.bias = nn.Parameter(original_linear.bias.data.clone().to(dtype)) if original_linear.bias is not None else None
        
        # 양자화 및 SVD를 위한 버퍼/파라미터
        self.register_buffer('w_quantized', original_linear.weight.data.clone().to(dtype))
        self.register_buffer('smooth_scale', torch.ones(self.weight.shape[1]).to(dtype))
        
        # Low Rank 브랜치
        self.lora_a = nn.Parameter(torch.zeros(rank, self.weight.shape[1]).to(dtype))
        self.lora_b = nn.Parameter(torch.zeros(self.weight.shape[0], rank).to(dtype))
        
        self.is_calibrated = False

    @torch.no_grad()
    def manual_calibrate_and_svd(self, x_max):
        x_max = x_max.clamp(min=1e-5).float()
        w_max = self.weight.abs().max(dim=0)[0].clamp(min=1e-5).float()

        # 1. Smooth Scale 계산 (alpha=0.5)
        self.smooth_scale.data = (w_max.pow(1 - self.alpha) / x_max.pow(self.alpha)).to(self.target_dtype)
        self.smooth_scale.data = self.smooth_scale.data.clamp(min=1e-4, max=1e4)
        # 2. 가중치 Smoothing 적용
        w_smoothed = self.weight.float() / self.smooth_scale.float().view(1, -1)
        
        # 3. Smoothing된 가중치를 양자화
        if self.wgt_mode == "NVFP4":
            w_q = quantize_to_nvfp4(w_smoothed, self.block_size)
        else:
            w_q = quantize_uniform(w_smoothed, self.block_size, mode=self.wgt_mode)
        
        # 4. 양자화된 가중치 저장 (Dequantized 상태)
        self.w_quantized.copy_(w_q.to(self.target_dtype))
        
        # 5. SVD를 위한 Error 계산
        # 핵심: 원래 W를 표현하기 위해선, 양자화된 W_q와 Error의 합이 W_smoothed와 같아야 함
        w_error = w_smoothed - w_q 
        
        # 6. SVD 분해
        U, S, Vh = torch.linalg.svd(w_error.float(), full_matrices=False)
        
        # 🎯 [핵심 수정] 
        # 실제 행렬이 가진 S의 길이와 설정한 rank 중 작은 값을 실제 rank로 사용합니다.
        actual_rank = min(self.rank, S.shape[0])
        
        sqrt_S = torch.sqrt(S[:actual_rank])

        # lora_a (입력 쪽)
        V_scaled = Vh[:actual_rank, :] * sqrt_S.unsqueeze(1)
        
        # 만약 실제 rank가 설정된 rank보다 작다면, 0으로 채워진(Zero-padded) 
        # 원본 크기의 텐서를 만들어서 lora_a, lora_b 에 덮어씌워야 구조가 깨지지 않습니다.
        lora_a_data = torch.zeros(self.rank, self.weight.shape[1], dtype=self.target_dtype, device=x_max.device)
        lora_a_data[:actual_rank, :] = V_scaled.to(self.target_dtype)
        self.lora_a.data = lora_a_data

        # lora_b (출력 쪽)
        U_scaled = U[:, :actual_rank] * sqrt_S.unsqueeze(0)
        lora_b_data = torch.zeros(self.weight.shape[0], self.rank, dtype=self.target_dtype, device=x_max.device)
        lora_b_data[:, :actual_rank] = U_scaled.to(self.target_dtype)
        self.lora_b.data = lora_b_data
        
        self.is_calibrated = True


    def forward(self, x):
        input_dtype = x.dtype
        if not self.is_calibrated: 
            return F.linear(x, self.weight.to(input_dtype), self.bias.to(input_dtype) if self.bias is not None else None)
        
        # 1. Activation Smoothing: X_smoothed = X * diag(smooth_scale)
        x_smoothed = x.to(self.target_dtype) * self.smooth_scale
        
        # 2. Activation 양자화
        if self.act_mode == "NVFP4":
            x_q = quantize_to_nvfp4(x_smoothed, self.block_size)
        else:
            x_q = quantize_uniform(x_smoothed, self.block_size, mode=self.act_mode)
            
        # 3. Base 브랜치 연산: Y_base = X_q * W_q^T
        base_out = F.linear(x_q, self.w_quantized)
        
        # 4. 🎯 [수정 2] SVD 브랜치 연산: 원래 입력 x를 그대로 사용합니다.
        # 위에서 lora_a에 이미 smooth_scale 역수를 반영했으므로, 
        # 여기서는 Smoothing 전의 순수 입력값(x)을 넣어야 수식이 완벽히 성립합니다.
        svd_out = F.linear(F.linear(x_smoothed, self.lora_a), self.lora_b)
        
        # 5. 최종 합산
        out = base_out + svd_out
        if self.bias is not None:
            out += self.bias
            
        return out.to(input_dtype)


class ManualRPCALinear(torch.nn.Module):
    def __init__(self, original_linear, act_mode, wgt_mode, alpha=0.5, rank=32, block_size=16, outlier_ratio=0.01, dtype=torch.float32):
        super().__init__()
        self.target_dtype = dtype
        self.act_mode = act_mode
        self.wgt_mode = wgt_mode
        self.alpha = alpha
        self.rank = rank
        self.block_size = block_size
        self.outlier_ratio = outlier_ratio # 🎯 [추가] 튀는 값을 얼마만큼 빼낼 것인지 (예: 1%)
        
        # 원본 파라미터 백업
        self.register_buffer('weight', original_linear.weight.data.clone().to(dtype))
        self.bias = nn.Parameter(original_linear.bias.data.clone().to(dtype)) if original_linear.bias is not None else None
        
        # RPCA 분리를 위한 버퍼 (S 행렬)
        self.register_buffer('w_sparse', torch.zeros_like(original_linear.weight.data).to(dtype))
        
        # 양자화 및 L 행렬(Dense) 처리를 위한 버퍼/파라미터
        self.register_buffer('w_quantized', original_linear.weight.data.clone().to(dtype))
        self.register_buffer('smooth_scale', torch.ones(self.weight.shape[1]).to(dtype))
        
        # Low Rank 브랜치 (L 행렬의 양자화 오차 보정용)
        self.lora_a = nn.Parameter(torch.zeros(rank, self.weight.shape[1]).to(dtype))
        self.lora_b = nn.Parameter(torch.zeros(self.weight.shape[0], rank).to(dtype))
        
        self.is_calibrated = False

    @torch.no_grad()
    def manual_calibrate_and_rpca(self, x_max):
        x_max = x_max.clamp(min=1e-5).float()
        
        # 🎯 1. RPCA 핵심: Sparse(S)와 Dense(L) 행렬 분리
        w_abs = self.weight.abs()
        # 전체 가중치 중 상위 outlier_ratio% 에 해당하는 기준값(threshold) 계산
        threshold = torch.quantile(w_abs.view(-1).float(), 1.0 - self.outlier_ratio)
        
        # Outlier 마스크 생성
        sparse_mask = w_abs >= threshold
        
        # S 행렬 (Sparse): Outlier만 남기고 나머지는 0. 이 값들은 양자화하지 않고 고정밀도 유지.
        self.w_sparse.copy_((self.weight * sparse_mask).to(self.target_dtype))
        
        # L 행렬 (Dense): Outlier가 제거되어 값의 분포가 매우 안정적인 상태.
        w_dense = self.weight * (~sparse_mask)

        # 🎯 2. Smooth Scale 계산 (주의: 거친 Outlier가 빠진 w_dense 기준으로 계산!)
        # 이렇게 하면 scale 값이 튀지 않아 초저정밀도 양자화에 훨씬 유리해집니다.
        w_max = w_dense.abs().max(dim=0)[0].clamp(min=1e-5).float()
        self.smooth_scale.data = (w_max.pow(1 - self.alpha) / x_max.pow(self.alpha)).to(self.target_dtype)
        self.smooth_scale.data = self.smooth_scale.data.clamp(min=1e-4, max=1e4)
        
        # 3. L 행렬 Smoothing 적용
        w_smoothed = w_dense.float() / self.smooth_scale.float().view(1, -1)
        
        # 4. Smoothing된 L 행렬을 양자화
        if self.wgt_mode == "NVFP4":
            w_q = quantize_to_nvfp4(w_smoothed, self.block_size)
        else:
            w_q = quantize_uniform(w_smoothed, self.block_size, mode=self.wgt_mode)
        
        self.w_quantized.copy_(w_q.to(self.target_dtype))
        
        # 5. SVD 분해를 위한 L 행렬의 양자화 Error 계산
        w_error = w_smoothed - w_q 
        
        # 6. Error에 대한 SVD 분해 (기존과 동일)
        U, S, Vh = torch.linalg.svd(w_error.float(), full_matrices=False)
        actual_rank = min(self.rank, S.shape[0])
        sqrt_S = torch.sqrt(S[:actual_rank])

        V_scaled = Vh[:actual_rank, :] * sqrt_S.unsqueeze(1)
        lora_a_data = torch.zeros(self.rank, self.weight.shape[1], dtype=self.target_dtype, device=x_max.device)
        lora_a_data[:actual_rank, :] = V_scaled.to(self.target_dtype)
        self.lora_a.data = lora_a_data

        U_scaled = U[:, :actual_rank] * sqrt_S.unsqueeze(0)
        lora_b_data = torch.zeros(self.weight.shape[0], self.rank, dtype=self.target_dtype, device=x_max.device)
        lora_b_data[:, :actual_rank] = U_scaled.to(self.target_dtype)
        self.lora_b.data = lora_b_data
        
        self.is_calibrated = True

    def forward(self, x):
        input_dtype = x.dtype
        if not self.is_calibrated: 
            return F.linear(x, self.weight.to(input_dtype), self.bias.to(input_dtype) if self.bias is not None else None)
        
        # 1. Activation Smoothing
        x_smoothed = x.to(self.target_dtype) * self.smooth_scale
        
        # 2. Activation 양자화
        if self.act_mode == "NVFP4":
            x_q = quantize_to_nvfp4(x_smoothed, self.block_size)
        else:
            x_q = quantize_uniform(x_smoothed, self.block_size, mode=self.act_mode)
            
        # 3. Base 브랜치 (L 행렬의 양자화 파트): Y_base = X_q * W_q^T
        base_out = F.linear(x_q, self.w_quantized)
        
        # 4. SVD 브랜치 (L 행렬의 오차 보정 파트): 원본 입력이 아닌 smoothed된 입력 사용
        svd_out = F.linear(F.linear(x_smoothed, self.lora_a), self.lora_b)
        
        # 🎯 5. Sparse 브랜치 (S 행렬 - Outlier 파트)
        # S 행렬은 smoothing의 영향을 받지 않았으므로 순수한 원본 입력 x를 곱해줍니다.
        sparse_out = F.linear(x.to(self.target_dtype), self.w_sparse)
        
        # 6. 최종 합산
        out = base_out + svd_out + sparse_out
        if self.bias is not None:
            out += self.bias
            
        return out.to(input_dtype)

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
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lowrank", type=int, default=32)
    quant_modes = ["NVFP4", "INT8", "INT4", "INT3", "INT2", "TERNARY"]
    parser.add_argument("--act_mode", type=str, default="NVFP4", choices=quant_modes)
    parser.add_argument("--wgt_mode", type=str, default="NVFP4", choices=quant_modes)
    parser.add_argument("--numeric_dtype", type=str, default="half")
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--dataset_name", type=str, default="MJHQ", choices=["MJHQ", "sDCI"])
    parser.add_argument("--img_base_dir", type=str, default="/data/jameskimh/james_dit_pixart_xl_mjhq")
    parser.add_argument("--quant_method", type=str, default="SVD", choices=["SVD", "RPCA"])
    args = parser.parse_args()

# 🎯 [Step 0] 카운트 및 프롬프트 설정
    # s_count: 최종 평가할 이미지 수, p_count: 캘리브레이션에 쓸 데이터 수
    s_target = 2 if args.test_run else args.num_samples
    
    accelerator = Accelerator()
    device = accelerator.device
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

    # 데이터셋에서 프롬프트 로드
    prompts = get_prompts(s_target, args)
    actual_count = len(prompts)

    if actual_count < s_target:
        print(f"⚠️ Requested {s_target} but only {actual_count} available. Adjusting...")
    
    s_count = actual_count
    p_count = 2 if args.test_run else min(64, s_count)
    t_count = 20

    # 1. Reference FP16 생성
    if accelerator.is_main_process:
        accelerator.print(f"🌟 Generating Reference images in: {dataset_ref_dir}")
        pipe_ref = PixArtAlphaPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)

        for i in range(s_count):
            ref_path = os.path.join(dataset_ref_dir, f"ref_{i}.png")
            if not os.path.exists(ref_path): # 이미 있으면 스킵
                gen = torch.Generator(device=device).manual_seed(42 + i)
                img = pipe_ref(prompts[i], num_inference_steps=t_count, generator=gen).images[0]
                img.save(ref_path)
        del pipe_ref
        torch.cuda.empty_cache()
        gc.collect()
    accelerator.wait_for_everyone()

    # 2. 양자화 모델 로드
    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16, 
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    transformer = pipe.transformer
    
    # ==========================================
    # 3. Target 선정 및 Calibration
    # ==========================================
    skip_keywords = [
        "x_embedder",  
        "t_embedder",  
        "proj_out"     
    ]
    
    target_linear_names = []
    for n, m in transformer.named_modules():
        if isinstance(m, nn.Linear):
            if any(skip in n for skip in skip_keywords):
                continue
            target_linear_names.append(n)

    if accelerator.is_main_process:
        print(f"\n🎯 Targeted {len(target_linear_names)} layers for NVFP4 Quantization.")

    all_samples = {}
    def hook_fn(name):
        def forward_hook(m, i, o):
            input_act = i[0].detach()
            if name not in all_samples:
                all_samples[name] = [] # 🎯 텐서가 아닌 리스트로 모읍니다.

            # 채널별 Max 추출
            flat_act = input_act.view(-1, input_act.shape[-1]).abs().float()
            step_max = flat_act.max(dim=0)[0].cpu()
            all_samples[name].append(step_max)
            
        return forward_hook

    hooks = []
    for name in target_linear_names:
        m = get_module_by_name(transformer, name)
        hooks.append(m.register_forward_hook(hook_fn(name)))
    
    print(f"⚙️ Calibrating with {p_count} prompts (Distributed)...", flush=True)
    with accelerator.split_between_processes(prompts[:p_count]) as local_prompts:
        for prompt in local_prompts:
            # 🎯 [핵심 복구] 생성할 때와 똑같은 스텝(t_count=20)으로 캘리브레이션 해야 분포가 틀어지지 않습니다!
            pipe(prompt, num_inference_steps=t_count, generator=torch.Generator(device=device).manual_seed(42))
        
    for h in hooks: h.remove()

    # 🎯 [핵심 복구] 스텝별 Max들의 평균을 내어 극단적인 픽셀 깨짐을 방지합니다.
    for name in all_samples: 
        local_mean_of_maxes = torch.stack(all_samples[name]).mean(dim=0).to(device)
        all_samples[name] = accelerator.reduce(local_mean_of_maxes, reduction="mean")

    accelerator.wait_for_everyone()

    # ==========================================
    # 4. 타겟 레이어 교체 (모든 GPU가 동일한 모델을 생성)
    # ==========================================
    # 타겟팅된 레이어 이름 리스트만 순회합니다.
    for name in tqdm(target_linear_names, desc="Quantizing Attn & MLP", disable=not accelerator.is_main_process):
        orig_m = get_module_by_name(transformer, name)
        
        curr_dev = next(orig_m.parameters()).device
        if curr_dev == device:
            # 1. 공통으로 사용할 Target Data Type 미리 정의
            target_dtype = torch.float16 if args.numeric_dtype == "half" else torch.float32
            
            if args.quant_method == "SVD":
                new_l = ManualSVDLinear(
                    orig_m, args.act_mode, args.wgt_mode, args.alpha, 
                    args.lowrank, args.block_size, target_dtype
                ).to(device)
                
                if name in all_samples:
                    # SVD 캘리브레이션 호출
                    new_l.manual_calibrate_and_svd(all_samples[name])
                    
            elif args.quant_method == "RPCA":
                # 🎯 args에 outlier_ratio가 세팅되어 있지 않다면 기본값(예: 0.01)을 사용하도록 안전장치 추가
                outlier_ratio = getattr(args, 'outlier_ratio', 0.01) 
                
                new_l = ManualRPCALinear(
                    orig_m, args.act_mode, args.wgt_mode, args.alpha, 
                    args.lowrank, args.block_size, outlier_ratio, target_dtype
                ).to(device)
                
                if name in all_samples:
                    # 🎯 RPCA 전용 캘리브레이션 함수 호출
                    new_l.manual_calibrate_and_rpca(all_samples[name])
            
            else:
                raise ValueError(f"지원하지 않는 양자화 방식입니다: {args.quant_method}")
            
            # 2. 완성된 새로운 레이어로 원본 트랜스포머의 레이어 교체
            set_module_by_name(transformer, name, new_l)

    # 교체 완료 후 다시 동기화
    accelerator.wait_for_everyone()

    # ==========================================
    # 5. 이미지 생성 및 지표 계산 (모든 GPU 참여)
    # ==========================================
    
    # 지표 객체는 모든 GPU에서 동일하게 생성해야 합니다 (if 블록 밖으로 이동)
    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
    is_m = InceptionScore().to(device)
    fid_m = FrechetInceptionDistance(feature=2048).to(device)
    
    # CLIP은 메인 프로세스에서만 로드 (메모리 절약)
    if accelerator.is_main_process:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # 🎯 [1] 이미지 생성 (이미 구현하신 방식 유지)
    indices = list(range(s_count))
    with accelerator.split_between_processes(indices) as local_indices:
        for i in local_indices:
            gen = torch.Generator(device=device).manual_seed(42 + i)
            q_img = pipe(prompts[i], num_inference_steps=t_count, generator=gen).images[0]
            save_path = os.path.join(dataset_save_dir, f"sample_{i}.png")
            q_img.save(save_path)
            
            # 생성 즉시 각 GPU의 지표 객체에 업데이트 (2배 빠름!)
            ref_path = os.path.join(dataset_ref_dir, f"ref_{i}.png")
            r_img = Image.open(ref_path).convert("RGB")

            q_ten = ToTensor()(q_img).unsqueeze(0).to(device)
            r_ten = ToTensor()(r_img).unsqueeze(0).to(device)
            
            psnr_m.update(q_ten, r_ten)
            ssim_m.update(q_ten, r_ten)
            lpips_m.update(q_ten * 2 - 1, r_ten * 2 - 1)
            
            img_uint8 = (q_ten * 255).to(torch.uint8)
            ref_uint8 = (r_ten * 255).to(torch.uint8)
            is_m.update(img_uint8)
            fid_m.update(ref_uint8, real=True)
            fid_m.update(img_uint8, real=False)

    # 🎯 [2] 모든 GPU가 업데이트를 마칠 때까지 대기
    accelerator.wait_for_everyone()

    # 🎯 [3] 지표 계산 (모든 GPU가 동시에 호출 - NCCL Timeout 방지 핵심)
    print(f"📊 GPU {accelerator.process_index} is computing metrics...", flush=True)
    res_psnr = psnr_m.compute()
    res_ssim = ssim_m.compute()
    res_lpips = lpips_m.compute()
    res_is, _ = is_m.compute()
    res_fid = fid_m.compute()

    # 🎯 [4] 최종 결과 저장 및 CLIP Score (메인 프로세스 전담)
    if accelerator.is_main_process:
        all_clip_scores = []
        print(f"🖼️ Calculating CLIP Scores for {s_count} images...", flush=True)
        
        for i in range(s_count):
            # 🎯 수정: dataset_save_dir에서 파일을 읽어야 합니다.
            q_img_path = os.path.join(dataset_save_dir, f"sample_{i}.png")
            
            if not os.path.exists(q_img_path):
                continue
                
            q_img = Image.open(q_img_path).convert("RGB")
            
            # CLIP 계산
            inputs = clip_processor(text=[prompts[i]], images=q_img, return_tensors="pt", padding=True).to(device)
            clip_out = clip_model(**inputs)
            all_clip_scores.append(float(clip_out.logits_per_image.item()))

        # 최종 결과 취합
        final_res = {
            "config": vars(args), 
            "averages": {
                "psnr": float(res_psnr),
                "ssim": float(res_ssim),
                "lpips": float(res_lpips),
                "clip": np.mean(all_clip_scores) if all_clip_scores else 0.0
            }, 
            "dist_metrics": {"FID": float(res_fid), "IS": float(res_is)}
        }
        
        # 🎯 수정: 메트릭 파일도 데이터셋 폴더 내부에 저장
        metrics_save_path = os.path.join(dataset_save_dir, "metrics.json")
        with open(metrics_save_path, "w") as f: 
            json.dump(final_res, f, indent=4)
            
        print(f"\n✅ All Done! Results saved to {dataset_save_dir}", flush=True)
        print(f"FID: {res_fid:.2f} | CLIP: {np.mean(all_clip_scores):.2f} | PSNR: {res_psnr:.2f}", flush=True)

    # 모든 작업 완료 후 동기화 및 종료
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("🚀 Process completed successfully.", flush=True)

    # 메모리 정리
    if 'pipe' in locals():
        del pipe
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__": 
    main()
