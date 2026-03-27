import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from diffusers import PixArtAlphaPipeline
from tqdm import tqdm
import copy

# ==========================================
# 1. 핵심 양자화 함수 및 클래스
# ==========================================

def manual_quant_act(x, block_size=16, mode="NVFP4"):
    """활성화 양자화: MX/NVFP4/INTn 지원"""
    curr_dtype = x.dtype
    orig_shape = x.shape
    x_flat = x.to(torch.float32).view(-1, block_size)
    
    if mode == "NVFP4": q_max = 6.0
    elif mode == "FP8": q_max = 448.0
    elif "INT" in mode:
        bit = int(mode.replace("INT", ""))
        q_max = float(2**(bit - 1) - 1)
    else: q_max = 6.0
        
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
        
        # 가중치 및 버퍼 설정
        self.register_buffer('weight', original_linear.weight.data.clone().to(dtype))
        self.register_buffer('w_ternary', torch.zeros_like(self.weight).to(dtype))
        self.register_buffer('smooth_scale', torch.ones(self.weight.shape[1]).to(dtype))
        
        if original_linear.bias is not None:
            self.register_buffer('bias', original_linear.bias.data.clone().to(dtype))
        else:
            self.bias = None
            
        self.lora_a = torch.nn.Parameter(torch.zeros(rank, self.weight.shape[1]).to(dtype))
        self.lora_b = torch.nn.Parameter(torch.zeros(self.weight.shape[0], rank).to(dtype))
        self.is_calibrated = False

    @torch.no_grad()
    def manual_calibrate_and_svd(self, x_samples, mode=None, act=None):
        w_mode = mode if mode is not None else self.weight_mode
        self.act_mode = act if act is not None else self.act_mode
        
        x_flat = x_samples.view(-1, x_samples.shape[-1]).float()
        x_max = x_flat.abs().max(dim=0)[0].clamp(min=1e-12)
        w_max = self.weight.abs().max(dim=0)[0].clamp(min=1e-12)

        # Alpha=0.5일 때 s=1이 되는 수식
        self.smooth_scale.data = ((x_max ** self.alpha) / (w_max ** (1 - self.alpha)) / 
                                  torch.sqrt(x_max / w_max)).to(self.target_dtype)
        
        w_smoothed = self.weight.float() / self.smooth_scale.float().view(1, -1)
        w_reshaped = w_smoothed.view(w_smoothed.shape[0], -1, self.block_size)
        t_scale = w_reshaped.abs().amax(dim=2, keepdim=True).clamp(min=1e-12)

        if w_mode in ["NVFP4", "MXFP4"]: q_max = 6.0
        elif w_mode == "INT4": q_max = 7.0
        elif w_mode in ["INT2", "TERNARY"]: q_max = 1.0
        elif w_mode == "BINARY": q_max = 1.0
        else: q_max = 1.0

        if w_mode == "BINARY":
            w_quant_reshaped = torch.sign(w_reshaped) * 1.0
        else:
            w_quant_reshaped = torch.clamp(torch.round(w_reshaped / t_scale * q_max), -q_max, q_max)

        w_quant = (w_quant_reshaped * t_scale / q_max).view(w_smoothed.shape)
        self.w_ternary.copy_(w_quant.to(self.target_dtype))

        # SVD 보정
        w_error = w_smoothed - w_quant
        U, S, Vh = torch.linalg.svd(w_error.float(), full_matrices=False)
        sqrt_S = torch.sqrt(S[:self.rank])
        
        self.lora_b.data = (U[:, :self.rank] * sqrt_S.unsqueeze(0)).to(self.target_dtype)
        self.lora_a.data = (Vh[:self.rank, :] * sqrt_S.unsqueeze(1)).to(self.target_dtype)
        
        self.is_calibrated = True

    def forward(self, x):
        input_dtype = x.dtype
        curr_x = x.to(self.target_dtype) * self.smooth_scale
        x_q = manual_quant_act(curr_x, self.block_size, mode=self.act_mode)
        
        base_out = F.linear(x_q, self.w_ternary)
        svd_out = F.linear(F.linear(x_q, self.lora_a), self.lora_b)
        
        res = base_out + svd_out + (self.bias if self.bias is not None else 0)
        return res.to(input_dtype)

# ==========================================
# 2. 유틸리티 함수 (레이어 교체용)
# ==========================================

def get_module_by_name(model, name):
    parts = name.split('.')
    module = model
    for part in parts:
        module = getattr(module, part)
    return module

def set_module_by_name(model, name, new_module):
    parts = name.split('.')
    parent_name = '.'.join(parts[:-1])
    child_name = parts[-1]
    parent = get_module_by_name(model, parent_name) if parent_name else model
    setattr(parent, child_name, new_module)

# ==========================================
# 3. 메인 실행 로직
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--act_mode", type=str, default="NVFP4")
    parser.add_argument("--wgt_mode", type=str, default="TERNARY")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--numeric_dtype", type=str, default="half")
    parser.add_argument("--lowrank", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--prompt", type=str, default="A stylish woman with a pearl necklace, highly detailed beads.")
    args = parser.parse_args()

    # Dtype & Device 설정
    dtype_map = {"half": torch.float16, "float": torch.float32}
    target_dtype = dtype_map.get(args.numeric_dtype, torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 모델 로드
    print(f"📦 Loading Model: {args.model_path}")
    pipe = PixArtAlphaPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
    transformer = pipe.transformer

    # 2. 데이터 수집 (Calibration용)
    print("🧪 Collecting Calibration Samples...")
    all_samples = {}
    def hook_fn(name):
        def forward_hook(module, input, output):
            if name not in all_samples:
                all_samples[name] = input[0].detach().cpu()
        return forward_hook

    hooks = []
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # 샘플 하나 실행하여 입력값 수집
    pipe(args.prompt, num_inference_steps=1)
    for hook in hooks: hook.remove()

    # 3. 레이어 교체 및 캘리브레이션
    print(f"🔄 Replacing Layers and Calibrating (Mode: {args.wgt_mode})...")
    linear_names = [name for name, m in transformer.named_modules() if isinstance(m, nn.Linear)]
    
    for name in tqdm(linear_names):
        original_module = get_module_by_name(transformer, name)
        
        # 새 레이어 생성 (James님의 주입 방식)
        new_layer = ManualSVDLinear(
            original_module, 
            act_mode=args.act_mode, 
            weight_mode=args.wgt_mode,
            alpha=args.alpha, 
            rank=args.lowrank, 
            block_size=args.block_size,
            dtype=target_dtype
        ).to(device)
        
        # 캘리브레이션 수행
        if name in all_samples:
            new_layer.manual_calibrate_and_svd(all_samples[name].to(device))
        
        # 모델 구조 업데이트
        set_module_by_name(transformer, name, new_layer)

    # 4. 최종 이미지 생성 (Inference)
    print("🎨 Generating Final Image...")
    generator = torch.Generator(device=device).manual_seed(42)
    image = pipe(args.prompt, num_inference_steps=20, generator=generator).images[0]

    # Preview 경로 가독성 조립
    b_idx = 0 
    preview_filename = (
        f"preview_step_"
        f"A{args.act_mode}_"
        f"W{args.wgt_mode}_"
        f"a{args.alpha}_"
        f"N{args.numeric_dtype}_"
        f"R{args.lowrank}_"
        f"{b_idx}.png"
    )
    preview_path = os.path.join(args.save_dir, preview_filename)
    
    image.save(preview_path)
    print(f"✅ Experiment Complete! Image saved at: {preview_path}")

if __name__ == "__main__":
    main()