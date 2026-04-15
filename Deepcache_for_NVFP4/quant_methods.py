"""
quant_methods.py

4가지 NVFP4 양자화 방법 구현.

M1 - ManualRTNLinear   : SmoothQuant + NVFP4 RTN (SVD 없음) — 가장 단순, format-only 기준선
M2 - SVDQuant          : mtq.NVFP4_SVDQUANT_DEFAULT_CFG (라이브러리 호출)
                         논문: SVDQuant (NeurIPS 2024 Spotlight)
M3 - MRGPTQLinear      : Micro-Rotation (H(16) per NVFP4 group) + GPTQ act_order + NVFP4 + SVD
                         논문: "MR-GPTQ: Bridging the Gap Between Promise and Performance
                                for Microscaling FP4 Quantization" (ICLR 2026)
                         핵심: NVFP4 group size(16)가 SmoothQuant을 무효화 →
                               대신 group-level H(16) rotation으로 분포 등방화
M4 - FourOverSixLinear : SmoothQuant + Adaptive block scaling (max=6 vs max=4 per block)
                         논문: "Four Over Six: More Accurate NVFP4 Quantization
                                with Adaptive Block Scaling" (MIT Han Lab, arXiv 2512.02010)
                         핵심: NVFP4 비균일 step size (4→6 구간 step=2)로 인한
                               near-maximal value (~5) 오류를 per-block max=4 선택으로 회피
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 공통 양자화 함수
# ---------------------------------------------------------------------------

def quantize_to_nvfp4(x, block_size=16):
    """Block-wise NVFP4 RTN. 8 non-uniform levels: [0, 0.5, 1, 1.5, 2, 3, 4, 6]."""
    orig_shape = x.shape
    x_flat = x.view(-1, block_size)
    nvfp4_levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=x.device, dtype=x.dtype,
    )
    amax = x_flat.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    scale = amax / 6.0
    x_norm = x_flat.abs() / scale
    distances = torch.abs(x_norm.unsqueeze(-1) - nvfp4_levels)
    closest_idx = torch.argmin(distances, dim=-1)
    x_q = torch.sign(x_flat) * nvfp4_levels[closest_idx] * scale
    return x_q.view(orig_shape)


def quantize_to_nvfp4_fouroversix(x, block_size=16):
    """
    Four Over Six adaptive block scaling NVFP4 (arXiv 2512.02010).

    NVFP4 levels: [0, 0.5, 1, 1.5, 2, 3, 4, 6] with step sizes [0.5, 0.5, 0.5, 0.5, 1, 1, 2].
    The 4→6 step (size 2) causes large error for values near 5.

    Per block, try two scales and pick the one with lower MSE:
      - max=6 (standard): scale = amax/6  (all 8 levels used)
      - max=4 (adaptive): scale = amax/4  (levels ≤4 used, avoids the step-2 region)
    """
    orig_shape = x.shape
    x_flat = x.view(-1, block_size)
    nvfp4_levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=x.device, dtype=x.dtype,
    )
    amax = x_flat.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-12)

    # Standard: scale = amax/6
    scale6  = amax / 6.0
    x_norm6 = x_flat.abs() / scale6
    idx6    = torch.argmin(torch.abs(x_norm6.unsqueeze(-1) - nvfp4_levels), dim=-1)
    x_q6    = torch.sign(x_flat) * nvfp4_levels[idx6] * scale6
    mse6    = ((x_flat - x_q6) ** 2).mean(dim=-1, keepdim=True)

    # Adaptive: scale = amax/4
    scale4  = amax / 4.0
    x_norm4 = x_flat.abs() / scale4
    idx4    = torch.argmin(torch.abs(x_norm4.unsqueeze(-1) - nvfp4_levels), dim=-1)
    x_q4    = torch.sign(x_flat) * nvfp4_levels[idx4] * scale4
    mse4    = ((x_flat - x_q4) ** 2).mean(dim=-1, keepdim=True)

    # Per-block: choose scale with lower MSE
    x_q = torch.where(mse4 < mse6, x_q4, x_q6)
    return x_q.view(orig_shape)


def quantize_uniform(x, block_size=16, mode="INT4"):
    """Block-wise symmetric uniform quantization (INT2~INT8, TERNARY)."""
    orig_shape = x.shape
    x_flat = x.view(-1, block_size)
    if mode == "TERNARY":
        q_max = 1.0
    elif mode.startswith("INT"):
        bits = int(mode.replace("INT", ""))
        q_max = float((2 ** (bits - 1)) - 1)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    amax = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / q_max
    x_q = torch.clamp(torch.round(x_flat / scale), -q_max, q_max) * scale
    return x_q.view(orig_shape)


# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------

def get_module_by_name(model, name):
    for part in name.split("."):
        model = getattr(model, part)
    return model


def set_module_by_name(model, name, new_module):
    parts = name.split(".")
    parent = get_module_by_name(model, ".".join(parts[:-1])) if len(parts) > 1 else model
    setattr(parent, parts[-1], new_module)


SKIP_KEYWORDS = ["x_embedder", "t_embedder", "proj_out"]


def get_target_linear_names(transformer):
    return [
        n for n, m in transformer.named_modules()
        if isinstance(m, nn.Linear)
        and not any(kw in n for kw in SKIP_KEYWORDS)
    ]


def _collect_x_max(pipe, transformer, target_names, prompts, p_count,
                   t_count, device, accelerator):
    """공통: forward hook으로 activation x_max 수집 + distributed reduce."""
    calib_data = {}

    def hook_fn(name):
        def forward_hook(m, inputs, output):
            x = inputs[0].detach().view(-1, inputs[0].shape[-1]).float()
            calib_data.setdefault(name, []).append(x.abs().max(dim=0)[0].cpu())
        return forward_hook

    hooks = [
        get_module_by_name(transformer, n).register_forward_hook(hook_fn(n))
        for n in target_names
    ]

    with accelerator.split_between_processes(prompts[:p_count]) as local_prompts:
        for prompt in local_prompts:
            pipe(prompt, num_inference_steps=t_count,
                 generator=torch.Generator(device=device).manual_seed(42))

    for h in hooks:
        h.remove()

    for name in calib_data:
        local_mean = torch.stack(calib_data[name]).mean(dim=0).to(device)
        calib_data[name] = accelerator.reduce(local_mean, reduction="mean")
    accelerator.wait_for_everyone()
    return calib_data


def _collect_hdiag(pipe, transformer, target_names, prompts, p_count,
                   t_count, device, accelerator):
    """공통: forward hook으로 H_diag(=E[x²]) 수집 + distributed reduce."""
    calib_data = {}

    def hook_fn(name):
        def forward_hook(m, inputs, output):
            x = inputs[0].detach().view(-1, inputs[0].shape[-1]).float()
            calib_data.setdefault(name, []).append(x.pow(2).mean(dim=0).cpu())
        return forward_hook

    hooks = [
        get_module_by_name(transformer, n).register_forward_hook(hook_fn(n))
        for n in target_names
    ]

    with accelerator.split_between_processes(prompts[:p_count]) as local_prompts:
        for prompt in local_prompts:
            pipe(prompt, num_inference_steps=t_count,
                 generator=torch.Generator(device=device).manual_seed(42))

    for h in hooks:
        h.remove()

    for name in calib_data:
        local_mean = torch.stack(calib_data[name]).mean(dim=0).to(device)
        calib_data[name] = accelerator.reduce(local_mean, reduction="mean")
    accelerator.wait_for_everyone()
    return calib_data


# ---------------------------------------------------------------------------
# M1: ManualRTNLinear (SmoothQuant + NVFP4 RTN, no SVD)
# ---------------------------------------------------------------------------

class ManualRTNLinear(nn.Module):
    """
    가장 단순한 NVFP4 양자화:
    SmoothQuant scale 적용 후 NVFP4 RTN. SVD error correction 없음.
    cache와 결합 시 다른 방법과 비교하는 기준선 (format-only 효과 측정).
    """

    def __init__(self, original_linear, alpha=0.5, block_size=16, dtype=torch.float16):
        super().__init__()
        self.target_dtype = dtype
        self.alpha = alpha
        self.block_size = block_size

        self.register_buffer("weight", original_linear.weight.data.clone().to(dtype))
        self.bias = nn.Parameter(
            original_linear.bias.data.clone().to(dtype)
        ) if original_linear.bias is not None else None

        self.register_buffer("w_quantized", original_linear.weight.data.clone().to(dtype))
        self.register_buffer("smooth_scale",
                             torch.ones(original_linear.in_features, dtype=dtype))
        self.is_calibrated = False

    @torch.no_grad()
    def calibrate(self, x_max):
        x_max = x_max.clamp(min=1e-5).float()
        w_max = self.weight.abs().max(dim=0)[0].clamp(min=1e-5).float()
        smooth = (w_max.pow(1 - self.alpha) / x_max.pow(self.alpha)).clamp(1e-4, 1e4)
        self.smooth_scale.data = smooth.to(self.target_dtype)
        w_smoothed = self.weight.float() / smooth.view(1, -1)
        self.w_quantized.copy_(quantize_to_nvfp4(w_smoothed, self.block_size).to(self.target_dtype))
        self.is_calibrated = True

    def forward(self, x):
        input_dtype = x.dtype
        if not self.is_calibrated:
            return F.linear(x, self.weight.to(input_dtype),
                            self.bias.to(input_dtype) if self.bias is not None else None)
        x_smoothed = x.to(self.target_dtype) * self.smooth_scale
        x_q = quantize_to_nvfp4(x_smoothed, self.block_size)
        out = F.linear(x_q, self.w_quantized)
        if self.bias is not None:
            out = out + self.bias
        return out.to(input_dtype)


def apply_rtn_quantization(pipe, transformer, accelerator, prompts, p_count,
                            t_count, device, args):
    """M1: SmoothQuant + NVFP4 RTN 적용."""
    target_names = get_target_linear_names(transformer)
    if accelerator.is_main_process:
        print(f"  [RTN] Targeting {len(target_names)} layers")

    print(f"  [RTN] Calibrating ({p_count} prompts)...", flush=True)
    calib_data = _collect_x_max(pipe, transformer, target_names,
                                 prompts, p_count, t_count, device, accelerator)

    for name in tqdm(target_names, desc="[RTN] Replacing layers",
                     disable=not accelerator.is_main_process):
        orig_m = get_module_by_name(transformer, name)
        if next(orig_m.parameters()).device == device:
            new_layer = ManualRTNLinear(
                orig_m, alpha=args.alpha, block_size=16, dtype=torch.float16,
            ).to(device)
            if name in calib_data:
                new_layer.calibrate(calib_data[name])
            set_module_by_name(transformer, name, new_layer)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("  [RTN] Quantization complete.")


# ---------------------------------------------------------------------------
# M2: SVDQuant — 메인 스크립트에서 직접 mtq.quantize 호출
# ---------------------------------------------------------------------------

def apply_svdquant_quantization(pipe, accelerator, prompts, p_count, t_count, device, args):
    """M2: mtq.NVFP4_SVDQUANT_DEFAULT_CFG 적용."""
    import modelopt.torch.quantization as mtq
    quant_config = copy.deepcopy(mtq.NVFP4_SVDQUANT_DEFAULT_CFG)
    if "algorithm" in quant_config:
        quant_config["algorithm"]["lowrank"] = args.lowrank

    if accelerator.is_main_process:
        print(f"  [SVDQuant] Applying mtq.NVFP4_SVDQUANT_DEFAULT_CFG (lowrank={args.lowrank})...")

    def forward_loop(model):
        for prompt in prompts[:p_count]:
            pipe(prompt, num_inference_steps=5,
                 generator=torch.Generator(device=device).manual_seed(42))

    with torch.no_grad():
        pipe.transformer = mtq.quantize(pipe.transformer, quant_config,
                                         forward_loop=forward_loop)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("  [SVDQuant] Quantization complete.")


# ---------------------------------------------------------------------------
# M3: MRGPTQLinear (Micro-Rotation NVFP4, ICLR 2026)
# ---------------------------------------------------------------------------

class MRGPTQLinear(nn.Module):
    """
    MR-GPTQ: Micro-Rotation NVFP4 양자화 (ICLR 2026).

    핵심 insight: NVFP4 group size(16)에서 SmoothQuant는 무효
      - SmoothQuant는 layer-wide smooth scale을 학습하지만,
        NVFP4는 이미 per-group(16) scale을 갖고 있어 benefit이 없음
      - 대신 H(16) Hadamard를 각 16-element group에 적용 →
        group 내부 outlier 분산, 분포 등방화

    구현:
      1. act_order: H_diag(=E[x²]) 기준 채널 내림차순 정렬
         → 중요 채널이 동일 NVFP4 group에 모임 → group scale 낭비 최소화
      2. H(16) Micro-Rotation: 각 16-element group에 H(16) 적용 (weight & activation)
      3. per-group NVFP4 quantization (group_size=16, SmoothQuant 없음)
      4. SVD error correction (rank=32)
    """

    def __init__(self, original_linear, rank=32, dtype=torch.float16):
        super().__init__()
        self.rank = rank
        self.target_dtype = dtype
        self.in_features  = original_linear.in_features
        self.out_features = original_linear.out_features
        self.block_size   = 16  # NVFP4 group size, fixed
        self.use_rotation = (self.in_features % self.block_size == 0)

        self.register_buffer("weight", original_linear.weight.data.clone().to(dtype))
        self.bias = nn.Parameter(
            original_linear.bias.data.clone().to(dtype)
        ) if original_linear.bias is not None else None

        # H(16) Hadamard matrix (normalized, orthogonal: H @ H.T = I)
        if self.use_rotation:
            from scipy.linalg import hadamard as scipy_hadamard
            import numpy as np
            H_np = scipy_hadamard(16) / (16 ** 0.5)
            self.register_buffer("H16", torch.from_numpy(H_np.astype(np.float32)).to(dtype))
        else:
            self.H16 = None

        # act_order permutation (set during calibration)
        self.register_buffer("perm",     torch.arange(self.in_features))
        self.register_buffer("inv_perm", torch.arange(self.in_features))
        self.register_buffer("w_quantized", original_linear.weight.data.clone().to(dtype))
        self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank, dtype=dtype))
        self.is_calibrated = False

    def _apply_h16(self, x):
        """Apply H(16) to each 16-element group in the last dimension."""
        shape = x.shape
        n_blocks = shape[-1] // self.block_size
        x_3d = x.reshape(*shape[:-1], n_blocks, self.block_size)
        return (x_3d @ self.H16.to(x.dtype)).reshape(shape)

    @torch.no_grad()
    def calibrate(self, h_diag):
        """
        h_diag: per-channel E[x²] tensor, shape [in_features].
        No x_max needed — SmoothQuant is not used.
        """
        device  = self.weight.device
        h_diag  = h_diag.float().to(device).clamp(min=1e-8)
        W       = self.weight.float()
        in_f    = self.in_features

        # Step 1: act_order — sort channels by importance (H_diag descending)
        perm     = torch.argsort(h_diag, descending=True)
        inv_perm = torch.argsort(perm)
        self.perm.data     = perm
        self.inv_perm.data = inv_perm
        W_reordered = W[:, perm]

        # Step 2: H(16) Micro-Rotation per group
        if self.use_rotation:
            W_rotated = self._apply_h16(W_reordered)
        else:
            W_rotated = W_reordered

        # Step 3: per-group NVFP4 quantization (group_size=16, no SmoothQuant)
        W_q = quantize_to_nvfp4(W_rotated, block_size=self.block_size)
        self.w_quantized.copy_(W_q.to(self.target_dtype))

        # Step 4: SVD error correction (in rotated space)
        W_err = W_rotated - W_q
        U, S, Vh = torch.linalg.svd(W_err.float(), full_matrices=False)
        r       = min(self.rank, S.shape[0])
        sqrt_S  = S[:r].sqrt()

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
            return F.linear(x, self.weight.to(input_dtype),
                            self.bias.to(input_dtype) if self.bias is not None else None)

        x_t = x.to(self.target_dtype)

        # Step 1: Reorder channels (act_order)
        x_perm = x_t[..., self.perm]

        # Step 2: H(16) Micro-Rotation
        x_rot = self._apply_h16(x_perm) if self.use_rotation else x_perm

        # Step 3: NVFP4 quantize
        x_q = quantize_to_nvfp4(x_rot, block_size=self.block_size)

        # Step 4: base (quantized) + SVD correction (in rotated space)
        # Note: W_q and lora_a are both in rotated+reordered space → outputs are correct
        out = F.linear(x_q, self.w_quantized) + F.linear(
            F.linear(x_rot, self.lora_a), self.lora_b
        )
        if self.bias is not None:
            out = out + self.bias
        return out.to(input_dtype)


def apply_mrgptq_quantization(pipe, transformer, accelerator, prompts, p_count,
                               t_count, device, args):
    """M3: MR-GPTQ Micro-Rotation NVFP4 적용 (SmoothQuant 없음, H(16) per group)."""
    target_names = get_target_linear_names(transformer)
    if accelerator.is_main_process:
        print(f"  [MRGPTQ] Targeting {len(target_names)} layers")

    print(f"  [MRGPTQ] Calibrating ({p_count} prompts, collecting H_diag)...", flush=True)
    calib_data = _collect_hdiag(
        pipe, transformer, target_names, prompts, p_count, t_count, device, accelerator
    )

    for name in tqdm(target_names, desc="[MRGPTQ] Replacing layers",
                     disable=not accelerator.is_main_process):
        orig_m = get_module_by_name(transformer, name)
        if next(orig_m.parameters()).device == device:
            new_layer = MRGPTQLinear(
                orig_m, rank=args.lowrank, dtype=torch.float16,
            ).to(device)
            if name in calib_data:
                new_layer.calibrate(calib_data[name])
            set_module_by_name(transformer, name, new_layer)

    del calib_data
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("  [MRGPTQ] Quantization complete.")


# ---------------------------------------------------------------------------
# M4: FourOverSixLinear (Adaptive block scaling NVFP4)
# ---------------------------------------------------------------------------

class FourOverSixLinear(nn.Module):
    """
    Four Over Six adaptive block scaling NVFP4 (arXiv 2512.02010).

    NVFP4 levels [0, 0.5, 1, 1.5, 2, 3, 4, 6] have step sizes [0.5×4, 1×2, 2×1].
    The step size 2 between levels 4 and 6 causes large quantization error for
    values near 5 (in normalized space). Fixing this without changing the format:

    Per 16-element block, choose between:
      - max=6 (standard): scale = amax/6, all 8 levels available
      - max=4 (adaptive): scale = amax/4, only levels ≤4 in practice
        (values in [0, amax] map to normalized [0, 4], avoiding the step-2 region)

    Selection criterion: pick whichever gives lower MSE per block.

    Implementation: SmoothQuant (for activation scaling) + Four-Over-Six NVFP4.
    No SVD correction — the contribution here is purely the better quantizer.
    """

    def __init__(self, original_linear, alpha=0.5, block_size=16, dtype=torch.float16):
        super().__init__()
        self.target_dtype = dtype
        self.alpha = alpha
        self.block_size = block_size

        self.register_buffer("weight", original_linear.weight.data.clone().to(dtype))
        self.bias = nn.Parameter(
            original_linear.bias.data.clone().to(dtype)
        ) if original_linear.bias is not None else None

        self.register_buffer("w_quantized", original_linear.weight.data.clone().to(dtype))
        self.register_buffer("smooth_scale",
                             torch.ones(original_linear.in_features, dtype=dtype))
        self.is_calibrated = False

    @torch.no_grad()
    def calibrate(self, x_max):
        x_max = x_max.clamp(min=1e-5).float()
        w_max = self.weight.abs().max(dim=0)[0].clamp(min=1e-5).float()
        smooth = (w_max.pow(1 - self.alpha) / x_max.pow(self.alpha)).clamp(1e-4, 1e4)
        self.smooth_scale.data = smooth.to(self.target_dtype)
        w_smoothed = self.weight.float() / smooth.view(1, -1)
        self.w_quantized.copy_(
            quantize_to_nvfp4_fouroversix(w_smoothed, self.block_size).to(self.target_dtype)
        )
        self.is_calibrated = True

    def forward(self, x):
        input_dtype = x.dtype
        if not self.is_calibrated:
            return F.linear(x, self.weight.to(input_dtype),
                            self.bias.to(input_dtype) if self.bias is not None else None)
        x_smoothed = x.to(self.target_dtype) * self.smooth_scale
        x_q = quantize_to_nvfp4_fouroversix(x_smoothed, self.block_size)
        out = F.linear(x_q, self.w_quantized)
        if self.bias is not None:
            out = out + self.bias
        return out.to(input_dtype)


def apply_fouroversix_quantization(pipe, transformer, accelerator, prompts, p_count,
                                    t_count, device, args):
    """M4: Four Over Six adaptive block scaling NVFP4 적용."""
    target_names = get_target_linear_names(transformer)
    if accelerator.is_main_process:
        print(f"  [FOUROVERSIX] Targeting {len(target_names)} layers")

    print(f"  [FOUROVERSIX] Calibrating ({p_count} prompts)...", flush=True)
    calib_data = _collect_x_max(pipe, transformer, target_names,
                                  prompts, p_count, t_count, device, accelerator)

    for name in tqdm(target_names, desc="[FOUROVERSIX] Replacing layers",
                     disable=not accelerator.is_main_process):
        orig_m = get_module_by_name(transformer, name)
        if next(orig_m.parameters()).device == device:
            new_layer = FourOverSixLinear(
                orig_m, alpha=args.alpha, block_size=16, dtype=torch.float16,
            ).to(device)
            if name in calib_data:
                new_layer.calibrate(calib_data[name])
            set_module_by_name(transformer, name, new_layer)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("  [FOUROVERSIX] Quantization complete.")
