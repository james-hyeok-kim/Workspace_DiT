"""
quant_methods.py — Step-aware Mixed Precision (W3/W4) for PixArt-Sigma.

Provides:
  - quantize_to_nvfp4 / quantize_uniform  : fake-quant primitives (from reference)
  - StepAwareMixedLinear                  : dual-buffer Linear, picks W3/W4 per step
  - classify_layer_type                   : assign one of 7 type labels to a named Linear
  - apply_step_aware_mixed_quantization   : replace transformer_block Linears in-place
  - apply_svdquant_quantization           : SVDQuant baseline (Phase 0 sanity)

W3/W4 are fake-quant (dequantized → FP16 F.linear).  Real VRAM is ~unchanged.
Memory savings are reported analytically via memory_accounting.py.
"""

import copy
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Common quantization primitives (copied from reference quant_methods.py)
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


def quantize_uniform(x, block_size=16, mode="INT3"):
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
# Layer-type classification
# ---------------------------------------------------------------------------

# 7 layer type labels used in bit_schedule keys
LAYER_TYPES = ("attn1_qkv", "attn1_out", "attn2_qkv", "attn2_out",
               "mlp_fc1", "mlp_fc2", "adaLN")


def classify_layer_type(module_name: str) -> str | None:
    """Map a dotted module name to one of LAYER_TYPES, or None to skip."""
    n = module_name
    if "attn1" in n:
        return "attn1_out" if "to_out" in n else "attn1_qkv"
    if "attn2" in n:
        return "attn2_out" if "to_out" in n else "attn2_qkv"
    if ".ff." in n or ".ff_context." in n:
        # GEGLU: net[0].proj is fc1, net[2] is fc2
        if "net.0" in n or (".proj" in n and "net.2" not in n):
            return "mlp_fc1"
        return "mlp_fc2"
    if "adaLN_modulation" in n or "scale_shift" in n or "adaln" in n.lower():
        return "adaLN"
    return None  # other Linears (e.g. proj_in, proj_out) — skip


# ---------------------------------------------------------------------------
# StepAwareMixedLinear
# ---------------------------------------------------------------------------

class StepAwareMixedLinear(nn.Module):
    """
    Holds pre-dequantized W4 (NVFP4) and W3 (INT3) weight buffers.
    At forward(), picks the buffer according to bit_schedule[(layer_type, step_idx)].

    step_idx is read from state.step_idx - 1 (transformer wrapper pre-increments).
    bit_schedule maps (layer_type_str, step_idx_int) -> "W3" | "W4".
    """

    def __init__(
        self,
        orig: nn.Linear,
        layer_type: str,
        bit_schedule: dict,
        state,        # DeepCacheState (or any object with step_idx attribute)
        block_size: int = 16,
    ):
        super().__init__()
        self.layer_type = layer_type
        self.bit_schedule = bit_schedule
        self.state = state
        self.in_features  = orig.in_features
        self.out_features = orig.out_features
        self._block_size  = block_size

        with torch.no_grad():
            W = orig.weight.data.float()
            # Pad if not divisible by block_size
            pad_in = (block_size - W.shape[1] % block_size) % block_size
            if pad_in:
                W = F.pad(W, (0, pad_in))

            w4 = quantize_to_nvfp4(W, block_size=block_size)
            w3 = quantize_uniform(W, block_size=block_size, mode="INT3")

            if pad_in:
                w4 = w4[:, :orig.weight.shape[1]]
                w3 = w3[:, :orig.weight.shape[1]]

        self.register_buffer("w4", w4.to(torch.float16))
        self.register_buffer("w3", w3.to(torch.float16))

        if orig.bias is not None:
            self.bias = nn.Parameter(orig.bias.data.clone().to(torch.float16))
        else:
            self.bias = None

    def _quantize_act(self, x: torch.Tensor, bits: str) -> torch.Tensor:
        """Block-wise activation quantization matching weight bit-width."""
        orig_shape = x.shape
        n = x.numel()
        pad = (self._block_size - n % self._block_size) % self._block_size
        x_flat = F.pad(x.reshape(-1).float(), (0, pad))
        if bits == "W3":
            x_q = quantize_uniform(x_flat, block_size=self._block_size, mode="INT3")
        else:
            x_q = quantize_to_nvfp4(x_flat, block_size=self._block_size)
        if pad:
            x_q = x_q.reshape(-1)[:n]
        return x_q.reshape(orig_shape).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        step = self.state.step_idx - 1  # 0-indexed current step
        key = (self.layer_type, max(step, 0))
        bits = self.bit_schedule.get(key, "W4")
        W = self.w3 if bits == "W3" else self.w4
        x_q = self._quantize_act(x, bits)   # W4→A4(NVFP4), W3→A3(INT3)
        return F.linear(x_q, W.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)


# ---------------------------------------------------------------------------
# Helper: iterate named modules within transformer blocks only
# ---------------------------------------------------------------------------

def _iter_block_linears(transformer):
    """Yield (full_name, block_idx, module, parent, attr) for Linear layers in transformer_blocks."""
    for b_idx, block in enumerate(transformer.transformer_blocks):
        for name, mod in block.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            # Find parent module and attribute name
            parts = name.split(".")
            parent = block
            for part in parts[:-1]:
                parent = getattr(parent, part)
            full_name = f"transformer_blocks.{b_idx}.{name}"
            yield full_name, b_idx, mod, parent, parts[-1]


# ---------------------------------------------------------------------------
# apply_step_aware_mixed_quantization
# ---------------------------------------------------------------------------

def apply_step_aware_mixed_quantization(transformer, bit_schedule, state, block_size=16):
    """
    Replace Linear layers in transformer_blocks with StepAwareMixedLinear in-place.

    bit_schedule: dict[(layer_type_str, step_idx_int) -> "W3" | "W4"]
    state: object with step_idx attribute (DeepCacheState or StepState).

    Returns mapping {full_name: layer_type} for all replaced layers.
    """
    replaced = {}
    skipped = []

    for full_name, b_idx, mod, parent, attr in list(_iter_block_linears(transformer)):
        layer_type = classify_layer_type(full_name)
        if layer_type is None:
            skipped.append(full_name)
            continue
        new_layer = StepAwareMixedLinear(mod, layer_type, bit_schedule, state, block_size)
        setattr(parent, attr, new_layer)
        replaced[full_name] = layer_type

    print(f"  [StepAware] Replaced {len(replaced)} Linear layers across "
          f"{len(transformer.transformer_blocks)} blocks.")
    if skipped:
        print(f"  [StepAware] Skipped {len(skipped)} Linears (type=None): "
              f"{skipped[:3]}{'...' if len(skipped) > 3 else ''}")
    return replaced


# ---------------------------------------------------------------------------
# SVDQuant baseline (Phase 0 sanity check)
# ---------------------------------------------------------------------------

def apply_svdquant_quantization(pipe, accelerator, prompts, p_count, t_count, device, args):
    """M2: mtq.NVFP4_SVDQUANT_DEFAULT_CFG. Reproduces reference baseline FID≈121.9 @ 10 steps."""
    import modelopt.torch.quantization as mtq
    quant_config = copy.deepcopy(mtq.NVFP4_SVDQUANT_DEFAULT_CFG)
    if "algorithm" in quant_config:
        quant_config["algorithm"]["lowrank"] = getattr(args, "lowrank", 32)

    if accelerator.is_main_process:
        print(f"  [SVDQuant] Applying NVFP4_SVDQUANT_DEFAULT_CFG "
              f"(lowrank={getattr(args, 'lowrank', 32)})...")

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
