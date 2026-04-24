"""
memory_accounting.py — Analytical memory footprint for step-aware mixed precision.

Since W3/W4 are fake-quant (dequant → FP16), real VRAM is unchanged.
This module reports *analytical* weight storage assuming real packed formats:
  - W4 (NVFP4): 4 bits / param
  - W3 (INT3):  3 bits / param

Two reporting modes:
  baseline_mb   : all-W4 analytical storage (Σ numel × 4 / 8 / 1e6)
  schedule_mb   : mixed-precision analytical storage (avg over steps)

The "avg over steps" interpretation: the model weights are shared across all
denoising steps. Under a step-aware schedule, the effective bit-width for a
given layer is the *average* bits used across the 10 steps:
  eff_bits(layer) = (n_W3_steps × 3 + n_W4_steps × 4) / num_steps

This represents the hypothetical storage if we could pack weights in a
step-specific compressed format (e.g., storing W3 for tolerant steps only).
"""

import torch
import torch.nn as nn

from quant_methods import classify_layer_type, LAYER_TYPES, StepAwareMixedLinear


def compute_analytical_memory(transformer, bit_schedule: dict, num_steps: int) -> dict:
    """
    Compute analytical weight storage for transformer_blocks.

    Args:
        transformer  : PixArtTransformer2DModel (after apply_step_aware_mixed_quantization)
        bit_schedule : {(layer_type, step_idx) -> "W3" | "W4"}
        num_steps    : total denoising steps

    Returns dict with:
        baseline_mb    : MB if all layers were W4
        schedule_mb    : MB with given schedule (average bits per step)
        savings_pct    : (baseline - schedule) / baseline × 100
        per_type_mb    : dict[layer_type -> schedule MB contribution]
        dual_buffer_mb : MB if both W3+W4 are stored simultaneously (actual experiment VRAM)
    """
    # Count numel per layer_type across all blocks
    type_numel: dict[str, int] = {lt: 0 for lt in LAYER_TYPES}
    other_numel: int = 0

    for b_idx, block in enumerate(transformer.transformer_blocks):
        for name, mod in block.named_modules():
            if isinstance(mod, StepAwareMixedLinear):
                # Already replaced: use stored layer_type and feature dims
                lt = mod.layer_type
                numel = mod.in_features * mod.out_features
            elif isinstance(mod, nn.Linear):
                full_name = f"transformer_blocks.{b_idx}.{name}"
                lt = classify_layer_type(full_name)
                numel = mod.in_features * mod.out_features
            else:
                continue
            if lt in type_numel:
                type_numel[lt] += numel
            else:
                other_numel += numel

    # Per layer_type: compute average bits under schedule
    bits_per_byte = 8
    baseline_bits = 0
    schedule_bits = 0
    per_type_mb = {}
    dual_buffer_bits = 0

    for lt in LAYER_TYPES:
        numel = type_numel[lt]
        if numel == 0:
            per_type_mb[lt] = 0.0
            continue

        n_w3 = sum(
            1 for s in range(num_steps)
            if bit_schedule.get((lt, s), "W4") == "W3"
        )
        n_w4 = num_steps - n_w3
        eff_bits = (n_w3 * 3 + n_w4 * 4) / num_steps

        baseline_bits += numel * 4  # all W4
        schedule_bits += numel * eff_bits

        # Dual-buffer: always store both W3 (3-bit) + W4 (4-bit)
        dual_buffer_bits += numel * (3 + 4)

        per_type_mb[lt] = numel * eff_bits / bits_per_byte / 1e6

    # Other linears (proj_in, proj_out, etc.) treated as W4 in both cases
    baseline_bits  += other_numel * 4
    schedule_bits  += other_numel * 4
    dual_buffer_bits += other_numel * 4

    baseline_mb   = baseline_bits  / bits_per_byte / 1e6
    schedule_mb   = schedule_bits  / bits_per_byte / 1e6
    dual_buffer_mb = dual_buffer_bits / bits_per_byte / 1e6
    savings_pct   = (baseline_mb - schedule_mb) / baseline_mb * 100 if baseline_mb > 0 else 0.0

    return {
        "baseline_mb":   baseline_mb,
        "schedule_mb":   schedule_mb,
        "savings_pct":   savings_pct,
        "per_type_mb":   per_type_mb,
        "dual_buffer_mb": dual_buffer_mb,
        "type_numel":    {lt: type_numel[lt] for lt in LAYER_TYPES},
    }


def print_memory_report(mem: dict, tag: str = ""):
    """Pretty-print analytical memory report."""
    print(f"\n{'─'*50}")
    if tag:
        print(f"  Memory Report: {tag}")
    print(f"  Baseline (all-W4) : {mem['baseline_mb']:.1f} MB")
    print(f"  Schedule (mixed)  : {mem['schedule_mb']:.1f} MB  "
          f"(savings {mem['savings_pct']:.1f}%)")
    print(f"  Dual-buffer (exp) : {mem['dual_buffer_mb']:.1f} MB  "
          f"(actual experiment VRAM for fake-quant)")
    print(f"  Per layer-type breakdown:")
    for lt in LAYER_TYPES:
        numel = mem['type_numel'].get(lt, 0)
        mb = mem['per_type_mb'].get(lt, 0.0)
        print(f"    {lt:15s}: {numel/1e6:.1f}M params → {mb:.1f} MB")
    print(f"  Note: analytical only. Real packed W3 kernel required for actual VRAM savings.")
    print(f"{'─'*50}\n")
