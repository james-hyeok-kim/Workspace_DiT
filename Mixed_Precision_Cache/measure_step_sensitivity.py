"""
measure_step_sensitivity.py — Per-(layer_type, step) sensitivity measurement.

Methodology (Metric A, ground-truth):
  Apply all-W4 StepAwareMixedLinear once (both W3/W4 buffers loaded).
  Teacher: run all-W4 schedule for n_calib prompts → z_ref latents.
  For each (layer_type L, step t) pair:
    Mutate shared bit_schedule[(L, t)] = "W3"
    Run n_calib prompts → z_Lt latents
    mse(L, t) = mean((z_Lt - z_ref)^2)
    Restore bit_schedule[(L, t)] = "W4"
  No pipeline reload needed — StepAwareMixedLinear reads dict at runtime.

Output: results/step_sensitivity.json with:
  - metric_A: {"attn1_qkv_step0": 0.000123, ...}  (up to 70 entries)
  - ranking_A: [[layer_type, step], ...] sorted low→high sensitivity
  - config: experiment settings

Runtime estimate:
  70 pairs × n_calib prompts × 10 steps × 1.44s/step ≈ 70 × 4 × 14.4s ≈ ~67 min
  (sensitivity loop only — no pipeline reloads)

Usage:
  python measure_step_sensitivity.py --num_steps 10 --n_calib 4
  python measure_step_sensitivity.py --num_steps 10 --n_calib 4 --gpu 0
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import json
import argparse

import torch
from diffusers import PixArtSigmaPipeline, DPMSolverMultistepScheduler

from eval_utils import get_prompts
from deepcache_utils import DeepCacheState, install_step_aware_quant
from quant_methods import (
    apply_step_aware_mixed_quantization,
    LAYER_TYPES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pipeline(model_path: str, device: str) -> PixArtSigmaPipeline:
    pipe = PixArtSigmaPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _run_latents(pipe, prompt: str, t_count: int, guidance_scale: float,
                 seed: int, device: str) -> torch.Tensor:
    """Run inference and return raw latents (before VAE decode). Shape: [1, C, H, W]."""
    gen = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        out = pipe(
            prompt,
            num_inference_steps=t_count,
            guidance_scale=guidance_scale,
            generator=gen,
            output_type="latent",
        )
    return out.images  # [1, C, H, W]


def _count_linears_per_type(transformer) -> dict:
    """Diagnostic: count nn.Linear layers per classify_layer_type label."""
    from quant_methods import classify_layer_type
    import torch.nn as nn
    counts = {lt: 0 for lt in LAYER_TYPES}
    counts["None"] = 0
    for b_idx, block in enumerate(transformer.transformer_blocks):
        for name, mod in block.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            full_name = f"transformer_blocks.{b_idx}.{name}"
            lt = classify_layer_type(full_name)
            key = lt if lt is not None else "None"
            counts[key] = counts.get(key, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Metric A: per-(layer_type, step) ablation via in-place dict mutation
# ---------------------------------------------------------------------------

def measure_metric_a(
    pipe,
    prompts: list,
    n_calib: int,
    t_count: int,
    guidance_scale: float,
    calib_seed_offset: int,
    device: str,
    output_path: str | None = None,
    resume: bool = True,
) -> dict:
    """
    For each of up to 70 (layer_type, step) pairs, perturb to W3 via in-place dict
    mutation, measure latent MSE vs all-W4.  No pipeline reloads.

    Steps:
      1. Build all-W4 schedule dict.
      2. Apply StepAwareMixedLinear once (loads both W3/W4 buffers).
      3. Install step counter once.
      4. Teacher pass: run n_calib prompts with all-W4 → z_ref.
      5. Ablation loop: for each (lt, s), set schedule[(lt,s)]="W3", run, restore "W4".

    If output_path is set and resume=True, skips already-computed pairs (crash-safe).
    """
    # Determine active layer types (skip types with 0 Linear layers)
    linear_counts = _count_linears_per_type(pipe.transformer)
    active_lts = [lt for lt in LAYER_TYPES if linear_counts.get(lt, 0) > 0]
    skipped_lts = [lt for lt in LAYER_TYPES if linear_counts.get(lt, 0) == 0]

    print(f"  [Diagnostic] Linear count per type: {linear_counts}")
    if skipped_lts:
        print(f"  [Diagnostic] Skipping (0 Linears): {skipped_lts}")

    n_pairs = len(active_lts) * t_count

    # Load existing partial results
    partial = {}
    if resume and output_path and os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        partial = existing.get("metric_A", {})
        print(f"  [Metric A] Resuming: {len(partial)}/{n_pairs} already done.")

    # ── Build shared schedule dict (all-W4) ────────────────────────────────
    schedule = {(lt, s): "W4" for lt in LAYER_TYPES for s in range(t_count)}

    # ── Apply quantization once ─────────────────────────────────────────────
    print("  [Metric A] Applying StepAwareMixedLinear (all-W4, once)...")
    state = DeepCacheState()
    with torch.no_grad():
        apply_step_aware_mixed_quantization(pipe.transformer, schedule, state)
    install_step_aware_quant(pipe.transformer, state)

    # ── Teacher pass (all-W4) ───────────────────────────────────────────────
    print(f"  [Metric A] Teacher pass ({n_calib} prompts, all-W4)...")
    z_ref = []
    for p_idx in range(n_calib):
        state.reset()
        z = _run_latents(pipe, prompts[p_idx], t_count, guidance_scale,
                         calib_seed_offset + p_idx, device)
        z_ref.append(z.cpu())
    print(f"  [Metric A] Teacher done. z_ref shapes: {[z.shape for z in z_ref]}")

    # ── Ablation loop (in-place dict mutation) ──────────────────────────────
    metric_a = dict(partial)
    done = 0
    total = n_pairs - len(partial)

    for lt in active_lts:
        for s in range(t_count):
            key = f"{lt}_step{s}"
            if key in metric_a:
                continue  # already done (resume)

            # Perturb: set only this (lt, s) pair to W3
            schedule[(lt, s)] = "W3"

            mse_list = []
            for p_idx in range(n_calib):
                state.reset()
                z_p = _run_latents(pipe, prompts[p_idx], t_count, guidance_scale,
                                   calib_seed_offset + p_idx, device)
                mse = float(((z_p.cpu() - z_ref[p_idx]) ** 2).mean().item())
                mse_list.append(mse)

            # Restore
            schedule[(lt, s)] = "W4"

            metric_a[key] = float(sum(mse_list) / len(mse_list))
            done += 1

            if done % 5 == 0 or done == total:
                print(f"    [{done}/{total}] ({lt}, step={s}): "
                      f"MSE={metric_a[key]:.6f}", flush=True)

            # Crash-safe incremental save
            if output_path:
                _save_partial(output_path, metric_a, t_count)

    return metric_a


def _save_partial(path: str, metric_a: dict, num_steps: int):
    """Incremental save to JSON (crash-safe)."""
    ranking = _rank_metric_a(metric_a)
    data = {
        "metric_A": metric_a,
        "ranking_A": ranking,
        "_partial": True,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _rank_metric_a(metric_a: dict) -> list:
    """Sort (layer_type, step) pairs by MSE ascending (low = tolerant = W3-safe)."""
    pairs = []
    for key, mse in metric_a.items():
        lt, s = key.rsplit("_step", 1)
        pairs.append((lt, int(s), mse))
    pairs.sort(key=lambda x: x[2])
    return [[lt, s] for lt, s, _ in pairs]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-(layer_type, step) sensitivity measurement"
    )
    parser.add_argument("--model_path", type=str,
                        default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    parser.add_argument("--num_steps", type=int, default=10, choices=[5, 10, 15, 20])
    parser.add_argument("--n_calib", type=int, default=4,
                        help="Number of calibration prompts per perturbation")
    parser.add_argument("--calib_seed_offset", type=int, default=1000)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--dataset_name", type=str, default="MJHQ")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from partial results (default: True)")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(
        args.output_dir,
        f"step_sensitivity_steps{args.num_steps}_cal{args.n_calib}"
        f"_seed{args.calib_seed_offset}.json"
    )

    print(f"\n{'='*60}")
    print(f"  Step Sensitivity Measurement (Metric A)")
    print(f"  Steps: {args.num_steps}  n_calib: {args.n_calib}")
    print(f"  Layer types: {LAYER_TYPES}")
    print(f"  Max pairs: {len(LAYER_TYPES) * args.num_steps}")
    print(f"  GPU: {device}")
    print(f"  Output: {out_path}")
    print(f"{'='*60}\n")

    prompts = get_prompts(args.n_calib, args.dataset_name)[:args.n_calib]

    print("[Setup] Loading pipeline...")
    pipe = _load_pipeline(args.model_path, device)

    metric_a = measure_metric_a(
        pipe=pipe,
        prompts=prompts,
        n_calib=args.n_calib,
        t_count=args.num_steps,
        guidance_scale=args.guidance_scale,
        calib_seed_offset=args.calib_seed_offset,
        device=device,
        output_path=out_path,
        resume=args.resume,
    )

    ranking = _rank_metric_a(metric_a)

    result = {
        "config": {
            "num_steps": args.num_steps,
            "n_calib": args.n_calib,
            "calib_seed_offset": args.calib_seed_offset,
            "guidance_scale": args.guidance_scale,
            "layer_types": list(LAYER_TYPES),
        },
        "metric_A": metric_a,
        "ranking_A": ranking,
        "_partial": False,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Done] Saved to: {out_path}")

    n_actual = len(metric_a)
    print(f"\n  Top-10 tolerant (layer_type, step) pairs (of {n_actual}):")
    for lt, s in ranking[:10]:
        key = f"{lt}_step{s}"
        print(f"    ({lt}, step={s}): MSE={metric_a[key]:.6f}")

    print(f"\n  Bottom-5 sensitive pairs:")
    for lt, s in ranking[-5:]:
        key = f"{lt}_step{s}"
        print(f"    ({lt}, step={s}): MSE={metric_a[key]:.6f}")

    print(f"\n  Per-layer-type MSE average:")
    active_lts = [lt for lt in LAYER_TYPES
                  if any(f"{lt}_step{s}" in metric_a for s in range(args.num_steps))]
    for lt in active_lts:
        vals = [metric_a[f"{lt}_step{s}"]
                for s in range(args.num_steps) if f"{lt}_step{s}" in metric_a]
        if vals:
            print(f"    {lt:15s}: {sum(vals)/len(vals):.6f}")


if __name__ == "__main__":
    main()
