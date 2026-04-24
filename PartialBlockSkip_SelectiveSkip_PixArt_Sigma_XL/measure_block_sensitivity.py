"""
measure_block_sensitivity.py

Per-block sensitivity measurement for selective skip experiment.
Produces sensitivity.json used by pixart_nvfp4_cache_compare.py --cache_mode selective_skip.

Metric A: Per-block skip ablation (ground truth)
  - For each block b in [0, 28): install selective skip {b}, run 4 prompts,
    compute MSE(z_b, z_ref) in latent space.  28 * 4 runs ≈ 6 min.

Metric C: Stale-vs-fresh residual diff (proxy, fast)
  - Single no-cache pass, hook every block to record r_i(t) = h_out_i - h_in_i.
  - Simulate interval=2 pattern: compare r(t_cached) vs r(t_last_fresh) per block.
  - 4 prompts * 20 steps in one forward sweep ≈ 2 min.

Output:
  /data/jameskimh/james_dit_pixart_sigma_xl_mjhq/SVDQUANT/MJHQ/sensitivity/
    SVDQUANT_steps{N}_cal{C}_seed{S}.json
"""

import os
import sys
import json
import argparse
import types
import gc

import numpy as np
import torch
from diffusers import PixArtSigmaPipeline, DPMSolverMultistepScheduler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_utils import get_prompts
from deepcache_utils import install_selective_skip


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pipeline(model_path: str, device: str):
    pipe = PixArtSigmaPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def _apply_svdquant(pipe, prompts, p_count, t_count, device, lowrank=32):
    import copy
    import modelopt.torch.quantization as mtq
    quant_config = copy.deepcopy(mtq.NVFP4_SVDQUANT_DEFAULT_CFG)
    if "algorithm" in quant_config:
        quant_config["algorithm"]["lowrank"] = lowrank

    def forward_loop(model):
        for prompt in prompts[:p_count]:
            pipe(prompt, num_inference_steps=5,
                 generator=torch.Generator(device=device).manual_seed(42))

    with torch.no_grad():
        pipe.transformer = mtq.quantize(pipe.transformer, quant_config,
                                        forward_loop=forward_loop)
    print("  [SVDQuant] Quantization complete.")


def _uninstall_selective(transformer, state, orig_tf_forward):
    """Undo install_selective_skip: restore patched blocks and transformer forward."""
    for i in state.skip_blocks:
        block = transformer.transformer_blocks[i]
        if "forward" in block.__dict__:
            del block.__dict__["forward"]
    transformer.forward = orig_tf_forward


def _run_latents(pipe, prompt, t_count, guidance_scale, seed, device):
    """Run inference and return raw latents (before VAE decode)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        out = pipe(
            prompt,
            num_inference_steps=t_count,
            guidance_scale=guidance_scale,
            generator=gen,
            output_type="latent",
        )
    return out.images  # Tensor [1, C, H, W] in latent space


# ---------------------------------------------------------------------------
# Metric C: stale-vs-fresh residual diff
# ---------------------------------------------------------------------------

def measure_metric_c(pipe, prompts, n_calib, t_count, guidance_scale,
                     deepcache_interval, calib_seed_offset, device):
    """Hook every block, simulate interval-2 stale pattern, compute per-block error."""
    transformer = pipe.transformer
    n_blocks = len(transformer.transformer_blocks)
    print(f"  [Metric C] Hooking {n_blocks} blocks, {n_calib} prompts × {t_count} steps...")

    # Per-block accumulators
    metric_c = {i: [] for i in range(n_blocks)}

    for p_idx in range(n_calib):
        prompt = prompts[p_idx]
        seed = calib_seed_offset + p_idx

        # Per-block running buffers for this prompt
        h_in_buf = {}    # block_i → h_in tensor at current step
        r_last_fresh = {}  # block_i → residual at last fresh step

        step_counter = [0]

        # Register hooks
        handles = []

        def make_pre_hook(b_idx):
            def pre_hook(module, inputs):
                h_in_buf[b_idx] = inputs[0].detach()
            return pre_hook

        def make_post_hook(b_idx):
            def post_hook(module, inputs, output):
                if b_idx not in h_in_buf:
                    return
                h_in = h_in_buf[b_idx]
                h_out = output if isinstance(output, torch.Tensor) else output[0]
                residual = (h_out - h_in).detach()
                t = step_counter[0]
                # Simulate interval=2: step 0 is always fresh (full_steps_set={0})
                # Even steps (0,2,4,...) = fresh; odd steps (1,3,5,...) = cached
                if t == 0 or t % deepcache_interval == 0:
                    r_last_fresh[b_idx] = residual
                else:
                    if b_idx in r_last_fresh:
                        stale = r_last_fresh[b_idx]
                        norm_fresh = residual.norm().item()
                        rel = (residual - stale).norm().item() / (norm_fresh + 1e-8)
                        metric_c[b_idx].append(rel)
            return post_hook

        for b_idx, block in enumerate(transformer.transformer_blocks):
            h = block.register_forward_pre_hook(make_pre_hook(b_idx))
            handles.append(h)
            h = block.register_forward_hook(make_post_hook(b_idx))
            handles.append(h)

        # Step counter via scheduler callback
        orig_step = transformer.forward.__func__ if hasattr(transformer.forward, "__func__") else None

        # We need to count denoising steps. Use a callback on the pipeline.
        def _step_cb(pipe_, step, timestep, kwargs):
            step_counter[0] = step
            return kwargs

        gen = torch.Generator(device=device).manual_seed(seed)
        with torch.no_grad():
            pipe(
                prompt,
                num_inference_steps=t_count,
                guidance_scale=guidance_scale,
                generator=gen,
                callback_on_step_end=_step_cb,
                output_type="latent",
            )

        for h in handles:
            h.remove()
        h_in_buf.clear()
        r_last_fresh.clear()
        step_counter[0] = 0
        gc.collect()
        torch.cuda.empty_cache()
        print(f"    prompt {p_idx+1}/{n_calib} done")

    result = {}
    for i in range(n_blocks):
        vals = metric_c[i]
        result[str(i)] = float(np.mean(vals)) if vals else 0.0
    return result


# ---------------------------------------------------------------------------
# Metric A: per-block skip ablation
# ---------------------------------------------------------------------------

def measure_metric_a(pipe, prompts, n_calib, t_count, guidance_scale,
                     deepcache_interval, calib_seed_offset, device):
    """For each block b, install skip={b}, run prompts, compute MSE vs teacher."""
    transformer = pipe.transformer
    n_blocks = len(transformer.transformer_blocks)

    print(f"  [Metric A] Teacher pass ({n_calib} prompts)...")

    # Teacher pass: no cache
    z_ref = []
    for p_idx in range(n_calib):
        z = _run_latents(pipe, prompts[p_idx], t_count, guidance_scale,
                         calib_seed_offset + p_idx, device)
        z_ref.append(z.cpu())
        gc.collect()
        torch.cuda.empty_cache()

    orig_tf_forward = transformer.forward  # save before any patching

    metric_a = {}
    for b in range(n_blocks):
        state = install_selective_skip(
            transformer,
            skip_blocks={b},
            cache_interval=deepcache_interval,
            num_full_steps=1,
        )
        mse_list = []
        for p_idx in range(n_calib):
            state.reset()
            z_b = _run_latents(pipe, prompts[p_idx], t_count, guidance_scale,
                               calib_seed_offset + p_idx, device)
            mse = float(((z_b.cpu() - z_ref[p_idx]) ** 2).mean().item())
            mse_list.append(mse)
        metric_a[str(b)] = float(np.mean(mse_list))

        _uninstall_selective(transformer, state, orig_tf_forward)
        gc.collect()
        torch.cuda.empty_cache()

        if (b + 1) % 4 == 0 or b == n_blocks - 1:
            print(f"    block {b+1}/{n_blocks}: MSE={metric_a[str(b)]:.6f}")

    return metric_a


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Block sensitivity measurement for selective skip")
    parser.add_argument("--quant_method", type=str, default="SVDQUANT",
                        choices=["SVDQUANT"], help="Quantization method (only SVDQUANT supported)")
    parser.add_argument("--model_path", type=str,
                        default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    parser.add_argument("--num_steps", type=int, default=20,
                        choices=[5, 10, 15, 20])
    parser.add_argument("--n_calib", type=int, default=4,
                        help="Number of calibration prompts")
    parser.add_argument("--calib_seed_offset", type=int, default=1000)
    parser.add_argument("--deepcache_interval", type=int, default=2)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--dataset_name", type=str, default="MJHQ")
    parser.add_argument("--lowrank", type=int, default=32)
    parser.add_argument("--metric", type=str, default="both",
                        choices=["A", "C", "both"])
    parser.add_argument("--output_dir", type=str,
                        default="/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/SVDQUANT/MJHQ/sensitivity")
    args = parser.parse_args()

    device = "cuda"
    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(
        args.output_dir,
        f"{args.quant_method}_steps{args.num_steps}_cal{args.n_calib}_seed{args.calib_seed_offset}.json"
    )
    if os.path.exists(out_path):
        print(f"[Skip] Sensitivity JSON already exists: {out_path}")
        with open(out_path) as f:
            print(json.dumps(json.load(f), indent=2)[:2000])
        return

    print(f"\n{'='*60}")
    print(f"  Block Sensitivity Measurement")
    print(f"  Quant: {args.quant_method}  Steps: {args.num_steps}")
    print(f"  n_calib: {args.n_calib}  seed_offset: {args.calib_seed_offset}")
    print(f"  Interval: {args.deepcache_interval}  Metric: {args.metric}")
    print(f"{'='*60}\n")

    prompts = get_prompts(args.n_calib, args.dataset_name)[:args.n_calib]

    # Load + quantize pipeline
    print("[Setup] Loading pipeline...")
    pipe = _load_pipeline(args.model_path, device)
    print("[Setup] Applying SVDQUANT quantization...")
    _apply_svdquant(pipe, prompts, len(prompts), args.num_steps, device, args.lowrank)
    pipe.set_progress_bar_config(disable=True)

    result = {
        "config": {
            "quant_method": args.quant_method,
            "num_steps": args.num_steps,
            "n_calib": args.n_calib,
            "calib_seed_offset": args.calib_seed_offset,
            "deepcache_interval": args.deepcache_interval,
            "guidance_scale": args.guidance_scale,
        }
    }

    # ── Metric C (fast) ──────────────────────────────────────────────────────
    if args.metric in ("C", "both"):
        print("\n[Metric C] Stale-vs-fresh residual diff...")
        result["metric_C"] = measure_metric_c(
            pipe, prompts, args.n_calib, args.num_steps, args.guidance_scale,
            args.deepcache_interval, args.calib_seed_offset, device,
        )
        result["ranking_C"] = sorted(
            range(len(pipe.transformer.transformer_blocks)),
            key=lambda i: result["metric_C"][str(i)]
        )
        print(f"  Ranking C (low→high): {result['ranking_C']}")

    # ── Metric A (ground truth) ──────────────────────────────────────────────
    if args.metric in ("A", "both"):
        print("\n[Metric A] Per-block skip ablation...")
        result["metric_A"] = measure_metric_a(
            pipe, prompts, args.n_calib, args.num_steps, args.guidance_scale,
            args.deepcache_interval, args.calib_seed_offset, device,
        )
        result["ranking_A"] = sorted(
            range(len(pipe.transformer.transformer_blocks)),
            key=lambda i: result["metric_A"][str(i)]
        )
        print(f"  Ranking A (low→high): {result['ranking_A']}")

    # ── Kendall tau comparison ───────────────────────────────────────────────
    if args.metric == "both" and "ranking_A" in result and "ranking_C" in result:
        from scipy.stats import kendalltau
        tau, _ = kendalltau(result["ranking_A"], result["ranking_C"])
        print(f"\n  Kendall tau (A vs C): {tau:.3f}")
        result["kendall_tau_A_vs_C"] = float(tau)

    # ── Save ─────────────────────────────────────────────────────────────────
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Done] Saved to: {out_path}")

    # Print top-10 skip candidates per metric
    n_blocks = len(pipe.transformer.transformer_blocks)
    if "ranking_A" in result:
        print(f"\n  Top-10 skip candidates (Metric A): {result['ranking_A'][:10]}")
    if "ranking_C" in result:
        print(f"  Top-10 skip candidates (Metric C): {result['ranking_C'][:10]}")


if __name__ == "__main__":
    main()
