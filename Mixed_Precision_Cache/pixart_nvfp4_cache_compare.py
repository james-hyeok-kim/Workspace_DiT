"""
pixart_nvfp4_cache_compare.py — Step-aware Mixed Precision (W3/W4) experiment.

실험 매트릭스:
  Phase 0: SVDQUANT      — baseline FID≈121.9 @ 10 steps
  Phase 1: STEP_AWARE_MIXED --low_bit_k 0     — all-W4 sanity (must match SVDQUANT FID ±5)
  Phase 1: STEP_AWARE_MIXED --low_bit_k 70    — all-W3 upper bound (quality cliff)
  Phase 3: STEP_AWARE_MIXED --schedule S1 --low_bit_k K  — K-sweep (K=10,20,30,40,50,60,70)
  Phase 4: STEP_AWARE_MIXED --schedule S2/S3             — step-uniform / layer-type ablations

CLI examples:
  # Phase 0 — SVDQuant baseline
  accelerate launch --num_processes 2 pixart_nvfp4_cache_compare.py \\
      --quant_method SVDQUANT --num_steps 10 --num_samples 100

  # Phase 1 — all-W4 sanity
  accelerate launch --num_processes 2 pixart_nvfp4_cache_compare.py \\
      --quant_method STEP_AWARE_MIXED --num_steps 10 --num_samples 100 \\
      --schedule_family S1 --low_bit_k 0

  # Phase 1 — all-W3 upper bound
  accelerate launch --num_processes 2 pixart_nvfp4_cache_compare.py \\
      --quant_method STEP_AWARE_MIXED --num_steps 10 --num_samples 100 \\
      --schedule_family S1 --low_bit_k 70

  # Phase 3 — K=30 S1 sweep
  accelerate launch --num_processes 2 pixart_nvfp4_cache_compare.py \\
      --quant_method STEP_AWARE_MIXED --num_steps 10 --num_samples 100 \\
      --schedule_family S1 --low_bit_k 30 \\
      --sensitivity_json results/step_sensitivity.json

  # smoke test (5 samples)
  accelerate launch --num_processes 2 pixart_nvfp4_cache_compare.py \\
      --quant_method STEP_AWARE_MIXED --num_steps 10 --num_samples 5 --test_run \\
      --schedule_family S1 --low_bit_k 20 \\
      --sensitivity_json results/step_sensitivity.json
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import csv
import json
import argparse

import torch
from accelerate import Accelerator
from diffusers import PixArtSigmaPipeline, DPMSolverMultistepScheduler
from transformers import CLIPModel, CLIPProcessor

from deepcache_utils import DeepCacheState, install_step_aware_quant
from eval_utils import get_prompts, generate_and_evaluate
from quant_methods import (
    apply_svdquant_quantization,
    apply_step_aware_mixed_quantization,
    LAYER_TYPES,
)
from memory_accounting import compute_analytical_memory


# ---------------------------------------------------------------------------
# Schedule builders
# ---------------------------------------------------------------------------

def build_schedule_s1(sensitivity_json: str, low_bit_k: int, num_steps: int) -> dict:
    """S1: top-K most tolerant (layer_type, step) pairs → W3, rest W4.

    Loads sensitivity.json produced by measure_step_sensitivity.py.
    If sensitivity_json is None or low_bit_k==0, returns all-W4 schedule.
    If low_bit_k >= total pairs (70), returns all-W3 schedule.
    """
    total_pairs = len(LAYER_TYPES) * num_steps
    schedule = {(lt, s): "W4" for lt in LAYER_TYPES for s in range(num_steps)}

    if low_bit_k == 0:
        return schedule  # all-W4

    if low_bit_k >= total_pairs:
        # all-W3 — no ranking needed
        for lt in LAYER_TYPES:
            for s in range(num_steps):
                schedule[(lt, s)] = "W3"
        print(f"  [Schedule S1] K={low_bit_k} (≥total): all pairs → W3")
        return schedule

    if sensitivity_json is None:
        print("  [Schedule S1] Warning: sensitivity_json not provided. Using all-W4.")
        return schedule

    if not os.path.exists(sensitivity_json):
        raise FileNotFoundError(
            f"Sensitivity JSON not found: {sensitivity_json}\n"
            f"Run measure_step_sensitivity.py first."
        )

    with open(sensitivity_json) as f:
        data = json.load(f)

    # ranking_A: list of (layer_type, step) pairs sorted low→high sensitivity
    ranking = data.get("ranking_A", [])  # list of [layer_type, step] or flat ordering

    if not ranking:
        print("  [Schedule] Warning: ranking_A empty in sensitivity JSON. Using all-W4.")
        return schedule

    # Assign W3 to top-K least sensitive pairs
    k = min(low_bit_k, total_pairs)
    for entry in ranking[:k]:
        lt, s = entry[0], int(entry[1])
        if (lt, s) in schedule:
            schedule[(lt, s)] = "W3"

    n_w3 = sum(1 for v in schedule.values() if v == "W3")
    print(f"  [Schedule S1] K={k}: {n_w3}/{total_pairs} (layer_type, step) pairs → W3")
    return schedule


def build_schedule_s2(sensitivity_json: str, low_bit_k: int, num_steps: int) -> dict:
    """S2: step-uniform — K most tolerant *steps* (averaged across layer types) → all-W3.

    Validates the 'sensitive steps are early/late' intuition.
    """
    schedule = {(lt, s): "W4" for lt in LAYER_TYPES for s in range(num_steps)}

    if low_bit_k == 0:
        return schedule

    if sensitivity_json is None or not os.path.exists(sensitivity_json):
        # Fallback: assign W3 to last K steps (heuristic: later steps less critical)
        steps_sorted = list(range(num_steps - 1, -1, -1))
    else:
        with open(sensitivity_json) as f:
            data = json.load(f)
        mse = data.get("metric_A", {})
        # Average MSE per step across layer types
        step_mse = {}
        for key_str, val in mse.items():
            lt, s = key_str.rsplit("_step", 1)
            s = int(s)
            step_mse.setdefault(s, []).append(val)
        step_avg = {s: sum(v) / len(v) for s, v in step_mse.items()}
        steps_sorted = sorted(step_avg, key=lambda s: step_avg[s])  # low→high

    k = min(low_bit_k, num_steps)
    w3_steps = set(steps_sorted[:k])
    for lt in LAYER_TYPES:
        for s in w3_steps:
            schedule[(lt, s)] = "W3"

    print(f"  [Schedule S2] K={k} steps → W3: {sorted(w3_steps)}")
    return schedule


def build_schedule_s3(sensitivity_json: str, low_bit_k: int, num_steps: int) -> dict:
    """S3: layer-type-uniform — K most tolerant *layer types* (averaged across steps) → all-W3."""
    schedule = {(lt, s): "W4" for lt in LAYER_TYPES for s in range(num_steps)}

    if low_bit_k == 0:
        return schedule

    if sensitivity_json is None or not os.path.exists(sensitivity_json):
        # Fallback: assign W3 to mlp types (heuristic)
        lt_sorted = list(LAYER_TYPES)
    else:
        with open(sensitivity_json) as f:
            data = json.load(f)
        mse = data.get("metric_A", {})
        lt_mse = {}
        for key_str, val in mse.items():
            lt, s = key_str.rsplit("_step", 1)
            lt_mse.setdefault(lt, []).append(val)
        lt_avg = {lt: sum(v) / len(v) for lt, v in lt_mse.items()}
        lt_sorted = sorted(lt_avg, key=lambda lt: lt_avg.get(lt, 1e9))

    k = min(low_bit_k, len(LAYER_TYPES))
    w3_types = set(lt_sorted[:k])
    for lt in w3_types:
        for s in range(num_steps):
            schedule[(lt, s)] = "W3"

    print(f"  [Schedule S3] K={k} layer types → W3: {sorted(w3_types)}")
    return schedule


def build_schedule_direct(bit_schedule_json: str, num_steps: int) -> dict:
    """Direct schedule from JSON: {\"attn1_qkv_step0\": \"W3\", ...} or nested."""
    with open(bit_schedule_json) as f:
        raw = json.load(f)
    schedule = {(lt, s): "W4" for lt in LAYER_TYPES for s in range(num_steps)}
    for key, val in raw.items():
        # Support key formats: "attn1_qkv_step3", "(attn1_qkv, 3)"
        if "_step" in key:
            lt, s = key.rsplit("_step", 1)
            schedule[(lt, int(s))] = val
        elif key.startswith("("):
            lt, s = key.strip("()").split(", ")
            schedule[(lt.strip(), int(s))] = val
    return schedule


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step-aware Mixed Precision experiment")

    parser.add_argument("--quant_method", type=str, required=True,
                        choices=["SVDQUANT", "STEP_AWARE_MIXED"])
    parser.add_argument("--num_steps", type=int, default=10, choices=[5, 10, 15, 20])
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--model_path", type=str,
                        default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    parser.add_argument("--dataset_name", type=str, default="MJHQ")
    parser.add_argument("--lowrank", type=int, default=32,
                        help="SVDQuant LoRA rank (Phase 0 only)")

    # Step-aware schedule
    parser.add_argument("--schedule_family", type=str, default="S1",
                        choices=["S1", "S2", "S3", "direct"],
                        help="S1=top-K (layer,step) pairs W3 | S2=step-uniform | "
                             "S3=layer-type-uniform | direct=from JSON")
    parser.add_argument("--low_bit_k", type=int, default=0,
                        help="K pairs/steps/types → W3 (0=all-W4, 70=all-W3 for S1)")
    parser.add_argument("--sensitivity_json", type=str, default=None,
                        help="Output of measure_step_sensitivity.py")
    parser.add_argument("--bit_schedule_json", type=str, default=None,
                        help="Direct schedule JSON (schedule_family=direct)")

    # Output
    parser.add_argument("--output_base", type=str,
                        default="/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/SVDQUANT/MJHQ")
    parser.add_argument("--ref_base", type=str,
                        default="/data/jameskimh/james_dit_pixart_sigma_xl_mjhq")
    parser.add_argument("--test_run", action="store_true",
                        help="Smoke test: skip saving, don't write results CSV")

    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    # ── Config tag ──────────────────────────────────────────────────────────
    if args.quant_method == "SVDQUANT":
        tag = f"SVDQUANT_steps{args.num_steps}"
    else:
        tag = (f"STEP_AWARE_{args.schedule_family}_k{args.low_bit_k}"
               f"_steps{args.num_steps}")

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"  Step-aware Mixed Precision")
        print(f"  Method: {args.quant_method}  Tag: {tag}")
        print(f"  Steps: {args.num_steps}  Samples: {args.num_samples}")
        print(f"  GPU: {device}")
        print(f"{'='*60}\n")

    # ── Prompts ─────────────────────────────────────────────────────────────
    n_prompts = args.num_samples if not args.test_run else min(args.num_samples, 5)
    prompts = get_prompts(n_prompts, args.dataset_name)

    # ── Phase A: FP16 reference images (if not already present) ─────────────
    ref_dir = os.path.join(args.ref_base, f"fp16_steps{args.num_steps}", args.dataset_name)
    if accelerator.is_main_process and not os.path.exists(
        os.path.join(ref_dir, f"ref_{min(args.num_samples, 100) - 1}.png")
    ):
        print("[Ref] Generating FP16 reference images...")
        os.makedirs(ref_dir, exist_ok=True)
        pipe_ref = PixArtSigmaPipeline.from_pretrained(
            args.model_path, torch_dtype=torch.float16
        ).to(device)
        pipe_ref.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe_ref.scheduler.config
        )
        pipe_ref.set_progress_bar_config(disable=True)
        for i, prompt in enumerate(prompts[:args.num_samples]):
            out_path = os.path.join(ref_dir, f"ref_{i}.png")
            if not os.path.exists(out_path):
                gen = torch.Generator(device=device).manual_seed(42 + i)
                img = pipe_ref(
                    prompt, num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance_scale, generator=gen
                ).images[0]
                img.save(out_path)
                if i % 10 == 0:
                    print(f"  [Ref] {i}/{args.num_samples}", flush=True)
        del pipe_ref
        gc.collect()
        torch.cuda.empty_cache()
        print("[Ref] Done.")
    accelerator.wait_for_everyone()

    # ── Load quantized pipeline ──────────────────────────────────────────────
    print("[Setup] Loading pipeline...")
    pipe = PixArtSigmaPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    calib_prompts = get_prompts(8, args.dataset_name)
    cache_state: DeepCacheState | None = None

    if args.quant_method == "SVDQUANT":
        apply_svdquant_quantization(
            pipe, accelerator, calib_prompts, 5, args.num_steps, device, args
        )

    elif args.quant_method == "STEP_AWARE_MIXED":
        # Build bit_schedule
        if args.schedule_family == "direct":
            if args.bit_schedule_json is None:
                raise ValueError("--bit_schedule_json required with --schedule_family direct")
            schedule = build_schedule_direct(args.bit_schedule_json, args.num_steps)
        elif args.schedule_family == "S1":
            schedule = build_schedule_s1(args.sensitivity_json, args.low_bit_k, args.num_steps)
        elif args.schedule_family == "S2":
            schedule = build_schedule_s2(args.sensitivity_json, args.low_bit_k, args.num_steps)
        else:
            schedule = build_schedule_s3(args.sensitivity_json, args.low_bit_k, args.num_steps)

        cache_state = DeepCacheState()
        if accelerator.is_main_process:
            print("[Setup] Applying StepAwareMixedLinear...")

        with torch.no_grad():
            replaced = apply_step_aware_mixed_quantization(
                pipe.transformer, schedule, cache_state
            )

        install_step_aware_quant(pipe.transformer, cache_state)

        if accelerator.is_main_process:
            # Analytical memory report
            mem = compute_analytical_memory(pipe.transformer, schedule, args.num_steps)
            print(f"  [Memory] W4 baseline: {mem['baseline_mb']:.1f} MB")
            print(f"  [Memory] Mixed schedule: {mem['schedule_mb']:.1f} MB")
            print(f"  [Memory] Savings: {mem['savings_pct']:.1f}%")
            print(f"  [Memory] Note: analytical only (fake-quant, real VRAM unchanged)")

    accelerator.wait_for_everyone()

    # ── CLIP model ──────────────────────────────────────────────────────────
    clip_model, clip_processor = None, None
    if accelerator.is_main_process:
        try:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            clip_model.eval()
        except Exception as e:
            print(f"  [CLIP] Load failed: {e}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    save_dir = os.path.join(args.output_base, tag)
    os.makedirs(save_dir, exist_ok=True)

    if accelerator.is_main_process:
        print(f"\n[Eval] Running {args.num_samples} samples → {save_dir}")

    results = generate_and_evaluate(
        pipe=pipe,
        cache_state=cache_state,
        t_count=args.num_steps,
        prompts=prompts,
        s_count=args.num_samples,
        ref_dir=ref_dir,
        save_dir=save_dir,
        device=device,
        accelerator=accelerator,
        clip_model=clip_model,
        clip_processor=clip_processor,
        config_tag=tag,
        guidance_scale=args.guidance_scale,
    )

    if accelerator.is_main_process:
        # Add memory info for STEP_AWARE_MIXED
        if args.quant_method == "STEP_AWARE_MIXED":
            mem = compute_analytical_memory(pipe.transformer, schedule, args.num_steps)
            results["analytical_mb"] = mem["schedule_mb"]
            results["analytical_mb_baseline"] = mem["baseline_mb"]
            results["memory_savings_pct"] = mem["savings_pct"]
            results["low_bit_k"] = args.low_bit_k
            results["schedule_family"] = args.schedule_family

        print(f"\n{'─'*50}")
        print(f"  Tag      : {tag}")
        print(f"  FID      : {results['fid']:.2f}")
        print(f"  CLIP     : {results.get('clip', 'N/A')}")
        print(f"  PSNR     : {results['psnr']:.2f}")
        print(f"  SSIM     : {results['ssim']:.4f}")
        print(f"  LPIPS    : {results['lpips']:.4f}")
        print(f"  TPI (s)  : {results['time_per_image_sec']:.3f}")
        if "analytical_mb" in results:
            print(f"  Mem (MB) : {results['analytical_mb']:.1f} "
                  f"(baseline {results['analytical_mb_baseline']:.1f}, "
                  f"savings {results['memory_savings_pct']:.1f}%)")
        print(f"{'─'*50}\n")

        if not args.test_run:
            metrics_path = os.path.join(save_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump({"tag": tag, "args": vars(args), "results": results}, f, indent=2)

            # Append to sweep CSV
            csv_path = os.path.join(os.path.dirname(save_dir), "sweep_results.csv")
            fieldnames = ["tag", "quant_method", "schedule_family", "low_bit_k",
                          "num_steps", "num_samples",
                          "fid", "clip", "psnr", "ssim", "lpips",
                          "time_per_image_sec", "analytical_mb", "memory_savings_pct"]
            write_header = not os.path.exists(csv_path)
            row = {
                "tag": tag,
                "quant_method": args.quant_method,
                "schedule_family": getattr(args, "schedule_family", ""),
                "low_bit_k": getattr(args, "low_bit_k", ""),
                "num_steps": args.num_steps,
                "num_samples": args.num_samples,
                **{k: results.get(k, "") for k in
                   ["fid", "clip", "psnr", "ssim", "lpips",
                    "time_per_image_sec", "analytical_mb", "memory_savings_pct"]},
            }
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            print(f"  [CSV] {csv_path}")


if __name__ == "__main__":
    main()
