"""
analyze_cache_similarity.py

MR-GPTQ negative cache penalty 심층 분석:
4가지 NVFP4 방법에서 per-block, per-step 간 residual의
cosine similarity와 magnitude ratio를 측정.

연구 질문:
  왜 MRGPTQ는 DeepCache 적용 시 FID가 오히려 낮아지는가 (cache_penalty = -14.64)?
  H(16) Micro-Rotation이 deep block residual의 방향을 안정화하여
  cache reuse가 더 정확해지는지 확인.

측정 지표 (per block, per step):
  cosine_sim   = cosine(residual[t], residual[t-interval])  → 방향 유사도
  magnitude_ratio = ||residual[t]|| / ||residual[t-interval]||  → 크기 비율
  l2_distance  = ||residual[t] - residual[t-interval]|| / ||residual[t]||  → 상대 L2

결과:
  results/analysis/
    cache_similarity_{METHOD}.csv
    cache_similarity_summary.csv
    cache_similarity_per_block.csv

사용법:
  python analyze_cache_similarity.py
  python analyze_cache_similarity.py --methods RTN SVDQUANT --num_samples 5
"""

import os
import gc
import csv
import argparse

import torch
import torch.nn.functional as F
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler

from eval_utils import get_prompts
from quant_methods import (
    apply_rtn_quantization,
    apply_svdquant_quantization,
    apply_mrgptq_quantization,
    apply_fouroversix_quantization,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_START    = 8
CACHE_END      = 20
CACHE_INTERVAL = 2     # stale = residual from 2 steps ago
NUM_STEPS      = 20
GUIDANCE_SCALE = 4.5
MODEL_PATH     = "PixArt-alpha/PixArt-XL-2-1024-MS"


# ---------------------------------------------------------------------------
# Per-block residual collection
# ---------------------------------------------------------------------------

class BlockResidualCollector:
    """
    Collects per-block residuals using forward hooks on transformer blocks.
    residuals[block_idx][step_idx][image_idx] = residual tensor (cpu, mean over tokens)
    """

    def __init__(self, cache_start: int, cache_end: int):
        self.cache_start = cache_start
        self.cache_end   = cache_end
        self.step_idx    = 0
        self.image_idx   = 0
        # {block_idx: {step_idx: list[Tensor]}}  list indexed by image
        self.residuals: dict[int, dict[int, list]] = {
            b: {} for b in range(cache_start, cache_end)
        }
        self._input_buf: dict[int, torch.Tensor] = {}
        self._hooks: list = []

    def reset_step(self):
        self.step_idx = 0

    def next_image(self):
        self.image_idx += 1
        self.step_idx  = 0

    def install(self, transformer):
        """Register pre/post hooks on deep blocks."""
        for b_idx in range(self.cache_start, self.cache_end):
            block = transformer.transformer_blocks[b_idx]

            def make_pre(bi):
                def pre_hook(module, args):
                    x = args[0].detach()
                    self._input_buf[bi] = x
                return pre_hook

            def make_post(bi):
                def post_hook(module, args, output):
                    if bi not in self._input_buf:
                        return
                    out = output.detach()
                    inp = self._input_buf[bi]
                    # residual = output - input, mean over batch & tokens → [hidden]
                    res = (out - inp).float().mean(dim=(0, 1)).cpu()
                    step = self.step_idx
                    if step not in self.residuals[bi]:
                        self.residuals[bi][step] = []
                    self.residuals[bi][step].append(res)
                    # Increment step_idx only at the last deep block
                    if bi == self.cache_end - 1:
                        self.step_idx += 1
                return post_hook

            h_pre  = block.register_forward_pre_hook(make_pre(b_idx))
            h_post = block.register_forward_hook(make_post(b_idx))
            self._hooks.extend([h_pre, h_post])

    def uninstall(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._input_buf.clear()


# ---------------------------------------------------------------------------
# Compute metrics from collected residuals
# ---------------------------------------------------------------------------

def compute_similarity_metrics(collector: BlockResidualCollector,
                                 method_name: str,
                                 num_samples: int,
                                 interval: int) -> list[dict]:
    """
    From collected residuals, compute per-(block, step, image) metrics.
    Returns list of dicts for CSV writing.
    """
    rows = []
    for b_idx in range(collector.cache_start, collector.cache_end):
        step_data = collector.residuals[b_idx]
        available_steps = sorted(step_data.keys())

        for step in available_steps:
            stale_step = step - interval
            if stale_step not in step_data:
                continue

            fresh_list = step_data[step]
            stale_list = step_data[stale_step]
            n = min(len(fresh_list), len(stale_list))
            if n == 0:
                continue

            for img_i in range(n):
                fresh = fresh_list[img_i]  # [hidden_dim]
                stale = stale_list[img_i]

                cos_sim = F.cosine_similarity(
                    fresh.unsqueeze(0), stale.unsqueeze(0)
                ).item()
                fresh_norm = fresh.norm().item()
                stale_norm = stale.norm().item()
                mag_ratio = fresh_norm / (stale_norm + 1e-8)
                l2_dist   = (fresh - stale).norm().item() / (fresh_norm + 1e-8)

                rows.append({
                    "method":        method_name,
                    "block_idx":     b_idx,
                    "step_idx":      step,
                    "image_idx":     img_i,
                    "cosine_sim":    round(cos_sim, 6),
                    "magnitude_ratio": round(mag_ratio, 6),
                    "l2_distance":   round(l2_dist, 6),
                    "fresh_norm":    round(fresh_norm, 4),
                    "stale_norm":    round(stale_norm, 4),
                })
    return rows


# ---------------------------------------------------------------------------
# Main analysis per method
# ---------------------------------------------------------------------------

def analyze_method(method_name: str, prompts: list[str], num_samples: int,
                   device, args) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"  Analyzing: {method_name}")
    print(f"{'='*60}")

    # Load fresh pipeline
    pipe = PixArtAlphaPipeline.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )
    transformer = pipe.transformer

    # Calibration setup (minimal)
    p_count = min(4, num_samples)

    class FakeArgs:
        lowrank    = 32
        alpha      = 0.5
        block_size = 16
        outlier_ratio = 0.05

    fake_args = FakeArgs()

    from contextlib import contextmanager

    class FakeAccelerator:
        is_main_process = True
        process_index   = 0
        @staticmethod
        @contextmanager
        def split_between_processes(x):
            yield x
        @staticmethod
        def reduce(x, reduction="mean"): return x
        @staticmethod
        def wait_for_everyone(): pass
        @staticmethod
        def print(*a, **k): print(*a, **k)

    accel = FakeAccelerator()

    # Apply quantization
    print(f"  [Phase 1] Applying {method_name} quantization...")
    if method_name == "RTN":
        apply_rtn_quantization(pipe, transformer, accel, prompts, p_count,
                               NUM_STEPS, device, fake_args)
    elif method_name == "SVDQUANT":
        apply_svdquant_quantization(pipe, accel, prompts, p_count,
                                    NUM_STEPS, device, fake_args)
        transformer = pipe.transformer
    elif method_name == "MRGPTQ":
        apply_mrgptq_quantization(pipe, transformer, accel, prompts, p_count,
                                   NUM_STEPS, device, fake_args)
    elif method_name == "FOUROVERSIX":
        apply_fouroversix_quantization(pipe, transformer, accel, prompts, p_count,
                                        NUM_STEPS, device, fake_args)

    # Install collector hooks
    print(f"  [Phase 2] Collecting residuals ({num_samples} samples × {NUM_STEPS} steps)...")
    collector = BlockResidualCollector(CACHE_START, CACHE_END)
    collector.install(transformer)

    for i in range(num_samples):
        gen = torch.Generator(device=device).manual_seed(42 + i)
        collector.reset_step()
        with torch.no_grad():
            pipe(
                prompts[i % len(prompts)],
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=gen,
            )
        collector.next_image()
        print(f"    sample {i+1}/{num_samples} done", flush=True)

    collector.uninstall()

    # Compute metrics
    print(f"  [Phase 3] Computing similarity metrics...")
    rows = compute_similarity_metrics(collector, method_name, num_samples,
                                       CACHE_INTERVAL)
    print(f"    {len(rows)} (block, step, image) tuples")

    # Cleanup
    del pipe, collector
    torch.cuda.empty_cache()
    gc.collect()

    return rows


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(all_rows: list[dict]) -> list[dict]:
    """Aggregate per-method and per-(method, block) statistics."""
    from collections import defaultdict
    import statistics

    # per-method
    method_data: dict[str, dict] = defaultdict(lambda: {
        "cos": [], "mag": [], "l2": []
    })
    # per-method, per-block
    block_data: dict[tuple, dict] = defaultdict(lambda: {
        "cos": [], "mag": [], "l2": []
    })

    for r in all_rows:
        m = r["method"]
        b = r["block_idx"]
        method_data[m]["cos"].append(r["cosine_sim"])
        method_data[m]["mag"].append(r["magnitude_ratio"])
        method_data[m]["l2"].append(r["l2_distance"])
        block_data[(m, b)]["cos"].append(r["cosine_sim"])
        block_data[(m, b)]["mag"].append(r["magnitude_ratio"])
        block_data[(m, b)]["l2"].append(r["l2_distance"])

    summary = []
    for m, d in sorted(method_data.items()):
        summary.append({
            "method":       m,
            "block_idx":    "ALL",
            "n":            len(d["cos"]),
            "mean_cos_sim": round(statistics.mean(d["cos"]), 4),
            "std_cos_sim":  round(statistics.stdev(d["cos"]) if len(d["cos"]) > 1 else 0, 4),
            "mean_mag_ratio": round(statistics.mean(d["mag"]), 4),
            "std_mag_ratio":  round(statistics.stdev(d["mag"]) if len(d["mag"]) > 1 else 0, 4),
            "mean_l2":      round(statistics.mean(d["l2"]), 4),
            "std_l2":       round(statistics.stdev(d["l2"]) if len(d["l2"]) > 1 else 0, 4),
        })
    for (m, b), d in sorted(block_data.items()):
        summary.append({
            "method":       m,
            "block_idx":    b,
            "n":            len(d["cos"]),
            "mean_cos_sim": round(statistics.mean(d["cos"]), 4),
            "std_cos_sim":  round(statistics.stdev(d["cos"]) if len(d["cos"]) > 1 else 0, 4),
            "mean_mag_ratio": round(statistics.mean(d["mag"]), 4),
            "std_mag_ratio":  round(statistics.stdev(d["mag"]) if len(d["mag"]) > 1 else 0, 4),
            "mean_l2":      round(statistics.mean(d["l2"]), 4),
            "std_l2":       round(statistics.stdev(d["l2"]) if len(d["l2"]) > 1 else 0, 4),
        })
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-block residual similarity analysis for MR-GPTQ negative penalty"
    )
    parser.add_argument("--methods", nargs="+",
                        default=["RTN", "SVDQUANT", "MRGPTQ", "FOUROVERSIX"],
                        choices=["RTN", "SVDQUANT", "MRGPTQ", "FOUROVERSIX"],
                        help="Methods to analyze")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of images to analyze per method")
    parser.add_argument("--save_dir", type=str, default="results/analysis")
    parser.add_argument("--dataset_name", type=str, default="MJHQ")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompts = get_prompts(args.num_samples, args.dataset_name)

    all_rows: list[dict] = []
    fieldnames = ["method", "block_idx", "step_idx", "image_idx",
                  "cosine_sim", "magnitude_ratio", "l2_distance",
                  "fresh_norm", "stale_norm"]

    for method in args.methods:
        rows = analyze_method(method, prompts, args.num_samples, device, args)
        all_rows.extend(rows)

        # Save per-method CSV immediately (in case of crash)
        csv_path = os.path.join(args.save_dir, f"cache_similarity_{method}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved: {csv_path}")

    # Save combined CSV
    all_csv = os.path.join(args.save_dir, "cache_similarity_all.csv")
    with open(all_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Save summary
    summary = compute_summary(all_rows)
    summary_fields = ["method", "block_idx", "n",
                      "mean_cos_sim", "std_cos_sim",
                      "mean_mag_ratio", "std_mag_ratio",
                      "mean_l2", "std_l2"]
    summary_csv = os.path.join(args.save_dir, "cache_similarity_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary)

    # Print method-level summary to console
    print(f"\n{'='*70}")
    print("  SUMMARY (ALL blocks, interval=2)")
    print(f"{'='*70}")
    method_rows = [r for r in summary if r["block_idx"] == "ALL"]
    hdr = f"{'Method':<14} {'N':>5} | {'cos_sim':>8} ± {'std':>6} | {'mag_ratio':>9} ± {'std':>6} | {'l2_dist':>7} ± {'std':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in method_rows:
        print(
            f"{r['method']:<14} {r['n']:>5} | "
            f"{r['mean_cos_sim']:>8.4f} ± {r['std_cos_sim']:>6.4f} | "
            f"{r['mean_mag_ratio']:>9.4f} ± {r['std_mag_ratio']:>6.4f} | "
            f"{r['mean_l2']:>7.4f} ± {r['std_l2']:>6.4f}"
        )

    print(f"\nAll results saved to: {args.save_dir}/")


if __name__ == "__main__":
    main()
