"""
benchmark_speed_memory.py

4가지 NVFP4 양자화 방법 × 3가지 캐시 모드(none/deepcache/cache_lora)에 대한
inference speedup 및 GPU 메모리 사용량 측정.

각 method당 모델을 한 번 로드하고 세 모드를 순서대로 벤치마크.

결과: results/benchmark/speed_memory_{METHOD}.json + summary CSV

사용법:
  python benchmark_speed_memory.py --method RTN
  python benchmark_speed_memory.py --method SVDQUANT
  python benchmark_speed_memory.py --all
"""

import os
import gc
import json
import time
import types
import argparse
import csv

import torch
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler

from deepcache_utils import install_deepcache, calibrate_cache_lora
from quant_methods import (
    apply_rtn_quantization,
    apply_svdquant_quantization,
    apply_mrgptq_quantization,
    apply_fouroversix_quantization,
)

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
MODEL_PATH    = "PixArt-alpha/PixArt-XL-2-1024-MS"
NUM_WARMUP    = 1      # timing 전 throwaway run
NUM_TIMED     = 5      # 실제 timing sample 수
NUM_CALIB_Q   = 20     # 양자화 calibration prompts (quant_methods 내부)
NUM_CALIB_L   = 4      # cache_lora calibration prompts
NUM_STEPS     = 20
GUIDANCE      = 4.5
LORA_RANK     = 4
CALIB_SEED_OFFSET = 1000
LORA_SEED_OFFSET  = 2000   # cache_lora calib seeds (eval: 42+i, quant_calib: 1000+i)

DEEPCACHE_START    = 8
DEEPCACHE_END      = 20
DEEPCACHE_INTERVAL = 2
DEEPCACHE_FULL_STEPS = {0}

PROMPT = "a professional photograph of a golden retriever puppy playing in the snow"
PROMPTS_CALIB = [
    "a cozy cabin in a snowy forest at dusk",
    "a colorful hot air balloon over a mountain range",
    "an astronaut floating in space with Earth in background",
    "a serene Japanese garden with a koi pond and cherry blossoms",
    "a bustling night market in an Asian city",
    "a lone lighthouse on a rocky cliff at sunset",
    "a polar bear standing on an ice floe in the Arctic",
    "a vintage steam train crossing a stone viaduct in the mountains",
    "a field of lavender in Provence with an old stone farmhouse",
    "a macro photograph of a dewdrop on a spider web",
    "a futuristic city with flying cars and neon lights",
    "a traditional Moroccan riad courtyard with intricate tile work",
    "a school of tropical fish around a coral reef",
    "a person meditating on a mountaintop at sunrise",
    "a cozy library with floor-to-ceiling bookshelves and a fireplace",
    "a pod of orca whales breaching in the open ocean",
    "a street artist painting a colorful mural in an urban alley",
    "a bowl of steaming ramen with perfectly sliced pork belly",
    "a medieval castle on a hill surrounded by autumn foliage",
    "a child building a sandcastle on a tropical beach",
]

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

class FakeAccelerator:
    """Minimal accelerator shim for quant_methods API compatibility."""
    def __init__(self, device):
        self.device = device
        self.process_index = 0
        self.is_main_process = True

    def print(self, *args, **kwargs):
        print(*args, **kwargs)

    def wait_for_everyone(self):
        pass

    def reduce(self, tensor, reduction="mean"):
        """Single-process: reduction is a no-op."""
        return tensor

    @staticmethod
    @__import__("contextlib").contextmanager
    def split_between_processes(x):
        yield x


def mem_mb():
    """Current GPU memory allocated in MB."""
    return torch.cuda.memory_allocated() / 1024 ** 2


def peak_mb():
    """Peak GPU memory allocated since last reset_peak_stats, in MB."""
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def reset_peak():
    torch.cuda.reset_peak_memory_stats()


def timed_inference(pipe, cache_state, num_warmup, num_timed, device, steps, guidance):
    """Run warm-up then timed inference. Returns list of wall-clock seconds per image."""
    times = []
    total = num_warmup + num_timed
    for i in range(total):
        if cache_state is not None:
            cache_state.reset()
        gen = torch.Generator(device=device).manual_seed(42 + i)
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        pipe(PROMPT, num_inference_steps=steps,
             guidance_scale=guidance, generator=gen)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        if i >= num_warmup:
            times.append(elapsed)
    return times


# ---------------------------------------------------------------------------
# Main benchmark for one method
# ---------------------------------------------------------------------------

def benchmark_method(method: str, device, save_dir: str):
    print(f"\n{'='*65}")
    print(f"  Benchmarking: {method}")
    print(f"{'='*65}\n")

    result = {"method": method}

    # ---- Load FP16 model ------------------------------------------------
    print("[1/6] Loading FP16 model...")
    reset_peak()
    t_load = time.perf_counter()
    pipe = PixArtAlphaPipeline.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    torch.cuda.synchronize(device)
    result["load_time_sec"] = round(time.perf_counter() - t_load, 2)
    result["mem_after_load_mb"] = round(mem_mb(), 1)
    result["mem_peak_load_mb"]  = round(peak_mb(), 1)
    print(f"  Load time: {result['load_time_sec']:.2f}s  "
          f"  Mem: {result['mem_after_load_mb']:.0f} MB")

    # ---- Quantize -------------------------------------------------------
    print(f"\n[2/6] Quantizing ({method})...")
    accelerator = FakeAccelerator(device)
    transformer  = pipe.transformer

    class _args:
        block_size  = 16
        lowrank     = 32
        alpha       = 0.5
        num_steps   = NUM_STEPS
        guidance_scale = GUIDANCE

    reset_peak()
    t_quant = time.perf_counter()

    if method == "RTN":
        apply_rtn_quantization(
            pipe, transformer, accelerator,
            PROMPTS_CALIB, NUM_CALIB_Q, NUM_STEPS, device, _args
        )
    elif method == "SVDQUANT":
        apply_svdquant_quantization(
            pipe, accelerator,
            PROMPTS_CALIB, NUM_CALIB_Q, NUM_STEPS, device, _args
        )
        transformer = pipe.transformer
    elif method == "MRGPTQ":
        apply_mrgptq_quantization(
            pipe, transformer, accelerator,
            PROMPTS_CALIB, NUM_CALIB_Q, NUM_STEPS, device, _args
        )
    elif method == "FOUROVERSIX":
        apply_fouroversix_quantization(
            pipe, transformer, accelerator,
            PROMPTS_CALIB, NUM_CALIB_Q, NUM_STEPS, device, _args
        )

    torch.cuda.synchronize(device)
    result["quant_time_sec"]      = round(time.perf_counter() - t_quant, 2)
    result["mem_after_quant_mb"]  = round(mem_mb(), 1)
    result["mem_peak_quant_mb"]   = round(peak_mb(), 1)
    print(f"  Quant time: {result['quant_time_sec']:.2f}s  "
          f"  Mem: {result['mem_after_quant_mb']:.0f} MB")

    # ---- Count model parameters -----------------------------------------
    total_params = sum(p.numel() for p in pipe.transformer.parameters())
    result["transformer_params_M"] = round(total_params / 1e6, 1)

    # ----------------------------------------------------------------
    # Mode A: no-cache
    # ----------------------------------------------------------------
    print(f"\n[3/6] Timing no-cache ({NUM_WARMUP}+{NUM_TIMED} runs)...")
    reset_peak()
    times_none = timed_inference(
        pipe, None, NUM_WARMUP, NUM_TIMED, device, NUM_STEPS, GUIDANCE
    )
    result["none_time_avg_sec"]  = round(sum(times_none) / len(times_none), 3)
    result["none_time_min_sec"]  = round(min(times_none), 3)
    result["none_mem_peak_mb"]   = round(peak_mb(), 1)
    print(f"  No-cache: avg={result['none_time_avg_sec']:.3f}s  "
          f"min={result['none_time_min_sec']:.3f}s  "
          f"peak_mem={result['none_mem_peak_mb']:.0f}MB")

    # ----------------------------------------------------------------
    # Mode B: deepcache
    # ----------------------------------------------------------------
    print(f"\n[4/6] Timing deepcache ({NUM_WARMUP}+{NUM_TIMED} runs)...")
    cache_state = install_deepcache(
        pipe.transformer,
        cache_start=DEEPCACHE_START,
        cache_end=DEEPCACHE_END,
        cache_interval=DEEPCACHE_INTERVAL,
        full_steps_set=DEEPCACHE_FULL_STEPS,
    )

    n_deep   = DEEPCACHE_END - DEEPCACHE_START
    n_total  = len(pipe.transformer.transformer_blocks)
    n_always = n_total - n_deep
    avg_blocks = (n_total + n_always * (DEEPCACHE_INTERVAL - 1)) / DEEPCACHE_INTERVAL
    speedup_est = n_total / avg_blocks

    reset_peak()
    times_dc = timed_inference(
        pipe, cache_state, NUM_WARMUP, NUM_TIMED, device, NUM_STEPS, GUIDANCE
    )
    result["dc_time_avg_sec"]    = round(sum(times_dc) / len(times_dc), 3)
    result["dc_time_min_sec"]    = round(min(times_dc), 3)
    result["dc_mem_peak_mb"]     = round(peak_mb(), 1)
    result["dc_speedup_actual"]  = round(result["none_time_avg_sec"] / result["dc_time_avg_sec"], 3)
    result["dc_speedup_theory"]  = round(speedup_est, 3)
    print(f"  DeepCache: avg={result['dc_time_avg_sec']:.3f}s  "
          f"min={result['dc_time_min_sec']:.3f}s  "
          f"speedup={result['dc_speedup_actual']:.2f}x (theory {speedup_est:.2f}x)  "
          f"peak_mem={result['dc_mem_peak_mb']:.0f}MB")

    # ----------------------------------------------------------------
    # Mode C: cache_lora — uninstall deepcache, calibrate, reinstall
    # ----------------------------------------------------------------
    print(f"\n[5/6] Cache-LoRA calibration (rank={LORA_RANK}, {NUM_CALIB_L} prompts)...")

    # Uninstall deepcache by restoring original forward
    # (reset the monkey-patch by reloading the method)
    pipe.transformer.forward = types.MethodType(
        type(pipe.transformer).forward, pipe.transformer
    )

    reset_peak()
    corrector_A, corrector_B, calib_time = calibrate_cache_lora(
        pipe=pipe,
        transformer=pipe.transformer,
        cache_start=DEEPCACHE_START,
        cache_end=DEEPCACHE_END,
        cache_interval=DEEPCACHE_INTERVAL,
        prompts=PROMPTS_CALIB,
        num_calib=NUM_CALIB_L,
        t_count=NUM_STEPS,
        guidance_scale=GUIDANCE,
        device=device,
        rank=LORA_RANK,
        calib_seed_offset=LORA_SEED_OFFSET,
    )
    result["lora_calib_time_sec"] = round(calib_time, 2)
    result["lora_mem_peak_calib_mb"] = round(peak_mb(), 1)
    result["lora_corrector_params"]  = corrector_A.numel() + corrector_B.numel()
    print(f"  Calib done: {calib_time:.1f}s  "
          f"corrector params={result['lora_corrector_params']:,}  "
          f"peak_mem={result['lora_mem_peak_calib_mb']:.0f}MB")

    # Reinstall deepcache with corrector
    print(f"\n[6/6] Timing cache_lora ({NUM_WARMUP}+{NUM_TIMED} runs)...")
    lora_state = install_deepcache(
        pipe.transformer,
        cache_start=DEEPCACHE_START,
        cache_end=DEEPCACHE_END,
        cache_interval=DEEPCACHE_INTERVAL,
        full_steps_set=DEEPCACHE_FULL_STEPS,
    )
    lora_state.corrector_A = corrector_A
    lora_state.corrector_B = corrector_B

    reset_peak()
    times_lora = timed_inference(
        pipe, lora_state, NUM_WARMUP, NUM_TIMED, device, NUM_STEPS, GUIDANCE
    )
    result["lora_time_avg_sec"]   = round(sum(times_lora) / len(times_lora), 3)
    result["lora_time_min_sec"]   = round(min(times_lora), 3)
    result["lora_mem_peak_mb"]    = round(peak_mb(), 1)
    result["lora_speedup_actual"] = round(result["none_time_avg_sec"] / result["lora_time_avg_sec"], 3)
    print(f"  Cache-LoRA: avg={result['lora_time_avg_sec']:.3f}s  "
          f"min={result['lora_time_min_sec']:.3f}s  "
          f"speedup={result['lora_speedup_actual']:.2f}x  "
          f"peak_mem={result['lora_mem_peak_mb']:.0f}MB")

    # ---- Cleanup ---------------------------------------------------------
    del pipe, corrector_A, corrector_B
    torch.cuda.empty_cache()
    gc.collect()

    # ---- Save ------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"bench_{method}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\n  Saved: {json_path}")

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default=None,
                        choices=["RTN", "SVDQUANT", "MRGPTQ", "FOUROVERSIX"])
    parser.add_argument("--all", action="store_true",
                        help="Run all 4 methods sequentially")
    parser.add_argument("--save_dir", type=str,
                        default="./results/benchmark")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    total_vram = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
    print(f"Total VRAM: {total_vram:.0f} MB")

    methods = (["RTN", "SVDQUANT", "MRGPTQ", "FOUROVERSIX"]
               if args.all else [args.method])

    all_results = []
    for method in methods:
        r = benchmark_method(method, device, args.save_dir)
        all_results.append(r)

    # Write summary CSV
    if len(all_results) > 1:
        csv_path = os.path.join(args.save_dir, "benchmark_summary.csv")
        fields = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSummary CSV: {csv_path}")

    # Print summary table
    print(f"\n{'='*85}")
    print(f"{'Method':<14} | {'No-cache':>10} | {'DeepCache':>10} {'Speedup':>8} | "
          f"{'CacheLora':>10} {'Speedup':>8} | {'Calib':>8}")
    print(f"{'-'*85}")
    for r in all_results:
        print(f"{r['method']:<14} | "
              f"{r['none_time_avg_sec']:>10.3f}s | "
              f"{r['dc_time_avg_sec']:>10.3f}s {r['dc_speedup_actual']:>7.2f}x | "
              f"{r['lora_time_avg_sec']:>10.3f}s {r['lora_speedup_actual']:>7.2f}x | "
              f"{r['lora_calib_time_sec']:>7.1f}s")
    print(f"\n{'Method':<14} | {'Mem@Load':>10} | {'Mem@Quant':>10} | {'Mem@NoneInf':>12} | "
          f"{'Mem@DCInf':>10} | {'Mem@LoraInf':>12} | {'Params(M)':>10}")
    print(f"{'-'*85}")
    for r in all_results:
        print(f"{r['method']:<14} | "
              f"{r['mem_after_load_mb']:>10.0f} | "
              f"{r['mem_after_quant_mb']:>10.0f} | "
              f"{r['none_mem_peak_mb']:>12.0f} | "
              f"{r['dc_mem_peak_mb']:>10.0f} | "
              f"{r['lora_mem_peak_mb']:>12.0f} | "
              f"{r['transformer_params_M']:>10.1f}")
    print(f"{'='*85}")


if __name__ == "__main__":
    main()
