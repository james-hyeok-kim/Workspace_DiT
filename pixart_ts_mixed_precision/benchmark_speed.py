"""
속도 벤치마크: config별 실제 inference time 측정
  - FP16 (no quant) 기준 speedup
  - UNIFORM_FP4 기준 speedup (SVD rank gating 효과 분리)
  - ms/image, images/sec 리포트
  - 결과를 speed_results.csv로 저장 및 results_summary.csv에 병합
"""

import os, time, csv, json
import torch
import torch.nn as nn
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler

# ── 현재 파일 기준 경로 ──────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")

# ── 실험 스크립트에서 필요한 것만 임포트 ────────────────────────
import sys
sys.path.insert(0, BASE_DIR)
from pixart_ts_mixed_precision import (
    ABLATION_CONFIGS, _TS_STATE,
    patch_transformer_for_ts, MixedPrecisionTSLinear,
    get_module_by_name, set_module_by_name,
    _block_avg,
)

# ── 설정 ────────────────────────────────────────────────────────
MODEL_PATH  = "PixArt-alpha/PixArt-XL-2-1024-MS"
DEVICE      = torch.device("cuda:0")
N_WARMUP    = 3     # 워밍업 run 수 (GPU 캐시 안정화)
N_MEASURE   = 7     # 측정 run 수
T_COUNT     = 20    # denoising steps
BLOCK_SIZE  = 128
ALPHA       = 0.5
N_GROUPS    = 3
SKIP_KW     = ["x_embedder", "t_embedder", "proj_out"]

PROMPT = "A professional high-quality photo of a futuristic city with neon lights"

def time_inference(pipe, n_warmup, n_measure):
    """워밍업 후 inference time 측정 (ms/image)"""
    gen = torch.Generator(device=DEVICE).manual_seed(42)
    with torch.no_grad():
        for _ in range(n_warmup):
            pipe(PROMPT, num_inference_steps=T_COUNT, generator=gen)
        torch.cuda.synchronize()

        times = []
        for _ in range(n_measure):
            gen = torch.Generator(device=DEVICE).manual_seed(42)
            t0 = time.perf_counter()
            pipe(PROMPT, num_inference_steps=T_COUNT, generator=gen)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)  # ms

    avg_ms = sum(times) / len(times)
    std_ms = (sum((t - avg_ms) ** 2 for t in times) / len(times)) ** 0.5
    return avg_ms, std_ms

def build_quantized_pipe(mp_config_name, calib_xmax_global):
    """지정한 config의 양자화 파이프라인 빌드 (calibration 간소화: global xmax 재사용)"""
    pipe = PixArtAlphaPipeline.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16
    ).to(DEVICE)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    transformer = pipe.transformer

    group_configs = ABLATION_CONFIGS[mp_config_name]
    target_names  = [
        n for n, m in transformer.named_modules()
        if isinstance(m, nn.Linear) and not any(kw in n for kw in SKIP_KW)
    ]

    patch_transformer_for_ts(transformer, n_groups=N_GROUPS)

    # calibration: global xmax만 사용 (각 group에 동일 stats 주입)
    for name in target_names:
        orig_m = get_module_by_name(transformer, name)
        new_l  = MixedPrecisionTSLinear(
            orig_m,
            group_configs=group_configs,
            n_groups=N_GROUPS,
            block_size=BLOCK_SIZE,
            alpha=ALPHA,
            wgt_bits="NVFP4",
            dtype=torch.float16,
        ).to(DEVICE)

        in_f = orig_m.in_features
        xmax = calib_xmax_global.get(name)
        if xmax is not None:
            grouped = {g: {"xmax": xmax.to(DEVICE), "hdiag": xmax.pow(2).to(DEVICE)}
                       for g in range(N_GROUPS)}
            new_l.calibrate(grouped)
        set_module_by_name(transformer, name, new_l)

    return pipe

# ────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Timestep-Aware Mixed Precision — Speed Benchmark")
print("=" * 60)

# ── Step 1: FP16 baseline 속도 측정 ─────────────────────────────
print("\n[1/7] FP16 baseline timing...")
pipe_fp16 = PixArtAlphaPipeline.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16
).to(DEVICE)
pipe_fp16.scheduler = DPMSolverMultistepScheduler.from_config(pipe_fp16.scheduler.config)

fp16_ms, fp16_std = time_inference(pipe_fp16, N_WARMUP, N_MEASURE)
print(f"  FP16: {fp16_ms:.1f} ± {fp16_std:.1f} ms/image")

# ── Step 2: Calibration (global xmax 한 번만) ────────────────────
print("\n[2/7] Calibrating (global xmax)...")
transformer_ref = pipe_fp16.transformer
target_names = [
    n for n, m in transformer_ref.named_modules()
    if isinstance(m, nn.Linear) and not any(kw in n for kw in SKIP_KW)
]

calib_xmax = {}
def hook_fn(name):
    def fwd(m, inp, out):
        x = inp[0].detach().view(-1, inp[0].shape[-1]).abs().float()
        if name not in calib_xmax:
            calib_xmax[name] = []
        calib_xmax[name].append(x.max(dim=0)[0].cpu())
    return fwd

hooks = [get_module_by_name(transformer_ref, n).register_forward_hook(hook_fn(n))
         for n in target_names]
with torch.no_grad():
    for _ in range(3):
        pipe_fp16(PROMPT, num_inference_steps=T_COUNT,
                  generator=torch.Generator(device=DEVICE).manual_seed(42))
for h in hooks: h.remove()

calib_xmax_global = {n: torch.stack(v).mean(0) for n, v in calib_xmax.items()}
del pipe_fp16
torch.cuda.empty_cache()
print(f"  Calibrated {len(calib_xmax_global)} layers.")

# ── Step 3: 각 config 속도 측정 ─────────────────────────────────
results_speed = []
CONFIGS = list(ABLATION_CONFIGS.keys())

for idx, cfg_name in enumerate(CONFIGS):
    print(f"\n[{idx+3}/7] {cfg_name} timing...")
    pipe_q = build_quantized_pipe(cfg_name, calib_xmax_global)
    avg_ms, std_ms = time_inference(pipe_q, N_WARMUP, N_MEASURE)
    speedup_vs_fp16 = fp16_ms / avg_ms

    results_speed.append({
        "config":           cfg_name,
        "ms_per_image":     round(avg_ms, 1),
        "std_ms":           round(std_ms, 1),
        "img_per_sec":      round(1000 / avg_ms, 3),
        "speedup_vs_fp16":  round(speedup_vs_fp16, 3),
    })
    print(f"  {cfg_name}: {avg_ms:.1f} ± {std_ms:.1f} ms  |  {speedup_vs_fp16:.3f}x vs FP16")

    del pipe_q
    torch.cuda.empty_cache()

# UNIFORM_FP4 기준 speedup 추가
uniform_ms = next(r["ms_per_image"] for r in results_speed if r["config"] == "UNIFORM_FP4")
for r in results_speed:
    r["speedup_vs_uniform"] = round(uniform_ms / r["ms_per_image"], 3)

# FP16 row 추가 (맨 앞)
results_speed.insert(0, {
    "config":           "FP16 (no quant)",
    "ms_per_image":     round(fp16_ms, 1),
    "std_ms":           round(fp16_std, 1),
    "img_per_sec":      round(1000 / fp16_ms, 3),
    "speedup_vs_fp16":  1.0,
    "speedup_vs_uniform": round(uniform_ms / fp16_ms, 3),
})

# ── Step 4: speed_results.csv 저장 ──────────────────────────────
speed_csv = os.path.join(RESULT_DIR, "speed_results.csv")
fields = ["config", "ms_per_image", "std_ms", "img_per_sec",
          "speedup_vs_fp16", "speedup_vs_uniform"]
with open(speed_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results_speed)
print(f"\nSpeed CSV saved: {speed_csv}")

# ── Step 5: results_summary.csv에 속도 컬럼 병합 ───────────────
summary_csv = os.path.join(RESULT_DIR, "results_summary.csv")
speed_map = {r["config"]: r for r in results_speed}

merged_rows = []
with open(summary_csv) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames + ["ms_per_image", "std_ms", "speedup_vs_fp16", "speedup_vs_uniform"]
    for row in reader:
        cfg = row["config"]
        sp  = speed_map.get(cfg, {})
        row["ms_per_image"]       = sp.get("ms_per_image", "N/A")
        row["std_ms"]             = sp.get("std_ms", "N/A")
        row["speedup_vs_fp16"]    = sp.get("speedup_vs_fp16", "N/A")
        row["speedup_vs_uniform"] = sp.get("speedup_vs_uniform", "N/A")
        merged_rows.append(row)

with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(merged_rows)
print(f"results_summary.csv 업데이트 완료 (속도 컬럼 추가)")

# ── 최종 비교표 출력 ────────────────────────────────────────────
print(f"\n{'Config':<28} {'ms/img':>8} {'±':>5} {'img/s':>7} {'vs FP16':>9} {'vs Uniform':>11}")
print("-" * 75)
for r in results_speed:
    print(f"  {r['config']:<26} {r['ms_per_image']:>7.1f} {r['std_ms']:>5.1f}"
          f" {r['img_per_sec']:>7.3f} {r['speedup_vs_fp16']:>9.3f}x {r['speedup_vs_uniform']:>10.3f}x")

print(f"\n완료. 결과 저장:")
print(f"  {speed_csv}")
print(f"  {summary_csv}")
