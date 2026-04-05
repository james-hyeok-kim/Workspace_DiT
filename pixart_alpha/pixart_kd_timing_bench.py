"""
pixart_kd_timing_bench.py
=========================
PTQ IALM 1회 + KD steps별 순수 훈련 시간 측정 (eval 없음).
기존 pixart_kd_finetune.py의 PTQ/KD 로직을 그대로 호출.

Usage:
  accelerate launch --multi_gpu --num_processes 2 pixart_kd_timing_bench.py
"""

import os, time, copy, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from accelerate import Accelerator
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler

from pixart_kd_finetune import (
    ManualRPCALinear,
    get_module_by_name, set_module_by_name,
    get_prompts, run_kd_finetuning,
)

# ── 설정 ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "PixArt-alpha/PixArt-XL-2-1024-MS"
DATASET      = "MJHQ"
ACT_MODE     = "NVFP4"
WGT_MODE     = "NVFP4"
LOWRANK      = 32
BLOCK_SIZE   = 16
ALPHA        = 0.5
KD_LR        = 1e-4
KD_PROMPTS   = 8
NUM_CALIB    = 20
T_COUNT      = 20
KD_STEPS_LIST = [30, 50, 70, 100]
SAVE_PATH    = "results/kd_timing_bench/timing_results.json"


def build_fake_args(kd_steps):
    """run_kd_finetuning이 요구하는 args namespace 생성."""
    import argparse
    a = argparse.Namespace()
    a.kd_steps    = kd_steps
    a.kd_lr       = KD_LR
    a.kd_prompts  = KD_PROMPTS
    a.dataset_name = DATASET
    return a


def main():
    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("  KD Timing Benchmark  (PTQ x1, KD 30/50/70/100)")
        print(f"  GPUs={accelerator.num_processes}  "
              f"Act={ACT_MODE}  Wgt={WGT_MODE}  rank={LOWRANK}")
        print("="*60 + "\n")

    prompts = get_prompts(max(NUM_CALIB, KD_PROMPTS),
                          build_fake_args(0))

    # ── PTQ ──────────────────────────────────────────────────────────────────
    accelerator.print("[PTQ] Loading model...")
    pipe = PixArtAlphaPipeline.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config)
    transformer = pipe.transformer

    accelerator.print("[PTQ] Deepcopying FP16 teacher to CPU...")
    teacher_transformer = copy.deepcopy(transformer).cpu()
    teacher_transformer.eval().requires_grad_(False)

    # calibration (pixart_kd_finetune.py와 동일한 방식)
    skip_kw = ["x_embedder", "t_embedder", "proj_out"]
    target_names = [
        n for n, m in transformer.named_modules()
        if isinstance(m, nn.Linear) and not any(kw in n for kw in skip_kw)
    ]
    accelerator.print(f"[PTQ] Targeted {len(target_names)} layers. Calibrating...")

    all_samples = {}
    def hook_fn(name):
        def fwd(m, i, o):
            flat = i[0].detach().view(-1, i[0].shape[-1]).abs().float()
            all_samples.setdefault(name, []).append(flat.max(dim=0)[0].cpu())
        return fwd

    hooks = [get_module_by_name(transformer, n).register_forward_hook(hook_fn(n))
             for n in target_names]
    p_count = min(NUM_CALIB, len(prompts))
    with accelerator.split_between_processes(prompts[:p_count]) as local_p:
        for prompt in tqdm(local_p, disable=not accelerator.is_main_process):
            pipe(prompt, num_inference_steps=T_COUNT,
                 generator=torch.Generator(device=device).manual_seed(42))
    for h in hooks:
        h.remove()

    for name in all_samples:
        local_mean = torch.stack(all_samples[name]).mean(dim=0).to(device)
        all_samples[name] = accelerator.reduce(local_mean, reduction="mean")
    accelerator.wait_for_everyone()

    # layer replacement (PTQ 시간 측정 시작)
    torch.cuda.synchronize()
    t_ptq = time.time()

    for name in tqdm(target_names, desc="Replacing layers (RPCA IALM)",
                     disable=not accelerator.is_main_process):
        orig_m = get_module_by_name(transformer, name)
        if next(orig_m.parameters()).device != device:
            continue
        new_layer = ManualRPCALinear(
            orig_m, ACT_MODE, WGT_MODE, ALPHA, LOWRANK, BLOCK_SIZE,
            rpca_lam=None, dtype=torch.float16
        ).to(device)
        if name in all_samples:
            new_layer.manual_calibrate_and_rpca(all_samples[name])
        set_module_by_name(transformer, name, new_layer)
    accelerator.wait_for_everyone()

    torch.cuda.synchronize()
    ptq_time = time.time() - t_ptq
    accelerator.print(
        f"[PTQ] Done. {ptq_time:.1f}s ({ptq_time/60:.1f}m)\n")

    # ── PTQ 후 lora 상태 저장 ─────────────────────────────────────────────
    init_lora = {
        n: p.data.clone()
        for n, p in pipe.transformer.named_parameters()
        if "lora_a" in n or "lora_b" in n
    }

    # teacher를 GPU로 (KD 동안 유지)
    teacher_transformer = teacher_transformer.to(device, dtype=torch.float16)
    torch.cuda.empty_cache()

    kd_prompts_list = get_prompts(KD_PROMPTS, build_fake_args(0))[:KD_PROMPTS]

    # ── KD timing per step count ──────────────────────────────────────────
    results = {}
    for n_steps in KD_STEPS_LIST:
        # lora 초기 상태로 reset
        with torch.no_grad():
            for n, p in pipe.transformer.named_parameters():
                if n in init_lora:
                    p.data.copy_(init_lora[n])

        args_kd = build_fake_args(n_steps)

        # warmup: 1step (CUDA kernel 캐시)
        args_warmup = build_fake_args(1)
        run_kd_finetuning(pipe, teacher_transformer, kd_prompts_list,
                          args_warmup, accelerator)
        # reset again after warmup
        with torch.no_grad():
            for n, p in pipe.transformer.named_parameters():
                if n in init_lora:
                    p.data.copy_(init_lora[n])

        torch.cuda.synchronize()
        t0 = time.time()
        loss_log = run_kd_finetuning(pipe, teacher_transformer,
                                     kd_prompts_list, args_kd, accelerator)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        s_per_step = elapsed / n_steps
        accelerator.print(
            f"[KD-{n_steps}] {elapsed:.1f}s ({elapsed/60:.1f}m)  "
            f"{s_per_step:.2f}s/step  "
            f"loss: {loss_log[0]:.5f}→{loss_log[-1]:.5f}")

        results[f"KD-{n_steps}"] = {
            "steps":       n_steps,
            "time_s":      round(elapsed, 1),
            "time_str":    f"{int(elapsed//60)}m{int(elapsed%60):02d}s",
            "s_per_step":  round(s_per_step, 2),
            "loss_init":   round(loss_log[0], 6),
            "loss_final":  round(loss_log[-1], 6),
        }

    results["PTQ_RPCA_IALM"] = {
        "time_s":   round(ptq_time, 1),
        "time_str": f"{int(ptq_time//60)}m{int(ptq_time%60):02d}s",
    }

    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        with open(SAVE_PATH, "w") as f:
            json.dump(results, f, indent=4)

        print("\n" + "="*60)
        print(f"  PTQ (RPCA IALM, 289 layers): {results['PTQ_RPCA_IALM']['time_str']}")
        print("-"*60)
        print(f"  {'Steps':<8} {'Time':>8}  {'s/step':>8}  Loss")
        print("-"*60)
        for k in [f"KD-{s}" for s in KD_STEPS_LIST]:
            v = results[k]
            print(f"  {v['steps']:<8} {v['time_str']:>8}  "
                  f"{v['s_per_step']:>7.2f}s  "
                  f"{v['loss_init']:.5f}→{v['loss_final']:.5f}")
        print("="*60)
        print(f"  Saved → {SAVE_PATH}")


if __name__ == "__main__":
    main()
