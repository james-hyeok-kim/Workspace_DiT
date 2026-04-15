#!/usr/bin/env python3
"""
generate_ref_images.py
──────────────────────
PixArt-XL 로 reference 이미지를 생성하여 precision별 디렉토리에 저장합니다.
이미 존재하는 이미지는 건너뜁니다 (재실행 안전).

저장 경로:
  {out_dir}/{precision}_steps{N}/ref_0.png
                                 ref_1.png
                                 ...

사용 예:
  # FP16 15-step ref 1000장
  accelerate launch --num_processes 1 generate_ref_images.py \
      --precision fp16 --num_inference_steps 15 --num_samples 1000

  # BF16 20-step ref 1000장
  accelerate launch --num_processes 1 generate_ref_images.py \
      --precision bf16 --num_inference_steps 20 --num_samples 1000
"""

import argparse
import os
import time

import torch
from accelerate import Accelerator
from datasets import load_dataset
from diffusers import DPMSolverMultistepScheduler, PixArtAlphaPipeline
from PIL import Image


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def get_prompts(num_samples: int, dataset_name: str) -> list[str]:
    if dataset_name == "MJHQ":
        path, split, key = "xingjianleng/mjhq30k", "test", "text"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset = load_dataset(path, split=split, streaming=True)
    prompts = []
    for i, entry in enumerate(dataset):
        if i >= num_samples:
            break
        prompts.append(entry[key])
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Reference image generator")
    parser.add_argument("--model_path",          type=str,
                        default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--dataset_name",        type=str, default="MJHQ")
    parser.add_argument("--precision",           type=str, default="fp16",
                        choices=list(DTYPE_MAP.keys()),
                        help="Model precision: fp16 | bf16 | fp32")
    parser.add_argument("--num_inference_steps", type=int, default=20,
                        help="Denoising steps")
    parser.add_argument("--guidance_scale",      type=float, default=4.5)
    parser.add_argument("--num_samples",         type=int, default=1000)
    parser.add_argument("--out_dir",             type=str,
                        default="/data/james_dit_pixart_xl_mjhq",
                        help="Root output directory")
    parser.add_argument("--seed",                type=int, default=42)
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    # ── 저장 경로 결정 ────────────────────────────────────────────────────────
    tag = f"{args.precision}_steps{args.num_inference_steps}"
    save_dir = os.path.join(args.out_dir, tag)
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        accelerator.print(f"Saving to: {save_dir}")
    accelerator.wait_for_everyone()

    # ── 프롬프트 로드 ─────────────────────────────────────────────────────────
    prompts = get_prompts(args.num_samples, args.dataset_name)
    s_count = len(prompts)
    accelerator.print(f"Loaded {s_count} prompts from {args.dataset_name}")

    # 이미 생성된 이미지 확인
    existing = sum(
        1 for i in range(s_count)
        if os.path.exists(os.path.join(save_dir, f"ref_{i}.png"))
    )
    if accelerator.is_main_process and existing > 0:
        accelerator.print(f"Found {existing} existing images — skipping those.")

    if existing == s_count:
        accelerator.print("All images already exist. Nothing to generate.")
        return

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    dtype = DTYPE_MAP[args.precision]
    accelerator.print(f"Loading {args.model_path} in {args.precision} ...")

    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_path, torch_dtype=dtype
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )

    # ── 이미지 생성 ───────────────────────────────────────────────────────────
    accelerator.print(
        f"\nGenerating {s_count} ref images "
        f"[{args.precision}, steps={args.num_inference_steps}, gs={args.guidance_scale}] ..."
    )

    indices = list(range(s_count))
    with accelerator.split_between_processes(indices) as local_indices:
        times = []
        for idx, i in enumerate(local_indices):
            save_path = os.path.join(save_dir, f"ref_{i}.png")
            if os.path.exists(save_path):
                continue

            gen = torch.Generator(device=device).manual_seed(args.seed + i)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            img = pipe(
                prompts[i],
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=gen,
            ).images[0]
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

            img.save(save_path)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(local_indices):
                avg = sum(times) / len(times)
                remaining = (len(local_indices) - idx - 1) * avg
                print(
                    f"[GPU {accelerator.process_index}] "
                    f"{idx+1}/{len(local_indices)} done "
                    f"({avg:.2f}s/img, ~{remaining/60:.1f}min left)",
                    flush=True,
                )

    accelerator.wait_for_everyone()

    # ── 완료 요약 ─────────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        total = sum(
            1 for i in range(s_count)
            if os.path.exists(os.path.join(save_dir, f"ref_{i}.png"))
        )
        accelerator.print(
            f"\n✅ Done: {total}/{s_count} images saved"
            f"\n   Path: {save_dir}"
            f"\n   Tag:  {tag}"
        )


if __name__ == "__main__":
    main()
