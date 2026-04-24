"""
pixart_step_sweep.py
NVFP4_SVDQUANT_DEFAULT_CFG에서 inference step 수별 FID/IS/속도 트레이드오프 측정.

각 step count마다:
  - FP16 ref 이미지 (동일 step count로 생성) — PSNR/SSIM/LPIPS/FID 기준
  - NVFP4 quantized 이미지 생성 및 평가
  - GPU 시간 측정 (time/image)

결과: results/{dataset}/step_sweep/summary.json
         results/{dataset}/step_sweep/summary.csv
"""

import os
import time
import gc
import copy
import csv
import json
import argparse

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from accelerate import Accelerator

from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler
import modelopt.torch.quantization as mtq

from datasets import load_dataset
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from transformers import CLIPModel, CLIPProcessor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def get_prompts(num_samples: int, args) -> list[str]:
    if args.dataset_name == "MJHQ":
        path, split, key = "xingjianleng/mjhq30k", "test", "text"
    else:
        path, split, key = "mit-han-lab/svdquant-datasets", "train", "prompt"

    try:
        dataset = load_dataset(path, split=split, streaming=True)
        prompts = []
        for i, entry in enumerate(dataset):
            if i >= num_samples:
                break
            prompts.append(entry[key])
        return prompts
    except Exception as e:
        print(f"Warning: Prompt loading failed: {e}")
        return ["A professional high-quality photo"] * num_samples


# ---------------------------------------------------------------------------
# Per-step-count evaluation
# ---------------------------------------------------------------------------

def evaluate_step_count(
    pipe,
    t_count: int,
    prompts: list[str],
    s_count: int,
    ref_base_dir: str,
    save_base_dir: str,
    dataset_name: str,
    device,
    accelerator,
    clip_model=None,
    clip_processor=None,
) -> dict:
    """Generate quantized images at t_count steps and compute metrics vs FP16 refs."""

    step_ref_dir  = os.path.join(ref_base_dir,  f"steps_{t_count}")
    step_save_dir = os.path.join(save_base_dir, f"fp16_steps{t_count}", dataset_name)

    if accelerator.is_main_process:
        os.makedirs(step_save_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Fresh metric objects per step count
    psnr_m  = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_m  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    is_m    = InceptionScore().to(device)
    fid_m   = FrechetInceptionDistance(feature=2048).to(device)

    local_times: list[float] = []

    indices = list(range(s_count))
    with accelerator.split_between_processes(indices) as local_indices:
        for i in local_indices:
            gen = torch.Generator(device=device).manual_seed(42 + i)

            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            q_img = pipe(prompts[i], num_inference_steps=t_count, generator=gen).images[0]
            torch.cuda.synchronize(device)
            local_times.append(time.perf_counter() - t0)

            save_path = os.path.join(step_save_dir, f"sample_{i}.png")
            q_img.save(save_path)

            ref_path = os.path.join(step_ref_dir, f"ref_{i}.png")
            r_img = Image.open(ref_path).convert("RGB")

            q_ten = ToTensor()(q_img).unsqueeze(0).to(device)
            r_ten = ToTensor()(r_img).unsqueeze(0).to(device)

            psnr_m.update(q_ten, r_ten)
            ssim_m.update(q_ten, r_ten)
            lpips_m.update(q_ten * 2 - 1, r_ten * 2 - 1)

            img_u8 = (q_ten * 255).to(torch.uint8)
            ref_u8 = (r_ten * 255).to(torch.uint8)
            is_m.update(img_u8)
            fid_m.update(ref_u8, real=True)
            fid_m.update(img_u8, real=False)

            print(f"[GPU {accelerator.process_index}] steps={t_count} sample_{i} "
                  f"({local_times[-1]:.2f}s)", flush=True)

    accelerator.wait_for_everyone()

    res_psnr  = float(psnr_m.compute())
    res_ssim  = float(ssim_m.compute())
    res_lpips = float(lpips_m.compute())
    res_is, _ = is_m.compute()
    res_is    = float(res_is)
    res_fid   = float(fid_m.compute())
    avg_time  = float(np.mean(local_times)) if local_times else 0.0

    clip_score = None
    if accelerator.is_main_process and clip_model is not None:
        scores = []
        for i in range(s_count):
            q_img = Image.open(os.path.join(step_save_dir, f"sample_{i}.png")).convert("RGB")
            inputs = clip_processor(
                text=[prompts[i]], images=q_img,
                return_tensors="pt", padding=True
            ).to(device)
            with torch.no_grad():
                scores.append(float(clip_model(**inputs).logits_per_image.item()))
        clip_score = float(np.mean(scores))

    return {
        "steps":              t_count,
        "fid":                res_fid,
        "is":                 res_is,
        "psnr":               res_psnr,
        "ssim":               res_ssim,
        "lpips":              res_lpips,
        "clip":               clip_score,
        "time_per_image_sec": avg_time,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NVFP4 inference step sweep")
    parser.add_argument("--num_samples",  type=int,  default=20)
    parser.add_argument("--test_run",     action="store_true",
                        help="2-sample smoke test")
    parser.add_argument("--ref_dir",      type=str,  default="./ref_images")
    parser.add_argument("--save_dir",     type=str,  default="/data/jameskimh/james_dit_pixart_xl_mjhq")
    parser.add_argument("--model_path",   type=str,
                        default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--dataset_name", type=str,  default="MJHQ",
                        choices=["MJHQ", "sDCI"])
    parser.add_argument("--lowrank",      type=int,  default=32,
                        help="SVDQuant low-rank branch dimension")
    parser.add_argument("--step_counts",  type=str,
                        default="5,8,10,12,15,20",
                        help="Comma-separated inference step counts to sweep")
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    step_counts = [int(x) for x in args.step_counts.split(",")]
    s_target = 2 if args.test_run else args.num_samples
    prompts  = get_prompts(s_target, args)
    s_count  = len(prompts)
    p_count  = 2 if args.test_run else min(64, s_count)

    dataset_ref_dir  = os.path.join(args.ref_dir,  args.dataset_name)
    dataset_save_dir = os.path.join(args.save_dir, "step_sweep", args.dataset_name)

    if accelerator.is_main_process:
        os.makedirs(dataset_ref_dir,  exist_ok=True)
        os.makedirs(dataset_save_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Phase 1: FP16 reference generation (main process, per step count)
    # -----------------------------------------------------------------------
    if accelerator.is_main_process:
        accelerator.print("Generating FP16 reference images per step count...")
        pipe_ref = PixArtAlphaPipeline.from_pretrained(
            args.model_path, torch_dtype=torch.float16
        ).to(device)

        for t in step_counts:
            step_ref_dir = os.path.join(dataset_ref_dir, f"steps_{t}")
            os.makedirs(step_ref_dir, exist_ok=True)

            missing = [
                i for i in range(s_count)
                if not os.path.exists(os.path.join(step_ref_dir, f"ref_{i}.png"))
            ]
            for i in missing:
                gen = torch.Generator(device=device).manual_seed(42 + i)
                img = pipe_ref(
                    prompts[i], num_inference_steps=t, generator=gen
                ).images[0]
                img.save(os.path.join(step_ref_dir, f"ref_{i}.png"))

            accelerator.print(f"  [steps={t}] {s_count} refs ready")

        del pipe_ref
        torch.cuda.empty_cache()
        gc.collect()

    accelerator.wait_for_everyone()

    # -----------------------------------------------------------------------
    # Phase 2: NVFP4 quantization (once)
    # -----------------------------------------------------------------------
    accelerator.print(f"Loading and quantizing model (NVFP4, lowrank={args.lowrank})...")

    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    quant_config = copy.deepcopy(mtq.NVFP4_SVDQUANT_DEFAULT_CFG)
    if "algorithm" in quant_config:
        quant_config["algorithm"]["lowrank"] = args.lowrank

    def forward_loop(model):
        for prompt in prompts[:p_count]:
            pipe(
                prompt,
                num_inference_steps=5,
                generator=torch.Generator(device=device).manual_seed(42),
            )

    with torch.no_grad():
        pipe.transformer = mtq.quantize(
            pipe.transformer, quant_config, forward_loop=forward_loop
        )

    accelerator.wait_for_everyone()
    accelerator.print("Quantization done.")

    # -----------------------------------------------------------------------
    # Phase 3: Step sweep (generate + evaluate per step count)
    # -----------------------------------------------------------------------
    if accelerator.is_main_process:
        clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    else:
        clip_model = clip_processor = None

    all_results: list[dict] = []

    for t in step_counts:
        accelerator.print(f"\n=== Evaluating steps={t} ===")
        result = evaluate_step_count(
            pipe, t, prompts, s_count,
            dataset_ref_dir, args.save_dir, args.dataset_name,
            device, accelerator,
            clip_model, clip_processor,
        )
        all_results.append(result)

        if accelerator.is_main_process:
            accelerator.print(
                f"  steps={t:2d} | FID={result['fid']:7.2f} | IS={result['is']:.3f} "
                f"| PSNR={result['psnr']:.2f} | time={result['time_per_image_sec']:.2f}s"
            )

    # -----------------------------------------------------------------------
    # Save summary
    # -----------------------------------------------------------------------
    if accelerator.is_main_process:
        summary = {
            "method":      "NVFP4_SVDQUANT_DEFAULT_CFG",
            "dataset":     args.dataset_name,
            "num_samples": s_count,
            "lowrank":     args.lowrank,
            "sweep_results": all_results,
        }
        summary_path = os.path.join(dataset_save_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        # CSV 저장
        csv_path = os.path.join(dataset_save_dir, "summary.csv")
        csv_fields = ["steps", "fid", "is", "psnr", "ssim", "lpips", "clip", "time_per_image_sec"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\n✅ Step sweep complete.")
        print(f"   JSON: {summary_path}")
        print(f"   CSV:  {csv_path}")
        print(f"\n{'Steps':>6} | {'FID':>8} | {'IS':>7} | {'PSNR':>7} | {'SSIM':>7} | {'CLIP':>7} | {'sec/img':>8}")
        print("-" * 65)
        for r in all_results:
            print(
                f"{r['steps']:>6} | {r['fid']:>8.2f} | {r['is']:>7.3f} | "
                f"{r['psnr']:>7.2f} | {r['ssim']:>7.4f} | "
                f"{(r['clip'] or 0):>7.2f} | {r['time_per_image_sec']:>8.2f}"
            )

    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
