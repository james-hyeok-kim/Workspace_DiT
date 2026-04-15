"""
pixart_tome_experiment.py
Phase 1: Token Merging (ToMe) FP16 sweep 실험.

여러 merge_ratio × block 범위 조합을 순서대로 평가하여
FID, IS, PSNR, SSIM, LPIPS, CLIP Score 및 속도를 비교한다.

CLI:
  --merge_ratio    0.2       # 각 block에서 merge할 토큰 비율 (0.0~0.5)
  --block_start    0
  --block_end      28
  --sweep                    # 미리 정의된 config 전체 sweep
  --num_samples    20
  --num_inference_steps 20
  --test_run                 # 2-sample smoke test
  --ref_dir        ./ref_images
  --save_dir       ./results
  --model_path     "PixArt-alpha/PixArt-XL-2-1024-MS"
  --dataset_name   MJHQ

결과:
  results/{dataset}/tome/{config_tag}/metrics.json
  results/{dataset}/tome/{config_tag}/metrics.csv
  results/{dataset}/tome/sweep_summary.json    (--sweep 시)
  results/{dataset}/tome/sweep_summary.csv     (--sweep 시)
"""

import os
import sys
import time
import gc
import csv
import json
import argparse

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from accelerate import Accelerator

from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler

from datasets import load_dataset
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from transformers import CLIPModel, CLIPProcessor

# 같은 디렉토리의 tome_patch
sys.path.insert(0, os.path.dirname(__file__))
from tome_patch import install_tome
from tome_core import compute_r


# ---------------------------------------------------------------------------
# Sweep configs: (merge_ratio, block_start, block_end, tag)
# ---------------------------------------------------------------------------

SWEEP_CONFIGS = [
    (0.0,  0, 28, "FP16_baseline"),   # ToMe 없는 FP16 baseline
    (0.10, 0, 28, "R10_ALL"),
    (0.20, 0, 28, "R20_ALL"),
    (0.30, 0, 28, "R30_ALL"),
    (0.50, 0, 28, "R50_ALL"),
    (0.20, 4, 24, "R20_MID"),
    (0.30, 4, 24, "R30_MID"),
]


# ---------------------------------------------------------------------------
# Prompts
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
        print(f"Warning: Prompt loading failed: {e}", flush=True)
        return ["A professional high-quality photo"] * num_samples


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def generate_and_evaluate(
    pipe,
    t_count: int,
    prompts: list[str],
    s_count: int,
    ref_dir: str,
    save_dir: str,
    device,
    accelerator,
    clip_model=None,
    clip_processor=None,
    config_tag: str = "",
    fp16_time_sec: float | None = None,
) -> dict:
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
    accelerator.wait_for_everyone()

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

            save_path = os.path.join(save_dir, f"sample_{i}.png")
            q_img.save(save_path)

            ref_path = os.path.join(ref_dir, f"ref_{i}.png")
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

            print(f"[GPU {accelerator.process_index}] {config_tag} "
                  f"sample_{i} ({local_times[-1]:.2f}s)", flush=True)

    accelerator.wait_for_everyone()

    res_psnr  = float(psnr_m.compute())
    res_ssim  = float(ssim_m.compute())
    res_lpips = float(lpips_m.compute())
    res_is, _ = is_m.compute()
    res_is    = float(res_is)
    res_fid   = float(fid_m.compute())
    avg_time  = float(np.mean(local_times)) if local_times else 0.0

    speedup_vs_fp16 = (fp16_time_sec / avg_time) if (fp16_time_sec and avg_time > 0) else None

    clip_score = None
    if accelerator.is_main_process and clip_model is not None:
        scores = []
        for i in range(s_count):
            img_path = os.path.join(save_dir, f"sample_{i}.png")
            if not os.path.exists(img_path):
                continue
            q_img = Image.open(img_path).convert("RGB")
            inputs = clip_processor(
                text=[prompts[i]], images=q_img,
                return_tensors="pt", padding=True,
            ).to(device)
            with torch.no_grad():
                scores.append(float(clip_model(**inputs).logits_per_image.item()))
        clip_score = float(np.mean(scores)) if scores else None

    return {
        "fid":                res_fid,
        "is":                 res_is,
        "psnr":               res_psnr,
        "ssim":               res_ssim,
        "lpips":              res_lpips,
        "clip":               clip_score,
        "time_per_image_sec": avg_time,
        "speedup_vs_fp16":    speedup_vs_fp16,
    }


# ---------------------------------------------------------------------------
# CSV 저장 헬퍼
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "config_tag", "merge_ratio", "block_start", "block_end",
    "num_steps", "num_samples",
    "fid", "is", "psnr", "ssim", "lpips", "clip",
    "time_per_image_sec", "speedup_vs_fp16",
]


def save_csv(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PixArt ToMe (FP16) sweep")
    parser.add_argument("--merge_ratio",          type=float, default=0.2)
    parser.add_argument("--block_start",          type=int,   default=0)
    parser.add_argument("--block_end",            type=int,   default=28)
    parser.add_argument("--sweep",                action="store_true")
    parser.add_argument("--num_samples",          type=int,   default=20)
    parser.add_argument("--num_inference_steps",  type=int,   default=20)
    parser.add_argument("--test_run",             action="store_true")
    parser.add_argument("--ref_dir",              type=str,   default="./ref_images")
    parser.add_argument("--save_dir",             type=str,   default="./results")
    parser.add_argument("--model_path",           type=str,
                        default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--dataset_name",         type=str,   default="MJHQ",
                        choices=["MJHQ", "sDCI"])
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    s_count = 2 if args.test_run else args.num_samples
    prompts = get_prompts(s_count, args)
    s_count = len(prompts)
    t_count = args.num_inference_steps

    dataset_ref_dir  = os.path.join(args.ref_dir,  args.dataset_name)
    dataset_save_dir = os.path.join(args.save_dir, args.dataset_name, "tome")

    if accelerator.is_main_process:
        os.makedirs(dataset_ref_dir,  exist_ok=True)
        os.makedirs(dataset_save_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Phase 1: FP16 reference 이미지 생성
    # -----------------------------------------------------------------------
    if accelerator.is_main_process:
        accelerator.print(f"[ToMe] Generating FP16 reference images ({t_count} steps)...")
        pipe_ref = PixArtAlphaPipeline.from_pretrained(
            args.model_path, torch_dtype=torch.float16
        ).to(device)

        for i in range(s_count):
            ref_path = os.path.join(dataset_ref_dir, f"ref_{i}.png")
            if not os.path.exists(ref_path):
                gen = torch.Generator(device=device).manual_seed(42 + i)
                img = pipe_ref(
                    prompts[i], num_inference_steps=t_count, generator=gen
                ).images[0]
                img.save(ref_path)

        del pipe_ref
        torch.cuda.empty_cache()
        gc.collect()
        accelerator.print(f"  {s_count} refs ready in {dataset_ref_dir}")

    accelerator.wait_for_everyone()

    # -----------------------------------------------------------------------
    # CLIP 모델 로드 (main process)
    # -----------------------------------------------------------------------
    if accelerator.is_main_process:
        clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    else:
        clip_model = clip_processor = None

    # -----------------------------------------------------------------------
    # Phase 2: ToMe 실험 sweep
    # -----------------------------------------------------------------------
    configs_to_run = (
        SWEEP_CONFIGS if args.sweep
        else [(args.merge_ratio, args.block_start, args.block_end,
               f"R{int(args.merge_ratio*100)}_s{args.block_start}_e{args.block_end}")]
    )

    sweep_results: list[dict] = []
    fp16_time_per_img: float | None = None   # FP16 baseline 시간 (speedup 계산용)

    for merge_ratio, blk_start, blk_end, config_tag in configs_to_run:
        N_TOKENS = 4096
        r = compute_r(N_TOKENS, merge_ratio)
        effective_tokens = N_TOKENS - r
        speedup_attn_est = (N_TOKENS / effective_tokens) ** 2 if effective_tokens > 0 else 1.0

        accelerator.print(
            f"\n=== Config: {config_tag} | ratio={merge_ratio:.2f} "
            f"| blocks [{blk_start}, {blk_end}) "
            f"| tokens {N_TOKENS}→{effective_tokens} "
            f"| attn speedup est ≈ {speedup_attn_est:.2f}x ==="
        )

        # 각 config마다 신선한 FP16 파이프라인 로드
        pipe = PixArtAlphaPipeline.from_pretrained(
            args.model_path, torch_dtype=torch.float16
        ).to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        # ToMe 설치 (merge_ratio > 0인 경우)
        if merge_ratio > 0.0:
            tome_cfg = install_tome(pipe.transformer, merge_ratio, blk_start, blk_end)
            accelerator.print(f"  ToMe installed: {tome_cfg}")
        else:
            accelerator.print("  ToMe NOT installed (FP16 baseline)")

        run_save_dir = os.path.join(dataset_save_dir, config_tag)
        metrics = generate_and_evaluate(
            pipe, t_count, prompts, s_count,
            dataset_ref_dir, run_save_dir,
            device, accelerator,
            clip_model, clip_processor,
            config_tag=config_tag,
            fp16_time_sec=fp16_time_per_img,
        )

        # FP16 baseline 시간 저장 (첫 번째 config = FP16_baseline)
        if merge_ratio == 0.0:
            fp16_time_per_img = metrics["time_per_image_sec"]

        entry = {
            "config_tag":         config_tag,
            "merge_ratio":        merge_ratio,
            "block_start":        blk_start,
            "block_end":          blk_end,
            "effective_tokens":   effective_tokens,
            "speedup_attn_est":   round(speedup_attn_est, 3),
            "num_steps":          t_count,
            "num_samples":        s_count,
            **metrics,
        }
        sweep_results.append(entry)

        # per-config 저장
        if accelerator.is_main_process:
            os.makedirs(run_save_dir, exist_ok=True)
            # JSON
            with open(os.path.join(run_save_dir, "metrics.json"), "w") as f:
                json.dump({"config": entry}, f, indent=4)
            # CSV
            save_csv(os.path.join(run_save_dir, "metrics.csv"), [entry])

            accelerator.print(
                f"  FID={metrics['fid']:.2f} | IS={metrics['is']:.3f} | "
                f"PSNR={metrics['psnr']:.2f} | CLIP={metrics['clip']} | "
                f"time={metrics['time_per_image_sec']:.2f}s | "
                f"speedup_vs_fp16={metrics['speedup_vs_fp16']}"
            )

        del pipe
        torch.cuda.empty_cache()
        gc.collect()
        accelerator.wait_for_everyone()

    # -----------------------------------------------------------------------
    # Sweep summary 저장
    # -----------------------------------------------------------------------
    if accelerator.is_main_process:
        summary = {
            "method":        "ToMe FP16",
            "dataset":       args.dataset_name,
            "num_samples":   s_count,
            "num_steps":     t_count,
            "sweep_results": sweep_results,
        }
        summary_json = os.path.join(dataset_save_dir, "sweep_summary.json")
        summary_csv  = os.path.join(dataset_save_dir, "sweep_summary.csv")

        with open(summary_json, "w") as f:
            json.dump(summary, f, indent=4)
        save_csv(summary_csv, sweep_results)

        print(f"\n{'=' * 100}")
        print(f"  ToMe FP16 Sweep 완료 | dataset={args.dataset_name} | n={s_count}")
        print(f"{'=' * 100}")
        print(
            f"\n{'config_tag':<22} | {'ratio':>5} | {'blks':>7} | "
            f"{'FID':>8} | {'IS':>7} | {'PSNR':>7} | "
            f"{'sec/img':>8} | {'speedup':>8}"
        )
        print("-" * 90)
        for r in sweep_results:
            spd = f"{r['speedup_vs_fp16']:.3f}x" if r["speedup_vs_fp16"] else "  N/A  "
            print(
                f"{r['config_tag']:<22} | {r['merge_ratio']:>5.2f} | "
                f"[{r['block_start']:>2},{r['block_end']:>2}) | "
                f"{r['fid']:>8.2f} | {r['is']:>7.3f} | {r['psnr']:>7.2f} | "
                f"{r['time_per_image_sec']:>8.2f} | {spd:>8}"
            )
        print(f"\n  JSON: {summary_json}")
        print(f"  CSV:  {summary_csv}")

    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
