"""
eval_utils.py

공통 평가 유틸리티: 프롬프트 로딩, 이미지 생성 + 메트릭 계산.

Extracted from pixart_caching/pixart_deepcache_experiment.py.
"""

import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from datasets import load_dataset
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from deepcache_utils import DeepCacheState


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def get_prompts(num_samples: int, dataset_name: str = "MJHQ") -> list[str]:
    if dataset_name == "MJHQ":
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
        fallback = [
            "A professional high-quality photo of a futuristic city with neon lights",
            "A beautiful landscape of mountains during sunset, cinematic lighting",
            "A cute robot holding a flower in a field, highly detailed digital art",
            "A gourmet burger with melting cheese and fresh vegetables on a wooden table",
            "An astronaut walking on a purple planet's surface under a starry sky",
        ]
        return (fallback * (num_samples // len(fallback) + 1))[:num_samples]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def generate_and_evaluate(
    pipe,
    cache_state: DeepCacheState | None,
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
    guidance_scale: float = 4.5,
) -> dict:
    """
    Generate images with pipe (+ optional DeepCache) and compute metrics.
    Returns dict with FID, IS, PSNR, SSIM, LPIPS, CLIP, time_per_image_sec.
    """
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

            if cache_state is not None:
                cache_state.reset()

            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            q_img = pipe(
                prompts[i], num_inference_steps=t_count,
                guidance_scale=guidance_scale, generator=gen,
            ).images[0]
            torch.cuda.synchronize(device)
            local_times.append(time.perf_counter() - t0)

            if cache_state is not None:
                cache_state.next_image()

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

    clip_score = None
    if accelerator.is_main_process and clip_model is not None:
        scores = []
        for i in range(s_count):
            q_img = Image.open(os.path.join(save_dir, f"sample_{i}.png")).convert("RGB")
            inputs = clip_processor(
                text=[prompts[i]], images=q_img,
                return_tensors="pt", padding=True, truncation=True,
            ).to(device)
            with torch.no_grad():
                scores.append(float(clip_model(**inputs).logits_per_image.item()))
        clip_score = float(np.mean(scores))

    return {
        "fid":                res_fid,
        "is":                 res_is,
        "psnr":               res_psnr,
        "ssim":               res_ssim,
        "lpips":              res_lpips,
        "clip":               clip_score,
        "time_per_image_sec": avg_time,
    }
