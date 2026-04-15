"""
pixart_nvfp4_cache_compare.py

4가지 NVFP4 양자화 방법 × DeepCache 호환성 비교 실험.

연구 질문:
  Cache-friendliness가 NVFP4 format 자체의 특성인가, 양자화 알고리즘에 의존하는가?

실험 매트릭스:
  Method: RTN | SVDQUANT | MRGPTQ | FOUROVERSIX
  Cache:  none | deepcache (interval=2, blocks[8,20))
  Steps:  15 | 20

핵심 분석 지표:
  cache_penalty = FID(with_cache) - FID(no_cache)
  → penalty가 method 간 유사: format-level property
  → penalty가 method 간 상이: algorithm이 cache-friendliness 결정

CLI 예시:
  # RTN + no cache, 20-step
  accelerate launch pixart_nvfp4_cache_compare.py --quant_method RTN --cache_mode none --num_steps 20

  # MR-GPTQ + deepcache, 20-step
  accelerate launch pixart_nvfp4_cache_compare.py --quant_method MRGPTQ --cache_mode deepcache --num_steps 20

  # Four Over Six + no cache, 15-step
  accelerate launch pixart_nvfp4_cache_compare.py --quant_method FOUROVERSIX --cache_mode none --num_steps 15

  # smoke test (2 samples)
  accelerate launch pixart_nvfp4_cache_compare.py --quant_method MRGPTQ --cache_mode deepcache --test_run

결과 위치:
  results/{dataset}/{quant_method}_{cache_mode}_steps{N}/
    sample_*.png, metrics.json, summary.csv
"""

import os
import gc
import csv
import json
import argparse

import torch
from accelerate import Accelerator
from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler
from transformers import CLIPModel, CLIPProcessor

from deepcache_utils import install_deepcache
from eval_utils import get_prompts, generate_and_evaluate
from quant_methods import (
    apply_rtn_quantization,
    apply_svdquant_quantization,
    apply_mrgptq_quantization,
    apply_fouroversix_quantization,
)


# ---------------------------------------------------------------------------
# DeepCache 기본 설정 (기존 실험에서 검증된 최적 설정)
# ---------------------------------------------------------------------------
DEEPCACHE_INTERVAL = 2
DEEPCACHE_START    = 8
DEEPCACHE_END      = 20
DEEPCACHE_FULL_STEPS = {0}   # 첫 step은 항상 full


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NVFP4 Quantization × DeepCache 호환성 비교"
    )

    # ── 실험 선택 ──────────────────────────────────────────────────────────────
    parser.add_argument("--quant_method", type=str, required=True,
                        choices=["RTN", "SVDQUANT", "MRGPTQ", "FOUROVERSIX"],
                        help="양자화 방법 선택")
    parser.add_argument("--cache_mode", type=str, default="none",
                        choices=["none", "deepcache"],
                        help="캐시 모드: none (no caching) | deepcache")
    parser.add_argument("--num_steps", type=int, default=20,
                        choices=[15, 20],
                        help="inference step 수")

    # ── 공통 설정 ─────────────────────────────────────────────────────────────
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--test_run", action="store_true",
                        help="2 samples로 smoke test")
    parser.add_argument("--model_path", type=str,
                        default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--dataset_name", type=str, default="MJHQ",
                        choices=["MJHQ", "sDCI"])
    parser.add_argument("--ref_dir", type=str, default="./ref_images")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--guidance_scale", type=float, default=4.5)

    # ── 양자화 하이퍼파라미터 ─────────────────────────────────────────────────
    parser.add_argument("--lowrank",       type=int,   default=32,
                        help="SVDQuant / GPTQ / RPCA low-rank 차원")
    parser.add_argument("--alpha",         type=float, default=0.5,
                        help="SmoothQuant alpha")
    parser.add_argument("--block_size",    type=int,   default=16,
                        help="NVFP4 group size (default: 16, fixed for all methods)")
    args = parser.parse_args()
    args.block_size = 16  # NVFP4 group size, fixed for all methods

    accelerator = Accelerator()
    device = accelerator.device

    s_count = 2 if args.test_run else args.num_samples
    prompts = get_prompts(s_count, args.dataset_name)
    s_count = len(prompts)
    p_count = 2 if args.test_run else min(64, s_count)
    t_count = args.num_steps

    # 결과 디렉토리
    run_tag     = f"{args.quant_method}_{args.cache_mode}_steps{t_count}"
    dataset_ref_dir  = os.path.join(args.ref_dir,  args.dataset_name)
    dataset_save_dir = os.path.join(args.save_dir, args.dataset_name, run_tag)

    if accelerator.is_main_process:
        os.makedirs(dataset_ref_dir,  exist_ok=True)
        os.makedirs(dataset_save_dir, exist_ok=True)
        print(f"\n{'='*65}")
        print(f"  Quant Method : {args.quant_method}")
        print(f"  Cache Mode   : {args.cache_mode}")
        print(f"  Steps        : {t_count}")
        print(f"  Samples      : {s_count}  (calib: {p_count})")
        print(f"  Save dir     : {dataset_save_dir}")
        print(f"{'='*65}\n")

    # -----------------------------------------------------------------------
    # Phase 1: FP16 reference 생성
    # -----------------------------------------------------------------------
    if accelerator.is_main_process:
        accelerator.print(f"[Phase 1] Generating FP16 references ({t_count}-step)...")
        pipe_ref = PixArtAlphaPipeline.from_pretrained(
            args.model_path, torch_dtype=torch.float16
        ).to(device)

        for i in range(s_count):
            ref_path = os.path.join(dataset_ref_dir, f"ref_{i}.png")
            if not os.path.exists(ref_path):
                gen = torch.Generator(device=device).manual_seed(42 + i)
                img = pipe_ref(
                    prompts[i], num_inference_steps=t_count,
                    guidance_scale=args.guidance_scale, generator=gen
                ).images[0]
                img.save(ref_path)

        del pipe_ref
        torch.cuda.empty_cache()
        gc.collect()
        accelerator.print(f"  {s_count} refs ready in {dataset_ref_dir}")

    accelerator.wait_for_everyone()

    # -----------------------------------------------------------------------
    # Phase 2: 양자화 모델 준비
    # -----------------------------------------------------------------------
    accelerator.print(f"\n[Phase 2] Quantizing with {args.quant_method}...")

    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    transformer = pipe.transformer

    if args.quant_method == "RTN":
        apply_rtn_quantization(
            pipe, transformer, accelerator, prompts, p_count, t_count, device, args
        )

    elif args.quant_method == "SVDQUANT":
        apply_svdquant_quantization(
            pipe, accelerator, prompts, p_count, t_count, device, args
        )
        transformer = pipe.transformer  # mtq가 내부적으로 교체할 수 있음

    elif args.quant_method == "MRGPTQ":
        apply_mrgptq_quantization(
            pipe, transformer, accelerator, prompts, p_count, t_count, device, args
        )

    elif args.quant_method == "FOUROVERSIX":
        apply_fouroversix_quantization(
            pipe, transformer, accelerator, prompts, p_count, t_count, device, args
        )

    accelerator.wait_for_everyone()

    # -----------------------------------------------------------------------
    # Phase 3: DeepCache 설치 (cache_mode == "deepcache")
    # -----------------------------------------------------------------------
    cache_state = None

    if args.cache_mode == "deepcache":
        accelerator.print(
            f"\n[Phase 3] Installing DeepCache "
            f"(interval={DEEPCACHE_INTERVAL}, "
            f"blocks=[{DEEPCACHE_START},{DEEPCACHE_END}))..."
        )
        cache_state = install_deepcache(
            pipe.transformer,
            cache_start=DEEPCACHE_START,
            cache_end=DEEPCACHE_END,
            cache_interval=DEEPCACHE_INTERVAL,
            full_steps_set=DEEPCACHE_FULL_STEPS,
        )

        # speedup 추정
        n_deep   = DEEPCACHE_END - DEEPCACHE_START
        n_total  = len(pipe.transformer.transformer_blocks)
        n_always = n_total - n_deep
        avg_blocks = (n_total + n_always * (DEEPCACHE_INTERVAL - 1)) / DEEPCACHE_INTERVAL
        speedup_est = n_total / avg_blocks
        accelerator.print(f"  Estimated speedup ≈ {speedup_est:.2f}x")
    else:
        accelerator.print("\n[Phase 3] No caching (cache_mode=none).")
        speedup_est = 1.0

    # -----------------------------------------------------------------------
    # CLIP 모델 로드
    # -----------------------------------------------------------------------
    if accelerator.is_main_process:
        clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    else:
        clip_model = clip_processor = None

    # -----------------------------------------------------------------------
    # Phase 4: 이미지 생성 + 메트릭 계산
    # -----------------------------------------------------------------------
    accelerator.print(f"\n[Phase 4] Generating {s_count} images and computing metrics...")

    metrics = generate_and_evaluate(
        pipe=pipe,
        cache_state=cache_state,
        t_count=t_count,
        prompts=prompts,
        s_count=s_count,
        ref_dir=dataset_ref_dir,
        save_dir=dataset_save_dir,
        device=device,
        accelerator=accelerator,
        clip_model=clip_model,
        clip_processor=clip_processor,
        config_tag=run_tag,
        guidance_scale=args.guidance_scale,
    )

    # -----------------------------------------------------------------------
    # 결과 저장
    # -----------------------------------------------------------------------
    if accelerator.is_main_process:
        result = {
            "quant_method":    args.quant_method,
            "cache_mode":      args.cache_mode,
            "num_steps":       t_count,
            "num_samples":     s_count,
            "lowrank":         args.lowrank,
            "speedup_est":     round(speedup_est, 3),
            "deepcache_start": DEEPCACHE_START if args.cache_mode == "deepcache" else None,
            "deepcache_end":   DEEPCACHE_END   if args.cache_mode == "deepcache" else None,
            "deepcache_interval": DEEPCACHE_INTERVAL if args.cache_mode == "deepcache" else None,
            **metrics,
        }

        # metrics.json
        json_path = os.path.join(dataset_save_dir, "metrics.json")
        with open(json_path, "w") as f:
            json.dump(result, f, indent=4)

        # metrics.csv
        csv_path = os.path.join(dataset_save_dir, "metrics.csv")
        csv_fields = ["quant_method", "cache_mode", "num_steps", "speedup_est",
                      "fid", "is", "psnr", "ssim", "lpips", "clip",
                      "time_per_image_sec"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(result)

        print(f"\n{'='*65}")
        print(f"  Config       : {run_tag}")
        print(f"  FID          : {metrics['fid']:.2f}")
        print(f"  IS           : {metrics['is']:.3f}")
        print(f"  PSNR         : {metrics['psnr']:.2f}")
        print(f"  SSIM         : {metrics['ssim']:.4f}")
        print(f"  LPIPS        : {metrics['lpips']:.4f}")
        if metrics["clip"] is not None:
            print(f"  CLIP         : {metrics['clip']:.2f}")
        print(f"  Time/img     : {metrics['time_per_image_sec']:.2f}s")
        print(f"  Speedup est  : {speedup_est:.2f}x")
        print(f"  Saved to     : {dataset_save_dir}")
        print(f"{'='*65}\n")

    accelerator.wait_for_everyone()

    # Cleanup
    del pipe
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
