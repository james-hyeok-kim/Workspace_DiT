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
from diffusers import PixArtSigmaPipeline, DPMSolverMultistepScheduler
from transformers import CLIPModel, CLIPProcessor

from deepcache_utils import (
    install_deepcache,
    calibrate_cache_lora,
    calibrate_cache_lora_phased,
    calibrate_cache_lora_timestep,
    calibrate_cache_lora_blockwise,
    calibrate_cache_lora_svdaware,
    calibrate_cache_lora_teacherforced,
)
from nonlinear_corrector import calibrate_nonlinear_corrector
from eval_utils import get_prompts, generate_and_evaluate
from quant_methods import (
    apply_rtn_quantization,
    apply_svdquant_quantization,
    apply_fp4dit_quantization,
    apply_hqdit_quantization,
    apply_convrot_quantization,
    apply_sixbit_quantization,
    strip_svdquant_lora,
)


# ---------------------------------------------------------------------------
# DeepCache 기본 설정 (기존 실험에서 검증된 최적 설정)
# ---------------------------------------------------------------------------
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
                        choices=["RTN", "SVDQUANT", "FP4DIT", "HQDIT", "CONVROT", "SIXBIT"],
                        help="양자화 방법 선택")
    parser.add_argument("--cache_mode", type=str, default="none",
                        choices=["none", "deepcache", "cache_lora",
                                 "cache_lora_phase", "cache_lora_ts",
                                 "cache_lora_block", "cache_lora_svd",
                                 "cache_lora_tf",
                                 "cache_nl_gelu", "cache_nl_mlp",
                                 "cache_nl_res", "cache_nl_film"],
                        help="캐시 모드")
    parser.add_argument("--nl_mid_dim", type=int, default=32,
                        help="Nonlinear corrector mid dimension (options 2-4)")
    parser.add_argument("--num_steps", type=int, default=20,
                        choices=[5, 10, 15, 20],
                        help="inference step 수")
    parser.add_argument("--cache_start", type=int, default=8,
                        help="DeepCache region start block index (default: 8)")
    parser.add_argument("--cache_end",   type=int, default=20,
                        help="DeepCache region end block index exclusive (default: 20)")

    # ── 공통 설정 ─────────────────────────────────────────────────────────────
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--test_run", action="store_true",
                        help="2 samples로 smoke test")
    parser.add_argument("--model_path", type=str,
                        default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")
    parser.add_argument("--dataset_name", type=str, default="MJHQ",
                        choices=["MJHQ", "sDCI"])
    parser.add_argument("--ref_dir", type=str,
                        default="/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/fp16_steps{steps}/MJHQ")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="이미지 저장 루트. 미지정 시 /data/jameskimh/james_dit_pixart_sigma_xl_mjhq/{quant_method}")
    parser.add_argument("--guidance_scale", type=float, default=4.5)

    # ── 양자화 하이퍼파라미터 ─────────────────────────────────────────────────
    parser.add_argument("--deepcache_interval", type=int, default=2,
                        help="DeepCache temporal interval (default: 2)")
    parser.add_argument("--disable_svdquant_lr", action="store_true",
                        help="Strip SVDQuant internal LoRA (rank-32) after quantization")
    parser.add_argument("--lowrank",       type=int,   default=32,
                        help="SVDQuant / GPTQ / RPCA low-rank 차원")
    parser.add_argument("--alpha",         type=float, default=0.5,
                        help="SmoothQuant alpha")
    parser.add_argument("--block_size",    type=int,   default=16,
                        help="NVFP4 group size (default: 16, fixed for all methods)")
    parser.add_argument("--lora_rank",     type=int,   default=8,
                        help="Cache-LoRA corrector rank (cache_mode=cache_lora only)")
    parser.add_argument("--lora_calib",    type=int,   default=4,
                        help="Cache-LoRA calibration sample count")
    parser.add_argument("--calib_seed_offset", type=int, default=1000,
                        help="Seed base for calibration (must not overlap with eval seeds 42+i)")
    args = parser.parse_args()
    args.block_size = 16  # NVFP4 group size, fixed for all methods
    if args.save_dir is None:
        args.save_dir = f"/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/{args.quant_method}"

    cache_start = args.cache_start
    cache_end   = args.cache_end

    accelerator = Accelerator()
    device = accelerator.device

    s_count = 2 if args.test_run else args.num_samples
    prompts = get_prompts(s_count, args.dataset_name)
    s_count = len(prompts)
    p_count = 2 if args.test_run else min(64, s_count)
    t_count = args.num_steps

    # 결과 디렉토리
    calib_tag = f"_cal{args.lora_calib}" if args.lora_calib != 4 else ""
    if "cache_nl" in args.cache_mode:
        nl_type = args.cache_mode.replace("cache_nl_", "")
        cache_tag = f"nl_{nl_type}_r{args.lora_rank}_m{args.nl_mid_dim}{calib_tag}"
    elif "cache_lora" in args.cache_mode:
        mode_short = args.cache_mode.replace("cache_lora", "cl")
        cache_tag  = f"{mode_short}_r{args.lora_rank}{calib_tag}"
    else:
        cache_tag = args.cache_mode
    method_tag = args.quant_method
    if getattr(args, 'disable_svdquant_lr', False):
        method_tag += "_nolr"
    interval_tag = f"_i{args.deepcache_interval}" if args.deepcache_interval != 2 else ""
    if args.cache_mode == "none":
        run_tag = f"{method_tag}_{cache_tag}{interval_tag}_steps{t_count}"
    else:
        run_tag = f"{method_tag}_{cache_tag}{interval_tag}_c{cache_start}-{cache_end}_steps{t_count}"
    # ref_dir may contain {steps} placeholder, e.g. "/data/.../fp16_steps{steps}/MJHQ"
    ref_dir_resolved = args.ref_dir.replace("{steps}", str(t_count))
    if "{steps}" in args.ref_dir:
        # path already includes dataset subdirectory
        dataset_ref_dir = ref_dir_resolved
    else:
        dataset_ref_dir = os.path.join(ref_dir_resolved, args.dataset_name)
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
        pipe_ref = PixArtSigmaPipeline.from_pretrained(
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

    pipe = PixArtSigmaPipeline.from_pretrained(
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
        if args.disable_svdquant_lr:
            if accelerator.is_main_process:
                print("  [SVDQuant] Stripping internal LoRA (--disable_svdquant_lr)...")
            strip_svdquant_lora(pipe.transformer)
            accelerator.wait_for_everyone()

    elif args.quant_method == "FP4DIT":
        apply_fp4dit_quantization(
            pipe, transformer, accelerator, prompts, p_count, t_count, device, args
        )

    elif args.quant_method == "HQDIT":
        apply_hqdit_quantization(
            pipe, transformer, accelerator, prompts, p_count, t_count, device, args
        )

    elif args.quant_method == "CONVROT":
        apply_convrot_quantization(
            pipe, transformer, accelerator, prompts, p_count, t_count, device, args
        )

    elif args.quant_method == "SIXBIT":
        apply_sixbit_quantization(
            pipe, transformer, accelerator, prompts, p_count, t_count, device, args
        )

    accelerator.wait_for_everyone()

    # -----------------------------------------------------------------------
    # Phase 2.5: Cache-LoRA calibration
    # -----------------------------------------------------------------------
    corrector_A = corrector_B = None
    phase_correctors = phase_t_count = None
    step_scales = None
    block_correctors = None
    svd_corrector_A = svd_corrector_B = None
    nl_corrector = None
    calib_time_sec = None

    if "cache_nl" in args.cache_mode:
        n_calib = 2 if args.test_run else args.lora_calib
        nl_type = args.cache_mode.replace("cache_nl_", "")  # gelu/mlp/res/film
        accelerator.print(
            f"\n[Phase 2.5] Nonlinear corrector calibration "
            f"(type={nl_type}, rank={args.lora_rank}, mid={args.nl_mid_dim}, "
            f"calib_samples={n_calib})..."
        )
        # corrector weights cache path — reuse across num_samples runs
        _corrector_cache_dir = os.path.join(
            "/data/jameskimh/james_dit_pixart_sigma_xl_mjhq_cache_adapter",
            args.quant_method,
        )
        _corrector_save_path = os.path.join(
            _corrector_cache_dir,
            f"nl_{nl_type}_r{args.lora_rank}_m{args.nl_mid_dim}"
            f"_cs{cache_start}_ce{cache_end}_steps{t_count}"
            f"_cal{args.lora_calib}_seed{args.calib_seed_offset}.pt"
        )

        nl_corrector, calib_time_sec = calibrate_nonlinear_corrector(
            pipe=pipe,
            transformer=pipe.transformer,
            cache_start=cache_start,
            cache_end=cache_end,
            cache_interval=args.deepcache_interval,
            prompts=prompts,
            num_calib=n_calib,
            t_count=t_count,
            guidance_scale=args.guidance_scale,
            device=device,
            corrector_type=nl_type,
            rank=args.lora_rank,
            mid_dim=args.nl_mid_dim,
            calib_seed_offset=args.calib_seed_offset,
            save_path=_corrector_save_path,
        )
        accelerator.print(f"  Calibration time: {calib_time_sec:.1f}s")

    elif "cache_lora" in args.cache_mode:
        n_calib = 2 if args.test_run else args.lora_calib
        accelerator.print(
            f"\n[Phase 2.5] Cache-LoRA calibration "
            f"(mode={args.cache_mode}, rank={args.lora_rank}, "
            f"calib_samples={n_calib})..."
        )

        _calib_kwargs = dict(
            pipe=pipe,
            transformer=pipe.transformer,
            cache_start=cache_start,
            cache_end=cache_end,
            cache_interval=args.deepcache_interval,
            prompts=prompts,
            num_calib=n_calib,
            t_count=t_count,
            guidance_scale=args.guidance_scale,
            device=device,
            rank=args.lora_rank,
            calib_seed_offset=args.calib_seed_offset,
        )

        if args.cache_mode == "cache_lora":
            corrector_A, corrector_B, calib_time_sec = calibrate_cache_lora(**_calib_kwargs)

        elif args.cache_mode == "cache_lora_phase":
            phase_correctors, phase_t_count, calib_time_sec = calibrate_cache_lora_phased(**_calib_kwargs)

        elif args.cache_mode == "cache_lora_ts":
            corrector_A, corrector_B, step_scales, calib_time_sec = calibrate_cache_lora_timestep(**_calib_kwargs)

        elif args.cache_mode == "cache_lora_block":
            block_correctors, calib_time_sec = calibrate_cache_lora_blockwise(**_calib_kwargs)

        elif args.cache_mode == "cache_lora_svd":
            svd_corrector_A, svd_corrector_B, calib_time_sec = calibrate_cache_lora_svdaware(**_calib_kwargs)

        elif args.cache_mode == "cache_lora_tf":
            corrector_A, corrector_B, calib_time_sec = calibrate_cache_lora_teacherforced(
                **_calib_kwargs,
                full_steps_set=DEEPCACHE_FULL_STEPS,
            )

        accelerator.print(f"  Calibration time: {calib_time_sec:.1f}s")

    accelerator.wait_for_everyone()

    # -----------------------------------------------------------------------
    # Phase 3: DeepCache 설치 (cache_mode == "deepcache" or "cache_lora")
    # -----------------------------------------------------------------------
    cache_state = None

    _cache_modes_with_dc = ("deepcache", "cache_lora", "cache_lora_phase",
                             "cache_lora_ts", "cache_lora_block", "cache_lora_svd",
                             "cache_lora_tf",
                             "cache_nl_gelu", "cache_nl_mlp",
                             "cache_nl_res", "cache_nl_film")
    if args.cache_mode in _cache_modes_with_dc:
        accelerator.print(
            f"\n[Phase 3] Installing DeepCache [{args.cache_mode}] "
            f"(interval={args.deepcache_interval}, "
            f"blocks=[{cache_start},{cache_end}))..."
        )
        cache_state = install_deepcache(
            pipe.transformer,
            cache_start=cache_start,
            cache_end=cache_end,
            cache_interval=args.deepcache_interval,
            full_steps_set=DEEPCACHE_FULL_STEPS,
        )

        # Attach correctors based on mode
        if nl_corrector is not None:
            cache_state.nl_corrector = nl_corrector
            cache_state.nl_needs_t = (args.cache_mode == "cache_nl_film")
            cache_state.nl_t_count = t_count
            n_params = sum(p.numel() for p in nl_corrector.parameters())
            accelerator.print(f"  Nonlinear corrector attached "
                              f"(type={args.cache_mode.replace('cache_nl_','')}, params={n_params:,})")
        elif phase_correctors is not None:
            cache_state.phase_correctors = phase_correctors
            cache_state.phase_t_count    = phase_t_count
            accelerator.print(f"  Phase-binned corrector attached ({len(phase_correctors)} phases, rank={args.lora_rank})")
        elif block_correctors is not None:
            cache_state.block_correctors = block_correctors
            accelerator.print(f"  Block-specific corrector attached ({len(block_correctors)} blocks, rank={args.lora_rank})")
        elif svd_corrector_A is not None:
            cache_state.svd_corrector_A = svd_corrector_A
            cache_state.svd_corrector_B = svd_corrector_B
            accelerator.print(f"  SVD-Aware corrector attached (rank={args.lora_rank})")
        elif corrector_A is not None:
            cache_state.corrector_A = corrector_A
            cache_state.corrector_B = corrector_B
            if step_scales is not None:
                cache_state.step_scales = step_scales
                accelerator.print(f"  TS corrector attached (rank={args.lora_rank}, {len(step_scales)} scales)")
            else:
                accelerator.print(f"  Cache-LoRA corrector attached (rank={args.lora_rank})")

        # speedup 추정
        n_deep   = cache_end - cache_start
        n_total  = len(pipe.transformer.transformer_blocks)
        n_always = n_total - n_deep
        avg_blocks = (n_total + n_always * (args.deepcache_interval - 1)) / args.deepcache_interval
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
        use_cache = args.cache_mode != "none"
        result = {
            "quant_method":    args.quant_method,
            "cache_mode":      args.cache_mode,
            "lora_rank":       args.lora_rank if "cache_lora" in args.cache_mode else None,
            "num_steps":       t_count,
            "num_samples":     s_count,
            "lowrank":         args.lowrank,
            "speedup_est":     round(speedup_est, 3),
            "deepcache_start": cache_start if use_cache else None,
            "deepcache_end":   cache_end   if use_cache else None,
            "deepcache_interval": args.deepcache_interval if use_cache else None,
            "calib_time_sec":     round(calib_time_sec, 2) if calib_time_sec is not None else None,
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
                      "time_per_image_sec", "calib_time_sec"]
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
        if calib_time_sec is not None:
            print(f"  Calib time   : {calib_time_sec:.1f}s")
        print(f"  Saved to     : {dataset_save_dir}")
        print(f"{'='*65}\n")

    accelerator.wait_for_everyone()

    # Cleanup
    del pipe
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
