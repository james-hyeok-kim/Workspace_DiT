"""
pixart_deepcache_experiment.py
NVFP4_SVDQUANT_DEFAULT_CFG + DeepCache 스타일 block-level activation caching.

핵심 아이디어:
  - 28개 transformer block을 3 구역으로 분할:
      shallow  (0 .. cache_start-1)  : 항상 연산
      deep     (cache_start .. cache_end-1) : 캐시 대상
      final    (cache_end .. 27)     : 항상 연산
  - Full step (매 cache_interval step):
      deep block 실행, residual = output - input 저장
  - Cached step:
      hidden_states += cached_residual  (deep block SKIP)
  - 구현: diffusers 소스 수정 없이 transformer.forward monkey-patch

CLI:
  --num_inference_steps  20
  --cache_interval       2       # 매 K step마다 full (나머지 cached)
  --cache_start          4
  --cache_end            24
  --full_steps           "0"     # 항상 full로 실행할 step 번호 (comma-sep)
  --sweep                        # ablation: 여러 config 자동 sweep
  --num_samples          20
  --test_run                     # 2-sample smoke test

결과: results/{dataset}/deepcache/interval{K}_s{start}_e{end}/metrics.json
      results/{dataset}/deepcache/sweep_summary.json  (--sweep 시)
"""

import os
import time
import gc
import copy
import json
import types
import argparse

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from accelerate import Accelerator

from diffusers import PixArtAlphaPipeline, DPMSolverMultistepScheduler
from diffusers.models.modeling_outputs import Transformer2DModelOutput
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
# DeepCache state & monkey-patch
# ---------------------------------------------------------------------------

class DeepCacheState:
    """Per-image-generation caching state. Reset before each pipe() call."""

    def __init__(self):
        self.step_idx: int = 0
        self.deep_residual_cache: torch.Tensor | None = None

    def reset(self):
        self.step_idx = 0
        self.deep_residual_cache = None


def _make_cached_forward(cache_start: int, cache_end: int,
                          cache_interval: int, full_steps_set: set,
                          state: DeepCacheState):
    """
    Return a new forward function (unbound method) for PixArtTransformer2DModel
    with deep block caching logic inserted in the block loop.

    The preamble / postamble are copied verbatim from
      diffusers/models/transformers/pixart_transformer_2d.py (forward, lines 227-362).
    """

    def cached_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        added_cond_kwargs: dict | None = None,
        cross_attention_kwargs: dict | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        # ---- preamble (identical to original) --------------------------------
        if self.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError(
                "`added_cond_kwargs` cannot be None when using additional "
                "conditions for `adaln_single`."
            )

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        batch_size = hidden_states.shape[0]
        height = hidden_states.shape[-2] // self.config.patch_size
        width  = hidden_states.shape[-1] // self.config.patch_size

        hidden_states = self.pos_embed(hidden_states)

        # adaln_single overwrites the `timestep` name in the original;
        # we rename to avoid shadowing the raw timestep arg.
        timestep_emb, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs,
            batch_size=batch_size, hidden_dtype=hidden_states.dtype,
        )

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1]
            )

        # ---- block loop with caching ----------------------------------------
        step_idx = state.step_idx
        is_full  = (
            step_idx in full_steps_set
            or state.deep_residual_cache is None  # first step always full
            or (step_idx % cache_interval == 0)
        )
        state.step_idx += 1

        block_kwargs = dict(
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep_emb,
            cross_attention_kwargs=cross_attention_kwargs,
            class_labels=None,
        )

        # Shallow blocks — always run
        for block in self.transformer_blocks[:cache_start]:
            hidden_states = block(hidden_states, **block_kwargs)

        if is_full:
            h_before_deep = hidden_states.clone()
            for block in self.transformer_blocks[cache_start:cache_end]:
                hidden_states = block(hidden_states, **block_kwargs)
            # Store residual that deep blocks contributed
            state.deep_residual_cache = hidden_states - h_before_deep
        else:
            # Skip deep blocks — apply cached residual instead
            hidden_states = hidden_states + state.deep_residual_cache

        # Final blocks — always run
        for block in self.transformer_blocks[cache_end:]:
            hidden_states = block(hidden_states, **block_kwargs)

        # ---- postamble (identical to original) ------------------------------
        shift, scale = (
            self.scale_shift_table[None]
            + embedded_timestep[:, None].to(self.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = (
            hidden_states * (1 + scale.to(hidden_states.device))
            + shift.to(hidden_states.device)
        )
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        hidden_states = hidden_states.reshape(
            shape=(-1, height, width,
                   self.config.patch_size, self.config.patch_size,
                   self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.out_channels,
                   height * self.config.patch_size,
                   width  * self.config.patch_size)
        )

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    return cached_forward


def install_deepcache(transformer, cache_start: int, cache_end: int,
                       cache_interval: int, full_steps_set: set) -> DeepCacheState:
    """
    Monkey-patch transformer.forward with caching logic.
    Returns the DeepCacheState object — call state.reset() before each pipe() call.
    """
    state = DeepCacheState()
    fn = _make_cached_forward(cache_start, cache_end, cache_interval,
                               full_steps_set, state)
    transformer.forward = types.MethodType(fn, transformer)
    return state


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def generate_and_evaluate(
    pipe,
    cache_state: DeepCacheState,
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
) -> dict:
    """
    Generate quantized+cached images and compute metrics.
    `ref_dir` should contain ref_0.png .. ref_{s_count-1}.png (FP16 20-step refs).
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

            # Reset cache state before each image generation
            cache_state.reset()

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

    clip_score = None
    if accelerator.is_main_process and clip_model is not None:
        scores = []
        for i in range(s_count):
            q_img = Image.open(os.path.join(save_dir, f"sample_{i}.png")).convert("RGB")
            inputs = clip_processor(
                text=[prompts[i]], images=q_img,
                return_tensors="pt", padding=True,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SWEEP_CONFIGS = [
    # (cache_interval, cache_start, cache_end)
    (1, 4, 24),   # baseline: no caching (interval=1 → every step full)
    (2, 4, 24),   # standard deepcache
    (3, 4, 24),   # aggressive
    (4, 4, 24),
    (2, 2, 26),   # wider block range
    (2, 8, 20),   # narrower block range
    (2, 4, 28),   # skip all but first 4 blocks on cached steps
]


def main():
    parser = argparse.ArgumentParser(description="NVFP4 + DeepCache experiment")
    parser.add_argument("--num_samples",         type=int,  default=20)
    parser.add_argument("--test_run",            action="store_true")
    parser.add_argument("--ref_dir",             type=str,  default="./ref_images")
    parser.add_argument("--save_dir",            type=str,  default="./results")
    parser.add_argument("--model_path",          type=str,
                        default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--dataset_name",        type=str,  default="MJHQ",
                        choices=["MJHQ", "sDCI"])
    parser.add_argument("--lowrank",             type=int,  default=32)
    parser.add_argument("--num_inference_steps", type=int,  default=20)
    parser.add_argument("--cache_interval",      type=int,  default=2,
                        help="Run full model every K steps; others use cache")
    parser.add_argument("--cache_start",         type=int,  default=4,
                        help="First block index in the cached deep segment")
    parser.add_argument("--cache_end",           type=int,  default=24,
                        help="First block index AFTER the cached deep segment")
    parser.add_argument("--full_steps",          type=str,  default="0",
                        help="Comma-separated step indices always run fully")
    parser.add_argument("--sweep",               action="store_true",
                        help="Sweep over predefined cache configs")
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    full_steps_set = {int(x) for x in args.full_steps.split(",") if x.strip()}

    s_target = 2 if args.test_run else args.num_samples
    prompts  = get_prompts(s_target, args)
    s_count  = len(prompts)
    p_count  = 2 if args.test_run else min(64, s_count)

    dataset_ref_dir  = os.path.join(args.ref_dir,  args.dataset_name)
    dataset_save_dir = os.path.join(args.save_dir, args.dataset_name, "deepcache")

    if accelerator.is_main_process:
        os.makedirs(dataset_ref_dir,  exist_ok=True)
        os.makedirs(dataset_save_dir, exist_ok=True)

    t_count = args.num_inference_steps

    # -----------------------------------------------------------------------
    # Phase 1: FP16 reference generation (20-step, main process only)
    # -----------------------------------------------------------------------
    if accelerator.is_main_process:
        accelerator.print(f"Generating FP16 reference images ({t_count} steps)...")
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
    # Phase 2: NVFP4 quantization (once)
    # -----------------------------------------------------------------------
    accelerator.print(f"Quantizing model (NVFP4, lowrank={args.lowrank})...")

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
    # CLIP model (main process only)
    # -----------------------------------------------------------------------
    if accelerator.is_main_process:
        clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    else:
        clip_model = clip_processor = None

    # -----------------------------------------------------------------------
    # Phase 3: Experiment(s)
    # -----------------------------------------------------------------------
    configs_to_run = (
        SWEEP_CONFIGS if args.sweep
        else [(args.cache_interval, args.cache_start, args.cache_end)]
    )

    sweep_results: list[dict] = []

    for interval, c_start, c_end in configs_to_run:
        config_tag = f"interval{interval}_s{c_start}_e{c_end}"
        accelerator.print(f"\n=== Config: {config_tag} | steps={t_count} ===")

        # Estimate theoretical speedup
        n_deep   = c_end - c_start
        n_total  = len(pipe.transformer.transformer_blocks)
        n_always = n_total - n_deep
        # On full steps: n_total blocks run; on cached steps: n_always blocks run
        # fraction of full steps = 1/interval
        avg_blocks = (n_total + n_always * (interval - 1)) / interval
        speedup_est = n_total / avg_blocks
        accelerator.print(
            f"  blocks: shallow={c_start}, deep={n_deep}, final={n_total-c_end} | "
            f"est. speedup ≈ {speedup_est:.2f}x"
        )

        # Install deepcache (replaces transformer.forward each iteration)
        cache_state = install_deepcache(
            pipe.transformer, c_start, c_end, interval, full_steps_set
        )

        run_save_dir = os.path.join(dataset_save_dir, config_tag)
        metrics = generate_and_evaluate(
            pipe, cache_state, t_count, prompts, s_count,
            dataset_ref_dir, run_save_dir,
            device, accelerator,
            clip_model, clip_processor,
            config_tag=config_tag,
        )

        entry = {
            "cache_interval": interval,
            "cache_start":    c_start,
            "cache_end":      c_end,
            "num_steps":      t_count,
            "speedup_est":    round(speedup_est, 3),
            **metrics,
        }
        sweep_results.append(entry)

        # Save per-config metrics
        if accelerator.is_main_process:
            with open(os.path.join(run_save_dir, "metrics.json"), "w") as f:
                json.dump({"config": entry}, f, indent=4)

            accelerator.print(
                f"  FID={metrics['fid']:.2f} | IS={metrics['is']:.3f} | "
                f"PSNR={metrics['psnr']:.2f} | time={metrics['time_per_image_sec']:.2f}s | "
                f"speedup_est={speedup_est:.2f}x"
            )

    # -----------------------------------------------------------------------
    # Save sweep summary
    # -----------------------------------------------------------------------
    if accelerator.is_main_process and args.sweep:
        summary = {
            "method":      "NVFP4_SVDQUANT_DEFAULT_CFG + DeepCache",
            "dataset":     args.dataset_name,
            "num_samples": s_count,
            "lowrank":     args.lowrank,
            "num_steps":   t_count,
            "sweep_results": sweep_results,
        }
        summary_path = os.path.join(dataset_save_dir, "sweep_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"\n✅ Sweep complete. Results: {summary_path}")
        print(
            f"\n{'interval':>9} | {'s':>3} | {'e':>3} | {'FID':>8} | "
            f"{'IS':>7} | {'PSNR':>7} | {'sec/img':>8} | {'speedup':>8}"
        )
        print("-" * 75)
        for r in sweep_results:
            print(
                f"{r['cache_interval']:>9} | {r['cache_start']:>3} | {r['cache_end']:>3} | "
                f"{r['fid']:>8.2f} | {r['is']:>7.3f} | {r['psnr']:>7.2f} | "
                f"{r['time_per_image_sec']:>8.2f} | {r['speedup_est']:>8.3f}x"
            )

    if accelerator.is_main_process and not args.sweep:
        r = sweep_results[0]
        print(
            f"\n✅ Done: interval={r['cache_interval']}, blocks [{r['cache_start']},{r['cache_end']})\n"
            f"   FID={r['fid']:.2f} | IS={r['is']:.3f} | PSNR={r['psnr']:.2f} | "
            f"time={r['time_per_image_sec']:.2f}s | speedup_est={r['speedup_est']:.2f}x"
        )

    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
