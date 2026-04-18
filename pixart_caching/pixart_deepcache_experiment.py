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

Analysis 모드 (3가지 연구 실험):
  Exp A (--profile_residual_error):
      캐시된 step에서 stale residual vs fresh residual 오차 측정
      → SVD low-rank branch가 cache error를 보정하는지 분석
      → --lowrank 0 / 32 두 번 실행해서 비교

  Exp B (--profile_blocks):
      각 block의 timestep 간 output drift 측정
      → cache 적합 block 범위 선택 근거 정량화
      → block별 relative drift → block_drift_profile.csv

  Exp C (--cache_aware_calib):
      NVFP4 calibration 시 DeepCache 활성화
      → cached step activation 분포를 calibration에 반영
      → full-step only calibration 대비 FID/IS 비교

CLI:
  --num_inference_steps  20
  --cache_interval       2
  --cache_start          4
  --cache_end            24
  --full_steps           "0"
  --sweep
  --num_samples          20
  --test_run
  --profile_residual_error   # Exp A
  --profile_blocks            # Exp B
  --cache_aware_calib         # Exp C

결과: results/{dataset}/deepcache/interval{K}_s{start}_e{end}_gs{G}/
      results/{dataset}/deepcache/residual_errors_rank{R}.csv   (Exp A)
      results/{dataset}/deepcache/block_drift_profile.csv        (Exp B)
"""

import os
import time
import gc
import copy
import csv
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
    """
    Per-image-generation caching state. reset() before each pipe() call.

    Analysis buffers (NOT reset between images — accumulate across run):
      residual_errors  : Exp A — relative error of stale vs fresh residual per cached step
      block_drifts     : Exp B — per-block relative output drift across consecutive full steps
      prev_block_outputs: Exp B — saved outputs from last full step

    Q&C approximation fields:
      use_vc           : Variance Calibration — scale cached residual by std ratio
      cached_deep_std  : std of deep block input at full step (for VC scaling)
    """

    def __init__(
        self,
        profile_residual_error: bool = False,
        profile_drift: bool = False,
        use_vc: bool = False,
    ):
        # Per-image state
        self.step_idx: int = 0
        self.deep_residual_cache: torch.Tensor | None = None

        # Exp A: SVD × cache residual error
        self.profile_residual_error = profile_residual_error
        self.residual_errors: list[dict] = []  # {image_idx, step_idx, abs_err, rel_err}

        # Exp B: block drift profiling
        self.profile_drift = profile_drift
        self.prev_block_outputs: dict[int, torch.Tensor] = {}
        self.block_drifts: dict[int, list[float]] = {}  # b_idx → [rel_drift, ...]

        # Q&C VC: Variance Calibration
        self.use_vc = use_vc
        self.cached_deep_std: torch.Tensor | None = None  # std at cache time

        # Internal: current image index for logging
        self._image_idx: int = 0

    def reset(self):
        """Reset per-image state. Analysis buffers are preserved."""
        self.step_idx = 0
        self.deep_residual_cache = None
        self.cached_deep_std = None
        # Reset Exp B per-step prev outputs (keep drifts)
        self.prev_block_outputs = {}

    def next_image(self):
        """Call after each image to advance image counter."""
        self._image_idx += 1


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _update_drift(state: DeepCacheState, b_idx: int, hidden_states: torch.Tensor):
    """Exp B: update per-block drift with current hidden_states (detached)."""
    curr = hidden_states.detach()
    if b_idx in state.prev_block_outputs:
        norm = curr.norm().item() + 1e-8
        drift = (curr - state.prev_block_outputs[b_idx]).norm().item() / norm
        state.block_drifts.setdefault(b_idx, []).append(drift)
    state.prev_block_outputs[b_idx] = curr


def save_residual_error_csv(state: DeepCacheState, save_dir: str, lowrank: int):
    """Save Exp A results: stale vs fresh residual error per cached step."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"residual_errors_rank{lowrank}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_idx", "step_idx", "abs_err", "rel_err"]
        )
        writer.writeheader()
        writer.writerows(state.residual_errors)
    print(f"[Exp A] Residual error CSV: {path} ({len(state.residual_errors)} rows)")
    if state.residual_errors:
        rel_errs = [e["rel_err"] for e in state.residual_errors]
        print(
            f"[Exp A] rel_err — mean={np.mean(rel_errs):.4f} "
            f"std={np.std(rel_errs):.4f} max={np.max(rel_errs):.4f}"
        )


def save_block_drift_csv(state: DeepCacheState, save_dir: str,
                          cache_start: int, cache_end: int):
    """Save Exp B results: per-block mean/std drift across full steps."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "block_drift_profile.csv")
    rows = []
    for b_idx in sorted(state.block_drifts.keys()):
        drifts = state.block_drifts[b_idx]
        in_cache = "deep" if cache_start <= b_idx < cache_end else (
            "shallow" if b_idx < cache_start else "final"
        )
        rows.append({
            "block_idx":    b_idx,
            "region":       in_cache,
            "mean_drift":   float(np.mean(drifts)),
            "std_drift":    float(np.std(drifts)),
            "max_drift":    float(np.max(drifts)),
            "num_samples":  len(drifts),
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["block_idx", "region", "mean_drift",
                           "std_drift", "max_drift", "num_samples"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Exp B] Block drift CSV: {path} ({len(rows)} blocks)")
    if rows:
        # Print top-5 most stable and most drifting blocks
        rows_sorted = sorted(rows, key=lambda x: x["mean_drift"])
        print("[Exp B] Most stable blocks (low drift):")
        for r in rows_sorted[:5]:
            print(f"  block {r['block_idx']:>2} ({r['region']:>7}): "
                  f"mean={r['mean_drift']:.4f} std={r['std_drift']:.4f}")
        print("[Exp B] Most drifting blocks:")
        for r in rows_sorted[-5:]:
            print(f"  block {r['block_idx']:>2} ({r['region']:>7}): "
                  f"mean={r['mean_drift']:.4f} std={r['std_drift']:.4f}")


# ---------------------------------------------------------------------------
# DeepCache forward (monkey-patch)
# ---------------------------------------------------------------------------

def _make_cached_forward(cache_start: int, cache_end: int,
                          cache_interval: int, full_steps_set: set,
                          state: DeepCacheState):
    """
    Return a new forward function for PixArtTransformer2DModel with:
      - block-level caching (core DeepCache logic)
      - Exp A: residual error measurement on cached steps
      - Exp B: per-block drift tracking on full steps

    Preamble/postamble copied from
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
            or state.deep_residual_cache is None
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

        # ── Shallow blocks: always run ────────────────────────────────────────
        for b_idx, block in enumerate(self.transformer_blocks[:cache_start]):
            hidden_states = block(hidden_states, **block_kwargs)
            if state.profile_drift and is_full:
                _update_drift(state, b_idx, hidden_states)

        # ── Deep blocks: full or cached ───────────────────────────────────────
        if is_full:
            h_before_deep = hidden_states.clone()

            # Q&C VC: save std of deep-block INPUT at full step
            if state.use_vc:
                state.cached_deep_std = hidden_states.std(dim=-1, keepdim=True).detach()

            for i, block in enumerate(self.transformer_blocks[cache_start:cache_end]):
                b_idx = cache_start + i
                hidden_states = block(hidden_states, **block_kwargs)
                if state.profile_drift:
                    _update_drift(state, b_idx, hidden_states)
            state.deep_residual_cache = hidden_states - h_before_deep
        else:
            # ── Exp A: measure stale residual error before applying cache ────
            if state.profile_residual_error:
                with torch.no_grad():
                    h_fresh = hidden_states.clone()
                    for block in self.transformer_blocks[cache_start:cache_end]:
                        h_fresh = block(h_fresh, **block_kwargs)
                    fresh_residual = h_fresh - hidden_states
                    stale_residual = state.deep_residual_cache
                    abs_err = (stale_residual - fresh_residual).norm().item()
                    rel_err = abs_err / (fresh_residual.norm().item() + 1e-8)
                    state.residual_errors.append({
                        "image_idx": state._image_idx,
                        "step_idx":  step_idx,
                        "abs_err":   abs_err,
                        "rel_err":   rel_err,
                    })

            # ── Q&C VC: scale cached residual by current/cached std ratio ────
            if state.use_vc and state.cached_deep_std is not None:
                current_std = hidden_states.std(dim=-1, keepdim=True)
                vc_scale = current_std / (state.cached_deep_std + 1e-8)
                hidden_states = hidden_states + state.deep_residual_cache * vc_scale
            else:
                hidden_states = hidden_states + state.deep_residual_cache

        # ── Final blocks: always run ──────────────────────────────────────────
        for i, block in enumerate(self.transformer_blocks[cache_end:]):
            b_idx = cache_end + i
            hidden_states = block(hidden_states, **block_kwargs)
            if state.profile_drift and is_full:
                _update_drift(state, b_idx, hidden_states)

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


def install_deepcache(
    transformer,
    cache_start: int,
    cache_end: int,
    cache_interval: int,
    full_steps_set: set,
    profile_residual_error: bool = False,
    profile_drift: bool = False,
    use_vc: bool = False,
) -> DeepCacheState:
    """
    Monkey-patch transformer.forward with caching + optional analysis/Q&C hooks.
    Returns DeepCacheState — call state.reset() before each pipe() call.

    use_vc: Q&C Variance Calibration — scale cached residual by current/cached std ratio
    """
    state = DeepCacheState(
        profile_residual_error=profile_residual_error,
        profile_drift=profile_drift,
        use_vc=use_vc,
    )
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
    guidance_scale: float = 4.5,
    lowrank: int = 32,
    skip_existing: bool = False,
) -> dict:
    """
    Generate quantized+cached images and compute metrics.
    Saves Exp A / Exp B analysis CSVs if profiling is enabled.
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
            save_path = os.path.join(save_dir, f"sample_{i}.png")

            if skip_existing and os.path.exists(save_path):
                q_img = Image.open(save_path).convert("RGB")
            else:
                cache_state.reset()

                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                q_img = pipe(
                    prompts[i], num_inference_steps=t_count,
                    guidance_scale=guidance_scale, generator=gen
                ).images[0]
                torch.cuda.synchronize(device)
                local_times.append(time.perf_counter() - t0)

                cache_state.next_image()
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

            t_str = f"{local_times[-1]:.2f}s" if local_times else "skipped"
            print(f"[GPU {accelerator.process_index}] {config_tag} "
                  f"sample_{i} ({t_str})", flush=True)

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
                truncation=True, max_length=77,
            ).to(device)
            with torch.no_grad():
                scores.append(float(clip_model(**inputs).logits_per_image.item()))
        clip_score = float(np.mean(scores))

    # ── Save analysis results (main process, Exp A & B) ─────────────────────
    if accelerator.is_main_process:
        if cache_state.profile_residual_error and cache_state.residual_errors:
            save_residual_error_csv(cache_state, save_dir, lowrank)
        if cache_state.profile_drift and cache_state.block_drifts:
            # Need cache_start/end from the state — retrieve from save_dir tag or pass explicitly
            # We save with defaults; caller may override if needed
            save_block_drift_csv(cache_state, save_dir, 0, 28)

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
# Main sweep configs
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

TEST_SWEEP_CONFIGS = [
    (4.5, False, False, "0"),
    (3.0, False, False, "0"),
    (3.5, False, False, "0"),
    (5.5, False, False, "0"),
    (6.0, False, False, "0"),
    (4.5, True,  False, "0"),
    (4.5, False, True,  "0"),
    (4.5, True,  False, "0,1"),
    (4.5, False, False, "0,1"),
    (4.5, False, False, "0,1,2"),
]


# ---------------------------------------------------------------------------
# test_sweep mode
# ---------------------------------------------------------------------------

def _run_test_sweep(
    pipe,
    base_scheduler_config,
    t_count: int,
    prompts: list[str],
    s_count: int,
    ref_dir: str,
    save_base_dir: str,
    device,
    accelerator,
    clip_model,
    clip_processor,
    args,
):
    sweep_results: list[dict] = []
    test_sweep_dir = os.path.join(save_base_dir, "test_sweep")
    if accelerator.is_main_process:
        os.makedirs(test_sweep_dir, exist_ok=True)

    for gs, karras, lu, fs_str in TEST_SWEEP_CONFIGS:
        fs_set = {int(x) for x in fs_str.split(",") if x.strip()}
        tag = f"gs{gs}_k{int(karras)}_lu{int(lu)}_fs{fs_str.replace(',','')}"
        accelerator.print(f"\n=== test_sweep: {tag} | steps={t_count} ===")

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            base_scheduler_config,
            use_karras_sigmas=karras,
            use_lu_lambdas=lu,
        )

        cache_state = install_deepcache(
            pipe.transformer, 8, 20, 2, fs_set,
            profile_residual_error=args.profile_residual_error,
            profile_drift=args.profile_blocks,
        )

        run_save_dir = os.path.join(test_sweep_dir, tag)
        metrics = generate_and_evaluate(
            pipe, cache_state, t_count, prompts, s_count,
            ref_dir, run_save_dir,
            device, accelerator,
            clip_model, clip_processor,
            config_tag=tag,
            guidance_scale=gs,
            lowrank=args.lowrank,
        )

        # Save block drift with correct cache range
        if accelerator.is_main_process and args.profile_blocks and cache_state.block_drifts:
            save_block_drift_csv(cache_state, run_save_dir, 8, 20)

        entry = {
            "guidance_scale":  gs,
            "karras":          karras,
            "lu":              lu,
            "full_steps":      fs_str,
            "cache_interval":  2,
            "cache_start":     8,
            "cache_end":       20,
            "num_steps":       t_count,
            **metrics,
        }
        sweep_results.append(entry)

        if accelerator.is_main_process:
            accelerator.print(
                f"  FID={metrics['fid']:.2f} | IS={metrics['is']:.3f} | "
                f"PSNR={metrics['psnr']:.2f} | time={metrics['time_per_image_sec']:.2f}s"
            )

    if accelerator.is_main_process:
        summary_path = os.path.join(test_sweep_dir, "test_sweep_summary.json")
        with open(summary_path, "w") as f:
            json.dump({"sweep_results": sweep_results}, f, indent=4)

        csv_path = os.path.join(test_sweep_dir, "test_sweep_summary.csv")
        csv_fields = ["guidance_scale", "karras", "lu", "full_steps",
                      "cache_interval", "cache_start", "cache_end", "num_steps",
                      "fid", "is", "psnr", "ssim", "lpips", "clip",
                      "time_per_image_sec"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sweep_results)

        print(f"\n✅ test_sweep complete.")
        print(f"   CSV: {csv_path}")
        print(f"\n{'gs':>5} | {'karras':>6} | {'lu':>5} | {'fs':>5} | "
              f"{'FID':>8} | {'IS':>7} | {'PSNR':>7} | {'time':>7}")
        print("-" * 70)
        for r in sweep_results:
            print(
                f"{r['guidance_scale']:>5.1f} | {str(r['karras']):>6} | "
                f"{str(r['lu']):>5} | {r['full_steps']:>5} | "
                f"{r['fid']:>8.2f} | {r['is']:>7.3f} | "
                f"{r['psnr']:>7.2f} | {r['time_per_image_sec']:>7.2f}s"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument("--cache_interval",      type=int,  default=2)
    parser.add_argument("--cache_start",         type=int,  default=4)
    parser.add_argument("--cache_end",           type=int,  default=24)
    parser.add_argument("--full_steps",          type=str,  default="0")
    parser.add_argument("--guidance_scale",      type=float, default=4.5)
    parser.add_argument("--use_karras_sigmas",   action="store_true")
    parser.add_argument("--use_lu_lambdas",      action="store_true")
    parser.add_argument("--sweep",               action="store_true")
    parser.add_argument("--test_sweep",          action="store_true")

    # ── Analysis modes ────────────────────────────────────────────────────────
    parser.add_argument(
        "--profile_residual_error", action="store_true",
        help="[Exp A] Log stale vs fresh residual error at each cached step. "
             "Run twice (--lowrank 0 and --lowrank 32) and compare CSVs."
    )
    parser.add_argument(
        "--profile_blocks", action="store_true",
        help="[Exp B] Log per-block output drift across consecutive full steps. "
             "Outputs block_drift_profile.csv to quantify which blocks are cache-stable."
    )
    parser.add_argument(
        "--cache_aware_calib", action="store_true",
        help="[Exp C] Enable DeepCache during NVFP4 calibration so cached-step "
             "activation distributions are included in scaling factor computation."
    )

    # ── Q&C approximation modes ───────────────────────────────────────────────
    parser.add_argument(
        "--use_tap", action="store_true",
        help="[Q&C TAP approx] Timestep-Aware Perturbation: run calibration at full "
             "num_inference_steps (not 5) to cover all timestep activation distributions."
    )
    parser.add_argument(
        "--use_vc", action="store_true",
        help="[Q&C VC approx] Variance Calibration: scale cached residual by "
             "current/cached std ratio to compensate distribution shift."
    )
    parser.add_argument(
        "--qandc", action="store_true",
        help="Shorthand for --use_tap --use_vc (Q&C full approximation)."
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip image generation if sample_{i}.png already exists in save_dir. "
             "Loads saved image instead — enables incremental sample count expansion."
    )

    args = parser.parse_args()

    # --qandc shorthand
    if args.qandc:
        args.use_tap = True
        args.use_vc  = True

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

    if args.test_sweep:
        s_target = 2
        prompts  = get_prompts(s_target, args)
        s_count  = len(prompts)
        p_count  = 2

    # -----------------------------------------------------------------------
    # Phase 1: FP16 reference generation (fixed: gs=4.5, default scheduler)
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
                    prompts[i], num_inference_steps=t_count,
                    guidance_scale=4.5, generator=gen
                ).images[0]
                img.save(ref_path)

        del pipe_ref
        torch.cuda.empty_cache()
        gc.collect()
        accelerator.print(f"  {s_count} refs ready in {dataset_ref_dir}")

    accelerator.wait_for_everyone()

    # -----------------------------------------------------------------------
    # Phase 2: NVFP4 quantization
    # -----------------------------------------------------------------------
    accelerator.print(f"Quantizing model (NVFP4, lowrank={args.lowrank})...")

    pipe = PixArtAlphaPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=args.use_karras_sigmas,
        use_lu_lambdas=args.use_lu_lambdas,
    )
    _base_scheduler_config = pipe.scheduler.config

    quant_config = copy.deepcopy(mtq.NVFP4_SVDQUANT_DEFAULT_CFG)
    if "algorithm" in quant_config:
        quant_config["algorithm"]["lowrank"] = args.lowrank

    # ── Q&C TAP: Timestep-Aware Perturbation calibration ─────────────────────
    # TAP 핵심: calibration을 짧은 5-step이 아닌 full num_inference_steps로 실행
    # → 모든 timestep 구간(고노이즈~저노이즈)의 activation 분포를 calibration에 포함
    # → NVFP4 micro-block scaling이 timestep별 분포 변화를 커버
    if args.use_tap:
        accelerator.print(
            f"[Q&C TAP] Timestep-Aware calibration: running full {t_count}-step "
            f"calibration (vs default 5-step) to cover all timestep distributions."
        )
        tap_calib_steps = t_count  # full steps
    else:
        tap_calib_steps = 5  # default

    # ── Exp C: cache-aware calibration ────────────────────────────────────────
    if args.cache_aware_calib:
        accelerator.print(
            "[Exp C] Cache-aware calibration: installing DeepCache before mtq.quantize. "
            f"Calibration forward passes will include cached-step activations "
            f"(interval={args.cache_interval}, blocks [{args.cache_start},{args.cache_end}))."
        )
        calib_cache_state = install_deepcache(
            pipe.transformer,
            args.cache_start,
            args.cache_end,
            args.cache_interval,
            full_steps_set,
        )

        def forward_loop(model):
            for prompt in prompts[:p_count]:
                calib_cache_state.reset()
                pipe(
                    prompt,
                    num_inference_steps=tap_calib_steps,
                    generator=torch.Generator(device=device).manual_seed(42),
                )
    else:
        def forward_loop(model):
            for prompt in prompts[:p_count]:
                pipe(
                    prompt,
                    num_inference_steps=tap_calib_steps,
                    generator=torch.Generator(device=device).manual_seed(42),
                )

    with torch.no_grad():
        pipe.transformer = mtq.quantize(
            pipe.transformer, quant_config, forward_loop=forward_loop
        )

    accelerator.wait_for_everyone()
    accelerator.print(
        "Quantization done."
        + (" [cache-aware calib]" if args.cache_aware_calib else "")
    )

    # ── Log Exp C calibration mode to JSON ────────────────────────────────────
    if accelerator.is_main_process and args.cache_aware_calib:
        calib_meta = {
            "cache_aware_calib": True,
            "cache_interval":    args.cache_interval,
            "cache_start":       args.cache_start,
            "cache_end":         args.cache_end,
            "lowrank":           args.lowrank,
            "num_calib_steps":   5,
        }
        meta_path = os.path.join(dataset_save_dir, "calib_meta.json")
        with open(meta_path, "w") as f:
            json.dump(calib_meta, f, indent=4)
        accelerator.print(f"[Exp C] Calibration meta: {meta_path}")

    # -----------------------------------------------------------------------
    # CLIP model
    # -----------------------------------------------------------------------
    if accelerator.is_main_process:
        clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    else:
        clip_model = clip_processor = None

    # -----------------------------------------------------------------------
    # Phase 3: Experiment(s)
    # -----------------------------------------------------------------------

    if args.test_sweep:
        _run_test_sweep(
            pipe, _base_scheduler_config, t_count, prompts, s_count,
            dataset_ref_dir, dataset_save_dir, device, accelerator,
            clip_model, clip_processor, args,
        )
        accelerator.wait_for_everyone()
        gc.collect()
        torch.cuda.empty_cache()
        return

    configs_to_run = (
        SWEEP_CONFIGS if args.sweep
        else [(args.cache_interval, args.cache_start, args.cache_end)]
    )

    sweep_results: list[dict] = []

    for interval, c_start, c_end in configs_to_run:
        config_tag = f"interval{interval}_s{c_start}_e{c_end}_gs{args.guidance_scale}_steps{t_count}"
        if args.cache_aware_calib:
            config_tag += "_calib_cache"
        if args.use_tap:
            config_tag += "_tap"
        if args.use_vc:
            config_tag += "_vc"
        accelerator.print(f"\n=== Config: {config_tag} | steps={t_count} ===")

        n_deep   = c_end - c_start
        n_total  = len(pipe.transformer.transformer_blocks)
        n_always = n_total - n_deep
        avg_blocks = (n_total + n_always * (interval - 1)) / interval
        speedup_est = n_total / avg_blocks
        accelerator.print(
            f"  blocks: shallow={c_start}, deep={n_deep}, final={n_total-c_end} | "
            f"est. speedup ≈ {speedup_est:.2f}x"
        )
        if args.profile_residual_error:
            accelerator.print(
                f"  [Exp A] Residual error profiling ON (rank={args.lowrank})"
            )
        if args.profile_blocks:
            accelerator.print(
                f"  [Exp B] Block drift profiling ON"
            )
        if args.cache_aware_calib:
            accelerator.print(
                f"  [Exp C] Cache-aware calibration was applied"
            )

        cache_state = install_deepcache(
            pipe.transformer, c_start, c_end, interval, full_steps_set,
            profile_residual_error=args.profile_residual_error,
            profile_drift=args.profile_blocks,
            use_vc=args.use_vc,
        )

        run_save_dir = os.path.join(dataset_save_dir, config_tag)
        metrics = generate_and_evaluate(
            pipe, cache_state, t_count, prompts, s_count,
            dataset_ref_dir, run_save_dir,
            device, accelerator,
            clip_model, clip_processor,
            config_tag=config_tag,
            guidance_scale=args.guidance_scale,
            lowrank=args.lowrank,
            skip_existing=args.skip_existing,
        )

        # Exp B: re-save with correct cache range
        if accelerator.is_main_process and args.profile_blocks and cache_state.block_drifts:
            save_block_drift_csv(cache_state, run_save_dir, c_start, c_end)

        entry = {
            "cache_interval":    interval,
            "cache_start":       c_start,
            "cache_end":         c_end,
            "num_steps":         t_count,
            "guidance_scale":    args.guidance_scale,
            "lowrank":           args.lowrank,
            "cache_aware_calib": args.cache_aware_calib,
            "karras":            args.use_karras_sigmas,
            "lu":                args.use_lu_lambdas,
            "full_steps":        args.full_steps,
            "speedup_est":       round(speedup_est, 3),
            **metrics,
        }
        sweep_results.append(entry)

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
            "method":            "NVFP4_SVDQUANT_DEFAULT_CFG + DeepCache",
            "dataset":           args.dataset_name,
            "num_samples":       s_count,
            "lowrank":           args.lowrank,
            "num_steps":         t_count,
            "cache_aware_calib": args.cache_aware_calib,
            "sweep_results":     sweep_results,
        }
        summary_path = os.path.join(dataset_save_dir, "sweep_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        csv_path = os.path.join(dataset_save_dir, "sweep_summary.csv")
        csv_fields = ["cache_interval", "cache_start", "cache_end", "num_steps",
                      "speedup_est", "fid", "is", "psnr", "ssim", "lpips", "clip",
                      "time_per_image_sec"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sweep_results)

        print(f"\n✅ Sweep complete. JSON: {summary_path} | CSV: {csv_path}")
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
        config_tag   = f"interval{r['cache_interval']}_s{r['cache_start']}_e{r['cache_end']}_gs{r.get('guidance_scale', 4.5)}"
        if args.cache_aware_calib:
            config_tag += "_calib_cache"
        run_save_dir = os.path.join(dataset_save_dir, config_tag)
        csv_path = os.path.join(run_save_dir, "metrics.csv")
        csv_fields = ["cache_interval", "cache_start", "cache_end", "num_steps",
                      "speedup_est", "fid", "is", "psnr", "ssim", "lpips", "clip",
                      "time_per_image_sec"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(r)
        print(
            f"\n✅ Done: interval={r['cache_interval']}, blocks [{r['cache_start']},{r['cache_end']})\n"
            f"   FID={r['fid']:.2f} | IS={r['is']:.3f} | PSNR={r['psnr']:.2f} | "
            f"time={r['time_per_image_sec']:.2f}s | speedup_est={r['speedup_est']:.2f}x\n"
            f"   CSV: {csv_path}"
        )

    accelerator.wait_for_everyone()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
