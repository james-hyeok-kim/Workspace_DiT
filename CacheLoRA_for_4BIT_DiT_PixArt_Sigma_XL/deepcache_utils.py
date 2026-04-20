"""
deepcache_utils.py

DeepCache block-level activation caching for PixArtTransformer2DModel.

Extracted from pixart_caching/pixart_deepcache_experiment.py.
Used by pixart_nvfp4_cache_compare.py for multi-method comparison.

핵심 아이디어:
  - 28개 transformer block을 3 구역으로 분할:
      shallow  (0 .. cache_start-1)  : 항상 연산
      deep     (cache_start .. cache_end-1) : 캐시 대상
      final    (cache_end .. 27)     : 항상 연산
  - Full step (매 cache_interval step):
      deep block 실행, residual = output - input 저장
  - Cached step:
      hidden_states += cached_residual  (deep block SKIP)
      [Cache-LoRA] cached_residual += corrector_B @ (corrector_A @ dx)
  - 구현: diffusers 소스 수정 없이 transformer.forward monkey-patch

Cache-LoRA Corrector:
  calibrate_cache_lora()로 calibration set에서 학습.
  deep region input delta → residual drift 를 rank-k로 근사.
"""

import os
import csv
import types

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.models.modeling_outputs import Transformer2DModelOutput


# ---------------------------------------------------------------------------
# DeepCache state
# ---------------------------------------------------------------------------

class DeepCacheState:
    """
    Per-image-generation caching state. reset() before each pipe() call.

    Analysis buffers (NOT reset between images — accumulate across run):
      residual_errors  : Exp A — relative error of stale vs fresh residual
      block_drifts     : Exp B — per-block relative output drift

    Q&C approximation:
      use_vc           : Variance Calibration — scale cached residual by std ratio

    Cache-LoRA corrector (set once after calibrate_cache_lora(), NOT reset):
      corrector_A      : [rank, hidden_dim]
      corrector_B      : [hidden_dim, rank]
      h_in_cached      : deep region input at last full step (reset per image)
    """

    def __init__(
        self,
        profile_residual_error: bool = False,
        profile_drift: bool = False,
        use_vc: bool = False,
    ):
        self.step_idx: int = 0
        self.deep_residual_cache: torch.Tensor | None = None

        self.profile_residual_error = profile_residual_error
        self.residual_errors: list[dict] = []

        self.profile_drift = profile_drift
        self.prev_block_outputs: dict[int, torch.Tensor] = {}
        self.block_drifts: dict[int, list[float]] = {}

        self.use_vc = use_vc
        self.cached_deep_std: torch.Tensor | None = None

        # Cache-LoRA corrector (calibrated once; not reset per image)
        self.corrector_A: torch.Tensor | None = None  # [rank, hidden_dim]
        self.corrector_B: torch.Tensor | None = None  # [hidden_dim, rank]
        self.h_in_cached: torch.Tensor | None = None  # deep region input at last full step

        self._image_idx: int = 0

    def reset(self):
        """Reset per-image state. Analysis buffers and correctors preserved."""
        self.step_idx = 0
        self.deep_residual_cache = None
        self.cached_deep_std = None
        self.prev_block_outputs = {}
        self.h_in_cached = None

    def next_image(self):
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
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "block_drift_profile.csv")
    rows = []
    for b_idx in sorted(state.block_drifts.keys()):
        drifts = state.block_drifts[b_idx]
        region = "deep" if cache_start <= b_idx < cache_end else (
            "shallow" if b_idx < cache_start else "final"
        )
        rows.append({
            "block_idx":   b_idx,
            "region":      region,
            "mean_drift":  float(np.mean(drifts)),
            "std_drift":   float(np.std(drifts)),
            "max_drift":   float(np.max(drifts)),
            "num_samples": len(drifts),
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["block_idx", "region", "mean_drift",
                           "std_drift", "max_drift", "num_samples"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Exp B] Block drift CSV: {path} ({len(rows)} blocks)")


# ---------------------------------------------------------------------------
# DeepCache forward (monkey-patch)
# ---------------------------------------------------------------------------

def _make_cached_forward(cache_start: int, cache_end: int,
                          cache_interval: int, full_steps_set: set,
                          state: DeepCacheState):
    """
    Return a new forward function for PixArtTransformer2DModel with
    block-level caching. Preamble/postamble copied from diffusers
    pixart_transformer_2d.py.
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
        # ---- preamble --------------------------------------------------------
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

        # ---- block loop with caching -----------------------------------------
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

        # Shallow blocks: always run
        for b_idx, block in enumerate(self.transformer_blocks[:cache_start]):
            hidden_states = block(hidden_states, **block_kwargs)
            if state.profile_drift and is_full:
                _update_drift(state, b_idx, hidden_states)

        # Deep blocks: full or cached
        if is_full:
            h_before_deep = hidden_states.clone()
            if state.use_vc:
                state.cached_deep_std = hidden_states.std(dim=-1, keepdim=True).detach()
            # Cache-LoRA: save deep region input for later dx computation
            if state.corrector_A is not None:
                state.h_in_cached = hidden_states.detach().clone()
            for i, block in enumerate(self.transformer_blocks[cache_start:cache_end]):
                b_idx = cache_start + i
                hidden_states = block(hidden_states, **block_kwargs)
                if state.profile_drift:
                    _update_drift(state, b_idx, hidden_states)
            state.deep_residual_cache = hidden_states - h_before_deep
        else:
            # Exp A: measure stale residual error
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

            # Cache-LoRA: compute correction from input delta
            lora_correction = None
            if (state.corrector_A is not None
                    and state.corrector_B is not None
                    and state.h_in_cached is not None):
                dx = hidden_states - state.h_in_cached          # [B, T, H]
                dtype = hidden_states.dtype
                lora_correction = F.linear(
                    F.linear(dx, state.corrector_A.to(dtype)),  # [B, T, rank]
                    state.corrector_B.to(dtype),                # [B, T, H]
                )

            # Q&C VC: scale cached residual by std ratio
            if state.use_vc and state.cached_deep_std is not None:
                current_std = hidden_states.std(dim=-1, keepdim=True)
                vc_scale = current_std / (state.cached_deep_std + 1e-8)
                residual = state.deep_residual_cache * vc_scale
            else:
                residual = state.deep_residual_cache

            if lora_correction is not None:
                hidden_states = hidden_states + residual + lora_correction
            else:
                hidden_states = hidden_states + residual

        # Final blocks: always run
        for i, block in enumerate(self.transformer_blocks[cache_end:]):
            b_idx = cache_end + i
            hidden_states = block(hidden_states, **block_kwargs)
            if state.profile_drift and is_full:
                _update_drift(state, b_idx, hidden_states)

        # ---- postamble -------------------------------------------------------
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


def calibrate_cache_lora(
    pipe,
    transformer,
    cache_start: int,
    cache_end: int,
    cache_interval: int,
    prompts: list,
    num_calib: int,
    t_count: int,
    guidance_scale: float,
    device,
    rank: int = 8,
    calib_seed_offset: int = 1000,
) -> tuple:
    """
    Learn a rank-k corrector for cached deep-block residuals.

    Runs `num_calib` prompts through the unmodified pipeline (no DeepCache),
    collecting deep-region inputs and residuals at each denoising step.
    Fits a rank-k linear mapping:
        drift[t] ≈ corrector_B @ (corrector_A @ dx[t])
    where:
        drift[t] = residual[t] - residual[t-interval]
        dx[t]    = h_in[t]    - h_in[t-interval]

    During inference, each cached step computes:
        correction = corrector_B @ (corrector_A @ (h_in_current - h_in_cached))
        hidden_states += deep_residual_cache + correction

    Args:
        calib_seed_offset: Seed base for calibration images (default 1000).
            Should not overlap with eval seeds (default 42+i) to avoid
            data contamination.

    Returns:
        (corrector_A, corrector_B, calib_time_sec) tensors on `device`
        corrector_A: [rank, hidden_dim]
        corrector_B: [hidden_dim, rank]
        calib_time_sec: float — wall-clock seconds for entire calibration
    """
    import time as _time
    _calib_t0 = _time.perf_counter()

    num_calib = min(num_calib, len(prompts))
    print(f"  [Cache-LoRA Calib] {num_calib} prompts × {t_count} steps, "
          f"interval={cache_interval}, rank={rank}, seed_offset={calib_seed_offset}")

    # Determine hidden_dim by inspecting any linear layer in a deep block
    hidden_dim = None
    for name, param in transformer.transformer_blocks[cache_start].named_parameters():
        if param.ndim >= 2:
            # Bias or weight; weight shape is [out, in] → use last dim
            hidden_dim = param.shape[-1]
            break
    if hidden_dim is None:
        raise RuntimeError("calibrate_cache_lora: cannot determine hidden_dim")

    # ---- Hook-based collection ------------------------------------------------
    # pre-hook on block[cache_start] → captures deep-region input
    # post-hook on block[cache_end-1] → captures deep-region output
    step_counter = [0]
    h_in_buf: dict[int, torch.Tensor] = {}
    h_out_buf: dict[int, torch.Tensor] = {}
    _tmp_in: list = [None]

    def _pre_hook(module, args):
        _tmp_in[0] = args[0].detach().cpu()

    def _post_hook(module, args, output):
        if _tmp_in[0] is not None:
            s = step_counter[0]
            h_in_buf[s]  = _tmp_in[0]
            h_out_buf[s] = output.detach().cpu()
            _tmp_in[0]   = None
            step_counter[0] += 1

    h_pre  = transformer.transformer_blocks[cache_start    ].register_forward_pre_hook(_pre_hook)
    h_post = transformer.transformer_blocks[cache_end  - 1 ].register_forward_hook(_post_hook)

    # Accumulate cross-covariance C = Σ drift_flat.T @ dx_flat  (shape [H, H])
    C = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float64)
    total_samples = 0

    try:
        for i in range(num_calib):
            step_counter[0] = 0
            h_in_buf.clear()
            h_out_buf.clear()
            _tmp_in[0] = None

            gen = torch.Generator(device=device).manual_seed(calib_seed_offset + i)
            with torch.no_grad():
                pipe(
                    prompts[i % len(prompts)],
                    num_inference_steps=t_count,
                    guidance_scale=guidance_scale,
                    generator=gen,
                )

            # Update cross-covariance using collected steps
            n_collected = step_counter[0]
            for s in range(cache_interval, n_collected):
                s_prev = s - cache_interval
                if s_prev not in h_in_buf:
                    continue
                h_in_c  = h_in_buf[s].float()       # [B, T, H]
                h_in_p  = h_in_buf[s_prev].float()
                h_out_c = h_out_buf[s].float()
                h_out_p = h_out_buf[s_prev].float()

                dx    = (h_in_c  - h_in_p ).reshape(-1, hidden_dim)   # [N, H]
                drift = ((h_out_c - h_in_c) - (h_out_p - h_in_p)).reshape(-1, hidden_dim)

                C += drift.double().T @ dx.double()
                total_samples += dx.shape[0]

            h_in_buf.clear()
            h_out_buf.clear()
            print(f"    calib {i+1}/{num_calib} done  (samples: {total_samples:,})", flush=True)

    finally:
        h_pre.remove()
        h_post.remove()

    if total_samples == 0:
        raise RuntimeError("calibrate_cache_lora: no calibration samples collected")

    # ---- SVD of cross-covariance ---------------------------------------------
    C_norm = (C / total_samples).float()
    print(f"  [Cache-LoRA Calib] SVD of [{hidden_dim}×{hidden_dim}] matrix...")
    U, S, Vt = torch.linalg.svd(C_norm, full_matrices=False)
    # U: [H, H],  S: [H],  Vt: [H, H]

    top_sv = S[:min(rank + 4, len(S))].tolist()
    print(f"  [Cache-LoRA Calib] Top singular values: {[f'{v:.4f}' for v in top_sv]}")

    sq = S[:rank].clamp(min=0.0).sqrt()
    corrector_A = (sq.unsqueeze(1) * Vt[:rank, :]).to(device)   # [rank, H]
    corrector_B = (U[:, :rank] * sq.unsqueeze(0)).to(device)    # [H, rank]

    calib_time_sec = _time.perf_counter() - _calib_t0
    print(f"  [Cache-LoRA Calib] Done: A{list(corrector_A.shape)}, B{list(corrector_B.shape)} "
          f"in {calib_time_sec:.1f}s")
    return corrector_A, corrector_B, calib_time_sec


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
    Monkey-patch transformer.forward with DeepCache.
    Returns DeepCacheState — call state.reset() before each pipe() call.
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
