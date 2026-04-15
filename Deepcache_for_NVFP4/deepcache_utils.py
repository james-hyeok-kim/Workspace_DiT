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
  - 구현: diffusers 소스 수정 없이 transformer.forward monkey-patch
"""

import os
import csv
import types

import numpy as np
import torch

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

        self._image_idx: int = 0

    def reset(self):
        """Reset per-image state. Analysis buffers preserved."""
        self.step_idx = 0
        self.deep_residual_cache = None
        self.cached_deep_std = None
        self.prev_block_outputs = {}

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

            # Q&C VC: scale cached residual by std ratio
            if state.use_vc and state.cached_deep_std is not None:
                current_std = hidden_states.std(dim=-1, keepdim=True)
                vc_scale = current_std / (state.cached_deep_std + 1e-8)
                hidden_states = hidden_states + state.deep_residual_cache * vc_scale
            else:
                hidden_states = hidden_states + state.deep_residual_cache

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
