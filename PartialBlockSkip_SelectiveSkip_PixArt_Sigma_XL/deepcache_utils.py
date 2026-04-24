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

        # Direction 2A: Phase-binned corrector (cache_lora_phase)
        self.phase_correctors = None   # list[(A:[rank,H], B:[H,rank])] per phase
        self.phase_t_count: int | None = None  # total step count for phase binning

        # Direction 2B: Timestep-conditional scale (cache_lora_ts)
        # Reuses corrector_A / corrector_B; adds per-step scalar
        self.step_scales: torch.Tensor | None = None  # [t_count] float tensor

        # Direction 3/5: Block-specific corrector (cache_lora_block)
        self.block_correctors = None       # list[(A_i:[rank,H], B_i:[H,rank])] per deep block
        self.residual_per_block = None     # list[Tensor[B,T,H]] — reset per image

        # Direction 4: SVD-Aware corrector (cache_lora_svd)
        self.svd_corrector_A: torch.Tensor | None = None  # [rank, 2H]
        self.svd_corrector_B: torch.Tensor | None = None  # [H, rank]
        self.lr_probe_cached: torch.Tensor | None = None  # [B,T,H] — reset per image

        # Nonlinear corrector (nn.Module, set after calibration)
        self.nl_corrector = None       # nn.Module with forward(dx, t_norm=None)
        self.nl_needs_t: bool = False  # True for FiLM (option 4)
        self.nl_t_count: int | None = None  # total steps for t_norm computation
        # fd mode: corrector predicts fresh_res directly; stale cache not added at inference
        self.nl_fd_mode: bool = False
        # train mode: skip no_grad wrapper so gradient flows through corrector (trajectory distill)
        self.nl_train_mode: bool = False

        self._image_idx: int = 0

        # Partial block skip (partial_attn / partial_mlp / partial_attn_mlp)
        self.partial_mode: str | None = None          # None | "attn" | "mlp" | "attn_mlp"
        self.attn1_pre_gate_cache: dict[int, torch.Tensor] = {}
        self.ff_pre_gate_cache:    dict[int, torch.Tensor] = {}

        # Selective skip (arbitrary skip_blocks set — install_selective_skip)
        self.skip_blocks: set[int] = set()
        self.block_residual_cache: dict[int, torch.Tensor] = {}

    def reset(self):
        """Reset per-image state. Analysis buffers and correctors preserved."""
        self.step_idx = 0
        self.deep_residual_cache = None
        self.cached_deep_std = None
        self.prev_block_outputs = {}
        self.h_in_cached = None
        self.residual_per_block = None
        self.lr_probe_cached = None
        self.attn1_pre_gate_cache.clear()
        self.ff_pre_gate_cache.clear()
        self.block_residual_cache.clear()

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
# SVD-Aware LR probe helper
# ---------------------------------------------------------------------------

def _collect_block_lr_probe(block, hidden_states: torch.Tensor) -> torch.Tensor:
    """Apply the LR branch of block.attn1.to_out[0] (SVDQuantLinear) to hidden_states.

    Used by cache_lora_svd: captures a low-rank probe signal conditioned on the
    current deep-region entry state.  Returns zeros if SVDQuant is not active.

    Args:
        block: a transformer block (BasicTransformerBlock)
        hidden_states: [B, T, H] float tensor on GPU

    Returns:
        [B, T, H] probe tensor (same device/dtype as hidden_states)
    """
    try:
        from modelopt.torch.quantization.nn.modules.quant_linear import SVDQuantLinear
    except ImportError:
        return torch.zeros_like(hidden_states)

    attn1 = getattr(block, "attn1", None)
    if attn1 is None:
        return torch.zeros_like(hidden_states)
    to_out_mod = getattr(attn1, "to_out", None)
    if to_out_mod is None or len(to_out_mod) == 0:
        return torch.zeros_like(hidden_states)
    to_out = to_out_mod[0]
    if not isinstance(to_out, SVDQuantLinear):
        return torch.zeros_like(hidden_states)

    try:
        lora_a = to_out.weight_quantizer.svdquant_lora_a  # [rank, in_dim]
        lora_b = to_out.weight_quantizer.svdquant_lora_b  # [out_dim, rank]
        if lora_a is None or lora_b is None:
            return torch.zeros_like(hidden_states)
        x_scaled = to_out._apply_pre_quant_scale(hidden_states)
        return F.linear(F.linear(x_scaled, lora_a), lora_b)
    except Exception:
        return torch.zeros_like(hidden_states)


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

            # Save deep-region entry for dx-based correctors
            needs_h_in = (
                state.corrector_A is not None
                or state.phase_correctors is not None
                or state.step_scales is not None
                or state.svd_corrector_A is not None
                or state.block_correctors is not None
                or state.nl_corrector is not None
                or getattr(state, '_tf_calibration_mode', False)
            )
            if needs_h_in:
                state.h_in_cached = hidden_states.detach().clone()

            # SVD-Aware: save LR probe at deep-region entry
            if state.svd_corrector_A is not None:
                state.lr_probe_cached = _collect_block_lr_probe(
                    self.transformer_blocks[cache_start], hidden_states
                ).detach()

            # Block-specific: initialize per-block residual cache
            if state.block_correctors is not None:
                state.residual_per_block = []

            for i, block in enumerate(self.transformer_blocks[cache_start:cache_end]):
                b_idx = cache_start + i
                if state.block_correctors is not None:
                    h_before_i = hidden_states.clone()
                hidden_states = block(hidden_states, **block_kwargs)
                if state.block_correctors is not None:
                    state.residual_per_block.append((hidden_states - h_before_i).detach())
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

            # ── Direction 9: Teacher-forced calibration ────────────────────────
            if getattr(state, '_tf_calibration_mode', False) and state.h_in_cached is not None:
                with torch.no_grad():
                    _stale_dx = (hidden_states - state.h_in_cached).detach().cpu()
                    h_fresh = hidden_states.clone()
                    for blk in self.transformer_blocks[cache_start:cache_end]:
                        h_fresh = blk(h_fresh, **block_kwargs)
                    _fresh_res = h_fresh - hidden_states
                    _drift = (_fresh_res - state.deep_residual_cache).detach().cpu()
                if not hasattr(state, '_tf_pairs'):
                    state._tf_pairs = []
                state._tf_pairs.append((_stale_dx.float(), _drift.float()))
                # Apply basic cached residual only (no correction during TF calibration)
                hidden_states = hidden_states + state.deep_residual_cache

            # ── Nonlinear corrector ──────────────────────────────────────
            elif state.nl_corrector is not None and state.h_in_cached is not None:
                dx = hidden_states - state.h_in_cached
                dtype = hidden_states.dtype
                def _apply_nl_corrector():
                    cdtype = next(state.nl_corrector.parameters()).dtype
                    if state.nl_needs_t and state.nl_t_count is not None:
                        t_val = step_idx / max(state.nl_t_count - 1, 1)
                        t_norm = torch.full(
                            (*dx.shape[:-1], 1), t_val,
                            device=dx.device, dtype=cdtype,
                        )
                        return state.nl_corrector(dx.to(cdtype), t_norm).to(dtype)
                    else:
                        return state.nl_corrector(dx.to(cdtype)).to(dtype)

                if state.nl_train_mode:
                    correction = _apply_nl_corrector()
                else:
                    with torch.no_grad():
                        correction = _apply_nl_corrector()
                if state.nl_fd_mode:
                    # fd: corrector predicts full residual; stale cache not used
                    hidden_states = hidden_states + correction
                else:
                    hidden_states = hidden_states + state.deep_residual_cache + correction

            # ── Direction 5: Block-specific corrector (global dx, per-block residual) ─
            elif state.block_correctors is not None and state.h_in_cached is not None:
                dx = hidden_states - state.h_in_cached  # single global dx
                dtype = hidden_states.dtype
                for i, (A_i, B_i) in enumerate(state.block_correctors):
                    corr_i = F.linear(F.linear(dx, A_i.to(dtype)), B_i.to(dtype))
                    hidden_states = (hidden_states
                                     + state.residual_per_block[i]
                                     + corr_i)

            else:
                # ── Compute lora_correction for global-residual modes ──────────
                lora_correction = None
                dtype = hidden_states.dtype

                if (state.svd_corrector_A is not None
                        and state.h_in_cached is not None
                        and state.lr_probe_cached is not None):
                    # Direction 4: SVD-Aware (augmented input [dx; lr_delta])
                    dx       = hidden_states - state.h_in_cached
                    lr_probe = _collect_block_lr_probe(
                        self.transformer_blocks[cache_start], hidden_states)
                    lr_delta = lr_probe - state.lr_probe_cached.to(hidden_states.device)
                    aug = torch.cat([dx, lr_delta], dim=-1)          # [B, T, 2H]
                    lora_correction = F.linear(
                        F.linear(aug, state.svd_corrector_A.to(dtype)),
                        state.svd_corrector_B.to(dtype))

                elif (state.phase_correctors is not None
                      and state.h_in_cached is not None
                      and state.phase_t_count is not None):
                    # Direction 2A: Phase-binned corrector
                    n_ph  = len(state.phase_correctors)
                    phase = min(step_idx * n_ph // state.phase_t_count, n_ph - 1)
                    A_p, B_p = state.phase_correctors[phase]
                    dx = hidden_states - state.h_in_cached
                    lora_correction = F.linear(
                        F.linear(dx, A_p.to(dtype)), B_p.to(dtype))

                elif (state.corrector_A is not None
                      and state.corrector_B is not None
                      and state.h_in_cached is not None):
                    # Standard cache_lora  OR  Direction 2B (ts) with same A/B
                    dx = hidden_states - state.h_in_cached
                    lora_correction = F.linear(
                        F.linear(dx, state.corrector_A.to(dtype)),
                        state.corrector_B.to(dtype))
                    if state.step_scales is not None:
                        # Direction 2B: multiply by per-step scale
                        s_idx = min(step_idx, state.step_scales.shape[0] - 1)
                        scale = state.step_scales[s_idx].to(dtype).to(hidden_states.device)
                        lora_correction = scale * lora_correction

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


def calibrate_cache_lora_phased(
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
    n_phases: int = 3,
    calib_seed_offset: int = 1000,
) -> tuple:
    """Direction 2A: Phase-binned corrector.

    Splits denoising steps into `n_phases` equal bins (early/mid/late) and
    learns a separate rank-k (A, B) per bin.  Returns a list of n_phases
    corrector pairs, plus t_count (needed at inference to bin the step index).

    Returns:
        (phase_correctors, t_count, calib_time_sec)
        phase_correctors: list of (A, B) tuples, A:[rank,H], B:[H,rank]
    """
    import time as _time
    _calib_t0 = _time.perf_counter()

    num_calib = min(num_calib, len(prompts))
    print(f"  [Cache-LoRA Phase Calib] {num_calib} prompts × {t_count} steps, "
          f"interval={cache_interval}, rank={rank}, n_phases={n_phases}")

    hidden_dim = None
    for _, param in transformer.transformer_blocks[cache_start].named_parameters():
        if param.ndim >= 2:
            hidden_dim = param.shape[-1]
            break
    if hidden_dim is None:
        raise RuntimeError("calibrate_cache_lora_phased: cannot determine hidden_dim")

    step_counter = [0]
    h_in_buf: dict = {}
    h_out_buf: dict = {}
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

    C_phases = [torch.zeros(hidden_dim, hidden_dim, dtype=torch.float64) for _ in range(n_phases)]
    counts   = [0] * n_phases

    try:
        for i in range(num_calib):
            step_counter[0] = 0
            h_in_buf.clear(); h_out_buf.clear()
            _tmp_in[0] = None

            gen = torch.Generator(device=device).manual_seed(calib_seed_offset + i)
            with torch.no_grad():
                pipe(prompts[i % len(prompts)], num_inference_steps=t_count,
                     guidance_scale=guidance_scale, generator=gen)

            n_collected = step_counter[0]
            for s in range(cache_interval, n_collected):
                s_prev = s - cache_interval
                if s_prev not in h_in_buf:
                    continue
                phase = min(s * n_phases // t_count, n_phases - 1)
                dx    = (h_in_buf[s].float() - h_in_buf[s_prev].float()).reshape(-1, hidden_dim)
                drift = ((h_out_buf[s].float() - h_in_buf[s].float())
                         - (h_out_buf[s_prev].float() - h_in_buf[s_prev].float())).reshape(-1, hidden_dim)
                C_phases[phase] += drift.double().T @ dx.double()
                counts[phase]   += dx.shape[0]

            h_in_buf.clear(); h_out_buf.clear()
            print(f"    calib {i+1}/{num_calib} done", flush=True)
    finally:
        h_pre.remove()
        h_post.remove()

    phase_correctors = []
    for p in range(n_phases):
        C_norm = (C_phases[p] / max(counts[p], 1)).float()
        U, S, Vt = torch.linalg.svd(C_norm, full_matrices=False)
        sq = S[:rank].clamp(min=0.0).sqrt()
        A_p = (sq.unsqueeze(1) * Vt[:rank, :]).to(device)
        B_p = (U[:, :rank] * sq.unsqueeze(0)).to(device)
        phase_correctors.append((A_p, B_p))
        print(f"    phase {p}: {counts[p]:,} samples, top-SV={S[0].item():.4f}")

    calib_time_sec = _time.perf_counter() - _calib_t0
    print(f"  [Cache-LoRA Phase Calib] Done in {calib_time_sec:.1f}s")
    return phase_correctors, t_count, calib_time_sec


def calibrate_cache_lora_timestep(
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
    """Direction 2B: Timestep-conditional low-rank corrector.

    Learns a global (A, B) corrector, then per-step scalar scale factors via
    least-squares:  scale_s = <drift_s, B@A@dx_s> / ||B@A@dx_s||²

    Returns:
        (corrector_A, corrector_B, step_scales, calib_time_sec)
        step_scales: [t_count] float tensor on `device`
    """
    import time as _time
    _calib_t0 = _time.perf_counter()

    num_calib = min(num_calib, len(prompts))
    print(f"  [Cache-LoRA TS Calib] {num_calib} prompts × {t_count} steps, "
          f"interval={cache_interval}, rank={rank}")

    hidden_dim = None
    for _, param in transformer.transformer_blocks[cache_start].named_parameters():
        if param.ndim >= 2:
            hidden_dim = param.shape[-1]
            break

    step_counter = [0]
    h_in_buf: dict = {}
    h_out_buf: dict = {}
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

    C = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float64)
    total_samples = 0
    dx_by_step:    dict = {s: [] for s in range(t_count)}
    drift_by_step: dict = {s: [] for s in range(t_count)}

    try:
        for i in range(num_calib):
            step_counter[0] = 0
            h_in_buf.clear(); h_out_buf.clear()
            _tmp_in[0] = None

            gen = torch.Generator(device=device).manual_seed(calib_seed_offset + i)
            with torch.no_grad():
                pipe(prompts[i % len(prompts)], num_inference_steps=t_count,
                     guidance_scale=guidance_scale, generator=gen)

            n_collected = step_counter[0]
            for s in range(cache_interval, n_collected):
                s_prev = s - cache_interval
                if s_prev not in h_in_buf:
                    continue
                dx    = (h_in_buf[s].float() - h_in_buf[s_prev].float()).reshape(-1, hidden_dim)
                drift = ((h_out_buf[s].float() - h_in_buf[s].float())
                         - (h_out_buf[s_prev].float() - h_in_buf[s_prev].float())).reshape(-1, hidden_dim)
                C += drift.double().T @ dx.double()
                total_samples += dx.shape[0]
                dx_by_step[s].append(dx)
                drift_by_step[s].append(drift)

            h_in_buf.clear(); h_out_buf.clear()
            print(f"    calib {i+1}/{num_calib} done", flush=True)
    finally:
        h_pre.remove()
        h_post.remove()

    # Global (A, B)
    C_norm = (C / max(total_samples, 1)).float()
    U, S, Vt = torch.linalg.svd(C_norm, full_matrices=False)
    sq = S[:rank].clamp(min=0.0).sqrt()
    corrector_A_cpu = sq.unsqueeze(1) * Vt[:rank, :]   # [rank, H] on CPU
    corrector_B_cpu = U[:, :rank] * sq.unsqueeze(0)    # [H, rank]

    # Per-step scale: scale_s = <drift, B@A@dx> / ||B@A@dx||²
    step_scales = torch.ones(t_count)
    for s in range(cache_interval, t_count):
        if not dx_by_step.get(s):
            continue
        dx_s    = torch.cat(dx_by_step[s],    dim=0)  # [N, H]
        drift_s = torch.cat(drift_by_step[s], dim=0)
        predicted = F.linear(F.linear(dx_s, corrector_A_cpu), corrector_B_cpu)
        num = (drift_s * predicted).sum()
        den = (predicted * predicted).sum().clamp(min=1e-8)
        step_scales[s] = (num / den).clamp(0.1, 10.0)

    print(f"  [Cache-LoRA TS] step_scales range: "
          f"[{step_scales.min():.3f}, {step_scales.max():.3f}]")

    calib_time_sec = _time.perf_counter() - _calib_t0
    print(f"  [Cache-LoRA TS Calib] Done in {calib_time_sec:.1f}s")
    return (corrector_A_cpu.to(device), corrector_B_cpu.to(device),
            step_scales.to(device), calib_time_sec)


def calibrate_cache_lora_blockwise(
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
    """Direction 3: Block-specific corrector.

    Learns an independent rank-k (A_i, B_i) for each deep block i in
    [cache_start, cache_end).  Runs num_calib passes per block (sequential),
    keeping memory usage bounded to one block at a time.

    Returns:
        (block_correctors, calib_time_sec)
        block_correctors: list of (A_i, B_i) — length = cache_end - cache_start
    """
    import time as _time
    _calib_t0 = _time.perf_counter()

    n_blocks  = cache_end - cache_start
    num_calib = min(num_calib, len(prompts))
    print(f"  [Cache-LoRA Block Calib] {num_calib} prompts × {t_count} steps, "
          f"interval={cache_interval}, rank={rank}, n_blocks={n_blocks}")

    hidden_dim = None
    for _, param in transformer.transformer_blocks[cache_start].named_parameters():
        if param.ndim >= 2:
            hidden_dim = param.shape[-1]
            break

    block_correctors = []

    for bi in range(n_blocks):
        block = transformer.transformer_blocks[cache_start + bi]
        step_counter = [0]
        h_in_buf: dict = {}
        h_out_buf: dict = {}
        _tmp_in: list = [None]

        def _pre_hook(module, args, _tmp=_tmp_in):
            _tmp[0] = args[0].detach().cpu()

        def _post_hook(module, args, output, _tmp=_tmp_in,
                       _sc=step_counter, _ib=h_in_buf, _ob=h_out_buf):
            if _tmp[0] is not None:
                s = _sc[0]
                _ib[s]  = _tmp[0]
                _ob[s]  = output.detach().cpu()
                _tmp[0] = None
                _sc[0] += 1

        h_pre  = block.register_forward_pre_hook(_pre_hook)
        h_post = block.register_forward_hook(_post_hook)

        C = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float64)
        total_samples = 0

        try:
            for i in range(num_calib):
                step_counter[0] = 0
                h_in_buf.clear(); h_out_buf.clear()
                _tmp_in[0] = None

                gen = torch.Generator(device=device).manual_seed(calib_seed_offset + i)
                with torch.no_grad():
                    pipe(prompts[i % len(prompts)], num_inference_steps=t_count,
                         guidance_scale=guidance_scale, generator=gen)

                n_collected = step_counter[0]
                for s in range(cache_interval, n_collected):
                    s_prev = s - cache_interval
                    if s_prev not in h_in_buf:
                        continue
                    dx    = (h_in_buf[s].float() - h_in_buf[s_prev].float()).reshape(-1, hidden_dim)
                    drift = ((h_out_buf[s].float() - h_in_buf[s].float())
                             - (h_out_buf[s_prev].float() - h_in_buf[s_prev].float())).reshape(-1, hidden_dim)
                    C += drift.double().T @ dx.double()
                    total_samples += dx.shape[0]

                h_in_buf.clear(); h_out_buf.clear()
        finally:
            h_pre.remove()
            h_post.remove()

        C_norm = (C / max(total_samples, 1)).float()
        U, S, Vt = torch.linalg.svd(C_norm, full_matrices=False)
        sq = S[:rank].clamp(min=0.0).sqrt()
        A_i = (sq.unsqueeze(1) * Vt[:rank, :]).to(device)
        B_i = (U[:, :rank] * sq.unsqueeze(0)).to(device)
        block_correctors.append((A_i, B_i))
        print(f"    block {cache_start+bi}: {total_samples:,} samples, "
              f"top-SV={S[0].item():.4f}", flush=True)

    calib_time_sec = _time.perf_counter() - _calib_t0
    print(f"  [Cache-LoRA Block Calib] Done in {calib_time_sec:.1f}s")
    return block_correctors, calib_time_sec


def calibrate_cache_lora_svdaware(
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
    """Direction 4: SVD-Aware Cache-LoRA.

    Augments the corrector input with the LR-branch probe from
    block[cache_start].attn1.to_out[0], giving the corrector an additional
    signal about SVDQUANT's low-rank correction.

    Cross-covariance shape: C = drift.T @ [dx; lr_probe_delta]  → [H, 2H]
    Corrector: A:[rank, 2H], B:[H, rank]

    Returns:
        (corrector_A, corrector_B, calib_time_sec)
        corrector_A: [rank, 2H]
        corrector_B: [H, rank]
    """
    import time as _time
    _calib_t0 = _time.perf_counter()

    num_calib = min(num_calib, len(prompts))
    print(f"  [Cache-LoRA SVD-Aware Calib] {num_calib} prompts × {t_count} steps, "
          f"interval={cache_interval}, rank={rank}")

    hidden_dim = None
    for _, param in transformer.transformer_blocks[cache_start].named_parameters():
        if param.ndim >= 2:
            hidden_dim = param.shape[-1]
            break

    block_cs = transformer.transformer_blocks[cache_start]

    step_counter = [0]
    h_in_buf:      dict = {}
    h_out_buf:     dict = {}
    lr_probe_buf:  dict = {}
    _tmp_in:       list = [None]
    _tmp_probe:    list = [None]

    def _pre_hook(module, args):
        h = args[0]
        _tmp_in[0]    = h.detach().cpu()
        _tmp_probe[0] = _collect_block_lr_probe(block_cs, h).detach().cpu()

    def _post_hook(module, args, output):
        if _tmp_in[0] is not None:
            s = step_counter[0]
            h_in_buf[s]     = _tmp_in[0]
            h_out_buf[s]    = output.detach().cpu()
            lr_probe_buf[s] = (_tmp_probe[0] if _tmp_probe[0] is not None
                               else torch.zeros_like(_tmp_in[0]))
            _tmp_in[0] = _tmp_probe[0] = None
            step_counter[0] += 1

    h_pre  = block_cs.register_forward_pre_hook(_pre_hook)
    h_post = transformer.transformer_blocks[cache_end - 1].register_forward_hook(_post_hook)

    # C shape: [H, 2H]  (drift vs [dx; lr_delta])
    C = torch.zeros(hidden_dim, 2 * hidden_dim, dtype=torch.float64)
    total_samples = 0

    try:
        for i in range(num_calib):
            step_counter[0] = 0
            h_in_buf.clear(); h_out_buf.clear(); lr_probe_buf.clear()
            _tmp_in[0] = _tmp_probe[0] = None

            gen = torch.Generator(device=device).manual_seed(calib_seed_offset + i)
            with torch.no_grad():
                pipe(prompts[i % len(prompts)], num_inference_steps=t_count,
                     guidance_scale=guidance_scale, generator=gen)

            n_collected = step_counter[0]
            for s in range(cache_interval, n_collected):
                s_prev = s - cache_interval
                if s_prev not in h_in_buf:
                    continue
                dx    = (h_in_buf[s].float() - h_in_buf[s_prev].float()).reshape(-1, hidden_dim)
                dp    = (lr_probe_buf[s].float() - lr_probe_buf[s_prev].float()).reshape(-1, hidden_dim)
                drift = ((h_out_buf[s].float() - h_in_buf[s].float())
                         - (h_out_buf[s_prev].float() - h_in_buf[s_prev].float())).reshape(-1, hidden_dim)
                aug = torch.cat([dx, dp], dim=-1)   # [N, 2H]
                C += drift.double().T @ aug.double()
                total_samples += dx.shape[0]

            h_in_buf.clear(); h_out_buf.clear(); lr_probe_buf.clear()
            print(f"    calib {i+1}/{num_calib} done", flush=True)
    finally:
        h_pre.remove()
        h_post.remove()

    C_norm = (C / max(total_samples, 1)).float()
    print(f"  [Cache-LoRA SVD-Aware] SVD of [{hidden_dim}×{2*hidden_dim}] matrix...")
    U, S, Vt = torch.linalg.svd(C_norm, full_matrices=False)

    sq = S[:rank].clamp(min=0.0).sqrt()
    corrector_A = (sq.unsqueeze(1) * Vt[:rank, :]).to(device)   # [rank, 2H]
    corrector_B = (U[:, :rank] * sq.unsqueeze(0)).to(device)    # [H, rank]

    calib_time_sec = _time.perf_counter() - _calib_t0
    print(f"  [Cache-LoRA SVD-Aware Calib] Done: A{list(corrector_A.shape)} "
          f"in {calib_time_sec:.1f}s")
    return corrector_A, corrector_B, calib_time_sec


def calibrate_cache_lora_teacherforced(
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
    full_steps_set: set | None = None,
) -> tuple:
    """
    Teacher-forced Cache-LoRA calibration.

    Unlike `calibrate_cache_lora` (which runs fresh inference), this function:
    1. Installs basic DeepCache on the transformer (no LoRA correction)
    2. For each calibration prompt, runs inference WITH DeepCache active
    3. At each cached step, records:
       - stale_dx = hidden_states_at_deep_entry - h_in_cached  (actual inference-time input)
       - drift    = fresh_residual - cached_residual           (what correction needs to add)
    4. Fits corrector from stale dx → drift (not fresh dx → drift)

    This eliminates the calibration-inference distribution mismatch: the corrector
    is trained on the same activation distribution it will see during inference.

    Returns:
        (corrector_A, corrector_B, calib_time_sec)
    """
    import time as _time
    _calib_t0 = _time.perf_counter()

    if full_steps_set is None:
        full_steps_set = {0}
    num_calib = min(num_calib, len(prompts))
    print(f"  [TF Cache-LoRA Calib] {num_calib} prompts × {t_count} steps, "
          f"interval={cache_interval}, rank={rank}, seed_offset={calib_seed_offset}")

    # Determine hidden_dim
    hidden_dim = None
    for name, param in transformer.transformer_blocks[cache_start].named_parameters():
        if param.ndim >= 2:
            hidden_dim = param.shape[-1]
            break
    if hidden_dim is None:
        raise RuntimeError("calibrate_cache_lora_teacherforced: cannot determine hidden_dim")

    # Backup original forward and install basic DeepCache
    orig_forward = transformer.forward
    tf_state = install_deepcache(
        transformer,
        cache_start=cache_start,
        cache_end=cache_end,
        cache_interval=cache_interval,
        full_steps_set=full_steps_set,
    )
    tf_state._tf_calibration_mode = True
    tf_state._tf_pairs = []

    try:
        for i in range(num_calib):
            tf_state.reset()
            tf_state._tf_calibration_mode = True  # reset() clears dynamic attrs — re-set
            gen = torch.Generator(device=device).manual_seed(calib_seed_offset + i)
            with torch.no_grad():
                pipe(
                    prompts[i % len(prompts)],
                    num_inference_steps=t_count,
                    guidance_scale=guidance_scale,
                    generator=gen,
                )
            n_pairs = len(tf_state._tf_pairs)
            print(f"    TF calib {i+1}/{num_calib} done  (pairs collected so far: {n_pairs})", flush=True)
    finally:
        # Restore original forward
        transformer.forward = orig_forward

    tf_pairs = tf_state._tf_pairs
    if len(tf_pairs) == 0:
        raise RuntimeError("calibrate_cache_lora_teacherforced: no pairs collected. "
                           "Ensure cache_interval > 1 so cached steps exist.")

    # Accumulate cross-covariance C = Σ drift.T @ stale_dx
    C = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float64)
    total_samples = 0
    for stale_dx, drift in tf_pairs:
        dx_flat = stale_dx.reshape(-1, hidden_dim)    # [N, H]
        dr_flat = drift.reshape(-1, hidden_dim)        # [N, H]
        C += dr_flat.double().T @ dx_flat.double()
        total_samples += dx_flat.shape[0]

    print(f"  [TF Cache-LoRA Calib] Total pairs={len(tf_pairs)}, tokens={total_samples:,}")
    print(f"  [TF Cache-LoRA Calib] SVD of [{hidden_dim}×{hidden_dim}] matrix...")

    C_norm = (C / total_samples).float()
    U, S, Vt = torch.linalg.svd(C_norm, full_matrices=False)

    top_sv = S[:min(rank + 4, len(S))].tolist()
    print(f"  [TF Cache-LoRA Calib] Top singular values: {[f'{v:.4f}' for v in top_sv]}")

    sq = S[:rank].clamp(min=0.0).sqrt()
    corrector_A = (sq.unsqueeze(1) * Vt[:rank, :]).to(device)   # [rank, H]
    corrector_B = (U[:, :rank] * sq.unsqueeze(0)).to(device)    # [H, rank]

    calib_time_sec = _time.perf_counter() - _calib_t0
    print(f"  [TF Cache-LoRA Calib] Done: A{list(corrector_A.shape)}, B{list(corrector_B.shape)} "
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


# ---------------------------------------------------------------------------
# Partial Block Skip
# ---------------------------------------------------------------------------

def _make_partial_block_forward(b_idx, state, cache_interval, full_steps_set):
    """Return a per-block forward that skips attn1 and/or ff on cached steps.

    Mirrors BasicTransformerBlock.forward (ada_norm_single branch).
    state.partial_mode: "attn" | "mlp" | "attn_mlp"
    """
    def partial_block_forward(self, hidden_states, attention_mask=None,
                               encoder_hidden_states=None, encoder_attention_mask=None,
                               timestep=None, cross_attention_kwargs=None,
                               class_labels=None, added_cond_kwargs=None):
        batch_size = hidden_states.shape[0]

        # Determine skip predicates
        step_idx = state.step_idx - 1   # transformer wrapper already bumped step_idx
        is_cached = (step_idx not in full_steps_set) and (step_idx % cache_interval != 0)
        skip_attn = state.partial_mode in ("attn", "attn_mlp")
        skip_mlp  = state.partial_mode in ("mlp",  "attn_mlp")

        # 1. adaLN — ALWAYS fresh (gate* are timestep-dependent)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)

        # 2. norm1 + optional pos_embed
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # GLIGEN: pop from kwargs (mirror diffusers)
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        # 3. attn1 (self-attention) — cache-skip candidate
        if skip_attn and is_cached and b_idx in state.attn1_pre_gate_cache:
            attn1_out = state.attn1_pre_gate_cache[b_idx]
        else:
            attn1_out = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if skip_attn:
                state.attn1_pre_gate_cache[b_idx] = attn1_out.detach()

        # 4. gate + residual
        hidden_states = gate_msa * attn1_out + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 4b. GLIGEN control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 5. attn2 (cross-attention) — ALWAYS fresh in all partial modes
        if self.attn2 is not None:
            # ada_norm_single: no norm2 before attn2 (just pass hidden_states directly)
            norm_hidden_states = hidden_states
            if self.pos_embed is not None:
                pass  # ada_norm_single skips pos_embed before attn2
            attn2_out = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn2_out + hidden_states

        # 6. norm2 for FF (ada_norm_single uses self.norm2 here)
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        # 7. ff (MLP) — cache-skip candidate
        if skip_mlp and is_cached and b_idx in state.ff_pre_gate_cache:
            ff_out = state.ff_pre_gate_cache[b_idx]
        else:
            ff_out = self.ff(norm_hidden_states)
            if skip_mlp:
                state.ff_pre_gate_cache[b_idx] = ff_out.detach()

        # 8. gate + residual
        hidden_states = gate_mlp * ff_out + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states

    return partial_block_forward


def install_partial_skip(
    transformer,
    cache_start: int,
    cache_end: int,
    cache_interval: int = 2,
    num_full_steps: int = 1,
    partial_mode: str = "attn",
) -> "DeepCacheState":
    """Install partial-block-skip on transformer_blocks[cache_start:cache_end].

    partial_mode:
      "attn"     — cache attn1 (self-attn pre-gate), run attn2+ff fresh
      "mlp"      — cache ff (pre-gate), run attn1+attn2 fresh
      "attn_mlp" — cache both attn1 and ff, run attn2 fresh
    """
    state = DeepCacheState()
    state.partial_mode = partial_mode
    full_steps_set = set(range(num_full_steps))

    # Patch each deep-range block
    for i in range(cache_start, cache_end):
        block = transformer.transformer_blocks[i]
        fn = _make_partial_block_forward(i, state, cache_interval, full_steps_set)
        block.forward = types.MethodType(fn, block)

    # Thin wrapper on transformer.forward to increment step_idx exactly once per step
    orig_forward = transformer.forward.__func__
    def _step_counter(self, *args, **kwargs):
        state.step_idx += 1
        return orig_forward(self, *args, **kwargs)
    transformer.forward = types.MethodType(_step_counter, transformer)

    return state


# ---------------------------------------------------------------------------
# Selective skip — arbitrary skip_blocks set (block-wise sensitivity experiment)
# ---------------------------------------------------------------------------

def _make_selective_block_forward(b_idx, state, cache_interval, full_steps_set, orig_fwd):
    """Return a per-block forward that skips b_idx on cached steps.

    On fresh step: run orig_fwd, store residual h_out - h_in.
    On cached step (cache hit): return h_in + stored residual (no compute).
    On cached step (cache miss, first time): falls through to fresh.
    """
    def selective_block_forward(self, hidden_states, *args, **kwargs):
        step_idx = state.step_idx - 1
        is_cached = (step_idx not in full_steps_set) and (step_idx % cache_interval != 0)
        if is_cached and b_idx in state.block_residual_cache:
            return hidden_states + state.block_residual_cache[b_idx]
        h_in = hidden_states
        h_out = orig_fwd(self, hidden_states, *args, **kwargs)
        state.block_residual_cache[b_idx] = (h_out - h_in).detach()
        return h_out
    return selective_block_forward


def install_selective_skip(
    transformer,
    skip_blocks,
    cache_interval: int = 2,
    num_full_steps: int = 1,
) -> "DeepCacheState":
    """Install selective block skip: patch only the blocks in skip_blocks.

    skip_blocks: iterable of block indices to skip on cached steps.
    Non-skipped blocks always run fresh (zero overhead — not patched at all).
    """
    state = DeepCacheState()
    state.skip_blocks = set(skip_blocks)
    full_steps_set = set(range(num_full_steps))

    for i in sorted(state.skip_blocks):
        block = transformer.transformer_blocks[i]
        orig_fwd = type(block).forward  # unbound class method — always the original diffusers forward
        fn = _make_selective_block_forward(i, state, cache_interval, full_steps_set, orig_fwd)
        block.forward = types.MethodType(fn, block)

    orig_forward = transformer.forward.__func__
    def _step_counter(self, *args, **kwargs):
        state.step_idx += 1
        return orig_forward(self, *args, **kwargs)
    transformer.forward = types.MethodType(_step_counter, transformer)

    return state


# ---------------------------------------------------------------------------
# Selective + Partial Block Skip
# ---------------------------------------------------------------------------

def install_selective_partial_skip(
    transformer,
    skip_blocks,
    partial_mode: str = "attn",
    cache_interval: int = 2,
    num_full_steps: int = 1,
) -> "DeepCacheState":
    """Install partial-block-skip on an arbitrary set of blocks (sensitivity-ranked).

    Combines selective block targeting with sub-module granularity: only
    skip_blocks are patched; within each patched block attn2 is always fresh
    (partial_mode='attn'/'attn_mlp') or attn1+attn2 fresh (partial_mode='mlp').
    """
    state = DeepCacheState()
    state.partial_mode = partial_mode
    state.skip_blocks  = set(skip_blocks)
    full_steps_set = set(range(num_full_steps))

    for i in sorted(state.skip_blocks):
        block = transformer.transformer_blocks[i]
        fn = _make_partial_block_forward(i, state, cache_interval, full_steps_set)
        block.forward = types.MethodType(fn, block)

    orig_forward_sp = transformer.forward.__func__
    def _step_counter_sp(self, *args, **kwargs):
        state.step_idx += 1
        return orig_forward_sp(self, *args, **kwargs)
    transformer.forward = types.MethodType(_step_counter_sp, transformer)

    return state


# ---------------------------------------------------------------------------
# Selective Skip + Nonlinear Corrector
# ---------------------------------------------------------------------------

def _make_selective_block_forward_nl(b_idx, is_first_skip, is_last_skip,
                                     state, cache_interval, full_steps_set, orig_fwd):
    """Selective block skip with nl_corrector support.

    First skip block (fresh step): save h_in as reference for next cached step.
    First skip block (cached step): compute dx = h_in(now) - h_in(prev fresh).
    Last skip block (cached step): apply nl_corrector(dx) after residual reuse.

    Mirrors DeepCache's dx = hidden_states - state.h_in_cached convention.
    """
    def selective_block_forward_nl(self, hidden_states, *args, **kwargs):
        step_idx = state.step_idx - 1
        is_cached = (step_idx not in full_steps_set) and (step_idx % cache_interval != 0)

        if is_first_skip:
            if not is_cached:
                # Fresh step: store h_in as reference for next cached step
                state.h_in_cached = hidden_states.detach()
            elif state.h_in_cached is not None:
                # Cached step: dx = current h_in - previous fresh h_in
                state._nl_dx = hidden_states - state.h_in_cached

        if is_cached and b_idx in state.block_residual_cache:
            hidden_states = hidden_states + state.block_residual_cache[b_idx]
        else:
            h_in = hidden_states
            hidden_states = orig_fwd(self, hidden_states, *args, **kwargs)
            state.block_residual_cache[b_idx] = (hidden_states - h_in).detach()

        if (is_last_skip and is_cached
                and state.nl_corrector is not None
                and getattr(state, '_nl_dx', None) is not None):
            dtype = hidden_states.dtype
            cdtype = next(state.nl_corrector.parameters()).dtype
            with torch.no_grad():
                correction = state.nl_corrector(state._nl_dx.to(cdtype)).to(dtype)
            hidden_states = hidden_states + correction

        return hidden_states
    return selective_block_forward_nl


def install_selective_skip_with_nl(
    transformer,
    skip_blocks,
    nl_corrector,
    cache_interval: int = 2,
    num_full_steps: int = 1,
) -> "DeepCacheState":
    """Selective block skip + nonlinear drift corrector.

    nl_corrector is applied at the last skip block on cached steps.
    Input dx = h_out_last_skip - h_in_first_skip captures the accumulated drift
    across the full skip span.
    """
    state = DeepCacheState()
    state.skip_blocks = set(skip_blocks)
    state.nl_corrector = nl_corrector
    full_steps_set = set(range(num_full_steps))

    sorted_blocks = sorted(state.skip_blocks)
    first_skip = sorted_blocks[0]
    last_skip  = sorted_blocks[-1]

    for i in sorted_blocks:
        block = transformer.transformer_blocks[i]
        orig_fwd = type(block).forward
        fn = _make_selective_block_forward_nl(
            i, i == first_skip, i == last_skip,
            state, cache_interval, full_steps_set, orig_fwd,
        )
        block.forward = types.MethodType(fn, block)

    orig_forward_snl = transformer.forward.__func__
    def _step_counter_snl(self, *args, **kwargs):
        state.step_idx += 1
        return orig_forward_snl(self, *args, **kwargs)
    transformer.forward = types.MethodType(_step_counter_snl, transformer)

    return state
